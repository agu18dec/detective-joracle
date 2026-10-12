"""The agent loop: tool calls under a budget, then the paper's forced reduction to ``finish``.

``run_tool_loop`` drives any tools object exposing ``call(name, args) -> str``, ``finished`` and
``log`` against a :class:`Backend` (one chat-completions call with tools). Everything the agent
saw and did is kept in the :class:`RunRecord` that the judges score and the viewer renders.
Context management (``trim_to_cap``, ``shrink_tool_results``) exists because a lens arm reads
whole pages of readouts and providers refuse very large contexts.

The loop body is ``run_tool_loop_async`` (sync or async backend and tools); ``run_tool_loop`` is
the sync wrapper every existing caller uses. Optional, off by default: :class:`Limits` (per-tool
call caps and unlock gates beside the token ``Budget``), a ``terminal`` tool name other than
``finish``, and ``run_many`` for running episodes concurrently.
"""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, TypeVar

from .prompts import IDLE_NUDGE, REDUCTION_TEMPLATE

T = TypeVar("T")

MAX_PREDICTIONS = 10


@dataclass
class ToolLog:
    """One tool call as the agent saw it: name, args, the returned text, cells it served."""

    name: str
    args: dict[str, Any]
    output: str
    cells_served: int = 0


@dataclass
class Turn:
    """One transcript entry: an assistant turn (text + tool calls + token usage) or a tool result."""

    role: str  # assistant | tool
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class RunRecord:
    """Everything one audit produced; ``to_json()`` is the on-disk record (``docs/architecture.md``)."""

    organism: str
    condition: str
    arm: str
    auditor: str
    seed: int
    turns: list[Turn] = field(default_factory=list)
    tool_log: list[ToolLog] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cells_served: int = 0
    chars_served: int = 0
    stopped_by: str = ""  # finish | budget | max_calls | error
    # stage 2 of the two-stage scoring: the same agent, in the same context, picking the planted
    # quirk out of the closed set after its free-form predictions are locked in.
    choice: dict[str, Any] | None = None
    consult_calls: int = 0  # ask_claude invocations (the "-ask" probe)
    select: str = "all"  # readout presentation (presentation.select): all | summary | llm | fve
    allow_reference: bool = True  # whether the agent could query the base model as a control
    trimmed: int = 0  # tool results replaced by a stub to keep the context under the cap

    def to_json(self) -> dict[str, Any]:
        """The record as plain JSON-able dicts (dataclasses expanded)."""
        return asdict(self)


# ``force_tool`` value meaning "the model must call SOME tool" (``tool_choice="required"``), as
# opposed to a named function. Forcing a named function with a nested-array schema came back
# shape-valid but empty on Gemini over OpenRouter; ``required`` lets the model produce the call
# the way it would have unforced. The loop uses it for the idle nudge only; the budget reduction
# still forces the terminal by name.
FORCE_ANY = "*"


class Backend(Protocol):
    """One chat-completions call with tools; returns (assistant_text, tool_calls, usage).

    ``force_tool`` names a function the model MUST call, or ``FORCE_ANY`` for "must call some
    tool"."""

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]: ...


class AsyncBackend(Protocol):
    """The same call as a coroutine (``async_openai_compatible_backend`` over ``openai.AsyncOpenAI``)."""

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]: ...


@dataclass
class Budget:
    """When the loop forces the reduction: output tokens (the paper's currency) or tool calls."""

    output_tokens: int = 25_000  # non-thinking generated tokens (the paper's currency)
    max_calls: int = 40
    cell_budget: int = 0  # readout cells a run may be served in total (0 = unlimited)


@dataclass
class Limits:
    """Per-tool call caps and unlock gates, additive to :class:`Budget` (which stays the currency).

    ``per_tool`` caps how many times each named tool may EXECUTE (calls the tool refused with a
    ``ValueError`` are not counted); a call to a capped-out tool answers ``BUDGET: …`` instead of
    running. ``unlock`` maps a tool to ``(prerequisite tool, count)``: until the prerequisite has
    executed that many times the tool answers ``LOCKED: …``. Tools absent from both are
    unlimited. When every tool in ``per_tool`` is exhausted the loop runs the same forced
    reduction it runs for an exhausted ``Budget``."""

    per_tool: dict[str, int]
    unlock: dict[str, tuple[str, int]] | None = None

    def exhausted(self, name: str, used: Mapping[str, int]) -> bool:
        """Whether ``name`` has executed its cap (``used`` = executions per tool)."""
        cap = self.per_tool.get(name)
        return cap is not None and used.get(name, 0) >= cap

    def locked(self, name: str, used: Mapping[str, int]) -> tuple[str, int] | None:
        """``(prerequisite, calls still needed)`` while ``name`` is gated, else ``None``."""
        req = (self.unlock or {}).get(name)
        if req is None:
            return None
        prereq, need = req
        have = used.get(prereq, 0)
        return None if have >= need else (prereq, need - have)

    def spent(self, used: Mapping[str, int]) -> bool:
        """Every capped tool is exhausted (an empty ``per_tool`` is never spent)."""
        return bool(self.per_tool) and all(self.exhausted(n, used) for n in self.per_tool)


SALVAGE_KEEP_CHARS = 2_000
# Total characters of tool output kept in the agent's context. Gemini refused ("blocked: OTHER",
# a 502) at ~1.55M characters of tool output — three full every-position readout pages — so
# the cap sits well under that. It applies to EVERY arm identically; black-box runs never reach
# it, so in practice it is the price of reading whole pages, paid only by the arms that read.
CONTEXT_CHAR_CAP = 800_000
TRIM_STUB = (
    "[This tool result ({n} characters) was trimmed from your context to stay within the "
    "provider's limits after you had read it. Your notes retain what mattered from it.]"
)


def shrink_tool_results(
    messages: list[dict[str, Any]], keep: int = SALVAGE_KEEP_CHARS
) -> list[dict[str, Any]]:
    """Truncate every oversized tool result (a new list), so a context the provider refused can
    be re-sent for one final ``finish`` turn."""
    out: list[dict[str, Any]] = []
    for m in messages:
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, str) and len(c) > keep:
            m = {**m, "content": c[:keep] + f"\n[... {len(c) - keep} characters dropped]"}
        out.append(m)
    return out


def trim_to_cap(messages: list[dict[str, Any]], cap: int = CONTEXT_CHAR_CAP) -> int:
    """Replace the OLDEST large tool results with a stub until the tool output in context fits
    under ``cap``. The newest result is never touched: it is the one the agent is about to read.
    Mutates in place; returns how many results were trimmed."""
    tools = [m for m in messages if m.get("role") == "tool" and isinstance(m.get("content"), str)]
    total = sum(len(m["content"]) for m in tools)
    n = 0
    for m in tools[:-1]:  # oldest first, newest exempt
        if total <= cap:
            break
        size = len(m["content"])
        if size < SALVAGE_KEEP_CHARS:
            continue
        m["content"] = TRIM_STUB.format(n=size)
        total -= size - len(m["content"])
        n += 1
    return n


async def run_tool_loop_async(
    tools: Any,
    schemas: list[dict[str, Any]],
    system: str,
    first_user: str,
    backend: Backend | AsyncBackend,
    budget: Budget,
    rec: RunRecord,
    *,
    limits: Limits | None = None,
    terminal: str = "finish",
) -> list[dict[str, Any]]:
    """The paper's agent loop over any tools object exposing ``call``, ``finished``, ``log``:
    tool calls until the terminal tool sets ``tools.finished``, the budget (output tokens or
    calls), then one forced reduction turn that MUST call ``terminal``. Fills ``rec.turns`` and
    ``rec.stopped_by``, and returns the message list so a caller can ask a follow-up question in
    the same context.

    ``backend`` and ``tools.call`` may each be sync or async (an awaitable result is awaited).
    A ``ValueError`` from ``tools.call`` is the model's mistake: it is returned as ``ERROR: …``,
    not counted toward ``limits``, and the loop continues; any other exception propagates.
    (``LiveTools.call`` never raises — it turns every error into text and logs the call.) A turn
    with no tool call is nudged; two in a row get the reduction with ``FORCE_ANY``; a fourth ends
    the run with ``stopped_by = "idle"``."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": first_user},
    ]
    reduction = REDUCTION_TEMPLATE.format(terminal=terminal)
    reduced = False
    salvaged = ""
    failures = 0
    force: str | None = None
    idle = 0
    used: dict[str, int] = {}  # executions per tool, the currency of ``limits``
    while True:
        rec.trimmed += trim_to_cap(messages)
        try:
            raw: Any = backend(messages, schemas, force)
            if inspect.isawaitable(raw):
                raw = await raw
            text, calls, usage = raw
            force = None
        except Exception as e:
            # A provider failure late in a run must not throw away everything the agent found.
            # First failure: shrink every tool result hard and retry the SAME turn, so the run
            # goes on. Second: force the terminal so it reports what it has. Third: give up.
            failures += 1
            salvaged = f"{type(e).__name__}: {str(e)[:200]}"
            if failures == 1:
                messages = shrink_tool_results(messages)
                continue
            if failures == 2 and not reduced:
                reduced = True
                messages.append({"role": "user", "content": reduction})
                force = terminal
                continue
            rec.stopped_by = f"error: {salvaged}"
            break
        out_tok = int(usage.get("output_tokens", 0))
        reason_tok = int(usage.get("reasoning_tokens", 0))
        rec.output_tokens += out_tok - reason_tok
        rec.reasoning_tokens += reason_tok
        rec.turns.append(Turn("assistant", text, calls, out_tok, reason_tok))
        # content=None is only valid alongside tool_calls; a null-content assistant turn with no
        # calls makes some providers reject the NEXT request with INVALID_ARGUMENT.
        msg: dict[str, Any] = {"role": "assistant", "content": text or (None if calls else "...")}
        if calls:
            msg["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                }
                for c in calls
            ]
        messages.append(msg)
        for c in calls:
            name = str(c["name"])
            gate = limits.locked(name, used) if limits else None
            if gate is not None:
                out = f"LOCKED: call {gate[0]} {gate[1]} more time(s) before using {name}."
            elif limits and limits.exhausted(name, used):
                out = (
                    f"BUDGET: you have used all {limits.per_tool[name]} of your {name} calls. "
                    "Work with what you have."
                )
            else:
                try:
                    out = tools.call(name, c["args"])
                    if inspect.isawaitable(out):
                        out = await out
                except ValueError as e:
                    # A malformed call never reached the tool: it costs a turn, not a call.
                    out = f"ERROR: {e}"
                else:
                    used[name] = used.get(name, 0) + 1
            rec.turns.append(Turn("tool", out))
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
            if tools.finished is not None:
                break
        if tools.finished is not None:
            rec.stopped_by = f"salvaged finish after {salvaged}" if salvaged else "finish"
            break
        n_calls = len(tools.log)
        spent = limits is not None and limits.spent(used)
        over = rec.output_tokens >= budget.output_tokens or n_calls >= budget.max_calls or spent
        if over and not reduced:
            reduced = True
            messages.append({"role": "user", "content": reduction})
            force = terminal  # the paper's reduction step: the next turn MUST be the terminal
            continue
        if over and reduced:
            rec.stopped_by = (
                "budget"
                if rec.output_tokens >= budget.output_tokens
                else "max_calls"
                if n_calls >= budget.max_calls
                else "limits"
            )
            break
        if not calls:  # plain text with no tool call: nudge, then force any tool, then give up
            idle += 1
            if idle >= 2 and not reduced:
                reduced = True
                messages.append({"role": "user", "content": reduction})
                force = FORCE_ANY
                continue
            if idle > 3:
                rec.stopped_by = "idle"
                break
            messages.append({"role": "user", "content": IDLE_NUDGE.format(terminal=terminal)})
        else:
            idle = 0
    return messages


def run_tool_loop(
    tools: Any,
    schemas: list[dict[str, Any]],
    system: str,
    first_user: str,
    backend: Backend | AsyncBackend,
    budget: Budget,
    rec: RunRecord,
    *,
    limits: Limits | None = None,
    terminal: str = "finish",
) -> list[dict[str, Any]]:
    """The sync entry point: :func:`run_tool_loop_async` on a private event loop. Inside a
    running event loop (a sync caller within async code) the loop runs on a worker thread —
    async backends should use ``run_tool_loop_async`` directly there."""
    coro = run_tool_loop_async(
        tools, schemas, system, first_user, backend, budget, rec, limits=limits, terminal=terminal
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


async def run_many(jobs: Iterable[Callable[[], Awaitable[T]]], concurrency: int) -> list[T]:
    """Run ``jobs`` (thunks returning awaitables) at most ``concurrency`` at a time; results in
    job order. An exception in one job propagates (``asyncio.gather`` semantics); wrap a job
    that must not take the batch down."""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def bounded(job: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            return await job()

    return list(await asyncio.gather(*(bounded(j) for j in jobs)))
