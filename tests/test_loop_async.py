"""The loop's general-purpose options: async backend/tools, Limits, uncharged malformed calls,
the FORCE_ANY idle nudge, a terminal tool other than finish, the async backend adapter."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from detective_joracle.agent.backends import (
    FakeBackend,
    async_openai_compatible_backend,
    openai_compatible_backend,
)
from detective_joracle.agent.loop import (
    FORCE_ANY,
    Budget,
    Limits,
    RunRecord,
    run_many,
    run_tool_loop,
    run_tool_loop_async,
)
from detective_joracle.agent.prompts import IDLE_NUDGE, REDUCTION, REDUCTION_TEMPLATE


class _Tools:
    """A tools stub that lets ``ValueError`` escape (unlike LiveTools) and knows its terminal."""

    def __init__(self, terminal: str = "finish") -> None:
        self.terminal = terminal
        self.finished: dict[str, Any] | None = None
        self.log: list[str] = []

    def call(self, name: str, args: Any) -> str:
        if name == "bad":
            raise ValueError("missing argument 'x'")
        self.log.append(name)
        if name == self.terminal:
            self.finished = {"predictions": args.get("predictions", [])}
            return "done"
        return f"{name} ok"


class _AsyncTools(_Tools):
    async def call(self, name: str, args: Any) -> str:  # type: ignore[override]
        await asyncio.sleep(0)
        return super().call(name, args)


class _AsyncBackend:
    """FakeBackend behind an awaitable call."""

    def __init__(self, inner: FakeBackend) -> None:
        self.inner = inner

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        await asyncio.sleep(0)
        return self.inner(messages, tools, force_tool)


def _rec() -> RunRecord:
    return RunRecord("o", "live", "blackbox", "fake", 0)


def test_async_loop_with_async_backend_and_async_tools_finishes() -> None:
    inner = FakeBackend(
        [("a", [("chat", {"user": "hi"})], 5), ("b", [("finish", {"predictions": [1]})], 5)]
    )
    t = _AsyncTools()
    rec = _rec()
    msgs = asyncio.run(run_tool_loop_async(t, [], "sys", "go", _AsyncBackend(inner), Budget(), rec))
    assert rec.stopped_by == "finish" and t.finished == {"predictions": [1]}
    assert t.log == ["chat", "finish"] and msgs[-1]["role"] == "tool"
    assert [tn.role for tn in rec.turns] == ["assistant", "tool", "assistant", "tool"]


def test_sync_wrapper_accepts_async_backend_even_inside_a_running_loop() -> None:
    inner = FakeBackend([("b", [("finish", {"predictions": []})], 5)])

    async def outer() -> RunRecord:
        rec = _rec()
        run_tool_loop(_AsyncTools(), [], "sys", "go", _AsyncBackend(inner), Budget(), rec)
        return rec

    assert asyncio.run(outer()).stopped_by == "finish"


def test_limits_exhausted_tool_answers_budget_text_and_is_not_executed() -> None:
    backend = FakeBackend(
        [
            ("a", [("readouts", {}), ("readouts", {})], 5),
            ("b", [("readouts", {}), ("chat", {})], 5),
            ("c", [("finish", {"predictions": []})], 5),
        ]
    )
    t = _Tools()
    rec = _rec()
    # chat is capped too (and not spent), so exhausting readouts alone does not force the reduction
    limits = Limits({"readouts": 2, "chat": 5})
    run_tool_loop(t, [], "sys", "go", backend, Budget(), rec, limits=limits)
    assert t.log == ["readouts", "readouts", "chat", "finish"]  # the third readouts never ran
    results = [tn.content for tn in rec.turns if tn.role == "tool"]
    assert results[2] == (
        "BUDGET: you have used all 2 of your readouts calls. Work with what you have."
    )
    assert results[3] == "chat ok" and rec.stopped_by == "finish"


def test_limits_unlock_gate_locks_until_the_prerequisite_count_is_met() -> None:
    backend = FakeBackend(
        [
            ("a", [("experiment", {})], 5),  # locked: 0 of 2 readouts so far
            ("b", [("readouts", {}), ("experiment", {})], 5),  # still 1 short
            ("c", [("readouts", {}), ("experiment", {})], 5),  # unlocked
            ("d", [("finish", {"predictions": []})], 5),
        ]
    )
    t = _Tools()
    rec = _rec()
    run_tool_loop(
        t,
        [],
        "sys",
        "go",
        backend,
        Budget(),
        rec,
        limits=Limits({}, unlock={"experiment": ("readouts", 2)}),
    )
    assert t.log == ["readouts", "readouts", "experiment", "finish"]
    results = [tn.content for tn in rec.turns if tn.role == "tool"]
    assert results[0] == "LOCKED: call readouts 2 more time(s) before using experiment."
    assert results[2] == "LOCKED: call readouts 1 more time(s) before using experiment."
    assert results[4] == "experiment ok"


def test_all_capped_tools_spent_forces_the_reduction() -> None:
    backend = FakeBackend(
        [("a", [("readouts", {})], 5), ("b", [("chat", {})], 5), ("c", [("chat", {})], 5)]
    )
    t = _Tools()
    rec = _rec()
    run_tool_loop(t, [], "sys", "go", backend, Budget(), rec, limits=Limits({"readouts": 1}))
    # after the only capped tool is spent: REDUCTION + a forced finish (FakeBackend has no such
    # step, so it finishes empty), never the later chat steps
    assert backend.forced[1] == "finish" and backend.seen[1][-1]["content"] == REDUCTION
    assert t.log == ["readouts", "finish"] and rec.stopped_by == "finish"


def test_value_error_from_tools_is_returned_as_error_and_not_charged() -> None:
    backend = FakeBackend(
        [
            ("a", [("bad", {})], 5),
            ("b", [("bad", {})], 5),
            ("c", [("chat", {})], 5),
            ("d", [("finish", {"predictions": []})], 5),
        ]
    )
    t = _Tools()
    rec = _rec()
    run_tool_loop(t, [], "sys", "go", backend, Budget(), rec, limits=Limits({"bad": 1}))
    results = [tn.content for tn in rec.turns if tn.role == "tool"]
    # both malformed calls come back as ERROR (the second is NOT "BUDGET": nothing was charged)
    assert results[:2] == ["ERROR: missing argument 'x'", "ERROR: missing argument 'x'"]
    assert t.log == ["chat", "finish"] and rec.stopped_by == "finish"
    assert len([tn for tn in rec.turns if tn.role == "assistant"]) == 4  # they cost turns


def test_other_exceptions_from_tools_propagate() -> None:
    class Boom(_Tools):
        def call(self, name: str, args: Any) -> str:
            raise RuntimeError("infrastructure")

    backend = FakeBackend([("a", [("chat", {})], 5)])
    with pytest.raises(RuntimeError, match="infrastructure"):
        run_tool_loop(Boom(), [], "sys", "go", backend, Budget(), _rec())


def test_idle_turns_nudge_then_force_any_then_stop() -> None:
    backend = FakeBackend([("t1", [], 5), ("t2", [], 5), ("t3", [], 5), ("t4", [], 5)])
    t = _Tools()
    rec = _rec()
    run_tool_loop(t, [], "sys", "go", backend, Budget(), rec)
    # turn 1 idle -> the plain nudge; turn 2 idle -> REDUCTION + FORCE_ANY. FakeBackend has no
    # step with a call, so it emits finish() with empty args under FORCE_ANY.
    assert backend.seen[1][-1]["content"] == IDLE_NUDGE.format(terminal="finish")
    assert backend.seen[2][-1]["content"] == REDUCTION and backend.forced[2] == FORCE_ANY
    assert t.finished == {"predictions": []} and rec.stopped_by == "finish"


def test_idle_stop_when_the_model_never_calls_a_tool() -> None:
    class Mute:
        """A backend that ignores force_tool and always answers prose."""

        def __init__(self) -> None:
            self.forced: list[str | None] = []

        def __call__(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            force_tool: str | None = None,
        ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
            self.forced.append(force_tool)
            return "prose", [], {"output_tokens": 1}

    backend = Mute()
    rec = _rec()
    run_tool_loop(_Tools(), [], "sys", "go", backend, Budget(), rec)
    assert rec.stopped_by == "idle"
    assert backend.forced == [None, None, FORCE_ANY, None]  # nudge, force any, nudge, stop
    assert len(rec.turns) == 4


def test_terminal_name_is_parametrized() -> None:
    backend = FakeBackend(
        [
            ("a", [("readouts", {})], 30_000),  # blows the token budget
            ("b", [("note", {"text": "ignored"})], 10),
        ],
        terminal="submit_hypothesis",
    )
    t = _Tools(terminal="submit_hypothesis")
    rec = _rec()
    run_tool_loop(t, [], "sys", "go", backend, Budget(), rec, terminal="submit_hypothesis")
    assert backend.forced[1] == "submit_hypothesis"
    expected = REDUCTION_TEMPLATE.format(terminal="submit_hypothesis")
    assert "Call submit_hypothesis() now" in expected
    assert backend.seen[1][-1]["content"] == expected
    assert t.finished == {"predictions": []} and rec.stopped_by == "finish"
    assert REDUCTION_TEMPLATE.format(terminal="finish") == REDUCTION


def test_run_many_bounds_concurrency_and_preserves_order() -> None:
    running = 0
    peak = 0

    def job(i: int) -> Any:
        async def go() -> int:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01 * (5 - i))  # later jobs finish first
            running -= 1
            return i

        return go

    out = asyncio.run(run_many([job(i) for i in range(5)], concurrency=2))
    assert out == [0, 1, 2, 3, 4] and peak == 2


class _AsyncCompletions:
    def __init__(self, cached: int | None = 7) -> None:
        self.n = 0
        self.kw: dict[str, Any] = {}
        self.cached = cached

    async def create(self, **kw: Any) -> Any:
        self.n += 1
        self.kw = kw
        if self.n == 1:
            return SimpleNamespace(choices=None)  # a provider hiccup: retried
        tc = SimpleNamespace(
            id="c1", function=SimpleNamespace(name="chat", arguments='{"user": "hi"}')
        )
        msg = SimpleNamespace(content="text", tool_calls=[tc])
        usage = SimpleNamespace(
            completion_tokens=30,
            prompt_tokens=500,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
            prompt_tokens_details=(
                SimpleNamespace(cached_tokens=self.cached) if self.cached is not None else None
            ),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def test_async_openai_compatible_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    comp = _AsyncCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    backend = async_openai_compatible_backend(
        client, "m", extra_body={"k": 1}, cache=True, effort="low", provider="google-vertex"
    )
    text, calls, usage = asyncio.run(backend([{"role": "system", "content": "s"}], [], FORCE_ANY))
    assert text == "text" and calls == [{"id": "c1", "name": "chat", "args": {"user": "hi"}}]
    assert usage == {
        "output_tokens": 30,
        "reasoning_tokens": 10,
        "prompt_tokens": 500,
        "cached_tokens": 7,
    }
    assert comp.kw["tool_choice"] == "required"
    assert comp.kw["extra_body"] == {
        "k": 1,
        "reasoning": {"effort": "low"},
        "provider": {"order": ["google-vertex"], "allow_fallbacks": False},
    }
    assert comp.kw["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # a named force and a missing prompt_tokens_details
    comp2 = _AsyncCompletions(cached=None)
    backend2 = async_openai_compatible_backend(
        SimpleNamespace(chat=SimpleNamespace(completions=comp2)), "m"
    )
    _, _, usage2 = asyncio.run(backend2([], [], "chat"))
    assert comp2.kw["tool_choice"] == {"type": "function", "function": {"name": "chat"}}
    assert comp2.kw["extra_body"] == {} and usage2["cached_tokens"] == 0
    _, _, _ = asyncio.run(backend2([], [], None))
    assert comp2.kw["tool_choice"] == "auto"


def test_sync_backend_maps_force_any_to_required() -> None:
    class Completions:
        def __init__(self) -> None:
            self.kw: dict[str, Any] = {}

        def create(self, **kw: Any) -> Any:
            self.kw = kw
            msg = SimpleNamespace(content="", tool_calls=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(completion_tokens=1)
            )

    comp = Completions()
    backend = openai_compatible_backend(
        SimpleNamespace(chat=SimpleNamespace(completions=comp)), "m"
    )
    assert backend([], [], FORCE_ANY) == ("", [], {"output_tokens": 1, "reasoning_tokens": 0})
    assert comp.kw["tool_choice"] == "required"
