"""LLM backends for the investigator: any OpenAI-compatible tool-calling endpoint, plus a
scripted :class:`FakeBackend` for offline tests and plumbing smokes.

``make_backend_factory("openrouter")`` builds the OpenRouter client (``OPENROUTER_API_KEY``);
``make_backend_factory("openai")`` builds a plain client for whatever ``OPENAI_API_KEY`` /
``OPENAI_BASE_URL`` name (a local vLLM/SGLang auditor, a proxy, OpenAI itself).
``async_client=True`` gives the awaitable twin (``async_openai_compatible_backend`` over
``openai.AsyncOpenAI``) for ``run_tool_loop_async`` / ``run_many``.
"""

import asyncio
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, overload

from .loop import FORCE_ANY, AsyncBackend, Backend


def with_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic prompt caching through OpenRouter: a ``cache_control`` breakpoint on the system
    prompt and on the newest message, so each turn re-reads the whole prefix from cache. An
    agent turn is the previous turn plus one exchange, so without this every one of ~200 calls
    per run pays full price for a 100-300k-token context."""
    out: list[dict[str, Any]] = []
    last = len(messages) - 1
    for i, m in enumerate(messages):
        if i in (0, last) and isinstance(m.get("content"), str) and m["content"]:
            m = {
                **m,
                "content": [
                    {"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}
                ],
            }
        out.append(m)
    return out


def tool_choice_of(force_tool: str | None) -> Any:
    """The ``tool_choice`` for a forced tool: ``required`` for ``FORCE_ANY``, the named function
    otherwise, ``auto`` when nothing is forced."""
    if force_tool == FORCE_ANY:
        return "required"
    if force_tool:
        return {"type": "function", "function": {"name": force_tool}}
    return "auto"


def parse_tool_calls(choice: Any) -> list[dict[str, Any]]:
    """``[{id, name, args}]`` from a choice; unparsable JSON arguments become ``{"_raw": …}``."""
    calls = []
    for tc in choice.message.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        calls.append({"id": tc.id, "name": tc.function.name, "args": args})
    return calls


def _usage_of(usage: Any) -> dict[str, int]:
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
    return {
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": reasoning,
    }


def openai_compatible_backend(
    client: Any,
    model: str,
    *,
    extra_body: Mapping[str, Any] | None = None,
    max_tokens: int = 4000,
    cache: bool = False,
) -> Backend:
    """``openai.OpenAI``-style client (OpenRouter with ``base_url`` set, or any other). Usage
    keys follow the OpenAI/OpenRouter schema; reasoning tokens are read from
    ``completion_tokens_details``. ``cache`` adds Anthropic cache breakpoints."""

    def call(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        # A provider occasionally returns an error-shaped body with `choices` absent; treat it as
        # transient rather than killing the whole run at whatever call it happened on.
        """One tool-calling completion; retries an empty ``choices`` body, raises after three."""
        resp: Any = None
        for attempt in range(3):
            resp = client.chat.completions.create(
                model=model,
                messages=with_cache_breakpoints(messages) if cache else messages,
                tools=tools,
                tool_choice=tool_choice_of(force_tool),
                max_tokens=max_tokens,
                extra_body=dict(extra_body or {}),
            )
            if getattr(resp, "choices", None):
                break
            time.sleep(2 * (attempt + 1))
        if resp is None or not getattr(resp, "choices", None):
            raise RuntimeError(f"provider returned no choices after 3 attempts: {resp}")
        choice = resp.choices[0]
        return choice.message.content or "", parse_tool_calls(choice), _usage_of(resp.usage)

    return call


def async_openai_compatible_backend(
    client: Any,
    model: str,
    *,
    extra_body: Mapping[str, Any] | None = None,
    max_tokens: int = 4000,
    cache: bool = False,
    effort: str | None = None,
    provider: str | None = None,
) -> AsyncBackend:
    """The awaitable twin of :func:`openai_compatible_backend` over an ``openai.AsyncOpenAI``-style
    client, for ``run_tool_loop_async``. ``effort`` sets OpenRouter's ``reasoning.effort``;
    ``provider`` pins one upstream provider (``provider.order``, no fallbacks). Usage also
    carries ``prompt_tokens`` and ``cached_tokens`` (``prompt_tokens_details.cached_tokens``) so
    a driver can see whether prompt caching is taking."""
    extra: dict[str, Any] = dict(extra_body or {})
    if effort is not None:
        extra["reasoning"] = {**dict(extra.get("reasoning") or {}), "effort": effort}
    if provider is not None:
        extra["provider"] = {"order": [provider], "allow_fallbacks": False}

    async def call(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        """One tool-calling completion; retries an empty ``choices`` body, raises after three."""
        resp: Any = None
        for attempt in range(3):
            resp = await client.chat.completions.create(
                model=model,
                messages=with_cache_breakpoints(messages) if cache else messages,
                tools=tools,
                tool_choice=tool_choice_of(force_tool),
                max_tokens=max_tokens,
                extra_body=dict(extra),
            )
            if getattr(resp, "choices", None):
                break
            await asyncio.sleep(2 * (attempt + 1))
        if resp is None or not getattr(resp, "choices", None):
            raise RuntimeError(f"provider returned no choices after 3 attempts: {resp}")
        choice = resp.choices[0]
        usage = _usage_of(resp.usage)
        usage["prompt_tokens"] = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
        pdetails = getattr(resp.usage, "prompt_tokens_details", None)
        usage["cached_tokens"] = int(getattr(pdetails, "cached_tokens", 0) or 0) if pdetails else 0
        return choice.message.content or "", parse_tool_calls(choice), usage

    return call


class FakeBackend:
    """A scripted backend for tests: each step is (text, [(name, args)], output_tokens).

    A forced tool is honoured like a compliant model would: skip ahead to the first scripted
    step that calls it (any tool call, for ``FORCE_ANY``), or, when no such step exists, call
    it with empty args (``terminal`` for ``FORCE_ANY``)."""

    def __init__(
        self,
        steps: Sequence[tuple[str, Sequence[tuple[str, dict[str, Any]]], int]],
        terminal: str = "finish",
    ) -> None:
        self.steps = list(steps)
        self.terminal = terminal
        self.seen: list[list[dict[str, Any]]] = []
        self.forced: list[str | None] = []
        self._n = 0

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        self.forced.append(force_tool)
        self.seen.append(list(messages))
        if not self.steps:
            raise RuntimeError("script exhausted")
        if force_tool:  # a compliant model: skip ahead to the scripted step that calls it,
            # or, with no such step, finish with what it has (empty predictions)
            while self.steps and not any(force_tool in (FORCE_ANY, n) for n, _ in self.steps[0][1]):
                self.steps.pop(0)
            if not self.steps:
                self._n += 1
                name = self.terminal if force_tool == FORCE_ANY else force_tool
                call = {"id": f"call_{self._n}", "name": name, "args": {}}
                return "", [call], {"output_tokens": 1, "reasoning_tokens": 0}
        text, calls, toks = self.steps.pop(0)
        out = []
        for name, args in calls:
            self._n += 1
            out.append({"id": f"call_{self._n}", "name": name, "args": dict(args)})
        return text, out, {"output_tokens": toks, "reasoning_tokens": 0}


def _gemini_extra() -> dict[str, Any]:
    extra: dict[str, Any] = {"reasoning": {"effort": "low"}}  # pinned low effort for every run
    # Content filters must not decide an audit: the readouts of a geopolitics organism are full
    # of politically sensitive text.
    extra["safety_settings"] = [
        {"category": c, "threshold": "BLOCK_NONE"}
        for c in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        )
    ]
    extra["provider"] = {"allow_fallbacks": True}
    return extra


@overload
def make_backend_factory(
    kind: str, *, async_client: Literal[False] = False
) -> Callable[[str], Backend]: ...


@overload
def make_backend_factory(
    kind: str, *, async_client: Literal[True]
) -> Callable[[str], AsyncBackend]: ...


def make_backend_factory(
    kind: str, *, async_client: bool = False
) -> Callable[[str], Backend] | Callable[[str], AsyncBackend]:
    """``openrouter`` -> an OpenAI client on OpenRouter (``OPENROUTER_API_KEY`` per command);
    ``openai`` -> a plain client on ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``; ``anthropic`` ->
    Anthropic's OpenAI-compatible endpoint on ``ANTHROPIC_API_KEY``. ``async_client``
    builds ``openai.AsyncOpenAI`` and the factory returns :class:`AsyncBackend` instances."""
    from openai import AsyncOpenAI, OpenAI

    cls: Any = AsyncOpenAI if async_client else OpenAI
    if kind == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key.startswith("sk-or-"):
            raise SystemExit("OPENROUTER_API_KEY missing or malformed (pass it per command)")
        # the SDK retries 429/5xx with exponential backoff and honours Retry-After
        client = cls(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=10)
    elif kind == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY missing (pass it per command; OPENAI_BASE_URL optional)"
            )
        client = cls(max_retries=10)
    elif kind == "anthropic":
        # Anthropic's OpenAI-compatible endpoint: Claude auditors with no OpenRouter in the
        # path. Tool calls, forced tools, tool-result turns and cache_control breakpoints all
        # work on it. Model ids are Anthropic's own (claude-opus-5, claude-haiku-4-5-20251001).
        akey = os.environ.get("ANTHROPIC_API_KEY", "")
        if not akey.startswith("sk-ant-"):
            raise SystemExit("ANTHROPIC_API_KEY missing or malformed (pass it per command)")
        client = cls(api_key=akey, base_url="https://api.anthropic.com/v1/", max_retries=6)
    else:
        raise ValueError(f"unknown backend kind {kind!r} (openrouter | openai | anthropic)")

    def _is_claude(kind_: str, model: str) -> bool:
        return kind_ == "anthropic" or model.startswith("anthropic/")

    def extra_for(model: str) -> dict[str, Any]:
        extra: dict[str, Any] = {"usage": {"include": True}} if kind == "openrouter" else {}
        if kind == "openrouter" and "gemini" in model:
            extra.update(_gemini_extra())
        return extra

    def factory(model: str) -> Backend:
        """A :class:`Backend` for ``model`` on this client (provider extras chosen by model id)."""
        return openai_compatible_backend(
            client, model, extra_body=extra_for(model), cache=_is_claude(kind, model)
        )

    def async_factory(model: str) -> AsyncBackend:
        """The :class:`AsyncBackend` twin of ``factory``."""
        return async_openai_compatible_backend(
            client, model, extra_body=extra_for(model), cache=_is_claude(kind, model)
        )

    return async_factory if async_client else factory
