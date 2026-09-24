"""LLM backends for the investigator: any OpenAI-compatible tool-calling endpoint, plus a
scripted :class:`FakeBackend` for offline tests and plumbing smokes.

``make_backend_factory("openrouter")`` builds the OpenRouter client (``OPENROUTER_API_KEY``);
``make_backend_factory("openai")`` builds a plain client for whatever ``OPENAI_API_KEY`` /
``OPENAI_BASE_URL`` name (a local vLLM/SGLang auditor, a proxy, OpenAI itself).
"""

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .loop import Backend


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
                tool_choice=(
                    {"type": "function", "function": {"name": force_tool}} if force_tool else "auto"
                ),
                max_tokens=max_tokens,
                extra_body=dict(extra_body or {}),
            )
            if getattr(resp, "choices", None):
                break
            time.sleep(2 * (attempt + 1))
        if resp is None or not getattr(resp, "choices", None):
            raise RuntimeError(f"provider returned no choices after 3 attempts: {resp}")
        choice = resp.choices[0]
        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        return (
            choice.message.content or "",
            calls,
            {
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "reasoning_tokens": reasoning,
            },
        )

    return call


class FakeBackend:
    """A scripted backend for tests: each step is (text, [(name, args)], output_tokens)."""

    def __init__(
        self, steps: Sequence[tuple[str, Sequence[tuple[str, dict[str, Any]]], int]]
    ) -> None:
        self.steps = list(steps)
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
            while self.steps and all(n != force_tool for n, _ in self.steps[0][1]):
                self.steps.pop(0)
            if not self.steps:
                self._n += 1
                call = {"id": f"call_{self._n}", "name": force_tool, "args": {}}
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


def make_backend_factory(kind: str, *, max_tokens: int = 4000) -> Callable[[str], Backend]:
    """``openrouter`` -> an OpenAI client on OpenRouter (``OPENROUTER_API_KEY`` per command);
    ``openai`` -> a plain client on ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``. ``max_tokens`` caps
    one turn's output; a ``finish`` whose arguments outgrow it comes back as invalid JSON."""
    from openai import OpenAI

    if kind == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key.startswith("sk-or-"):
            raise SystemExit("OPENROUTER_API_KEY missing or malformed (pass it per command)")
        # the SDK retries 429/5xx with exponential backoff and honours Retry-After
        client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=10)
    elif kind == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY missing (pass it per command; OPENAI_BASE_URL optional)"
            )
        client = OpenAI(max_retries=10)
    else:
        raise ValueError(f"unknown backend kind {kind!r} (openrouter | openai)")

    def factory(model: str) -> Backend:
        """A :class:`Backend` for ``model`` on this client (provider extras chosen by model id)."""
        extra: dict[str, Any] = {"usage": {"include": True}} if kind == "openrouter" else {}
        if kind == "openrouter" and "gemini" in model:
            extra.update(_gemini_extra())
        return openai_compatible_backend(
            client,
            model,
            extra_body=extra,
            max_tokens=max_tokens,
            cache=model.startswith("anthropic/"),
        )

    return factory
