"""Route :mod:`client` through OpenRouter, tuned for a fast bulk judge.

``client`` builds a plain ``AsyncOpenAI`` client, which honours ``OPENAI_BASE_URL``; this module
points it at OpenRouter and swaps the per-call function for one that

* sends ``reasoning`` (some models cannot turn reasoning off, so the fastest legal setting is
  ``{"effort": "minimal"}``),
* asks OpenRouter to route only to providers that honour the structured-output schema,
* paces requests under a process-wide requests-per-minute cap (``OPENROUTER_RPM``, default 240)
  — a burst past a model's platform limit turns a large share of cells into unjudged ``None``s,
* retries transient failures (429 / 5xx / timeouts / connection) patiently with jittered backoff,
* tallies ``usage.cost`` so the run reports what it spent.

The key is ``OPENROUTER_API_KEY``, passed per command (never exported). ``install()`` must run
before the first ``async_json`` call; ``route.async_json_route`` does that for you.
"""

import asyncio
import json
import os
import random
import time
from typing import Any

from . import client as llm_client

OPENROUTER = "https://openrouter.ai/api/v1"
SPEND = {"usd": 0.0, "calls": 0, "errors": 0, "retries": 0}
_NEXT = [0.0]  # monotonic time of the next free request slot (process-wide, loop-independent)


async def _pace(rpm: float) -> None:
    """Reserve the next request slot. Reading and advancing ``_NEXT`` has no await in between, so
    it is atomic under asyncio; the value is plain time, so it survives each ``asyncio.run``."""
    now = time.monotonic()
    slot = max(now, _NEXT[0])
    _NEXT[0] = slot + 60.0 / rpm
    if slot > now:
        await asyncio.sleep(slot - now)


def install(reasoning: dict[str, Any] | None = None, max_tokens: int = 8000) -> None:
    """Point ``client`` at OpenRouter and replace its per-call function with the paced, retrying one.

    Raises ``SystemExit`` if ``OPENROUTER_API_KEY`` is missing or not an ``sk-or-`` key."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key.startswith("sk-or-"):
        raise SystemExit("OPENROUTER_API_KEY missing or malformed (pass it per command)")
    os.environ["OPENAI_API_KEY"] = key  # process-local: the openai SDK reads it at client init
    os.environ["OPENAI_BASE_URL"] = OPENROUTER
    os.environ.setdefault("JUDGE_HTTP_TIMEOUT", "180")
    rung = reasoning if reasoning is not None else {"effort": "minimal"}
    rpm = float(os.environ.get("OPENROUTER_RPM", "240"))

    async def _one(
        client: Any,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        temperature: float | None = None,
        timeout: float = 180.0,
    ) -> dict[str, Any] | None:
        # lone surrogates (from a detokenizer) make the API reject the whole request with a 400;
        # deterministic, so retries cannot help — sanitize instead
        system = system.encode("utf-8", "replace").decode("utf-8")
        user = user.encode("utf-8", "replace").decode("utf-8")
        for attempt in range(12):
            await _pace(rpm)
            try:
                resp = await client.chat.completions.create(
                    timeout=timeout,
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_schema", "json_schema": schema},
                    extra_body={
                        "reasoning": rung,
                        "usage": {"include": True},
                        "provider": {"require_parameters": True},
                    },
                    **({} if temperature is None else {"temperature": temperature}),
                )
                SPEND["calls"] += 1
                usage = getattr(resp, "usage", None)
                cost = getattr(usage, "cost", None) if usage is not None else None
                if cost is None and usage is not None:
                    cost = (getattr(usage, "model_extra", None) or {}).get("cost")
                SPEND["usd"] += float(cost or 0.0)
                content = resp.choices[0].message.content or ""
                content = content.strip()
                if content.startswith("```"):
                    content = content.strip("`").removeprefix("json").strip()
                data: dict[str, Any] = json.loads(content)
                return data
            except Exception as e:
                status = getattr(e, "status_code", None)
                name = type(e).__name__
                transient = (
                    status in (408, 409, 429, 500, 502, 503, 504, 529)
                    or "Timeout" in name
                    or "Connection" in name
                    or "JSONDecodeError" in name  # truncated/garbled body: resample
                )
                if transient and attempt < 11:
                    SPEND["retries"] += 1
                    cap = 90.0 if status == 429 else 30.0
                    await asyncio.sleep(min(cap, 2.0 * 2**attempt) * (0.5 + random.random()))
                    continue
                SPEND["errors"] += 1
                print(f"  llm error: {name}: {str(e)[:200]}")
                return None
        return None

    llm_client._one = _one


def preflight(model: str) -> None:
    """One live call; a capped or wrong key fails loudly here instead of writing all-None."""
    s = llm_client.schema_block("t", {"a": {"type": "integer"}}, ["a"])
    out = llm_client.async_json(
        [("Reply JSON.", 'Return {"a":1}')], schema=s, model=model, concurrency=1
    )
    if out != [{"a": 1}]:
        raise SystemExit(f"preflight failed for {model}: {out}")
    print(f"[openrouter] preflight ok ({model})")


def report() -> str:
    """One line of spend/retry/error counters for the process so far."""
    return (
        f"[openrouter] calls={SPEND['calls']} retries={SPEND['retries']} "
        f"errors={SPEND['errors']} spend=${SPEND['usd']:.2f}"
    )
