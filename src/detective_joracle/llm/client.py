"""A minimal OpenAI-compatible structured-JSON caller.

:func:`async_json` takes a batch of ``(system, user)`` prompt pairs plus a JSON schema, runs them
concurrently against the Chat Completions API (``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``), and
returns the parsed object per item — or ``None`` on any API error, so a judge outage degrades to
"unjudged" and never crashes a run. The SDK is imported lazily so importing the package (and
running the offline tests) never needs a key.

Judge calls run at the model's DEFAULT sampling temperature unless one is given, so verdicts
carry small rerun-to-rerun variance; treat single-digit deltas between reruns as noise.
"""

import asyncio
import json
import math
import os
from collections.abc import Sequence
from typing import Any

DEFAULT_MODEL = "google/gemini-3.8-flash"  # an OpenRouter id; any Chat Completions model works
CONCURRENCY = 32  # bound in-flight requests; firing all N at once exhausts the pool and stalls


def schema_block(name: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """A strict ``json_schema`` response-format block (the Chat Completions structured shape)."""
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        },
    }


def _sampling(temperature: float | None) -> dict[str, float]:
    """Extra request kwargs for an explicit sampling temperature; empty = provider default."""
    return {} if temperature is None else {"temperature": temperature}


async def _one(
    client: Any,
    system: str,
    user: str,
    schema: dict[str, Any],
    model: str,
    temperature: float | None = None,
    timeout: float = 90.0,
) -> dict[str, Any] | None:
    """One structured completion; returns the parsed dict, or ``None`` on any error.

    Providers without schema enforcement (JSON mode only) reject the strict ``response_format``
    with a 400 naming it; the call is then retried ONCE in ``json_object`` mode with the schema
    spelled out in the system prompt. Callers validate the payload shape anyway.
    """
    fmt: dict[str, Any] = {"type": "json_schema", "json_schema": schema}
    sys_prompt = system
    for attempt in range(2):
        try:
            resp = await client.chat.completions.create(
                timeout=timeout,
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user},
                ],
                response_format=fmt,
                **_sampling(temperature),
            )
            data: dict[str, Any] = json.loads(resp.choices[0].message.content)
            return data
        except Exception as e:  # any API/parse error degrades to None
            if attempt == 0 and _wants_json_object_fallback(e):
                fmt = {"type": "json_object"}
                sys_prompt = system + _schema_suffix(schema)
                continue
            print(f"  llm error: {type(e).__name__}: {e}")
            return None
    return None


def _wants_json_object_fallback(exc: Exception) -> bool:
    """A 400 that names the response_format / json_schema field = provider lacks schema mode."""
    if getattr(exc, "status_code", None) != 400:
        return False
    msg = str(exc).lower()
    return "response_format" in msg or "json_schema" in msg


def _schema_suffix(schema: dict[str, Any]) -> str:
    inner = schema.get("schema", schema)
    return (
        "\n\nRespond with a single JSON object and nothing else, matching exactly this JSON "
        f"schema:\n{json.dumps(inner)}"
    )


def _openai_timeout() -> float:
    """Per-call HTTP timeout (``JUDGE_HTTP_TIMEOUT``, seconds, default 90).

    Validated once in ``async_json`` before its degrade-to-``None`` guard, and again here, once
    per batch — never inside the per-call ``try``. A bad value swallowed by either handler would
    return ``None`` for every call and write a plausible all-zero artifact.
    """
    raw = os.environ.get("JUDGE_HTTP_TIMEOUT", "90")
    try:
        secs = float(raw)
    except ValueError:
        raise ValueError(f"JUDGE_HTTP_TIMEOUT={raw!r} is not a number") from None
    if not secs > 0 or math.isinf(secs):
        raise ValueError(f"JUDGE_HTTP_TIMEOUT={raw!r} must be a positive, finite number of seconds")
    return secs


async def _batch(
    items: list[tuple[str, str]],
    schema: dict[str, Any],
    model: str,
    concurrency: int,
    api_keys: Sequence[str] | None = None,
    temperature: float | None = None,
) -> list[dict[str, Any] | None]:
    # One client (+ own in-flight cap) per key; items round-robin across them. Rate limits are
    # per-org, so two keys from different orgs double the ceiling — ``concurrency`` is PER KEY.
    from openai import AsyncOpenAI  # lazy: only the live path needs the dep

    keys: Sequence[str | None] = api_keys if api_keys else [None]  # None -> env key
    clients = [AsyncOpenAI(api_key=k) for k in keys]
    sems = [asyncio.Semaphore(concurrency) for _ in clients]
    timeout = _openai_timeout()  # read + validate ONCE, before any call

    async def _guarded(i: int, system: str, user: str) -> dict[str, Any] | None:
        j = i % len(clients)
        async with sems[j]:  # cap concurrent calls — an unbounded gather hangs the judge
            # ``_one`` is looked up at call time so ``openrouter.install`` can replace it
            return await _one(clients[j], system, user, schema, model, temperature, timeout)

    try:
        return await asyncio.gather(*(_guarded(i, s, u) for i, (s, u) in enumerate(items)))
    finally:
        for client in clients:
            await client.close()


def async_json(
    items: list[tuple[str, str]],
    *,
    schema: dict[str, Any],
    model: str = DEFAULT_MODEL,
    concurrency: int = CONCURRENCY,
    api_keys: Sequence[str] | None = None,
    temperature: float | None = None,
) -> list[dict[str, Any] | None]:
    """Run ``(system, user)`` prompt pairs concurrently (at most ``concurrency`` in flight); each
    result is the parsed object or None.

    Returns all-``None`` (never raises) if the client/key is unavailable. The ONE exception is a
    malformed ``JUDGE_HTTP_TIMEOUT``: that is a config error, validated up front and allowed to
    raise.
    """
    if not items:
        return []
    _openai_timeout()  # raises on a malformed value, BEFORE the degrade-to-None guard below
    try:
        return asyncio.run(_batch(items, schema, model, concurrency, api_keys, temperature))
    except Exception as e:  # missing key / no event loop / import error
        print(f"llm unavailable ({type(e).__name__}: {e}); results are None")
        return [None] * len(items)
