"""One plain sentence per mechanism, for the page.

Investigators write mechanisms for other investigators: dense, hedged, full of the run's own
vocabulary. A reader of the viewer wants the claim in one sentence first and the long form on
demand. This pass rewrites every mechanism of every run into at most ~30 plain words that keep the
causal claim (what in the prompt does what to the model, and what that produces) and drop the
rest. It is a rewrite of the investigator's words, not a new judgement: nothing may be added.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm.route import async_json_route, schema_block

SYSTEM = (
    "Rewrite a researcher's explanation of why a language model misbehaves on a prompt as ONE "
    "plain-English sentence of at most 30 words, for a general reader. Keep the causal claim — "
    "what in the prompt does what to the model, and what that produces — and drop jargon, hedges, "
    "evidence and citations. Do not add anything the original does not say. Refer to the model as "
    "'the model'."
)
USER = "<behavior>{behavior}</behavior>\n<original>\n{mechanism}\n</original>\n\nReturn the one-sentence rewrite."
SCHEMA = schema_block("short", {"short": {"type": "string"}}, ["short"])
MAX_WORDS = 34


def rewrite(
    items: Sequence[tuple[str, str, str]],
    *,
    model: str,
    concurrency: int = 16,
) -> dict[str, str]:
    """``(key, behavior, mechanism)`` rows in; ``{key: short}`` out for the rows that came back."""
    outs = async_json_route(
        [(SYSTEM, USER.format(behavior=b, mechanism=m[:2500])) for _, b, m in items],
        schema=SCHEMA,
        model=model,
        concurrency=concurrency,
    )
    result: dict[str, str] = {}
    for (key, _, _), o in zip(items, outs, strict=True):
        short = str((o or {}).get("short", "")).strip()
        if not short:
            continue
        words = short.split()
        # keep the promise to the reader; the long form is one click away
        if len(words) > MAX_WORDS:
            short = " ".join(words[:MAX_WORDS]).rstrip(",;:") + "…"
        result[key] = short
    return result


def keys_for(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    """Every mechanism of every run as ``(pattern_key#arm#index, behavior, mechanism)``."""
    rows: list[tuple[str, str, str]] = []
    for blob in records:
        rec = blob.get("record") or {}
        pat = blob.get("pattern") or {}
        arm = str(rec.get("arm", "olens"))
        for i, m in enumerate((rec.get("result") or {}).get("mechanisms") or []):
            text = str(m.get("mechanism", "")).strip()
            if text:
                rows.append(
                    (f"{pat.get('pattern_key')}#{arm}#{i}", str(pat.get("behavior_name", "")), text)
                )
    return rows


def dumps(blob: Mapping[str, Any]) -> str:
    return json.dumps(blob, ensure_ascii=False, indent=0)


__all__ = ["MAX_WORDS", "SCHEMA", "dumps", "keys_for", "rewrite"]
