"""Did the lens change what the investigator concluded? The lens arm against the black-box arm.

Both arms get the same brief, the same chat tools and the same budget; only ``readouts`` differs.
For each pattern an LLM reader is shown the two ranked mechanism lists blind (A/B, order
randomised per pattern) and asked which mechanisms in each list have a counterpart in the other,
and which are unique — the same-mechanism judgement is a paraphrase question, so it is left to a
reader rather than to string overlap. Nothing here says which arm is *right*; it measures whether
the two arms told the same story, and what the lens arm said that the black-box arm did not.
"""

import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm.route import async_json_route, schema_block

SYSTEM = (
    "You compare two independent explanations of the same phenomenon. Two investigators, given "
    "the same prompt and the same behavior of a language model, each proposed a ranked list of "
    "MECHANISMS (why the model does it). Decide, for every mechanism in list A, whether list B "
    "contains the same mechanism stated differently (a counterpart), and for every mechanism in "
    "list B whether list A does. Two mechanisms are counterparts when they make the same causal "
    "claim about what the model does with this prompt, even if one is more specific or cites "
    "different evidence; they are NOT counterparts when they name different causes. Also say "
    "whether the two TOP-RANKED mechanisms make the same claim."
)
USER = (
    "<prompt>\n{prompt}\n</prompt>\n<behavior>{behavior}</behavior>\n\n"
    "<list_A>\n{a}\n</list_A>\n\n<list_B>\n{b}\n</list_B>\n\n"
    "Return, for each mechanism id in A, the id of its counterpart in B or null; for each id in B, "
    "its counterpart in A or null; whether the top mechanisms match; and one sentence on what A "
    "says that B does not (or 'nothing')."
)
SCHEMA = schema_block(
    "agreement",
    {
        "a_to_b": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "counterpart": {"type": ["string", "null"]},
                },
                "required": ["id", "counterpart"],
            },
        },
        "b_to_a": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "counterpart": {"type": ["string", "null"]},
                },
                "required": ["id", "counterpart"],
            },
        },
        "top_match": {"type": "boolean"},
        "a_only_summary": {"type": "string"},
    },
    ["a_to_b", "b_to_a", "top_match", "a_only_summary"],
)


def _fmt(mechs: Sequence[Mapping[str, Any]], tag: str, chars: int = 700) -> str:
    return "\n".join(
        f"[{tag}{i}] (confidence {m.get('confidence')}) {str(m.get('mechanism', ''))[:chars]}"
        for i, m in enumerate(mechs)
    )


def compare(
    pattern: Mapping[str, Any],
    lens: Sequence[Mapping[str, Any]],
    blackbox: Sequence[Mapping[str, Any]],
    *,
    model: str,
    seed: int = 0,
) -> dict[str, Any]:
    """One pattern: the lens arm's mechanisms against the black-box arm's, read blind."""
    flip = random.Random(f"{pattern.get('pattern_key')}:{seed}").random() < 0.5
    a, b = (blackbox, lens) if flip else (lens, blackbox)
    out = async_json_route(
        [
            (
                SYSTEM,
                USER.format(
                    prompt=str(pattern.get("prompt", ""))[:2000],
                    behavior=pattern.get("behavior_name", ""),
                    a=_fmt(a, "A"),
                    b=_fmt(b, "B"),
                ),
            )
        ],
        schema=SCHEMA,
        model=model,
        concurrency=1,
    )[0]
    if not out:
        return {"pattern_key": pattern.get("pattern_key"), "error": "reader returned nothing"}
    a_to_b = {str(x["id"]): x["counterpart"] for x in out.get("a_to_b") or []}
    b_to_a = {str(x["id"]): x["counterpart"] for x in out.get("b_to_a") or []}
    lens_map, bb_map = (b_to_a, a_to_b) if flip else (a_to_b, b_to_a)
    lens_tag, bb_tag = ("B", "A") if flip else ("A", "B")
    lens_matched = [i for i in range(len(lens)) if lens_map.get(f"{lens_tag}{i}")]
    bb_matched = [i for i in range(len(blackbox)) if bb_map.get(f"{bb_tag}{i}")]
    return {
        "pattern_key": pattern.get("pattern_key"),
        "behavior_id": pattern.get("behavior_id"),
        "n_lens": len(lens),
        "n_blackbox": len(blackbox),
        "lens_with_counterpart": len(lens_matched),
        "blackbox_with_counterpart": len(bb_matched),
        "lens_only": [
            {"mechanism": lens[i].get("mechanism"), "confidence": lens[i].get("confidence")}
            for i in range(len(lens))
            if i not in lens_matched
        ],
        "blackbox_only": [
            {"mechanism": blackbox[i].get("mechanism"), "confidence": blackbox[i].get("confidence")}
            for i in range(len(blackbox))
            if i not in bb_matched
        ],
        "top_match": bool(out.get("top_match")),
        "lens_only_summary": str(out.get("a_only_summary", "")) if not flip else "",
        "blackbox_only_summary": str(out.get("a_only_summary", "")) if flip else "",
        "order": "blackbox_first" if flip else "lens_first",
        "model": model,
    }


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    n_lens = sum(r["n_lens"] for r in ok)
    n_bb = sum(r["n_blackbox"] for r in ok)
    return {
        "n_patterns": len(ok),
        "top_match": sum(1 for r in ok if r["top_match"]),
        "lens_mechanisms": n_lens,
        "lens_with_counterpart": sum(r["lens_with_counterpart"] for r in ok),
        "blackbox_mechanisms": n_bb,
        "blackbox_with_counterpart": sum(r["blackbox_with_counterpart"] for r in ok),
        "lens_only_total": sum(len(r["lens_only"]) for r in ok),
        "blackbox_only_total": sum(len(r["blackbox_only"]) for r in ok),
        "errors": len(rows) - len(ok),
    }


def dumps(blob: Mapping[str, Any]) -> str:
    return json.dumps(blob, ensure_ascii=False, indent=1)


__all__ = ["SCHEMA", "compare", "dumps", "summarise"]
