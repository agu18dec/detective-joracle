"""Clustering the mechanisms across runs: what recurs, and where it does not.

Every run explains ONE pattern, and a mechanism that only ever fits one prompt explains that
prompt rather than the behavior. This pass groups the mechanisms into themes and keeps each
theme's members attributed, so a reader can go from "the model adopts the user's framing" back to
the three patterns and the cells that claim it. It is one LLM call over the finished records — no
judging, no scoring: nothing here verifies a mechanism, it only counts how often one was proposed.
"""

import json
from collections.abc import Sequence
from typing import Any

from ..llm.route import async_json_route, schema_block

SYSTEM = (
    "You are organising the findings of an interpretability study. Researchers explained, one "
    "prompt at a time, why a language model shows a documented unwanted behavior. Group their "
    "proposed mechanisms into themes. A theme is a mechanism stated at the level that its "
    "members actually share — not a topic label ('drunk driving') and not a restatement of the "
    "behavior ('the model is unsafe'). Keep the researchers' own distinctions: if two mechanisms "
    "differ in what they claim the model does, they are different themes even when they concern "
    "the same prompt. Every member must be one of the given items, referenced by its item id; "
    "do not invent members and do not assign an item to more than one theme."
)

USER = (
    "Here are the mechanisms, one per line, each with an item id, the behavior it was proposed "
    "for, and the pattern it came from.\n\n{items}\n\n"
    "Return {n_max} themes at most, ordered by how many items they cover. For each theme give a "
    "short name (at most 8 words), a description of the mechanism in two or three sentences, and "
    "the item ids that belong to it. Items that fit no theme go in no theme."
)

SCHEMA: dict[str, Any] = schema_block(
    "themes",
    {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "description", "item_ids"],
            },
        }
    },
    ["clusters"],
)


def collect(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten every finished run's mechanisms into items the clusterer can reference."""
    items: list[dict[str, Any]] = []
    for blob in records:
        rec = blob.get("record") or {}
        pat = blob.get("pattern") or {}
        for i, m in enumerate((rec.get("result") or {}).get("mechanisms") or []):
            items.append(
                {
                    "item_id": f"{pat.get('pattern_key', '?')}#{i}",
                    "pattern_key": pat.get("pattern_key", ""),
                    "behavior_id": pat.get("behavior_id", ""),
                    "behavior_name": pat.get("behavior_name", ""),
                    "group_summary": pat.get("group_summary", ""),
                    "mechanism": str(m.get("mechanism", "")),
                    "evidence": str(m.get("evidence", "")),
                    "readout_cells": str(m.get("readout_cells", "")),
                    "confidence": m.get("confidence"),
                    "would_test_by": str(m.get("would_test_by", "")),
                }
            )
    return items


def cluster(
    items: Sequence[dict[str, Any]],
    *,
    model: str = "google/gemini-3.8-flash",
    n_max: int = 12,
    mechanism_chars: int = 600,
) -> dict[str, Any]:
    """Group the mechanisms into themes; members keep their item ids, so nothing is unattributed."""
    if not items:
        return {"clusters": [], "model": model, "n_items": 0, "n_records": 0}
    lines = "\n".join(
        f"[{it['item_id']}] behavior={it['behavior_name']!r} pattern={it['group_summary'][:90]!r} "
        f"mechanism={it['mechanism'][:mechanism_chars]!r}"
        for it in items
    )
    out = async_json_route(
        [(SYSTEM, USER.format(items=lines, n_max=n_max))],
        schema=SCHEMA,
        model=model,
        concurrency=1,
    )[0]
    by_id = {it["item_id"]: it for it in items}
    clusters: list[dict[str, Any]] = []
    for c in (out or {}).get("clusters") or []:
        members = [by_id[str(i)] for i in c.get("item_ids") or [] if str(i) in by_id]
        if not members:
            continue
        clusters.append(
            {
                "name": str(c.get("name", "")),
                "description": str(c.get("description", "")),
                "behavior_ids": sorted({m["behavior_id"] for m in members}),
                "members": [
                    {
                        "pattern_key": m["pattern_key"],
                        "behavior_id": m["behavior_id"],
                        "mechanism": m["mechanism"],
                        "evidence": m["evidence"],
                        "readout_cells": m["readout_cells"],
                        "would_test_by": m["would_test_by"],
                    }
                    for m in members
                ],
            }
        )
    clusters.sort(key=lambda c: -len(c["members"]))
    clustered = {m["pattern_key"] + "#" for c in clusters for m in c["members"]}
    return {
        "clusters": clusters,
        "model": model,
        "n_items": len(items),
        "n_clustered": sum(len(c["members"]) for c in clusters),
        "n_patterns": len({it["pattern_key"] for it in items}),
        "unclustered_patterns": sorted(
            {it["pattern_key"] for it in items} - {k[:-1] for k in clustered}
        ),
    }


def dumps(blob: dict[str, Any]) -> str:
    return json.dumps(blob, ensure_ascii=False, indent=1)


__all__ = ["SCHEMA", "cluster", "collect", "dumps"]
