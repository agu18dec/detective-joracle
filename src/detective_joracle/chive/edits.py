"""The CHIVE yes/no items: would this edit to the conversation change the behaviour?

Each edit was applied to the conversation and the target model re-run 30 times. YES items are the
claim bank's supported edits (``ground_truth.movers``: moved the behaviour rate by 50 points or
more); NO items are its inert edits (``ruled_out`` with kind inert and basis claim_bank: moved it
by 15 points or less). Nothing lies in between, by construction. Inert items of basis
experiment_le_15pp are left out: their text is the investigator's summary and states the result.
The claim-bank edit texts describe the change, not its outcome (checked on all 737: the few that
mention numbers or "the assistant now ..." describe the edited conversation).
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .data import Incident


@dataclass(frozen=True)
class EditItem:
    id: str  # E1, E2, ... in a fixed per-incident order
    text: str
    label: bool  # True: the edit moved the behaviour rate by >= 50 points
    effect_pp: float


def edit_items(inc: Incident) -> list[EditItem]:
    gt = inc.ground_truth
    raw = [(m["edit"], True, float(m.get("effect_pp") or 0.0)) for m in gt.get("movers", [])]
    raw += [
        (x["edit"], False, float(x.get("effect_pp") or 0.0))
        for x in gt.get("ruled_out", [])
        if x.get("kind") == "inert" and x.get("basis") == "claim_bank"
    ]
    raw.sort(key=lambda r: hashlib.sha1(f"{inc.prompt_id}:{r[0]}".encode()).hexdigest())
    return [EditItem(f"E{i}", t, lab, pp) for i, (t, lab, pp) in enumerate(raw, start=1)]


def score_items(items: list[EditItem], probs: dict[str, float]) -> dict[str, Any]:
    """Per-incident scores of predicted P(YES): Brier, log loss (p clipped to [0.01, 0.99]),
    accuracy at 0.5; the items themselves, for pooled AUC."""
    ps = [float(probs[it.id]) for it in items]
    ys = [it.label for it in items]
    clip = [min(0.99, max(0.01, p)) for p in ps]
    n = len(items)
    return {
        "items": [
            {"id": it.id, "label": it.label, "p": p, "effect_pp": it.effect_pp}
            for it, p in zip(items, ps, strict=True)
        ],
        "brier": sum((p - y) ** 2 for p, y in zip(ps, ys, strict=True)) / n,
        "logloss": -sum(math.log(c if y else 1 - c) for c, y in zip(clip, ys, strict=True)) / n,
        "accuracy": sum((p >= 0.5) == y for p, y in zip(ps, ys, strict=True)) / n,
        "n": n,
    }
