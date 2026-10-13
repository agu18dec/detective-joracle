"""CHIVE's binary claim eval (adamkarvonen/chive, chive/binary_eval): the claims of CHIVE's own
eval sets (claim bank -> concreteness >= 3 -> balanced 50/50), each a BEHAVIOR / BASELINE /
INTERVENTION / PREDICTION (moves the rate by >= 30 points) statement. Scored by pooled AUROC
(CHIVE's headline), within-incident AUROC, Brier and accuracy at 0.5."""

import math
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Claim:
    claim_id: str  # <prompt_id>:<edit hash>
    prompt_id: str
    label: bool  # True: the edit moved the behaviour rate by >= 50 points
    effect_pp: float
    intervention: str
    statement: str
    question: str = ""  # CHIVE's investigator question ("Why did you ...?")
    transcript: str = ""  # CHIVE's readable transcript

    @staticmethod
    def from_chive(row: dict[str, Any]) -> "Claim":
        """A row of CHIVE's eval_set_*.jsonl (chive/binary_eval/build_eval_set.py), verbatim."""
        return Claim(
            claim_id=row["claim_id"],
            prompt_id=row["prompt_id"],
            label=bool(row["is_true"]),
            effect_pp=float(row["effect_pp"]),
            intervention=row["intervention"],
            statement=row["statement"],
            question=row["question"],
            transcript=row["transcript"],
        )

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- scoring

Scored = dict[str, tuple[list[float], bool, str]]  # claim_id -> (p per draw, label, prompt_id)


def auc(pairs: list[tuple[float, bool]]) -> float:
    """Mann-Whitney AUROC; ties count half; nan with one class."""
    pos = [p for p, y in pairs if y]
    neg = [p for p, y in pairs if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def metrics(scored: Scored, claim_ids: list[str]) -> dict[str, float]:
    """Pooled AUROC and within-incident AUROC on the draw-mean p; Brier, log loss and accuracy
    per draw, averaged over claims."""
    rows = [scored[c] for c in claim_ids if c in scored]
    if not rows:
        return {k: float("nan") for k in ("auroc", "within_auroc", "brier", "logloss", "acc")}
    mean = [(sum(ps) / len(ps), y, pid) for ps, y, pid in rows]
    by_inc: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for p, y, pid in mean:
        by_inc[pid].append((p, y))
    within = [a for a in (auc(v) for v in by_inc.values()) if a == a]

    def per_draw(f: Callable[[float, bool], float]) -> float:
        return sum(sum(f(p, y) for p in ps) / len(ps) for ps, y, _ in rows) / len(rows)

    def ll(p: float, y: bool) -> float:
        c = min(0.99, max(0.01, p))
        return -math.log(c if y else 1 - c)

    return {
        "auroc": auc([(p, y) for p, y, _ in mean]),
        "within_auroc": sum(within) / len(within) if within else float("nan"),
        "brier": per_draw(lambda p, y: (p - y) ** 2),
        "logloss": per_draw(ll),
        "acc": per_draw(lambda p, y: float((p >= 0.5) == y)),
    }


def paired(a: Scored, b: Scored, claim_ids: list[str], metric: str) -> str:
    """a minus b on claims both scored; seeded bootstrap over incidents (claims cluster)."""
    common = [c for c in claim_ids if c in a and c in b]
    by_inc: dict[str, list[str]] = defaultdict(list)
    for c in common:
        by_inc[a[c][2]].append(c)
    incs = sorted(by_inc)
    if not incs:
        return "-"
    point = metrics(a, common)[metric] - metrics(b, common)[metric]
    rng = random.Random(0)
    diffs = []
    for _ in range(2000):
        smp = [c for i in (rng.choice(incs) for _ in incs) for c in by_inc[i]]
        d = metrics(a, smp)[metric] - metrics(b, smp)[metric]
        if d == d:
            diffs.append(d)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("*" if lo > 0 or hi < 0 else "")
