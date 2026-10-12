"""CHIVE's binary claim eval, in its own form (adamkarvonen/chive, chive/binary_eval).

One claim per counterfactual edit: BEHAVIOR, BASELINE (~k%), INTERVENTION, and the unsigned
PREDICTION that the edit moves the behaviour rate by at least 30 points in either direction. TRUE
claims are the claim bank's supported edits (moved >= 50 pp, ``ground_truth.movers``), FALSE ones
its inert edits (<= 15 pp, ``ruled_out`` kind inert, basis claim_bank); 30 pp sits in the gap, so
the label is unambiguous. The set is balanced 50/50 (seeded subsample of the larger class), as in
CHIVE's eval sets. Scored by pooled AUROC (CHIVE's headline), within-incident AUROC, Brier and
accuracy at 0.5.
"""

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .data import Incident

THRESHOLD_PP = 30


@dataclass(frozen=True)
class Claim:
    claim_id: str  # <prompt_id>:<edit hash>
    prompt_id: str
    label: bool  # True: the edit moved the behaviour rate by >= 50 points
    effect_pp: float
    intervention: str
    statement: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def statement(inc: Incident, intervention: str) -> str:
    """CHIVE's claim text (build_claims.compose_statement): baseline as a percentage, unsigned."""
    base = round(inc.k / inc.n * 100)
    return (
        f'BEHAVIOR (the completion is scored "yes" when): {inc.behaviour}\n\n'
        f"BASELINE: on the original prompt at temperature 1.0, this behavior occurs in "
        f"~{base}% of completions.\n\n"
        f"INTERVENTION: {intervention}\n\n"
        f"PREDICTION: applying this intervention (everything else unchanged) would change how "
        f"often the behavior occurs by at least {THRESHOLD_PP} percentage points (in either "
        f"direction) from the ~{base}% baseline."
    )


def build_claims(incs: list[Incident], seed: int = 0) -> list[Claim]:
    out: list[Claim] = []
    for inc in incs:
        gt = inc.ground_truth
        edits = [(m["edit"], True, float(m.get("effect_pp") or 0)) for m in gt.get("movers", [])]
        edits += [
            (x["edit"], False, float(x.get("effect_pp") or 0))
            for x in gt.get("ruled_out", [])
            if x.get("kind") == "inert" and x.get("basis") == "claim_bank"
        ]
        for text, label, pp in edits:
            h = hashlib.sha1(text.encode()).hexdigest()[:8]
            out.append(
                Claim(f"{inc.prompt_id}:{h}", inc.prompt_id, label, pp, text, statement(inc, text))
            )
    yes = [c for c in out if c.label]
    no = [c for c in out if not c.label]
    rng = random.Random(seed)
    big, small = (yes, no) if len(yes) > len(no) else (no, yes)
    keep = small + rng.sample(big, len(small))
    return sorted(keep, key=lambda c: c.claim_id)


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
