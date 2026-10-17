"""Pass@1 (old pass judge) per reader, overall and by difficulty strata defined WITHOUT the
compared arms.

difficulty = the no-tools plain answers' primary-cause recall (point eval v1: mean (score-1)/9 over
the primary points, averaged over the 3 plain answers). Strata: overall | plain-graded subset |
hardest third (lowest plain recall) | hard = plain recall < 20%. Also shown (biased, flagged):
'transcript fails' = transcript pass@1 <= 0.2 (selection on transcript draws favours other arms).
Each row: per-arm pass@1 and each reader minus transcript with a paired bootstrap 95% CI (1000
draws, seed 0, one generator per row).

    python strata.py [batches ...] [--arms transcript,olens_rl_B,...] [--runs DIR] [--points DIR]
"""

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
from common import complete, judged, paired_bootstrap  # noqa: E402

DEFAULT_ARMS = "transcript,olens_rl_B,jlens_B,mlnla_B"


def plain_recall(points: Path, batches: list[str]) -> dict[tuple[str, str], float]:
    """(batch, pid) -> the plain answers' primary-cause recall (incidents with 3 plain grades)."""
    out = {}
    for b in batches:
        for f in sorted((points / b / "tags").glob("*.json")):
            pid = f.name[:-5]
            prim = json.loads(f.read_text())["primary"]
            gs = [
                json.loads(x.read_text())["point_scores"]
                for x in sorted((points / b / "grades/plain").glob(f"{pid}.d*.json"))[:3]
            ]
            idx = [i for i, t in enumerate(prim) if t]
            if len(gs) == 3 and idx and all(len(g) == len(prim) for g in gs):
                out[(b, pid)] = sum(sum(g[i] - 1 for i in idx) / (9 * len(idx)) for g in gs) / 3
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("batches", nargs="*", default=paths.BATCHES)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--runs", type=Path, default=paths.RUNS)
    ap.add_argument("--points", type=Path, default=paths.POINTS)
    a = ap.parse_args()
    arms = a.arms.split(",")
    d = judged(a.runs, a.batches, arms)
    plain = plain_recall(a.points, a.batches)
    ks = complete(d, arms)

    def p1(arm: str, k: tuple[str, str]) -> float:
        return sum(d[k][arm]) / 5

    def show(name: str, sub: list[tuple[str, str]]) -> None:
        if not sub:
            return
        rng = random.Random(0)
        line = f"{name:34s} n={len(sub):4d} | " + "  ".join(
            f"{arm.split('_')[0]} {100 * sum(p1(arm, k) for k in sub) / len(sub):5.1f}"
            for arm in arms
        )
        for arm in arms[1:]:
            m, lo, hi = paired_bootstrap([p1(arm, k) - p1(arms[0], k) for k in sub], rng, 1000)
            line += f" | {arm.split('_')[0]}−tr {100 * m:+.1f} [{100 * lo:+.0f},{100 * hi:+.0f}]"
        print(line)

    show("overall", ks)
    pk = sorted([k for k in ks if k in plain], key=lambda k: plain[k])
    show("plain-graded subset", pk)
    show("hardest third (plain recall)", pk[: len(pk) // 3])
    show("hard: plain recall < 20%", [k for k in pk if plain[k] < 0.2])
    show("[biased] transcript pass@1 <= 0.2", [k for k in ks if p1(arms[0], k) <= 0.2])


if __name__ == "__main__":
    main()
