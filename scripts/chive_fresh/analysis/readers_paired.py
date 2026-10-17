"""Paired pass@1 / pass@5 (old pass judge) for transcript, OLens, J-lens and multi-layer NLA, plus
optional extra arms, on incidents where every listed arm has 5 clean judged base draws.

    python readers_paired.py [batches ...] [--single] [--arms a,b,c] [--runs DIR]

Prints pass@1 / pass@5 per arm, each reader minus transcript and OLens (olens_rl_B) minus each
other reader with a paired incident-bootstrap 95% CI (2000 draws, seed 0). ``--single`` adds
olens_single (single-readout OLens); ``--arms`` replaces the arm list (first arm = reference).
"""

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
from common import complete, judged, paired_bootstrap  # noqa: E402

DEFAULT_ARMS = ["transcript", "olens_rl_B", "jlens_B", "mlnla_B"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("batches", nargs="*", default=paths.BATCHES)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--runs", type=Path, default=paths.RUNS)
    a = ap.parse_args()
    arms = a.arms.split(",") if a.arms else DEFAULT_ARMS + (["olens_single"] if a.single else [])
    d = judged(a.runs, a.batches, arms)
    ks = complete(d, arms)
    print(
        f"paired incidents: {len(ks)}  ({', '.join(f'{b} {sum(k[0] == b for k in ks)}' for b in a.batches)})"
    )

    def p1(arm: str, k: tuple[str, str]) -> float:
        return sum(d[k][arm]) / 5

    for arm in arms:
        print(
            f"  {arm:13s} pass@1 {100 * sum(p1(arm, k) for k in ks) / len(ks):.1f}%  "
            f"pass@5 {100 * sum(any(d[k][arm]) for k in ks) / len(ks):.1f}%"
        )
    rng = random.Random(0)

    def ci(x: str, y: str) -> str:
        m, lo, hi = paired_bootstrap([p1(x, k) - p1(y, k) for k in ks], rng)
        return f"{100 * m:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"

    for arm in arms[1:]:
        print(f"  {arm} - {arms[0]}: {ci(arm, arms[0])}")
    if "olens_rl_B" in arms:
        for arm in arms[arms.index("olens_rl_B") + 1 :]:
            print(f"  olens_rl_B - {arm}: {ci('olens_rl_B', arm)}")


if __name__ == "__main__":
    main()
