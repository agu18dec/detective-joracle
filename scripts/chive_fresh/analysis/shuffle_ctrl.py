"""Shuffled-readout control: pass@1 with an incident's real readouts vs another incident's.

On each batch's control incidents (``splits/shuffle_ctrl.txt``; 150 over fresh1-4), compares each
real-reader arm with its ``:shuffled`` twin (run_chive.py serves the readouts of a different
incident, same instrument), with transcript for reference: real - shuffled with a paired bootstrap
95% CI (2000 draws, seed 0). Only incidents where all three arms have 5 clean judged base draws.

    python shuffle_ctrl.py [batches ...] [--runs DIR] [--datasets DIR] [--split shuffle_ctrl.txt]
"""

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
from common import BASE_DRAWS, paired_bootstrap, record_pass  # noqa: E402

PAIRS = [
    ("olens_rl_B", "olens_shufc"),
    ("jlens_B", "jlens_shufc"),
    ("mlnla_k1", "mlnla_k1_shufc"),
    ("mlnla_B", "mlnla_k1_shufc"),
]


def control_pass1(
    runs: Path, datasets: Path, batches: list[str], arms: set[str], split: str
) -> dict[tuple[str, str], dict[str, float]]:
    """(batch, pid) -> arm -> pass@1 over exactly 5 clean judged base draws."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for b in batches:
        for pid in (datasets / b / "splits" / split).read_text().split():
            for a in arms:
                xs = [record_pass(f) for f in (runs / b / a).glob(BASE_DRAWS.replace("*", pid, 1))]
                ok = [x for x in xs if x is not None]
                if len(ok) == 5:
                    out.setdefault((b, pid), {})[a] = sum(ok) / 5
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("batches", nargs="*", default=paths.BATCHES)
    ap.add_argument("--runs", type=Path, default=paths.RUNS)
    ap.add_argument("--datasets", type=Path, default=paths.DATASETS)
    ap.add_argument("--split", default="shuffle_ctrl.txt")
    a = ap.parse_args()
    arms = {x for p in PAIRS for x in p} | {"transcript"}
    d = control_pass1(a.runs, a.datasets, a.batches, arms, a.split)
    rng = random.Random(0)
    for real, shuf in PAIRS:
        ks = [k for k in d if real in d[k] and shuf in d[k] and "transcript" in d[k]]
        if not ks:
            print(real, "vs", shuf, ": not ready")
            continue

        def m(arm: str, ks: list[tuple[str, str]] = ks) -> float:
            return 100 * sum(d[k][arm] for k in ks) / len(ks)

        mean, lo, hi = paired_bootstrap([d[k][real] - d[k][shuf] for k in ks], rng, 2000)
        print(
            f"{real:11s} real {m(real):.1f}% | shuffled {m(shuf):.1f}% | transcript "
            f"{m('transcript'):.1f}% | real−shuffled {100 * mean:+.1f} [{100 * lo:+.1f}, "
            f"{100 * hi:+.1f}] (n={len(ks)})"
        )


if __name__ == "__main__":
    main()
