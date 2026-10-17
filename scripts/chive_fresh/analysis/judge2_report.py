"""Aggregate judge v2 selections (any completed subset) with paired bootstrap CIs vs transcript.

Per incident and arm (incidents where every arm has >= 1 selection), averaged over draws: share of
CHIVE's true points ASSERTED, of the non-obvious ones ASSERTED, of true points DENIED, and of
refuted rivals ASSERTED. Tables: overall and the pre-registered least-confident third (no-tools
Opus top_p from $CHIVE_CONFIDENCE/<batch>/confidence.json, scripts/confidence_chive.py). Paired
bootstrap 95% CI vs transcript (1000 draws, one seed-0 generator per table; incidents in sorted
order, so the CIs are reproducible).

    python judge2_report.py <judge2_hyp|judge2> [batches ...] [--arms a,b,c] [--root DIR]

``judge2_hyp`` = hypothesis-only grading (primary), ``judge2`` = hypothesis + evidence + rival.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
from common import paired_bootstrap  # noqa: E402

DEFAULT_ARMS = "transcript,olens_rl_B,jlens_B,mlnla_B"
KEYS = (
    ("true", "true pts asserted"),
    ("nonobv", "non-obvious"),
    ("denied", "true pts denied"),
    ("rival", "rivals asserted"),
)


def _mean(v: list[float]) -> float:
    return sum(v) / len(v)


def incident_meta(b: str, pid: str) -> tuple[list[bool], int]:
    """(obvious flag per point, number of refuted rivals) for one incident."""
    pts = json.loads((paths.POINTS / b / "points" / f"{pid}.json").read_text())["points"]
    pl = [
        json.loads(x.read_text())["point_scores"]
        for x in sorted((paths.POINTS / b / "grades/plain").glob(f"{pid}.d*.json"))[:3]
    ]
    ob = (
        [sum(x[i] >= 7 for x in pl) >= 2 for i in range(len(pts))]
        if len(pl) == 3
        else [False] * len(pts)
    )
    with (paths.DATASETS / b / "data/ground_truth.jsonl").open() as fh:
        gt = next(r for r in map(json.loads, fh) if r["prompt_id"] == pid)
    nr = sum(1 for x in gt["ruled_out"] if x.get("kind") == "refuted" and x.get("cause"))
    return ob, nr


def selection_scores(labels: dict[str, str], ob: list[bool], nr: int) -> dict[str, float | None]:
    """One answer's four shares (None where undefined: no non-obvious point / no rival)."""
    t = [labels.get(f"P{i + 1}", "ABSENT") for i in range(len(ob))]
    r = [labels.get(f"R{j + 1}", "ABSENT") for j in range(nr)]
    n_nonobv = max(1, sum(not o for o in ob))
    return {
        "true": sum(x == "ASSERTED" for x in t) / len(t),
        "nonobv": (sum(x == "ASSERTED" for x, o in zip(t, ob, strict=True) if not o) / n_nonobv)
        if not all(ob)
        else None,
        "denied": sum(x == "DENIED" for x in t) / len(t),
        "rival": (sum(x == "ASSERTED" for x in r) / nr) if nr else None,
    }


def collect(
    root: Path, batches: list[str], arms: list[str]
) -> tuple[dict[tuple[str, str], dict[str, dict[str, list[float]]]], dict[tuple[str, str], Any]]:
    M: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
    strat: dict[tuple[str, str], Any] = {}
    for b in batches:
        conf: dict[str, Any] = {}
        try:
            conf = json.loads((paths.CONFIDENCE / b / "confidence.json").read_text())
        except Exception:
            pass
        # sorted: the pod original iterated a set, so its CIs varied with PYTHONHASHSEED
        pids = sorted(
            {f.name.split(".d")[0] for f in (root / b / "select" / arms[0]).glob("*.json")}
        )
        for pid in pids:
            try:
                ob, nr = incident_meta(b, pid)
            except Exception:
                continue
            per = {}
            for a in arms:
                xs = [
                    selection_scores((json.loads(f.read_text()) or {}).get("labels", {}), ob, nr)
                    for f in (root / b / "select" / a).glob(f"{pid}.d*.json")
                ]
                if xs:
                    per[a] = xs
            if all(a in per for a in arms):
                M[(b, pid)] = {
                    a: {k: [x[k] for x in per[a] if x[k] is not None] for k, _ in KEYS}
                    for a in arms
                }
                if pid in conf:
                    strat[(b, pid)] = conf[pid].get("top_p")
    return M, strat


def table(
    M: dict[tuple[str, str], dict[str, dict[str, list[float]]]],
    arms: list[str],
    name: str,
    keys: list[tuple[str, str]],
) -> str:
    if not keys:
        return ""
    out = [f"\n## {name} (n={len(keys)} incidents)"]
    out.append(
        f"{'metric':18s}"
        + "".join(f"{a:>12s}" for a in arms)
        + "   "
        + "   ".join(f"{a}−tr [95% CI]" for a in arms[1:])
    )
    rng = random.Random(0)
    for k, lab in KEYS:
        ks = [x for x in keys if all(M[x][a][k] for a in arms)]
        vals = {a: [_mean(M[x][a][k]) for x in ks] for a in arms}
        line = f"{lab:18s}" + "".join(f"{100 * _mean(vals[a]):11.1f}%" for a in arms)
        for a in arms[1:]:
            m, lo, hi = paired_bootstrap(
                [p - q for p, q in zip(vals[a], vals[arms[0]], strict=True)], rng, 1000
            )
            line += f"   {100 * m:+5.1f} [{100 * lo:+.1f},{100 * hi:+.1f}]"
        out.append(line)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("judge_dir", help="judge2_hyp or judge2 (under --root)")
    ap.add_argument("batches", nargs="*", default=paths.BATCHES)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--root", type=Path, default=paths.SCALE)
    a = ap.parse_args()
    arms = a.arms.split(",")
    M, strat = collect(a.root / a.judge_dir, a.batches, arms)
    keys = list(M)
    ck = sorted([k for k in keys if strat.get(k) is not None], key=lambda k: strat[k])
    for name, sub in (
        ("overall", keys),
        ("least-confident third (no-tools top_p)", ck[: len(ck) // 3]),
    ):
        if sub:
            print(table(M, arms, name, sub))


if __name__ == "__main__":
    main()
