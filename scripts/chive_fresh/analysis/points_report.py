"""Point eval v1 report for the fresh batches (the docs/chive_points_and_claims.md metrics).

Per answer (base draws), averaged over the 5 runs, then over incidents (only incidents where every
graded arm has 5 graded answers and the plain baseline has 3): non-obvious recall and primary-cause
recall (mean (score-1)/9 over those points), disproved (contradicted) claims per answer, claims per
answer, precision = supported / (supported + contradicted) pooled per incident; rivals rejected /
asserted pooled over runs. Point wins / losses vs transcript: a non-obvious point stated (>= 7) in
>= 3/5 runs of the arm and <= 1/5 of transcript (loss = the reverse). Paired bootstrap 95% CI vs
transcript (2000 draws, seed 0 per metric). A point is OBVIOUS if the no-tools plain answer scores
it >= 7 in >= 2 of its 3 answers.

    python points_report.py <batch> [<batch> ...] [--points DIR]

Reads $CHIVE_POINTS/<batch>/{tags,grades/<arm>,grades/plain,rivals/<arm>}/ (scripts/points_chive.py
output; run_points.sh runs every stage).
"""

import argparse
import collections
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

Rows = dict[tuple[str, str], dict[str, dict[str, Any]]]


def load(d: Path) -> dict[str, dict[str, Any]]:
    """pid -> draw (string) -> record, for every ``<pid>.d<k>.json`` in ``d``."""
    out: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    for f in d.glob("*.json"):
        r = json.loads(f.read_text())
        out[r["prompt_id"]][str(f).split(".d")[-1].split(".")[0]] = r
    return out


def collect(points: Path, batches: list[str]) -> tuple[Rows, collections.Counter[tuple[str, str]]]:
    """Per (batch, pid) and arm: per-answer metric means (+ ``_sum`` totals); point win/loss counts."""
    rows: Rows = collections.defaultdict(dict)
    pointwin: collections.Counter[tuple[str, str]] = collections.Counter()
    for b in batches:
        D = points / b
        tags = {f.name[:-5]: json.loads(f.read_text())["primary"] for f in D.glob("tags/*.json")}
        plain = load(D / "grades/plain")
        arms = sorted(p.name for p in (D / "grades").iterdir() if p.name != "plain")
        G = {a: load(D / "grades" / a) for a in arms}
        RV = {a: load(D / "rivals" / a) for a in arms}
        for pid in tags:
            if not all(len(G[a].get(pid, {})) >= 5 for a in arms) or len(plain.get(pid, {})) < 3:
                continue
            npts = len(tags[pid])
            ps = [g["point_scores"] for g in plain[pid].values()]
            if any(len(x) != npts for x in ps):
                continue
            obvious = [sum(x[i] >= 7 for x in ps) >= 2 for i in range(npts)]
            stated = {}
            for a in arms:
                m: dict[str, list[float]] = collections.defaultdict(list)
                st = [0] * npts
                for d, g in sorted(G[a][pid].items())[:5]:
                    sc = g["point_scores"]
                    if len(sc) != npts:
                        continue
                    for i, s in enumerate(sc):
                        st[i] += s >= 7
                    no = [i for i in range(npts) if not obvious[i]]
                    pr = [i for i in range(npts) if tags[pid][i]]
                    if no:
                        m["nonobv"].append(sum(sc[i] - 1 for i in no) / (9 * len(no)))
                    if pr:
                        m["primary"].append(sum(sc[i] - 1 for i in pr) / (9 * len(pr)))
                    lab = collections.Counter(c["label"] for c in g.get("claims", []))
                    m["disproved"].append(lab["contradicted"])
                    m["claims"].append(sum(lab.values()))
                    m["sup"].append(lab["supported"])
                    m["con"].append(lab["contradicted"])
                    rv = (RV[a].get(pid, {}).get(d) or {}).get("rivals", [])
                    m["rv_n"].append(len(rv))
                    m["rv_rej"].append(rv.count("rejected"))
                    m["rv_ass"].append(rv.count("asserted"))
                stated[a] = st
                rows[(b, pid)][a] = {k: sum(v) / len(v) for k, v in m.items() if v} | {
                    "_sum": {k: sum(v) for k, v in m.items()}
                }
            for a in arms:
                if a == "transcript":
                    continue
                for i in range(npts):
                    if obvious[i]:
                        continue
                    if stated[a][i] >= 3 and stated["transcript"][i] <= 1:
                        pointwin[(a, "won")] += 1
                    if stated["transcript"][i] >= 3 and stated[a][i] <= 1:
                        pointwin[(a, "lost")] += 1
    return rows, pointwin


def report(rows: Rows, pointwin: collections.Counter[tuple[str, str]], batches: list[str]) -> str:
    arms = sorted({a for r in rows.values() for a in r})
    keys = list(rows)

    def metric(a: str, k: str) -> list[float]:
        if k == "precision":
            return [
                r[a]["_sum"]["sup"] / max(1, r[a]["_sum"]["sup"] + r[a]["_sum"]["con"])
                for r in (rows[x] for x in keys)
            ]
        return [rows[x][a].get(k, 0) for x in keys]

    def pooled(a: str, num: str) -> float:
        s = [rows[x][a]["_sum"] for x in keys]
        return sum(z[num] for z in s) / max(1, sum(z["rv_n"] for z in s))

    def ci(a: str, k: str) -> tuple[float, float, float]:
        d = [x - y for x, y in zip(metric(a, k), metric("transcript", k), strict=True)]
        return paired_bootstrap(d, random.Random(0), 2000)

    out = [f"incidents with complete grades: {len(keys)} ({', '.join(batches)})\n"]
    out.append(
        f"{'metric':22s}"
        + "".join(f"{a:>14s}" for a in arms)
        + "   "
        + "  ".join(f"{a}−transcript [95% CI]" for a in arms if a != "transcript")
    )
    for k, lab, pct in (
        ("nonobv", "non-obvious recall", 1),
        ("primary", "primary-cause recall", 1),
        ("disproved", "disproved / answer", 0),
        ("claims", "claims / answer", 0),
        ("precision", "precision", 1),
    ):
        scale = 100 if pct else 1
        prec = 1 if pct else 2
        vals = "".join(f"{scale * sum(metric(a, k)) / len(keys):>14.{prec}f}" for a in arms)
        cis = "  ".join(
            "{:+.{p}f} [{:+.{p}f}, {:+.{p}f}]".format(*[scale * v for v in ci(a, k)], p=prec)
            for a in arms
            if a != "transcript"
        )
        out.append(f"{lab:22s}{vals}   {cis}")
    for num, lab in (("rv_rej", "rivals rejected"), ("rv_ass", "rivals asserted")):
        out.append(f"{lab:22s}" + "".join(f"{100 * pooled(a, num):>14.1f}" for a in arms))
    out.append(
        "\nnon-obvious points won / lost vs transcript: "
        + ", ".join(
            f"{a} {pointwin[(a, 'won')]}/{pointwin[(a, 'lost')]}" for a in arms if a != "transcript"
        )
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--points", type=Path, default=paths.POINTS)
    a = ap.parse_args()
    rows, pointwin = collect(a.points, a.batches)
    print(report(rows, pointwin, a.batches))


if __name__ == "__main__":
    main()
