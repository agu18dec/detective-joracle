"""One-off point-eval analyses behind docs/chive_points_and_claims.md that stage=report does not
print: claims / supported / disproved / not-covered per answer, precision, non-obvious points
found by one arm only, judge pass@1 per incident, and the reader-split table (each reader's
pass@1 gain vs transcript, split by how the other readers do) with solo rescues.

    python scripts/points_extra.py runs/points
"""

import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/points")
ARMS = ["transcript", "olens_rl_B", "mlnla_B", "jlens_B"]
JUDGE = {
    "transcript": "/workspace/agam/chive_runs_dj/climb_test/transcript",
    "olens_rl_B": "/workspace/agam/chive_runs_dj/climb_test/olens_rl_B",
    "nla_B": "/workspace/agam/chive_runs_dj/climb_test/nla_B",
    "mlnla_B": "/tmp/chive_mlnla/climb_test/mlnla_B",
    "jlens_B": "/workspace/agam/chive_runs_dj/climb_test/jlens_B",
}


def load(d):
    g = defaultdict(lambda: defaultdict(list))
    for f in Path(d).glob("*/*.json"):
        r = json.loads(f.read_text())
        g[r["arm"]][r["prompt_id"]].append(r)
    return g


g = load(OUT / "grades")
plain = {p: v for p, v in g.pop("plain").items() if len(v) == 3}
tags = {f.stem: json.loads(f.read_text())["primary"] for f in (OUT / "tags").glob("*.json")}
judge = {}
for a, d in JUDGE.items():
    s = defaultdict(list)
    for f in Path(d).glob("*.d*.json"):
        r = json.loads(f.read_text())
        j = r.get("judge") or {}
        if j.get("score") is not None:
            s[r["prompt_id"]].append(bool(j["pass"]))
    judge[a] = {p: sum(v) / len(v) for p, v in s.items() if len(v) == 5}
common = sorted(
    set.intersection(*[{p for p, v in g[a].items() if len(v) == 5} for a in ARMS])
    & set(plain)
    & set(tags)
    & set.intersection(*[set(v) for v in judge.values()])
)
rivals = {a: defaultdict(list) for a in ARMS}
for a in ARMS:
    for f in (OUT / "rivals" / a).glob("*.json"):
        r = json.loads(f.read_text())
        rivals[a][r["prompt_id"]] += r["rivals"]

M = {a: {} for a in ARMS}
for p in common:
    n = len(tags[p])
    nonob = [i for i in range(n) if sum(r["point_scores"][i] >= 7 for r in plain[p]) < 2]
    found = {a: [max(r["point_scores"][i] for r in g[a][p]) >= 7 for i in range(n)] for a in ARMS}
    for a in ARMS:
        rs = g[a][p]
        counts = Counter(c.get("label") for r in rs for c in r["claims"])

        def lab(label: str, counts: Counter[str] = counts) -> float:
            return counts[label] / 5

        sup, con, nc, rej = (
            lab("supported"),
            lab("contradicted"),
            lab("not_covered"),
            lab("rejected"),
        )
        others = [b for b in ARMS if b != a]
        row = {
            "claims/answer": sup + con + nc + rej,
            "asserted/answer": sup + con + nc,
            "supported/answer": sup,
            "disproved/answer": con,
            "uncheckable/answer": nc,
            "rejected/answer": rej,
            "disproved rate": con / max(sup + con + nc, 1e-9),
            "precision (sup/(sup+dis))": sup / (sup + con) if sup + con else float("nan"),
            "points stated (>=7)/answer": sum(
                r["point_scores"][i] >= 7 for r in rs for i in range(n)
            )
            / 5,
            "recall all": sum((r["point_scores"][i] - 1) / 9 for r in rs for i in range(n))
            / (5 * n),
            "recall non-obvious": sum((r["point_scores"][i] - 1) / 9 for r in rs for i in nonob)
            / (5 * len(nonob))
            if nonob
            else float("nan"),
            "non-obvious points found only by this arm (any draw)": sum(
                found[a][i] and not any(found[b][i] for b in others) for i in nonob
            ),
            "pass@1 (judge)": judge[a][p],
        }
        rv = rivals[a].get(p, [])
        row["rivals rejected"] = sum(x == "rejected" for x in rv) / len(rv) if rv else float("nan")
        row["rivals asserted"] = sum(x == "asserted" for x in rv) / len(rv) if rv else float("nan")
        M[a][p] = row


def diff_ci(a, k, ids):
    xs = [
        M[a][p][k] - M["transcript"][p][k]
        for p in ids
        if M[a][p][k] == M[a][p][k] and M["transcript"][p][k] == M["transcript"][p][k]
    ]
    rng = random.Random(0)
    m = sum(xs) / len(xs)
    bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(2000))
    star = "*" if bs[50] > 0 or bs[1949] < 0 else ""
    return m, bs[50], bs[1949], star


def table(ids, title):
    print(f"\n### {title} ({len(ids)} incidents)\n")
    print("| metric | " + " | ".join(ARMS) + " | OLens − transcript |")
    print("|---|" + "---|" * (len(ARMS) + 1))
    for k in M["transcript"][common[0]]:
        vals = []
        for a in ARMS:
            xs = [M[a][p][k] for p in ids if M[a][p][k] == M[a][p][k]]
            vals.append(sum(xs) / len(xs))
        pct = k not in {
            "claims/answer",
            "asserted/answer",
            "supported/answer",
            "disproved/answer",
            "uncheckable/answer",
            "rejected/answer",
            "points stated (>=7)/answer",
            "non-obvious points found only by this arm (any draw)",
        }
        f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.2f}")
        m, lo, hi, s = diff_ci("olens_rl_B", k, ids)
        sc = 100 if pct else 1
        print(
            f"| {k} | "
            + " | ".join(f(v) for v in vals)
            + f" | {sc * m:+.2f} [{sc * lo:+.2f}, {sc * hi:+.2f}]{s} |"
        )


table(common, "all")


READERS = ["olens_rl_B", "nla_B", "mlnla_B", "jlens_B"]


def ci(xs: list[float]) -> str:
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(2000))
    star = "*" if bs[50] > 0 or bs[1949] < 0 else ""
    return f"{100 * sum(xs) / len(xs):+.1f} [{100 * bs[50]:+.1f}, {100 * bs[1949]:+.1f}]{star}"


print("\n### pass@1 gain vs transcript, split by the other three readers' mean pass@1\n")
for x in READERS:
    oth = [r for r in READERS if r != x]
    m = {p: sum(judge[r][p] for r in oth) / 3 for p in common}
    bands = [
        [p for p in common if m[p] <= 0.4],
        [p for p in common if 0.4 < m[p] < 0.8],
        [p for p in common if m[p] >= 0.8],
    ]
    print(
        x,
        " | ".join(
            f"n={len(b)} {ci([judge[x][p] - judge['transcript'][p] for p in b])}"
            for b in bands
            if b
        ),
    )
print(
    "\nrescues on incidents transcript fails (pass@1 <= 20%): reader pass@1 >= 60%; solo = every other reader <= 20%"
)
for x in READERS:
    oth = [r for r in READERS if r != x]
    rescue = [p for p in common if judge["transcript"][p] <= 0.2 and judge[x][p] >= 0.6]
    solo = [p for p in rescue if all(judge[r][p] <= 0.2 for r in oth)]
    print(f"  {x}: {len(rescue)} rescues, solo {solo}")
gains = {x: [judge[x][p] - judge["transcript"][p] for p in common] for x in READERS}
for a, b in itertools.combinations(READERS, 2):
    xa, xb = gains[a], gains[b]
    ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
    num = sum((u - ma) * (v - mb) for u, v in zip(xa, xb, strict=True))
    den = (sum((u - ma) ** 2 for u in xa) * sum((v - mb) ** 2 for v in xb)) ** 0.5
    print(f"  gain correlation {a} ~ {b}: {num / den:.2f}")
