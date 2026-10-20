"""Replicate analysis: original (published per-incident judge means) vs A (original banks) vs B (rebuilt).

  python3 analyze.py   -> prints markdown, writes ../analysis.json

pass@k: unbiased estimator from the 5 judged draws (incidents with all 5 draws judged, like
passk_chive.py). Original arms: per-incident judge mean m over 5 draws -> c = round(5 m).
Paired 95% bootstrap CIs over incidents (4000 resamples, seed 0, as passk_chive.py).
Strata: all 204; least-confident third = lowest-top_p third of confidence.json over all 300,
∩ the 204 (the original's definition: 66 incidents); hard = hard92 ∩ 204 (46).
"""
import json, random
from collections import defaultdict
from math import comb
from pathlib import Path

R = Path("/workspace/agam/chive_scale/replicate_runs")
H = Path("/workspace/agam/chive_scale/harness/data/chive/splits")
IDS = H.joinpath("climb_test204.txt").read_text().split()
HARD = set(H.joinpath("hard92.txt").read_text().split()) & set(IDS)
ARMS = {  # label -> (eval dir, arm)
    "transcript": ("rep_transcript", "transcript"),
    "olens_orig": ("rep_olens_orig", "olens_rl_B"), "nla_orig": ("rep_nla_orig", "nla_B"),
    "jlens_orig": ("rep_jlens_orig", "jlens_B"), "mlnla_orig": ("rep_mlnla_orig", "mlnla_B"),
    "olens_rebuilt": ("rep_olens_rebuilt", "olens_rl_B"), "nla_rebuilt": ("rep_nla_rebuilt", "nla_B"),
    "jlens_rebuilt": ("rep_jlens_rebuilt", "jlens_B"), "mlnla_rebuilt": ("rep_mlnla_rebuilt", "mlnla_B"),
    "mlnla_rebuilt_k1": ("rep_mlnla_rebuilt_k1", "mlnla_k1"),
    "olens_orig_10draws": ("rep_olens_orig", "olens_rl_B"), "mlnla_orig_10draws": ("rep_mlnla_orig", "mlnla_B"),
}
ORIG = {"olens": "olens_rl_B", "nla": "nla_B", "jlens": "jlens_B", "mlnla": "mlnla_B", "transcript": "transcript"}


def pak(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def load():
    out = {}
    for lab, (ev, arm) in ARMS.items():
        d = defaultdict(list)
        maxd = 10 if lab.endswith("_10draws") else 5  # primary tables: draws d0-d4, as published
        for f in (R / ev / arm).glob("*.d*.json"):
            if int(f.name.rsplit(".d", 1)[1].split(".")[0]) >= maxd:
                continue
            r = json.loads(f.read_text()); j = r.get("judge") or {}
            if j.get("score") is not None and int(j.get("n_replicates") or 0) >= 3:
                d[r["prompt_id"]].append(bool(j["pass"]))
        out[lab] = {p: (len(v), sum(v)) for p, v in d.items() if len(v) >= maxd and p in IDS}
    o = json.loads((R / "original/judge_pass1.json").read_text())
    for k, a in ORIG.items():
        out[f"{k}_published"] = {p: (5, round(5 * m)) for p, m in o[a].items()}
    return out


def per(data, lab, k):
    return {p: pak(n, c, k) for p, (n, c) in data[lab].items()}


def mean(data, lab, k, sub):
    xs = [v for p, v in per(data, lab, k).items() if p in sub]
    return (100 * sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)


def paired(data, a, b, k, sub):
    pa, pb = per(data, a, k), per(data, b, k)
    ps = [p for p in pa if p in pb and p in sub]
    if len(ps) < 5:
        return None
    d = [pa[p] - pb[p] for p in ps]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
    m, lo, hi = 100 * sum(d) / len(d), 100 * bs[100], 100 * bs[3899]
    return {"d": m, "lo": lo, "hi": hi, "n": len(ps), "sig": lo > 0 or hi < 0}


def fmt(x):
    return "-" if not x else f"{x['d']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]{'*' if x['sig'] else ''} (n={x['n']})"


def main():
    data = load()
    strata = {"all": set(IDS), "hard": HARD}
    cf = R / "confidence.json"
    if cf.exists():
        conf = json.loads(cf.read_text())
        ranked = sorted(conf, key=lambda p: conf[p]["top_p"])
        strata["least-confident third"] = set(ranked[: len(ranked) // 3]) & set(IDS)
        c204 = sorted((p for p in conf if p in IDS), key=lambda p: conf[p]["top_p"])
        strata["least-confident third (within 204)"] = set(c204[: len(c204) // 3])
    res = {"strata_n": {k: len(v) for k, v in strata.items()}, "arms": {}, "paired": {}}
    labs = [l for l in data if data[l]]
    print("## pass@1 / pass@5 by stratum\n")
    for sn, sub in strata.items():
        print(f"### {sn} ({len(sub)})\n\n| arm | n | pass@1 | pass@5 | Δpass@1 vs transcript | Δpass@5 vs transcript |\n|---|---|---|---|---|---|")
        for l in labs:
            m1, n = mean(data, l, 1, sub); m5, _ = mean(data, l, 5, sub)
            tx = "transcript_published" if l.endswith("_published") else "transcript"
            c1 = paired(data, l, tx, 1, sub) if l != tx else None
            c5 = paired(data, l, tx, 5, sub) if l != tx else None
            res["arms"].setdefault(l, {})[sn] = {"n": n, "pass1": m1, "pass5": m5, "vs_tx_1": c1, "vs_tx_5": c5}
            print(f"| {l} | {n} | {m1:.1f} | {m5:.1f} | {fmt(c1)} | {fmt(c5)} |")
        print()
    pairs = [("olens_published", "mlnla_published"), ("olens_orig", "mlnla_orig"), ("olens_rebuilt", "mlnla_rebuilt"),
             ("olens_rebuilt", "mlnla_rebuilt_k1"), ("olens_orig_10draws", "mlnla_orig_10draws"), ("mlnla_rebuilt", "mlnla_rebuilt_k1")]
    pairs += [(f"{r}_orig", f"{r}_rebuilt") for r in ("olens", "nla", "jlens", "mlnla")]
    pairs += [(f"{r}_orig", f"{r}_published") for r in ("olens", "nla", "jlens", "mlnla")] + [("transcript", "transcript_published")]
    pairs += [(f"{r}_rebuilt", f"{r}_published") for r in ("olens", "nla", "jlens", "mlnla")]
    print("## pairwise (a − b), paired by incident\n\n| a − b | stratum | Δpass@1 | Δpass@5 |\n|---|---|---|---|")
    for a, b in pairs:
        if not data.get(a) or not data.get(b):
            continue
        for sn, sub in strata.items():
            x1, x5 = paired(data, a, b, 1, sub), paired(data, a, b, 5, sub)
            res["paired"][f"{a} - {b} | {sn}"] = {"p1": x1, "p5": x5}
            print(f"| {a} − {b} | {sn} | {fmt(x1)} | {fmt(x5)} |")
    (R / "analysis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
