"""Agam's held-out graph (plot_heldout_methods.py: pass@1, pass@5, pass@1 least-confident third) for
(1) the published 204 runs, (2) the replicate's 5 new draws, (3) both pooled (10 draws), (4) the fresh 915.
Plus paired OLens x4 − multi-layer NLA x4 differences. Unbiased pass@k from n draws."""
import json, glob, collections, sys
from math import comb
import numpy as np
sys.path.insert(0, "/workspace/agam/chive_scale/report")
R = "/workspace/agam/chive_scale/replicate_runs"
OUT = "/workspace/agam/chive_scale/analysis/olens_vs_nla204"
pub = json.load(open(f"{R}/original/judge_pass1.json"))
conf = json.load(open(f"{R}/confidence.json"))
ranked = sorted(conf, key=lambda p: conf[p]["top_p"]); low = set(ranked[: len(ranked) // 3])
def pak(n, c, k): return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
def rep(ev, arm):
    d = collections.defaultdict(list)
    for f in glob.glob(f"{R}/{ev}/{arm}/*.json"):
        r = json.load(open(f))
        if str(r.get("stopped_by", "")).startswith(("error", "salv")): continue
        j = r.get("judge") or {}
        if j.get("score") is not None: d[r["prompt_id"]].append(j["score"] >= 7)
    return {p: (5, sum(v)) for p, v in d.items() if len(v) == 5}
P = {a: {p: (5, round(5 * m)) for p, m in pub[a].items()} for a in ["olens_rl_B", "jlens_B", "mlnla_B", "transcript"]}
Rp = {"olens_rl_B": rep("rep_olens_orig", "olens_rl_B"), "mlnla_B": rep("rep_mlnla_orig", "mlnla_B"), "transcript": rep("rep_transcript", "transcript")}
Pool = {a: {p: (10, P[a][p][1] + Rp[a][p][1]) for p in Rp[a] if p in P[a]} for a in Rp}
# fresh 915
from common import pass_table
T = pass_table(["olens_rl_B", "jlens_L2060", "mlnla_B", "transcript"])
import common
tp = common.top_p(); fr = sorted(tp, key=lambda k: tp[k]); flow = set(fr[: len(fr) // 3])
F = {a: {k: (5, sum(v[a])) for k, v in T.items() if a in v} for a in ["olens_rl_B", "jlens_L2060", "mlnla_B", "transcript"]}
SETS = [("Published (your run, 5 draws)", P, low), ("Replicate (5 new draws, same banks)", Rp, low),
        ("Published + replicate pooled (10 draws)", Pool, low), ("Fresh 915 incidents (5 draws)", F, flow)]
LAB = {"olens_rl_B": "OLens", "jlens_B": "J-lens", "jlens_L2060": "J-lens", "mlnla_B": "multi-layer NLA", "transcript": "transcript only"}
COL = {"olens_rl_B": "#1f6f8b", "jlens_B": "#7a5ea8", "jlens_L2060": "#7a5ea8", "mlnla_B": "#c77d2e", "transcript": "#9a9a9a"}
rng = np.random.default_rng(0)
def bci(x):
    x = np.asarray(x, float); bs = np.sort(x[rng.integers(0, len(x), (4000, len(x)))].mean(1)); return x.mean(), bs[100], bs[3899]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(len(SETS), 3, figsize=(20, 5.2 * len(SETS)), sharey=True)
lines = ["| data | panel | n | OLens | NLA ×4 | transcript | OLens − NLA, paired [95% CI] |", "|---|---|---|---|---|---|---|"]
for row, (name, D, lowset) in zip(axes, SETS):
    common_ids = sorted(set.intersection(*[set(v) for v in D.values()]))
    panels = [("pass@1, all", common_ids, 1), ("pass@5, all", common_ids, 5), ("pass@1, least-confident third", [p for p in common_ids if p in lowset], 1)]
    for ax, (title, ids, k) in zip(row, panels):
        val = {a: [pak(*D[a][p], k) for p in ids] for a in D}
        for i, a in enumerate(D):
            m, lo, hi = bci(val[a]); m, lo, hi = 100 * m, 100 * lo, 100 * hi
            ax.bar(i, m, 0.75, color=COL[a]); ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="#333", capsize=4)
            ax.text(i, hi + 1, f"{m:.1f}", ha="center")
        d, lo, hi = bci(np.array(val["olens_rl_B"]) - np.array(val["mlnla_B"]))
        ax.set_xticks(range(len(D)), [LAB[a] for a in D], rotation=12); ax.set_ylim(40, 100); ax.grid(axis="y", alpha=.3)
        ax.set_title(f"{title} (n={len(ids)})\nOLens − NLA paired: {100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]", loc="left", fontsize=12)
        lines.append(f"| {name} | {title} | {len(ids)} | {100*np.mean(val['olens_rl_B']):.1f} | {100*np.mean(val['mlnla_B']):.1f} | {100*np.mean(val['transcript']):.1f} | {100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] |")
    row[0].set_ylabel(name, fontsize=13, fontweight="bold")
fig.suptitle("Agam's held-out graph re-made: published vs replicate vs pooled vs fresh 915 (95% CI over incidents; pass = judge ≥ 7)", x=0.01, ha="left", fontsize=15)
fig.tight_layout(); fig.savefig(f"{OUT}/olens_vs_nla_heldout_vs_replicate.png", dpi=130, bbox_inches="tight")
open(f"{OUT}/table.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
