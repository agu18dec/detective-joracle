"""One graph for nla_variance.md: (A) per-incident pass counts over 5 draws on the 915, (B) run-to-run
agreement, (C) where the outcome is decided (pre-readout committed hypothesis vs final, same run)."""
import json, collections, numpy as np, shutil
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("/tmp/var_scores.json"))
ARMS = [("transcript", "transcript only", "#9a9a9a"), ("olens_rl_B", "OLens x4", "#1f6f8b"), ("mlnla_B", "multi-layer NLA x4", "#c77d2e")]
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(1, 3, figsize=(20, 5.6), gridspec_kw={"width_ratios": [1.4, 1, 1.1]})
w = 0.27
for i, (a, lab, col) in enumerate(ARMS):
    c = collections.Counter(sum(s >= 7 for s in v[a]) for v in S.values()); n = len(S)
    xs = np.arange(6) + (i - 1) * w
    ax[0].bar(xs, [100 * c[k] / n for k in range(6)], w, color=col, label=f"{lab}  (mixed: {100 * sum(c[k] for k in range(1, 5)) / n:.0f}%)")
ax[0].set_xticks(range(6), [f"{k}/5" for k in range(6)]); ax[0].set_ylabel("% of incidents")
ax[0].set_title("A. Draws passing per incident (915 fresh incidents, 5 draws)", loc="left"); ax[0].legend(frameon=False)
# B: agreement
pts = [("915: draws 0-1 vs 2-3\n(split-half)", {"transcript only": .82, "OLens x4": .76, "multi-layer NLA x4": .77}),
       ("915: x4 run vs x1 run\n(independent 5-draw runs)", {"OLens x4": .88, "multi-layer NLA x4": .89}),
       ("204: original run vs re-run\n(5 draws each)", {"transcript only": .91, "OLens x4": .89, "multi-layer NLA x4": .86})]
for j, (lab, d) in enumerate(pts):
    for i, (a, al, col) in enumerate(ARMS):
        if al in d: ax[1].plot(j + (i - 1) * 0.12, d[al], "o", ms=11, color=col); ax[1].text(j + (i - 1) * 0.12, d[al] + 0.012, f"{d[al]:.2f}", ha="center", fontsize=9)
ax[1].set_xticks(range(3), [p[0] for p in pts], fontsize=10); ax[1].set_ylim(0.6, 1.0); ax[1].grid(axis="y", alpha=.3)
ax[1].set_title("B. Same incidents, separate runs: per-incident\npass-rate correlation (higher = more reproducible)", loc="left")
# C: committed -> final
R = [json.loads(l) for l in open("/workspace/agam/chive_scale/analysis/judge_audit/raw.jsonl")]
by = collections.defaultdict(dict)
for r in R: by[(r["batch"], r["pid"])][r["kind"]] = r["score"]
cats = [("pass→pass", "#4c9a5b"), ("fail→fail", "#c44e52"), ("fail→pass", "#9ccf9f"), ("pass→fail", "#f2a8a8")]
for i, (a, lab, col) in enumerate(ARMS[1:]):
    t = collections.Counter()
    for v in by.values():
        if v.get(f"committed_{a}") is not None and v.get(f"retest_{a}") is not None:
            t[("pass" if v[f"committed_{a}"] >= 7 else "fail") + "→" + ("pass" if v[f"retest_{a}"] >= 7 else "fail")] += 1
    n = sum(t.values()); left = 0
    for c, cc in cats:
        v = 100 * t[c] / n; ax[2].barh(i, v, left=left, color=cc, label=c if i == 0 else None)
        ax[2].text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=10); left += v
ax[2].set_yticks(range(2), [l for _, l, _ in ARMS[1:]]); ax[2].set_xlim(0, 100); ax[2].set_xlabel("% of runs (240 incidents, draw 0)")
ax[2].set_title("C. Before readouts → after readouts (same run):\n76% of outcomes are fixed before any readout", loc="left")
ax[2].legend(ncol=4, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18))
fig.suptitle("Multi-layer NLA is not noisier than OLens: run-to-run variance comes from the investigating agent, equally for every arm",
             x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout()
out = "/workspace/agam/experiments/chive/nla_variance.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
shutil.copy(out, "/workspace/agam/chive_graphs/nla_variance.png")
