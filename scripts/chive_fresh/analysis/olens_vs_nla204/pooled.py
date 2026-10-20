"""climb_test204 held-out graph with every judged draw pooled per incident (published 5 + replicate draws).

Same panels and colours as Agam's plot_heldout_methods.py: pass@1 all, pass@5 all, pass@1 least-confident
third. pass@1 = mean over an incident's draws; pass@5 = unbiased estimator from its n draws. Bars: 95%
bootstrap CI over incidents; the paired OLens − multi-layer NLA difference is printed per panel. Salvaged /
error episodes (cut short by rate limits) are dropped. J-lens has only the 5 published draws (hatched).
Writes chive_graphs/climb_test204_pooled.png and analysis/olens_vs_nla204/pooled_table.md.
"""
import json, glob, collections, os, shutil
from math import comb
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "/workspace/agam/chive_scale/replicate_runs"
OUT = "/workspace/agam/chive_scale/analysis/olens_vs_nla204"
G = "/workspace/agam/chive_graphs"
ARMS = [("OLens", "olens_rl_B", "rep_olens_orig", "#1f6f8b"), ("J-lens", "jlens_B", "rep_jlens_orig", "#7a5ea8"),
        ("multi-layer NLA", "mlnla_B", "rep_mlnla_orig", "#c77d2e"), ("transcript only", "transcript", "rep_transcript", "#9a9a9a")]
pub = json.load(open(f"{R}/original/judge_pass1.json"))
conf = json.load(open(f"{R}/confidence.json"))
ranked = sorted(conf, key=lambda p: conf[p]["top_p"]); low = set(ranked[: len(ranked) // 3])
# archived original NLA records: use them directly so its salvaged episodes can be dropped
arch = collections.defaultdict(list)
for f in glob.glob(f"{R}/archive_orig/chive_mlnla/climb_test/mlnla_B/*.json"):
    r = json.load(open(f)); j = r.get("judge") or {}
    if j.get("score") is not None and not str(r.get("stopped_by", "")).startswith(("error", "salv")):
        arch[r["prompt_id"]].append(j["score"] >= 7)


def counts(arm, ev):
    """{pid: (n, c)} pooled over published + replicate draws."""
    rep = collections.defaultdict(list)
    for f in glob.glob(f"{R}/{ev}/{arm}/*.json"):
        r = json.load(open(f)); j = r.get("judge") or {}
        if j.get("score") is not None and not str(r.get("stopped_by", "")).startswith(("error", "salv")):
            rep[r["prompt_id"]].append(j["score"] >= 7)
    out = {}
    for p, m in pub[arm].items():
        if arm == "mlnla_B": n0, c0 = len(arch[p]), sum(arch[p])
        else: n0, c0 = 5, round(5 * m)
        n, c = n0 + len(rep.get(p, [])), c0 + sum(rep.get(p, []))
        if n >= 5: out[p] = (n, c)
    return out


def pak(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


D = {lab: counts(arm, ev) for lab, arm, ev, _ in ARMS}
ndraw = {lab: np.mean([v[0] for v in D[lab].values()]) for lab in D}
ids = sorted(set.intersection(*[set(v) for v in D.values()]))
rng = np.random.default_rng(0)


def bci(x):
    x = np.asarray(x, float); bs = np.sort(x[rng.integers(0, len(x), (4000, len(x)))].mean(1))
    return 100 * x.mean(), 100 * bs[100], 100 * bs[3899]


plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15})
fig, axes = plt.subplots(1, 3, figsize=(21, 7.2), sharey=True)
panels = [(f"pass@1, all {len(ids)} incidents", ids, 1), (f"pass@5, all {len(ids)} incidents", ids, 5),
          (f"pass@1, least-confident third ({sum(p in low for p in ids)})", [p for p in ids if p in low], 1)]
rows = []
for ax, (title, S, k) in zip(axes, panels):
    val = {lab: [pak(*D[lab][p], k) for p in S] for lab in D}
    for i, (lab, arm, ev, col) in enumerate(ARMS):
        m, lo, hi = bci(val[lab])
        ax.bar(i, m, 0.75, color=col, hatch="//" if ndraw[lab] < 6 else None, edgecolor="white" if ndraw[lab] < 6 else col)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="#333", lw=1.4, capsize=4)
        ax.text(i, hi + 1, f"{m:.1f}", ha="center", fontsize=13)
        rows.append((title, lab, len(S), m, lo, hi))
    d, lo, hi = bci(np.array(val["OLens"]) - np.array(val["multi-layer NLA"]))
    rows.append((title, "OLens − NLA (paired)", len(S), d, lo, hi))
    ax.set_xticks(range(len(ARMS)), [f"{lab}\n({ndraw[lab]:.0f} draws)" for lab, *_ in ARMS], fontsize=12)
    ax.set_title(title, loc="left")
    ax.text(0.0, 0.97, f"OLens − NLA, paired: {d:+.1f} [{lo:+.1f}, {hi:+.1f}]", transform=ax.transAxes, fontsize=12.5,
            va="top", color="#222", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb"))
    ax.set_ylim(40, 100); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("% of incidents (judge >= 7)")
fig.suptitle("Held-out free-form explanations, climb_test204, all draws pooled (published + re-runs; "
             "95% CI over incidents; J-lens hatched = 5 published draws only)", x=0.01, ha="left", fontsize=15)
fig.tight_layout()
fig.savefig(f"{OUT}/climb_test204_pooled.png", dpi=150, bbox_inches="tight")
for d in (G, "/workspace/agam/experiments/chive"):
    shutil.copy(f"{OUT}/climb_test204_pooled.png", d)
L = ["| panel | arm | n | value | 95% CI |", "|---|---|---|---|---|"] + \
    [f"| {t} | {a} | {n} | {m:.1f} | [{lo:.1f}, {hi:.1f}] |" for t, a, n, m, lo, hi in rows]
L.append("\ndraws per incident: " + ", ".join(f"{k} {v:.1f}" for k, v in ndraw.items()))
open(f"{OUT}/pooled_table.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
