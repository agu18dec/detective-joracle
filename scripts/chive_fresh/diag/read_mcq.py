"""MCQ read: accuracy (picked truth), probability on truth, lure rate (picked a CHIVE refuted rival) per arm,
paired by incident vs transcript and OLens vs NLA. Usage: read_mcq.py <suffix: mcq|mcqh> <prefix: dev|devh>"""
import json, glob, collections, sys
import numpy as np
sys.path.insert(0, "/workspace/agam/chive_scale/report")
from common import boot
C = "/workspace/agam/chive_scale"; suf, pre = sys.argv[1], sys.argv[2]
sets = {}
for b in ("fresh1", "fresh2"):
    for l in open(f"{C}/analysis/mcq/{pre}_{b}_run.jsonl"):
        r = json.loads(l); sets[(b, r["prompt_id"])] = r
ARMS = ["transcript", "olens_rl_B", "mlnla_B", "jlens_L2060", "olens_single_k16"]
D = collections.defaultdict(lambda: collections.defaultdict(list))
for b in ("fresh1", "fresh2"):
    for a in ARMS:
        for f in glob.glob(f"{C}/harness_runs/{b}_{suf}/{a}/*.json"):
            try: r = json.load(open(f))
            except Exception: continue
            if str(r.get("stopped_by", "")).startswith(("error", "salv")) or not r.get("choice_eval"): continue
            k = (b, r["prompt_id"]); s = sets.get(k)
            if not s: continue
            ch = r["choice_eval"]["choice"]; src = {o["letter"]: o["source"] for o in s["options"]}
            pr = (r.get("result") or {}).get("probabilities") or {}
            pt = float(pr.get(s["truth"], np.nan)) if isinstance(pr, dict) else np.nan
            D[k][a].append((float(ch == s["truth"]), float(src.get(ch) == "tested_alternative"), pt))
def metric(k, a, i): return np.nanmean([x[i] for x in D[k][a]])
print(f"{suf}: {len(sets)} option sets")
for i, name in [(0, "accuracy"), (2, "p(truth)"), (1, "lure rate")]:
    print(f"== {name}")
    for a in ARMS:
        ks = [k for k in D if D[k][a] and D[k]["transcript"]]
        if len(ks) < 10: continue
        m, r, d, lo, hi = boot([metric(k, a, i) for k in ks], [metric(k, "transcript", i) for k in ks])
        print(f"  {a:17s} {100*m:5.1f}  vs transcript {100*r:5.1f}  Δ {100*d:+5.1f} [{100*lo:+.1f},{100*hi:+.1f}]  n={len(ks)}")
    ks = [k for k in D if D[k]["olens_rl_B"] and D[k]["mlnla_B"]]
    if ks:
        m, r, d, lo, hi = boot([metric(k, "olens_rl_B", i) for k in ks], [metric(k, "mlnla_B", i) for k in ks])
        print(f"  OLens − NLA: {100*d:+.1f} [{100*lo:+.1f},{100*hi:+.1f}] n={len(ks)}")

# ---- vertical bar graph: accuracy / p(truth) / lure rate per arm (95% bootstrap CI over incidents)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, shutil
sys.path.insert(0, "/workspace/agam/chive_scale/report"); from common import boot_ci
LAB = {"olens_rl_B": ("OLens ×4", "#1f6f8b"), "olens_single_k16": ("single-bullet\nOLens ×16", "#5fa8c7"), "jlens_L2060": ("J-lens", "#7a5ea8"),
       "mlnla_B": ("multi-layer\nNLA ×4", "#c77d2e"), "transcript": ("transcript\nonly", "#9a9a9a")}
order = ["olens_rl_B", "olens_single_k16", "jlens_L2060", "mlnla_B", "transcript"]
plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
ks_all = [k for k in D if all(D[k][a] for a in order)]
for ax, (i, name, chance) in zip(axes, [(0, "accuracy (picked the true cause)", 10), (2, "probability on the true cause", 10), (1, "lure rate (picked a refuted rival)", None)]):
    for x, a in enumerate(order):
        m, lo, hi = boot_ci([metric(k, a, i) for k in ks_all]); m, lo, hi = 100 * m, 100 * lo, 100 * hi
        ax.bar(x, m, 0.72, color=LAB[a][1]); ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="#333", capsize=4)
        ax.text(x, hi + 0.8, f"{m:.1f}", ha="center")
    _, _, d, lo, hi = boot([metric(k, "olens_rl_B", i) for k in ks_all], [metric(k, "mlnla_B", i) for k in ks_all])
    ax.text(0.02, 0.97, f"OLens − NLA, paired: {100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]", transform=ax.transAxes, va="top",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb"))
    if chance: ax.axhline(chance, ls="--", color="#999", lw=1); ax.text(4.4, chance + 0.5, "chance", fontsize=9, color="#777", ha="right")
    ax.set_xticks(range(len(order)), [LAB[a][0] for a in order], fontsize=10.5); ax.set_title(name + " (%)", loc="left"); ax.grid(axis="y", alpha=.3)
    ax.set_ylim(0, 85)
fig.suptitle(f"CHIVE MCQ ({'v3 hard' if suf == 'mcqh' else 'v2'}): 10 candidate causes, 1 true; dev incidents passing the leak check (n={len(ks_all)}), ~3 draws; 95% CI",
             x=0.01, ha="left", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.94))
name = {'mcqh': 'mcq_v3hard_bars', 'mcqb': 'mcq_v5blind_bars'}.get(suf, 'mcq_v2_bars')
fig.savefig(f"/workspace/agam/chive_graphs/{name}.png", dpi=150, bbox_inches="tight")
shutil.copy(f"/workspace/agam/chive_graphs/{name}.png", "/workspace/agam/experiments/chive/")
print("graph:", f"/workspace/agam/chive_graphs/{name}.png")
