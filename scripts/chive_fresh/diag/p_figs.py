"""Figures for the chive-diag write-up (PNGs in /workspace/agam/experiments/chive-diag/). Reads the data/*.csv tables.
Palette = report/common.py FAM_COLOR (dataviz reference instance; OLens blue, NLA green, J-lens orange, transcript gray).
usage: python p_figs.py [flips judge3 hypgen gate budget ceiling]
"""
import csv, sys, numpy as np
sys.path.insert(0, "/workspace/agam/chive_scale/report")
from common import mpl, forest, FAM_COLOR, INK, INK2, MUTED, GRID, AXIS, SURF, LABEL
EXP = "/workspace/agam/experiments/chive-diag"
plt = mpl()

def rd(name):
    with open(f"{EXP}/data/{name}") as f: return list(csv.DictReader(f))

def save(fig, name):
    fig.savefig(f"{EXP}/{name}.png", dpi=170, bbox_inches="tight", facecolor=SURF); plt.close(fig); print(f"{EXP}/{name}.png")

def bars(ax, cats, series, title, pct=True):
    """series: [(label, color, [values])]; grouped horizontal-free vertical bars with direct value labels."""
    x = np.arange(len(cats)); w = 0.8 / len(series)
    for i, (lab, col, vals) in enumerate(series):
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, [100 * v if pct else v for v in vals], width=w - 0.03, color=col, label=lab, zorder=2)
        for xx, v in zip(xs, vals):
            ax.text(xx, (100 * v if pct else v) + 0.8, f"{100 * v:.1f}" if pct else f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_title(title, loc="left"); ax.legend(frameon=False, fontsize=8, loc="upper right")

def flips():
    T = {r["arm"]: r for r in rd("p3a_transitions.csv")}
    cats = ["committed\n(transcript only)", "final\n(after readouts)", "pass → fail", "fail → pass"]
    keys = ["committed_pass", "final_pass", "pass_fail", "fail_pass"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    bars(axs[0], cats, [("OLens ×4", FAM_COLOR["olens"], [float(T["olens_rl_B"][k]) for k in keys]),
                        ("Multi-layer NLA ×4", FAM_COLOR["mlnla"], [float(T["mlnla_B"][k]) for k in keys])],
         f"Before vs after readouts (% of episodes, n={T['olens_rl_B']['n']}/arm)")
    axs[0].set_ylim(0, 85)
    C = rd("p3a_codes.csv"); codes = ["DILUTION", "LURE", "REFOCUS", "ARGUED_AWAY", "MISREAD", "JUDGE_NOISE"]
    get = lambda a: [sum(float(r["share"]) for r in C if r["arm"] == a and r["direction"] == "pass->fail" and r["code"] == c) for c in codes]
    nn = lambda a: sum(int(r["n"]) for r in C if r["arm"] == a and r["direction"] == "pass->fail")
    bars(axs[1], ["dilution", "lure", "re-focus", "argued\naway", "misread", "judge\nnoise"],
         [("OLens ×4", FAM_COLOR["olens"], get("olens_rl_B")), ("Multi-layer NLA ×4", FAM_COLOR["mlnla"], get("mlnla_B"))],
         f"Why pass → fail (n={nn('olens_rl_B')}/{nn('mlnla_B')}, LLM-coded)")
    axs[1].set_ylim(0, 60)
    fig.text(0, -0.04, "Pass = 1-10 judge mean of 3 ≥ 7, same judge for committed and final. 915 incidents × draws 0-4.", fontsize=7.5, color=MUTED)
    save(fig, "flips_committed_vs_final")

def forest_panels(csvname, mets, titles, fname, suptitle, arms=("olens_rl_B", "mlnla_B", "jlens_B"), armmap=None, pct=True):
    R = [r for r in rd(csvname) if r["stratum"] == "overall"]
    fig, axs = plt.subplots(1, len(mets), figsize=(3.3 * len(mets), 2.3 + 0.25 * len(arms)))
    for i, (ax, met, ttl) in enumerate(zip(np.atleast_1d(axs), mets, titles)):
        rows = []
        for a in arms:
            r = next((x for x in R if x["metric"] == met and x["arm"] == a and x["ref"] == "transcript"), None)
            if not r: continue
            rows.append(dict(arm=(armmap or {}).get(a, a), mean=float(r["arm_mean"]), diff=float(r["diff"]), lo=float(r["lo"]),
                             hi=float(r["hi"]), n=int(r["n"]), transcript=float(r["ref_mean"])))
        forest(ax, rows, ttl, pct=pct, show_labels=(i == 0), show_n=False)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=9.5, color=INK, y=1.04)
    save(fig, fname)

def judge3():
    forest_panels("p1b_judge3.csv", ["precision", "wrong", "halluc", "recall", "recall_nonobv"],
                  ["Precision (true / specific claims)", "Wrong (rival + contradicted)", "Unsupported specifics", "Recall, all points", "Recall, non-obvious points"],
                  "judge3_claims", "Judge v3, claims in the final hypothesis vs transcript-only agents (paired, 95% CI; value at right = arm's mean)")

def hypgen():
    mp = {"olens": "olens_rl_B", "mlnla": "mlnla_B"}
    R = rd("p4a_hypgen.csv")
    for r in R: r["arm"] = mp.get(r["arm"], r["arm"]); r["ref"] = "transcript" if r["ref"] == "transcript" else mp.get(r["ref"], r["ref"])
    with open(f"{EXP}/data/_hypgen_fig.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(R[0])); w.writeheader(); w.writerows(R)
    forest_panels("_hypgen_fig.csv", ["match1", "match5", "primary@1", "recall_nonobv", "rivals_named"],
                  ["Top-1 hypothesis true", "Any of top-5 true", "Top-1 = primary cause", "Non-obvious recall (top-5)", "Refuted rivals named"],
                  "hypgen_top5", "No-agent hypothesis generation: transcript vs + decision-window readouts (paired, 95% CI)", arms=("olens_rl_B", "mlnla_B"))


def budget():
    R = rd("p5_dev_compare.csv")
    g = lambda a, ev, ref: next(r for r in R if r["arm"] == a and r["ref"] == ref and r["ref_eval"] == ev)
    vals = {}
    for fam, a3, a0 in [("transcript", "transcript_rx3", "transcript"), ("olens", "olens_rx3", "olens_rl_B"), ("mlnla", "mlnla_rx3", "mlnla_B")]:
        r0 = g(a3, "{b}", a0); r1 = g(a3, "{b}_exp1n30", a0)
        vals[fam] = [float(r0["ref_pass"]), float(r1["ref_pass"]), float(r0["arm_pass"])]
    n = g("olens_rx3", "{b}", "olens_rl_B")["n"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars(ax, ["0 experiments\n(frozen harness)", "1 experiment\n(frozen prompt)", "3 experiments\n('experiments decide')"],
         [("Transcript only", FAM_COLOR["transcript"], vals["transcript"]), ("OLens ×4", FAM_COLOR["olens"], vals["olens"]),
          ("Multi-layer NLA ×4", FAM_COLOR["mlnla"], vals["mlnla"])], f"Pass@1 by experiment budget (dev, n={n} incidents × 2 draws)")
    ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    fig.text(0, -0.05, "n=30 target samples per experiment; baseline re-measured by the same executor. Same incidents and draws in every bar.", fontsize=7.5, color=MUTED)
    save(fig, "budget_curve_dev")


def heldout():
    R = rd("p6_heldout.csv")
    keep = ["PRIMARY A: OLens x16 + gate vs NLA + gate (budget 0)", "PRIMARY B: OLens x16 + gate vs NLA + gate (budget 3)",
            "gate vs frozen, OLens x16", "gate vs frozen, NLA x4", "frozen OLens x4 vs frozen NLA x4 (reference)",
            "budget 3: OLens x16 + gate vs transcript", "budget 3: NLA + gate vs transcript",
            "transcript: 3 experiments vs none", "OLens x16 + gate: 3 experiments vs none", "NLA + gate: 3 experiments vs none"]
    nice = {"PRIMARY A: OLens x16 + gate vs NLA + gate (budget 0)": "Primary A: OLens ×16+gate − NLA+gate, 0 exp",
            "PRIMARY B: OLens x16 + gate vs NLA + gate (budget 3)": "Primary B: OLens ×16+gate − NLA+gate, 3 exp",
            "gate vs frozen, OLens x16": "Gate − frozen, OLens ×16", "gate vs frozen, NLA x4": "Gate − frozen, NLA ×4",
            "frozen OLens x4 vs frozen NLA x4 (reference)": "Frozen OLens ×4 − NLA ×4 (reference)",
            "budget 3: OLens x16 + gate vs transcript": "3 exp: OLens ×16+gate − transcript", "budget 3: NLA + gate vs transcript": "3 exp: NLA+gate − transcript",
            "transcript: 3 experiments vs none": "Transcript: 3 exp − 0 exp", "OLens x16 + gate: 3 experiments vs none": "OLens ×16+gate: 3 exp − 0 exp",
            "NLA + gate: 3 experiments vs none": "NLA+gate: 3 exp − 0 exp"}
    rows = [next(r for r in R if r["label"] == k) for k in keep]
    fig, ax = plt.subplots(figsize=(7.6, 4.6)); ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color=AXIS, lw=1); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for y, r in zip(ys, rows):
        col = FAM_COLOR["transcript"] if "exp − 0 exp" in nice[r["label"]] else FAM_COLOR["olens"] if "OLens" in nice[r["label"]] else FAM_COLOR["mlnla"]
        d, lo, hi = (100 * float(r[k]) for k in ("diff", "lo", "hi"))
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round"); ax.plot([d], [y], "o", ms=7, color=col)
        ax.text(1.02, y, f"{d:+.1f} pp", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([nice[r["label"]] for r in rows], fontsize=8); ax.tick_params(length=0)
    ax.set_xlabel("paired difference in pass@1 (pp), 95% CI", color=MUTED, fontsize=8)
    ax.set_title(f"Held-out test: {rows[0]['n']} incidents × 3 draws (picks frozen on dev)", loc="left")
    save(fig, "heldout_test")

if __name__ == "__main__":
    for name in (sys.argv[1:] or ["flips", "judge3", "hypgen"]):
        globals()[name]()
