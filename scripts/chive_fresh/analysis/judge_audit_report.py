"""Report for judge_audit.py: control pass rates, test-retest, and pre-readout (committed) vs final per arm.
Writes analysis/judge_audit/report.md, dash_lines.md and chive_graphs/judge_audit.png."""
import json, collections, pathlib, sys
import numpy as np
sys.path.insert(0, "/workspace/agam/chive_scale/report")
from common import mpl, save, boot, boot_ci, FAM_COLOR, family, LABEL, INK, INK2, MUTED, GRID, AXIS

D = pathlib.Path("/workspace/agam/chive_scale/analysis/judge_audit")
R = [json.loads(l) for l in open(D / "raw.jsonl") if l.strip()]
R = [r for r in R if r["score"] is not None]
by = collections.defaultdict(dict)
for r in R: by[(r["batch"], r["pid"])][r["kind"]] = r
P = lambda s: float(s >= 7)
L = ["# Judge audit (fresh CHIVE, old 1-10 judge, pass = mean of 3 >= 7)", "", f"incidents: {len(by)}", "",
     "## Controls", "| control | n | pass rate | mean score |", "|---|---|---|---|"]
ctrl = {}
for k, lab in [("pos_T1", "ground-truth cause T1 verbatim"), ("neg_X1", "refuted/inert rival X1 verbatim"),
               ("neg_other", "transcript hypothesis from another incident"), ("generic", "generic 'prompt ambiguity'")]:
    xs = [v[k]["score"] for v in by.values() if k in v]
    if xs:
        ctrl[k] = (np.mean([P(x) for x in xs]), np.mean(xs), len(xs))
        L.append(f"| {lab} | {len(xs)} | {100 * ctrl[k][0]:.1f}% | {ctrl[k][1]:.2f} |")
L += ["", "## Test-retest (same stored d0 final hypothesis, judged again)",
      "| arm | n | stored pass | re-judged pass | pass flips | mean abs score change |", "|---|---|---|---|---|---|"]
flips = {}
for a in ["transcript", "olens_rl_B", "mlnla_B", "jlens_L2060"]:
    xs = [(v[f"retest_{a}"]["stored"], v[f"retest_{a}"]["score"]) for v in by.values() if f"retest_{a}" in v and v[f"retest_{a}"]["stored"] is not None]
    if not xs: continue
    st, re_ = np.array(xs).T; fl = np.mean((st >= 7) != (re_ >= 7)); flips[a] = fl
    L.append(f"| {LABEL[a]} | {len(xs)} | {100 * np.mean(st >= 7):.1f}% | {100 * np.mean(re_ >= 7):.1f}% | {100 * fl:.1f}% | {np.mean(np.abs(st - re_)):.2f} |")
L += ["", "## Pre-readout committed text hypothesis vs final (same run, same incident; judged in this audit)",
      "Committed = the agent's text-only hypothesis written BEFORE readout tools unlock; final = submitted after readouts.",
      "| arm | n | committed pass | final pass (re-judged) | final − committed [95% CI] | transcript arm (re-judged) |", "|---|---|---|---|---|---|"]
gain = []
for a in ["olens_rl_B", "mlnla_B", "jlens_L2060"]:
    ks = [k for k, v in by.items() if f"committed_{a}" in v and f"retest_{a}" in v and "retest_transcript" in v]
    if not ks: continue
    c = [P(by[k][f"committed_{a}"]["score"]) for k in ks]; f = [P(by[k][f"retest_{a}"]["score"]) for k in ks]
    t = [P(by[k]["retest_transcript"]["score"]) for k in ks]
    mf, mc, d, lo, hi = boot(f, c)
    gain.append(dict(arm=a, n=len(ks), committed=mc, final=mf, diff=d, lo=lo, hi=hi, transcript=np.mean(t)))
    L.append(f"| {LABEL[a]} | {len(ks)} | {100 * mc:.1f}% | {100 * mf:.1f}% | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {100 * np.mean(t):.1f}% |")
(D / "report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

dl = ["## Judge audit (n=%d incidents)" % len(by)]
if "pos_T1" in ctrl: dl.append(f"- Judge controls: true cause T1 passes {100 * ctrl['pos_T1'][0]:.0f}%, refuted rival X1 {100 * ctrl['neg_X1'][0]:.0f}%, other-incident hypothesis {100 * ctrl['neg_other'][0]:.0f}%, generic {100 * ctrl['generic'][0]:.0f}%.")
if flips: dl.append("- Test-retest pass flips per record: " + ", ".join(f"{LABEL[a]} {100 * v:.0f}%" for a, v in flips.items()) + ".")
if gain: dl.append("- Readouts' own effect (final − pre-readout committed, same run): " + ", ".join(f"{LABEL[g['arm']]} {100 * g['diff']:+.1f} pp [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}]" for g in gain) + ".")
(D / "dash_lines.md").write_text("\n".join(dl) + "\n")

plt = mpl()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.2]})
ax = axes[0]; names = [("pos_T1", "true cause T1"), ("neg_X1", "refuted rival X1"), ("neg_other", "other incident's hyp"), ("generic", "generic")]
names = [(k, l) for k, l in names if k in ctrl]
ax.barh(range(len(names))[::-1], [100 * ctrl[k][0] for k, _ in names], color=[FAM_COLOR["olens"]] + [FAM_COLOR["transcript"]] * (len(names) - 1))
for y, (k, _) in zip(range(len(names))[::-1], names): ax.text(100 * ctrl[k][0] + 1, y, f"{100 * ctrl[k][0]:.0f}%", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels([l for _, l in names]); ax.set_xlim(0, 110)
ax.set_title("Judge controls: pass rate", loc="left"); ax.tick_params(length=0); ax.grid(axis="x", color=GRID)
ax = axes[1]
for i, gg in enumerate(gain):
    col = FAM_COLOR[family(gg["arm"])]; y = len(gain) - 1 - i
    ax.plot([100 * gg["committed"], 100 * gg["final"]], [y, y], color=col, lw=2)
    ax.plot([100 * gg["committed"]], [y], "o", mfc="white", mec=col, ms=8, mew=2)
    ax.plot([100 * gg["final"]], [y], "o", color=col, ms=8)
    ax.plot([100 * gg["transcript"]], [y], "D", color=FAM_COLOR["transcript"], ms=6)
ax.set_yticks(range(len(gain))[::-1]); ax.set_yticklabels([LABEL[g["arm"]] for g in gain]); ax.tick_params(length=0)
ax.grid(axis="x", color=GRID); ax.set_xlabel("pass rate (%)", color=MUTED, fontsize=8)
ax.set_title("Same run: before readouts (hollow) → after (filled); ◆ transcript arm", loc="left")
fig.suptitle(f"Judge audit on {len(by)} fresh incidents (d0)", x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
fig.tight_layout(); save(fig, "judge_audit", export="judge_audit")
