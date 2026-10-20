"""(a) Free-form hypothesis generation: pass@1 / pass@5 per arm (old 1-10 judge, pass = score >= 7),
overall + hardest third + least-confident third, on MATCHED incidents (every arm present has 5 clean
judged base draws). Paired incident bootstrap 95% CI vs transcript. Also the shuffled-readout control.
Outputs: data/a_passk.csv, data/a_passk_pairwise.csv, fig/a_passk.png, data/a_shuffle.csv, fig/a_shuffle.png,
data/a_meta.json"""
import json, numpy as np
from common import *

def main():
    T = pass_table(ARMS + ["olens_shufc", "jlens_shufc", "mlnla_k1_shufc"])
    S = strata()
    present = [a for a in ARMS if sum(a in v for v in T.values()) >= 20]
    missing = [a for a in ARMS if a not in present]
    matched = sorted(k for k, v in T.items() if all(a in v for a in present))
    rows = []
    for st in STRATA:
        ks = in_stratum(matched, S, st)
        for metric in ("pass@1", "pass@5"):
            f = (lambda xs: sum(xs) / 5) if metric == "pass@1" else (lambda xs: float(any(xs)))
            ref = [f(T[k]["transcript"]) for k in ks]
            for a in present:
                m, rm, d, lo, hi = boot([f(T[k][a]) for k in ks], ref)
                rows.append(dict(stratum=st, metric=metric, arm=a, n=len(ks), mean=m, transcript=rm, diff=d, lo=lo, hi=hi))
    # head-to-head: each OLens arm vs each comparator reader (matched set)
    hh = []
    for st in STRATA:
        ks = in_stratum(matched, S, st)
        for o in [a for a in OLENS if a in present]:
            for c in [a for a in ("jlens_L2060", "jlens_B", "mlnla_k1", "mlnla_B") if a in present]:
                for metric, f in (("pass@1", lambda xs: sum(xs) / 5), ("pass@5", lambda xs: float(any(xs)))):
                    m, rm, d, lo, hi = boot([f(T[k][o]) for k in ks], [f(T[k][c]) for k in ks])
                    hh.append(dict(stratum=st, metric=metric, olens=o, other=c, n=len(ks), olens_mean=m, other_mean=rm, diff=d, lo=lo, hi=hi))
    write_csv("a_olens_vs_readers", hh, ["stratum", "metric", "olens", "other", "n", "olens_mean", "other_mean", "diff", "lo", "hi"])
    write_csv("a_passk", rows, ["stratum", "metric", "arm", "n", "mean", "transcript", "diff", "lo", "hi"])
    # pairwise (each arm vs transcript on every incident both completed) -- larger n, interim-friendly
    pw = []
    for st in STRATA:
        for a in ARMS[1:]:
            ks = in_stratum(sorted(k for k, v in T.items() if a in v and "transcript" in v), S, st)
            if not ks: continue
            for metric, f in (("pass@1", lambda xs: sum(xs) / 5), ("pass@5", lambda xs: float(any(xs)))):
                m, rm, d, lo, hi = boot([f(T[k][a]) for k in ks], [f(T[k]["transcript"]) for k in ks])
                pw.append(dict(stratum=st, metric=metric, arm=a, n=len(ks), mean=m, transcript=rm, diff=d, lo=lo, hi=hi))
    write_csv("a_passk_pairwise", pw, ["stratum", "metric", "arm", "n", "mean", "transcript", "diff", "lo", "hi"])

    interim = len(matched) < 800
    small = []
    if interim:
        ov = {r["arm"]: r["n"] for r in pw if r["stratum"] == "overall" and r["metric"] == "pass@1"}
        small = [a for a in present if a != "transcript" and ov.get(a, 0) < 150]
        rows = [r for r in pw if r["arm"] in present and r["arm"] not in small]
    plt = mpl()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="row", gridspec_kw={"wspace": 0.45})
    for i, metric in enumerate(("pass@1", "pass@5")):
        for j, st in enumerate(STRATA):
            rr = [r for r in rows if r["stratum"] == st and r["metric"] == metric]
            if not rr: axes[i, j].axis("off"); continue
            if interim:
                forest(axes[i, j], rr, f"{metric} · {st}", show_labels=(j == 0), show_n=True,
                       ref_text="transcript on each arm's incidents: " + "/".join(f"{100 * r['transcript']:.0f}" for r in rr) + "%")
            else:
                forest(axes[i, j], rr, f"{metric} · {st} (n={rr[0]['n']})", show_labels=(j == 0),
                       ref_text=f"transcript = {100 * rr[0]['transcript']:.1f}%")
    fig.suptitle("Free-form hypothesis generation: difference vs transcript-only (old 1-10 judge, pass = score >= 7)",
                 x=0.01, ha="left", color=INK, fontsize=11, fontweight="bold")
    fig.text(0.01, 0.935, (f"INTERIM ({len(matched)} fully matched incidents so far): each arm vs transcript on the incidents both have complete, so n differs per arm. " if interim else "") + f"Matched incidents (every arm has 5 clean judged draws d0-d4). Dot = mean paired difference, bar = 95% incident bootstrap CI. "
             f"Right-hand labels show each arm's absolute rate. Hollow = secondary arm." + (f"  Not yet available: {', '.join(missing + small)}." if missing or small else ""),
             color=INK2, fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.86, bottom=0.08, hspace=0.55); save(fig, "a_passk", "freeform_hypothesis"); plt.close(fig)

    # ---- shuffled-readout control on the 150 control incidents
    ctrl = set()
    for b in BATCHES:
        ctrl |= {(b, p) for p in open(C / "datasets" / b / "splits" / "shuffle_ctrl.txt").read().split()}
    pairs = [("olens_rl_B", "olens_shufc"), ("jlens_B", "jlens_shufc"), ("mlnla_k1", "mlnla_k1_shufc")]
    sh = []
    for real, shuf in pairs:
        ks = sorted(k for k in ctrl if k in T and all(a in T[k] for a in (real, shuf, "transcript")))
        if not ks: continue
        p1 = lambda a: [sum(T[k][a]) / 5 for k in ks]
        mr, mt, dr, lr, hr = boot(p1(real), p1("transcript"))
        ms, _, ds, ls_, hs = boot(p1(shuf), p1("transcript"))
        _, _, drs, lrs, hrs = boot(p1(real), p1(shuf))
        sh.append(dict(reader=real, shuffled_arm=shuf, n=len(ks), real=mr, shuffled=ms, transcript=mt,
                       real_minus_tr=dr, real_minus_tr_lo=lr, real_minus_tr_hi=hr,
                       shuf_minus_tr=ds, shuf_minus_tr_lo=ls_, shuf_minus_tr_hi=hs,
                       real_minus_shuf=drs, real_minus_shuf_lo=lrs, real_minus_shuf_hi=hrs))
    write_csv("a_shuffle", sh, list(sh[0].keys()) if sh else ["reader"])
    if sh:
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.axvline(0, color=AXIS, lw=1)
        y = 0; yt, yl = [], []
        for r in sh[::-1]:
            col = FAM_COLOR[family(r["reader"])]
            for kind, lab in (("shuf", "shuffled readouts"), ("real", "real readouts")):
                m, lo, hi = (r["real_minus_tr"], r["real_minus_tr_lo"], r["real_minus_tr_hi"]) if kind == "real" else (r["shuf_minus_tr"], r["shuf_minus_tr_lo"], r["shuf_minus_tr_hi"])
                ax.plot([100 * lo, 100 * hi], [y, y], color=col, lw=2, solid_capstyle="round")
                ax.plot([100 * m], [y], "o", ms=7, mfc=col if kind == "real" else SURF, mec=col, mew=1.5 if kind == "real" else 2)
                yt.append(y); yl.append(f"{LABEL[r['reader']]}, {lab}  {100 * (r['real'] if kind == 'real' else r['shuffled']):.1f}%")
                y += 1
            y += 0.6
        ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(length=0)
        ax.set_xlabel("pass@1 difference vs transcript (pp)", color=MUTED, fontsize=8)
        ax.set_title(f"Shuffled-readout control: another incident's readouts (n={sh[0]['n']} control incidents)", loc="left")
        ax.text(0, -0.25, f"transcript = {100 * sh[0]['transcript']:.1f}%. Filled = real readouts, hollow = shuffled. 95% paired incident bootstrap CIs.",
                transform=ax.transAxes, color=MUTED, fontsize=7.5)
        fig.tight_layout(); save(fig, "a_shuffle", "shuffled_control"); plt.close(fig)
    json.dump({"present": present, "missing": missing, "n_matched": len(matched),
               "per_batch": {b: sum(k[0] == b for k in matched) for b in BATCHES},
               "coverage": {a: sum(a in v for v in T.values()) for a in ARMS}}, open(REP / "data" / "a_meta.json", "w"), indent=1)
    print("matched", len(matched), "missing", missing)

if __name__ == "__main__":
    main()
