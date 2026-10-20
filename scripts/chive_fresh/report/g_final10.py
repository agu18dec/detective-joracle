"""Final free-form graph at up to 10 draws per incident (fresh 915): pass@1 (mean over draws) and unbiased pass@5
per arm, transcript baseline row, paired 95% bootstrap CIs vs transcript, strata overall / hardest third /
least-confident third, plus the paired OLens − NLA difference. An incident counts for an arm when it has >= MIN
clean judged draws (salvaged/error dropped); comparisons use incidents both arms have.
Writes chive_graphs/final_freeform_10draws.png and report/data/final_freeform_10draws.csv.
"""
import os
from math import comb
import numpy as np
from common import load_records, strata, boot, forest, mpl, save, write_csv, BATCHES, LABEL, STRATA, in_stratum, INK, INK2

ARMS = ["transcript", "olens_rl_B", "jlens_L2060", "mlnla_B"]
MIN = int(os.environ.get("FINAL_MIN_DRAWS", "8"))


def pak(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def main():
    R = load_records(BATCHES, ARMS, draws=range(10))
    P = {}
    for (b, a, pid, d), r in R.items():
        if r["score"] is not None: P.setdefault((b, pid), {}).setdefault(a, []).append(r["score"] >= 7)
    P = {k: {a: v for a, v in arms.items() if len(v) >= MIN} for k, arms in P.items()}
    S = strata(); rows = []
    for st in STRATA:
        for metric, k in (("pass@1", 1), ("pass@5", 5)):
            for a in ARMS[1:]:
                ks = in_stratum(sorted(x for x, v in P.items() if a in v and "transcript" in v), S, st)
                f = lambda xs: (np.mean(xs) if k == 1 else pak(len(xs), sum(xs), 5))
                m, rm, d, lo, hi = boot([f(P[x][a]) for x in ks], [f(P[x]["transcript"]) for x in ks])
                rows.append(dict(stratum=st, metric=metric, arm=a, n=len(ks), mean=m, transcript=rm, diff=d, lo=lo, hi=hi,
                                 draws=np.mean([len(P[x][a]) for x in ks]) if ks else 0))
            ks = in_stratum(sorted(x for x, v in P.items() if "olens_rl_B" in v and "mlnla_B" in v), S, st)
            f = lambda xs: (np.mean(xs) if k == 1 else pak(len(xs), sum(xs), 5))
            m, rm, d, lo, hi = boot([f(P[x]["olens_rl_B"]) for x in ks], [f(P[x]["mlnla_B"]) for x in ks])
            rows.append(dict(stratum=st, metric=metric, arm="olens_minus_nla", n=len(ks), mean=m, transcript=rm, diff=d, lo=lo, hi=hi))
    write_csv("final_freeform_10draws", rows, ["stratum", "metric", "arm", "n", "mean", "transcript", "diff", "lo", "hi", "draws"])
    plt = mpl()
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.4))
    for i, metric in enumerate(("pass@1", "pass@5")):
        for j, st in enumerate(STRATA):
            rr = [r for r in rows if r["stratum"] == st and r["metric"] == metric and r["arm"] != "olens_minus_nla"]
            on = [r for r in rows if r["stratum"] == st and r["metric"] == metric and r["arm"] == "olens_minus_nla"][0]
            forest(axes[i, j], rr, f"{metric} · {st} (n≈{rr[0]['n']})", show_labels=(j == 0), show_n=False,
                   ref_text=f"OLens − NLA, paired: {100 * on['diff']:+.1f} [{100 * on['lo']:+.1f}, {100 * on['hi']:+.1f}]")
    nd = np.mean([r["draws"] for r in rows if r.get("draws")])
    fig.suptitle(f"Free-form hypotheses, 915 fresh CHIVE incidents, ~{nd:.1f} draws per incident (unbiased pass@5); difference vs transcript",
                 x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "final_freeform_10draws", export="final_freeform_10draws")
    for r in rows:
        if r["stratum"] == "overall": print(r["metric"], r["arm"], r["n"], f"{100 * r['mean']:.1f}", f"{100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]")


if __name__ == "__main__":
    main()
