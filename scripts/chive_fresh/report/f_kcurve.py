"""Single-bullet OLens x k curve: free-form pass@1 and pass@5 vs k (samples per cell, k = 1,2,4,8,16).

Matched incidents: every k-arm (and transcript) has all 5 base draws judged. Points are means with 95%
bootstrap CIs over incidents; the transcript level is drawn as a dashed reference line. Arms whose runs are
not in yet are skipped, so the curve fills in as k=2/8/16 finish.
Writes chive_graphs/single_bullet_k_curve.png and report/data/single_bullet_k_curve.csv.
"""
import numpy as np
from common import pass_table, boot_ci, mpl, save, write_csv, INK, INK2, MUTED, GRID, AXIS, FAM_COLOR, family

K_ARMS = [(1, "olens_single_k1"), (2, "olens_single_k2"), (4, "olens_single"), (8, "olens_single_k8"),
          (16, "olens_single_k16")]


def main():
    T = pass_table([a for _, a in K_ARMS] + ["transcript"])
    present = [(k, a) for k, a in K_ARMS if sum(a in v for v in T.values()) >= 50]
    keys = [key for key, v in T.items() if "transcript" in v and all(a in v for _, a in present)]
    rows = []
    for k, a in present:
        for metric, f in (("pass@1", lambda xs: float(np.mean(xs))), ("pass@5", lambda xs: float(any(xs)))):
            m, lo, hi = boot_ci([f(T[key][a]) for key in keys])
            rows.append({"k": k, "arm": a, "metric": metric, "mean": m, "lo": lo, "hi": hi, "n": len(keys)})
    tr = {metric: boot_ci([f(T[key]["transcript"]) for key in keys])[0]
          for metric, f in (("pass@1", lambda xs: float(np.mean(xs))), ("pass@5", lambda xs: float(any(xs))))}
    write_csv("single_bullet_k_curve", rows + [{"k": 0, "arm": "transcript", "metric": m, "mean": v, "n": len(keys)}
                                               for m, v in tr.items()], ["k", "arm", "metric", "mean", "lo", "hi", "n"])
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    col = FAM_COLOR[family("olens_single")]
    for ax, metric in zip(axes, ("pass@1", "pass@5")):
        R = [r for r in rows if r["metric"] == metric]
        xs = [r["k"] for r in R]
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.fill_between(xs, [100 * r["lo"] for r in R], [100 * r["hi"] for r in R], color=col, alpha=0.15, lw=0, zorder=1)
        ax.plot(xs, [100 * r["mean"] for r in R], "-o", color=col, ms=6, lw=2, zorder=3)
        for r in R:
            ax.annotate(f"{100 * r['mean']:.1f}", (r["k"], 100 * r["mean"]), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=INK2)
        ax.axhline(100 * tr[metric], color=AXIS, ls="--", lw=1.2, zorder=2)
        ax.text(1.0, 100 * tr[metric], f" transcript {100 * tr[metric]:.1f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8, color=MUTED)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlabel("samples per cell (k)", color=MUTED, fontsize=8)
        ax.set_title(f"{metric} (%)", loc="left")
        ax.tick_params(length=0)
    fig.suptitle(f"Single-bullet OLens × k, free-form hypotheses (n={len(keys)} matched incidents; 95% CI band)",
                 x=0.01, ha="left", fontsize=9.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "single_bullet_k_curve", export="single_bullet_k_curve")
    print("k-curve:", [(r["k"], r["metric"], round(100 * r["mean"], 1)) for r in rows], "n", len(keys))


if __name__ == "__main__":
    main()
