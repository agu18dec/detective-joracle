"""Vertical bar version of the final free-form graph (fresh 915, up to 10 draws per incident), in the style of
Agam's held-out graph: one bar per arm with a 95% bootstrap CI over incidents; panels pass@1 / unbiased pass@5 x
overall / hardest third / least-confident third; paired OLens − NLA printed per panel.
Writes chive_graphs/final_freeform_10draws_bars.png.
"""
import os
from math import comb
import numpy as np
from common import load_records, strata, boot, boot_ci, save, BATCHES, STRATA, in_stratum

ARMS = [("olens_rl_B", "OLens ×4", "#1f6f8b"), ("jlens_L2060", "J-lens", "#7a5ea8"),
        ("mlnla_B", "multi-layer NLA ×4", "#c77d2e"), ("transcript", "transcript only", "#9a9a9a")]
MIN = int(os.environ.get("FINAL_MIN_DRAWS", "8"))
ARMS = [x for x in ARMS if x[0] not in os.environ.get("FINAL_SKIP", "").split(",")]


def pak(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def main():
    R = load_records(BATCHES, [a for a, *_ in ARMS], draws=range(10))
    P = {}
    for (b, a, pid, d), r in R.items():
        if r["score"] is not None: P.setdefault((b, pid), {}).setdefault(a, []).append(r["score"] >= 7)
    P = {k: v for k, v in P.items() if all(len(v.get(a, [])) >= MIN for a, *_ in ARMS)}
    nd = np.mean([len(v[a]) for v in P.values() for a, *_ in ARMS])
    S = strata()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14})
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharey="row")
    for i, (metric, k) in enumerate((("pass@1", 1), ("pass@5", 5))):
        f = (lambda xs: float(np.mean(xs))) if k == 1 else (lambda xs: pak(len(xs), sum(xs), 5))
        for j, st in enumerate(STRATA):
            ax = axes[i, j]; ks = in_stratum(sorted(P), S, st)
            for x, (a, lab, col) in enumerate(ARMS):
                m, lo, hi = boot_ci([f(P[q][a]) for q in ks])
                m, lo, hi = 100 * m, 100 * lo, 100 * hi
                ax.bar(x, m, 0.72, color=col); ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="#333", capsize=4, lw=1.3)
                ax.text(x, hi + 0.8, f"{m:.1f}", ha="center", fontsize=12)
            _, _, d, dlo, dhi = boot([f(P[q]["olens_rl_B"]) for q in ks], [f(P[q]["mlnla_B"]) for q in ks])
            ax.set_title(f"{metric}, {st} (n={len(ks)})", loc="left")
            ax.text(0.02, 0.97, f"OLens − NLA, paired: {100 * d:+.1f} [{100 * dlo:+.1f}, {100 * dhi:+.1f}]", transform=ax.transAxes,
                    va="top", fontsize=11.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb"))
            ax.set_xticks(range(len(ARMS)), [lab for _, lab, _ in ARMS], rotation=12, fontsize=11.5)
            ax.set_ylim(40 if k == 1 else 55, 100); ax.grid(axis="y", alpha=0.3)
        axes[i, 0].set_ylabel(f"{metric}: % of incidents (judge ≥ 7)")
    fig.suptitle(f"Free-form explanations, 915 fresh CHIVE incidents, same harness for every arm "
                 f"(~{nd:.1f} draws per incident; 95% CI over incidents; pass@5 unbiased)", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "final_freeform_10draws_bars", export="final_freeform_10draws_bars")
    print("n incidents", len(P), "draws", round(nd, 1))


if __name__ == "__main__":
    main()
