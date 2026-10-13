"""Point-level results: recall of non-obvious points and the disproved-claim rate, per method
and stratum, with 95% bootstrap CIs over incidents. Draws the current run (distinct points,
0-10 scores) and, below it, the earlier run (0/1/2 scores, original points).

    python scripts/plot_points.py plots=plots/chive
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pydra  # noqa: E402

ARMS = {
    "olens_rl_B": ("OLens", "#1f6f8b"),
    "mlnla_B": ("multi-layer NLA", "#c77d2e"),
    "jlens_B": ("J-lens", "#7a5ea8"),
    "transcript": ("transcript only", "#9a9a9a"),
}
RUNS = [
    ("/tmp/chive_points", 10, "distinct points, scored 0-10"),
    ("/tmp/chive_points_v1", 2, "earlier run: original points, scored 0/1/2"),
]
BROKEN = {"prompt_002961"}  # v1: its point list came back as a string (1,303 'points')
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14})


def per_incident(run: str, scale: int) -> dict[str, dict[str, tuple[float, float]]]:
    g: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in Path(run, "grades").glob("*/*.json"):
        r = json.loads(f.read_text())
        if r["prompt_id"] not in BROKEN:
            g[r["arm"]][r["prompt_id"]].append(r)
    full = {a: {p: v for p, v in d.items() if len(v) == 5} for a, d in g.items()}
    common = sorted(set.intersection(*[set(x) for x in full.values()]))
    hi = 7 if scale == 10 else 2
    out: dict[str, dict[str, tuple[float, float]]] = {a: {} for a in full}
    for p in common:
        tr = full["transcript"][p]
        n = len(tr[0]["point_scores"])
        non = [i for i in range(n) if sum(r["point_scores"][i] >= hi for r in tr) < 3]
        for a in full:
            rs = full[a][p]
            rec = (
                sum(sum(r["point_scores"][i] for i in non) / (scale * len(non)) for r in rs) / 5
                if non
                else float("nan")
            )
            claims = [c for r in rs for c in r["claims"]]
            con = sum(c["label"] == "contradicted" for c in claims) / max(len(claims), 1)
            out[a][p] = (rec, con)
    return out


def mean_ci(xs: list[float]) -> tuple[float, float, float]:
    xs = [x for x in xs if x == x]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(2000))
    return 100 * sum(xs) / len(xs), 100 * bs[50], 100 * bs[1949]


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.plots = "plots/chive"
        self.confidence = "/workspace/agam/chive_runs_dj/confidence.json"
        self.hard = "data/chive/splits/hard92.txt"

    def finalize(self) -> None:
        self.plots = Path(self.plots)


@pydra.main(Config)
def main(cfg: Config) -> None:
    conf = json.loads(Path(cfg.confidence).read_text())
    ranked = sorted(conf, key=lambda p: conf[p]["top_p"])
    low = set(ranked[: len(ranked) // 3])
    hard = set(Path(cfg.hard).read_text().split())
    fig, axes = plt.subplots(len(RUNS), 2, figsize=(20, 6.2 * len(RUNS)))
    for row, (run, scale, label) in enumerate(RUNS):
        data = per_incident(run, scale)
        ids = sorted(next(iter(data.values())))
        strata = {
            f"all ({len(ids)})": ids,
            f"least-confident\nthird ({sum(p in low for p in ids)})": [p for p in ids if p in low],
            f"hard ({sum(p in hard for p in ids)})": [p for p in ids if p in hard],
        }
        for col, (metric, title) in enumerate(
            [
                (0, "recall of NON-OBVIOUS points"),
                (1, "share of claims DISPROVED by CHIVE's experiments"),
            ]
        ):
            ax = axes[row][col]
            w = 0.8 / len(ARMS)
            for j, sub in enumerate(strata.values()):
                for i, (arm, (name, colour)) in enumerate(ARMS.items()):
                    m, lo, hi_ = mean_ci([data[arm][p][metric] for p in sub])
                    x = j + (i - (len(ARMS) - 1) / 2) * w
                    ax.bar(x, m, w * 0.9, color=colour, label=name if j == 0 else None)
                    ax.errorbar(x, m, yerr=[[m - lo], [hi_ - m]], color="#333", lw=1.2, capsize=3)
            ax.set_xticks(range(len(strata)), list(strata))
            ax.set_title(f"{title} — {label}", loc="left")
            ax.set_ylabel("%" + (" (higher = better)" if metric == 0 else " (lower = better)"))
            ax.grid(axis="y", alpha=0.3)
    axes[0][1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle(
        "Point-level grading of held-out free-form explanations (204 incidents x 5 runs; "
        "95% CI over incidents)",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    cfg.plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.plots / "points.png", dpi=160, bbox_inches="tight")
    print(f"wrote {cfg.plots / 'points.png'}")


if __name__ == "__main__":
    main()
