"""Point-level results (points_chive.py): recall of NON-OBVIOUS points, recall of PRIMARY-cause
points, and the disproved-claim rate, per method and stratum, 95% bootstrap CIs over incidents.
Definitions match points_chive.py report: obvious = an independent no-tools answer scores the
point >= 7 (1-10 scale) in >= 2 of 3 tries; recall = mean (score - 1) / 9; disproved rate =
contradicted / asserted claims (rivals the answer rejects excluded); rivals rejected = share of
CHIVE's ruled-out alternatives the answer explicitly rules out.

    python scripts/plot_points.py out=/tmp/chive_points4 plots=plots/chive
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

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
PANELS = [
    ("nonobvious", "recall of NON-OBVIOUS points (higher = better)"),
    ("primary", "recall of PRIMARY-cause points (higher = better)"),
    ("disproved", "share of asserted claims DISPROVED (lower = better)"),
    ("rival_rejected", "ruled-out alternatives REJECTED (higher = better)"),
]
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14})


def per_incident(out: Path) -> dict[str, dict[str, dict[str, float]]]:
    g: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for f in (out / "grades").glob("*/*.json"):
        r = json.loads(f.read_text())
        g[r["arm"]][r["prompt_id"]].append(r)
    plain = {p: v for p, v in g.pop("plain").items() if len(v) == 3}
    tags = {f.stem: json.loads(f.read_text())["primary"] for f in (out / "tags").glob("*.json")}
    full = {a: {p: v for p, v in d.items() if len(v) == 5} for a, d in g.items()}
    common = set.intersection(*[set(x) for x in full.values()]) & set(plain) & set(tags)
    res: dict[str, dict[str, dict[str, float]]] = {a: {} for a in full}
    for p in sorted(common):
        n = len(tags[p])
        idx = {
            "nonobvious": [
                i for i in range(n) if sum(r["point_scores"][i] >= 7 for r in plain[p]) < 2
            ],
            "primary": [i for i in range(n) if tags[p][i]],
        }
        for a in full:
            rs = full[a][p]
            row = {
                k: (
                    sum(sum(r["point_scores"][i] - 1 for i in ix) / (9 * len(ix)) for r in rs) / 5
                    if ix
                    else float("nan")
                )
                for k, ix in idx.items()
            }
            claims = [c for r in rs for c in r["claims"] if c.get("label") != "rejected"]
            row["disproved"] = sum(c["label"] == "contradicted" for c in claims) / max(
                len(claims), 1
            )
            rv = [
                x
                for f in (out / "rivals" / a).glob(f"{p}.d*.json")
                for x in json.loads(f.read_text())["rivals"]
            ]
            row["rival_rejected"] = (
                sum(x == "rejected" for x in rv) / len(rv) if rv else float("nan")
            )
            res[a][p] = row
    return res


def mean_ci(xs: list[float]) -> tuple[float, float, float]:
    xs = [x for x in xs if x == x]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(2000))
    return 100 * sum(xs) / len(xs), 100 * bs[50], 100 * bs[1949]


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.out = "/tmp/chive_points4"
        self.plots = "plots/chive"
        self.confidence = "/workspace/agam/chive_runs_dj/confidence.json"
        self.hard = "data/chive/splits/hard92.txt"

    def finalize(self) -> None:
        self.out, self.plots = Path(self.out), Path(self.plots)


@pydra.main(Config)
def main(cfg: Config) -> None:
    data = per_incident(cfg.out)
    ids = sorted(next(iter(data.values())))
    conf = json.loads(Path(cfg.confidence).read_text())
    ranked = sorted(conf, key=lambda p: conf[p]["top_p"])
    low = set(ranked[: len(ranked) // 3])
    hard = set(Path(cfg.hard).read_text().split())
    strata = {
        f"all ({len(ids)})": ids,
        f"least-confident\nthird ({sum(p in low for p in ids)})": [p for p in ids if p in low],
        f"hard ({sum(p in hard for p in ids)})": [p for p in ids if p in hard],
    }
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    w = 0.8 / len(ARMS)
    for ax, (metric, title) in zip(axes.flat, PANELS, strict=True):
        for j, sub in enumerate(strata.values()):
            for i, (arm, (name, colour)) in enumerate(ARMS.items()):
                m, lo, hi = mean_ci([data[arm][p][metric] for p in sub])
                x = j + (i - (len(ARMS) - 1) / 2) * w
                ax.bar(x, m, w * 0.9, color=colour, label=name if j == 0 else None)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="#333", lw=1.2, capsize=3)
        ax.set_xticks(range(len(strata)), list(strata))
        ax.set_title(title, loc="left")
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.3)
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle(
        "Point-level grading of held-out free-form explanations (5 runs per incident; "
        "95% CI over incidents)",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    cfg.plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.plots / "points.png", dpi=160, bbox_inches="tight")
    print(f"wrote {cfg.plots / 'points.png'} ({len(ids)} incidents)")


if __name__ == "__main__":
    main()
