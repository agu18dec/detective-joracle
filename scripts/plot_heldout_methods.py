"""Held-out free-form (204 incidents x 5 draws, frozen harness): pass@1 and pass@5 per method,
all incidents and the least-confident third, with 95% bootstrap CIs over incidents.

    python scripts/plot_heldout_methods.py plots=plots/chive
"""

import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pydra  # noqa: E402

ARMS = {  # label -> (record dir, colour)
    "OLens": ("/workspace/agam/chive_runs_dj/climb_test/olens_rl_B", "#1f6f8b"),
    "J-lens": ("/workspace/agam/chive_runs_dj/climb_test/jlens_B", "#7a5ea8"),
    "multi-layer NLA": ("/tmp/chive_mlnla/climb_test/mlnla_B", "#c77d2e"),
    "transcript only": ("/workspace/agam/chive_runs_dj/climb_test/transcript", "#9a9a9a"),
}
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16})


def passes(d: str) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = defaultdict(list)
    for f in Path(d).glob("*.d*.json"):
        r = json.loads(f.read_text())
        j = r.get("judge") or {}
        if j.get("score") is not None:
            out[r["prompt_id"]].append(bool(j["pass"]))
    return {p: v for p, v in out.items() if len(v) == 5}


def pass_at(v: list[bool], k: int) -> float:
    return 1 - comb(5 - sum(v), k) / comb(5, k)


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.plots = "plots/chive"
        self.confidence = "/workspace/agam/chive_runs_dj/confidence.json"

    def finalize(self) -> None:
        self.plots = Path(self.plots)


@pydra.main(Config)
def main(cfg: Config) -> None:
    cfg.plots.mkdir(parents=True, exist_ok=True)
    data = {a: passes(d) for a, (d, _) in ARMS.items()}
    common = sorted(set.intersection(*[set(x) for x in data.values()]))
    conf = json.loads(Path(cfg.confidence).read_text())
    ranked = sorted(conf, key=lambda p: conf[p]["top_p"])
    low = set(ranked[: len(ranked) // 3])
    panels = [
        (f"pass@1, all {len(common)} incidents", common, 1),
        (f"pass@5, all {len(common)} incidents", common, 5),
        (
            f"pass@1, least-confident third ({sum(p in low for p in common)})",
            [p for p in common if p in low],
            1,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    for ax, (title, ids, k) in zip(axes, panels, strict=True):
        for i, (label, (_, colour)) in enumerate(ARMS.items()):
            xs = [pass_at(data[label][p], k) for p in ids]
            rng = random.Random(0)
            bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(2000))
            m = 100 * sum(xs) / len(xs)
            lo, hi = 100 * bs[50], 100 * bs[1949]
            ax.bar(i, m, 0.75, color=colour, label=label)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="#333", lw=1.4, capsize=4)
            ax.text(i, hi + 1, f"{m:.1f}", ha="center", fontsize=13)
        ax.set_xticks(range(len(ARMS)), list(ARMS), rotation=15)
        ax.set_title(title, loc="left")
        ax.set_ylim(40, 100)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("% of incidents (judge >= 7)")
    fig.suptitle(
        "Held-out free-form explanations, same harness for every method (95% CI over incidents)",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(cfg.plots / "heldout_methods.png", dpi=160, bbox_inches="tight")
    print(f"wrote {cfg.plots / 'heldout_methods.png'} ({len(common)} incidents)")


if __name__ == "__main__":
    main()
