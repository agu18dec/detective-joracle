"""Figures for the CHIVE forensics evals: E1, E2 (+ concluder), candidate selection, top-20
coverage and the arbiter. Every figure is drawn from the judged run records.

    python scripts/plot_chive.py out=/workspace/agam/chive_runs_dj plots=plots/chive
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pydra  # noqa: E402

from detective_joracle.chive.data import read_ids  # noqa: E402

LABEL = {
    "transcript": "transcript only",
    "olens_rl": "OLens (RL)",
    "olens_distill": "OLens (distill)",
    "nla": "NLA",
    "jlens": "J-lens",
    "olens_rl_posshuffled": "OLens, positions shuffled",
    "olens_rl_shuffled": "OLens, other incident's readouts",
}
COLOR = {
    "transcript": "#9a9a9a",
    "olens_rl": "#1f6f8b",
    "olens_distill": "#5aa7c2",
    "nla": "#c77d2e",
    "jlens": "#7a5ea8",
    "olens_rl_posshuffled": "#9fc7d4",
    "olens_rl_shuffled": "#d9d9d9",
}
plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16})


def per_incident(out: Path, ev: str, key: str = "judge") -> dict[str, dict[str, float]]:
    """arm -> prompt_id -> pass@1 averaged over draws."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in (out / ev).glob("*/*.d*.json"):
        r = json.loads(f.read_text())
        j = r.get(key) or {}
        if j.get("score") is not None:
            acc[r["arm"]][r["prompt_id"]].append(float(j["pass"]))
    return {a: {p: sum(v) / len(v) for p, v in d.items()} for a, d in acc.items()}


def mean_ci(xs: list[float]) -> tuple[float, float, float]:
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(2000))
    return 100 * sum(xs) / len(xs), 100 * bs[50], 100 * bs[1949]


def paired_ci(
    a: dict[str, float], b: dict[str, float], sub: set[str]
) -> tuple[float, float, float]:
    d = [a[p] - b[p] for p in a if p in b and p in sub]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
    return 100 * sum(d) / len(d), 100 * bs[100], 100 * bs[3899]


def grouped_bars(ax, data, arms, strata, title, ylabel="pass@1 (%)"):  # type: ignore[no-untyped-def]
    w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        for j, sub in enumerate(strata.values()):
            xs = [v for p, v in data.get(arm, {}).items() if p in sub]
            if not xs:
                continue
            m, lo, hi = mean_ci(xs)
            x = j + (i - (len(arms) - 1) / 2) * w
            ax.bar(
                x,
                m,
                w * 0.92,
                color=COLOR.get(arm, "#888"),
                edgecolor="white",
                label=LABEL.get(arm, arm) if j == 0 else None,
            )
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="#333", lw=1.4, capsize=3)
    ax.set_xticks(range(len(strata)), list(strata))
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.out = pydra.REQUIRED
        self.plots = "plots/chive"
        self.ids = "data/chive/splits/all300.txt"
        self.hard = "data/chive/splits/hard92.txt"

    def finalize(self) -> None:
        self.out, self.plots = Path(self.out), Path(self.plots)


@pydra.main(Config)
def main(cfg: Config) -> None:
    cfg.plots.mkdir(parents=True, exist_ok=True)
    ids, hard = set(read_ids(Path(cfg.ids))), set(read_ids(Path(cfg.hard)))
    conf = {p: v["top_p"] for p, v in json.loads((cfg.out / "confidence.json").read_text()).items()}
    ranked = sorted(conf, key=lambda p: conf[p])
    low = set(ranked[: len(ranked) // 3])
    strata = {"all 300": ids, "least-confident third": low, "hard 92 (selected\non errors)": hard}

    # E1 overview
    e1 = per_incident(cfg.out, "e1_readonly")
    arms = [
        "transcript",
        "olens_rl",
        "olens_distill",
        "nla",
        "jlens",
        "olens_rl_posshuffled",
        "olens_rl_shuffled",
    ]
    fig, ax = plt.subplots(figsize=(16, 7))
    grouped_bars(ax, e1, arms, strata, "E1: read-only investigations (2 draws per incident)")
    fig.tight_layout()
    fig.savefig(cfg.plots / "e1_overview.png", dpi=160, bbox_inches="tight")

    # E1 paired vs transcript
    comps = [a for a in arms if a != "transcript"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), sharey=True)
    for ax, (name, sub) in zip(
        axes, [("all 300", ids), ("least-confident third", low)], strict=True
    ):
        for i, arm in enumerate(comps):
            m, lo, hi = paired_ci(e1[arm], e1["transcript"], sub)
            ax.errorbar(
                m, i, xerr=[[m - lo], [hi - m]], fmt="o", ms=10, lw=2.5, capsize=5, color=COLOR[arm]
            )
        ax.axvline(0, color="#444", lw=1.2)
        ax.set_title(name, loc="left")
        ax.set_xlabel("pass@1 minus transcript only (pp), paired, 95% CI")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(comps)), [LABEL[a] for a in comps])
    fig.suptitle("E1: what each method adds over reading the transcript", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(cfg.plots / "e1_vs_transcript.png", dpi=160, bbox_inches="tight")

    # E2 + concluder
    e2 = per_incident(cfg.out, "e2_one_experiment")
    e2c = per_incident(cfg.out, "e2_one_experiment", key="concluder_judge")
    arms2 = ["transcript", "olens_rl", "olens_distill", "nla", "jlens"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), sharey=True)
    grouped_bars(
        axes[0], e2, arms2, {"all 300": ids, "hard 92": hard}, "E2: one experiment allowed"
    )
    grouped_bars(
        axes[1],
        e2c,
        arms2,
        {"all 300": ids, "hard 92": hard},
        "E2 two-agent: a fresh agent sees\nonly the experiment and its result",
    )
    axes[0].get_legend().remove()
    fig.tight_layout()
    fig.savefig(cfg.plots / "e2_overview.png", dpi=160, bbox_inches="tight")

    # candidate selection by prior
    rows = {
        json.loads(line)["prompt_id"]: json.loads(line)
        for line in Path("data/chive/choices/all300_8opt.jsonl").read_text().splitlines()
    }
    oo = {p: r["options_only_correct"] / r["options_only_samples"] for p, r in rows.items()}
    cs = per_incident(cfg.out, "choice8", key="choice_eval")
    cstrata = {
        "all 199": set(rows),
        "prior-misleading\n(blind picker 0/5)": {p for p, x in oo.items() if x == 0},
        "mixed": {p for p, x in oo.items() if 0 < x < 0.8},
        "prior-solvable\n(blind picker >= 4/5)": {p for p, x in oo.items() if x >= 0.8},
    }
    fig, ax = plt.subplots(figsize=(16, 7))
    grouped_bars(
        ax,
        cs,
        arms2,
        cstrata,
        "Candidate selection: 8 options, chance 12.5%",
        ylabel="picks the true cause (%)",
    )
    ax.axhline(12.5, color="#c0392b", ls="--", lw=1.5)
    fig.tight_layout()
    fig.savefig(cfg.plots / "candidate_selection.png", dpi=160, bbox_inches="tight")

    # top-20 coverage (generation vs narrowing)
    lists = [json.loads(f.read_text()) for f in (cfg.out / "top20").glob("*.json")]
    lists = [x for x in lists if x.get("item_scores")]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for name, sub, c in [("all 300", ids, "#1f6f8b"), ("hard 92", hard, "#c0392b")]:
        ls = [x for x in lists if x["prompt_id"] in sub]
        ys = [
            100 * sum(any(s >= 7 for s in x["item_scores"][:k]) for x in ls) / len(ls)
            for k in range(1, 21)
        ]
        ax.plot(range(1, 21), ys, "o-", lw=3, ms=6, color=c, label=name)
    ax.set_xlabel("top k of the transcript-only hypothesis list")
    ax.set_ylabel("incidents with a passing hypothesis (%)")
    ax.set_title("The true cause is usually already conceivable from the text", loc="left")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(cfg.plots / "top20_coverage.png", dpi=160)

    # arbiter (E1)
    md = (cfg.out / "e1_readonly" / "arbiter.md").read_text().splitlines()
    pts = []
    for line in md:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if (
            len(cells) > 3
            and cells[3].endswith("%")
            and cells[0] in ("olens_rl", "nla", "jlens", "transcript:self", "olens_rl:self")
        ):
            pts.append((cells[0], float(cells[3].rstrip("%"))))
    names = {
        "olens_rl": "transcript + OLens (RL)",
        "nla": "transcript + NLA",
        "jlens": "transcript + J-lens",
        "transcript:self": "transcript + transcript\n(control)",
        "olens_rl:self": "OLens + OLens\n(control)",
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    xs = list(range(len(pts)))
    ax.bar(
        xs,
        [v for _, v in pts],
        color=[
            "#1f6f8b" if "OLens" in names[k] and "control" not in names[k] else "#bbbbbb"
            for k, _ in pts
        ],
    )
    for x, (_, v) in zip(xs, pts, strict=True):
        ax.text(x, v + 0.4, f"{v:.1f}%", ha="center", fontsize=14)
    ax.axhline(
        100 * sum(e1["transcript"].values()) / len(e1["transcript"]),
        color="#444",
        ls="--",
        lw=1.5,
        label="transcript only",
    )
    ax.set_xticks(xs, [names[k] for k, _ in pts], fontsize=12)
    ax.set_ylim(55, 72)
    ax.set_ylabel("pass@1 of the arbiter's pick (%)")
    ax.set_title("Arbiter: a fresh model picks between two final answers (E1, all 300)", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.plots / "e1_arbiter.png", dpi=160)
    print(
        "wrote e1_overview, e1_vs_transcript, e2_overview, candidate_selection, top20_coverage, "
        "e1_arbiter"
    )


if __name__ == "__main__":
    main()
