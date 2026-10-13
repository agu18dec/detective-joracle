"""Figure for the CHIVE claim eval: AUROC per setting and method (all claims, least-confident
third), and accuracy on the claims the text-only plain answerer gets wrong (selected on its first
draw, every method scored on other runs). 95% CIs: bootstrap over incidents.

    python scripts/plot_claims.py out=/tmp/chive_claims claims=data/chive/claims/claims300.jsonl \\
        confidence=<confidence.json> plots=plots/chive
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

from detective_joracle.chive.claims import auc  # noqa: E402

SETTINGS = [  # (eval dir, label)
    ("plain", "plain answerer\n(no tools)"),
    ("focused", "focused episode\nper claim"),
    ("two_agent_ro", "two-agent:\nA read-only"),
    ("two_agent", "two-agent:\nA + 1 experiment"),
]
METHODS = {  # arm name per setting -> method
    "plain": "transcript",
    "transcript": "transcript",
    "transcript_A": "transcript",
    "olens_rl": "OLens",
    "olens_rl_A": "OLens",
    "nla": "NLA",
    "nla_A": "NLA",
    "jlens": "J-lens",
    "jlens_A": "J-lens",
    "olens_rl_otherq": "OLens, other incident's readouts",
}
COLOR = {
    "transcript": "#9a9a9a",
    "OLens": "#1f6f8b",
    "NLA": "#c77d2e",
    "J-lens": "#7a5ea8",
    "OLens, other incident's readouts": "#bfdbe5",
}
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15})

Rows = dict[str, dict[int, tuple[float, bool, str]]]  # claim -> draw -> (p, label, prompt_id)


def load(out: Path, ev: str) -> dict[str, Rows]:
    d: dict[str, Rows] = defaultdict(lambda: defaultdict(dict))
    for f in (out / ev).glob("*/*.d*.json"):
        r = json.loads(f.read_text())
        p = (r.get("result") or {}).get("probability_true")
        if p is None:
            continue
        draw = int(f.name.rsplit(".d", 1)[1].split(".")[0])
        c = r["claim"]
        d[r["arm"]][c["claim_id"]][draw] = (float(p), bool(c["label"]), c["prompt_id"])
    return d


def boot(rows: Rows, ids: list[str], stat: Any) -> tuple[float, float, float]:
    by_inc: dict[str, list[str]] = defaultdict(list)
    for c in ids:
        if c in rows:
            by_inc[next(iter(rows[c].values()))[2]].append(c)
    incs = sorted(by_inc)
    point = stat([c for i in incs for c in by_inc[i]])
    rng = random.Random(0)
    bs = sorted(
        x
        for x in (
            stat([c for i in (rng.choice(incs) for _ in incs) for c in by_inc[i]])
            for _ in range(500)
        )
        if x == x
    )
    return point, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.out = pydra.REQUIRED
        self.claims = "data/chive/claims/claims300.jsonl"
        self.confidence = pydra.REQUIRED
        self.plots = "plots/chive"

    def finalize(self) -> None:
        self.out, self.plots = Path(self.out), Path(self.plots)


@pydra.main(Config)
def main(cfg: Config) -> None:
    cfg.plots.mkdir(parents=True, exist_ok=True)
    claims = [json.loads(line) for line in Path(cfg.claims).open()]
    conf = json.loads(Path(cfg.confidence).read_text())
    ranked = sorted(conf, key=lambda p: conf[p]["top_p"])
    low = set(ranked[: len(ranked) // 3])
    data = {ev: load(cfg.out, ev) for ev, _ in SETTINGS}
    plain = data["plain"]["plain"]
    misread = [c for c, v in plain.items() if 0 in v and (v[0][0] >= 0.5) != v[0][1]]
    all_ids = [c["claim_id"] for c in claims]
    strata = {
        "AUROC, all 688 claims": (all_ids, "auroc"),
        "AUROC, least-confident third": (
            [c["claim_id"] for c in claims if c["prompt_id"] in low],
            "auroc",
        ),
        f"accuracy on the {len(misread)} claims the\ntext-only answerer gets wrong": (
            misread,
            "acc",
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    methods = list(COLOR)
    for ax, (title, (ids, metric)) in zip(axes, strata.items(), strict=True):
        for j, (ev, _) in enumerate(SETTINGS):
            arms = [a for a in data[ev] if a in METHODS]
            arms.sort(key=lambda a: methods.index(METHODS[a]))
            w = 0.8 / max(len(arms), 1)
            for i, a in enumerate(arms):
                rows = data[ev][a]
                # misread claims: score a run other than the selecting draw for the plain answerer
                dr = 1 if (ev == "plain" and metric == "acc") else 0

                def stat(
                    sub: list[str], rows: Rows = rows, dr: int = dr, metric: str = metric
                ) -> float:
                    pts = [
                        rows[c].get(dr) or next(iter(rows[c].values())) for c in sub if c in rows
                    ]
                    if metric == "acc":
                        return 100 * sum((p >= 0.5) == y for p, y, _ in pts) / max(len(pts), 1)
                    return auc([(p, y) for p, y, _ in pts])

                m, lo, hi = boot(rows, ids, stat)
                x = j + (i - (len(arms) - 1) / 2) * w
                meth = METHODS[a]
                ax.bar(x, m, w * 0.9, color=COLOR[meth], label=meth if j == 1 else None)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="#333", lw=1.2, capsize=2)
        ax.set_xticks(range(len(SETTINGS)), [lab for _, lab in SETTINGS], fontsize=12)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.3)
        if metric == "auroc":
            ax.set_ylim(0.75, 1.0)
            ax.axhline(0.5, color="#c0392b", ls="--", lw=1)
        else:
            ax.set_ylim(0, 60)
            ax.set_ylabel("% claims answered correctly")
    axes[0].set_ylabel("AUROC (0.5 = chance)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m], label=m) for m in methods]
    axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle(
        "CHIVE claim eval (CHIVE's own format): would this edit move the behaviour by >= 30 points? "
        "(95% CI, bootstrap over incidents)",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(cfg.plots / "claims_auroc.png", dpi=160, bbox_inches="tight")
    print(f"wrote {cfg.plots / 'claims_auroc.png'}; {len(misread)} misread claims")


if __name__ == "__main__":
    main()
