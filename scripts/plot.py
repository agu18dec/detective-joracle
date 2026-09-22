"""The grid as one figure: detection by arm per organism, then what the judges add.

    python scripts/plot.py [auditor=google/gemini-3.8-flash] [out_root=outputs/live]
    -> <out_root>/plots/live_grid.png

Needs matplotlib (``pip install -e ".[plot]"``). Arms and organisms are discovered from the
records; arms are ordered black-box first, then alphabetically.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from detective_joracle.stats import wilson_interval  # noqa: E402

PALETTE = ["#555555", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def load(root: Path, auditor: str) -> dict[tuple[str, str], dict[str, list[int]]]:
    """``{(organism, arm): {hit, close, plaus, chat: [per-run values]}}`` for one auditor."""
    cells: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"hit": [], "close": [], "plaus": [], "chat": []}
    )
    aud_dir = auditor.replace("/", "_")
    for f in (root / "runs/live").glob(f"*/*/{aud_dir}/seed_*.json"):
        r = json.loads(f.read_text())
        if "judged" not in r or not r.get("result") or r["organism"] == "base":
            continue
        arm = f.parents[2].name
        q = next(iter(r["judged"]))
        jv = (r.get("judges") or {}).get(q) or {}
        c = cells[(r["organism"], arm)]
        c["hit"].append(int(r["judged"][q]["success"]["match"]))
        for k, src in (
            ("close", "closeness"),
            ("plaus", "plausible5"),
            ("chat", "chat_sufficient"),
        ):
            if jv.get(src) is not None:
                c[k].append(int(jv[src]["score"]))
    return cells


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args: dict[str, str] = dict(a.split("=", 1) for a in sys.argv[1:])
    auditor = args.get("auditor", "google/gemini-3.8-flash")
    root = REPO / args.get("out_root", "outputs/live")
    cells = load(root, auditor)
    if not cells:
        raise SystemExit(f"no judged records for {auditor} under {root}")
    orgs = sorted({o for o, _ in cells})
    arms = sorted({a for _, a in cells}, key=lambda a: (not a.startswith("blackbox"), a))
    colors = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(arms)}
    fig, axes = plt.subplots(3, 1, figsize=(17, 15), sharex=True)
    x = list(range(len(orgs)))
    w = 0.8 / len(arms)
    panels: list[tuple[str, str, tuple[float, float], bool]] = [
        ("hit", "detection (any of the ranked predictions matches; Wilson 95%)", (0, 1.05), True),
        (
            "close",
            "closeness of the TOP hypothesis to the planted quirk (judge, 1-10)",
            (0, 10.5),
            False,
        ),
        (
            "plaus",
            "picked the true quirk out of true + 4 plausible distractors (judge)",
            (0, 1.05),
            True,
        ),
    ]
    for ax, (key, title, ylim, is_rate) in zip(axes, panels, strict=True):
        for j, arm in enumerate(arms):
            xs: list[float] = []
            ys: list[float] = []
            lo: list[float] = []
            hi: list[float] = []
            for i, org in enumerate(orgs):
                v = cells.get((org, arm), {}).get(key) or []
                if not v:
                    continue
                m = sum(v) / len(v)
                xs.append(i + (j - len(arms) / 2 + 0.5) * w)
                ys.append(m)
                if is_rate:
                    a, b = wilson_interval(sum(v), len(v))
                    lo.append(m - a)
                    hi.append(b - m)
                else:
                    lo.append(0)
                    hi.append(0)
            err: Any = [lo, hi] if is_rate else None
            ax.bar(
                xs,
                ys,
                w,
                color=colors[arm],
                label=arm,
                yerr=err,
                capsize=2,
                error_kw={"lw": 0.8, "alpha": 0.6},
            )
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=20, loc="left")
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(axis="y", alpha=0.3)
        if key == "plaus":
            ax.axhline(0.2, ls="--", color="k", lw=1, alpha=0.5)
            ax.text(len(orgs) - 0.55, 0.22, "chance", fontsize=14)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(orgs, fontsize=18, rotation=15, ha="right")
    axes[0].legend(
        ncol=len(arms), fontsize=15, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False
    )
    fig.suptitle(f"In-the-loop audit — investigator {auditor.split('/')[-1]}", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = root / "plots/live_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
