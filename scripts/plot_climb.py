"""Figures for the harness climb: dev A/B forest, dev keep vs rescue, held-out pass@k.

python scripts/plot_climb.py out=/workspace/agam/chive_runs_dj plots=plots/chive
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

from detective_joracle.chive.data import read_ids  # noqa: E402

# dev arms: label -> (eval dir, arm name)
DEV = {
    "OLens + harness": ("climb_textfirst_absence", "olens_rl_tfa"),
    "OLens + harness + readout digest": ("climb_tfa_digest", "olens_rl_tfa_digest"),
    "OLens + harness + 1 experiment": ("climb_tfa_experiment", "olens_rl_tfa_exp"),
    "OLens + harness + J-lens tool": ("climb_tfa_jlens", "olens_rl_tfa_jlens"),
    "OLens + harness + continue_from (guided)": ("climb_tfa_continue_guided", "olens_rl_tfa_cont2"),
    "OLens + harness + continue_from": ("climb_textfirst_absence_continue", "olens_rl_tfac"),
    "OLens + harness + boundary readouts": ("climb_tfa_boundary", "olens_rl_tfa_boundary"),
    "NLA + harness": ("climb_readers_improved", "nla_tfa"),
    "J-lens + harness": ("climb_readers_improved", "jlens_tfa"),
    "transcript + 1 experiment": ("climb_tfa_experiment", "transcript_exp"),
    "transcript + continue_from": ("climb_textfirst_absence_continue", "transcript_cont"),
    "transcript": ("climb_textfirst_absence", "transcript"),
}
OLENS = "#1f6f8b"  # OLens harness variants
OTHER = "#9a9a9a"  # baselines and other readers
HARD = "#c0392b"
plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16})


def draws(out: Path, ev: str, arm: str, ids: set[str]) -> dict[str, list[float]]:
    d: dict[str, list[float]] = defaultdict(list)
    for f in (out / ev / arm).glob("*.json"):
        r = json.loads(f.read_text())
        j = r.get("judge") or {}
        if j.get("score") is not None and r["prompt_id"] in ids:
            d[r["prompt_id"]].append(float(j["pass"]))
    return d


def paired(a: dict[str, list[float]], b: dict[str, list[float]]) -> tuple[float, float, float]:
    ps = [p for p in a if p in b]
    d = [sum(a[p]) / len(a[p]) - sum(b[p]) / len(b[p]) for p in ps]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
    return 100 * sum(d) / len(d), 100 * bs[100], 100 * bs[3899]


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.out = pydra.REQUIRED
        self.plots = "plots/chive"
        self.dev = "data/chive/splits/climb_dev96.txt"
        self.test = "data/chive/splits/climb_test204.txt"
        self.hard = "data/chive/splits/hard92.txt"

    def finalize(self) -> None:
        self.out, self.plots = Path(self.out), Path(self.plots)


def forest(cfg: Config, dev: set[str]) -> None:
    data = {k: draws(cfg.out, *v, dev) for k, v in DEV.items()}
    base = data["OLens + harness"]
    rows = []
    for label, d in data.items():
        if not d:
            continue
        m = sum(sum(v) / len(v) for v in d.values()) / len(d)
        rows.append((label, 100 * m, *(paired(d, base) if d is not base else (0.0, 0.0, 0.0))))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(rows) + 1.8))
    for i, (label, m, dm, lo, hi) in enumerate(rows):
        c = OLENS if label.startswith("OLens") else OTHER
        ax.errorbar(dm, i, xerr=[[dm - lo], [hi - dm]], fmt="o", color=c, ms=9, lw=2.5, capsize=5)
        ax.text(1.02, i, f"{m:.1f}%", transform=ax.get_yaxis_transform(), va="center", fontsize=14)
    ax.axvline(0, color="#444", lw=1.2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("pass@1 minus OLens + harness (pp), paired, 95% CI")
    ax.set_title("Harness climb on dev (96 incidents x 2 draws)", loc="left")
    ax.text(
        1.02,
        len(rows) - 0.3,
        "pass@1",
        transform=ax.get_yaxis_transform(),
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.plots / "climb_dev_forest.png", dpi=160)


DEV_READERS = {  # label -> (eval dir, arm, style key in TEST_STYLE)
    "OLens + harness": ("climb_textfirst_absence", "olens_rl_tfa", "olens_rl_B"),
    "NLA + harness": ("climb_readers_improved", "nla_tfa", "nla_B"),
    "J-lens + harness": ("climb_readers_improved", "jlens_tfa", "jlens_B"),
    "transcript + continue_from": (
        "climb_textfirst_absence_continue",
        "transcript_cont",
        "transcript_cont",
    ),
    "transcript": ("climb_textfirst_absence", "transcript", "transcript"),
}


def keep_rescue(cfg: Config, dev: set[str]) -> None:
    """Dev: each reader under the original vs the harness (tool variants: forest)."""
    tr: dict[tuple[str, int], float] = {}
    for f in (cfg.out / "climb_textfirst_absence" / "transcript").glob("*.json"):
        r = json.loads(f.read_text())
        tr[(r["prompt_id"], int(r["seed"]))] = float(r["judge"]["pass"])
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for label, (ev, arm, style) in DEV_READERS.items():
        k = rr = nk = nr = 0.0
        for f in (cfg.out / ev / arm).glob("*.json"):
            r = json.loads(f.read_text())
            if r["prompt_id"] not in dev or (r.get("judge") or {}).get("score") is None:
                continue
            t = tr.get((r["prompt_id"], 1 - int(r["seed"])))
            if t is None:
                continue
            if t:
                k, nk = k + r["judge"]["pass"], nk + 1
            else:
                rr, nr = rr + r["judge"]["pass"], nr + 1
        if not nk:
            continue
        c, _ = TEST_STYLE[style]
        x, y = 100 * rr / nr, 100 * k / nk
        ax.scatter(
            x,
            y,
            s=170,
            color=c,
            edgecolor="white",
            lw=2,
            zorder=3,
            marker="s" if "original" in label else "o",
        )
        ax.annotate(label, (x, y), xytext=(9, -4), textcoords="offset points", fontsize=12)
    ax.set_xlim(10, 50)
    ax.set_ylim(75, 96)
    ax.set_xlabel("rescue: pass@1 where a transcript draw FAILED (%)")
    ax.set_ylabel("keep: pass@1 where a transcript draw PASSED (%)")
    ax.set_title("Keep vs rescue on dev (96 incidents x 2 draws)", loc="left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.plots / "climb_dev_keep_rescue.png", dpi=160)


TEST_ARMS = {
    "OLens + harness": "olens_rl_B",
    "OLens + harness, other question's readouts": "olens_rl_B_otherq",
    "NLA + harness": "nla_B",
    "J-lens + harness": "jlens_B",
    "transcript + continue_from": "transcript_cont",
    "transcript": "transcript",
}
TEST_STYLE = {  # colour per method; solid = harness, dashed = original
    "olens_rl_B": ("#1f6f8b", "-"),
    "olens_rl": ("#1f6f8b", "--"),
    "nla_B": ("#c77d2e", "-"),
    "nla": ("#c77d2e", "--"),
    "jlens_B": ("#7a5ea8", "-"),
    "jlens": ("#7a5ea8", "--"),
    "transcript_cont": ("#555555", ":"),
    "transcript": ("#9a9a9a", "-"),
}


def complete(cfg: Config, arm: str, sub: set[str]) -> dict[str, list[float]]:
    d = {p: v for p, v in draws(cfg.out, "climb_test", arm, sub).items() if len(v) >= 5}
    return d if len(d) >= 0.9 * len(sub) else {}  # an arm still running is left out


def passk(cfg: Config, test: set[str], hard: set[str], low: set[str]) -> None:
    strata = {
        "all 204": test,
        "least-confident third": test & low,
        "hard 46 (selected\non errors)": test & hard,
        "normal 158": test - hard,
    }
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.4))
    for ax, (name, sub) in zip(axes, strata.items(), strict=True):
        for label, arm in TEST_ARMS.items():
            d = complete(cfg, arm, sub)
            if not d:
                continue
            ys = [
                100
                * sum(
                    1.0 if 5 - sum(v) < k else 1 - comb(5 - int(sum(v)), k) / comb(5, k)
                    for v in d.values()
                )
                / len(d)
                for k in range(1, 6)
            ]
            c, ls = TEST_STYLE[arm]
            ax.plot(range(1, 6), ys, marker="o", ls=ls, lw=3, ms=7, label=label, color=c)
        ax.set_title(f"{name} (n={len(sub)})")
        ax.set_xlabel("k")
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pass@k (%)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Held-out test (204 incidents x 5 draws, read-only)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(cfg.plots / "heldout_passk.png", dpi=160, bbox_inches="tight")


def heldout_keep_rescue(cfg: Config, test: set[str]) -> None:
    """Each transcript draw selects (passed / failed); a method is scored on its OTHER draws."""
    tr = draws(cfg.out, "climb_test", "transcript", test)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    pts = []
    best = {"olens_rl_B", "nla_B", "jlens_B", "transcript_cont", "transcript", "olens_rl_B_otherq"}
    for label, arm in TEST_ARMS.items():
        d = complete(cfg, arm, test)
        if arm not in best or not d or arm == "transcript" and not tr:
            continue
        k = r = nk = nr = 0.0
        for p, v in d.items():
            t = tr.get(p)
            if not t:
                continue
            for i, sel in enumerate(t):
                others = [x for j, x in enumerate(v) if j != i]
                score = sum(others) / len(others)
                if sel:
                    k, nk = k + score, nk + 1
                else:
                    r, nr = r + score, nr + 1
        c, _ = TEST_STYLE[arm]
        pts.append((100 * r / nr, 100 * k / nk, label, c))
    placed: list[tuple[float, float]] = []
    for x, y, label, c in pts:
        near = any(abs(x - px) < 4 and abs(y - py) < 1.5 for px, py in placed)
        placed.append((x, y))
        ax.scatter(
            x,
            y,
            s=150,
            color=c,
            edgecolor="white",
            lw=2,
            zorder=3,
            marker="o" if "original" not in label else "s",
        )
        left = label.startswith("NLA")  # NLA and J-lens sit close together: label NLA leftwards
        ax.annotate(
            label,
            (x, y),
            xytext=(-10 if left else 8, -4 if left else (-18 if near else 5)),
            textcoords="offset points",
            fontsize=12,
            ha="right" if left else "left",
        )
    ax.set_xlim(10, 70)  # fixed, honest ranges: keep and rescue on comparable scales
    ax.set_ylim(70, 100)
    ax.set_xlabel("rescue: pass@1 where a transcript draw FAILED (%)")
    ax.set_ylabel("keep: pass@1 where a transcript draw PASSED (%)")
    ax.set_title("Keep vs rescue on the held-out test (204 incidents x 5 draws)", loc="left")
    ax.text(
        0.01,
        0.02,
        "keep = right where the transcript-only agent was right (no harm)\n"
        "rescue = right where the transcript-only agent was wrong (added value)",
        transform=ax.transAxes,
        fontsize=12,
        color="#444",
        va="bottom",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.plots / "heldout_keep_rescue.png", dpi=160)


def heldout_pareto(cfg: Config, test: set[str]) -> None:
    """Keep vs rescue at pass@1 and pass@4 (leave-one-out: each transcript draw defines right /
    wrong; every method is scored on the other 4 draws of that incident; averaged over the 5)."""
    tr = draws(cfg.out, "climb_test", "transcript", test)
    best = ["olens_rl_B", "nla_B", "jlens_B", "olens_rl_B_otherq", "transcript_cont", "transcript"]
    label_of = {v: k for k, v in TEST_ARMS.items()}
    rows: dict[str, dict[int, tuple[float, float]]] = {}
    for arm in best:
        d = complete(cfg, arm, test)
        if not d:
            continue
        rows[arm] = {}
        for k in (1, 4):
            kk = rr = nk = nr = 0.0
            for p, v in d.items():
                t = tr.get(p)
                if not t or len(t) < 5:
                    continue
                for i, sel in enumerate(t):
                    rest = [x for j, x in enumerate(v) if j != i]
                    n, c = len(rest), int(sum(rest))
                    score = 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
                    if sel:
                        kk, nk = kk + score, nk + 1
                    else:
                        rr, nr = rr + score, nr + 1
            rows[arm][k] = (100 * rr / nr, 100 * kk / nk)
    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    for ax, k in zip(axes, (1, 4), strict=True):
        pts = [(rows[a][k][0], rows[a][k][1], a) for a in rows]
        front = sorted(
            [p for p in pts if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in pts)]
        )
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            color="#bbbbbb",
            lw=2,
            ls="--",
            zorder=1,
            label="Pareto front",
        )
        for x, y, a in pts:
            c, _ = TEST_STYLE[a]
            ax.scatter(x, y, s=170, color=c, edgecolor="white", lw=2, zorder=3)
            left = a == "nla_B"
            ax.annotate(
                label_of[a],
                (x, y),
                xytext=(-10 if left else 8, -4 if left else 5),
                textcoords="offset points",
                fontsize=12,
                ha="right" if left else "left",
            )
        ax.set_title(f"pass@{k}", loc="left")
        ax.set_xlabel("rescue: right where a transcript run was WRONG (%)")
        ax.set_xlim(10, 80 if k == 4 else 60)
        ax.set_ylim(80 if k == 4 else 70, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("keep: right where a transcript run was RIGHT (%)")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle(
        "Held-out keep vs rescue (204 incidents; each scored on the 4 runs not used to "
        "label right/wrong)",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(cfg.plots / "heldout_pareto.png", dpi=160, bbox_inches="tight")
    for arm, r in rows.items():
        print(
            f"{label_of[arm]:42s} pass@1 keep {r[1][1]:.1f} rescue {r[1][0]:.1f} | "
            f"pass@4 keep {r[4][1]:.1f} rescue {r[4][0]:.1f}"
        )


@pydra.main(Config)
def main(cfg: Config) -> None:
    cfg.plots.mkdir(parents=True, exist_ok=True)
    dev, test = set(read_ids(Path(cfg.dev))), set(read_ids(Path(cfg.test)))
    hard = set(read_ids(Path(cfg.hard)))
    conf = json.loads((cfg.out / "confidence.json").read_text())
    ranked = sorted(conf, key=lambda p: conf[p]["top_p"])
    low = set(ranked[: len(ranked) // 3])
    forest(cfg, dev)
    keep_rescue(cfg, dev)
    passk(cfg, test, hard, low)
    heldout_keep_rescue(cfg, test)
    heldout_pareto(cfg, test)
    print(
        f"wrote {cfg.plots}/: climb_dev_forest, climb_dev_keep_rescue, heldout_passk, "
        "heldout_keep_rescue"
    )


if __name__ == "__main__":
    main()
