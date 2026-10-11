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
    "OLens, improved harness": ("climb_textfirst_absence", "olens_rl_tfa"),
    "OLens improved + readout digest": ("climb_tfa_digest", "olens_rl_tfa_digest"),
    "OLens improved + 1 experiment": ("climb_tfa_experiment", "olens_rl_tfa_exp"),
    "OLens improved + J-lens tool": ("climb_tfa_jlens", "olens_rl_tfa_jlens"),
    "OLens improved + continue_from (guided)": ("climb_tfa_continue_guided", "olens_rl_tfa_cont2"),
    "OLens improved + continue_from": ("climb_textfirst_absence_continue", "olens_rl_tfac"),
    "OLens improved + boundary readouts": ("climb_tfa_boundary", "olens_rl_tfa_boundary"),
    "OLens, original harness": ("climb_textfirst_absence", "olens_rl"),
    "NLA": ("climb_textfirst_absence", "nla"),
    "J-lens": ("climb_textfirst_absence", "jlens"),
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
    base = data["OLens, improved harness"]
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
    ax.set_xlabel("pass@1 minus OLens, improved harness (pp), paired, 95% CI")
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


def keep_rescue(cfg: Config, dev: set[str]) -> None:
    tr: dict[tuple[str, int], float] = {}
    for f in (cfg.out / "climb_textfirst_absence" / "transcript").glob("*.json"):
        r = json.loads(f.read_text())
        tr[(r["prompt_id"], int(r["seed"]))] = float(r["judge"]["pass"])
    fig, ax = plt.subplots(figsize=(11, 8.5))
    pts: list[tuple[float, float, str, str]] = []
    for label, (ev, arm) in DEV.items():
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
        c = OLENS if label.startswith("OLens") else OTHER
        pts.append((100 * rr / nr, 100 * k / nk, label, c))
    groups: dict[tuple[float, float], list[str]] = defaultdict(list)
    colors: dict[tuple[float, float], str] = {}
    for x, y, label, c in pts:  # identical points share one label
        groups[(round(x, 1), round(y, 1))].append(label)
        colors.setdefault((round(x, 1), round(y, 1)), c)
    placed: list[tuple[float, float]] = []
    for (x, y), labels in groups.items():
        ax.scatter(x, y, s=140, color=colors[(x, y)], edgecolor="white", lw=2, zorder=3)
        near = any(abs(x - px) < 3.0 and abs(y - py) < 0.8 for px, py in placed)
        crowded_right = any(0 < gx - x < 9 and abs(gy - y) < 0.6 for gx, gy in groups)
        ax.annotate(
            " = ".join(labels),
            (x, y),
            xytext=(-7 if crowded_right else 7, -16 if near else 5),
            textcoords="offset points",
            fontsize=12,
            ha="right" if crowded_right else "left",
        )
        placed.append((x, y))
    xs = [x for x, _ in groups]
    ax.set_xlim(min(xs) - 7, max(xs) + 12)  # room for labels on both sides
    ax.set_xlabel("rescue: pass@1 where a transcript draw FAILED (%)")
    ax.set_ylabel("keep: pass@1 where a transcript draw PASSED (%)")
    ax.set_title("Keep vs rescue on dev (up and right is better)", loc="left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.plots / "climb_dev_keep_rescue.png", dpi=160)


def passk(cfg: Config, test: set[str], hard: set[str]) -> None:
    arms = {
        "OLens, improved harness": "olens_rl_B",
        "transcript": "transcript",
        "transcript + continue_from": "transcript_cont",
        "NLA, improved harness": "nla_B",
        "J-lens, improved harness": "jlens_B",
        "OLens, original harness": "olens_rl",
    }
    strata = {"all 204": test, "hard 46": test & hard, "normal 158": test - hard}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=False)
    for ax, (name, sub) in zip(axes, strata.items(), strict=True):
        for label, arm in arms.items():
            d = {p: v for p, v in draws(cfg.out, "climb_test", arm, sub).items() if len(v) >= 5}
            if len(d) < len(sub) * 0.9:  # an arm still running: skip until complete
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
            c = OLENS if arm == "olens_rl_B" else (OTHER if arm == "transcript" else None)
            ax.plot(range(1, 6), ys, "o-", lw=3, ms=8, label=label, color=c)
        ax.set_title(name)
        ax.set_xlabel("k")
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pass@k (%)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Held-out test (204 incidents x 5 draws, read-only)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(cfg.plots / "heldout_passk.png", dpi=160, bbox_inches="tight")


@pydra.main(Config)
def main(cfg: Config) -> None:
    cfg.plots.mkdir(parents=True, exist_ok=True)
    dev, test = set(read_ids(Path(cfg.dev))), set(read_ids(Path(cfg.test)))
    hard = set(read_ids(Path(cfg.hard)))
    forest(cfg, dev)
    keep_rescue(cfg, dev)
    passk(cfg, test, hard)
    print(f"wrote {cfg.plots}/climb_dev_forest.png, climb_dev_keep_rescue.png, heldout_passk.png")


if __name__ == "__main__":
    main()
