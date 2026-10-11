"""The pre-registered comparisons (docs/chive_preregistration.md) for one eval.

Paired by incident (pass@1 averaged over draws), 95% bootstrap CIs. Strata come from
transcript-only uncertainty (confidence.json, top_p) and never from correctness; the hard-92
stratum (selected on transcript errors) is shown alongside, flagged. Writes
<out>/<eval>/strata.md.

    python scripts/strata_chive.py eval=e1_readonly out=/workspace/agam/chive_runs_dj \\
        "pairs=['olens_rl-olens_rl_posshuffled','olens_rl-olens_rl_shuffled','olens_rl-transcript']"
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import pydra

from detective_joracle.chive.data import read_ids


def load(out: Path, ev: str) -> dict[str, dict[str, list[float]]]:
    """arm -> prompt_id -> pass per draw (judged records only)."""
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted((out / ev).glob("*/*.d*.json")):
        r = json.loads(f.read_text())
        j = r.get("judge") or {}
        if j.get("score") is not None:
            by[r["arm"]][r["prompt_id"]].append(1.0 if j["pass"] else 0.0)
    return by


def mean_of(by: dict[str, dict[str, list[float]]], arm: str, sub: set[str]) -> tuple[float, int]:
    xs = [sum(v) / len(v) for p, v in by[arm].items() if p in sub]
    return (sum(xs) / len(xs) if xs else float("nan")), len(xs)


def paired(by: dict[str, dict[str, list[float]]], a: str, b: str, sub: set[str]) -> str:
    """a minus b, matched by incident, with a 95% bootstrap CI ('*' = CI excludes 0)."""
    ps = [p for p in sub if p in by[a] and p in by[b]]
    if not ps:
        return "-"
    d = [sum(by[a][p]) / len(by[a][p]) - sum(by[b][p]) / len(by[b][p]) for p in ps]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
    lo, hi = 100 * bs[100], 100 * bs[3899]
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{100 * sum(d) / len(d):+.1f} [{lo:+.1f}, {hi:+.1f}]{star} n={len(ps)}"


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.eval = "e1_readonly"
        self.out = pydra.REQUIRED
        self.ids = "data/chive/splits/all300.txt"
        self.hard = "data/chive/splits/hard92.txt"
        self.confidence = ""  # default: <out>/confidence.json
        self.pairs: list[str] = ["olens_rl-transcript"]  # "A-B": A minus B

    def finalize(self) -> None:
        self.out = Path(self.out)
        self.confidence = Path(self.confidence) if self.confidence else self.out / "confidence.json"


@pydra.main(Config)
def main(cfg: Config) -> None:
    ids = set(read_ids(Path(cfg.ids)))
    hard = set(read_ids(Path(cfg.hard)))
    conf = {p: v["top_p"] for p, v in json.loads(cfg.confidence.read_text()).items() if p in ids}
    ranked = sorted(conf, key=lambda p: conf[p])
    n = len(ranked)
    strata = {
        "all": ids,
        "top_p in [0.25, 0.75] (primary)": {p for p in conf if 0.25 <= conf[p] <= 0.75},
        "tercile 1 (least confident)": set(ranked[: n // 3]),
        "tercile 2": set(ranked[n // 3 : 2 * n // 3]),
        "tercile 3 (most confident)": set(ranked[2 * n // 3 :]),
        "hard 92 (selected on errors: flagged)": hard & ids,
    }
    by = load(cfg.out, cfg.eval)
    arms = sorted(by)
    lines = [
        f"# Pre-registered comparisons: `{cfg.eval}`",
        "",
        "See docs/chive_preregistration.md. pass@1 averaged over draws, then incidents; "
        "differences paired by incident with 95% bootstrap CIs (* = excludes 0). Strata from "
        "transcript-only uncertainty (top_p), never from correctness, except the flagged hard 92.",
        "",
        "## pass@1 by stratum",
        "",
        "| stratum | n | " + " | ".join(arms) + " |",
        "|---|---|" + "---|" * len(arms),
    ]
    for name, sub in strata.items():
        cells = [f"{mean_of(by, a, sub)[0]:.1%}" for a in arms]
        lines.append(f"| {name} | {len(sub)} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences", ""]
    lines += ["| stratum | " + " | ".join(cfg.pairs) + " |", "|---|" + "---|" * len(cfg.pairs)]
    for name, sub in strata.items():
        cells = [paired(by, *pair.split("-", 1), sub) for pair in cfg.pairs]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if conf:
        tp = sorted(conf.values())
        lines += [
            "",
            f"top_p over {len(tp)} incidents: quartiles {tp[len(tp) // 4]:.2f} / "
            f"{tp[len(tp) // 2]:.2f} / {tp[3 * len(tp) // 4]:.2f}; tercile cuts "
            f"{conf[ranked[n // 3]]:.2f}, {conf[ranked[2 * n // 3]]:.2f}.",
        ]
    text = "\n".join(lines)
    (cfg.out / cfg.eval / "strata.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
