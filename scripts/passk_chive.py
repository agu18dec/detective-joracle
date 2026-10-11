"""pass@k (k = 1..n) per arm for one eval, with paired comparisons against a reference arm.

Per incident with n judged draws of which c pass, the unbiased estimator
pass@k = 1 - C(n - c, k) / C(n, k); averaged over incidents. Differences are paired by incident
with 95% bootstrap CIs. Strata: all, hard, normal, and all minus an exclusion list (the leakage
check: incidents read in depth while designing the harness). Writes <out>/<eval>/passk.md.

    python scripts/passk_chive.py eval=climb_test out=/workspace/agam/chive_runs_dj \\
        ids=data/chive/splits/climb_test204.txt reference=olens_rl_B
"""

import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

import pydra

from detective_joracle.chive.data import read_ids

LABEL = {
    "olens_rl_B": "OLens, improved harness",
    "nla_B": "NLA, improved harness",
    "jlens_B": "J-lens, improved harness",
    "olens_rl": "OLens, original harness",
    "nla": "NLA, original harness",
    "jlens": "J-lens, original harness",
    "transcript": "transcript only",
    "transcript_cont": "transcript + continue_from",
}


def pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.eval = "climb_test"
        self.out = pydra.REQUIRED
        self.ids = "data/chive/splits/climb_test204.txt"
        self.hard = "data/chive/splits/hard92.txt"
        self.exclude = ""  # optional file of incident ids for the leakage check
        self.reference = "olens_rl_B"
        self.draws = 5

    def finalize(self) -> None:
        self.out = Path(self.out)


@pydra.main(Config)
def main(cfg: Config) -> None:
    ids = set(read_ids(Path(cfg.ids)))
    hard = set(read_ids(Path(cfg.hard)))
    excl = set(read_ids(Path(cfg.exclude))) if cfg.exclude else set()
    by: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted((cfg.out / cfg.eval).glob("*/*.d*.json")):
        r = json.loads(f.read_text())
        j = r.get("judge") or {}
        if r["prompt_id"] in ids and j.get("score") is not None:
            by[r["arm"]][r["prompt_id"]].append(bool(j["pass"]))
    # only incidents with all draws judged, so every arm's estimator uses the same n
    full = {a: {p: v for p, v in d.items() if len(v) >= cfg.draws} for a, d in by.items()}
    ks = list(range(1, cfg.draws + 1))

    def per_incident(arm: str, k: int) -> dict[str, float]:
        return {p: pass_at_k(len(v), sum(v), k) for p, v in full[arm].items()}

    def paired(a: str, b: str, k: int, sub: set[str]) -> str:
        pa, pb = per_incident(a, k), per_incident(b, k)
        ps = [p for p in pa if p in pb and p in sub]
        if not ps:
            return "-"
        d = [pa[p] - pb[p] for p in ps]
        rng = random.Random(0)
        bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
        lo, hi = 100 * bs[100], 100 * bs[3899]
        return (
            f"{100 * sum(d) / len(d):+.1f} [{lo:+.1f}, {hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}"
        )

    strata = {
        "all": ids,
        "hard": ids & hard,
        "normal": ids - hard,
        **({"all minus deep-read": ids - excl} if excl else {}),
    }
    arms = sorted(full, key=lambda a: -sum(per_incident(a, 1).values()))
    lines = [f"# pass@k: `{cfg.eval}` ({cfg.draws} draws; incidents with every draw judged)", ""]
    for name, sub in strata.items():
        lines += [
            f"## {name}",
            "",
            "| arm | n | "
            + " | ".join(f"pass@{k}" for k in ks)
            + f" | pass@1 vs {LABEL.get(cfg.reference, cfg.reference)} "
            f"| pass@{cfg.draws} vs {LABEL.get(cfg.reference, cfg.reference)} |",
            "|---|---|" + "---|" * (len(ks) + 2),
        ]
        for arm in arms:
            vals = []
            for k in ks:
                xs = [x for p, x in per_incident(arm, k).items() if p in sub]
                vals.append(f"{sum(xs) / len(xs):.1%}" if xs else "-")
            n = sum(1 for p in full[arm] if p in sub)
            ref = arm == cfg.reference or cfg.reference not in full
            c1 = "-" if ref else paired(arm, cfg.reference, 1, sub)
            ck = "-" if ref else paired(arm, cfg.reference, cfg.draws, sub)
            lines.append(
                f"| {LABEL.get(arm, arm)} | {n} | " + " | ".join(vals) + f" | {c1} | {ck} |"
            )
        lines.append("")
    text = "\n".join(lines)
    (cfg.out / cfg.eval / "passk.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
