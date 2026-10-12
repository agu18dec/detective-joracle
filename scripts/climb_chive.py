"""Harness-climb A/B table: every climb_* arm vs the base arm and vs transcript + continue_from.

Reads every judged record under <out>/climb_*/<arm>/ for the dev incidents (an arm name is
unique across rounds). Per arm: pass@1 (2 draws), paired difference vs `base` and vs
`reference` with 95% bootstrap CIs, keep / rescue (split on one transcript draw, scored on the
other), hard / normal pass@1, tool calls per episode, and unsupported quotes (quoted text in
the answer that neither the conversation nor the arm's readout banks back -- the hallucination
proxy of analyze_chive.py). Writes <out>/climb_ab.md.

    python scripts/climb_chive.py out=runs/chive base=olens_rl_tfa bank=<OLens readout bank>
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pydra

sys.path.insert(0, str(Path(__file__).parent))
from analyze_chive import bank_index, bank_text, quote_check  # noqa: E402
from detective_joracle.chive.data import DATASET, load_incidents, read_ids  # noqa: E402


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = DATASET  # HF dataset id or a local directory (see run_chive.py)
        self.out = pydra.REQUIRED
        self.ids = "data/chive/splits/climb_dev96.txt"
        self.hard = "data/chive/splits/hard92.txt"
        self.base = "olens_rl_tfa"
        self.reference = "transcript_cont"
        self.bank = pydra.REQUIRED  # OLens readout bank, for the quote check

    def finalize(self) -> None:
        self.out, self.bank = Path(self.out), Path(self.bank)


def paired(a: dict[str, list[float]], b: dict[str, list[float]]) -> str:
    ps = [p for p in a if p in b]
    if not ps:
        return "-"
    d = [sum(a[p]) / len(a[p]) - sum(b[p]) / len(b[p]) for p in ps]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
    lo, hi = 100 * bs[100], 100 * bs[3899]
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{100 * sum(d) / len(d):+.1f} [{lo:+.1f}, {hi:+.1f}]{star}"


@pydra.main(Config)
def main(cfg: Config) -> None:
    ids = set(read_ids(Path(cfg.ids)))
    hard = set(read_ids(Path(cfg.hard)))
    incs = load_incidents(sorted(ids), cfg.dataset)
    recs: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    seen: set[Path] = set()
    for f in sorted(cfg.out.glob("climb_*/*/*.d*.json")):
        real = f.resolve()
        if real in seen:  # baselines are linked into every round
            continue
        seen.add(real)
        r = json.loads(f.read_text())
        if r["prompt_id"] in ids and (r.get("judge") or {}).get("score") is not None:
            recs[r["arm"]][r["prompt_id"]].append(r)
    passes = {
        a: {p: [float(r["judge"]["pass"]) for r in rs] for p, rs in by.items()}
        for a, by in recs.items()
    }
    tdraw = {
        (p, int(r["seed"])): r["judge"]["pass"] for p, rs in recs["transcript"].items() for r in rs
    }
    idx_cache: dict[str, Any] = {}
    convo = {p: "\n".join(m["content"] for m in i.messages) + i.completion for p, i in incs.items()}

    def keep_rescue(arm: str) -> tuple[float, float]:
        k = r_ = nk = nr = 0.0
        for p, rs in recs[arm].items():
            for r in rs:
                t = tdraw.get((p, 1 - int(r["seed"])))
                if t is None:
                    continue
                if t:
                    k, nk = k + r["judge"]["pass"], nk + 1
                else:
                    r_, nr = r_ + r["judge"]["pass"], nr + 1
        return k / max(1, nk), r_ / max(1, nr)

    lines = [
        "# Harness climb: A/B on climb-dev (96 incidents x 2 draws, read-only)",
        "",
        f"Each arm vs the base `{cfg.base}` and vs the reference `{cfg.reference}`, paired by "
        "incident (95% bootstrap CI, * = excludes 0). keep/rescue: split on one transcript draw, "
        "scored on the other. unsupported = quoted text backed by neither the conversation nor "
        "the OLens bank (support < 0.5; the hallucination proxy).",
        "",
        "| arm | pass@1 | vs base | vs reference | keep | rescue | hard | normal | calls/ep | "
        "unsupported quotes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in sorted(recs, key=lambda a: -sum(map(sum, passes[a].values()))):
        pa = passes[arm]
        m = sum(sum(v) / len(v) for v in pa.values()) / len(pa)
        h = [sum(v) / len(v) for p, v in pa.items() if p in hard]
        nh = [sum(v) / len(v) for p, v in pa.items() if p not in hard]
        kk, rr = keep_rescue(arm)
        eps = [r for rs in recs[arm].values() for r in rs]
        calls = sum(len(r.get("events", [])) for r in eps) / len(eps)
        q: Counter[str] = Counter()
        if arm.startswith("olens"):
            for p, rs in recs[arm].items():
                if p not in idx_cache:
                    idx_cache[p] = bank_index(bank_text(cfg.bank, p))
                for r in rs:
                    q.update(quote_check(r, convo[p], idx_cache[p], tokens=False))
        tot = sum(q.values())
        unsup = (
            f"{q['quotes_unsupported']}/{tot} ({q['quotes_unsupported'] / tot:.1%})" if tot else "-"
        )
        vs_base = paired(pa, passes[cfg.base]) if arm != cfg.base and cfg.base in passes else "-"
        vs_ref = (
            paired(pa, passes[cfg.reference])
            if arm != cfg.reference and cfg.reference in passes
            else "-"
        )
        lines.append(
            f"| {arm} | {m:.1%} | {vs_base} | {vs_ref} | {kk:.1%} | {rr:.1%} | "
            f"{sum(h) / max(1, len(h)):.1%} | {sum(nh) / max(1, len(nh)):.1%} | {calls:.1f} | "
            f"{unsup} |"
        )
    tool_lines = ["", "## Tool calls per episode", ""]
    for arm in sorted(recs):
        eps = [r for rs in recs[arm].values() for r in rs]
        c = Counter(e["tool"] for r in eps for e in r.get("events", []))
        tool_lines.append(
            f"- {arm}: " + ", ".join(f"{k} {v / len(eps):.1f}" for k, v in c.most_common())
        )
    text = "\n".join(lines + tool_lines)
    (cfg.out / "climb_ab.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
