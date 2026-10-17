"""Recount readouts from the raw harness banks (not agent summaries) at given positions.

    python hverify.py <batch> <pid> <pos[,pos..]> <regex> [--bank s3d_rl600_all] [--show N] [--others]

Per layer: matching lines / all bullet lines (4 samples x ~4 bullets per cell), plus the
visible-text check (does the regex already match the conversation itself?) and, with --others,
NLA (L42) / multi-layer NLA lines and J-lens top tokens at the same position. Banks are read from
$CHIVE_BANKS/<bank>/<pid>.json.gz, tokens from $CHIVE_DATASETS/<batch>/data/tokens.jsonl.
Rule of thumb from the case work: always recount from the bank before quoting a readout.
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402


def load_bank(banks: Path, name: str, pid: str) -> dict[str, dict[str, list[str]]]:
    with gzip.open(banks / name / f"{pid}.json.gz", "rt") as f:
        return json.load(f)["readouts"]


def lines(cell: list[Any] | None) -> list[str]:
    """A cell's samples split into non-empty lines (a sample without line breaks stays whole)."""
    out: list[str] = []
    for s in cell or []:
        ls = [x.strip() for x in str(s).split("\n") if x.strip()]
        out += ls or [str(s)]
    return out


def count(
    readouts: dict[str, dict[str, list[str]]], pos: int, rx: re.Pattern[str]
) -> dict[int, tuple[list[str], int]]:
    """layer -> (matching lines, all lines) at ``pos``."""
    out = {}
    for layer in sorted(readouts, key=int):
        ls = lines(readouts[layer].get(str(pos)))
        out[int(layer)] = ([x for x in ls if rx.search(x)], len(ls))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("batch")
    ap.add_argument("pid")
    ap.add_argument("positions")
    ap.add_argument("regex")
    ap.add_argument("--bank", default="s3d_rl600_all")
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--others", action="store_true")
    ap.add_argument("--banks", type=Path, default=paths.BANKS)
    ap.add_argument("--datasets", type=Path, default=paths.DATASETS)
    a = ap.parse_args()
    with (a.datasets / a.batch / "data/tokens.jsonl").open() as fh:
        tok = next(r for r in map(json.loads, fh) if r["prompt_id"] == a.pid)
    pieces = tok["pieces"]
    rx = re.compile(a.regex, re.I)
    ol = load_bank(a.banks, a.bank, a.pid)
    vis = rx.findall("".join(pieces))
    print(f"{a.pid}  regex /{a.regex}/  visible-text hits: {len(vis)} {vis[:3]}")
    for p in [int(x) for x in a.positions.split(",")]:
        ctx = "".join(pieces[max(0, p - 25) : p]).replace("\n", "⏎")
        print(f"\n== t{p} {pieces[p]!r}   context: …{ctx[-90:]}[[{pieces[p]}]]")
        tot = hit = 0
        for layer, (h, n) in count(ol, p, rx).items():
            tot += n
            hit += len(h)
            eg = ("   e.g. " + " | ".join(x[:150] for x in h[: a.show])) if h else ""
            print(f"  L{layer}: {len(h)}/{n}{eg}")
        print(f"  OLens total {hit}/{tot}")
        if a.others:
            for name in ("nla_k4", "mlnla_k4_chive"):
                try:
                    o = load_bank(a.banks, name, a.pid)
                except FileNotFoundError:
                    continue
                for layer, (h, n) in count(o, p, rx).items():
                    print(
                        f"  {name} L{layer}: {len(h)}/{n}" + ("   e.g. " + h[0][:150] if h else "")
                    )
            j = load_bank(a.banks, "jlens_all", a.pid)
            print(
                "  jlens L40-60 top tokens:",
                " | ".join(
                    f"L{L}:{' '.join(map(str, j[L].get(str(p), [])[:6]))}"
                    for L in sorted(j, key=int)
                    if int(L) in (40, 48, 56, 60)
                ),
            )


if __name__ == "__main__":
    main()
