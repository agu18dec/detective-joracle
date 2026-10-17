"""Compare a freshly built harness bank record against an existing one (validation).

  python3 compare_banks.py NEW.json.gz REF.json.gz [--show 3]
  python3 compare_banks.py NEW.json.gz MIRROR_DIR --olens-mirror   # chive_cases_mirror/.../r/<pid>/

J-lens (k=1, token lists): exact cell match rate and mean top-10 set overlap per layer.
Sampled readers: format, coverage, k, length stats, a few side-by-side cells, and a crude
content-similarity score (word-set Jaccard between new and ref samples at the same cell vs
at a shifted cell, i.e. is a cell more like its own reference than like another position's).
"""

import argparse
import gzip
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def load_mirror(d: Path) -> dict[str, Any]:
    readouts = {}
    for f in sorted(d.glob("olens_bank_L*.json")):
        layer = re.search(r"_L(\d+)\.json", f.name).group(1)
        cells = json.loads(f.read_text())
        readouts[layer] = {str(p): c for p, c in enumerate(cells)}
    return {"readouts": readouts, "token_ids": None, "k": 4}


def words(texts: list[str]) -> set[str]:
    return {w for t in texts for w in re.findall(r"[A-Za-z]{4,}", t.lower())}


def jacc(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--olens-mirror", action="store_true")
    ap.add_argument("--show", type=int, default=2)
    a = ap.parse_args()
    new = load(a.new)
    ref = load_mirror(a.ref) if a.olens_mirror else load(a.ref)
    print(f"new keys: {sorted(new)}")
    if ref.get("token_ids") is not None:
        print(
            f"token_ids equal: {new['token_ids'] == ref['token_ids']} (n={len(new['token_ids'])})"
        )
    nl, rl = sorted(new["readouts"], key=int), sorted(ref["readouts"], key=int)
    print(f"layers new={nl[0]}..{nl[-1]} ({len(nl)})  ref={rl[0]}..{rl[-1]} ({len(rl)})")
    print(f"k new={new.get('k')} ref={ref.get('k')}")
    common = [layer for layer in nl if layer in ref["readouts"]]
    jl = new.get("k", 1) == 1 and all(
        isinstance(c, list) and len(c) == 1 and c[0].startswith("'") or c[0].startswith('"')
        for c in list(new["readouts"][nl[0]].values())[:5]
    )
    rng = random.Random(0)
    for layer in common:
        n, r = new["readouts"][layer], ref["readouts"][layer]
        pos = sorted(set(n) & set(r), key=int)
        cov = f"pos new={len(n)} ref={len(r)} common={len(pos)}"
        if jl:
            exact = sum(n[p] == r[p] for p in pos) / max(1, len(pos))
            tok = lambda c: set(re.findall(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", c[0]))  # noqa: E731
            ov = statistics.mean(len(tok(n[p]) & tok(r[p])) / 10 for p in pos)
            print(f"L{layer}: {cov} exact-cell={exact:.4f} mean-top10-overlap={ov:.4f}")
            continue
        ks = {len(n[p]) for p in pos}
        ln = statistics.mean(len(s) for p in pos for s in n[p])
        lr = statistics.mean(len(s) for p in pos for s in r[p])
        empty = sum(1 for p in pos for s in n[p] if not s.strip())
        same = statistics.mean(jacc(words(n[p]), words(r[p])) for p in pos)
        sh = [
            jacc(words(n[p]), words(r[pos[(i + len(pos) // 2) % len(pos)]]))
            for i, p in enumerate(pos)
        ]
        print(
            f"L{layer}: {cov} k_new={ks} mean_chars new={ln:.0f} ref={lr:.0f} empty={empty} "
            f"| jaccard same-cell={same:.3f} vs shifted-cell={statistics.mean(sh):.3f}"
        )
    if not jl:
        for layer in common[:: max(1, len(common) // 2)][:2]:
            pos = sorted(set(new["readouts"][layer]) & set(ref["readouts"][layer]), key=int)
            for p in rng.sample(pos, min(a.show, len(pos))):
                print(f"\n=== L{layer} pos {p}")
                print("  NEW:", new["readouts"][layer][p][0][:300].replace("\n", " | "))
                print("  REF:", ref["readouts"][layer][p][0][:300].replace("\n", " | "))


if __name__ == "__main__":
    main()
