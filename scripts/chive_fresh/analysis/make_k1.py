"""k=1 readout bank from a k=4 bank: keep the first sample of every cell.

Same instrument, a quarter of the readout tokens: the ``olens_k1`` and ``mlnla_k1`` arms
(docs/chive_fresh_scaleup.md) test whether OLens' edge comes from reading 4 samples per cell.
Files already in DST are skipped (resumable); every output records its source under
``provenance.k1_from``.

    python make_k1.py SRC_BANK_DIR DST_BANK_DIR [--workers 16]
    # e.g. make_k1.py $CHIVE_BANKS/s3d_rl600_all $CHIVE_BANKS/olens_k1
"""

import argparse
import concurrent.futures as cf
import gzip
import json
from pathlib import Path
from typing import Any


def k1_record(rec: dict[str, Any], source: str) -> dict[str, Any]:
    """The record with only the first sample of every (layer, position) cell, and ``k`` = 1."""
    rec["readouts"] = {
        layer: {pos: samples[:1] for pos, samples in cells.items()}
        for layer, cells in rec["readouts"].items()
    }
    rec["k"] = 1
    rec.setdefault("provenance", {})["k1_from"] = source
    return rec


def convert_one(f: Path, dst: Path) -> int:
    out = dst / f.name
    if out.exists():
        return 0
    with gzip.open(f, "rt") as fh:
        rec = json.load(fh)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(k1_record(rec, str(f)), fh)
    tmp.rename(out)
    return 1


def convert(src: Path, dst: Path, workers: int = 16) -> tuple[int, int]:
    """(new files written, source files) for every ``*.json.gz`` in ``src``."""
    dst.mkdir(parents=True, exist_ok=True)
    fs = sorted(src.glob("*.json.gz"))
    with cf.ProcessPoolExecutor(workers) as ex:
        n = sum(ex.map(convert_one, fs, [dst] * len(fs)))
    return n, len(fs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    n, total = convert(a.src, a.dst, a.workers)
    print(f"{n} new k=1 files -> {a.dst} ({total} source)")


if __name__ == "__main__":
    main()
