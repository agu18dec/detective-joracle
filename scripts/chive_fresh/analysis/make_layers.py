"""Layer-restricted readout bank: keep only the given layers.

Used for the ``jlens_L2060`` arm (docs/chive_fresh_scaleup.md): J-lens restricted to the OLens /
NLA ladder L20-60 step 4, so J-lens is compared at the same layers instead of all 63. Files already
in DST are skipped (resumable); every output records its source under ``provenance.layers_from``.

    python make_layers.py SRC_BANK_DIR DST_BANK_DIR [--layers 20:61:4] [--workers 16]
    # e.g. make_layers.py $CHIVE_BANKS/jlens_all $CHIVE_BANKS/jlens_L20-60
"""

import argparse
import concurrent.futures as cf
import gzip
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

DEFAULT_LAYERS = "20:61:4"


def parse_layers(spec: str) -> set[str]:
    """``start:stop:step`` (Python range) or a comma list -> the layer keys (strings) to keep."""
    if ":" in spec:
        start, stop, step = (int(x) for x in spec.split(":"))
        return {str(layer) for layer in range(start, stop, step)}
    return {str(int(x)) for x in spec.split(",") if x.strip()}


def layers_record(rec: dict[str, Any], keep: Collection[str], source: str) -> dict[str, Any]:
    """The record with only the ``keep`` layers in ``readouts``."""
    rec["readouts"] = {layer: v for layer, v in rec["readouts"].items() if layer in keep}
    rec.setdefault("provenance", {})["layers_from"] = source
    return rec


def convert_one(f: Path, dst: Path, keep: Collection[str]) -> int:
    out = dst / f.name
    if out.exists():
        return 0
    with gzip.open(f, "rt") as fh:
        rec = json.load(fh)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(layers_record(rec, keep, str(f)), fh)
    tmp.rename(out)
    return 1


def convert(src: Path, dst: Path, keep: Collection[str], workers: int = 16) -> tuple[int, int]:
    """(new files written, source files) for every ``*.json.gz`` in ``src``."""
    dst.mkdir(parents=True, exist_ok=True)
    fs = sorted(src.glob("*.json.gz"))
    with cf.ProcessPoolExecutor(workers) as ex:
        n = sum(ex.map(convert_one, fs, [dst] * len(fs), [set(keep)] * len(fs)))
    return n, len(fs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    n, _ = convert(a.src, a.dst, parse_layers(a.layers), a.workers)
    print(f"{n} files -> {a.dst}")


if __name__ == "__main__":
    main()
