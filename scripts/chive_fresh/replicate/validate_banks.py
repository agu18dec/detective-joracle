"""Load every bank record through the harness load_reader (token-id check) for the 204 ids.
  python validate_banks.py <ids> <dataset> BANK_DIR:KIND ..."""
import sys
from pathlib import Path
from detective_joracle.chive.data import load_incidents, read_ids
from detective_joracle.chive.readers import load_reader
ids = read_ids(Path(sys.argv[1])); incs = load_incidents(ids, sys.argv[2])
for spec in sys.argv[3:]:
    bank, kind = spec.rsplit(":", 1); ok = bad = 0; lay = set(); ks = set(); cover = []
    for p in ids:
        try:
            r = load_reader("x", kind, Path(bank), incs[p]); ok += 1
            lay.add(tuple(r.layers)); ks.add(r.k); cover.append(len(r.cells) / max(1, r.n_tokens))
        except Exception as e:
            bad += 1; print("  FAIL", p, type(e).__name__, str(e)[:150])
    print(f"{bank}: ok={ok} fail={bad} layers={[f'{l[0]}..{l[-1]}({len(l)})' for l in lay]} k={ks} "
          f"pos coverage min={min(cover):.3f} mean={sum(cover)/len(cover):.3f}", flush=True)
