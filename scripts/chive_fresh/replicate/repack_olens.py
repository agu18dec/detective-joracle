"""Repack the viewer's per-layer OLens copies (r/<pid>/olens_bank_L{L}.json: list per position of
k=4 samples) into a harness-format bank record <out>/<pid>.json.gz, token ids from the HF dataset.

  python3 repack_olens.py <mirror_r_dir> <tokens.jsonl> <ids.txt> <out_dir> [prefix=olens_bank]
"""
import gzip, json, sys
from pathlib import Path

mirror, tokens_f, ids_f, out = map(Path, sys.argv[1:5])
prefix = sys.argv[5] if len(sys.argv) > 5 else "olens_bank"
LAYERS = list(range(20, 61, 4))
ids = ids_f.read_text().split()
tok = {}
for line in tokens_f.open():
    d = json.loads(line)
    if d["prompt_id"] in ids:
        tok[d["prompt_id"]] = [int(t) for t in d["token_ids"]]
out.mkdir(parents=True, exist_ok=True)
bad = 0
for pid in ids:
    ro = {}
    for L in LAYERS:
        cells = json.loads((mirror / pid / f"{prefix}_L{L}.json").read_text())
        if len(cells) != len(tok[pid]):
            print(f"{pid} L{L}: {len(cells)} cells vs {len(tok[pid])} tokens"); bad += 1
        ro[str(L)] = {str(p): [str(s) for s in c] for p, c in enumerate(cells)}
    ks = {len(c) for per in ro.values() for c in per.values()}
    rec = {"prompt_id": pid, "reader": "s3d_rl600_all", "n_tokens": len(tok[pid]), "layers": LAYERS,
           "k": 4, "coverage_note": "every token", "token_ids": tok[pid], "readouts": ro,
           "provenance": {"source": f"repacked from viewer copies chive-cases-data:r/{pid}/{prefix}_L*.json(.gz) "
                          "(local mirror gw-olens-uplift/outputs/chive_cases_mirror)",
                          "original": "OLens RL s3d_rl600 bank s3d_rl600_all, k=4, every position, L20-60 step 4",
                          "k_per_cell_seen": sorted(ks)}}
    with gzip.open(out / f"{pid}.json.gz", "wt") as f:
        json.dump(rec, f)
print(f"wrote {len(ids)} records to {out}; {bad} length mismatches")
