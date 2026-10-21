"""Shared paths/helpers for the chive-diag analysis (Phase 1-4)."""
import csv, json, pathlib, sys, collections
C = pathlib.Path("/workspace/agam/chive_scale")
EXP = pathlib.Path("/workspace/agam/experiments/chive-diag")
OUT = EXP / "data"; OUT.mkdir(parents=True, exist_ok=True)
DIAG = C / "diag"  # LLM outputs (cached per item)
BATCHES = ["fresh1", "fresh2", "fresh3", "fresh4"]

def write_rows(name, rows):
    if not rows: return
    with open(OUT / name, "w", newline="") as f:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

def jrows(f):
    return {r["prompt_id"]: r for r in map(json.loads, open(f))}

_GT, _INC = {}, {}
def gt(b):
    if b not in _GT: _GT[b] = jrows(C / "datasets" / b / "data" / "ground_truth.jsonl")
    return _GT[b]
def inc(b):
    if b not in _INC: _INC[b] = jrows(C / "datasets" / b / "data" / "incidents.jsonl")
    return _INC[b]

def record(b, arm, pid, d):
    f = C / "harness_runs" / b / arm / f"{pid}.d{d}.json"
    return json.loads(f.read_text()) if f.exists() else None
