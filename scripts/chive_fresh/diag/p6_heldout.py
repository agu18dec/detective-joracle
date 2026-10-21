"""Held-out test (analysis/diag/heldout/PREREG.md): 110+110 incidents from fresh3/4, draws 0-2. Paired incident bootstrap."""
import sys, json
sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
import p5_dev_compare as P
from dcommon import *
from common import boot
P.DEV = {b: open(C / "analysis/diag/heldout" / f"{b}.txt").read().split() for b in ("fresh3", "fresh4")}
G = range(3)
specs = [  # (arm, eval, ref, ref eval, draws, label)
    ("olens_s16_gate", "{b}", "mlnla_gate_h", "{b}", G, "PRIMARY A: OLens x16 + gate vs NLA + gate (budget 0)"),
    ("olens_s16_gx3", "{b}", "mlnla_gx3", "{b}", G, "PRIMARY B: OLens x16 + gate vs NLA + gate (budget 3)"),
    ("olens_s16_gate", "{b}", "olens_single_k16", "{b}", G, "gate vs frozen, OLens x16"),
    ("mlnla_gate_h", "{b}", "mlnla_B", "{b}", G, "gate vs frozen, NLA x4"),
    ("olens_s16_gate", "{b}", "mlnla_B", "{b}", G, "OLens x16 + gate vs frozen NLA x4"),
    ("olens_rl_B", "{b}", "mlnla_B", "{b}", G, "frozen OLens x4 vs frozen NLA x4 (reference)"),
    ("olens_single_k16", "{b}", "mlnla_B", "{b}", G, "frozen OLens x16 vs frozen NLA x4 (reference)"),
    ("olens_s16_gx3", "{b}", "transcript_rx3h", "{b}", G, "budget 3: OLens x16 + gate vs transcript"),
    ("mlnla_gx3", "{b}", "transcript_rx3h", "{b}", G, "budget 3: NLA + gate vs transcript"),
    ("transcript_rx3h", "{b}", "transcript", "{b}", G, "transcript: 3 experiments vs none"),
    ("olens_s16_gx3", "{b}", "olens_s16_gate", "{b}", G, "OLens x16 + gate: 3 experiments vs none"),
    ("mlnla_gx3", "{b}", "mlnla_gate_h", "{b}", G, "NLA + gate: 3 experiments vs none"),
    ("transcript_rx3h", "{b}", "transcript", "{b}_exp1n30", range(2), "transcript: 3 experiments vs 1"),
    ("olens_s16_gx3", "{b}", "olens_rl_B", "{b}_exp1n30", range(2), "OLens x16+gate b3 vs OLens x4 b1"),
    ("mlnla_gx3", "{b}", "mlnla_B", "{b}_exp1n30", range(2), "NLA+gate b3 vs NLA x4 b1"),
]
rows = []
for a, ea, r, er, dr, lab in specs:
    A, B = P.passes(ea, a, dr), P.passes(er, r, dr)
    ks, va, vb = P.paired(A, B)
    if not ks: print("no data", lab); continue
    m1, m2, d, lo, hi = boot(va, vb)
    rows.append(dict(label=lab, arm=a, ref=r, ref_eval=er, n=len(ks), arm_pass=m1, ref_pass=m2, diff=d, lo=lo, hi=hi))
    print(f"{lab:55s} n={len(ks):3d} {m1:.3f} vs {m2:.3f} d={d:+.3f} [{lo:+.3f},{hi:+.3f}]")
write_rows("p6_heldout.csv", rows)
