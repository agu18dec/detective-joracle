"""Phase 2 report: bank ceiling vs what agents surface.

2a ceiling (diag/ceiling/<b>/<bank>/<pid>.json): per bank, share of TRUE points (split obvious / non-obvious) and of refuted
   rivals that the bank states (>= 1 hit line; also >= 3 lines = recurring), paired by incident across banks.
   Shuffled banks (fresh1) give the false-hit floor.
2b ceiling minus agent: for each (incident, non-obvious point) the bank states, per episode of the matching arm (d0-d4):
   asserted = judge v3 final labels contain TRUE(Pi);
   served   = any hit line (first 60 chars) appears in the episode's tool outputs;
   -> asserted | served-not-asserted (interpretation gap) | not-served (search gap).
   Also for rivals the bank states: asserted-rival rate when the lure was served vs not.
Out: experiments/chive-diag/data/p2_ceiling.csv, p2_gap.csv
"""
import json, sys, collections, numpy as np
sys.path.insert(0, "/workspace/agam/chive_scale/report"); sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from common import boot
from dcommon import *

BANK_ARM = {"olens": "olens_rl_B", "mlnla": "mlnla_B", "jlens": None}

def load_ceiling():
    out = collections.defaultdict(dict)  # (b,pid) -> bank -> items
    for b in BATCHES:
        base = C / "diag" / "ceiling" / b
        if not base.exists(): continue
        for bd in base.iterdir():
            if bd.name.startswith("_"): continue
            for f in bd.glob("*.json"):
                try: r = json.loads(f.read_text())
                except Exception: continue
                if any(it.get("hits") is None for it in r["items"]): continue
                out[(b, f.stem)][bd.name] = r["items"]
    return out

def ceiling_table(CE):
    rows = []
    for bank in ["olens", "mlnla", "jlens", "olens_shuf", "mlnla_shuf"]:
        for kind in ["true_nonobv", "true_obv", "rival"]:
            for thr in (1, 3):
                vals = []
                for k, v in CE.items():
                    if bank not in v: continue
                    its = [it for it in v[bank] if (kind == "rival" and it["kind"] == "rival") or
                           (kind == "true_nonobv" and it["kind"] == "true" and it["obvious"] is False) or
                           (kind == "true_obv" and it["kind"] == "true" and it["obvious"] is True)]
                    if its: vals.append(np.mean([len(it["hits"]) >= thr for it in its]))
                if vals:
                    rows.append(dict(bank=bank, kind=kind, min_lines=thr, n_incidents=len(vals), share=float(np.mean(vals))))
    # paired OLens - NLA on the same incidents
    for kind in ["true_nonobv", "true_obv", "rival"]:
        a, r = [], []
        for k, v in CE.items():
            if "olens" in v and "mlnla" in v:
                f = lambda its: [len(it["hits"]) >= 1 for it in its if (kind == "rival" and it["kind"] == "rival") or
                                 (kind == "true_nonobv" and it["kind"] == "true" and it["obvious"] is False) or
                                 (kind == "true_obv" and it["kind"] == "true" and it["obvious"] is True)]
                x, y = f(v["olens"]), f(v["mlnla"])
                if x: a.append(np.mean(x)); r.append(np.mean(y))
        m1, m2, d, lo, hi = boot(a, r)
        rows.append(dict(bank="olens-mlnla", kind=kind, min_lines=1, n_incidents=len(a), share=d, lo=lo, hi=hi))
    return rows

def gap_table(CE):
    rows = []; agg = collections.defaultdict(collections.Counter)
    for (b, pid), v in CE.items():
        for bank, arm in BANK_ARM.items():
            if not arm or bank not in v: continue
            for d in range(5):
                cf = C / "diag" / "judge3" / b / "cls_final" / arm / f"{pid}.d{d}.json"
                rf = C / "harness_runs" / b / arm / f"{pid}.d{d}.json"
                if not cf.exists() or not rf.exists(): continue
                j = json.loads(cf.read_text()); rec = json.loads(rf.read_text())
                served_txt = "\n".join(str(t.get("output", "")) for t in rec.get("tool_log", []))
                asserted = {str(x.get("match")) for x in j.get("labels", []) if str(x.get("label", "")).upper() == "TRUE"}
                riv = {str(x.get("match")) for x in j.get("labels", []) if str(x.get("label", "")).upper() == "RIVAL"}
                for it in v[bank]:
                    if not it["hits"]: continue
                    served = any(h["line"][:60] in served_txt for h in it["hits"])
                    if it["kind"] == "true" and it["obvious"] is False:
                        c = agg[(bank, "nonobv")]
                        c["n"] += 1
                        if it["id"] in asserted: c["asserted"] += 1; c["asserted_served"] += served
                        elif served: c["served_not_asserted"] += 1
                        else: c["not_served"] += 1
                    elif it["kind"] == "rival":
                        c = agg[(bank, "rival")]
                        c["n"] += 1; c["served"] += served
                        c["asserted_if_served"] += served and it["id"] in riv
                        c["asserted_if_not_served"] += (not served) and it["id"] in riv
    for (bank, kind), c in agg.items():
        n = c["n"] or 1
        if kind == "nonobv":
            rows.append(dict(bank=bank, kind=kind, n=c["n"], asserted=c["asserted"] / n, served_not_asserted=c["served_not_asserted"] / n,
                             not_served=c["not_served"] / n, asserted_and_served=c["asserted_served"] / n))
        else:
            s = c["served"] or 1; ns = (c["n"] - c["served"]) or 1
            rows.append(dict(bank=bank, kind=kind, n=c["n"], served=c["served"] / n,
                             p_assert_rival_if_served=c["asserted_if_served"] / s, p_assert_rival_if_not_served=c["asserted_if_not_served"] / ns))
    return rows

if __name__ == "__main__":
    CE = load_ceiling(); print(f"incidents with ceiling data: {len(CE)}")
    r1 = ceiling_table(CE); write_rows("p2_ceiling.csv", r1)
    for r in r1: print(r)
    r2 = gap_table(CE); write_rows("p2_gap.csv", r2)
    print()
    for r in r2: print(r)
