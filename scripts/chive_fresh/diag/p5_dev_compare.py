"""Phase 5 dev comparison: new harness variants vs the frozen harness on dev240 (analysis/harness_improve/dev/fresh{1,2}.txt).

Pass = record judge mean >= 7. Per incident: mean pass over the draws both arms have (d0-d2 gate; d0-d1 budget arms).
Paired incident bootstrap. Pairs:
  gate (echotag+revgate) vs frozen:      olens_gate - olens_rl_B, mlnla_gate - mlnla_B, olens_s16_gate - olens_single_k16
  gate OLens - gate NLA vs frozen OLens - frozen NLA
  budget: *_rx3 (3 experiments, rexp) vs budget-1 (<b>_exp1n30, frozen prompt) and budget-0 (frozen) on d0-d1
Out: data/p5_dev_compare.csv
"""
import json, sys, glob, collections
sys.path.insert(0, "/workspace/agam/chive_scale/report"); sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from common import boot
from dcommon import *

DEV = {b: open(C / "analysis/harness_improve/dev" / f"{b}.txt").read().split() for b in ("fresh1", "fresh2")}

def passes(ev, arm, draws):
    out = collections.defaultdict(dict)
    for b, ids in DEV.items():
        e = ev.format(b=b)
        for pid in ids:
            for d in draws:
                f = C / "harness_runs" / e / arm / f"{pid}.d{d}.json"
                if not f.exists(): continue
                try: r = json.loads(f.read_text())
                except Exception: continue
                if str(r.get("stopped_by", "")).startswith(("error", "salvaged")) or not r.get("judge") or r["judge"].get("score") is None: continue
                out[(b, pid)][d] = r["judge"]["score"] >= 7
    return out

def paired(A, B):
    ks = [k for k in A if k in B and set(A[k]) & set(B[k])]
    a = [sum(A[k][d] for d in set(A[k]) & set(B[k])) / len(set(A[k]) & set(B[k])) for k in ks]
    b = [sum(B[k][d] for d in set(A[k]) & set(B[k])) / len(set(A[k]) & set(B[k])) for k in ks]
    return ks, a, b

def main():
    rows = []
    G = range(3); X = range(2)
    specs = [("olens_gate", "{b}", "olens_rl_B", "{b}", G), ("mlnla_gate", "{b}", "mlnla_B", "{b}", G),
             ("olens_s16_gate", "{b}", "olens_single_k16", "{b}", G),
             ("olens_gate", "{b}", "mlnla_gate", "{b}", G), ("olens_rl_B", "{b}", "mlnla_B", "{b}", G),
             ("olens_s16_gate", "{b}", "mlnla_gate", "{b}", G), ("olens_s16_gate", "{b}", "mlnla_B", "{b}", G),
             ("transcript_rx3", "{b}", "transcript", "{b}_exp1n30", X), ("olens_rx3", "{b}", "olens_rl_B", "{b}_exp1n30", X),
             ("mlnla_rx3", "{b}", "mlnla_B", "{b}_exp1n30", X), ("olens_rx3", "{b}", "olens_rl_B", "{b}", X),
             ("mlnla_rx3", "{b}", "mlnla_B", "{b}", X), ("transcript_rx3", "{b}", "transcript", "{b}", X),
             ("olens_rx3", "{b}", "mlnla_rx3", "{b}", X), ("olens_rx3", "{b}", "transcript_rx3", "{b}", X),
             ("mlnla_rx3", "{b}", "transcript_rx3", "{b}", X)]
    for a, ea, r, er, dr in specs:
        A, B = passes(ea, a, dr), passes(er, r, dr)
        ks, va, vb = paired(A, B)
        if not ks: continue
        m1, m2, d, lo, hi = boot(va, vb)
        rows.append(dict(arm=a, arm_eval=ea, ref=r, ref_eval=er, draws=len(list(dr)), n=len(ks), arm_pass=m1, ref_pass=m2, diff=d, lo=lo, hi=hi))
        print(f"{a:15s} vs {r:17s} ({er:12s}) n={len(ks):3d}  {m1:.3f} vs {m2:.3f}  d={d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    # difference-in-differences: (gate OLens - gate NLA) - (frozen OLens - frozen NLA)
    og, ng, of, nf = (passes("{b}", x, G) for x in ("olens_gate", "mlnla_gate", "olens_rl_B", "mlnla_B"))
    ks = [k for k in og if k in ng and k in of and k in nf]
    def m(D, k): return sum(D[k].values()) / len(D[k])
    if ks:
        v, r_, d, lo, hi = boot([m(og, k) - m(ng, k) for k in ks], [m(of, k) - m(nf, k) for k in ks])
        rows.append(dict(arm="(olens-nla) gate", ref="(olens-nla) frozen", n=len(ks), arm_pass=v, ref_pass=r_, diff=d, lo=lo, hi=hi))
        print(f"DiD (OLens-NLA)gate - (OLens-NLA)frozen n={len(ks)} {v:+.3f} vs {r_:+.3f} d={d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    write_rows("p5_dev_compare.csv", rows)

if __name__ == "__main__":
    main()
