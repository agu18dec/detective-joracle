"""Phase 1a: claim-level precision / hallucination from the EXISTING point-eval v1 grades (no new LLM calls).

v1 grader labelled up to 10 of each answer's own claims (answer = hypothesis+evidence+rival) as
supported | contradicted | rejected | not_covered against CHIVE's judge_reference.
  precision      = sup / (sup + con)
  not_covered    = claims GT neither supports nor contradicts (hallucination PROXY)
  nc_internal    = not_covered claims that are about internals (layers/readouts/'internally') -> unverifiable by GT,
                   not necessarily hallucinations; nc_external = the rest (claims about the conversation/model behaviour)
Per incident: pooled over d0-d4; paired bootstrap vs transcript and OLens x4 vs NLA x4.
Out: /workspace/agam/experiments/chive-diag/data/p1a_v1claims.csv
"""
import sys, re, json, collections, numpy as np
sys.path.insert(0, "/workspace/agam/chive_scale/report"); sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from common import *
import c_points as CP
from dcommon import OUT, write_rows

ARMS4 = ["transcript", "olens_rl_B", "mlnla_B", "jlens_B"]
INTERNAL = re.compile(r"\b(layer|layers|readout|internal(ly)?|latent|activation|lens|L\d{2}|representation|residual|hidden state|mid-layer|late layer|early layer)\b", re.I)

def main():
    CP.prefetch(); info = CP.incident_info(); S = strata()
    M = collections.defaultdict(dict)
    for b in BATCHES:
        for a in ARMS4:
            G = CP.load_dir(C / "points" / b / "grades" / a)
            for pid, ds in G.items():
                if (b, pid) not in info or len(ds) < 5: continue
                c = collections.Counter(); n_ans = 0
                for d in range(5):
                    g = CP.jl(ds[d])
                    if not g: break
                    n_ans += 1
                    for cl in g.get("claims", []):
                        L = cl.get("label"); c[L] += 1; c["all"] += 1
                        if L == "not_covered": c["nc_int" if INTERNAL.search(cl.get("claim", "")) else "nc_ext"] += 1
                if n_ans < 5 or not c["all"]: continue
                M[(b, pid)][a] = {
                    "precision": c["supported"] / max(1, c["supported"] + c["contradicted"]),
                    "supported_share": c["supported"] / c["all"],
                    "contradicted_share": c["contradicted"] / c["all"],
                    "not_covered_share": c["not_covered"] / c["all"],
                    "nc_internal_share": c["nc_int"] / c["all"],
                    "nc_external_share": c["nc_ext"] / c["all"],
                    "claims_per_answer": c["all"] / 5,
                    "contradicted_per_answer": c["contradicted"] / 5,
                    "supported_per_answer": c["supported"] / 5,
                }
    keys = [k for k, v in M.items() if all(a in v for a in ARMS4[:3])]
    rows = []
    for st in STRATA:
        ks = in_stratum(keys, S, st)
        for met in ["precision", "supported_share", "contradicted_share", "not_covered_share", "nc_internal_share", "nc_external_share",
                    "claims_per_answer", "supported_per_answer", "contradicted_per_answer"]:
            for a, ref in [("olens_rl_B", "transcript"), ("mlnla_B", "transcript"), ("jlens_B", "transcript"), ("olens_rl_B", "mlnla_B")]:
                kk = [k for k in ks if a in M[k] and ref in M[k]]
                v, r, d, lo, hi = boot([M[k][a][met] for k in kk], [M[k][ref][met] for k in kk])
                rows.append(dict(stratum=st, metric=met, arm=a, ref=ref, n=len(kk), arm_mean=v, ref_mean=r, diff=d, lo=lo, hi=hi))
    write_rows("p1a_v1claims.csv", rows)
    for r in rows:
        if r["stratum"] == "overall":
            print(f"{r['metric']:24s} {r['arm']:11s} vs {r['ref']:10s} n={r['n']:4d}  {r['arm_mean']:.3f} vs {r['ref_mean']:.3f}  "
                  f"d={r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]")

if __name__ == "__main__":
    main()
