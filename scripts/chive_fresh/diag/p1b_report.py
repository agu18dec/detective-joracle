"""Phase 1b report: judge v3 (claim-centric) precision / recall / hallucination per arm, and committed -> final deltas.

Per answer (labels from diag/judge3/<b>/cls_{final,commit}/<arm>/<pid>.d<k>.json):
  specific claims = TRUE + RIVAL + CONTRADICTED + UNSUPPORTED (VAGUE and NOT-claims excluded)
  precision       = TRUE / specific
  halluc          = UNSUPPORTED / specific          (specific causal claim GT neither supports nor contradicts)
  wrong           = (RIVAL + CONTRADICTED) / specific
  recall          = distinct true points matched / all points; recall_nonobv over non-obvious points;
                    recall_primary over T1-tagged points
  rival_rate      = distinct refuted rivals asserted / refuted rivals (incidents with >= 1)
  deny_true       = any true point wrongly rejected ('NOT:' claim)
  n_specific, n_vague
Per incident: mean over the draws present (d0-d4); paired incident bootstrap vs transcript and OLens - NLA; strata as
report/common.py. Commit->final: per episode delta (final - committed) for reader arms, then per incident.
Out: experiments/chive-diag/data/p1b_judge3.csv, p1b_commit_delta.csv
"""
import json, sys, collections, numpy as np
sys.path.insert(0, "/workspace/agam/chive_scale/report"); sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from common import boot, strata, in_stratum, STRATA
from dcommon import *

ARMS = ["transcript", "olens_rl_B", "mlnla_B", "jlens_B"]
METS = ["precision", "halluc", "wrong", "recall", "recall_nonobv", "recall_primary", "rival_rate", "deny_true", "n_specific", "n_vague"]

def flags(b, pid, _c={}):
    if (b, pid) not in _c:
        pts = json.loads((C / "points" / b / "points" / f"{pid}.json").read_text())["points"]
        pl = [json.loads(p.read_text())["point_scores"] for p in sorted((C / "points" / b / "grades/plain").glob(f"{pid}.d*.json"))[:3]]
        ob = [sum(x[i] >= 7 for x in pl) >= 2 for i in range(len(pts))] if len(pl) == 3 and all(len(x) == len(pts) for x in pl) else None
        tf = C / "points" / b / "tags" / f"{pid}.json"
        prim = json.loads(tf.read_text()).get("primary") if tf.exists() else None
        _c[(b, pid)] = (ob, prim)
    return _c[(b, pid)]

def metrics(j, b, pid):
    L = j.get("labels", []); npts, nrv = j.get("npts", 0), j.get("nrv", 0)
    lab = [str(x.get("label", "")).upper() for x in L]; mat = [str(x.get("match") or "") for x in L]
    c = collections.Counter(lab); spec = c["TRUE"] + c["RIVAL"] + c["CONTRADICTED"] + c["UNSUPPORTED"]
    tp = {m for l, m in zip(lab, mat) if l == "TRUE" and m.startswith("P")}
    rv = {m for l, m in zip(lab, mat) if l == "RIVAL" and m.startswith("R")}
    ob, prim = flags(b, pid)
    out = {"n_specific": spec, "n_vague": c["VAGUE"], "deny_true": float(c["DENY_TRUE"] > 0),
           "recall": len(tp) / npts if npts else np.nan}
    if spec:
        out.update(precision=c["TRUE"] / spec, halluc=c["UNSUPPORTED"] / spec, wrong=(c["RIVAL"] + c["CONTRADICTED"]) / spec)
    if ob and not all(ob):
        no = [f"P{i + 1}" for i, o in enumerate(ob) if not o]; out["recall_nonobv"] = len(tp & set(no)) / len(no)
    if prim and any(prim):
        pr = [f"P{i + 1}" for i, t in enumerate(prim) if t]; out["recall_primary"] = len(tp & set(pr)) / len(pr)
    if nrv: out["rival_rate"] = len(rv) / nrv
    return out

def load(which):
    M = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))  # (b,pid)->arm->met->[draw vals]
    E = {}  # (b,arm,pid,d) -> metrics
    for b in BATCHES:
        base = C / "diag" / "judge3" / b / f"cls_{which}"
        if not base.exists(): continue
        for ad in base.iterdir():
            for f in ad.glob("*.json"):
                pid, d = f.name[:-5].split(".d")
                try: j = json.loads(f.read_text())
                except Exception: continue
                m = metrics(j, b, pid); E[(b, ad.name, pid, int(d))] = m
                for k, v in m.items():
                    if v == v: M[(b, pid)][ad.name][k].append(v)
    return M, E

def main():
    S = strata(); M, EF = load("final"); _, EC = load("commit")
    rows = []
    for st in STRATA:
        for met in METS:
            for a, ref in [("olens_rl_B", "transcript"), ("mlnla_B", "transcript"), ("jlens_B", "transcript"), ("olens_rl_B", "mlnla_B")]:
                ks = [k for k in in_stratum(M.keys(), S, st) if M[k][a].get(met) and M[k][ref].get(met)]
                v, r, d, lo, hi = boot([np.mean(M[k][a][met]) for k in ks], [np.mean(M[k][ref][met]) for k in ks])
                rows.append(dict(stratum=st, metric=met, arm=a, ref=ref, n=len(ks), arm_mean=v, ref_mean=r, diff=d, lo=lo, hi=hi))
    write_rows("p1b_judge3.csv", rows)
    print("== judge v3, final hypothesis (overall)")
    for r in rows:
        if r["stratum"] == "overall":
            print(f"{r['metric']:15s} {r['arm']:10s}-{r['ref']:10s} n={r['n']:4d} {r['arm_mean']:.3f} vs {r['ref_mean']:.3f} d={r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]")
    # committed -> final (reader arms), per incident mean of per-episode deltas
    D = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for (b, a, pid, d), mc in EC.items():
        mf = EF.get((b, a, pid, d))
        if not mf: continue
        for k in METS:
            if k in mc and k in mf and mc[k] == mc[k] and mf[k] == mf[k]: D[(b, pid)][a][k].append(mf[k] - mc[k])
    rows2 = []
    print("\n== committed -> final change (reader arms), per incident")
    for a in ["olens_rl_B", "mlnla_B", "jlens_B"]:
        for met in METS:
            ks = [k for k in D if D[k][a].get(met)]
            if not ks: continue
            v = [np.mean(D[k][a][met]) for k in ks]; m, lo, hi = (lambda x: (np.mean(x), *np.percentile([np.mean(np.random.default_rng(i).choice(x, len(x))) for i in range(1000)], [2.5, 97.5])))(np.array(v))
            rows2.append(dict(arm=a, metric=met, n=len(ks), delta=m, lo=lo, hi=hi))
            print(f"{a:10s} {met:15s} n={len(ks):4d} final-commit {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    # OLens - NLA difference in the delta
    for met in METS:
        ks = [k for k in D if D[k]["olens_rl_B"].get(met) and D[k]["mlnla_B"].get(met)]
        if ks:
            v, r, d, lo, hi = boot([np.mean(D[k]["olens_rl_B"][met]) for k in ks], [np.mean(D[k]["mlnla_B"][met]) for k in ks])
            rows2.append(dict(arm="olens-nla", metric=met, n=len(ks), delta=d, lo=lo, hi=hi))
    write_rows("p1b_commit_delta.csv", rows2)

if __name__ == "__main__":
    main()
