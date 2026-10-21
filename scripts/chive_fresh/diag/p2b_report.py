"""Ceiling v2 report (diag/ceiling2): statable items only, VALID lines only, prompt side. Paired OLens - NLA, shuffled floor."""
import json, sys, collections, numpy as np
sys.path.insert(0, "/workspace/agam/chive_scale/report"); sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from common import boot
from dcommon import *
D = collections.defaultdict(dict); stat = collections.Counter()
for b in BATCHES:
    for bd in (C / "diag" / "ceiling2" / b).glob("[a-z]*"):
        for f in bd.glob("*.json"):
            r = json.loads(f.read_text()); D[(b, f.stem)][bd.name] = r["items"]
            if bd.name == "olens":
                for it in r["items"]:
                    k = it["kind"] + ("" if it["kind"] == "rival" else ("_obv" if it["obvious"] else "_nonobv"))
                    stat[(k, it["statable"])] += 1
print("statable share:", {k: f"{stat[(k, True)]}/{stat[(k, True)] + stat[(k, False)]}" for k in ("true_nonobv", "true_obv", "rival")})
def sel(its, kind):
    return [it for it in its if it["statable"] and it["hits"] is not None and (
        (kind == "rival" and it["kind"] == "rival") or (kind == "true_nonobv" and it["kind"] == "true" and it["obvious"] is False)
        or (kind == "true_obv" and it["kind"] == "true" and it["obvious"] is True))]
rows = []
for kind in ("true_nonobv", "true_obv", "rival"):
    for thr in (1, 3):
        for bank in ("olens", "mlnla", "olens_shuf", "mlnla_shuf"):
            v = [np.mean([len(it["hits"]) >= thr for it in sel(x[bank], kind)]) for x in D.values() if bank in x and sel(x[bank], kind)]
            m, lo, hi = (np.mean(v), *np.percentile([np.mean(np.random.default_rng(i).choice(v, len(v))) for i in range(1000)], [2.5, 97.5])) if v else (np.nan,) * 3
            rows.append(dict(kind=kind, min_lines=thr, bank=bank, n=len(v), share=m, lo=lo, hi=hi))
            print(f"{kind:12s} >={thr} {bank:11s} n={len(v):3d} {m:.3f} [{lo:.3f},{hi:.3f}]")
        ks = [k for k, x in D.items() if "olens" in x and "mlnla" in x and sel(x["olens"], kind) and sel(x["mlnla"], kind)]
        a = [np.mean([len(it["hits"]) >= thr for it in sel(D[k]["olens"], kind)]) for k in ks]
        r = [np.mean([len(it["hits"]) >= thr for it in sel(D[k]["mlnla"], kind)]) for k in ks]
        m1, m2, d, lo, hi = boot(a, r); rows.append(dict(kind=kind, min_lines=thr, bank="olens-mlnla", n=len(ks), share=d, lo=lo, hi=hi))
        print(f"{kind:12s} >={thr} OLens-NLA   n={len(ks):3d} {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
write_rows("p2b_ceiling2.csv", rows)
