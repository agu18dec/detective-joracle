"""(c) Point-based eval: every judge/version side by side per arm, same strata, paired incident bootstrap vs transcript.

Judges (each matched over the arms it has graded; an incident counts only if every such arm has all 5 base draws):
  old   : old 1-10 pass judge, pass@1 (score >= 7)                         [harness_runs]
  v1    : point eval v1 (points/<b>/grades, rivals): non-obvious recall, primary-cause recall,
          disproved claims / answer, precision = sup/(sup+con), rivals asserted / rejected (share of refuted rivals)
  regrade: blind lenient non-obvious re-grade (points/regrade_all; OLens x4 vs transcript, fresh1+2)
  j2hyp / j2full: judge v2 (blind bullets, ASSERTED/DENIED/ABSENT) on final hypothesis only / hyp+evidence+rival:
          true pts asserted, non-obvious asserted, true pts denied, refuted rivals asserted
Obvious point = no-tools plain answers score it >= 7 in >= 2 of 3 (points_report.py definition).
Outputs: data/c_points.csv, fig/c_points_<stratum>.png (-> chive_graphs/point_eval_by_judge*.png)"""
import json, glob, collections, numpy as np
from common import *

def load_dir(d):
    out = collections.defaultdict(dict)
    for f in glob.glob(str(d / "*.json")):
        name = f.split("/")[-1]
        try: pid, dr = name[:-5].split(".d"); dr = int(dr)
        except ValueError: continue
        if dr < 5: out[pid][dr] = f
    return out

_JL = {}
def jl(f):
    if f in _JL: return _JL[f]
    try: return json.load(open(f))
    except Exception: return None

def prefetch():
    """read every grade/rival/judge2/regrade JSON in parallel (network volume is latency-bound)."""
    import concurrent.futures as cf
    fs = []
    for b in BATCHES:
        for pat in ("points/{b}/grades/*/*.json", "points/{b}/rivals/*/*.json", "points/regrade_all/{b}/*/*.json",
                    "judge2_hyp/{b}/select/*/*.json", "judge2/{b}/select/*/*.json", "points/{b}/points/*.json", "points/{b}/tags/*.json"):
            fs += glob.glob(str(C / pat.format(b=b)))
    def rd(f):
        try: return f, json.load(open(f))
        except Exception: return f, None
    with cf.ThreadPoolExecutor(64) as ex:
        _JL.update(ex.map(rd, fs))

def incident_info():
    """per (b,pid): npts, primary flags, obvious flags, n refuted rivals with cause (judge v2)."""
    info = {}
    for b in BATCHES:
        gt = {}
        for l in open(C / "datasets" / b / "data" / "ground_truth.jsonl"):
            r = json.loads(l); gt[r["prompt_id"]] = sum(1 for x in r["ruled_out"] if x.get("kind") == "refuted" and x.get("cause"))
        for f in glob.glob(str(C / "points" / b / "points" / "*.json")):
            pid = f.split("/")[-1][:-5]
            pts = jl(f)["points"]
            tf = str(C / "points" / b / "tags" / f"{pid}.json")
            prim = (jl(tf) or {}).get("primary")
            pl = [jl(x)["point_scores"] for x in sorted(glob.glob(str(C / "points" / b / "grades/plain" / f"{pid}.d*.json")))[:3]]
            if len(pl) < 3 or any(len(x) != len(pts) for x in pl): continue
            ob = [sum(x[i] >= 7 for x in pl) >= 2 for i in range(len(pts))]
            info[(b, pid)] = dict(npts=len(pts), primary=prim, obvious=ob, nr=gt.get(pid, 0))
    return info

def v1_metrics(info):
    """-> {(b,pid): {arm: {metric: value}}} for arms with 5 graded draws."""
    M = collections.defaultdict(dict)
    for b in BATCHES:
        gdir = C / "points" / b / "grades"
        if not gdir.exists(): continue
        for ad in gdir.iterdir():
            a = ad.name
            if a == "plain" or a not in ARMS: continue
            G = load_dir(ad); RV = load_dir(C / "points" / b / "rivals" / a)
            for pid, ds in G.items():
                I = info.get((b, pid))
                if not I or len(ds) < 5: continue
                m = collections.defaultdict(list); ok = True; s = collections.Counter()
                for d in range(5):
                    g = jl(ds.get(d)) if d in ds else None
                    if not g or len(g["point_scores"]) != I["npts"]: ok = False; break
                    sc = g["point_scores"]
                    no = [i for i in range(I["npts"]) if not I["obvious"][i]]
                    pr = [i for i in range(I["npts"]) if I["primary"] and I["primary"][i]]
                    if no: m["nonobv"].append(sum(sc[i] - 1 for i in no) / (9 * len(no)))
                    if pr: m["primary"].append(sum(sc[i] - 1 for i in pr) / (9 * len(pr)))
                    lab = collections.Counter(c.get("label") for c in g.get("claims", []))
                    m["disproved"].append(lab["contradicted"]); s["sup"] += lab["supported"]; s["con"] += lab["contradicted"]
                    rv = (jl(RV[pid][d]) or {}).get("rivals", []) if d in RV.get(pid, {}) else None
                    if rv is None: ok = False; break
                    s["rv_n"] += len(rv); s["rv_ass"] += rv.count("asserted"); s["rv_rej"] += rv.count("rejected")
                if not ok: continue
                out = {k: float(np.mean(v)) for k, v in m.items() if v}
                if s["sup"] + s["con"]: out["precision"] = s["sup"] / (s["sup"] + s["con"])
                if s["rv_n"]: out["rv_ass"] = s["rv_ass"] / s["rv_n"]; out["rv_rej"] = s["rv_rej"] / s["rv_n"]
                M[(b, pid)][a] = out
    return M

def regrade_metrics(info):
    M = collections.defaultdict(dict)
    base = C / "points" / "regrade_all"
    for b in BATCHES:
        if not (base / b).exists(): continue
        for ad in (base / b).iterdir():
            a = ad.name; G = load_dir(ad)
            for pid, ds in G.items():
                I = info.get((b, pid))
                if not I or len(ds) < 5: continue
                n = sum(not o for o in I["obvious"])
                xs = [jl(ds[d]) for d in range(5) if d in ds]
                if len(xs) < 5 or any(not x or len(x["scores"]) != n for x in xs): continue
                M[(b, pid)][a] = {"nonobv": float(np.mean([sum(s - 1 for s in x["scores"]) / (9 * n) for x in xs]))}
    return M

def j2_metrics(info, J):
    M = collections.defaultdict(dict)
    for b in BATCHES:
        sd = C / J / b / "select"
        if not sd.exists(): continue
        for ad in sd.iterdir():
            a = ad.name
            if a not in ARMS: continue
            G = load_dir(ad)
            for pid, ds in G.items():
                I = info.get((b, pid))
                if not I or len(ds) < 5: continue
                xs = collections.defaultdict(list); ok = True
                for d in range(5):
                    r = jl(ds.get(d)) if d in ds else None
                    if r is None: ok = False; break
                    lab = (r or {}).get("labels", {}) or {}
                    t = [lab.get(f"P{i + 1}", "ABSENT") for i in range(I["npts"])]; rv = [lab.get(f"R{j + 1}", "ABSENT") for j in range(I["nr"])]
                    xs["true"].append(sum(x == "ASSERTED" for x in t) / len(t))
                    if not all(I["obvious"]): xs["nonobv"].append(sum(x == "ASSERTED" for x, o in zip(t, I["obvious"]) if not o) / sum(not o for o in I["obvious"]))
                    xs["denied"].append(sum(x == "DENIED" for x in t) / len(t))
                    if I["nr"]: xs["rival"].append(sum(x == "ASSERTED" for x in rv) / I["nr"])
                if ok: M[(b, pid)][a] = {k: float(np.mean(v)) for k, v in xs.items() if v}
    return M

def old_metrics():
    T = pass_table(ARMS)
    return {k: {a: {"pass1": sum(v[a]) / 5} for a in v} for k, v in T.items()}

SPECS = [  # (judge key, metric key, label, pct)
    ("old", "pass1", "Old 1-10 judge: pass@1", True),
    ("v1", "nonobv", "Point eval v1: non-obvious recall", True),
    ("v1", "primary", "Point eval v1: primary-cause recall", True),
    ("v1", "disproved", "Point eval v1: disproved claims / answer (lower = better)", False),
    ("v1", "precision", "Point eval v1: precision", True),
    ("v1", "rv_ass", "Point eval v1: refuted rivals asserted (lower = better)", True),
    ("v1", "rv_rej", "Point eval v1: refuted rivals rejected", True),
    ("regrade", "nonobv", "Blind re-grade: non-obvious recall (fresh1+2)", True),
    ("j2hyp", "true", "Judge v2 hyp-only: true points asserted", True),
    ("j2hyp", "nonobv", "Judge v2 hyp-only: non-obvious asserted", True),
    ("j2hyp", "denied", "Judge v2 hyp-only: true points denied (lower = better)", True),
    ("j2hyp", "rival", "Judge v2 hyp-only: refuted rivals asserted (lower = better)", True),
    ("j2full", "true", "Judge v2 full: true points asserted", True),
    ("j2full", "nonobv", "Judge v2 full: non-obvious asserted", True),
    ("j2full", "denied", "Judge v2 full: true points denied (lower = better)", True),
    ("j2full", "rival", "Judge v2 full: refuted rivals asserted (lower = better)", True),
]

def main():
    prefetch(); info = incident_info()
    J = {"old": old_metrics(), "v1": v1_metrics(info), "regrade": regrade_metrics(info),
         "j2hyp": j2_metrics(info, "judge2_hyp"), "j2full": j2_metrics(info, "judge2")}
    S = strata(); rows = []; meta = {}
    for jk, M in J.items():
        cov = {a: sum(a in v for v in M.values()) for a in ARMS}
        if cov["transcript"] < 20: continue
        # interim rule: an arm joins the matched set once it covers >= 80% of transcript's incidents
        present = [a for a in ARMS if cov[a] >= 0.8 * cov["transcript"]]
        matched = sorted(k for k, v in M.items() if all(a in v for a in present))
        meta[jk] = {"present": present, "n_matched": len(matched), "coverage": cov}
        for (jj, mk, lab, pct) in SPECS:
            if jj != jk: continue
            for st in STRATA:
                ks = [k for k in in_stratum(matched, S, st) if all(mk in M[k][a] for a in present)]
                for a in present:
                    m, rm, d, lo, hi = boot([M[k][a][mk] for k in ks], [M[k]["transcript"][mk] for k in ks])
                    rows.append(dict(judge=jk, metric=mk, label=lab, stratum=st, arm=a, n=len(ks), mean=m, transcript=rm, diff=d, lo=lo, hi=hi))
    write_csv("c_points", rows, ["judge", "metric", "label", "stratum", "arm", "n", "mean", "transcript", "diff", "lo", "hi"])
    json.dump(meta, open(REP / "data" / "c_meta.json", "w"), indent=1)
    plt = mpl()
    for st, exp in (("overall", "point_eval_by_judge"), ("hardest third", "point_eval_by_judge_hardest_third"), ("least-confident third", "point_eval_by_judge_least_confident_third")):
        specs = [s for s in SPECS if any(r["judge"] == s[0] and r["metric"] == s[1] and r["stratum"] == st for r in rows)]
        if not specs: continue
        nc = 4; nr = (len(specs) + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(21, 2.9 * nr + 1), gridspec_kw={"wspace": 1.0, "hspace": 0.8})
        axes = np.atleast_2d(axes)
        for ax in axes.flat: ax.axis("off")
        for i, (jj, mk, lab, pct) in enumerate(specs):
            ax = axes.flat[i]; ax.axis("on")
            rr = [r for r in rows if r["judge"] == jj and r["metric"] == mk and r["stratum"] == st]
            forest(ax, rr, lab.replace(": ", ":\n", 1) + f" (n={rr[0]['n']})", pct=pct, show_labels=True,
                   ref_text=f"transcript = {(100 if pct else 1) * rr[0]['transcript']:.{1 if pct else 2}f}{'%' if pct else ''}")
            ax.title.set_fontsize(8.5)
        fig.suptitle(f"Point-based evaluation, every judge side by side · {st}", x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
        fig.text(0.01, 1 - 0.55 / (2.9 * nr + 1), "Each panel: paired difference vs transcript-only (dot) with 95% incident bootstrap CI; right labels = the arm's absolute value. "
                 "Each judge is matched over the arms it has graded (n in title). Hollow = secondary arm.", color=INK2, fontsize=8.5)
        fig.subplots_adjust(left=0.09, right=0.97, top=1 - 1.1 / (2.9 * nr + 1), bottom=0.04)
        save(fig, "c_points_" + st.replace(" ", "_").replace("-", "_"), exp); plt.close(fig)
    print(json.dumps(meta))

if __name__ == "__main__":
    main()
