"""(b) Budgeted counterfactual (1 experiment, n=30 Qwen samples, draws d0-d1, evals <batch>_exp1n30).
Per episode: final pass (judge >= 7), concluder pass (concluder_judge >= 7), mover (|exp rate - re-measured
baseline| >= 30 pp), reduction (baseline - exp rate >= 30 pp). Episodes without an experiment count as
non-movers. Per incident = mean over draws; matched incidents (every present arm has both draws judged);
paired incident bootstrap 95% CI vs transcript; strata as in (a).
Outputs: data/b_budget.csv, data/b_meta.json, fig/b_budget.png (-> chive_graphs/budgeted_counterfactual.png)"""
import json, collections, numpy as np
from common import *
DRAWS = range(2)
METRICS = [("final", "final pass@1"), ("conc", "concluder pass@1"), ("mover", "mover rate (|Δ| >= 30 pp)"), ("reduce", "reduction rate (Δ <= -30 pp)")]

def main():
    base = json.load(open(C / "exp1" / "baselines.json"))
    R = load_records([f"{b}_exp1n30" for b in BATCHES], ARMS, DRAWS)
    E = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (ev, a, pid, d), r in R.items():
        if r["score"] is None or pid not in base: continue
        b0 = base[pid]["rate"]; ex = r["exp"]
        delta = (ex["rate"] - b0) if ex and ex.get("rate") is not None else 0.0
        conc = r["conc_score"] if r["conc_score"] is not None else (None if ex else r["score"])  # no experiment -> no concluder step
        E[(ev.split("_")[0], pid)][a][d] = {"final": float(r["score"] >= 7), "conc": None if conc is None else float(conc >= 7),
                                           "mover": float(abs(delta) >= 0.30 - 1e-9), "reduce": float(-delta >= 0.30 - 1e-9)}
    S = strata(); rows = []; meta = {}
    for m, lab in METRICS:  # each metric matched separately (concluder judging lags the final judge)
        inc = {}
        for k, v in E.items():
            for a, dd in v.items():
                xs = [dd[d][m] for d in DRAWS if d in dd]
                if len(xs) == len(DRAWS) and None not in xs: inc.setdefault(k, {})[a] = float(np.mean(xs))
        cov = {a: sum(a in v for v in inc.values()) for a in ARMS}
        if cov["transcript"] < 20: continue
        present = [a for a in ARMS if cov[a] >= max(20, 0.8 * cov["transcript"])]  # interim: arm joins once it covers 80% of transcript
        matched = sorted(k for k, v in inc.items() if all(a in v for a in present))
        meta[lab] = {"present": present, "missing": [a for a in ARMS if a not in present], "n_matched": len(matched), "coverage": cov}
        for st in STRATA:
            ks = in_stratum(matched, S, st)
            for a in present:
                mm, rm, d, lo, hi = boot([inc[k][a] for k in ks], [inc[k]["transcript"] for k in ks])
                rows.append(dict(stratum=st, metric=lab, arm=a, n=len(ks), mean=mm, transcript=rm, diff=d, lo=lo, hi=hi))
    write_csv("b_budget", rows, ["stratum", "metric", "arm", "n", "mean", "transcript", "diff", "lo", "hi"])
    allm = set().union(*[set(v["missing"]) for v in meta.values()]) if meta else set(ARMS)
    missing = [a for a in ARMS if a in allm]
    json.dump({"missing": missing, "n_matched": min((v["n_matched"] for v in meta.values()), default=0), "per_metric": meta},
              open(REP / "data" / "b_meta.json", "w"), indent=1)
    matched = []
    if not rows: print("no data"); return
    plt = mpl()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex="row", gridspec_kw={"wspace": 0.45})
    for i, (m, lab) in enumerate(METRICS):
        for j, st in enumerate(STRATA):
            rr = [r for r in rows if r["stratum"] == st and r["metric"] == lab]
            if not rr: axes[i, j].axis("off"); continue
            forest(axes[i, j], rr, f"{lab} · {st} (n={rr[0]['n']})", show_labels=(j == 0), ref_text=f"transcript = {100 * rr[0]['transcript']:.1f}%")
    fig.suptitle("Budgeted counterfactual (1 experiment, 30 Qwen samples): difference vs transcript-only", x=0.01, ha="left", color=INK, fontsize=11, fontweight="bold")
    fig.text(0.01, 0.94, "Matched incidents per metric (concluder judging lags), draws d0-d1, per-incident mean. Mover = the single experiment moves the behaviour rate >= 30 pp vs the baseline re-measured with the same executor. "
             "95% paired incident bootstrap CIs." + (f"  Not yet available: {', '.join(missing)}." if missing else ""), color=INK2, fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.91, bottom=0.05, hspace=0.6); save(fig, "b_budget", "budgeted_counterfactual"); plt.close(fig)
    print({k: (v["n_matched"], v["missing"]) for k, v in meta.items()})

if __name__ == "__main__":
    main()
