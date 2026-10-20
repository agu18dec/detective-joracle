"""(c, part 2) Claims eval: yes/no claim AUROC per arm (claims_runs/<b>/focused/<arm>, probability_true;
salvaged/error records dropped). Each arm vs transcript on the claims both answered (pairwise: the jlens_B /
mlnla_k1 claims runs are a fresh1-only subset), pooled AUROC (Mann-Whitney), paired incident bootstrap 95% CI
of the AUROC difference (2000 resamples of incidents). Strata as elsewhere.
Outputs: data/c_claims.csv, fig/c_claims.png (-> chive_graphs/claims_auroc.png)"""
import json, glob, collections, numpy as np
import concurrent.futures as cf
from common import *

def auc(p, y):
    p = np.asarray(p, float); y = np.asarray(y, bool); pos = y.sum(); neg = len(y) - pos
    if not pos or not neg: return np.nan
    order = np.argsort(p, kind="mergesort"); ps = p[order]; ranks = np.empty(len(p))
    i = 0
    while i < len(ps):
        j = i
        while j < len(ps) and ps[j] == ps[i]: j += 1
        ranks[order[i:j]] = (i + j + 1) / 2; i = j
    return (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg)

def main():
    claims = {}; files = []
    for b in BATCHES:
        f = C / "claims_runs" / b / "claims.jsonl"
        if not f.exists(): continue
        for l in open(f):
            c = json.loads(l); claims[c["claim_id"]] = (b, c["prompt_id"], bool(c["label"]))
        files += glob.glob(str(C / "claims_runs" / b / "focused" / "*" / "*.json"))
    def rd(f):
        try: return json.load(open(f))
        except Exception: return None
    P = collections.defaultdict(lambda: collections.defaultdict(list))
    with cf.ThreadPoolExecutor(64) as ex:
        for r in ex.map(rd, files):
            if not r or bad(r): continue
            p = (r.get("result") or {}).get("probability_true"); cid = r["claim"]["claim_id"]
            if p is None or cid not in claims: continue
            P[r["arm"]][cid].append(float(p))
    S = strata(); rows = []
    arms = [a for a in ARMS if a in P]
    for st in STRATA:
        for a in arms:
            common = [c for c in P[a] if c in P["transcript"]]
            common = [c for c in common if S[st] is None or claims[c][:2] in S[st]]
            if not common: continue
            incs = sorted({claims[c][:2] for c in common}); by = collections.defaultdict(list)
            for c in common: by[claims[c][:2]].append(c)
            def stat(cs):
                y = [claims[c][2] for c in cs]
                return auc([np.mean(P[a][c]) for c in cs], y), auc([np.mean(P["transcript"][c]) for c in cs], y)
            m, rm = stat(common)
            rng = np.random.default_rng(0); ds = []
            for _ in range(2000):
                cs = [c for i in rng.integers(0, len(incs), len(incs)) for c in by[incs[i]]]
                x, y = stat(cs); ds.append(x - y)
            ds = np.sort(ds)
            rows.append(dict(stratum=st, arm=a, n_claims=len(common), n=len(incs), n_true=sum(claims[c][2] for c in common),
                             mean=m, transcript=rm, diff=m - rm, lo=ds[50], hi=ds[1949]))
    write_csv("c_claims", rows, ["stratum", "arm", "n_claims", "n", "n_true", "mean", "transcript", "diff", "lo", "hi"])
    plt = mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.3), gridspec_kw={"wspace": 0.55})
    for j, st in enumerate(STRATA):
        rr = [r for r in rows if r["stratum"] == st]
        if not rr: axes[j].axis("off"); continue
        forest(axes[j], rr, f"Claim AUROC · {st}", pct=False, xlabel="AUROC difference vs transcript", show_labels=(j == 0), show_n=True,
               ref_text="transcript AUROC on each arm's claims: " + "/".join(f"{r['transcript']:.3f}" for r in rr if r["arm"] != "transcript"))
    fig.suptitle("Claims eval: does the agent judge a yes/no counterfactual claim correctly? (pooled AUROC)", x=0.01, ha="left", color=INK, fontsize=11, fontweight="bold")
    fig.text(0.01, 0.9, "Each arm vs transcript on the claims both answered (n = incidents). J-lens L0-62 and multi-layer NLA x1 claims runs cover a fresh1 subset only. "
             "95% paired incident bootstrap CI.", color=INK2, fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.7, bottom=0.17)
    save(fig, "c_claims", "claims_auroc"); plt.close(fig)
    for r in rows: print(r["stratum"], r["arm"], r["n_claims"], f"{r['mean']:.3f}", f"{r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]")

if __name__ == "__main__":
    main()
