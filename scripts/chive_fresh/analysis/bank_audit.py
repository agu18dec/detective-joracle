"""Bank integrity: per bank, over a sample of incidents: layers, coverage (positions with readouts / n_tokens),
samples per cell vs k, empty samples, near-cap-length samples, identical samples within a cell, mean chars."""
import gzip, json, glob, random, sys, collections, statistics as st
from concurrent.futures import ProcessPoolExecutor
def one(f):
    r = json.load(gzip.open(f, "rt")); n = len(r["token_ids"]); k = int(r.get("k", 1))
    L = sorted(int(x) for x in r["readouts"]); pos = set(); cells = 0; badk = 0; empty = 0; samp = 0; dup = 0; ch = []
    for l, pp in r["readouts"].items():
        for p, s in pp.items():
            pos.add(int(p)); cells += 1; badk += len(s) != k; samp += len(s)
            empty += sum(not str(x).strip() for x in s); ch += [len(str(x)) for x in s]
            dup += len(s) > 1 and len(set(s)) == 1
    return dict(n=n, k=k, L=tuple(L), cov=len(pos) / n, cells=cells, badk=badk / max(cells, 1), empty=empty / max(samp, 1),
                dup=dup / max(cells, 1), chars=st.mean(ch) if ch else 0, p95=sorted(ch)[int(.95 * len(ch))] if ch else 0,
                note=bool(r.get("coverage_note")))
def audit(d, m=60):
    fs = sorted(glob.glob(d + "/*.json.gz")); random.Random(0).shuffle(fs)
    with ProcessPoolExecutor(16) as ex: R = list(ex.map(one, fs[:m]))
    Ls = collections.Counter(r["L"] for r in R)
    f = lambda key: round(st.mean(r[key] for r in R), 4)
    print(f"{d.split('chive_scale/')[1]:52s} files={len(fs):4d} k={collections.Counter(r['k'] for r in R).most_common(1)[0][0]} layers={Ls.most_common(1)[0][0][:3]}..{Ls.most_common(1)[0][0][-1]}({len(Ls.most_common(1)[0][0])}) layer-sets={len(Ls)} cov={f('cov')} min_cov={round(min(r['cov'] for r in R),3)} badk={f('badk')} empty={f('empty')} dup={f('dup')} chars={f('chars'):.0f} p95={f('p95'):.0f} note={sum(r['note'] for r in R)}")
B = "/workspace/agam/chive_scale/"
for d in ["replicate_banks/orig/s3d_rl600_all", "banks/s3d_rl600_all", "replicate_banks/orig/mlnla_k4_chive", "banks/mlnla_k4_chive",
          "replicate_banks/orig/jlens_all", "banks/jlens_L20-60", "banks/s3_single_rl", "banks/mlnla_k1", "banks/olens_k1"]:
    audit(B + d)
