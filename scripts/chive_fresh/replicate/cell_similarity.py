"""Same-cell agreement between two banks of the same reader on the same incidents.

  python3 cell_similarity.py <ids.txt> <n_pos> BANK_A BANK_B

Per sampled (incident, position, layer) cell, word sets (>=4 letters) of the samples:
  A~B same cell   Jaccard(words(A[0:2]), words(B[0:2]))  (2 vs 2 samples)
  A~A same cell   Jaccard(words(A[0:2]), words(A[2:4]))  (noise ceiling: two independent pairs from one bank)
  B~B same cell   Jaccard(words(B[0:2]), words(B[2:4]))
  A~B shifted     Jaccard(words(A[0:2]), words(B at a far position, same layer))  (floor)
If B reproduces A's distribution, A~B ≈ A~A ≈ B~B, well above the shifted floor.
"""
import gzip, json, random, re, statistics, sys
from collections import defaultdict
from pathlib import Path
ids = Path(sys.argv[1]).read_text().split(); n = int(sys.argv[2]); A, B = Path(sys.argv[3]), Path(sys.argv[4])
W = lambda ss: {w for s in ss for w in re.findall(r"[a-z]{4,}", s.lower())}
J = lambda a, b: len(a & b) / max(1, len(a | b))
res = defaultdict(lambda: defaultdict(list))
for pid in ids:
    a = json.load(gzip.open(A / f"{pid}.json.gz", "rt"))["readouts"]; b = json.load(gzip.open(B / f"{pid}.json.gz", "rt"))["readouts"]
    for L in a:
        npos = len(a[L]); rng = random.Random(f"{pid}:{L}")
        for p in rng.sample(range(npos), min(n, npos)):
            q = (p + npos // 2) % npos
            ac, bc, bq = a[L][str(p)], b[L][str(p)], b[L][str(q)]
            r = res[int(L)]
            r["AB"].append(J(W(ac[:2]), W(bc[:2]))); r["AA"].append(J(W(ac[:2]), W(ac[2:4])))
            r["BB"].append(J(W(bc[:2]), W(bc[2:4]))); r["AB_shift"].append(J(W(ac[:2]), W(bq[:2])))
print("layer  A~B  A~A  B~B  A~B(shifted)")
tot = defaultdict(list)
for L in sorted(res):
    m = {k: statistics.mean(v) for k, v in res[L].items()}
    for k, v in res[L].items(): tot[k] += v
    print(f"L{L:<4} {m['AB']:.3f} {m['AA']:.3f} {m['BB']:.3f} {m['AB_shift']:.3f}")
m = {k: statistics.mean(v) for k, v in tot.items()}
print(f"all   {m['AB']:.3f} {m['AA']:.3f} {m['BB']:.3f} {m['AB_shift']:.3f}  (cells={len(tot['AB'])})")
