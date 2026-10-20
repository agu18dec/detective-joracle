"""Bank content comparison: format, within-cell duplication and grounding of sampled readouts.

  python3 bank_compare.py <tokens.jsonl> <ids.txt|-> <n_pos_per_incident> LABEL=BANK_DIR ...

Per bank, over the same sampled positions (seeded by prompt id; positions before the response):
  chars/sample, lines/sample, empty %, share of samples with tag residue (<explanation>, </...>),
  share of samples truncated-looking (no final punctuation/newline and >= 95th pct length),
  within-cell duplicates (share of cells with any 2 identical samples), distinct-sample ratio,
  grounding: share of content words (>=4 letters, not a stopword) in a sample that occur in the
  conversation prefix (pieces[:p+1]); also in the whole conversation; and in neither ("novel").
"""
import collections, gzip, json, random, re, statistics, sys
from pathlib import Path

STOP = set("""this that with from have what will would there their they them then than been were
which when where your about into also just like some such more most only other over these those
very here does doing done each make made many much must need should could being after before
because while both same between through user model text response answer question token tokens
layer position context conversation message assistant""".split())
tokf, idsf, npos = sys.argv[1], sys.argv[2], int(sys.argv[3])
banks = [a.split("=", 1) for a in sys.argv[4:]]
ids = None if idsf == "-" else set(Path(idsf).read_text().split())
toks = {}
for line in open(tokf):
    d = json.loads(line)
    if ids is None or d["prompt_id"] in ids:
        toks[d["prompt_id"]] = d
W = lambda s: [w for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in STOP]
for label, bdir in banks:
    B = Path(bdir); c = collections.Counter(); lens = []; lines = []; gp = []; ga = []; dup = 0; cells = 0; dist = []
    pids = sorted(p for p in toks if (B / f"{p}.json.gz").exists())
    for pid in pids:
        t = toks[pid]; pieces = t["pieces"]; roles = [r for _, r in t["turn_of"]]
        try: last_user = max(i for i, r in enumerate(roles) if r == "user")
        except ValueError: last_user = len(pieces) - 1
        R = json.load(gzip.open(B / f"{pid}.json.gz", "rt"))["readouts"]
        cand = list(range(5, max(6, last_user)))
        pos = random.Random(pid).sample(cand, min(npos, len(cand)))
        allw = set(W("".join(pieces)))
        for p in pos:
            pre = set(W("".join(pieces[: p + 1])))
            for L, per in R.items():
                ss = [str(s) for s in per.get(str(p), [])]
                if not ss: continue
                cells += 1; dup += len(set(ss)) < len(ss); dist.append(len(set(ss)) / len(ss))
                for s in ss:
                    lens.append(len(s)); lines.append(len([x for x in s.split("\n") if x.strip()]))
                    c["n"] += 1; c["empty"] += not s.strip()
                    c["tag"] += bool(re.search(r"</?explanation>|<\|im_|</?think>", s))
                    ws = W(s)
                    if ws:
                        gp.append(sum(w in pre for w in ws) / len(ws)); ga.append(sum(w in allw for w in ws) / len(ws))
    if not c["n"]:
        print(f"{label}: no data"); continue
    p95 = sorted(lens)[int(0.95 * len(lens))]
    print(f"{label:22s} inc={len(pids)} cells={cells} samples={c['n']} chars/sample={statistics.mean(lens):.0f} "
          f"(median {statistics.median(lens):.0f}, p95 {p95}) lines/sample={statistics.mean(lines):.2f} "
          f"empty={100*c['empty']/c['n']:.2f}% tag-residue={100*c['tag']/c['n']:.2f}% "
          f"cells-with-dup={100*dup/cells:.2f}% distinct/k={statistics.mean(dist):.3f} "
          f"grounded-in-prefix={100*statistics.mean(gp):.1f}% in-conversation={100*statistics.mean(ga):.1f}%", flush=True)
