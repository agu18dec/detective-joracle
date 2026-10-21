"""Phase 3b: how agents use the readout tools (no LLM). Per reader episode (d0-d4):
  n_search, n_inspect, n_readout_calls, n_positions served (search hits + inspected), decision-window touched
  (any served position within 12 tokens before .. 4 into the response), share of searched patterns that are
  'hypothesis-confirming' (contain words from the committed hypothesis), output tokens, pass.
Writes data/p3b_episodes.csv and prints arm x pass/fail means.
"""
import json, re, sys, collections, concurrent.futures as cf
sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from dcommon import *

ARMS = ["olens_rl_B", "mlnla_B", "jlens_B"]
STOP = set("the a an of to and or in on for is it that this with as by be model its from at are was not but".split())

def resp_start(b, pid, _c={}):
    if b not in _c:
        _c[b] = {}
        for r in map(json.loads, open(C / "datasets" / b / "data/tokens.jsonl")):
            t = r["turn_of"]; la = max(i for i, (tt, k) in enumerate(t) if k == "assistant"); lt = t[la][0]
            _c[b][r["prompt_id"]] = next(i for i, (tt, k) in enumerate(t) if tt == lt and k == "assistant")
    return _c[b][pid]

def feats(args):
    b, a, f = args
    try: r = json.loads(f.read_text())
    except Exception: return None
    if str(r.get("stopped_by", "")).startswith(("error", "salvaged")) or not r.get("judge"): return None
    pid = f.name.split(".d")[0]; d = int(f.name.split(".d")[1][:-5]); rs = resp_start(b, pid)
    dwin = set(range(rs - 12, rs + 4))
    commit = ((r.get("committed_text_hypothesis") or {}).get("hypothesis") or "").lower()
    cw = {w for w in re.findall(r"[a-z]{4,}", commit) if w not in STOP}
    ns = ni = 0; served = set(); conf = 0; npat = 0; first_ro = None
    for i, e in enumerate(r.get("events", [])):
        if e["tool"] in ("search", "inspect"):
            if first_ro is None: first_ro = i
            ps = e.get("positions") or []; served |= set(ps)
            if e["tool"] == "search":
                ns += 1; pat = str((e.get("args") or {}).get("pattern", "")).lower()
                pw = set(re.findall(r"[a-z]{4,}", pat)); npat += 1
                conf += bool(pw & cw)
            else: ni += 1
    return dict(batch=b, arm=a, pid=pid, draw=d, passed=int(r["judge"]["score"] >= 7), score=r["judge"]["score"],
                n_search=ns, n_inspect=ni, n_readout=ns + ni, n_served=len(served), dwin=int(bool(served & dwin)),
                post_resp=int(any(p >= rs for p in served)), confirm_share=conf / npat if npat else None,
                out_tokens=r.get("output_tokens"), n_calls=len(r.get("events", [])))

if __name__ == "__main__":
    jobs = [(b, a, f) for b in BATCHES for a in ARMS for d in range(5) for f in (C / "harness_runs" / b / a).glob(f"*.d{d}.json")]
    for b in BATCHES: resp_start(b, next(iter(gt(b))))
    with cf.ThreadPoolExecutor(48) as ex: rows = [x for x in ex.map(feats, jobs) if x]
    write_rows("p3b_episodes.csv", rows)
    g = collections.defaultdict(list)
    for x in rows: g[(x["arm"], x["passed"])].append(x)
    keys = ["n_search", "n_inspect", "n_served", "dwin", "post_resp", "confirm_share", "out_tokens", "n_calls"]
    print("arm pass n " + " ".join(keys))
    for k in sorted(g):
        xs = g[k]
        print(k[0], k[1], len(xs), " ".join(f"{sum(x[m] for x in xs if x[m] is not None)/max(1,sum(x[m] is not None for x in xs)):.2f}" for m in keys))
