"""Phase 4a: open-ended hypothesis generation WITHOUT the agent loop. Does a reader's readouts let a model propose the
right causes (and fewer refuted ones)?

Per incident and condition (transcript | olens | mlnla), 2 samples:
  generate: Opus sees the conversation, the response, the behaviour and (reader conditions) EVERY readout line at the
            decision window (12 tokens before the response .. 4 into it; 2 samples per cell, all 11 layers, deduplicated,
            labelled t<pos> L<layer>) and writes 5 distinct ranked hypotheses (one sentence each, a specific feature +
            mechanism). Readouts are described as noisy single-token decodings; no ground truth anywhere.
  grade   : judge v3's claim grader (judge3.CLS_SYS, arm-blind) labels each hypothesis TRUE(Pi)/RIVAL(Rj)/CONTRADICTED/
            UNSUPPORTED/VAGUE against CHIVE's ground truth.
Metrics: match@1 (hyp 1 is TRUE), match@5 (any TRUE), recall / non-obvious recall / primary recall over distinct points,
rivals proposed (share of refuted rivals named), precision over specific hypotheses.
Out: diag/hypgen/<b>/<cond>/<pid>.s<k>.json ; report with --report
usage: python p4a_hypgen.py <batch> [--conds transcript,olens,mlnla] [--samples 2] [--limit N]
"""
import gzip, json, os, sys, random, pathlib, collections, concurrent.futures as cf
C = pathlib.Path("/workspace/agam/chive_scale")
BANKS = {"olens": "s3d_rl600_all", "mlnla": "mlnla_k4_chive"}
GEN_MODEL = "claude-opus-5"; GRADE_MODEL = "claude-opus-5"
GEN_SYS = """You are an interpretability researcher. You see a conversation with a language model (the target model), its
response, and a behaviour in that response that recurs in a fraction of re-runs. Propose the 5 most likely causes:
specific features of the conversation (an instruction or its wording, a phrase, a persona, a format, something an earlier
turn set up, a default the model applies when something is left open) and how each drives the behaviour. Make them
DISTINCT causes (not rephrasings), ranked most likely first, one or two sentences each.{readouts}
Output JSON {{"hypotheses": ["...", "...", "...", "...", "..."]}}."""
RO_NOTE = """
You also get READOUTS: natural-language decodings of the target model's internal activations at single tokens just before
(and at the start of) its response, several samples per token and layer. They are noisy: single lines can borrow details
from similar contexts; lines that merely restate the visible text or preview the response's own wording say WHAT the model
does, not WHY. Content that recurs and that the text does not state can reveal how the model reads the request
(inferences, framings, assumptions, plans, judgements). Use them where they help; ignore them where they don't."""

def oauth():
    sys.path.insert(0, str(C)); import anthropic_oauth; anthropic_oauth.install(); import anthropic
    return anthropic, anthropic.Anthropic(max_retries=6)

def ask(an, cli, model, system, user, max_tokens=2500):
    for _ in range(4):
        try:
            m = cli.messages.create(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
            t = "".join(x.text for x in m.content if x.type == "text"); return json.loads(t[t.find("{"): t.rfind("}") + 1])
        except an.AuthenticationError: raise SystemExit("401")
        except Exception: pass

def cached(path, fn):
    if path.exists(): return json.loads(path.read_text())
    r = fn()
    if r is not None:
        path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False)); os.replace(tmp, path)
    return r

def window_readouts(bank, pid, rs, per_cell=2):
    rec = json.load(gzip.open(C / "banks" / bank / f"{pid}.json.gz", "rt"))["readouts"]
    seen, out = set(), []
    for pos in range(max(0, rs - 12), rs + 4):
        for layer in sorted(rec, key=int):
            for s in (rec[layer].get(str(pos)) or [])[:per_cell]:
                for ln in str(s).split("\n"):
                    ln = ln.strip().lstrip("-•* ").strip()
                    if len(ln) >= 8 and ln not in seen:
                        seen.add(ln); out.append(f"t{pos} L{layer}: {ln[:260]}")
    return out

def main(b, conds, S, limit):
    an, cli = oauth()
    sys.argv = [sys.argv[0], b]  # judge3 parses argv at import; give it the batch
    sys.path.insert(0, str(C / "analysis" / "diag")); import judge3 as J3
    D = C / "datasets" / b
    INC = {r["prompt_id"]: r for r in map(json.loads, open(D / "data/incidents.jsonl"))}
    TOK = {r["prompt_id"]: r["turn_of"] for r in map(json.loads, open(D / "data/tokens.jsonl"))}
    ids = sorted(p.stem for p in (C / "points" / b / "points").glob("*.json"))[: limit or None]
    def rstart(t):
        la = max(i for i, (tt, k) in enumerate(t) if k == "assistant"); lt = t[la][0]
        return next(i for i, (tt, k) in enumerate(t) if tt == lt and k == "assistant")
    def one(job):
        pid, cond, k = job
        out = C / "diag" / "hypgen" / b / cond / f"{pid}.s{k}.json"
        if out.exists(): return
        inc = INC[pid]
        conv = "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in inc["messages"])
        if len(conv) > 24000: conv = conv[:12000] + "\n[...]\n" + conv[-12000:]
        u = f"CONVERSATION:\n{conv}\n\n[RESPONSE]: {inc['completion']}\n\nBEHAVIOUR ({inc['k']}/{inc['n']} re-runs): {inc['behaviour']}"
        if cond != "transcript":
            lines = window_readouts(BANKS[cond], pid, rstart(TOK[pid]))
            random.Random(f"{pid}{k}").shuffle(lines)
            u += "\n\nREADOUTS (decision window, shuffled order):\n" + "\n".join(lines[:900])
        g = ask(an, cli, GEN_MODEL, GEN_SYS.format(readouts=RO_NOTE if cond != "transcript" else ""), u)
        hyps = [str(h) for h in (g or {}).get("hypotheses", [])][:5]
        if len(hyps) < 3: return
        cl = J3.classify(pid, hyps)
        if cl is None: return
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"hypotheses": hyps, "labels": cl["labels"], "npts": cl["npts"], "nrv": cl["nrv"]}, ensure_ascii=False))
    jobs = [(p, c, k) for p in ids for c in conds for k in range(S)]
    random.Random(0).shuffle(jobs)
    with cf.ThreadPoolExecutor(int(os.environ.get("HG_THREADS", "12"))) as ex: list(ex.map(one, jobs))
    print(f"HYPGEN {b} DONE {len(jobs)}", flush=True)

def report():
    sys.path.insert(0, str(C / "report")); sys.path.insert(0, str(C / "analysis" / "diag"))
    from common import boot, strata, in_stratum, STRATA
    from dcommon import write_rows
    from p1b_report import flags
    import numpy as np
    M = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for b in ["fresh1", "fresh2", "fresh3", "fresh4"]:
        for cd in (C / "diag" / "hypgen" / b).glob("*"):
            for f in cd.glob("*.json"):
                pid = f.name.split(".s")[0]; j = json.loads(f.read_text())
                lab = [str(x.get("label", "")).upper() for x in j["labels"]]; mat = [str(x.get("match") or "") for x in j["labels"]]
                tp = {m for l, m in zip(lab, mat) if l == "TRUE"}; rv = {m for l, m in zip(lab, mat) if l == "RIVAL"}
                spec = sum(l in ("TRUE", "RIVAL", "CONTRADICTED", "UNSUPPORTED") for l in lab)
                ob, prim = flags(b, pid); m = M[(b, pid)][cd.name]
                m["match1"].append(float(lab[0] == "TRUE")); m["match5"].append(float("TRUE" in lab))
                m["recall"].append(len(tp) / j["npts"]) if j["npts"] else None
                if ob and not all(ob):
                    no = {f"P{i + 1}" for i, o in enumerate(ob) if not o}; m["recall_nonobv"].append(len(tp & no) / len(no))
                    m["any_nonobv"].append(float(bool(tp & no)))
                if prim and any(prim):
                    pr = {f"P{i + 1}" for i, t in enumerate(prim) if t}; m["primary@5"].append(float(bool(tp & pr)))
                    m["primary@1"].append(float(lab[0] == "TRUE" and mat[0] in pr))
                if j["nrv"]: m["rivals_named"].append(len(rv) / j["nrv"])
                if spec: m["precision"].append(lab.count("TRUE") / spec); m["unsupported"].append(lab.count("UNSUPPORTED") / spec)
    S = strata(); rows = []
    mets = ["match1", "match5", "primary@1", "primary@5", "recall", "recall_nonobv", "any_nonobv", "rivals_named", "precision", "unsupported"]
    for st in STRATA:
        for met in mets:
            for a, ref in [("olens", "transcript"), ("mlnla", "transcript"), ("olens", "mlnla")]:
                ks = [k for k in in_stratum(M.keys(), S, st) if M[k][a].get(met) and M[k][ref].get(met)]
                v, r, d, lo, hi = boot([np.mean(M[k][a][met]) for k in ks], [np.mean(M[k][ref][met]) for k in ks])
                rows.append(dict(stratum=st, metric=met, arm=a, ref=ref, n=len(ks), arm_mean=v, ref_mean=r, diff=d, lo=lo, hi=hi))
                if st == "overall" or met in ("recall_nonobv", "primary@1"):
                    print(f"{st[:12]:12s} {met:14s} {a:6s}-{ref:10s} n={len(ks):4d} {v:.3f} vs {r:.3f} d={d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    write_rows("p4a_hypgen.csv", rows)

if __name__ == "__main__":
    if sys.argv[1] == "--report": report()
    else:
        opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
        main(sys.argv[1], opt("--conds", "transcript,olens,mlnla").split(","), int(opt("--samples", "2")), int(opt("--limit", "0")))
