"""Phase 2 v2: bank ceiling with the audit's fixes (analysis/diag/ceiling_audit.md: v1 hit precision was ~13% OLens / ~0% NLA).

Changes vs p2_ceiling.py (v1):
  1. STATABLE points only. A Sonnet call per incident rewrites each true point / refuted rival into the mechanism clause a
     model could represent WHILE READING the conversation (a reading, inference, assumption, judgement, plan), or marks it
     not statable (pure intervention outcomes like "removing X eliminates Y"). Only statable items are scored.
  2. PROMPT SIDE only: candidate lines come from positions before the response (pos < response_start + 2); no response-side
     lines (v1: 53% of OLens hits were the response being written).
  3. STRICT per-line labels by Opus: VALID / ECHO (restates visible text) / FORECAST (predicts the response's wording) /
     TOPICAL / CONTRADICTS / HALLUCINATED (entities or facts not in the conversation). Only VALID counts. The judge sees the
     conversation, the first 600 chars of the response, and the item.
  4. Same pipeline on shuffled banks (another incident's bank) for the false-hit floor.
Retrieval as v1 (TF-IDF over all prompt-side lines with the mechanism + 6 paraphrases as queries; top 30 + 15 decision-window).
Sample: 80 random incidents per batch (seed 0) = 320.
Out: diag/ceiling2/<b>/<bank>/<pid>.json
usage: python p2b_ceiling2.py <batch> [--banks olens,mlnla,olens_shuf,mlnla_shuf] [--n 80] [--procs 4]
"""
import gzip, json, os, sys, pathlib, random, concurrent.futures as cf
sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from p2_ceiling import BANKS, QX_SYS, oauth_client, ask, cached, bank_lines

C = pathlib.Path("/workspace/agam/chive_scale")
STAT_SYS = """Each numbered statement explains why a language model behaved as it did; some are phrased as experiment outcomes.
For each, give the MECHANISM a model could represent internally WHILE READING the conversation, before or as it starts
responding: how it reads or interprets something, an inference about the user or task, an assumption or default it applies,
a judgement, or a plan. Write it as one short sentence about this conversation ("the model reads 'private devices' as
personal hardware"). If the statement is ONLY an intervention outcome or a frequency claim with no such mechanism
(e.g. "removing X largely eliminates Y", "a plain request rarely triggers Z"), set statable false.
Output JSON {"items": {"<id>": {"statable": true|false, "mechanism": "<sentence or empty>"}}}."""
JUDGE2_SYS = """You check whether single readout lines (natural-language decodings of a language model's activations at one token
of the conversation, read BEFORE its response) express a given MECHANISM. You see the conversation (clipped), the start of
the response, the mechanism, and numbered lines with their token position. Label EVERY line:
- VALID: states or clearly implies the mechanism as the model's own reading/inference/assumption/judgement/plan about THIS
  conversation, with content not merely copied from the visible text.
- ECHO: restates or paraphrases text visible in the conversation (including instructions) without the mechanism's content.
- FORECAST: predicts the wording of the response or of the next text, without the mechanism's content.
- TOPICAL: same topic, but not the mechanism.
- CONTRADICTS: points the opposite way.
- HALLUCINATED: relies on entities, numbers or facts that are not in the conversation.
Be strict: VALID needs the mechanism itself, not keywords. Output JSON {"labels": {"<n>": "VALID|ECHO|FORECAST|TOPICAL|CONTRADICTS|HALLUCINATED"}}."""

def work(args):
    b, pid, banks, partner = args
    anthropic, cli = oauth_client()
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    D = C / "datasets" / b
    g = next(r for r in map(json.loads, open(D / "data/ground_truth.jsonl")) if r["prompt_id"] == pid)
    inc = next(r for r in map(json.loads, open(D / "data/incidents.jsonl")) if r["prompt_id"] == pid)
    tok = next(r for r in map(json.loads, open(D / "data/tokens.jsonl")) if r["prompt_id"] == pid)
    t = tok["turn_of"]; la = max(i for i, (tt, k) in enumerate(t) if k == "assistant"); lt = t[la][0]
    rs = next(i for i, (tt, k) in enumerate(t) if tt == lt and k == "assistant")
    dwin = set(range(max(0, rs - 12), rs + 2))
    pts = json.loads((C / "points" / b / "points" / f"{pid}.json").read_text())["points"]
    plain = [json.loads(p.read_text())["point_scores"] for p in sorted((C / "points" / b / "grades/plain").glob(f"{pid}.d*.json"))[:3]]
    obvious = [sum(x[i] >= 7 for x in plain) >= 2 for i in range(len(pts))] if len(plain) == 3 and all(len(x) == len(pts) for x in plain) else [None] * len(pts)
    items = [dict(id=f"P{i + 1}", kind="true", obvious=obvious[i], text=p) for i, p in enumerate(pts)]
    items += [dict(id=f"R{j + 1}", kind="rival", obvious=None, text=x["cause"])
              for j, x in enumerate(x for x in g["ruled_out"] if x.get("kind") == "refuted" and x.get("cause"))]
    base = C / "diag" / "ceiling2" / b
    st = cached(base / "_statable" / f"{pid}.json", lambda: ask(cli, anthropic, STAT_SYS,
                f"CONVERSATION (clipped):\n{json.dumps(inc['messages'])[:6000]}\n\nBEHAVIOUR: {inc['behaviour']}\n\n" +
                "\n".join(f"{it['id']}: {it['text']}" for it in items)))
    if not st: return pid, "nostat"
    for it in items:
        s = (st.get("items") or {}).get(it["id"]) or {}
        it["statable"] = bool(s.get("statable")) and bool(str(s.get("mechanism") or "").strip())
        it["mechanism"] = str(s.get("mechanism") or "")
    sitems = [it for it in items if it["statable"]]
    qx = cached(base / "_queries" / f"{pid}.json", lambda: ask(cli, anthropic, QX_SYS,
                "\n".join(f"{it['id']}: {it['mechanism']}" for it in sitems))) if sitems else {"q": {}}
    qx = qx or {"q": {}}
    conv = "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in inc["messages"])
    conv = (conv[:3500] + "\n[...]\n" + conv[-3500:]) if len(conv) > 7500 else conv
    conv += f"\n\n[RESPONSE START]: {inc['completion'][:600]}"
    for bk in banks:
        outf = base / bk / f"{pid}.json"
        if outf.exists(): continue
        try: lines, _ = bank_lines(BANKS[bk], pid, partner)
        except FileNotFoundError: continue
        lines = [l for l in lines if l[0] < rs + 2]
        texts = list(dict.fromkeys(l[2] for l in lines)); first = {}
        for p_, L_, t_ in lines: first.setdefault(t_, (p_, L_))
        indw = {t_ for p_, L_, t_ in lines if p_ in dwin}
        X = vec = None
        if texts and sitems:
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", max_features=300000)
            X = vec.fit_transform(texts)
        from collections import Counter
        def judge_item(it):
            if not it["statable"] or not texts:
                return dict(it, hits=[], labels={}, n_cand=0)
            qs = [it["mechanism"]] + list((qx.get("q") or {}).get(it["id"], []))[:6]
            sc = (X @ vec.transform(qs).T).max(axis=1).toarray().ravel(); order = np.argsort(-sc)
            cand = [texts[i] for i in order[:30] if sc[i] > 0]
            cand += [texts[i] for i in order if texts[i] in indw and texts[i] not in cand][:15]
            if not cand:
                return dict(it, hits=[], labels={}, n_cand=0)
            u = (f"CONVERSATION:\n{conv}\n\nMECHANISM: {it['mechanism']}\n\nLINES:\n"
                 + "\n".join(f"{i + 1}. [t{first[c][0]}{' (just before the response)' if first[c][0] in dwin else ''}] {c[:300]}" for i, c in enumerate(cand)))
            j = None
            for _ in range(2):
                j = ask(cli, anthropic, JUDGE2_SYS, u, 1500)
                if j and isinstance(j.get("labels"), dict): break
            if not j or not isinstance(j.get("labels"), dict):
                return dict(it, hits=None, labels={}, n_cand=len(cand))
            labs = {int(k): str(v).upper() for k, v in j["labels"].items() if str(k).isdigit() and 0 < int(k) <= len(cand)}
            hits = [dict(pos=first[cand[n - 1]][0], layer=first[cand[n - 1]][1], line=cand[n - 1], dwin=cand[n - 1] in indw)
                    for n, l in sorted(labs.items()) if l == "VALID"]
            return dict(it, hits=hits, labels=dict(Counter(labs.values())), n_cand=len(cand))
        with cf.ThreadPoolExecutor(6) as tp:
            res = list(tp.map(judge_item, items))
        outf.parent.mkdir(parents=True, exist_ok=True)
        outf.write_text(json.dumps(dict(prompt_id=pid, bank=BANKS[bk], partner=partner if "shuf" in bk else None,
                                        resp_start=rs, items=res), ensure_ascii=False))
    return pid, "ok"

if __name__ == "__main__":
    import p2_ceiling as P1
    P1.MODEL = "claude-opus-5"  # ask() reads the module global: Opus for the statable rewrite, queries and the strict judge
    b = sys.argv[1]; opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    banks = opt("--banks", "olens,mlnla,olens_shuf,mlnla_shuf").split(",")
    ids = sorted(p.stem for p in (C / "points" / b / "points").glob("*.json"))
    rng = random.Random(0); sample = sorted(rng.sample(ids, min(int(opt("--n", "80")), len(ids))))
    partners = sample[:]; random.Random(b).shuffle(partners)
    pairs = [(b, p, banks, q if q != p else partners[(i + 1) % len(partners)]) for i, (p, q) in enumerate(zip(sample, partners))]
    n = 0
    with cf.ProcessPoolExecutor(int(opt("--procs", "4"))) as ex:
        for pid, stt in ex.map(work, pairs):
            n += 1
            if stt != "ok" or n % 10 == 0: print(f"[{b}] {n}/{len(pairs)} {pid} {stt}", flush=True)
    print(f"CEILING2 {b} DONE", flush=True)
