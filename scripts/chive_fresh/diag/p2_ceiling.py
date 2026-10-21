"""Phase 2a: bank ceiling. Does a reader's readout bank STATE each CHIVE point / refuted rival anywhere, regardless of
whether the agent found it? Same retrieval + judge for every bank, so banks are comparable.

Per incident:
  items   = every true point (points/<b>/points, obvious flag from plain answers) + every refuted rival (ground truth)
  queries = Sonnet writes 6 short paraphrases / keyword sets per item (cached, shared across banks) so lexical retrieval
            also finds NLA-style or terse wording
  lines   = every readout line in the bank: (pos, layer, sample, line) with OLens bullets / NLA lines split on newlines
  retrieve: word TF-IDF (1-2 grams) over all lines, max cosine over the item's queries -> top 30 distinct lines,
            + up to 15 distinct lines from the decision window (12 tokens before the response .. 4 into it), by the same
            score -> <= 45 candidate lines
  judge   : Sonnet sees the behaviour, the conversation (clipped), the item and the numbered lines, and returns the lines
            that state the item's proposition as the model's own consideration (not merely quoting the visible text)
Output: diag/ceiling/<b>/<bank>/<pid>.json = {items:[{id,kind,obvious,text,hits:[{pos,layer,line}],n_cand}], ...}

usage: python p2_ceiling.py <batch> [--banks olens,mlnla,jlens] [--limit N] [--procs 12]
"""
import gzip, json, os, re, sys, pathlib, random, concurrent.futures as cf
C = pathlib.Path("/workspace/agam/chive_scale")
BANKS = {"olens": "s3d_rl600_all", "mlnla": "mlnla_k4_chive", "jlens": "jlens_L20-60",
         "olens_shuf": "s3d_rl600_all@shuf", "mlnla_shuf": "mlnla_k4_chive@shuf"}
MODEL = "claude-sonnet-5"
QX_SYS = ("For each numbered statement about why a language model behaves as it does, write 6 short alternative phrasings "
          "a reader of the model's internal 'thoughts' might use: terse paraphrases, keyword lists, and how the model itself "
          "might put it in first person (e.g. 'the user wants X so I should Y'). Output JSON {\"q\": {\"<id>\": [6 strings]}}.")
JUDGE_SYS = ("You check whether readouts of a language model's internal states express a given proposition. You see the "
             "conversation (clipped), the behaviour, a PROPOSITION (a hypothesised cause of the behaviour) and numbered readout "
             "lines (each decoded from the model's activations at one token). List the lines that state or clearly imply the "
             "proposition as something the model itself is considering, believing, planning or reacting to. Do NOT count "
             "lines that merely quote or paraphrase text visible in the conversation without the causal/evaluative content, "
             "or lines that only share the topic. Output JSON {\"hits\": [line numbers], \"note\": \"<10 words\"}.")

def oauth_client():
    sys.path.insert(0, str(C)); import anthropic_oauth; anthropic_oauth.install()
    import anthropic; return anthropic, anthropic.Anthropic(max_retries=6)

def ask(cli, anthropic, system, user, max_tokens=2500):
    """JSON reply. Budget is raised to >= 8000 because thinking can use the whole budget (stop=max_tokens with empty
    text); on a safety refusal it retries once with Sonnet, else returns None."""
    model = MODEL
    for attempt in range(4):
        try:
            m = cli.messages.create(model=model, max_tokens=max(8000, max_tokens), system=system, messages=[{"role": "user", "content": user}])
            t = "".join(x.text for x in m.content if x.type == "text")
            if "{" not in t:
                print(f"[ask] no JSON: model={model} stop={m.stop_reason} text={t[:120]!r}", flush=True)
                if m.stop_reason == "refusal":
                    if model == "claude-sonnet-5": return None
                    model = "claude-sonnet-5"
                continue
            return json.loads(t[t.find("{"): t.rfind("}") + 1])
        except anthropic.AuthenticationError:
            raise SystemExit("401")
        except Exception as e:
            print(f"[ask] {type(e).__name__}: {str(e)[:200]}", flush=True)
    return None

def cached(path, fn):
    if path.exists(): return json.loads(path.read_text())
    r = fn()
    if r is not None:
        path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False)); os.replace(tmp, path)
    return r

def bank_lines(bank, pid, partner):
    name, shuf = (bank.split("@") + [None])[:2]
    src = partner if shuf else pid
    rec = json.load(gzip.open(C / "banks" / name / f"{src}.json.gz", "rt"))
    out = []
    for layer, per in rec["readouts"].items():
        for pos, samples in per.items():
            for si, s in enumerate(samples):
                for ln in str(s).split("\n"):
                    ln = ln.strip().lstrip("-•* ").strip()
                    if len(ln) >= 8: out.append((int(pos), int(layer), ln))
    return out, rec.get("n_tokens", 0)

def work(args):
    b, pid, banks, partner = args
    anthropic, cli = oauth_client()
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    D = C / "datasets" / b
    gt = next(r for r in map(json.loads, open(D / "data/ground_truth.jsonl")) if r["prompt_id"] == pid)
    inc = next(r for r in map(json.loads, open(D / "data/incidents.jsonl")) if r["prompt_id"] == pid)
    tok = next(r for r in map(json.loads, open(D / "data/tokens.jsonl")) if r["prompt_id"] == pid)
    turns = tok["turn_of"]; last_t = turns[-1][0] if turns[-1][1] != "control" else max(t for t, k in turns if k == "assistant")
    rs = next(i for i, (t, k) in enumerate(turns) if t == last_t and k == "assistant")
    dwin = set(range(max(0, rs - 12), rs + 4))
    pts = json.loads((C / "points" / b / "points" / f"{pid}.json").read_text())["points"]
    plain = [json.loads(p.read_text())["point_scores"] for p in sorted((C / "points" / b / "grades/plain").glob(f"{pid}.d*.json"))[:3]]
    obvious = [sum(x[i] >= 7 for x in plain) >= 2 for i in range(len(pts))] if len(plain) == 3 else [None] * len(pts)
    items = [dict(id=f"P{i + 1}", kind="true", obvious=obvious[i], text=p) for i, p in enumerate(pts)]
    items += [dict(id=f"R{j + 1}", kind="rival", obvious=None, text=x["cause"])
              for j, x in enumerate(x for x in gt["ruled_out"] if x.get("kind") == "refuted" and x.get("cause"))]
    qx = cached(C / "diag" / "ceiling" / b / "_queries" / f"{pid}.json", lambda: ask(cli, anthropic, QX_SYS,
                "\n".join(f"{it['id']}: {it['text']}" for it in items)))
    if not qx: return pid, "noqx"
    conv = "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in inc["messages"])
    conv = (conv[:3000] + "\n[...]\n" + conv[-3000:]) if len(conv) > 6500 else conv
    conv += f"\n\n[ASSISTANT RESPONSE]: {inc['completion'][:1500]}"
    for bk in banks:
        outf = C / "diag" / "ceiling" / b / bk / f"{pid}.json"
        if outf.exists(): continue
        try: lines, _ = bank_lines(BANKS[bk], pid, partner)
        except FileNotFoundError: continue
        texts = list(dict.fromkeys(l[2] for l in lines))
        first = {}
        for p_, L_, t_ in lines:
            first.setdefault(t_, (p_, L_))
        indw = {t_ for p_, L_, t_ in lines if p_ in dwin}
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english", max_features=300000)
        X = vec.fit_transform(texts)
        res = []
        for it in items:
            qs = [it["text"]] + list((qx.get("q") or {}).get(it["id"], []))[:6]
            sc = (X @ vec.transform(qs).T).max(axis=1).toarray().ravel()
            order = np.argsort(-sc)
            cand = [texts[i] for i in order[:30] if sc[i] > 0]
            cand += [texts[i] for i in order if texts[i] in indw and texts[i] not in cand][:15]
            if not cand:
                res.append(dict(it, hits=[], n_cand=0)); continue
            u = (f"CONVERSATION:\n{conv}\n\nBEHAVIOUR: {inc['behaviour']}\n\nPROPOSITION: {it['text']}\n\nREADOUT LINES:\n"
                 + "\n".join(f"{i + 1}. {t[:300]}" for i, t in enumerate(cand)))
            j = ask(cli, anthropic, JUDGE_SYS, u, 800)
            if j is None:
                res.append(dict(it, hits=None, n_cand=len(cand), note="judgefail")); continue
            hits = [int(h) - 1 for h in j.get("hits", []) if str(h).isdigit() and 0 < int(h) <= len(cand)]
            res.append(dict(it, hits=[dict(pos=first[cand[h]][0], layer=first[cand[h]][1], line=cand[h],
                                           dwin=cand[h] in indw) for h in hits], n_cand=len(cand), note=j.get("note")))
        outf.parent.mkdir(parents=True, exist_ok=True)
        outf.write_text(json.dumps(dict(prompt_id=pid, bank=BANKS[bk], partner=partner if "shuf" in bk else None,
                                        resp_start=rs, items=res), ensure_ascii=False))
    return pid, "ok"

if __name__ == "__main__":
    b = sys.argv[1]; opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    banks = opt("--banks", "olens,mlnla,jlens").split(",")
    ids = sorted(p.stem for p in (C / "points" / b / "points").glob("*.json"))
    rng = random.Random(b); partners = ids[:]; rng.shuffle(partners)
    pairs = [(b, p, banks, q if q != p else partners[0]) for p, q in zip(ids, partners)]
    if "--limit" in sys.argv: pairs = pairs[:int(opt("--limit", "0"))]
    n = 0
    with cf.ProcessPoolExecutor(int(opt("--procs", "12"))) as ex:
        for pid, st in ex.map(work, pairs):
            n += 1
            if st != "ok" or n % 25 == 0: print(f"[{b}] {n}/{len(pairs)} {pid} {st}", flush=True)
    print(f"CEILING {b} DONE", flush=True)
