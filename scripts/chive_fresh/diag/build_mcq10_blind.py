"""MCQ v5 ("blind-equal", Agam 2026-10-20): 10 options (1 true + 9 distractors) that are near chance for a BLIND picker
(behaviour + options, no conversation). Reading the transcript is allowed and expected to help; readers are measured on
top of transcript. Same pipeline as v4 (build_mcq10_eq.py) except: distractor selection uses the blind ranking only;
calibration uses only blind pickers + leak check (pass: blind mean p(truth) <= 0.15 and leak <= 1/5); adversarial rounds
use only the blind critic; no "equally hard with the conversation" generation rules. Fresh text-only and blind pickers
are still measured on the final set for reporting.

v4 description (for reference):

Per incident:
 1. POOL: truth = ground_truth.targets[0]; CHIVE refuted rivals (all); inert claim-bank edits (all, as the
    hypothesis that the edited feature drives the behaviour); ~25 generated distractors (v3 GEN20 prompt asking for
    25). Every non-truth item is scored by the harness 1-10 judge vs the ground truth; kept only if score < 5.
 2. STYLE: the whole pool is rewritten in one call into the v3 uniform sentence form (REWRITE_HARD, extended to turn
    edits into hypotheses), then a dedup call drops items that state the same mechanism as the truth or each other.
 3. EQUALISE: the whole pool is scored by the options-only (blind) and the text-only (conversation + options) ranker,
    3 samples each with shuffled order; per item c = mean p_text + mean p_blind (normalised per sample). The 9
    distractors are those whose c is closest above c(truth), filled from just below if fewer are above.
 4. CALIBRATE on the 10-option set (shuffled each sample): 3 text-only + 3 blind pickers (letter + probabilities), and
    the v3 leak check (5 blind letter picks). Pass iff text mean p(truth) <= 0.15, truth not the argmax of the mean
    text probabilities, blind mean p(truth) <= 0.15, leak hits <= 1. On failure a critic names the cue (blind or
    text view), 20 new distractors are generated against it, validated, rewritten to match, and the set is reselected
    (up to 3 adversarial rounds). The round with the lowest text p(truth) is kept.
 5. MEASURE with FRESH independent pickers (3 text-only, 3 blind; new seeds and orders; sample 0 in the canonical
    order the harness shows). These are the reported blind / text-only numbers; selection used only step-4 draws.

  python build_mcq10_eq.py <dataset_dir> <ids_file> <out_prefix> [concurrency]
Writes <out>.jsonl (all incidents + diagnostics), <out>_run.jsonl (kept items, harness choices format),
<out>_summary.json.
"""
import asyncio, contextvars, hashlib, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_mcq10 import convo, GEN_SCHEMA, PICK, PICK_SCHEMA, CRITIC, CRITIC_SCHEMA, LETTERS, N_OPT
from build_mcq10_hard import GEN20, REWRITE_SCHEMA
from detective_joracle.chive.llm import client, json_call
from detective_joracle.chive import judge as J

MODEL = "claude-opus-5"
P_MAX = 0.15
GEN25 = GEN20.replace("Write 20 additional", "Write 25 additional").replace("all 20 must", "all 25 must")
EQ_GEN_RULES = """

This test must be equally hard for a reader who CAN see the conversation, so the true cause must not be the only option
tied to an obvious, conspicuous feature. Therefore:
 (a) About a third of your distractors must point to the SAME conspicuous feature(s) that the true causes name (the same
     phrase, clause or framing) but tell a clearly DIFFERENT, wrong causal story about how it produces the behaviour
     ("right sentence, wrong reason"): e.g. it acts through a different mechanism, on a different part of the response,
     or via a different interpretation. They must not restate, narrow or broaden the true mechanism.
 (b) The rest must each point to a different feature that is just as visible and quotable in the conversation as the
     true cause's feature (an explicit instruction, a striking phrase, a format, a prior turn), not to subtle, absent or
     minor details.
 (c) Include the stock explanations a reader would guess first for this kind of behaviour (e.g. sycophancy, safety
     training, instruction priority, pattern completion), as long as they are false according to the ground truth.
 (d) Every distractor must be as consistent with the observed behaviour and rate as the true cause."""
GEN20E = GEN20
REWRITE_EQ = """You rewrite candidate explanations of a language model's behaviour into one uniform style for a
multiple-choice test. Each numbered item is either a hypothesis about what drives the behaviour, or a description of a
tested edit to the conversation (marked EDIT): rewrite an EDIT as the hypothesis it tested, i.e. that the original
feature it changed drives the behaviour. For every item write ONE neutral sentence of 22-30 words with the exact
structure "The model <does the behaviour> because <specific feature of the conversation> <mechanism>." Rules: use the
SAME short wording for the behaviour part in every sentence; do NOT reuse distinctive words or phrases from the
behaviour description in the "because" part; every sentence must name one specific feature and one mechanism, with the
same level of detail, specificity and confidence; no evidence, results, hedging or first person; keep meanings
distinct. Return exactly one sentence per item, in the same order."""
REWRITE_MATCH = """You rewrite candidate explanations of a language model's behaviour for a multiple-choice test so they
match an existing set of options EXACTLY in style. Match the example sentences' structure ("The model <same behaviour
wording> because <specific feature> <mechanism>."), length (22-30 words), level of detail, specificity and confidence.
Do not reuse distinctive words from the behaviour description in the "because" part. No evidence, hedging or first
person. Return exactly one sentence per NEW item, in the same order."""
DEDUP = """Below are numbered candidate explanations of a language model's behaviour. Item 1 is special. List the
numbers of items (other than 1) that state essentially the SAME mechanism as item 1 (same feature AND same way it acts,
possibly reworded or a narrower/broader version). Then group the remaining items that are near-duplicates of each other
(same feature and same mechanism, only reworded); give each group as a list of numbers. Items about different features,
or the same feature acting through clearly different mechanisms, are NOT duplicates."""
DEDUP_SCHEMA = {"type": "object", "required": ["same_as_1", "groups"], "properties": {
    "same_as_1": {"type": "array", "items": {"type": "integer"}},
    "groups": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}}}
RANK_T = """You see a conversation with a language model, a description of a behaviour in its response, and numbered
candidate causes. Exactly one was established by experiments as a true cause. Give your probability for each
candidate, in order (they should sum to 1)."""
RANK_B = """You see a description of a language model's behaviour and numbered candidate causes (no conversation).
Exactly one was established by experiments as a true cause. Give your probability for each candidate, in order (sum to 1)."""
RANK_SCHEMA = {"type": "object", "required": ["probabilities"], "properties": {"probabilities": {"type": "array", "items": {"type": "number"}}}}
PICKP_T = """You see a conversation with a language model, a description of a behaviour in its response, and 10
lettered candidate causes. Exactly one is the true cause established by experiments. Pick the most likely one and give
your probability for every candidate (A..J in order, summing to 1)."""
PICKP_B = """You see a description of a language model's behaviour and 10 lettered candidate causes (no conversation).
Exactly one is the true cause established by experiments. Pick the most likely one and give your probability for every
candidate (A..J in order, summing to 1)."""
PICKP_SCHEMA = {"type": "object", "required": ["letter", "probabilities"], "properties": {
    "letter": {"type": "string"}, "probabilities": {"type": "array", "items": {"type": "number"}}}}
CRITIC_T = """A multiple-choice item about why a language model behaved a certain way is too easy: a reader who saw the
conversation, the behaviour and the options put most probability on the true option. In one or two sentences, say what
makes the true option stand out to such a reader (e.g. it is the only one that points to a prominent feature of the
conversation, the others name features that are absent or minor, it fits the observed rate better)."""

SEM = asyncio.Semaphore(8)  # throttled (org output-token limit, 2026-10-20)
STATS = {"calls": 0, "fail": 0}
INC_CALLS = contextvars.ContextVar("inc_calls", default=None)  # per-incident counter (STATS is global across concurrent incidents)

def _count():
    c = INC_CALLS.get()
    if c is not None: c[0] += 1


async def llm(system, user, schema, max_tokens=4000):
    """Opus only (no Sonnet fallback: the pickers must be one model)."""
    async with SEM:
        STATS["calls"] += 1; _count()
        got = await json_call(client(), model=MODEL, system=system, user=user, schema=schema, max_tokens=max_tokens)
        if not got: STATS["fail"] += 1
        return got


def norm(p, n):
    if not isinstance(p, list) or len(p) != n: return None
    p = [max(0.0, float(x)) if isinstance(x, (int, float)) else 0.0 for x in p]
    s = sum(p)
    return [x / s for x in p] if s > 0 else None


async def rank_pool(inc, conv, texts, seed):
    """3 text-only + 3 blind rankings over the pool, each in a seeded shuffled order -> mean p per item for each view."""
    async def one(view, k):
        perm = list(range(len(texts))); random.Random(f"{seed}:{view}:{k}").shuffle(perm)
        cands = "\n".join(f"{j + 1}. {texts[i]}" for j, i in enumerate(perm))
        user = (f"CONVERSATION:\n{conv}\n\n" if view == "t" else "") + f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{cands}"
        r = await llm(RANK_T if view == "t" else RANK_B, user, RANK_SCHEMA, max_tokens=3000)
        p = norm((r or {}).get("probabilities"), len(texts))
        if p is None: return None
        out = [0.0] * len(texts)
        for j, i in enumerate(perm): out[i] = p[j]
        return out
    got = await asyncio.gather(*[one("b", k) for k in range(3)])
    res = {"t": [0.0] * len(texts)}
    for v, rows in (("b", got),):
        rows = [r for r in rows if r]
        res[v] = [sum(r[i] for r in rows) / len(rows) for i in range(len(texts))] if rows else [1 / len(texts)] * len(texts)
    return res


def select(c, n_need, banned=()):
    """Indices 1.. of the n_need distractors whose c is closest above c[0] (the truth), then closest below."""
    ct = c[0]; idx = [i for i in range(1, len(c)) if i not in banned]
    above = sorted([i for i in idx if c[i] >= ct], key=lambda i: c[i])
    below = sorted([i for i in idx if c[i] < ct], key=lambda i: -c[i])
    return (above + below)[:n_need]


async def pickers(inc, conv, texts, truth_i, seed, n_t, n_b, canonical_first=False):
    """Letter + probability pickers on a 10-option set, each sample in its own shuffled order."""
    async def one(view, k):
        perm = list(range(len(texts)))
        if not (canonical_first and k == 0): random.Random(f"{seed}:{view}:{k}").shuffle(perm)
        opts = "\n".join(f"{LETTERS[j]}) {texts[i]}" for j, i in enumerate(perm))
        user = (f"CONVERSATION:\n{conv}\n\n" if view == "t" else "") + f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{opts}"
        r = await llm(PICKP_T if view == "t" else PICKP_B, user, PICKP_SCHEMA, max_tokens=1500)
        p = norm((r or {}).get("probabilities"), len(texts)); L = str((r or {}).get("letter", "")).strip().upper()[:1]
        if p is None or L not in LETTERS[:len(texts)]: return None
        pp = [0.0] * len(texts)
        for j, i in enumerate(perm): pp[i] = p[j]
        return {"pick": perm[LETTERS.index(L)], "p": pp}
    got = await asyncio.gather(*[one("t", k) for k in range(n_t)], *[one("b", k) for k in range(n_b)])
    def agg(rows):
        rows = [r for r in rows if r]
        if not rows: return None
        mp = [sum(r["p"][i] for r in rows) / len(rows) for i in range(len(texts))]
        return {"n": len(rows), "correct": sum(r["pick"] == truth_i for r in rows), "p_truth": mp[truth_i],
                "argmax_truth": max(range(len(mp)), key=lambda i: mp[i]) == truth_i, "mean_p": mp, "picks": [r["pick"] for r in rows]}
    return agg(got[:n_t]), agg(got[n_t:])


async def leak_check(inc, texts, truth_i, seed):
    """v3 leak check: 5 blind letter picks (PICK prompt), shuffled per sample; hits on the truth."""
    async def one(k):
        perm = list(range(len(texts))); random.Random(f"{seed}:leak:{k}").shuffle(perm)
        opts = "\n".join(f"{LETTERS[j]}) {texts[i]}" for j, i in enumerate(perm))
        r = await llm(PICK, f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{opts}", PICK_SCHEMA, max_tokens=800)
        L = str((r or {}).get("letter", "")).strip().upper()[:1]
        return L in LETTERS[:len(texts)] and perm[LETTERS.index(L)] == truth_i
    return sum(await asyncio.gather(*[one(k) for k in range(5)]))


async def gen_valid(inc, conv, prompt, n, cue="", avoid=()):
    extra = ""
    if cue: extra += (f"\n\nA previous option set was too easy: {cue}\nWrite distractors that share whatever made the true "
                      f"cause stand out (equally tied to prominent features of this conversation, equally consistent with "
                      f"the observed behaviour), so that cue no longer identifies it. They must still be false.")
    if avoid: extra += "\n\nAlready used (do not repeat these mechanisms):\n" + "\n".join(f"- {x}" for x in avoid)
    g = await llm(prompt, f"CONVERSATION:\n{conv}\n\nBEHAVIOUR: {inc.behaviour}\n\nGROUND TRUTH:\n{inc.criteria}{extra}", GEN_SCHEMA, max_tokens=9000)
    gen = [str(x).strip() for x in (g or {}).get("distractors", []) if str(x).strip()][:n]
    return gen


async def validate(inc, items):
    async def one(d):
        async with SEM:
            STATS["calls"] += 1; _count()
            return await J.score(inc.behaviour, inc.criteria, d, replicates=1)
    scores = await asyncio.gather(*[one(d) for d in items])
    return [s["score"] for s in scores]


async def rewrite_all(inc, items, is_edit):
    body = "\n".join(f"{i + 1}. {'EDIT: ' if e else ''}{x}" for i, (x, e) in enumerate(zip(items, is_edit)))
    w = await llm(REWRITE_EQ, f"BEHAVIOUR: {inc.behaviour}\n\nITEMS:\n{body}", REWRITE_SCHEMA, max_tokens=8000)
    s = [str(x).strip() for x in (w or {}).get("sentences", [])]
    return s if len(s) == len(items) else None


async def rewrite_match(inc, examples, items):
    ex = "\n".join(f"- {x}" for x in examples)
    body = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items))
    w = await llm(REWRITE_MATCH, f"BEHAVIOUR: {inc.behaviour}\n\nEXISTING OPTIONS (style to match):\n{ex}\n\nNEW ITEMS:\n{body}", REWRITE_SCHEMA, max_tokens=6000)
    s = [str(x).strip() for x in (w or {}).get("sentences", [])]
    return s if len(s) == len(items) else None


async def dedup(texts):
    """Drop indices that duplicate the truth (index 0) or an earlier item."""
    d = await llm(DEDUP, "\n".join(f"{i + 1}. {x}" for i, x in enumerate(texts)), DEDUP_SCHEMA, max_tokens=2000)
    if not d: return set()
    drop = {i - 1 for i in d.get("same_as_1", []) if isinstance(i, int) and 2 <= i <= len(texts)}
    for g in d.get("groups", []):
        g = sorted({i - 1 for i in g if isinstance(i, int) and 2 <= i <= len(texts)} - drop)
        drop.update(g[1:])
    return drop


def passes(cal_t, cal_b, leak):
    return bool(cal_b and cal_b["p_truth"] <= P_MAX and leak <= 1)


async def build(inc):
    cnt = [0]; INC_CALLS.set(cnt)
    pid = inc.prompt_id; gt = inc.ground_truth; conv = convo(inc)
    truth = gt["targets"][0]["cause"]
    raw = [(truth, "truth", False)]
    raw += [(x["cause"], "tested_alternative", False) for x in gt["ruled_out"] if x["kind"] == "refuted" and x.get("cause")]
    raw += [(x["edit"], "inert_edit", True) for x in gt["ruled_out"] if x["kind"] == "inert" and x.get("edit")]
    gen = await gen_valid(inc, conv, GEN25, 25)
    raw += [(g, "generated", False) for g in gen]
    sents = await rewrite_all(inc, [k[0] for k in raw], [k[2] for k in raw])
    if sents is None: return {"prompt_id": pid, "skip": "rewrite failed"}
    # validity is judged on the REWRITTEN sentence (what the reader sees; an inert edit rewritten as a hypothesis can
    # land on a true cause)
    sc = await validate(inc, sents[1:])
    pool = [{"text": sents[0], "source": "truth", "orig": truth}]
    pool += [{"text": t, "source": k[1], "orig": k[0], "judge": s} for t, k, s in zip(sents[1:], raw[1:], sc) if s is not None and s < 5]
    n_invalid = len(raw) - len(pool)
    drop = await dedup([p["text"] for p in pool])
    pool = [p for i, p in enumerate(pool) if i not in drop]
    n_need = N_OPT - 1
    if len(pool) - 1 < n_need: return {"prompt_id": pid, "skip": f"too few distractors ({len(pool) - 1})", "n_invalid": n_invalid}
    rk = await rank_pool(inc, conv, [p["text"] for p in pool], f"{pid}:rank0")
    for p, a, b in zip(pool, rk["t"], rk["b"]): p["pt0"], p["pb0"] = a, b
    rounds, best, cue_hist = [], None, []
    cur = [0] + select([p["pt0"] + p["pb0"] for p in pool], n_need)
    for rnd in range(4):  # round 0 + up to 3 adversarial rounds
        texts = [pool[i]["text"] for i in cur]
        (cal_t, cal_b), leak = await asyncio.gather(pickers(inc, conv, texts, 0, f"{pid}:cal{rnd}", 0, 3), leak_check(inc, texts, 0, f"{pid}:cal{rnd}"))
        ok = passes(cal_t, cal_b, leak)
        R = {"round": rnd, "set": cur, "cal_text": cal_t, "cal_blind": cal_b, "leak_hits": leak, "pass": ok}
        rounds.append(R)
        key = (not ok, (cal_b or {}).get("p_truth", 1.0))
        if best is None or key < best[0]: best = (key, R)
        if ok or rnd == 3: break
        if rnd == 0 and (not cal_b or cal_b["p_truth"] > 0.5):
            R["gave_up"] = True; break
        if rnd >= 1 and (not cal_b or cal_b["p_truth"] > 0.4):
            R["gave_up"] = True; break  # hopeless after one adversarial round: save the remaining rounds
        # adversarial round: critic names the cue for the view that failed, regenerate distractors against it
        opts = "\n".join(f"{LETTERS[j]}) {t}" for j, t in enumerate(texts))
        if False:
            c = await llm(CRITIC_T, f"CONVERSATION:\n{conv}\n\nBEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{opts}\n\nTRUE: A", CRITIC_SCHEMA, max_tokens=1500)
        else:
            c = await llm(CRITIC, f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{opts}\n\nTRUE: A", CRITIC_SCHEMA, max_tokens=1500)
        cue = (c or {}).get("cue", ""); cue_hist.append(cue)
        new = await gen_valid(inc, conv, GEN20E, 20, cue=cue, avoid=[pool[i]["orig"] for i in cur[1:]])
        ns = await rewrite_match(inc, texts, new) if new else None
        if ns:
            nsc = await validate(inc, ns)
            ok_new = [(t, g, s) for t, g, s in zip(ns, new, nsc) if s is not None and s < 5]
            n_invalid += len(ns) - len(ok_new)
            if ok_new:
                base = len(pool)
                pool += [{"text": t, "source": "generated", "orig": g, "judge": s, "round": rnd + 1} for t, g, s in ok_new]
                cand = cur + list(range(base, len(pool)))
                drop = await dedup([pool[i]["text"] for i in cand])
                cand = [i for j, i in enumerate(cand) if j not in drop]
                rk = await rank_pool(inc, conv, [pool[i]["text"] for i in cand], f"{pid}:rank{rnd + 1}")
                c_sub = [rk["b"][j] for j in range(len(cand))]
                if len(cand) - 1 >= n_need:
                    cur = [0] + [cand[j] for j in select(c_sub, n_need)]
                    R["rerank"] = {"cand": cand, "c": c_sub}
                    continue
        R["regen_failed"] = True
        # fall back: swap the currently selected distractor the pickers liked least for the next one in the pool
        c0 = [p["pt0"] + p["pb0"] if "pt0" in p else 0 for p in pool]
        worst = min(range(1, len(cur)), key=lambda j: (cal_b["mean_p"][j] if cal_b else 0))
        rest = [i for i in select(c0, len(pool)) if i not in cur]
        if not rest: break
        cur = cur[:worst] + [rest[0]] + cur[worst + 1:]
    R = best[1]; sel = R["set"]
    order = sorted(range(N_OPT), key=lambda j: hashlib.sha1(f"{pid}:mcq10blind:{j}".encode()).hexdigest())
    texts_c = [pool[sel[j]]["text"] for j in order]  # canonical order the harness shows
    options = [{"letter": LETTERS[k], "text": pool[sel[j]]["text"], "truth": j == 0, "source": pool[sel[j]]["source"]} for k, j in enumerate(order)]
    tl = next(o["letter"] for o in options if o["truth"]); ti = LETTERS.index(tl)
    fr_t, fr_b = await pickers(inc, conv, texts_c, ti, f"{pid}:fresh", 3, 3, canonical_first=True)
    def slim(a): return None if a is None else {k: a[k] for k in ("n", "correct", "p_truth", "argmax_truth")}
    return {"prompt_id": pid, "options": options, "truth": tl, "keep": R["pass"],
            "options_only_correct": R["leak_hits"], "options_only_samples": 5,
            "text_only_correct": (fr_t or {}).get("correct", 0), "text_only_samples": (fr_t or {}).get("n", 0),
            "fresh_text": fr_t, "fresh_blind": fr_b, "cal_text": slim(R["cal_text"]), "cal_blind": slim(R["cal_blind"]),
            "adversarial_rounds": len(rounds) - 1, "kept_round": R["round"], "cues": cue_hist,
            "rounds": [{k: (slim(v) if k.startswith("cal_") else v) for k, v in r.items() if k != "rerank"} for r in rounds],
            "n_pool": len(pool), "n_invalid": n_invalid, "pool": pool, "calls": cnt[0],
            "sources": {s: sum(o["source"] == s for o in options) for s in ("tested_alternative", "inert_edit", "generated")}}


async def main(ds, ids, out, conc="6"):
    from detective_joracle.chive.data import load_incidents, read_ids
    incs = load_incidents(read_ids(Path(ids)), ds)
    out = Path(out); full = out.with_suffix(".jsonl")
    done = {json.loads(l)["prompt_id"] for l in open(full)} if full.exists() else set()
    fh = open(full, "a"); sem = asyncio.Semaphore(int(conc))
    async def one(inc):
        async with sem:
            try: r = await build(inc)
            except Exception as e:
                import traceback; traceback.print_exc()
                r = {"prompt_id": inc.prompt_id, "skip": f"{type(e).__name__}: {str(e)[:200]}"}
        fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{inc.prompt_id} keep={r.get('keep')} skip={r.get('skip')} calls={STATS['calls']} fail={STATS['fail']}", flush=True)
    await asyncio.gather(*[one(i) for p, i in incs.items() if p not in done]); fh.close()
    summarize(out)


def summarize(out):
    out = Path(out); rows = [json.loads(l) for l in open(out.with_suffix(".jsonl"))]
    ok = [r for r in rows if "options" in r]; run = [r for r in ok if r["keep"]]
    run_rows = [{k: r[k] for k in ("prompt_id", "options", "truth", "options_only_correct", "options_only_samples",
                                   "text_only_correct", "text_only_samples", "fresh_text", "fresh_blind", "cal_text", "cal_blind",
                                   "adversarial_rounds", "sources")} for r in run]
    out.with_name(out.name + "_run.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in run_rows))
    def m(rs, key, f):
        v = [f(r[key]) for r in rs if r.get(key)]
        return round(sum(v) / len(v), 4) if v else None
    acc = lambda a: a["correct"] / a["n"]; pt = lambda a: a["p_truth"]
    summ = {"incidents": len(rows), "built": len(ok), "kept": len(run), "skips": sum("skip" in r for r in rows),
            "kept_fresh_text_acc": m(run, "fresh_text", acc), "kept_fresh_text_p_truth": m(run, "fresh_text", pt),
            "kept_fresh_blind_acc": m(run, "fresh_blind", acc), "kept_fresh_blind_p_truth": m(run, "fresh_blind", pt),
            "kept_cal_text_p_truth": m(run, "cal_text", pt), "kept_cal_blind_p_truth": m(run, "cal_blind", pt),
            "all_fresh_text_acc": m(ok, "fresh_text", acc), "all_fresh_blind_acc": m(ok, "fresh_blind", acc),
            "kept_round_hist": {str(k): sum(r["kept_round"] == k for r in run) for k in range(4)},
            "calls_per_incident": m(ok, "calls", lambda c: c) if all("calls" in r for r in ok) else None}
    out.with_name(out.name + "_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))


if __name__ == "__main__":
    if sys.argv[1] == "summarize": summarize(sys.argv[2])
    else: asyncio.run(main(*sys.argv[1:5]))
