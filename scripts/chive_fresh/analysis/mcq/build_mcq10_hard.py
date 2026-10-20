"""Harder MCQ option sets (v3): same format and validity rules as build_mcq10.py, built to lower blind-picker accuracy.

Changes vs v2:
 - GEN asks for 20 distractors; if fewer than needed survive the judge filter (score < 5), a second GEN call lists
   the rejected ones as "too close to a true cause" and asks for 20 more.
 - Distractors are ranked by p_blind + p_text: the probability an options-only picker (behaviour + candidates) and a
   text-only picker (conversation + candidates) put on each, so they look as right as the truth from both views.
 - The rewrite forbids echoing the behaviour description's wording and requires the same structure for all options.
 - Up to 3 adversarial rounds (a critic names the cue that gave the truth away; distractors are regenerated to share
   it); the set with the fewest blind hits is kept.

  python build_mcq10_hard.py <dataset_dir> <ids_file> <out_prefix>
"""
import asyncio, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_mcq10 import convo, call, GEN, GEN_SCHEMA, RANK, RANK_SCHEMA, PICK, PICK_SCHEMA, CRITIC, CRITIC_SCHEMA, LETTERS, N_OPT
from detective_joracle.chive.data import load_incidents, read_ids
from detective_joracle.chive import judge as J

GEN20 = GEN.replace("Write 12 additional", "Write 20 additional").replace("all 12 must", "all 20 must")
BLIND_RANK = """You see a description of a language model's behaviour and numbered candidate causes (no conversation).
Exactly one was established by experiments as a true cause. Give your probability for each candidate (sum to 1)."""
REWRITE_HARD = """You rewrite candidate explanations of a language model's behaviour into one uniform style for a
multiple-choice test. For every numbered item (a hypothesis about what drives the behaviour), write ONE neutral sentence
of 22-30 words with the exact structure "The model <does the behaviour> because <specific feature of the conversation>
<mechanism>." Rules: use the SAME short wording for the behaviour part in every sentence; do NOT reuse distinctive words
or phrases from the behaviour description in the "because" part; every sentence must name one specific feature and one
mechanism, with the same level of detail and confidence; no evidence, results, hedging or first person; keep meanings
distinct. Return the sentences in the same order."""
REWRITE_SCHEMA = {"type": "object", "required": ["sentences"], "properties": {"sentences": {"type": "array", "items": {"type": "string"}}}}


async def gen_valid(inc, conv, need, cue):
    valid, rejected = [], []
    for attempt in range(2):
        extra = ""
        if cue: extra += (f"\n\nA previous set leaked the answer: {cue} Write distractors that share whatever made the true cause "
                          f"stand out, so that cue no longer identifies it.")
        if rejected: extra += "\n\nThese were rejected as too close to a true cause; avoid their mechanisms:\n" + "\n".join(f"- {x}" for x in rejected[:20])
        g = await call(GEN20, f"CONVERSATION:\n{conv}\n\nBEHAVIOUR: {inc.behaviour}\n\nGROUND TRUTH:\n{inc.criteria}{extra}", GEN_SCHEMA, max_tokens=8000)
        gen = [str(x).strip() for x in (g or {}).get("distractors", []) if str(x).strip()][:20]
        scores = await asyncio.gather(*[J.score(inc.behaviour, inc.criteria, d, replicates=1) for d in gen])
        for d, s in zip(gen, scores):
            (valid if s["score"] is not None and s["score"] < 5 else rejected).append(d)
        if len(valid) >= need + 4: break
    return valid


async def make_set(inc, conv, truth, excl, valid, need):
    pool = [truth, *excl, *valid]
    cands = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(pool))
    rt, rb = await asyncio.gather(call(RANK, f"CONVERSATION:\n{conv}\n\nBEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{cands}", RANK_SCHEMA),
                                  call(BLIND_RANK, f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{cands}", RANK_SCHEMA))
    def probs(r):
        p = (r or {}).get("probabilities") or []
        return p if len(p) == len(pool) else [0.0] * len(pool)
    pt, pb = probs(rt), probs(rb)
    ranked = sorted(range(1 + len(excl), len(pool)), key=lambda i: -(pt[i] + pb[i]))[:need]
    items = [truth, *excl, *[pool[i] for i in ranked]]
    src = ["truth"] + ["tested_alternative"] * len(excl) + ["generated"] * need
    w = await call(REWRITE_HARD, f"BEHAVIOUR: {inc.behaviour}\n\nITEMS:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items)), REWRITE_SCHEMA)
    sents = [str(x).strip() for x in (w or {}).get("sentences", [])]
    if len(sents) != len(items): return None
    order = sorted(range(len(sents)), key=lambda i: hashlib.sha1(f"{inc.prompt_id}:mcq10h:{i}".encode()).hexdigest())
    options = [{"letter": LETTERS[j], "text": sents[i], "truth": i == 0, "source": src[i]} for j, i in enumerate(order)]
    tl = next(o["letter"] for o in options if o["truth"])
    opts = "\n".join(f"{o['letter']}) {o['text']}" for o in options)
    blind = await asyncio.gather(*[call(PICK, f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{opts}", PICK_SCHEMA, max_tokens=800) for _ in range(5)])
    hits = sum(bool(p and str(p.get("letter", "")).strip().upper()[:1] == tl) for p in blind)
    return {"options": options, "truth": tl, "opts": opts, "hits": hits}


async def build(inc):
    gt = inc.ground_truth
    truth = gt["targets"][0]["cause"]
    excl = [x["cause"] for x in gt["ruled_out"] if x["kind"] == "refuted" and x.get("cause")][:4]
    need = N_OPT - 1 - len(excl); conv = convo(inc)
    best, cue, rounds, n_valid = None, "", 0, 0
    for rounds in range(3):
        valid = await gen_valid(inc, conv, need, cue); n_valid = len(valid)
        if len(valid) < need: continue
        s = await make_set(inc, conv, truth, excl, valid, need)
        if s and (best is None or s["hits"] < best["hits"]): best = s
        if best and best["hits"] <= 1: break
        if s:
            c = await call(CRITIC, f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{s['opts']}\n\nTRUE: {s['truth']}", CRITIC_SCHEMA, max_tokens=1500)
            cue = (c or {}).get("cue", "")
    if best is None: return {"prompt_id": inc.prompt_id, "skip": f"too few valid distractors ({n_valid})"}
    text = await asyncio.gather(*[call(PICK, f"CONVERSATION:\n{conv}\n\nBEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{best['opts']}", PICK_SCHEMA, max_tokens=800) for _ in range(3)])
    th = sum(bool(p and str(p.get("letter", "")).strip().upper()[:1] == best["truth"]) for p in text)
    return {"prompt_id": inc.prompt_id, "options": best["options"], "truth": best["truth"],
            "options_only_correct": best["hits"], "options_only_samples": 5, "text_only_correct": th, "text_only_samples": 3,
            "adversarial_rounds": rounds, "n_tested": len(excl)}


async def main(ds, ids, out):
    incs = load_incidents(read_ids(Path(ids)), ds)
    out = Path(out); full = out.with_suffix(".jsonl")
    done = {json.loads(l)["prompt_id"] for l in open(full)} if full.exists() else set()
    fh = open(full, "a"); sem = asyncio.Semaphore(10)
    async def one(inc):
        async with sem:
            try: r = await build(inc)
            except Exception as e: r = {"prompt_id": inc.prompt_id, "skip": f"{type(e).__name__}: {str(e)[:200]}"}
        fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
    await asyncio.gather(*[one(i) for p, i in incs.items() if p not in done]); fh.close()
    rows = [json.loads(l) for l in open(full)]
    ok = [r for r in rows if "options" in r]
    run = [r for r in ok if r["options_only_correct"] <= 1]
    out.with_name(out.name + "_run.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in run))
    summ = {"incidents": len(rows), "built": len(ok), "pass_leak_check": len(run),
            "blind_acc": sum(r["options_only_correct"] for r in ok) / max(1, 5 * len(ok)),
            "text_only_acc": sum(r["text_only_correct"] for r in ok) / max(1, 3 * len(ok)), "skips": sum("skip" in r for r in rows)}
    out.with_name(out.name + "_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:4]))
