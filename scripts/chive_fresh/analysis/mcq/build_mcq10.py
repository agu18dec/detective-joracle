"""CHIVE MCQ option sets: 10 equally hard candidate causes per incident (1 true + 9 distractors).

Per incident:
 1. truth = the first established cause (ground_truth.targets[0]); tested alternatives = CHIVE's refuted causes and
    inert edits (ruled_out), as in harness/scripts/build_tested_choices.py.
 2. GEN: Opus sees the conversation, the behaviour and the FULL ground truth (true causes + ruled-out list) and writes
    12 further plausible, specific mechanisms that are none of the listed true causes.
 3. VALIDITY: each generated distractor is scored by the harness's 1-10 judge against the ground truth (1 replicate);
    any scoring >= 5 (might match a true cause) is dropped.
 4. HARDNESS: a text-only picker (conversation + candidates, no tools) gives a probability per candidate over the truth,
    the tested alternatives and the valid generated ones; distractors are tested alternatives first, then the
    generated ones with the highest probability, to 9.
 5. STYLE: one call rewrites all 10 into neutral 20-35-word sentences ("The model ... because/when ...").
 6. CHECKS on the final set (shuffled by hash): blind picker (behaviour + options only) x5 -> options_only_correct;
    text-only picker (conversation + options) x3 -> text_only_correct.
Writes <out>.jsonl (every incident, harness choices= format + diagnostics) and <out>_run.jsonl (sets passing the
leak check: blind picker right <= 1 of 5).

  python build_mcq10.py <dataset_dir> <ids_file> <out_prefix>
"""
import asyncio, hashlib, json, sys
from pathlib import Path
from detective_joracle.chive.data import load_incidents, read_ids
from detective_joracle.chive.llm import client, json_call
from detective_joracle.chive import judge as J

LETTERS = "ABCDEFGHIJ"
N_OPT = 10
GEN = """You design distractors for a multiple-choice test about why a language model behaved a certain way in a
conversation. You are given the conversation, the behaviour, and the ground truth established by experiments: the
TRUE causes and a list of causes/edits that were tested and ruled out. Write 12 additional candidate causes that an
expert investigator reading this conversation would find genuinely plausible. Each must name a specific feature of
THIS conversation (a phrase, instruction, format, framing, missing information, prior turn) and a concrete mechanism.
Rules: none may state or imply the same mechanism as any TRUE cause (not even partially, not as a broader or narrower
version); avoid repeating the ruled-out items; make them as specific, as directly tied to the behaviour, and as convincing as the true causes -- a reader
who sees ONLY the behaviour description and the options must find each of them as likely as the true cause; all 12
must be distinct from each other and from the ruled-out items.""" 
GEN_SCHEMA = {"type": "object", "required": ["distractors"], "properties": {"distractors": {"type": "array", "items": {"type": "string"}}}}
RANK = """You see a conversation with a language model, a description of a behaviour in its response, and numbered
candidate causes. Exactly one was established by experiments as a true cause. Give your probability for each
candidate (they should sum to 1)."""
RANK_SCHEMA = {"type": "object", "required": ["probabilities"], "properties": {"probabilities": {"type": "array", "items": {"type": "number"}}}}
REWRITE = """You rewrite candidate explanations of a language model's behaviour into one uniform style. For every
numbered item (a hypothesis, or a description of a tested edit: rewrite that as the hypothesis it tested, i.e. that
the edited feature drives the behaviour), write ONE neutral sentence of 20-35 words of the form "The model ...
because/when ...", stating only the proposed cause and mechanism, naming the specific feature it depends on -- no
evidence, no results, no hedging, no first person. Make every sentence equally detailed and equally direct about how it produces the behaviour: do
not make any one more elaborate, more hedged or more closely worded to the behaviour description than the others.
Keep the items' meanings distinct. Return the sentences in the same order."""
REWRITE_SCHEMA = {"type": "object", "required": ["sentences"], "properties": {"sentences": {"type": "array", "items": {"type": "string"}}}}
CRITIC = """A multiple-choice item about why a language model behaved a certain way leaked its answer: a reader who saw
only the behaviour and the options picked the true option. In one or two sentences, say what cue makes the true
option stand out (e.g. it is the only one that mentions X, it echoes the behaviour's wording, it is more general)."""
CRITIC_SCHEMA = {"type": "object", "required": ["cue"], "properties": {"cue": {"type": "string"}}}
PICK = """You see a description of a language model's behaviour and candidate explanations. Exactly one is the true
cause established by experiments. Pick the most likely one."""
PICK_SCHEMA = {"type": "object", "required": ["letter"], "properties": {"letter": {"type": "string"}}}


def convo(inc, limit=40000):
    t = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in inc.messages)
    t += f"\n\n[assistant response in which the behaviour was observed]\n{inc.completion}"
    return t if len(t) <= limit else t[: limit // 2] + "\n...[truncated]...\n" + t[-limit // 2:]


async def call(system, user, schema, model="claude-opus-5", max_tokens=4000):
    for m in (model, "claude-sonnet-5"):
        try:
            got = await json_call(client(), model=m, system=system, user=user, schema=schema, max_tokens=max_tokens)
        except Exception:
            got = None
        if got: return got
    return None


async def build(inc, _retry=""):
    gt = inc.ground_truth
    truth = gt["targets"][0]["cause"]
    excl = [x["cause"] for x in gt["ruled_out"] if x["kind"] == "refuted" and x.get("cause")][:4]
    conv = convo(inc)
    extra = (f"\n\nA previous set leaked the answer: {_retry} Write distractors that share whatever made the true cause "
             f"stand out, so that cue no longer identifies it.") if _retry else ""
    g = await call(GEN, f"CONVERSATION:\n{conv}\n\nBEHAVIOUR: {inc.behaviour}\n\nGROUND TRUTH:\n{inc.criteria}{extra}", GEN_SCHEMA, max_tokens=6000)
    gen = [str(x).strip() for x in (g or {}).get("distractors", []) if str(x).strip()][:12]
    scores = await asyncio.gather(*[J.score(inc.behaviour, inc.criteria, d, replicates=1) for d in gen])
    valid = [(d, s["score"]) for d, s in zip(gen, scores) if s["score"] is not None and s["score"] < 5]
    need = N_OPT - 1 - len(excl)
    if need > len(valid): return {"prompt_id": inc.prompt_id, "skip": f"only {len(valid)} valid generated distractors"}
    pool = [truth, *excl, *[d for d, _ in valid]]
    r = await call(RANK, f"CONVERSATION:\n{conv}\n\nBEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(pool)), RANK_SCHEMA)
    pr = (r or {}).get("probabilities") or []
    pr = pr if len(pr) == len(pool) else [0.0] * len(pool)
    gen_ranked = sorted(range(1 + len(excl), len(pool)), key=lambda i: -pr[i])[:need]
    items = [truth, *excl, *[pool[i] for i in gen_ranked]]
    src = ["truth"] + ["tested_alternative"] * len(excl) + ["generated"] * need
    w = await call(REWRITE, f"BEHAVIOUR: {inc.behaviour}\n\nITEMS:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items)), REWRITE_SCHEMA)
    sents = [str(x).strip() for x in (w or {}).get("sentences", [])]
    if len(sents) != len(items): return {"prompt_id": inc.prompt_id, "skip": "rewrite failed"}
    order = sorted(range(len(sents)), key=lambda i: hashlib.sha1(f"{inc.prompt_id}:mcq10:{i}".encode()).hexdigest())
    options = [{"letter": LETTERS[j], "text": sents[i], "truth": i == 0, "source": src[i]} for j, i in enumerate(order)]
    tl = next(o["letter"] for o in options if o["truth"])
    opts = "\n".join(f"{o['letter']}) {o['text']}" for o in options)
    blind = await asyncio.gather(*[call(PICK, f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{opts}", PICK_SCHEMA, max_tokens=800) for _ in range(5)])
    hit0 = sum(bool(p and str(p.get("letter", "")).strip().upper()[:1] == tl) for p in blind)
    if hit0 >= 2 and not _retry:  # adversarial round: name the cue, regenerate distractors without it
        cue = await call(CRITIC, f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{opts}\n\nTRUE: {tl}", CRITIC_SCHEMA, max_tokens=1500)
        return await build(inc, _retry=(cue or {}).get("cue", ""))
    blind_first = hit0
    text = await asyncio.gather(*[call(PICK, f"CONVERSATION:\n{conv}\n\nBEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n{opts}", PICK_SCHEMA, max_tokens=800) for _ in range(3)])
    hit = lambda ps: sum(bool(p and str(p.get("letter", "")).strip().upper()[:1] == tl) for p in ps)
    return {"prompt_id": inc.prompt_id, "options": options, "truth": tl,
            "options_only_correct": hit(blind), "options_only_samples": 5,
            "text_only_correct": hit(text), "text_only_samples": 3,
            "adversarial_round": bool(_retry), "n_tested": len(excl), "n_generated_valid": len(valid), "gen_judge_scores": [s for _, s in valid]}


async def main(ds, ids, out):
    incs = load_incidents(read_ids(Path(ids)), ds)
    out = Path(out); full = out.with_suffix(".jsonl")
    done = {json.loads(l)["prompt_id"] for l in open(full)} if full.exists() else set()
    fh = open(full, "a"); sem = asyncio.Semaphore(12)
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
            "text_only_acc": sum(r["text_only_correct"] for r in ok) / max(1, 3 * len(ok)),
            "skips": sum("skip" in r for r in rows)}
    out.with_name(out.name + "_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:4]))
