"""CHIVE validity audit, part B/C: replicate decisive experiments and re-grade CHIVE's own completions.

Sample: 80 incidents (seed 0) across fresh1-4. Per incident, from runs/<b>/investigations.jsonl experiment_log (exact
edited messages, 30 completions, CHIVE's Sonnet verdicts):
  B. REPLICATION with our executor (Qwen3.6-27B vLLM, T=1, 500 tokens; Opus grader with CHIVE's grader prompt):
     baseline (original conversation) n=30; the top MOVER edit (largest-effect claim-bank mover) n=30; one claim-bank
     INERT edit n=60 (2 x 30, for power). Compare rates with CHIVE's numbers; does the mover still move >= 30 pp from our
     own baseline, does the inert edit stay within 15 pp?
  C. GRADER AGREEMENT: re-grade CHIVE's stored 30 baseline completions and 30 mover-experiment completions with our
     Opus grader; per-response agreement with CHIVE's Sonnet verdicts and the k difference.
Out: diag/replicate/<pid>.json ; report with --report
usage (from harness/ with its venv): python p7_replicate.py run [--n 80] | python p7_replicate.py report
"""
import ast, asyncio, json, os, random, sys, pathlib
C = pathlib.Path("/workspace/agam/chive_scale")
U = "https://safety-sahan-2-agam--chive-qwen-vllm-serve.modal.run"

def load_inv(b):
    out = {}
    for l in open(C / "runs" / b / "investigations.jsonl"):
        r = json.loads(l); out[r["prompt_id"]] = r
    return out

def msgs(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

async def main(n):
    from detective_joracle.chive.executor import Executor, GRADER_SYSTEM, VERDICT_SCHEMA
    from detective_joracle.chive.llm import client, json_call
    ex = Executor(U); ex.N = 30
    async def grade(messages, behaviour, completions):
        convo = "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in messages)
        user = (f"## Conversation\n{convo}\n\n## Question (answer for EACH response independently)\n\n{behaviour}\n\n"
                f"## Responses ({len(completions)})\n" + "".join(f"\n--- Response {i}/{len(completions)} ---\n{c}\n" for i, c in enumerate(completions, 1)))
        got = await json_call(client(), model=ex.grader_model, system=GRADER_SYSTEM, user=user, schema=VERDICT_SCHEMA, max_tokens=8000)
        by = {int(v["completion"]): v["answer"] == "yes" for v in (got or {}).get("verdicts", [])}
        if sorted(by) != list(range(1, len(completions) + 1)): return None
        return [by[i] for i in range(1, len(completions) + 1)]
    async def run(messages, behaviour, reps=1):
        ks, ns, comps = 0, 0, []
        for _ in range(reps):
            c = await asyncio.to_thread(ex._complete, messages); v = await grade(messages, behaviour, c)
            if v is None: return None
            ks += sum(v); ns += len(v); comps += c
        return {"k": ks, "n": ns, "rate": ks / ns}
    pool = []
    for b in ["fresh1", "fresh2", "fresh3", "fresh4"]:
        gts = [json.loads(l) for l in open(C / "datasets" / b / "data/ground_truth.jsonl")]
        pool += [(b, g) for g in gts]
    random.Random(0).shuffle(pool)
    invs = {b: load_inv(b) for b in ["fresh1", "fresh2", "fresh3", "fresh4"]}
    incs = {}
    for b in ["fresh1", "fresh2", "fresh3", "fresh4"]:
        for l in open(C / "datasets" / b / "data/incidents.jsonl"):
            r = json.loads(l); incs[r["prompt_id"]] = r
    sem = asyncio.Semaphore(int(os.environ.get("REP_CONC", "6")))
    async def one(b, g):
        pid = g["prompt_id"]; out = C / "diag" / "replicate" / f"{pid}.json"
        if out.exists(): return
        inv = invs[b].get(pid)
        if not inv: return
        el = {str(e["exp_num"]): e for e in inv.get("experiment_log") or []}
        beh = incs[pid]["behaviour"]; base_msgs = incs[pid]["messages"]
        movers = sorted([m for m in g.get("movers", []) if m.get("experiment_numbers")], key=lambda m: -m["effect_pp"])
        inert = [r for r in g["ruled_out"] if r.get("kind") == "inert" and r.get("basis") == "claim_bank" and r.get("experiment_numbers")]
        mv = next((m for m in movers if str(m["experiment_numbers"][0]) in el), None)
        it = next((r for r in inert if str(r["experiment_numbers"][0]) in el), None)
        async with sem:
            rec = {"prompt_id": pid, "batch": b, "chive_base": {"k": g["k"], "n": g["n"]}}
            rec["base"] = await run(base_msgs, beh)
            if mv:
                e = el[str(mv["experiment_numbers"][0])]; m = msgs(e["messages"])
                rec["mover"] = {"edit": mv["edit"], "cleanliness": mv.get("edit_cleanliness"), "chive": mv["rate_counts"], "chive_base": mv["baseline_rate"],
                                "ours": await run(m, beh)}
                v = await grade(m, beh, e["completions"])
                cv = msgs(e["grader_verdicts"]) if isinstance(e["grader_verdicts"], str) else e["grader_verdicts"]
                if v: rec["regrade_mover"] = {"agree": sum(a == bool(c) for a, c in zip(v, cv)) / len(v), "k_ours": sum(v), "k_chive": sum(bool(c) for c in cv)}
            if it:
                e = el[str(it["experiment_numbers"][0])]; m = msgs(e["messages"])
                rec["inert"] = {"edit": it["edit"], "chive": it["rate_counts"], "chive_base": it["baseline_rate"], "ours": await run(m, beh, reps=2)}
            v = await grade(base_msgs, beh, inv["completions"])
            cv = inv["baseline_grader_verdicts"]
            if v and cv and len(cv) == len(v): rec["regrade_base"] = {"agree": sum(a == bool(c) for a, c in zip(v, cv)) / len(v), "k_ours": sum(v), "k_chive": sum(bool(c) for c in cv)}
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec))
        print(pid, "done", flush=True)
    await asyncio.gather(*[one(b, g) for b, g in pool[:n]])
    print("REPLICATE DONE", flush=True)

def report():
    import statistics as S
    R = [json.loads(f.read_text()) for f in (C / "diag" / "replicate").glob("*.json")]
    print("incidents", len(R))
    bd = [abs(r["base"]["rate"] - r["chive_base"]["k"] / r["chive_base"]["n"]) for r in R if r.get("base")]
    print(f"baseline |ours - CHIVE|: median {S.median(bd):.3f}, >=0.3: {sum(x >= .3 for x in bd)}/{len(bd)}")
    mv = [r for r in R if r.get("mover") and r["mover"]["ours"] and r.get("base")]
    eff = [r["base"]["rate"] - r["mover"]["ours"]["rate"] for r in mv]
    ch = [r["mover"]["chive_base"] - int(r["mover"]["chive"].split("/")[0]) / int(r["mover"]["chive"].split("/")[1]) for r in mv]
    print(f"mover: CHIVE effect median {S.median(ch):.2f}; our effect (vs our baseline) median {S.median(eff):.2f}; replicates (|effect| >= 0.30 same sign): {sum(abs(e) >= .30 and e * c > 0 for e, c in zip(eff, ch))}/{len(mv)}; >= 0.15: {sum(abs(e) >= .15 and e * c > 0 for e, c in zip(eff, ch))}/{len(mv)}")
    for cl in ("minor_targeted", "large_single_factor", "mixed", "confounded"):
        xs = [(e, c) for r, e, c in zip(mv, eff, ch) if r["mover"]["cleanliness"] == cl]
        if xs: print(f"   {cl:20s} n={len(xs)} replicate>=30pp {sum(abs(e) >= .30 and e * c > 0 for e, c in xs)}")
    it = [r for r in R if r.get("inert") and r["inert"]["ours"] and r.get("base")]
    ie = [abs(r["base"]["rate"] - r["inert"]["ours"]["rate"]) for r in it]
    print(f"inert (n=60): |effect| median {S.median(ie):.2f}; stays <= 0.15: {sum(x <= .15 for x in ie)}/{len(it)}; >= 0.30 (inert verdict wrong): {sum(x >= .30 for x in ie)}/{len(it)}")
    for k in ("regrade_base", "regrade_mover"):
        g = [r[k] for r in R if r.get(k)]
        if g: print(f"{k}: per-response agreement Opus vs CHIVE Sonnet {S.mean(x['agree'] for x in g):.3f}; mean |k diff| {S.mean(abs(x['k_ours'] - x['k_chive']) for x in g):.2f}/30; n={len(g)}")

if __name__ == "__main__":
    if sys.argv[1] == "report": report()
    else: asyncio.run(main(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 80))
