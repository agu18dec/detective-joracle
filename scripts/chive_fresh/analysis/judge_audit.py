"""Judge audit for the CHIVE fresh free-form eval (the 1-10 hypothesis judge, pass = mean of 3 >= 7).

Per sampled incident, scores with the SAME judge (detective_joracle.chive.judge.score):
  pos_T1     the ground-truth cause T1 statement itself            (should pass ~always)
  neg_X1     the first NON-MATCHING (refuted/inert) item's statement (should score <= 2)
  neg_other  a transcript-arm final hypothesis from a DIFFERENT incident (should fail)
  generic    a generic "prompt ambiguity / priors" hypothesis       (should fail)
  retest_<arm>     the stored d0 final hypothesis re-judged (test-retest vs stored score)
  committed_<arm>  the d0 pre-readout committed text hypothesis of a reader arm (A/A + readout gain)

  python analysis/judge_audit.py N_PER_BATCH   -> analysis/judge_audit/{raw.jsonl, report.md}
"""
import asyncio, json, random, re, sys, pathlib
from detective_joracle.chive.judge import score

C = pathlib.Path("/workspace/agam/chive_scale")
OUT = C / "analysis" / "judge_audit"; OUT.mkdir(exist_ok=True)
BATCHES = ["fresh1", "fresh2", "fresh3", "fresh4"]
ARMS = ["transcript", "olens_rl_B", "mlnla_B", "jlens_L2060"]
GENERIC = ("The behaviour comes from ambiguity in the prompt: the request is underspecified, so the model "
           "falls back on its general training priors and default response style.")


def item(ref, tag):
    m = re.search(r"^\[" + tag + r"\][^\n]*", ref, re.M)
    if not m: return None
    s = re.sub(r"^\[\w+\]\s*(\((refuted|inert)\)\s*)?", "", m.group(0))
    return s.split(" — ")[0].strip()


def load(b, a, pid):
    f = C / "harness_runs" / b / a / f"{pid}.d0.json"
    if not f.exists(): return None
    r = json.loads(f.read_text())
    if str(r.get("stopped_by", "")).startswith(("error", "salvaged")): return None
    return r


async def main(n):
    rng = random.Random(0); jobs = []
    for b in BATCHES:
        gt = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(C / "datasets" / b / "data" / "ground_truth.jsonl")}
        inc = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(C / "datasets" / b / "data" / "incidents.jsonl")}
        pids = sorted(p for p in gt if all(load(b, a, p) for a in ARMS))
        pick = rng.sample(pids, min(n, len(pids)))
        for i, pid in enumerate(pick):
            ref = gt[pid]["judge_reference"]; beh = inc[pid]["behaviour"]
            other = load(b, "transcript", pick[(i + 1) % len(pick)])["result"]["hypothesis"]
            cand = {"pos_T1": item(ref, "T1"), "neg_X1": item(ref, "X1"), "neg_other": other, "generic": GENERIC}
            for a in ARMS:
                r = load(b, a, pid)
                cand[f"retest_{a}"] = r["result"]["hypothesis"]
                cand[f"stored_{a}"] = (r.get("judge") or {}).get("score")
                if a != "transcript":
                    cand[f"committed_{a}"] = (r.get("committed_text_hypothesis") or {}).get("hypothesis")
            for k, h in cand.items():
                if k.startswith("stored_"): continue
                if h: jobs.append((b, pid, k, beh, ref, h, cand))
    sem = asyncio.Semaphore(24); done = {}
    raw = OUT / "raw.jsonl"
    if raw.exists():
        for l in open(raw):
            x = json.loads(l); done[(x["batch"], x["pid"], x["kind"])] = x
    fh = open(raw, "a")
    async def one(j):
        b, pid, k, beh, ref, h, cand = j
        if (b, pid, k) in done: return
        async with sem:
            try: s = await score(beh, ref, h)
            except Exception as e: print("err", pid, k, type(e).__name__); return
        x = {"batch": b, "pid": pid, "kind": k, "score": s["score"], "reps": [g["score"] for g in s["replicates"]],
             "stored": cand.get("stored_" + k[7:]) if k.startswith("retest_") else None}
        done[(b, pid, k)] = x; fh.write(json.dumps(x) + "\n"); fh.flush()
    await asyncio.gather(*[one(j) for j in jobs])
    print("judged", len(done))


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 60))
