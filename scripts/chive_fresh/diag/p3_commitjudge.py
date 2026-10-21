"""Phase 3a input: score every reader episode's COMMITTED (pre-readout, transcript-only) hypothesis with the same
1-10 judge as the final (detective_joracle.chive.judge.score, 3 replicates, mean). Gives committed->final flips per episode.

usage (from harness/ with its venv): python p3_commitjudge.py <batch> [--arms olens_rl_B,mlnla_B] [--draws 0,1,2,3,4]
Out: diag/commitjudge/<b>/<arm>/<pid>.d<k>.json = {score, replicates}
"""
import asyncio, json, os, sys, pathlib
from detective_joracle.chive.judge import score

C = pathlib.Path("/workspace/agam/chive_scale")
b = sys.argv[1]; opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ARMS = opt("--arms", "olens_rl_B,mlnla_B").split(","); DRAWS = [int(x) for x in opt("--draws", "0,1,2,3,4").split(",")]
GT = {r["prompt_id"]: r for r in map(json.loads, open(C / "datasets" / b / "data/ground_truth.jsonl"))}
INC = {r["prompt_id"]: r for r in map(json.loads, open(C / "datasets" / b / "data/incidents.jsonl"))}

async def main():
    sem = asyncio.Semaphore(int(os.environ.get("CJ_CONC", "12"))); n = [0, 0]
    async def one(a, f):
        out = C / "diag" / "commitjudge" / b / a / f.name
        if out.exists(): return
        r = json.loads(f.read_text())
        if str(r.get("stopped_by", "")).startswith(("error", "salvaged")): return
        h = (r.get("committed_text_hypothesis") or {}).get("hypothesis"); pid = f.name.split(".d")[0]
        if not h or pid not in GT: return
        async with sem:
            try: s = await score(INC[pid]["behaviour"], GT[pid]["judge_reference"], h)
            except Exception as e:
                n[1] += 1
                if "401" in str(e): raise SystemExit("401")
                return
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s if isinstance(s, dict) else {"score": s}, default=str)); n[0] += 1
        if n[0] % 500 == 0: print(f"[{b}] {n}", flush=True)
    fs = [(a, f) for a in ARMS for d in DRAWS for f in (C / "harness_runs" / b / a).glob(f"*.d{d}.json")]
    await asyncio.gather(*(one(a, f) for a, f in fs))
    print(f"COMMITJUDGE {b} DONE ok={n[0]} err={n[1]}", flush=True)

asyncio.run(main())
