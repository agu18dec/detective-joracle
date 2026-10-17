"""CHIVE's mechanism-concreteness classifier (chive/binary_eval/concreteness.py, unmodified prompt
and model constant) on the incidents of claims_runs/<batch>/claims.jsonl, keyless.
Run under the CHIVE venv ($CHIVE_PY) with cwd $CHIVE_DIR. Appends to
$CHIVE_CLAIMS_RUNS/<batch>/mechanism_concreteness.jsonl (resumable).
usage: concreteness_fresh.py <batch> [model]"""

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

sys.path.insert(0, str(paths.CHIVE_DIR))
import anthropic_oauth  # noqa: E402

anthropic_oauth.install()
from chive.binary_eval import concreteness as cc  # noqa: E402
from chive.pipeline import llm_client as llmc  # noqa: E402
from chive.pipeline.run_files import load_model_understanding_eval_entries  # noqa: E402

b = sys.argv[1]
if len(sys.argv) > 2:
    cc.MODEL = sys.argv[2]
d = paths.CLAIMS_RUNS / b
want = {json.loads(line)["prompt_id"] for line in open(d / "claims.jsonl")}
out = d / "mechanism_concreteness.jsonl"
have = {json.loads(line)["prompt_id"] for line in open(out)} if out.exists() else set()
entries = load_model_understanding_eval_entries(str(paths.RUN_DIRS / b), min_verification_score=8)
cache = {
    e.prompt_id: {
        "question": e.second_person_question,
        "baseline_behavior": e.behavior_summary,
        "structured_answer": e.gt_structured_text,
        "commentary": e.commentary,
    }
    for e in entries
    if e.prompt_id in want and e.prompt_id not in have
}
print(
    f"{b}: {len(want)} incidents, {len(have)} done, {len(cache)} to score with {cc.MODEL}; "
    f"{len(want - have - set(cache))} not loadable",
    flush=True,
)
llmc.set_max_concurrency(16)
client = llmc.make_llm_client(cc.MODEL)


async def go():
    with out.open("a") as f:
        for fut in asyncio.as_completed([cc.classify_one(client, p, r) for p, r in cache.items()]):
            r = await fut
            if r is not None:
                f.write(json.dumps({**r, "model": cc.MODEL}) + "\n")
                f.flush()


asyncio.run(go())
