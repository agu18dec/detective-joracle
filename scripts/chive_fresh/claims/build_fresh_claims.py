"""Build per-batch claim files for claims_chive.py from the fresh CHIVE claim banks, with CHIVE's
build_eval_set.py balancing (seed 0, k=min(T,F), rng.sample(T,k)+rng.sample(F,k), shuffle).
Concreteness is NOT applied here (no mechanism_concreteness for fresh runs); it is a report stratum.
Restricted to incidents in $CHIVE_DATASETS/<b>/splits/all.txt that have an s3d_rl600_all bank file.
Writes $CHIVE_CLAIMS_RUNS/<b>/{claims.jsonl, claims_meta.json}.
usage: build_fresh_claims.py [batches...]"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

sys.path.insert(0, str(paths.REPO / "src"))
from detective_joracle.chive.claims import Claim  # noqa: E402

for b in sys.argv[1:] or paths.BATCHES:
    ds = paths.DATASETS / b
    have = set((ds / "splits/all.txt").read_text().split())
    bank = paths.BANKS / "s3d_rl600_all"
    rows = [json.loads(line) for line in open(ds / "work/binary_counterfactual_questions.jsonl")]
    kept = [
        r for r in rows if r["prompt_id"] in have and (bank / f"{r['prompt_id']}.json.gz").exists()
    ]
    rng = random.Random(0)
    T = [r for r in kept if r["is_true"]]
    F = [r for r in kept if not r["is_true"]]
    k = min(len(T), len(F))
    out = rng.sample(T, k) + rng.sample(F, k)
    rng.shuffle(out)
    d = paths.CLAIMS_RUNS / b
    d.mkdir(parents=True, exist_ok=True)
    (d / "claims.jsonl").write_text(
        "".join(json.dumps(Claim.from_chive(r).to_row(), ensure_ascii=False) + "\n" for r in out)
    )
    meta = {
        r["claim_id"]: {
            k2: r[k2]
            for k2 in ("edit_cleanliness", "effect_pp", "verification_score", "claim_type")
        }
        for r in out
    }
    (d / "claims_meta.json").write_text(json.dumps(meta))
    print(
        b,
        f"bank {len(rows)} -> in split+bank {len(kept)} ({len(T)}T/{len(F)}F) -> balanced {2 * k} ({k}T/{k}F), {len({r['prompt_id'] for r in out})} incidents",
    )
