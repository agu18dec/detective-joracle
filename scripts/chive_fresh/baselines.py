"""Re-measure each incident's baseline with the experiment executor itself.

The budgeted-counterfactual protocol (docs/chive_fresh_scaleup.md) compares an agent's experiment
against the unedited conversation measured the SAME way: CHIVE_EXP_N (default here 30) Qwen samples
from the harness's target endpoint, graded by the harness's Opus CHIVE grader. CHIVE's own baseline
(k/n in incidents.jsonl) came from a different sampler and grader, so its rate can differ (e.g.
21/30 vs 8/30). Output: a JSON file {prompt_id: {k, n, rate, chive_k, chive_n}}, written after
every incident (resumable: incidents already in it are skipped). Point the harness at it with
``CHIVE_BASELINES=<file>``.

    source scripts/chive_fresh/env.sh
    $HARNESS_PY scripts/chive_fresh/baselines.py [--out FILE] [--batches fresh1,fresh2,...] \
        [--url TARGET_URL] [--concurrency 16]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CHIVE_EXP_N", "30")  # before the executor reads it at import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

from detective_joracle.chive.data import Incident, load_incidents, read_ids  # noqa: E402
from detective_joracle.chive.executor import Executor  # noqa: E402


async def measure(
    out: Path, batches: list[str], url: str, concurrency: int, datasets: Path
) -> dict[str, Any]:
    done: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
    ex = Executor(url)
    sem = asyncio.Semaphore(concurrency)
    for b in batches:
        d = datasets / b
        incs = load_incidents([i for i in read_ids(d / "splits/all.txt") if i not in done], str(d))

        async def one(inc: Incident) -> None:
            async with sem:
                for _ in range(3):
                    try:
                        keep_all = [{"keep": i + 1} for i in range(len(inc.messages))]
                        r = await ex.run(inc, keep_all)
                        done[inc.prompt_id] = {
                            "k": r["k"],
                            "n": r["n"],
                            "rate": r["rate"],
                            "chive_k": inc.k,
                            "chive_n": inc.n,
                        }
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_text(json.dumps(done))
                        return
                    except Exception as e:
                        print("retry", inc.prompt_id, type(e).__name__, str(e)[:120], flush=True)

        await asyncio.gather(*(one(i) for i in incs.values()))
        print(b, "done;", len(done), "baselines", flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=paths.SCALE / "exp1" / "baselines.json")
    ap.add_argument("--batches", default=",".join(paths.BATCHES))
    ap.add_argument("--url", default=paths.TARGET_URL)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--datasets", type=Path, default=paths.DATASETS)
    a = ap.parse_args()
    asyncio.run(measure(a.out, a.batches.split(","), a.url, a.concurrency, a.datasets))


if __name__ == "__main__":
    main()
