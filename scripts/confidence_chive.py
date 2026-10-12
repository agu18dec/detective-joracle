"""Transcript-only uncertainty per incident: a stratification that does not select on errors.

Selecting incidents where the transcript investigator FAILED inflates any tool's gain (regression
to the mean, plus: anything that pushes the agent off its default answer helps when the default
is wrong). Instead, the investigator's model (Opus 5, adaptive thinking at effort=max, no tools)
reads each incident and spreads probability over up to five distinct candidate causes plus
"something else" -- the probability that controlled experiments would establish each one --
REPLICATES times. Uncertainty is read off that distribution without looking at correctness:

  top_p    mean (over samples) probability of the top candidate
  entropy  mean entropy (bits) of the distribution, "something else" included

Writes <out>/confidence.json: prompt_id -> {top_p, entropy, samples: [...]}. Pre-registered use:
docs/chive_preregistration.md.

    python scripts/confidence_chive.py out=/workspace/agam/chive_runs_dj
"""

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any

import pydra

from detective_joracle.agent.loop import run_many
from detective_joracle.chive.data import DATASET, Incident, load_incidents, read_ids
from detective_joracle.chive.llm import anthropic_client
from detective_joracle.chive.prompts import seed

MODELS = ("claude-opus-5", "claude-sonnet-5")  # the second covers content-filter refusals
REPLICATES = 3
SYSTEM = """You see a conversation with a language model, a behaviour in its response, and how \
often it recurs when the model is re-run. Controlled experiments (editing the conversation and \
re-running the model) have established which feature of the conversation drives the behaviour; \
you do not know the result.

List up to five DISTINCT candidate causes -- each one specific feature of the conversation and \
how it drives the behaviour -- and give each the probability that it is the cause the \
experiments established. Put the remaining probability on "something else". The probabilities \
must sum to 1. Be calibrated: do not inflate your favourite, and do not spread evenly out of \
caution.

End your reply with only this JSON:
{"candidates": [{"cause": "...", "p": 0.0}, ...], "something_else": 0.0}"""


def parse(text: str) -> list[float] | None:
    """Candidate probabilities plus 'something else', renormalised; None if unparseable."""
    m = re.search(r"\{[^{}]*\"candidates\".*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        ps = [float(c["p"]) for c in d["candidates"]][:5] + [float(d.get("something_else", 0))]
    except (ValueError, KeyError, TypeError):
        return None
    total = sum(max(0.0, x) for x in ps)
    if total <= 0 or not 0.8 <= total <= 1.2:  # badly off: not a distribution
        return None
    return [max(0.0, x) / total for x in ps]


def entropy(ps: list[float]) -> float:
    return -sum(x * math.log2(x) for x in ps if x > 0)


async def ask(c: Any, inc: Incident) -> dict[str, Any] | None:
    user = seed(inc).rsplit("Positions are tokens", 1)[0]
    for model in MODELS:
        for _ in range(2):  # one retry on an unparseable reply
            try:
                r = await c.messages.create(
                    model=model,
                    max_tokens=16000,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": user}],
                    thinking={"type": "adaptive"},
                    output_config={"effort": "max"},
                )
            except Exception:
                continue
            text = "".join(b.text for b in r.content if b.type == "text")
            ps = parse(text)
            if ps is not None:
                return {"model": model, "ps": ps, "text": text[-3000:]}
    return None


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = DATASET  # HF dataset id or a local directory (see run_chive.py)
        self.out = pydra.REQUIRED
        self.ids = "data/chive/splits/all300.txt"
        self.concurrency = 16

    def finalize(self) -> None:
        self.out = Path(self.out)


@pydra.main(Config)
def main(cfg: Config) -> None:
    f = cfg.out / "confidence.json"
    done: dict[str, Any] = json.loads(f.read_text()) if f.exists() else {}
    incs = load_incidents(read_ids(Path(cfg.ids)), cfg.dataset)
    c = anthropic_client()

    async def job(inc: Incident) -> None:
        got = [s for s in await asyncio.gather(*[ask(c, inc) for _ in range(REPLICATES)]) if s]
        if len(got) < REPLICATES:
            print(f"{inc.prompt_id}: {len(got)}/{REPLICATES} samples; will retry")
            return
        done[inc.prompt_id] = {
            "top_p": sum(max(s["ps"][:-1] or [0.0]) for s in got) / len(got),
            "entropy": sum(entropy(s["ps"]) for s in got) / len(got),
            "samples": got,
        }

    todo = [i for p, i in incs.items() if p not in done]
    print(f"confidence: {len(todo)} incidents to ask")
    asyncio.run(run_many([lambda i=i: job(i) for i in todo], cfg.concurrency))
    f.write_text(json.dumps(done, ensure_ascii=False))
    tp = sorted(v["top_p"] for v in done.values())
    q = [tp[int(len(tp) * x)] for x in (0.25, 0.5, 0.75)]
    print(
        f"{len(done)} incidents; top_p quartiles {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}; "
        f"in [0.25, 0.75]: {sum(0.25 <= x <= 0.75 for x in tp)}"
    )


if __name__ == "__main__":
    main()
