"""Transcript-only confidence per incident: a stratification that does not select on errors.

Selecting incidents where the transcript investigator FAILED inflates any tool's gain (regression
to the mean, plus: anything that pushes the agent off its default answer helps when the default
is wrong). Instead, Opus reads each incident (no tools) and gives its best hypothesis and the
probability that it is the true cause, REPLICATES times; the mean probability stratifies
incidents by how uncertain the transcript leaves them, without looking at correctness.
Writes <out>/confidence.json: prompt_id -> {"p": mean, "ps": [...], "hypotheses": [...]}.

    python scripts/confidence_chive.py out=/workspace/agam/chive_runs_dj
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pydra

from detective_joracle.agent.loop import run_many
from detective_joracle.chive.data import Incident, load_incidents, read_ids
from detective_joracle.chive.llm import client, json_call
from detective_joracle.chive.prompts import seed

MODELS = ("claude-opus-5", "claude-sonnet-5")  # the second covers content-filter refusals
REPLICATES = 3
SYSTEM = """You see a conversation with a language model, a behaviour in its response, and how \
often it recurs. Give the ONE feature of the conversation you think most likely drives the \
behaviour, and the probability (0-1) that it is the true cause as controlled experiments \
(editing the conversation and re-running the model) would establish. Be calibrated."""
SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["hypothesis", "probability"],
    "properties": {"hypothesis": {"type": "string"}, "probability": {"type": "number"}},
}


async def one(inc: Incident) -> dict[str, Any] | None:
    user = seed(inc).rsplit("Positions are tokens", 1)[0]

    async def ask() -> dict[str, Any] | None:
        for model in MODELS:
            got = await json_call(client(), model=model, system=SYSTEM, user=user, schema=SCHEMA)
            if got and isinstance(got.get("probability"), int | float):
                return {**got, "model": model}
        return None

    got = [g for g in await asyncio.gather(*[ask() for _ in range(REPLICATES)]) if g]
    if not got:
        return None
    ps = [min(1.0, max(0.0, float(g["probability"]))) for g in got]
    return {
        "p": sum(ps) / len(ps),
        "ps": ps,
        "hypotheses": [g["hypothesis"] for g in got],
        "models": [g["model"] for g in got],
    }


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.out = pydra.REQUIRED
        self.ids = "data/chive/splits/all300.txt"
        self.incidents = "/workspace/agam/chive_bank/incidents"
        self.concurrency = 16

    def finalize(self) -> None:
        self.out, self.incidents = Path(self.out), Path(self.incidents)


@pydra.main(Config)
def main(cfg: Config) -> None:
    f = cfg.out / "confidence.json"
    done: dict[str, Any] = json.loads(f.read_text()) if f.exists() else {}
    incs = load_incidents(cfg.incidents, read_ids(Path(cfg.ids)))

    async def job(inc: Incident) -> None:
        got = await one(inc)
        if got:
            done[inc.prompt_id] = got

    todo = [i for p, i in incs.items() if p not in done]
    print(f"confidence: {len(todo)} incidents to ask")
    asyncio.run(run_many([lambda i=i: job(i) for i in todo], cfg.concurrency))
    f.write_text(json.dumps(done, ensure_ascii=False))
    ps = sorted(v["p"] for v in done.values())
    print(
        f"{len(done)} incidents; p quartiles {ps[len(ps) // 4]:.2f} / {ps[len(ps) // 2]:.2f} / "
        f"{ps[3 * len(ps) // 4]:.2f}; 25-75% band: {sum(0.25 <= x <= 0.75 for x in ps)}"
    )


if __name__ == "__main__":
    main()
