"""Tested-alternatives candidate sets: the true cause against the alternatives CHIVE tested.

For each incident: the truth is the first established cause (ground_truth.targets[0]); the
distractors are the incident's ruled-out alternatives -- hypotheses its experiments refuted, and
edits that did not move the behaviour (rewritten as the hypothesis they tested). One model call restates
every option as one neutral sentence in the same style, without evidence or results, so wording
does not give the answer away. Options are lettered in an order fixed by a hash. A blind picker
that sees only the behaviour and the options gives the leak check. Writes a jsonl in the format
run_chive.py's `choices=` reads, plus <name>_leak.json.

    python scripts/build_tested_choices.py ids=data/chive/splits/climb_test204.txt \\
        out=data/chive/choices/tested_test204.jsonl
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pydra

from detective_joracle.agent.loop import run_many
from detective_joracle.chive.data import Incident, load_incidents, read_ids
from detective_joracle.chive.llm import client, json_call

MODELS = ("claude-opus-5", "claude-sonnet-5")  # the second covers content-filter refusals
LETTERS = "ABCDEFGHIJ"
MAX_DISTRACTORS = 7

REWRITE = """You rewrite candidate explanations of a language model's behaviour into one uniform \
style. You get the behaviour and a numbered list of items. Each item is either a hypothesis \
about what drives the behaviour, possibly followed by evidence, or a description of an edit to \
the conversation that was tested (rewrite that one as the hypothesis the edit tested: that the \
edited feature drives the behaviour). For every item, write ONE neutral sentence of 20-35 words \
of the form "The model ... because/when ...", stating only the proposed cause and mechanism -- no \
evidence, no results, no hedging, no first person. Keep the items' meanings distinct. Return \
the sentences in the same order."""
REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["sentences"],
    "properties": {"sentences": {"type": "array", "items": {"type": "string"}}},
}
PICK = """You see a description of a language model's behaviour and candidate explanations. \
Exactly one is the true cause established by experiments. Pick the most likely one."""
PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["letter"],
    "properties": {"letter": {"type": "string"}},
}


async def call(system: str, user: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    for model in MODELS:
        got = await json_call(client(), model=model, system=system, user=user, schema=schema)
        if got:
            return got
    return None


async def build(inc: Incident) -> dict[str, Any] | None:
    gt = inc.ground_truth
    truth = gt["targets"][0]["cause"]
    excl = [x["cause"] if x["kind"] == "refuted" else x["edit"] for x in gt["ruled_out"]]
    excl = excl[:MAX_DISTRACTORS]
    if len(excl) < 2:
        return None
    items = [truth, *excl]
    user = f"BEHAVIOUR: {inc.behaviour}\n\nITEMS:\n" + "\n".join(
        f"{i + 1}. {x}" for i, x in enumerate(items)
    )
    got = await call(REWRITE, user, REWRITE_SCHEMA)
    if not got or len(got.get("sentences", [])) != len(items):
        return None
    sents = [str(x).strip() for x in got["sentences"]]
    order = sorted(
        range(len(sents)),
        key=lambda i: hashlib.sha1(f"{inc.prompt_id}:{i}".encode()).hexdigest(),
    )
    options = [
        {
            "letter": LETTERS[j],
            "text": sents[i],
            "truth": i == 0,
            "source": "truth" if i == 0 else "tested_alternative",
        }
        for j, i in enumerate(order)
    ]
    truth_letter = next(o["letter"] for o in options if o["truth"])
    blind = f"BEHAVIOUR: {inc.behaviour}\n\nCANDIDATES:\n" + "\n".join(
        f"{o['letter']}) {o['text']}" for o in options
    )
    picks = await asyncio.gather(*[call(PICK, blind, PICK_SCHEMA) for _ in range(3)])
    hits = sum(
        bool(p and str(p.get("letter", "")).strip().upper()[:1] == truth_letter) for p in picks
    )
    return {
        "prompt_id": inc.prompt_id,
        "options": options,
        "truth": truth_letter,
        "options_only_correct": hits,
        "options_only_samples": 3,
    }


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.ids = "data/chive/splits/climb_test204.txt"
        self.out = "data/chive/choices/tested_test204.jsonl"
        self.concurrency = 16

    def finalize(self) -> None:
        self.out = Path(self.out)


@pydra.main(Config)
def main(cfg: Config) -> None:
    incs = load_incidents(read_ids(Path(cfg.ids))).values()
    rows: list[dict[str, Any]] = []

    async def job(inc: Incident) -> None:
        r = await build(inc)
        if r:
            rows.append(r)

    asyncio.run(run_many([lambda i=i: job(i) for i in incs], cfg.concurrency))
    rows.sort(key=lambda r: r["prompt_id"])
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n_opt = [len(r["options"]) for r in rows]
    chance = sum(1 / n for n in n_opt) / len(n_opt)
    leak = sum(r["options_only_correct"] for r in rows) / (3 * len(rows))
    summary = {
        "n_incidents": len(rows),
        "options_mean": sum(n_opt) / len(n_opt),
        "chance": chance,
        "options_only": leak,
    }
    cfg.out.with_name(cfg.out.stem + "_leak.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
