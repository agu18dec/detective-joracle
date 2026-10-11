"""Offline tests for the CHIVE module: a tiny incident and bank, a scripted agent, no network."""

import asyncio
import gzip
import json
from pathlib import Path

import pytest

from detective_joracle.agent.backends import FakeBackend
from detective_joracle.agent.loop import Budget, RunRecord, run_tool_loop_async
from detective_joracle.chive.data import Incident
from detective_joracle.chive.executor import apply_edit
from detective_joracle.chive.prompts import seed, system_prompt
from detective_joracle.chive.readers import load_reader
from detective_joracle.chive.tools import SUBMIT, ChiveTools, schemas

PIECES = [
    "<|im_start|>",
    "user",
    "\n",
    "Write",
    " a",
    " poem",
    " about",
    " rain",
    "<|im_end|>",
    "\n",
    "<|im_start|>",
    "assistant",
    "\n",
    "Grey",
    " skies",
    " weep",
]
TURNS = [(0, "control")] * 3 + [(1, "user")] * 5 + [(0, "control")] * 4 + [(2, "assistant")] * 4


def incident() -> Incident:
    return Incident(
        prompt_id="p1",
        messages=[{"role": "user", "content": "Write a poem about rain"}],
        completion="Grey skies weep",
        behaviour="Is the poem sad?",
        k=20,
        n=30,
        token_ids=list(range(len(PIECES))),
        pieces=PIECES,
        turn_of=TURNS,
        criteria="[T1] the word rain cues sadness",
    )


def write_bank(root: Path, inc: Incident, text: str) -> None:
    readouts = {
        str(layer): {
            str(p): [f"{text} at t{p} L{layer} s{i}" for i in range(4)] for p in range(inc.n_tokens)
        }
        for layer in (20, 40, 60)
    }
    with gzip.open(root / f"{inc.prompt_id}.json.gz", "wt") as f:
        json.dump({"token_ids": inc.token_ids, "k": 4, "readouts": readouts}, f)


def test_incident_positions() -> None:
    inc = incident()
    assert inc.response_start == 12
    assert inc.where(7) == "t7 ' rain' (T1 user)"
    assert "[[ rain]]" in inc.context(7)


def test_reader_and_shuffle(tmp_path: Path) -> None:
    inc = incident()
    write_bank(tmp_path, inc, "sad mood")
    r = load_reader("olens_rl", "olens", tmp_path, inc)
    assert r.layers == [20, 40, 60] and r.k == 4 and len(r.cell(7)[40]) == 4
    other = Incident(**{**inc.__dict__, "prompt_id": "p2"})
    write_bank(tmp_path, other, "happy mood")
    s = load_reader("olens_rl", "olens", tmp_path, inc, partner=other)
    assert s.source_id == "p2" and s.cell(7)[40][0].startswith("happy")


def test_scripted_episode(tmp_path: Path) -> None:
    inc = incident()
    write_bank(tmp_path, inc, "the user wants a sad mood")
    reader = load_reader("olens_rl", "olens", tmp_path, inc)
    tools = ChiveTools(inc, reader)
    backend = FakeBackend(
        [
            ("", [("show_tokens", {"turn": "T1", "phrase": "rain"})], 10),
            ("", [("search", {"pattern": "sad", "positions": "t3-t8", "k": 3})], 10),
            ("", [("inspect", {"position": 7, "window": 1})], 10),
            ("", [("inspect", {"position": 999})], 10),  # malformed: ERROR, run continues
            ("", [("notebook", {"action": "write", "text": "rain -> sad"})], 10),
            (
                "",
                [
                    (
                        SUBMIT,
                        {
                            "hypothesis": "the word rain cues a sad mood",
                            "evidence": "t7 L40",
                            "rival": "poem genre",
                            "rival_ruled_out_by": "readouts name rain",
                        },
                    )
                ],
                10,
            ),
        ],
        terminal=SUBMIT,
    )
    rec = RunRecord(organism="p1", condition="test", arm="olens_rl", auditor="fake", seed=0)
    asyncio.run(
        run_tool_loop_async(
            tools,
            schemas(reader, 0),
            system_prompt(reader, 0, 12),
            seed(inc),
            backend,
            Budget(),
            rec,
            terminal=SUBMIT,
        )
    )
    assert rec.stopped_by == "finish"
    assert tools.finished and "rain" in tools.finished["hypothesis"]
    assert [e["tool"] for e in tools.events][:3] == ["show_tokens", "search", "inspect"]
    assert tools.events[2]["positions"] == [7]
    assert any(e.get("error") for e in tools.events)  # the malformed inspect was logged
    assert "neighbours" in tools.log[2].output


def test_transcript_arm_has_no_readout_tools() -> None:
    names = [s["function"]["name"] for s in schemas(None, 0)]
    assert names == ["show_tokens", "notebook", SUBMIT]
    assert "readouts" not in system_prompt(None, 0, 12).lower()


def test_apply_edit() -> None:
    msgs = [{"role": "user", "content": "Write a poem about rain"}]
    out = apply_edit(msgs, [{"keep": "T1", "replace": [{"old": "rain", "new": "sun"}]}])
    assert out == [{"role": "user", "content": "Write a poem about sun"}]
    with pytest.raises(ValueError):
        apply_edit(msgs, [{"keep": 1, "replace": [{"old": "snow", "new": "x"}]}])
    with pytest.raises(ValueError):
        apply_edit(msgs, [{"keep": 1}, {"role": "assistant", "content": "x"}])


def test_candidate_selection(tmp_path: Path) -> None:
    inc = incident()
    cands = [("A", "the word rain cues sadness"), ("B", "poems are sad by default")]
    first = seed(inc, cands)
    assert "## Candidate causes" in first and "B) poems are sad by default" in first
    sub = schemas(None, 0, ["A", "B"])[-1]["function"]["parameters"]
    assert sub["required"] == ["choice", "hypothesis"] and sub["properties"]["choice"]["enum"]
    tools = ChiveTools(inc, None, letters=["A", "B"])
    backend = FakeBackend(
        [
            ("", [(SUBMIT, {"hypothesis": "rain", "choice": "C"})], 10),  # invalid: ERROR, continue
            ("", [(SUBMIT, {"hypothesis": "rain cues sadness", "choice": "a)"})], 10),
        ],
        terminal=SUBMIT,
    )
    rec = RunRecord(organism="p1", condition="test", arm="transcript", auditor="fake", seed=0)
    asyncio.run(
        run_tool_loop_async(
            tools,
            schemas(None, 0, ["A", "B"]),
            "sys",
            first,
            backend,
            Budget(),
            rec,
            terminal=SUBMIT,
        )
    )
    assert tools.finished and tools.finished["choice"] == "A"
    assert tools.events[0].get("error")
