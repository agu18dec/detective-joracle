"""Offline tests for the CHIVE module: a tiny incident and bank, a scripted agent, no network."""

import asyncio
import gzip
import json
from pathlib import Path

import pytest

from detective_joracle.agent.backends import FakeBackend
from detective_joracle.agent.loop import Budget, RunRecord, run_tool_loop_async
from detective_joracle.chive.data import DATASET, Incident, load_incidents
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
        ground_truth={"targets": [{"id": "T1", "cause": "the word rain cues sadness"}]},
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
    assert sub["required"] == ["choice", "probabilities", "hypothesis"]
    tools = ChiveTools(inc, None, letters=["A", "B"])
    backend = FakeBackend(
        [
            ("", [(SUBMIT, {"hypothesis": "rain", "choice": "C"})], 10),  # invalid: ERROR, continue
            ("", [(SUBMIT, {"hypothesis": "rain", "choice": "A"})], 10),  # no probabilities: ERROR
            (
                "",
                [
                    (
                        SUBMIT,
                        {
                            "hypothesis": "rain cues sadness",
                            "choice": "a)",
                            "probabilities": {"A": 3, "B": 1},
                        },
                    )
                ],
                10,
            ),
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
    assert tools.finished["probabilities"] == {"A": 0.75, "B": 0.25}  # normalised
    assert tools.events[0].get("error") and tools.events[1].get("error")


def test_position_shuffle_control(tmp_path: Path) -> None:
    inc = incident()
    write_bank(tmp_path, inc, "sad mood")
    r = load_reader("olens_rl", "olens", tmp_path, inc, permute=True)
    again = load_reader("olens_rl", "olens", tmp_path, inc, permute=True)
    assert r.source_id == "p1" and r.cells == again.cells  # same incident, fixed permutation
    moved = [p for p in range(inc.n_tokens) if r.cell(p)[40][0] != f"sad mood at t{p} L40 s0"]
    assert len(moved) > inc.n_tokens // 2  # most positions read another position's readouts


def test_aux_reader_logs_real_tool_name(tmp_path: Path) -> None:
    inc = incident()
    write_bank(tmp_path, inc, "sad mood")
    main = load_reader("olens_rl", "olens", tmp_path, inc)
    aux = load_reader("jlens", "olens", tmp_path, inc)
    tools = ChiveTools(inc, main, aux=aux)
    asyncio.run(tools.call("jlens_inspect", {"position": 7}))
    assert tools.events[-1]["tool"] == "jlens_inspect" and tools.reader is main


def test_load_incidents_joins_the_dataset_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inc = incident()
    rows = {
        "incidents": {
            "prompt_id": "p1",
            "messages": inc.messages,
            "completion": inc.completion,
            "behaviour": inc.behaviour,
            "k": inc.k,
            "n": inc.n,
        },
        "tokens": {
            "prompt_id": "p1",
            "token_ids": inc.token_ids,
            "pieces": inc.pieces,
            "turn_of": [list(t) for t in inc.turn_of],
        },
        "ground_truth": {"prompt_id": "p1", "judge_reference": inc.criteria, **inc.ground_truth},
    }
    cache = tmp_path / "detective_joracle" / DATASET
    cache.mkdir(parents=True)
    for config, row in rows.items():  # pre-filled cache: no download
        (cache / f"{config}.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    got = load_incidents(["p1"])["p1"]
    assert got.turn_of == inc.turn_of and got.criteria == inc.criteria
    assert got.ground_truth["targets"] == inc.ground_truth["targets"]
    with pytest.raises(KeyError):
        load_incidents(["missing"])


def test_load_incidents_from_a_local_dataset_dir(tmp_path: Path) -> None:
    inc = incident()
    (tmp_path / "data").mkdir()
    rows = {
        "incidents": {
            "prompt_id": "p1",
            "messages": inc.messages,
            "completion": inc.completion,
            "behaviour": inc.behaviour,
            "k": inc.k,
            "n": inc.n,
        },
        "tokens": {
            "prompt_id": "p1",
            "token_ids": inc.token_ids,
            "pieces": inc.pieces,
            "turn_of": [list(t) for t in inc.turn_of],
        },
        "ground_truth": {"prompt_id": "p1", "judge_reference": inc.criteria},
    }
    for config, row in rows.items():
        (tmp_path / "data" / f"{config}.jsonl").write_text(json.dumps(row) + "\n")
    assert load_incidents(["p1"], str(tmp_path))["p1"].pieces == inc.pieces


def test_describe_flags_sparse_readouts(tmp_path: Path) -> None:
    from detective_joracle.chive.readers import Reader, describe

    cells = {3: {20: ["a"]}, 5: {20: ["b"]}}
    sparse = Reader("olens_rl", "olens", [20], 1, cells, "p1", n_tokens=16, coverage_note="x")
    assert "only 2 of 16 tokens (x)" in describe(sparse)
    assert "every token" not in describe(sparse)
    jl = Reader("jlens", "jlens", [20], 1, cells, "p1", n_tokens=16)
    assert "every token" not in describe(jl)
    dense = Reader("olens_rl", "olens", [20], 1, {p: {20: ["a"]} for p in range(4)}, "p1", 4)
    assert "only" not in describe(dense)


def test_yesno_items_tools_and_scoring() -> None:
    from detective_joracle.chive.edits import edit_items, score_items

    gt = {
        "movers": [{"edit": "Remove the word rain.", "effect_pp": 70.0}],
        "ruled_out": [
            {"kind": "inert", "basis": "claim_bank", "edit": "Add a title.", "effect_pp": 5.0},
            {
                "kind": "inert",
                "basis": "experiment_le_15pp",
                "edit": "states result",
                "effect_pp": 3,
            },
            {"kind": "refuted", "cause": "not an edit"},
        ],
    }
    inc = Incident(**{**incident().__dict__, "ground_truth": gt})
    items = edit_items(inc)
    assert [it.id for it in items] == ["E1", "E2"]
    assert sorted(it.label for it in items) == [False, True]
    assert edit_items(inc) == items  # fixed order
    ids = [it.id for it in items]
    sch = {s["function"]["name"]: s["function"]["parameters"] for s in schemas(None, 0, edits=ids)}
    assert "predictions" in sch[SUBMIT]["required"]
    tools = ChiveTools(inc, None, edits=ids)
    with pytest.raises(ValueError):
        asyncio.run(tools.call(SUBMIT, {"hypothesis": "h", "predictions": {"E1": 0.9}}))
    yes = next(it.id for it in items if it.label)
    no = next(it.id for it in items if not it.label)
    asyncio.run(tools.call(SUBMIT, {"hypothesis": "h", "predictions": {yes: 0.9, no: 0.2}}))
    assert tools.finished is not None
    s = score_items(items, tools.finished["predictions"])
    assert s["accuracy"] == 1.0 and abs(s["brier"] - (0.01 + 0.04) / 2) < 1e-9
    assert "E1)" in seed(inc, edits=[(it.id, it.text) for it in items])
