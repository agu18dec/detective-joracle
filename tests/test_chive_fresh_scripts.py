"""Offline tests for the fresh-incident pipeline scripts in scripts/chive_fresh (no Modal, no API):
bank derivations (make_k1, make_layers), judge v2 candidate / aggregation logic, the analysis
helpers, CHIVE runtime patch helpers and build_incidents on a tiny synthetic CHIVE run dir."""

import gzip
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

FRESH = Path(__file__).resolve().parents[1] / "scripts" / "chive_fresh"


def _load(rel: str) -> ModuleType:
    """Import a script by path (its own dir and scripts/chive_fresh on sys.path, as when run)."""
    path = FRESH / rel
    for p in (str(FRESH), str(path.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    name = "chive_fresh_" + rel.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------ banks
def _write_bank(root: Path, pid: str, layers: list[int], k: int, n: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rec = {
        "prompt_id": pid,
        "token_ids": list(range(n)),
        "k": k,
        "layers": layers,
        "readouts": {
            str(L): {str(p): [f"{pid} L{L} t{p} s{i}" for i in range(k)] for p in range(n)}
            for L in layers
        },
        "provenance": {"reader": "olens"},
    }
    with gzip.open(root / f"{pid}.json.gz", "wt") as f:
        json.dump(rec, f)


def _read(path: Path) -> Any:
    with gzip.open(path, "rt") as f:
        return json.load(f)


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_make_k1_keeps_the_first_sample(tmp_path: Path) -> None:
    mk = _load("analysis/make_k1.py")
    src, dst = tmp_path / "s3d", tmp_path / "olens_k1"
    for pid in ("p1", "p2"):
        _write_bank(src, pid, [20, 24], k=4)
    assert mk.convert(src, dst, workers=2) == (2, 2)
    rec = _read(dst / "p1.json.gz")
    assert rec["k"] == 1 and rec["token_ids"] == [0, 1, 2]
    assert rec["readouts"]["24"]["2"] == ["p1 L24 t2 s0"]
    assert rec["provenance"] == {"reader": "olens", "k1_from": str(src / "p1.json.gz")}
    assert mk.convert(src, dst, workers=2) == (0, 2)  # resumable: existing files skipped
    assert not list(dst.glob("*.tmp"))


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_make_layers_restricts_to_the_ladder(tmp_path: Path) -> None:
    ml = _load("analysis/make_layers.py")
    keep = ml.parse_layers("20:61:4")
    assert keep == {str(x) for x in (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60)}
    assert ml.parse_layers("3, 7") == {"3", "7"}
    src, dst = tmp_path / "jlens_all", tmp_path / "jlens_L20-60"
    _write_bank(src, "p1", list(range(0, 63, 1)), k=1)
    assert ml.convert(src, dst, keep, workers=1) == (1, 1)
    rec = _read(dst / "p1.json.gz")
    assert sorted(rec["readouts"], key=int) == sorted(keep, key=int)
    assert rec["readouts"]["40"] == _read(src / "p1.json.gz")["readouts"]["40"]
    assert rec["provenance"]["layers_from"].endswith("p1.json.gz") and rec["k"] == 1


# --------------------------------------------------------------------------- judge v2
def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _judge_batch(tmp_path: Path) -> Any:
    j2 = _load("analysis/judge2.py")
    ds, pts, runs = tmp_path / "datasets", tmp_path / "points", tmp_path / "runs"
    _jsonl(
        ds / "b1/data/incidents.jsonl",
        [
            {
                "prompt_id": "p1",
                "messages": [{"role": "user", "content": "Write about rain"}],
                "completion": "Grey skies",
                "behaviour": "Is it sad?",
                "k": 6,
                "n": 10,
            }
        ],
    )
    _jsonl(
        ds / "b1/data/ground_truth.jsonl",
        [
            {
                "prompt_id": "p1",
                "ruled_out": [
                    {"kind": "refuted", "cause": "poem genre"},
                    {"kind": "inert", "edit": "swap title"},
                    {"kind": "refuted", "cause": "user mood"},
                ],
            }
        ],
    )
    _jsonl(
        ds / "b1/work/binary_counterfactual_questions.jsonl",
        [
            {"prompt_id": "p1", "intervention": "remove rain", "is_true": True},
            {"prompt_id": "p1", "intervention": "swap title", "is_true": False},
        ],
    )
    (pts / "b1/points").mkdir(parents=True)
    (pts / "b1/points/p1.json").write_text(
        json.dumps({"points": ["rain cues sadness", "grey cues gloom", "short poem"]})
    )
    for d, scores in enumerate(([8, 2, 9], [7, 3, 8], [2, 2, 9])):  # P1, P3 obvious
        _jsonl(pts / f"b1/grades/plain/p1.d{d}.json", [{"point_scores": scores}])
    for arm in ("transcript", "olens_rl_B"):
        _jsonl(
            runs / f"b1/{arm}/p1.d0.json",
            [{"result": {"hypothesis": "rain", "evidence": "L40 says rain", "rival": "genre"}}],
        )
    b = j2.load_batch("b1", runs=runs, datasets=ds, points=pts, out_root=tmp_path / "out")
    return j2, b


def test_judge2_candidates(tmp_path: Path) -> None:
    j2, b = _judge_batch(tmp_path)
    assert j2.obvious_flags(b, "p1", 3) == [True, False, True]
    cands = j2.candidates(b, "p1")
    by = {c["id"]: c for c in cands}
    assert sorted(by) == ["P1", "P2", "P3", "R1", "R2"]  # inert ruled-out items are not rivals
    assert by["P2"] == {"id": "P2", "text": "grey cues gloom", "true": True, "obvious": False}
    assert (
        by["R2"]["text"] == "user mood (this is what drives the behaviour)" and not by["R2"]["true"]
    )
    assert [c["id"] for c in cands] == [c["id"] for c in j2.candidates(b, "p1")]  # seeded by pid
    assert j2.answer_text(b, "olens_rl_B", "p1", 0).startswith("HYPOTHESIS: rain\n\nEVIDENCE:")
    b.hyp_only = True
    assert j2.answer_text(b, "olens_rl_B", "p1", 0) == "HYPOTHESIS: rain"


def test_judge2_cached_run_and_aggregation(tmp_path: Path) -> None:
    j2, b = _judge_batch(tmp_path)
    assert b.out == tmp_path / "out" / "judge2" / "b1"
    labels = {
        "transcript": {"P1": "ASSERTED", "P2": "ABSENT", "P3": "DENIED", "R1": "ASSERTED"},
        "olens_rl_B": {"P1": "ASSERTED", "P2": "ASSERTED", "R1": "DENIED", "R2": "DENIED"},
    }
    preds = {"transcript": {"E1": 0.4, "E2": 0.6}, "olens_rl_B": {"E1": 0.9}}  # E2 missing -> 0.5
    for arm in labels:
        for stage, obj in (
            ("bullets", {"bullets": ["rain drives it"]}),
            ("select", {"labels": labels[arm]}),
            ("predict", {"p": preds[arm]}),
        ):
            _jsonl(b.out / stage / arm / "p1.d0.json", [obj])
    _jsonl(b.out / "predict/_null/p1.json", [{"p": {"E1": 0.5, "E2": 0.5}}])

    def no_api(*_: Any) -> Any:
        raise AssertionError("every stage is cached: no call expected")

    res = [j2.run_one(b, no_api, arm, "p1", 0, True) for arm in labels]
    null = {"p1": j2.run_null(b, no_api, "p1")}
    agg = j2.aggregate(b, res, null)
    t, o = agg["transcript"], agg["olens_rl_B"]
    assert t["true_asserted"].count(True) == 1 and len(t["true_asserted"]) == 3
    assert t["nonobv_asserted"] == [False] and o["nonobv_asserted"] == [True]
    assert t["true_denied"].count(True) == 1
    assert sorted(t["rival_asserted"]) == [False, True] and o["rival_denied"] == [True, True]
    assert (o["pos"], o["neg"]) == ([0.9], [0.5]) and (t["pos"], t["neg"]) == ([0.4], [0.6])
    assert j2.auroc(o["pos"], o["neg"]) == 1.0 and j2.auroc(t["pos"], t["neg"]) == 0.0
    assert j2.auroc(agg[j2.NULL_ARM]["pos"], agg[j2.NULL_ARM]["neg"]) == 0.5  # ties count half
    assert j2.auroc([], [0.1]) is None
    table = j2.table(agg, ["transcript", "olens_rl_B"])
    assert "(no hypothesis)" in table and " 0.000" in table and " 1.000" in table
    # without predictions requested, stage 3 is skipped even when cached
    assert j2.run_one(b, no_api, "transcript", "p1", 0, False)["pred"] is None


def test_judge2_report_selection_scores() -> None:
    jr = _load("analysis/judge2_report.py")
    s = jr.selection_scores({"P1": "ASSERTED", "P2": "DENIED", "R1": "ASSERTED"}, [True, False], 2)
    assert s == {"true": 0.5, "nonobv": 0.0, "denied": 0.5, "rival": 0.5}
    s = jr.selection_scores({}, [True], 0)
    assert s["nonobv"] is None and s["rival"] is None and s["true"] == 0.0


# --------------------------------------------------------------------- analysis helpers
def test_common_judged_draws_and_bootstrap(tmp_path: Path) -> None:
    c = _load("analysis/common.py")
    arm = tmp_path / "b1" / "olens_rl_B"
    arm.mkdir(parents=True)
    recs = [
        {"judge": {"score": 8}},
        {"judge": {"score": 3}},
        {"judge": {"score": 9}, "stopped_by": "salvaged finish"},
        {"judge": {"score": 9}, "stopped_by": "error: 429"},
        {"judge": {}},
        {"judge": {"score": 7}},  # d5: outside the base draws
    ]
    for d, r in enumerate(recs):
        (arm / f"p1.d{d}.json").write_text(json.dumps(r))
    got = c.judged(tmp_path, ["b1"], ["olens_rl_B"])
    assert got == {("b1", "p1"): {"olens_rl_B": [True, False]}}
    assert c.complete(got, ["olens_rl_B"], draws=2) == [("b1", "p1")]
    assert c.complete(got, ["olens_rl_B"]) == []
    m, lo, hi = c.paired_bootstrap([0.2, 0.2, 0.2], random.Random(0), 1000)
    assert m == pytest.approx(0.2) and lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
    m, lo, hi = c.paired_bootstrap([0.0, 1.0], random.Random(0), 2000)
    assert m == 0.5 and lo <= m <= hi and lo >= 0 and hi <= 1


def test_runtime_patch_helpers() -> None:
    k = _load("run_chive_keyless.py")
    assert [k.effort_for_budget(b) for b in (1024, 2048, 2049, 4096, 8192)] == [
        "low",
        "low",
        "medium",
        "medium",
        "high",
    ]
    assert k.decode_tool_arg("verdicts", '{"verdicts": [1, 2]}') == [1, 2]
    assert k.decode_tool_arg("verdicts", ' [{"a": 1}]') == [{"a": 1}]
    assert k.decode_tool_arg("x", '{"y": 1}') == {"y": 1}  # other keys: decoded, not unwrapped
    assert k.decode_tool_arg("x", "{not json") == "{not json"
    assert k.decode_tool_arg("x", "plain text") == "plain text"
    assert k.decode_tool_arg("x", [1]) == [1]


# --------------------------------------------------------------------- build_incidents
REPORT = """## Observed behavior
The model writes a sad poem.

## Core causes
1. The word rain cues sadness
   **Status:** supported
   - removing rain dropped the rate to 0/10
2. Poem form
   Status: not supported

## Negative evidence
A. Poem genre
   - a genre swap did nothing
"""


def _investigation(pid: str, effect_yes: int) -> dict[str, Any]:
    return {
        "prompt_id": pid,
        "classifier_question": "Is the poem sad?",
        "baseline_grader_yes": 6,
        "baseline_grader_verdicts": [True] * 6 + [False] * 4,
        "completions": ["Sunny day", "Grey skies weep"],
        "messages": [{"role": "user", "content": "Write a poem about rain."}],
        "structured_findings": {
            "structured_answer": REPORT,
            "experiment_summaries": [
                {"experiment_number": 2, "summary": "swap genre", "baseline_rate": "6/10"},
                {"experiment_number": 1, "summary": "remove rain", "is_key": True},
            ],
        },
        "experiment_log": [
            {
                "exp_num": 1,
                "grader_verdicts": [v < effect_yes for v in range(10)],
                "grader_yes": effect_yes,
            },
            {
                "exp_num": 2,
                "grader_verdicts": [v < 6 for v in range(10)],
                "grader_yes": 6,
                "turns_spec": [{"keep": 1}],
            },
        ],
    }


def _run_dir(root: Path) -> Path:
    run = root / "run"
    invs = [_investigation("p1", 0), _investigation("p2", 0), _investigation("p3", 5)]
    _jsonl(run / "investigations.jsonl", invs)
    _jsonl(
        run / "screening.jsonl",
        [{"prompt_id": i["prompt_id"], "messages": i["messages"]} for i in invs],
    )
    _jsonl(
        run / "verification.jsonl",
        [
            {"prompt_id": "p1", "score": 9},
            {"prompt_id": "p2", "score": 5},
            {"prompt_id": "p3", "score": 9},
        ],
    )
    _jsonl(
        run / "prompts.jsonl",
        [{"id": p, "source": "wildchat", "conversation_hash": f"h{p}"} for p in ("p1", "p2", "p3")],
    )
    return run


def test_build_incidents_gate_and_rows(tmp_path: Path) -> None:
    bi = _load("build_incidents.py")
    run = bi.load_run(_run_dir(tmp_path))
    drops: Any = __import__("collections").Counter()
    kept = bi.gate(run, 8.0, 30.0, drops)
    assert list(kept) == ["p1"]  # p2: verification 5 < 8; p3: max effect 10pp < 30
    assert drops == {"verification": 1, "effect": 1}
    g = kept["p1"]
    assert [e["number"] for e in g["exps"]] == [1, 2] and g["max_effect"] == pytest.approx(60.0)
    assert g["exps"][0]["baseline_rate"] == 0.6 and g["exps"][1]["turns"] == [{"keep": 1}]
    assert [t["title"] for t in g["targets"]] == ["The word rain cues sadness"]

    gt = bi.ground_truth_row("p1", g, [])
    assert gt["targets"][0] == {
        "id": "T1",
        "cause": "The word rain cues sadness",
        "status": "supported",
        "evidence": ["removing rain dropped the rate to 0/10"],
    }
    assert [(x["id"], x["basis"]) for x in gt["ruled_out"]] == [
        ("X1", "investigator"),
        ("X2", "experiment_le_15pp"),
    ]
    assert gt["movers"] == [] and gt["max_effect_pp"] == pytest.approx(60.0)
    ref = gt["judge_reference"]
    assert "[T1] The word rain cues sadness" in ref and "(small_effect) swap genre" in ref
    assert "(refuted) Poem genre — a genre swap did nothing" in ref

    claim = {
        "prompt_id": "p1",
        "claim_index": 0,
        "intervention": "remove the word rain",
        "k_baseline": 6,
        "n_baseline": 10,
        "k_counterfactual": 0,
        "n_counterfactual": 10,
        "counterfactual_rate": "0/10",
        "effect_pp": 60.0,
        "edit_cleanliness": "minor_targeted",
        "experiment_numbers": [1],
        "is_true": True,
        "chosen_completion": "Grey skies weep",
    }
    gt = bi.ground_truth_row("p1", g, [claim])
    assert gt["movers"][0]["edit"] == "remove the word rain" and gt["movers"][0]["rate"] == 0.0
    assert "remove the word rain -> rate moved 60 points" in gt["judge_reference"]
    assert bi.choose_completion(g["inv"], [claim]) == "Grey skies weep"
    assert bi.choose_completion(g["inv"], []) == "Sunny day"  # first baseline completion graded yes
    row = bi.incident_row("p1", g, run["pool"]["p1"], g["inv"]["messages"], "Grey skies weep")
    assert row["source_dataset"] == "allenai/WildChat-4.8M" and (row["k"], row["n"]) == (6, 10)


class _FakeTok:
    """Chat-template + offsets tokenizer stand-in (Qwen-style template, regex pieces)."""

    PIECE = re.compile(r"<\|im_(?:start|end)\|>|\n| ?[^\s<]+|\s|<")

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def apply_chat_template(self, msgs: list[dict[str, str]], tokenize: bool, **_: Any) -> Any:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text: str, **_: Any) -> dict[str, Any]:
        spans = [(m.start(), m.end()) for m in self.PIECE.finditer(text)]
        assert "".join(text[a:b] for a, b in spans) == text
        ids = [self.vocab.setdefault(text[a:b], len(self.vocab)) for a, b in spans]
        return {"input_ids": ids, "offset_mapping": spans}


def test_build_incidents_tokenize_transcript() -> None:
    bi = _load("build_incidents.py")
    msgs = [{"role": "user", "content": "Write a poem about rain."}]
    out = bi.tokenize_transcript(_FakeTok(), msgs, "Grey skies weep.")
    pieces, turn_of = out["pieces"], out["turn_of"]
    assert "".join(pieces).startswith("<|im_start|>user\nWrite")
    by_piece = dict(zip(pieces, turn_of, strict=True))
    assert by_piece[" rain."] == [1, "user"] and by_piece[" weep."] == [2, "assistant"]
    assert by_piece["<|im_start|>"] == [0, "control"] and by_piece["assistant"] == [0, "control"]
    with pytest.raises(ValueError, match="tokens >"):
        bi.tokenize_transcript(_FakeTok(), [{"role": "user", "content": "x " * 8000}], "y")


# ------------------------------------------------------------------ bank container code
def test_gw_src_defaults_to_the_vendored_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("modal")
    monkeypatch.delenv("GW_SRC", raising=False)
    rm = _load("banks/readers_modal.py")
    vendored = FRESH / "banks" / "vendor" / "gw_src" / "src"
    assert rm.GW_SRC == vendored
    for f in (
        "global_workspace/olens_suite/runner.py",
        "global_workspace/ola/verbalizer.py",
        "jlens/hooks.py",
    ):
        assert (rm.GW_SRC / f).is_file(), f
    assert "515f5977244f6837216a48d997299fb766f00f91" in (vendored.parent / "README.md").read_text()
    monkeypatch.setenv("GW_SRC", "/elsewhere/src")
    assert _load("banks/readers_modal.py").GW_SRC == Path("/elsewhere/src")
