"""Offline tests for the CHIVE fresh scale-up additions: atomic record writes, the `recur` search view,
the `llmfilter` fallback, and variant gating of the system prompt (no variant = unchanged prompt)."""

import asyncio
import importlib.util
import json
import threading
from pathlib import Path

from detective_joracle.chive import tools as T
from detective_joracle.chive.prompts import system_prompt
from detective_joracle.chive.readers import Reader
from detective_joracle.chive.tools import ChiveTools

from test_chive import incident

ROOT = Path(__file__).resolve().parents[1]


def _run_chive():
    spec = importlib.util.spec_from_file_location("run_chive_mod", ROOT / "scripts" / "run_chive.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reader(inc, hits: dict[int, int], single: tuple[int, ...] = ()) -> Reader:
    """k=4 samples x 3 layers per position; `hits[p]` samples per layer at p say 'rain is sad';
    positions in `single` have exactly one such sample, at layer 20 only."""
    cells = {}
    for p in range(inc.n_tokens):
        n = hits.get(p, 0)
        cells[p] = {L: ["rain is sad"] * n + ["something else"] * (4 - n) for L in (20, 40, 60)}
        if p in single:
            cells[p][20] = ["rain is sad"] + ["something else"] * 3
    return Reader(name="olens_rl", kind="olens", layers=[20, 40, 60], k=4, cells=cells, source_id=inc.prompt_id,
                  n_tokens=inc.n_tokens)


def test_atomic_write_concurrent(tmp_path: Path) -> None:
    rc = _run_chive()
    f = tmp_path / "rec.d0.json"
    payloads = [json.dumps({"i": i, "pad": "x" * (5000 - 400 * i)}) for i in range(10)]
    ts = [threading.Thread(target=rc._atomic_write, args=(f, p)) for p in payloads]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert json.loads(f.read_text())["i"] in range(10)  # always one complete record, never interleaved
    assert not list(tmp_path.glob(".*.tmp"))  # no temp files left behind, and the record glob never sees them
    assert sorted(p.name for p in tmp_path.glob("*.d*.json")) == ["rec.d0.json"]


def test_recur_marks_recurring_vs_single() -> None:
    inc = incident()
    r = _reader(inc, {7: 4}, single=(3,))  # t7 ' rain': every sample, every layer; t3: one sample, one layer
    tools = ChiveTools(inc, r, None, 0, recur=True)
    out = tools.search({"pattern": "rain is sad", "k": 5}, {})
    assert "Base rate:" in out
    line7 = next(x for x in out.splitlines() if x.startswith("t7 "))
    line3 = next(x for x in out.splitlines() if x.startswith("t3 "))
    assert "RECURRING" in line7 and "12/12" in line7
    assert "SINGLE/WEAK" in line3


def test_recur_off_keeps_old_search_output() -> None:
    inc = incident()
    r = _reader(inc, {7: 4})
    out = ChiveTools(inc, r, None, 0).search({"pattern": "rain is sad"}, {})
    assert "Base rate:" not in out and "RECURRING" not in out


def test_llmfilter_falls_back_to_raw_on_outage(monkeypatch) -> None:
    inc = incident()
    r = _reader(inc, {7: 4})

    def boom():
        raise ConnectionError("no tunnel")

    monkeypatch.setattr("detective_joracle.chive.llm.client", boom)
    tools = ChiveTools(inc, r, None, 0, llmfilter=True)
    out = asyncio.run(tools.call("inspect", {"position": 7}))
    assert "=== t7" in out and "filter unavailable" in out


def test_variant_gating_leaves_default_prompt_unchanged() -> None:
    inc = incident()
    r = _reader(inc, {})
    base = system_prompt(r, 0, 12, ("textfirst", "absence"))
    for v, marker in (("recur", "Recurrence is the evidence"), ("llmfilter", "Filtered readouts"),
                      ("multicand", "Several candidates"), ("antilure", "Weighing a single vivid readout")):
        assert marker not in base
        with_v = system_prompt(r, 0, 12, ("textfirst", "absence", v))
        assert marker in with_v and with_v.startswith(base.split("## Absences")[0])
    assert system_prompt(None, 0, 12, ("recur", "llmfilter", "multicand", "antilure")) == system_prompt(None, 0, 12, ())
