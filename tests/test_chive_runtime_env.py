"""Offline tests for the pod runtime switches of the CHIVE harness (docs/chive_fresh_scaleup.md):
keyless OAuth from the Modal Dict, CHIVE_EXP_N, CHIVE_BASELINES, POINTS_ARMS / POINTS_DRAW_GLOB.
No network, no Modal: the Dict is a fake module injected into ``sys.modules``."""

import asyncio
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from detective_joracle.chive import llm
from detective_joracle.chive.data import Incident
from detective_joracle.chive.executor import Executor, samples_per_experiment
from detective_joracle.chive.tools import ChiveTools, baseline_counts

ROOT = Path(__file__).resolve().parents[1]


def _incident() -> Incident:
    return Incident(
        prompt_id="p1",
        messages=[{"role": "user", "content": "hi"}],
        completion="hello",
        behaviour="Is it a greeting?",
        k=20,
        n=30,
        token_ids=[0, 1],
        pieces=["hi", "hello"],
        turn_of=[(1, "user"), (2, "assistant")],
        criteria="",
        ground_truth={},
    )


# ------------------------------------------------------------------ keyless OAuth token
class _FakeDict:
    store: dict[str, Any] = {}
    calls: list[tuple[str, str | None]] = []

    @classmethod
    def from_name(cls, name: str, environment_name: str | None = None) -> "_FakeDict":
        cls.calls.append((name, environment_name))
        return cls()

    def get(self, key: str) -> Any:
        return self.store.get(key)


@pytest.fixture
def fake_modal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[_FakeDict]:
    mod = types.ModuleType("modal")
    mod.Dict = _FakeDict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", mod)
    monkeypatch.setattr(llm, "SOCK", str(tmp_path / "no-such.sock"))  # no opsec tunnel
    _FakeDict.store, _FakeDict.calls = {}, []
    return _FakeDict


def test_token_from_modal_dict_without_tunnel(fake_modal: type[_FakeDict]) -> None:
    fake_modal.store["token"] = {"token": "tok-1", "pushed_at": 1.0}
    assert llm._token_once() == "tok-1"
    assert fake_modal.calls == [(llm.DICT_NAME, llm.DICT_ENV)]
    assert (llm.DICT_NAME, llm.DICT_ENV) == ("anthropic-oauth", "agam")


@pytest.mark.parametrize("entry", [None, "tok", {"pushed_at": 1.0}, {"token": ""}])
def test_missing_dict_token_asks_for_a_push(fake_modal: type[_FakeDict], entry: Any) -> None:
    fake_modal.store["token"] = entry
    with pytest.raises(RuntimeError, match="push-judge-token"):
        llm._token_once()


def test_no_modal_package_is_a_connection_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(llm, "SOCK", str(tmp_path / "no-such.sock"))
    monkeypatch.setitem(sys.modules, "modal", None)  # import modal -> ImportError
    with pytest.raises(ConnectionError):
        llm._token_once()


def test_token_refresh_period_and_repush(
    fake_modal: type[_FakeDict], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert llm.REFRESH_S == 5 * 60
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    tok = llm._Token()
    fake_modal.store["token"] = {"token": "old"}
    assert tok.get() == "old"
    fake_modal.store["token"] = {"token": "new"}  # re-pushed from the Mac
    now[0] += llm.REFRESH_S - 1
    assert tok.get() == "old"  # cached within the refresh period
    now[0] += 2
    assert tok.get() == "new"
    # an empty Dict while a token is held: keep the held one, retry a minute later
    fake_modal.store.clear()
    now[0] += llm.REFRESH_S + 1
    assert tok.get() == "new"
    tok.invalidate("new")
    fake_modal.store["token"] = {"token": "newer"}
    assert tok.get() == "newer"


def test_first_fetch_without_any_token_raises(fake_modal: type[_FakeDict]) -> None:
    with pytest.raises(RuntimeError):
        llm._Token().get()


def test_api_key_wins_over_the_dict(
    fake_modal: type[_FakeDict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = llm.client()
    assert c.api_key == "sk-test" and fake_modal.calls == []


# --------------------------------------------------------------------- CHIVE_EXP_N
def test_samples_per_experiment_env() -> None:
    assert samples_per_experiment({}) == 12
    assert samples_per_experiment({"CHIVE_EXP_N": "30"}) == 30
    assert Executor.N == samples_per_experiment()  # class attribute, read at import


def test_exp_n_is_read_at_import(tmp_path: Path) -> None:
    import os
    import subprocess

    code = "from detective_joracle.chive.executor import Executor; print(Executor.N)"
    env = {**os.environ, "CHIVE_EXP_N": "30", "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "30", out.stderr


# ------------------------------------------------------------------ CHIVE_BASELINES
def test_baseline_counts_override(tmp_path: Path) -> None:
    inc = _incident()
    assert baseline_counts(inc, {}) == (20, 30)
    f = tmp_path / "baselines.json"
    f.write_text(json.dumps({"p1": {"k": 9, "n": 30, "rate": 0.3, "chive_k": 20, "chive_n": 30}}))
    assert baseline_counts(inc, {"CHIVE_BASELINES": str(f)}) == (9, 30)
    other = Incident(**{**inc.__dict__, "prompt_id": "p2"})
    assert baseline_counts(other, {"CHIVE_BASELINES": str(f)}) == (20, 30)  # not re-measured


class _FakeExecutor:
    async def run(self, inc: Incident, turns: Any) -> dict[str, Any]:
        return {"k": 3, "n": 4, "rate": 0.75, "completions": ["a"] * 4, "verdicts": [True] * 4}


def test_experiment_reports_the_remeasured_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = ChiveTools(_incident(), None, executor=_FakeExecutor(), experiments=2)  # type: ignore[arg-type]
    out = asyncio.run(tools.run_experiment({"turns": [{"keep": 1}]}, {}))
    assert "3 of 4 responses (75%); baseline 20/30 (67%)" in out
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"p1": {"k": 9, "n": 30}}))
    monkeypatch.setenv("CHIVE_BASELINES", str(f))
    out = asyncio.run(tools.run_experiment({"turns": [{"keep": 1}]}, {}))
    assert "baseline 9/30 (30%)" in out


# -------------------------------------------------------- POINTS_ARMS / POINTS_DRAW_GLOB
def _load_points(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> Any:
    for k in ("POINTS_ARMS", "POINTS_DRAW_GLOB"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location(
        "points_chive", ROOT / "scripts" / "points_chive.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_points_default_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_points(monkeypatch, {})
    assert mod.ARMS == mod.DEFAULT_ARMS and mod.DRAW_GLOB == "*.d*.json"


def test_points_env_arms_and_draw_glob(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for arm in ("transcript", "olens_k1"):
        (tmp_path / arm).mkdir()
        for d in range(7):
            (tmp_path / arm / f"p1.d{d}.json").write_text("{}")
    arms = {a: str(tmp_path / a) for a in ("transcript", "olens_k1")}
    mod = _load_points(
        monkeypatch, {"POINTS_ARMS": json.dumps(arms), "POINTS_DRAW_GLOB": "*.d[0-4].json"}
    )
    assert mod.ARMS == arms and mod.DRAW_GLOB == "*.d[0-4].json"
    args = types.SimpleNamespace(out=tmp_path / "out")
    (args.out / "points").mkdir(parents=True)
    (args.out / "points" / "p1.json").write_text("{}")  # only incidents with points are graded
    files = mod.answer_files(args, {"p1": _incident()})
    assert sorted({a for a, _ in files}) == ["olens_k1", "transcript"]
    assert {f.name for _, f in files} == {f"p1.d{d}.json" for d in range(5)}
