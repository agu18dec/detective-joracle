"""Run an unmodified CHIVE entrypoint with keyless Claude and the runtime fixes it needs today.

    python run_chive_keyless.py MODULE [ARGS ...]
    # e.g. (cwd = the CHIVE checkout, its venv):
    #   python run_chive_keyless.py chive.pipeline.run_pipeline --config RUN_DIR/config.json

Single-turn WildChat runs: CHIVE's own prompts are correct as-is (no on-policy rewording). The
patches below are applied in-process (CHIVE's files are never edited):

1. Keyless OAuth (``anthropic_oauth.install``): every Anthropic client reads the token pushed into
   the Modal Dict ``anthropic-oauth``; a re-push reaches the running pipeline within 5 minutes.
2. Modal 303 follow-redirects: Modal web endpoints answer requests running > ~150 s with a 303 to
   a result URL; httpx does not follow redirects by default, so the vLLM client raised
   APIStatusError 303 and killed stage 1. ``httpx.AsyncClient`` now defaults
   ``follow_redirects=True``.
3. Opus thinking -> adaptive effort: claude-opus-5 400s on ``thinking.type.enabled`` ("use
   thinking.type.adaptive and output_config.effort"). CHIVE's stages pass ``thinking_budget``;
   it is mapped to an adaptive effort (``effort_for_budget``: <= 2048 -> low, <= 4096 -> medium,
   larger -> high; the grader's 2048 at medium overran its 8192 max_tokens and quarantined ~4% of
   investigations).
4. Stringified tool args: Sonnet 5 sometimes returns a nested array argument as a JSON string,
   e.g. the grader's ``verdicts='{"verdicts":[...]}'``, which fails pydantic and quarantined ~6%
   of fresh1's investigations. Such strings are decoded (unwrapping a same-key object) before
   validation (``decode_tool_arg``).
"""

import json
import runpy
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def effort_for_budget(thinking_budget: int) -> str:
    """CHIVE's extended-thinking budget -> the adaptive-thinking effort used instead."""
    if thinking_budget <= 2048:
        return "low"
    return "medium" if thinking_budget <= 4096 else "high"


def decode_tool_arg(key: str, v: Any) -> Any:
    """A JSON-looking string argument decoded; ``{key: x}`` unwrapped to ``x``; else unchanged."""
    if isinstance(v, str) and v.lstrip()[:1] in "[{":
        try:
            d = json.loads(v)
        except ValueError:
            return v
        if isinstance(d, dict) and set(d) == {key}:
            d = d[key]
        return d
    return v


def install_patches() -> None:
    """Apply patches 1-4 (needs the CHIVE package and global-workspace's judge_auth)."""
    import anthropic_oauth

    anthropic_oauth.install()

    import httpx

    _init = httpx.AsyncClient.__init__

    def _follow(self: Any, *a: Any, **k: Any) -> None:
        k.setdefault("follow_redirects", True)
        _init(self, *a, **k)

    httpx.AsyncClient.__init__ = _follow  # type: ignore[method-assign]

    from chive.pipeline import llm_client as _lc

    _call = _lc.AnthropicLLMClient.call

    async def _adaptive(
        self: Any, *a: Any, thinking_budget: Any = None, effort: Any = None, **k: Any
    ) -> Any:
        if thinking_budget and effort is None:
            effort = effort_for_budget(thinking_budget)
            thinking_budget = None
        return await _call(self, *a, thinking_budget=thinking_budget, effort=effort, **k)

    _lc.AnthropicLLMClient.call = _adaptive

    from chive.pipeline import tool_validate as _tv

    _cft = _tv.call_for_tool

    async def _call_for_tool(*a: Any, **k: Any) -> Any:
        out = await _cft(*a, **k)
        if out is not None and isinstance(out[0], dict):
            out = ({key: decode_tool_arg(key, v) for key, v in out[0].items()}, *out[1:])
        return out

    _tv.call_for_tool = _call_for_tool


def main() -> None:
    install_patches()
    mod, sys.argv = sys.argv[1], [sys.argv[1]] + sys.argv[2:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
