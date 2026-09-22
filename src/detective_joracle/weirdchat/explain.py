"""Explain mode: one investigator run on one WeirdChat pattern.

The quirk game's tools are reused as they are (``LiveTools``): chat, raw completion and lens
readouts on any conversation, against the stock model served as ``base``. Two things differ.

1. The study's own rollouts are loaded as conversations before the agent starts (``w000`` …), so
   the lens can be read on the exact matched/unmatched pair the judge labelled rather than only on
   samples the agent draws itself.
2. ``finish`` takes mechanisms, not predictions — there is no planted quirk to name, no judge and
   no closed-set stage, so a run ends with hypotheses and the experiments that would test them.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..agent.loop import Backend, Budget, RunRecord, run_tool_loop
from ..tools.live import LIVE_CONDITION, READ_LAYERS, Conversation, LensClient, LiveTools
from ..tools.live import tool_schemas as live_tool_schemas
from . import prompts as wp
from .data import Pattern

CONDITION = "weirdchat"
ORGANISM = "base"  # the stock model: WeirdChat found these behaviors in the released weights
ARM = "olens"
MAX_MECHANISM_CHARS = 4000


class ExplainTools(LiveTools):
    """``LiveTools`` with a mechanisms-shaped ``finish`` and seeded study conversations."""

    def seed_conversation(self, cid: str, user: str, completion: str) -> Conversation:
        """Load a rollout the study produced as a readable conversation under ``cid``."""
        conv = Conversation(
            cid,
            [{"role": "user", "content": user}],
            completion,
            "organism",
            "assistant",
        )
        self.convs[cid] = conv
        return conv

    def finish_mechanisms(self, mechanisms: Sequence[Mapping[str, Any]], summary: str = "") -> str:
        """Lock in the ranked mechanisms (empty ones dropped, at most ``MAX_MECHANISMS``)."""
        out = [
            {
                "mechanism": str(m.get("mechanism", "")).strip()[:MAX_MECHANISM_CHARS],
                "evidence": str(m.get("evidence", "")).strip()[:MAX_MECHANISM_CHARS],
                "readout_cells": str(m.get("readout_cells", "")).strip()[:MAX_MECHANISM_CHARS],
                "confidence": float(m.get("confidence", 0.5) or 0.0),
                "would_test_by": str(m.get("would_test_by", "")).strip()[:MAX_MECHANISM_CHARS],
            }
            for m in mechanisms
            if str(m.get("mechanism", "")).strip()
        ][: wp.MAX_MECHANISMS]
        self.finished = {"mechanisms": out, "summary": str(summary).strip()[:MAX_MECHANISM_CHARS]}
        return f"finished with {len(out)} mechanisms"

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        """Dispatch, intercepting ``finish`` (which carries mechanisms in this mode)."""
        if name != "finish":
            return super().call(name, args)
        try:
            out = self.finish_mechanisms(
                list(args.get("mechanisms") or []), str(args.get("summary", "") or "")
            )
        except Exception as e:  # the agent sees the error and can retry, as with any tool
            out = f"tool error: {type(e).__name__}: {e}"
        from ..agent.loop import ToolLog

        self.log.append(ToolLog(name, dict(args), out, 0))
        return out


def tool_schemas(arm: str = ARM, layers: Sequence[int] = READ_LAYERS) -> list[dict[str, Any]]:
    """The arm's schemas with the quirk-shaped ``finish`` swapped for the mechanisms one."""
    out = [s for s in live_tool_schemas(arm, layers, False) if s["function"]["name"] != "finish"]
    return [*out, wp.FINISH_SCHEMA]


def rollout_ids(pattern: Pattern, n_side: int) -> list[tuple[str, bool, str]]:
    """``(conversation id, shows the behavior, reply text)`` for the seeded study rollouts."""
    out: list[tuple[str, bool, str]] = []
    for i, s in enumerate(pattern.matched[:n_side]):
        out.append((f"w{i:03d}m", True, s.text))
    for i, s in enumerate(pattern.unmatched[:n_side]):
        out.append((f"w{i:03d}u", False, s.text))
    return out


def run_explain_agent(
    pattern: Pattern,
    client: LensClient,
    backend: Backend,
    *,
    auditor: str,
    seed: int,
    arm: str = ARM,
    layers: Sequence[int] = READ_LAYERS,
    budget: Budget | None = None,
    aux_model: str = "google/gemini-3.8-flash",
    select: str = "all",
    n_side: int = 2,
    rollout_chars: int = 3000,
) -> tuple[RunRecord, ExplainTools]:
    """One explain run on one pattern; the tools come back too, for the record's extras."""
    tools = ExplainTools(
        arm,
        ORGANISM,
        client,
        seed,
        layers=layers,
        aux_model=aux_model,
        select=select,
        allow_reference=False,
    )
    rollouts = rollout_ids(pattern, n_side)
    for cid, _, text in rollouts:
        tools.seed_conversation(cid, pattern.prompt, text)
    rec = RunRecord(ORGANISM, CONDITION, arm, auditor, seed)
    run_tool_loop(
        tools,
        tool_schemas(arm, layers),
        wp.system_prompt(),
        wp.brief(
            behavior_name=pattern.behavior_name,
            rubric=pattern.transcript_rubric,
            group_summary=pattern.group_summary,
            prompt=pattern.prompt,
            match_rate=pattern.published_match_rate,
            n_samples=len(pattern.samples),
            rollouts=[(cid, m, t[:rollout_chars]) for cid, m, t in rollouts],
        ),
        backend,
        budget or Budget(max_calls=400),
        rec,
        reduction=wp.REDUCTION,
    )
    rec.tool_log = tools.log
    rec.notes = tools.notes
    rec.result = tools.finished
    rec.cells_served = tools.cells_served
    rec.chars_served = tools.chars_served
    rec.select = select
    rec.allow_reference = False
    return rec, tools


def dumps(rec: RunRecord, pattern: Pattern, extras: Mapping[str, Any]) -> str:
    """The on-disk run record: the agent run, the pattern it was given, and server usage."""
    return json.dumps(
        {"record": rec.to_json(), "pattern": pattern.to_json(), "extras": dict(extras)},
        ensure_ascii=False,
        indent=1,
    )


__all__ = [
    "ARM",
    "CONDITION",
    "LIVE_CONDITION",
    "ORGANISM",
    "ExplainTools",
    "dumps",
    "rollout_ids",
    "run_explain_agent",
    "tool_schemas",
]
