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
from ..tools.arms import lens_of
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
        if "_raw" in args:  # the backend could not parse the arguments: the turn was cut short
            out = (
                f"tool error: your finish() arguments were not valid JSON ({len(str(args['_raw']))} "
                "characters received — the turn was truncated). Call finish() again with SHORTER "
                "fields: one or two sentences of evidence per mechanism, cells as brief "
                "references, and no more mechanisms than you have evidence for."
            )
        else:
            try:
                mechanisms, summary = _shape(args)
                if not mechanisms and any(args.get(k) for k in _FIELDS):
                    raise ValueError(
                        "finish() needs a `mechanisms` ARRAY of objects {mechanism, evidence, "
                        "readout_cells, confidence, would_test_by}; you sent prose in top-level "
                        "fields. Call finish() again with each mechanism as its own object."
                    )
                out = self.finish_mechanisms(mechanisms, summary)
            except Exception as e:  # the agent sees the error and can retry, as with any tool
                out = f"tool error: {type(e).__name__}: {e}"
        from ..agent.loop import ToolLog

        self.log.append(ToolLog(name, dict(args), out, 0))
        return out


_FIELDS = ("mechanism", "evidence", "readout_cells", "would_test_by", "summary")


def _shape(args: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], str]:
    """``(mechanisms, summary)`` from a finish() call, tolerating the flat shape a model
    sometimes produces: one mechanism's fields at top level, its statement under ``summary``."""
    raw = args.get("mechanisms") or []
    if isinstance(raw, str):  # the array serialised as text — decode it, or treat it as one item
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [{"mechanism": raw}]
    if isinstance(raw, Mapping):
        raw = [raw]
    mechanisms: list[Mapping[str, Any]] = []
    for m in raw if isinstance(raw, list) else []:
        if isinstance(m, Mapping):
            mechanisms.append(m)
        elif isinstance(m, str) and m.strip():  # a list of statements
            mechanisms.append({"mechanism": m})
    summary = str(args.get("summary", "") or "")
    if mechanisms:
        return mechanisms, summary
    statement = str(args.get("mechanism", "") or "")
    if not statement and summary and (args.get("evidence") or args.get("would_test_by")):
        statement, summary = summary, ""  # the statement was filed under summary
    if statement:
        one = {k: args.get(k) for k in ("evidence", "readout_cells", "confidence", "would_test_by")}
        return [
            {"mechanism": statement, **{k: v for k, v in one.items() if v is not None}}
        ], summary
    return [], summary


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
    rollouts = [(cid, m, t[:rollout_chars]) for cid, m, t in rollout_ids(pattern, n_side)]
    for cid, _, text in rollouts:  # the agent reads exactly the text it was shown
        tools.seed_conversation(cid, pattern.prompt, text)
    lens = lens_of(arm)
    rec = RunRecord(ORGANISM, CONDITION, arm, auditor, seed)
    run_tool_loop(
        tools,
        tool_schemas(arm, layers),
        wp.system_prompt(lens),
        wp.brief(
            behavior_name=pattern.behavior_name,
            rubric=pattern.transcript_rubric,
            group_summary=pattern.group_summary,
            prompt=pattern.prompt,
            match_rate=pattern.published_match_rate,
            n_samples=len(pattern.samples),
            rollouts=rollouts,
            lens=lens is not None,
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
