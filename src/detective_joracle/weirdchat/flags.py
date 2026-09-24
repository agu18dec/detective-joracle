"""Marking the readout cells worth a reader's attention, with an LLM flagger.

A lens grid for one reply is hundreds of positions by eleven layers; almost every cell paraphrases
the prompt or echoes the reply. This pass hands the grid to a reader model in windows of positions
and asks for the cells that say something the text at that point does not: a role the model has
taken on, two continuations live at once, a hidden referent decoded, a capability disclaimer
sitting next to the claim, a cell that contradicts the surrounding text. Every flag quotes the
cell verbatim; the quote is checked against the grid before it is kept, so the viewer never marks
a position on the strength of a paraphrase.

The flagger sees the reply and knows which side of the judge's label the reply fell on; it is an
attention pass, not a blind judge, and the page says so.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm.route import async_json_route, schema_block

CATEGORIES = (
    "role_adoption",  # the model represents itself as something other than a chat assistant
    "both_branches",  # the compliant and the honest continuation live in one cell or position
    "hidden_referent",  # a euphemism, premise or target decoded to what it actually means
    "disclaimer_present",  # the capability/identity disclaimer, next to where the claim is made
    "contradicts_text",  # the cell says something the reply at that point does not
    "commitment_point",  # where the reply's direction gets fixed
    "other",
)

SYSTEM = (
    "You read interpretability-lens output for a language model. Each cell is a sentence a lens "
    "decoded from the model's internal state at ONE token position and ONE layer while it read a "
    "user prompt and wrote a reply. You are given the prompt, the reply, whether a judge said the "
    "reply shows a specified behavior, and a window of positions with every layer's cell. Mark the "
    "cells worth a reader's attention: cells that say something the TEXT at that point does not — "
    "a role the model has taken on (role_adoption), both the compliant and the honest continuation "
    "present together (both_branches), a euphemism or premise decoded to its real meaning "
    "(hidden_referent), a capability or identity disclaimer sitting where a claim is made "
    "(disclaimer_present), a cell that contradicts the surrounding text (contradicts_text), the "
    "point where the reply's direction is fixed (commitment_point). Do NOT flag cells that merely "
    "paraphrase the prompt or restate the reply; do not flag more than 8 cells per window; prefer "
    "the boundary/header positions and the first reply tokens when they qualify. Quote the cell "
    "VERBATIM (copy at least 25 consecutive characters exactly as written, including any Chinese). "
    "One sentence of why per flag."
)
USER = (
    "<behavior>{behavior}</behavior>\n<judge_says_reply_shows_it>{flagged}</judge_says_reply_shows_it>\n"
    "<prompt>\n{prompt}\n</prompt>\n<reply>\n{reply}\n</reply>\n\n"
    "Window {w}/{n_windows}, positions {lo}–{hi} (region in brackets, token in quotes, then one "
    "line per layer):\n\n{cells}\n\n"
    "Return the flags for this window."
)
SCHEMA = schema_block(
    "flags",
    {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "position": {"type": "integer"},
                    "layer": {"type": "integer"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "quote": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["position", "layer", "category", "quote", "why"],
            },
        }
    },
    ["flags"],
)
WINDOW = 48  # positions per call: ~48 x 11 layers x ~300 chars ≈ 40k tokens
CELL_CHARS = 320
MIN_QUOTE = 25


def windows(positions: Sequence[str], size: int = WINDOW) -> list[list[str]]:
    return [list(positions[i : i + size]) for i in range(0, len(positions), size)]


def _render(read: Mapping[str, Any], pos: Sequence[str]) -> str:
    tokens, tags, cells = read["tokens"], read["tags"], read["readouts"]
    layers = sorted(cells, key=int)
    out = []
    for p in pos:
        t = tags.get(p, {})
        out.append(f"pos {p} [{t.get('region', '?')}] {tokens.get(p, '')!r}")
        for L in layers:
            samples = cells[L].get(p) or []
            if samples:
                out.append(f"  L{L}: " + " | ".join(s[:CELL_CHARS] for s in samples))
    return "\n".join(out)


def verify(read: Mapping[str, Any], flag: Mapping[str, Any]) -> bool:
    """A flag is kept only when its quote is a verbatim substring of a cell at that position and
    layer (any sample), and long enough to be a quote rather than a word."""
    q = str(flag.get("quote", ""))
    if len(q) < MIN_QUOTE:
        return False
    samples = read["readouts"].get(str(flag.get("layer")), {}).get(str(flag.get("position"))) or []
    return any(q in s for s in samples)


def flag_read(
    read: Mapping[str, Any],
    *,
    behavior: str,
    prompt: str,
    reply: str,
    flagged: bool,
    model: str,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Every window of one read through the flagger; verified flags out, with the miss count."""
    pos = sorted(read["tokens"], key=int)
    wins = windows(pos)
    items = [
        (
            SYSTEM,
            USER.format(
                behavior=behavior,
                flagged="yes" if flagged else "no",
                prompt=prompt[:3000],
                reply=reply[:4000],
                w=i + 1,
                n_windows=len(wins),
                lo=w[0],
                hi=w[-1],
                cells=_render(read, w),
            ),
        )
        for i, w in enumerate(wins)
    ]
    outs = async_json_route(items, schema=SCHEMA, model=model, concurrency=concurrency)
    kept: list[dict[str, Any]] = []
    proposed = failed_calls = unverified = 0
    for o in outs:
        if o is None:
            failed_calls += 1
            continue
        for f in o.get("flags") or []:
            proposed += 1
            if verify(read, f):
                kept.append(
                    {
                        "position": int(f["position"]),
                        "layer": int(f["layer"]),
                        "category": str(f["category"]),
                        "quote": str(f["quote"]),
                        "why": str(f["why"])[:400],
                    }
                )
            else:
                unverified += 1
    kept.sort(key=lambda f: (f["position"], f["layer"]))
    return {
        "flags": kept,
        "n_windows": len(wins),
        "failed_calls": failed_calls,
        "proposed": proposed,
        "unverified": unverified,
        "model": model,
    }


def dumps(blob: Mapping[str, Any]) -> str:
    return json.dumps(blob, ensure_ascii=False, indent=1)


__all__ = ["CATEGORIES", "SCHEMA", "WINDOW", "dumps", "flag_read", "verify", "windows"]


# ------------------------------------------------------- contrastive annotations
ANNOTATE_SYSTEM = (
    "You annotate interpretability-lens output for a language model. An investigator explained "
    "why the model shows a behavior on a prompt, as a ranked list of mechanisms. You are given "
    "those mechanisms, the prompt, and — for a window of token positions — the lens cells of TWO "
    "replies to that prompt: one the judge FLAGGED as showing the behavior, one judged CLEAN. Up to "
    "the point where the two replies diverge the cells come from the identical text, so a "
    "difference there is lens noise; after it, a difference can be real. For each mechanism, mark "
    "the cells that CONTRASTIVELY support it: a cell where the flagged reply's activation says "
    "something the clean reply's does not at the same position (or the reverse), and that is what "
    "the mechanism claims. Also mark cells where both replies share the same readout that the "
    "mechanism rests on (propensity evidence), labelled as such. Quote the cell VERBATIM (at least "
    "25 consecutive characters exactly as written). At most 6 annotations per window; skip windows "
    "with nothing that bears on a mechanism."
)
ANNOTATE_USER = (
    "<behavior>{behavior}</behavior>\n<prompt>\n{prompt}\n</prompt>\n\n<mechanisms>\n{mechs}\n</mechanisms>\n\n"
    "Replies diverge from position {fork} onward (positions before it are identical text).\n"
    "Window {w}/{n_windows}, positions {lo}–{hi}. For each position: the token, then per layer the "
    "FLAGGED cell and the CLEAN cell.\n\n{cells}\n\nReturn the annotations for this window."
)
ANNOTATE_SCHEMA = schema_block(
    "annotations",
    {
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mechanism": {"type": "integer"},
                    "position": {"type": "integer"},
                    "layer": {"type": "integer"},
                    "side": {"type": "string", "enum": ["flagged", "clean", "both"]},
                    "quote": {"type": "string"},
                    "contrast": {"type": "string"},
                },
                "required": ["mechanism", "position", "layer", "side", "quote", "contrast"],
            },
        }
    },
    ["annotations"],
)
ANNOTATE_WINDOW = 24


def _render_pair(flagged: Mapping[str, Any], clean: Mapping[str, Any], pos: Sequence[str]) -> str:
    layers = sorted(set(flagged["readouts"]) | set(clean["readouts"]), key=int)
    out = []
    for p in pos:
        tf = flagged["tokens"].get(p)
        tc = clean["tokens"].get(p)
        tok = f"{tf!r}" if tf == tc else f"flagged {tf!r} / clean {tc!r}"
        out.append(
            f"pos {p} [{flagged['tags'].get(p, clean['tags'].get(p, {})).get('region', '?')}] {tok}"
        )
        for L in layers:
            f = " | ".join(s[:CELL_CHARS] for s in flagged["readouts"].get(L, {}).get(p) or [])
            c = " | ".join(s[:CELL_CHARS] for s in clean["readouts"].get(L, {}).get(p) or [])
            if f or c:
                out.append(f"  L{L} FLAGGED: {f or '—'}\n  L{L} CLEAN:   {c or '—'}")
    return "\n".join(out)


def _verify_side(
    flagged: Mapping[str, Any], clean: Mapping[str, Any], a: Mapping[str, Any]
) -> str | None:
    """Which read(s) hold the quote verbatim at (position, layer): 'flagged', 'clean', 'both', or None."""
    q = str(a.get("quote", ""))
    if len(q) < MIN_QUOTE:
        return None
    L, p = str(a.get("layer")), str(a.get("position"))
    in_f = any(q in s for s in flagged["readouts"].get(L, {}).get(p) or [])
    in_c = any(q in s for s in clean["readouts"].get(L, {}).get(p) or [])
    return "both" if in_f and in_c else "flagged" if in_f else "clean" if in_c else None


def annotate_contrast(
    flagged: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    behavior: str,
    prompt: str,
    mechanisms: Sequence[Mapping[str, Any]],
    fork_position: int | None,
    model: str,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Cells that contrastively support the investigator's mechanisms, verified verbatim in the
    read they are attributed to (the side is corrected to what the grids actually show)."""
    pos = sorted(set(flagged["tokens"]) | set(clean["tokens"]), key=int)
    wins = windows(pos, ANNOTATE_WINDOW)
    mechs = "\n".join(
        f"[{i}] {str(m.get('short') or m.get('mechanism', ''))[:400]}"
        for i, m in enumerate(mechanisms)
    )
    items = [
        (
            ANNOTATE_SYSTEM,
            ANNOTATE_USER.format(
                behavior=behavior,
                prompt=prompt[:3000],
                mechs=mechs,
                fork=fork_position if fork_position is not None else "(unknown)",
                w=i + 1,
                n_windows=len(wins),
                lo=w[0],
                hi=w[-1],
                cells=_render_pair(flagged, clean, w),
            ),
        )
        for i, w in enumerate(wins)
    ]
    outs = async_json_route(items, schema=ANNOTATE_SCHEMA, model=model, concurrency=concurrency)
    kept: list[dict[str, Any]] = []
    proposed = failed = unverified = 0
    for o in outs:
        if o is None:
            failed += 1
            continue
        for a in o.get("annotations") or []:
            proposed += 1
            side = _verify_side(flagged, clean, a)
            mi = int(a.get("mechanism", -1))
            if side is None or not (0 <= mi < len(mechanisms)):
                unverified += 1
                continue
            kept.append(
                {
                    "mechanism": mi,
                    "position": int(a["position"]),
                    "layer": int(a["layer"]),
                    "side": side,
                    "claimed_side": str(a.get("side")),
                    "quote": str(a["quote"]),
                    "contrast": str(a.get("contrast", ""))[:400],
                }
            )
    kept.sort(key=lambda a: (a["mechanism"], a["position"], a["layer"]))
    return {
        "annotations": kept,
        "n_windows": len(wins),
        "failed_calls": failed,
        "proposed": proposed,
        "unverified": unverified,
        "model": model,
    }


__all__ += ["ANNOTATE_SCHEMA", "ANNOTATE_WINDOW", "annotate_contrast"]


# ------------------------------------------------------- grading: contrastive or shared?
GRADE_SYSTEM = (
    "You check whether a lens cell is CONTRASTIVE evidence. You are given a hypothesis about why "
    "a model shows a behavior on a prompt, and — at one token position and one layer — the lens "
    "cell(s) from a reply the judge FLAGGED and the cell(s) from a CLEAN reply to the same prompt. "
    "The annotation claims the quoted cell supports the hypothesis. Decide: does the flagged side "
    "say something that bears on the hypothesis which the clean side does NOT say at this cell (or "
    "the reverse)? If both sides carry essentially the same content, the cell is SHARED propensity "
    "evidence, not contrastive. Answer for each item."
)
GRADE_USER = (
    "<hypothesis>{hyp}</hypothesis>\n<position>{pos} L{layer}</position>\n"
    "<flagged_cell>\n{f}\n</flagged_cell>\n<clean_cell>\n{c}\n</clean_cell>\n<quoted>{quote}</quoted>\n\n"
    "Is the quoted cell contrastive (one side says something bearing on the hypothesis that the "
    "other does not) or shared? One sentence of reason."
)
GRADE_SCHEMA = schema_block(
    "grade",
    {"contrastive": {"type": "boolean"}, "reason": {"type": "string"}},
    ["contrastive", "reason"],
)


def grade_contrast(
    annotations: Sequence[dict[str, Any]],
    flagged: Mapping[str, Any],
    clean: Mapping[str, Any],
    mechanisms: Sequence[Mapping[str, Any]],
    *,
    model: str,
    concurrency: int = 16,
) -> int:
    """Mark each annotation ``contrastive`` (True/False/None on a failed call), in place; returns
    how many were graded contrastive. The reader sees both sides' cells at that exact cell."""

    def cells(read: Mapping[str, Any], a: Mapping[str, Any]) -> str:
        got = read["readouts"].get(str(a["layer"]), {}).get(str(a["position"])) or []
        return "\n".join(f"- {s[:CELL_CHARS]}" for s in got) or "—"

    items = [
        (
            GRADE_SYSTEM,
            GRADE_USER.format(
                hyp=str(
                    mechanisms[a["mechanism"]].get("short")
                    or mechanisms[a["mechanism"]].get("mechanism", "")
                )[:400],
                pos=a["position"],
                layer=a["layer"],
                f=cells(flagged, a),
                c=cells(clean, a),
                quote=a["quote"],
            ),
        )
        for a in annotations
    ]
    outs = async_json_route(items, schema=GRADE_SCHEMA, model=model, concurrency=concurrency)
    n = 0
    for a, o in zip(annotations, outs, strict=True):
        if o is None:
            a["contrastive"] = None
            a["grade_reason"] = ""
            continue
        a["contrastive"] = bool(o.get("contrastive"))
        a["grade_reason"] = str(o.get("reason", ""))[:300]
        n += int(a["contrastive"])
    return n


__all__ += ["GRADE_SCHEMA", "grade_contrast"]
