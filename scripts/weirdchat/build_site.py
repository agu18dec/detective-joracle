"""Build the WeirdChat explanation viewer — a small index page plus per-pattern data files.

Walks ``<out_root>/{patterns,diag,runs}`` plus an optional ``synth.json`` and writes into
``<site_dir>`` (default ``<out_root>/site``):

    index.html                 page + JS + the small payload (pattern metadata, per-run
                               mechanisms, behaviors table, synth clusters) — no samples/grids
    data/<pattern_key>.json    the heavy payload (samples, lens reads, full agent transcripts),
                               fetched on demand when a pattern page opens
    data/<pattern_key>.agent.json   only when the pattern's file would blow the per-file budget:
                               the agent's own lens reads, fetched when one is first selected

    python scripts/weirdchat/build_site.py [out=outputs/weirdchat] [site=<dir>] [single=true]
        [translations=<cache.json>] [highlights=<hand_picks.json>]
    cd outputs/weirdchat/site && python -m http.server 8905

``single=true`` inlines every pattern into ONE self-contained index.html (no data/ dir, no
fetch). ``translations`` is a ``{sample_text: english}`` cache rendered under each sample;
the build also writes ``cells_to_translate.json`` (every non-Latin-script sample) for the
script that fills it. ``highlights`` adds hand-picked cells; one that does not verify verbatim
against the read is a build error. Auto highlights are every mechanism-quoted cell that does.

The page follows the design contract in ``docs/viewer_design.md`` (the wsbench viewer's light
app shell): a top bar of pickers and searches, metadata panes, then two resizable panes — the
read tokens of one lens read on the left, every token clickable; that position's readout at
every layer on the right, one column per compared read. A "read" is one OLens readout over one conversation — from the
diagnostic pass (``diag/``) or parsed back out of the agent's ``readouts`` tool pages.

Serve the directory: ``fetch()`` of ``data/…`` is blocked from a ``file://`` origin in most
browsers (the page says so when it happens). The overview and cluster pages need no fetch.
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))  # so the brief can be rebuilt with the repo's own prompts

# a single OLens sample keeps at most this much text (measured max is 431, so nothing is clipped)
CELL_CAP = 450
# a tool result / an assistant turn in the transcript keeps at most this much text
TOOL_OUTPUT_CAP = 12000
THINK_CAP = 6000
# hard size budgets: each data/*.json, and everything written together
# (the host caps a version at 64 MB and a file at 16 MB; 4 MB per file is a soft warning)
DATA_FILE_BUDGET = 4 * 1024 * 1024
TOTAL_BUDGET = 62 * 1024 * 1024

# the agent's readouts tool page (detective_joracle.tools.live._format_readouts)
POS_RE = re.compile(r"^  pos (\d+) \[([^\]]+)\] token=(.*)$")
LAYER_RE = re.compile(r"^    L(\d+)(?: \([^)]*\))?(?: \[kept: [^\]]*\])?: (.*)$")
REGION_RE = re.compile(r"^== ([A-Za-z]+) \((\d+) positions\)")
CLAIM_RE = re.compile(r"(\d+) positions x (\d+) layers(?: x (\d+) samples)?")
CONV_RE = re.compile(r"readouts on conversation (\S+) \(")
# the page header starts with the lens's display name (detective_joracle.tools.arms.LENS_NAME)
LENS_BY_HEAD = (
    ("Jacobian lens", "jlens"),
    ("NLA-RL", "nla"),
    ("OLens", "olens"),
    ("Logit lens", "logit"),
)
BAG_LENSES = {"jlens", "nla"}  # cells joined with " | ": a J-lens token bag, or NLA's k samples
TEXT_LINE_RE = re.compile(r"^  \[(system|user|assistant reply)\] (.*)$")
STUDY_CONV_RE = re.compile(r"^w\d+([mu])$")


# ------------------------------------------------------------------------- io
def read_json(path: Path) -> dict[str, Any] | None:
    """Parse a JSON object, returning None for anything missing or unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_text(value: Any) -> str:
    """Any JSON value as display text (dicts/lists collapse to compact JSON)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def clip(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[:cap] + f" … [+{len(text) - cap} chars]"


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sorted_int_keys(mapping: dict[str, Any]) -> list[str]:
    """Numeric-looking keys in numeric order; anything else after, lexically."""
    nums = sorted((k for k in mapping if as_int(k) is not None), key=lambda k: as_int(k) or 0)
    return nums + sorted(k for k in mapping if as_int(k) is None)


# -------------------------------------------------------------------- reads
# one read = {id, source, label, conv_id, messages, completion, layers, rows, fork_pos, …};
# rows = [{pos, tok, region, kind, samples: [[str, …] per layer]}] in position order
def samples_of(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clip(as_text(v), CELL_CAP) for v in value if as_text(v)]
    text = as_text(value)
    return [clip(text, CELL_CAP)] if text else []


def diag_read(entry: dict[str, Any]) -> dict[str, Any]:
    """One diagnostic read: the exact conversation read plus its per-position, per-layer grid."""
    conv = as_dict(entry.get("conversation"))
    readout = as_dict(entry.get("readout"))
    tokens = as_dict(readout.get("tokens"))
    tags = as_dict(readout.get("tags"))
    by_layer = as_dict(readout.get("readouts"))
    layer_keys = sorted_int_keys(by_layer)
    positions: set[str] = set(tokens) | set(tags)
    for layer in layer_keys:
        positions |= set(as_dict(by_layer.get(layer)))
    rows = []
    for pos in sorted_int_keys(dict.fromkeys(positions)):
        tag = as_dict(tags.get(pos))
        rows.append(
            {
                "pos": as_int(pos) if as_int(pos) is not None else pos,
                "tok": as_text(tokens.get(pos)),
                "region": as_text(tag.get("region")) or "—",
                "kind": as_text(tag.get("kind")),
                "samples": [samples_of(as_dict(by_layer.get(L)).get(pos)) for L in layer_keys],
            }
        )
    label = as_text(entry.get("label")) or "read"
    sample_index = as_int(entry.get("sample_index"))
    return {
        "id": f"diag:{label}:{sample_index if sample_index is not None else '?'}",
        "source": "diag",
        "label": label,
        "sample_index": sample_index,
        "conv_id": "",
        "tool_index": None,
        "messages": [
            {"role": as_text(m.get("role")), "content": as_text(m.get("content"))}
            for m in as_list(conv.get("messages"))
            if isinstance(m, dict)
        ],
        "completion": as_text(conv.get("completion")),
        "text_source": "diag",
        "lens": as_text(readout.get("lens")) or "olens",
        "layers": [as_int(k) if as_int(k) is not None else k for k in layer_keys],
        "rows": rows,
        "claimed": {"positions": as_int(readout.get("n_tokens")), "layers": len(layer_keys)},
        "parse_error": None,
        "fork_pos": None,
    }


def decode_token(repr_text: str) -> str:
    try:
        value = ast.literal_eval(repr_text.strip())
    except (ValueError, SyntaxError):
        return repr_text
    return value if isinstance(value, str) else repr_text


def parse_readout_page(text: str) -> dict[str, Any]:
    """Parse one agent `readouts` tool page back into rows; never raises, records parse_error."""
    out: dict[str, Any] = {
        "conv_id": "",
        "lens": "",
        "claimed": {"positions": None, "layers": None, "k": 1},
        "messages": [],
        "completion": "",
        "layers": [],
        "rows": [],
        "parse_error": None,
    }
    try:
        _parse_page_into(out, text)
    except Exception as exc:  # a bad page must become a read with parse_error, never raise
        out["parse_error"] = f"{type(exc).__name__}: {exc}"
    if not out["rows"] and not out["parse_error"]:
        first = text.strip().split("\n", 1)[0] if text.strip() else "(empty output)"
        out["parse_error"] = f"no positions parsed — page starts: {first[:160]}"
    return out


def _parse_page_into(out: dict[str, Any], text: str) -> None:
    lines = text.split("\n")
    head = lines[0] if lines else ""
    m = CONV_RE.search(head)
    if m:
        out["conv_id"] = m.group(1)
    out["lens"] = next((lens for name, lens in LENS_BY_HEAD if head.startswith(name)), "")
    m = CLAIM_RE.search(head)
    k = 1
    if m:
        k = int(m.group(3) or 1)
        out["claimed"] = {"positions": int(m.group(1)), "layers": int(m.group(2)), "k": k}
    region = "—"
    row: dict[str, Any] | None = None
    open_layer: int | None = None
    in_text = False
    rows: list[dict[str, Any]] = []
    for line in lines:
        if line.startswith("TEXT THAT WAS READ"):
            in_text = True
            continue
        if in_text:
            tm = TEXT_LINE_RE.match(line)
            if tm:
                role, content = tm.group(1), tm.group(2)
                if role == "assistant reply":
                    out["completion"] = "" if content == "(empty)" else content
                else:
                    out["messages"].append({"role": role, "content": content})
                continue
            if not (line.startswith("REGIONS:") or line.startswith("== ")):
                continue
            in_text = False
        rm = REGION_RE.match(line)
        if rm:
            region, row, open_layer = rm.group(1).lower(), None, None
            continue
        pm = POS_RE.match(line)
        if pm:
            row = {
                "pos": int(pm.group(1)),
                "tok": decode_token(pm.group(3)),
                "region": region,
                "kind": pm.group(2),
                "cells": {},
            }
            rows.append(row)
            open_layer = None
            continue
        lm = LAYER_RE.match(line)
        if lm and row is not None:
            open_layer = int(lm.group(1))
            row["cells"][open_layer] = lm.group(2)
            continue
        if row is not None and open_layer is not None:
            if line.startswith("        base model here:"):
                open_layer = None
                continue
            row["cells"][open_layer] += "\n" + line  # a multi-line sample continues
    layers = sorted({L for r in rows for L in r["cells"]})
    for r in rows:
        cells = r.pop("cells")
        r["samples"] = []
        for L in layers:
            raw = cells.get(L, "").rstrip()
            split = k > 1 or out["lens"] in BAG_LENSES
            parts = raw.split(" | ") if (split and raw) else ([raw] if raw else [])
            r["samples"].append([clip(p, CELL_CAP) for p in parts if p])
    rows.sort(key=lambda r: r["pos"])
    out["layers"] = layers
    out["rows"] = rows


def agent_read(
    tool_index: int,
    logged: dict[str, Any],
    used_ids: set[str],
    diag_reads: list[dict[str, Any]],
    arm: str = "olens",
) -> dict[str, Any]:
    """One `readouts` tool call as a read; borrows the full text from a matching diag read.

    Ids stay ``agent:<conv>`` for the OLens arm (deep links in the wild) and become
    ``agent:<lens>:<conv>`` for the other lens arms, which read the same conversations."""
    args = as_dict(logged.get("args"))
    parsed = parse_readout_page(as_text(logged.get("output")))
    conv_id = as_text(args.get("conversation")) or parsed["conv_id"] or f"call{tool_index}"
    lens = parsed["lens"] or (arm if arm != "blackbox" else "olens")
    rid = f"agent:{conv_id}" if lens == "olens" else f"agent:{lens}:{conv_id}"
    if rid in used_ids:
        rid = f"{rid}#{tool_index}"
    used_ids.add(rid)
    sm = STUDY_CONV_RE.match(conv_id)
    label = {"m": "matched", "u": "unmatched"}[sm.group(1)] if sm else "agent"
    twin = None
    if sm:
        head = parsed["completion"][:400]
        for d in diag_reads:
            if d["label"] == label and (not head or d["completion"].startswith(head)):
                twin = d
                break
    messages, completion, text_source = parsed["messages"], parsed["completion"], "tool page"
    if twin is not None:
        messages, completion, text_source = twin["messages"], twin["completion"], twin["id"]
    return {
        "id": rid,
        "source": "agent",
        "label": label,
        "sample_index": None,
        "conv_id": conv_id,
        "tool_index": tool_index,
        "messages": messages,
        "completion": completion,
        "text_source": text_source,
        "same_rollout_as": twin["id"] if twin else None,
        "lens": lens,
        "layers": parsed["layers"],
        "rows": parsed["rows"],
        "claimed": parsed["claimed"],
        "parse_error": parsed["parse_error"],
        "fork_pos": None,
    }


def fork_position(read: dict[str, Any], fork: dict[str, Any]) -> int | None:
    """Reply position whose token first passes fork.prefix_chars in the completion (≈, thinned)."""
    prefix_chars = as_int(fork.get("prefix_chars"))
    completion = read.get("completion") or ""
    if prefix_chars is None or not completion or prefix_chars >= len(completion):
        return None
    cursor = 0
    for row in read["rows"]:
        tok = row["tok"]
        if row["region"] != "reply" or not tok or tok.startswith("<|"):
            continue
        idx = completion.find(tok, cursor)
        if idx < 0:
            continue
        cursor = idx + len(tok)
        if cursor > prefix_chars:
            return int(row["pos"])
    return None


# ------------------------------------------------------------------ the brief
N_SIDE = 2  # study rollouts per side handed to the investigator (run_weirdchat.py default)


def load_prompts() -> tuple[Any, Any, str]:
    """(brief, system_prompt, how) from detective_joracle.weirdchat.prompts.

    The package import needs the repo's dependencies (requests, openai). When they are missing
    the module is loaded from its source instead — same templates, same code — with its one
    relative import (MAX_PREDICTIONS) read out of agent/loop.py."""
    try:
        from detective_joracle.weirdchat.prompts import brief, system_prompt

        return brief, system_prompt, "detective_joracle.weirdchat.prompts (imported)"
    except Exception as exc:  # fall back to the source: the brief must still be faithful
        reason = f"{type(exc).__name__}: {exc}"
    try:
        src_dir = REPO / "src" / "detective_joracle"
        tree = ast.parse((src_dir / "weirdchat" / "prompts.py").read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
        max_pred = 10
        for node in ast.parse((src_dir / "agent" / "loop.py").read_text()).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MAX_PREDICTIONS" for t in node.targets
            ):
                max_pred = int(ast.literal_eval(node.value))
        ns: dict[str, Any] = {"MAX_PREDICTIONS": max_pred, "__name__": "wc_prompts"}
        exec(compile(tree, "prompts.py", "exec"), ns)  # noqa: S102 — the repo's own module
        return ns["brief"], ns["system_prompt"], f"prompts.py loaded from source ({reason})"
    except Exception as exc:
        return None, None, f"unavailable: {reason}; source fallback failed: {exc}"


def rollout_ids(samples: list[dict[str, Any]], n_side: int = N_SIDE) -> list[tuple[str, bool, str]]:
    """The seeded study rollouts exactly as explain.rollout_ids orders them: w000m, w001m, w000u, w001u."""
    out: list[tuple[str, bool, str]] = []
    for i, smp in enumerate([x for x in samples if x["matched"]][:n_side]):
        out.append((f"w{i:03d}m", True, smp["text"]))
    for i, smp in enumerate([x for x in samples if not x["matched"]][:n_side]):
        out.append((f"w{i:03d}u", False, smp["text"]))
    return out


def brief_of(pat: dict[str, Any], brief_fn: Any) -> dict[str, Any] | None:
    """The opening message the investigator received for this pattern, rebuilt from the pattern."""
    if brief_fn is None:
        return None
    rollouts = [(cid, m, t[:3000]) for cid, m, t in rollout_ids(pat["samples"])]
    try:
        text = brief_fn(
            behavior_name=pat["behavior_name"],
            rubric=pat["rubric"],
            group_summary=pat["group_summary"],
            prompt=pat["prompt"],
            match_rate=pat["published_match_rate"] or 0.0,
            n_samples=len(pat["samples"]),
            rollouts=rollouts,
        )
    except Exception as exc:
        return {"text": None, "ids": [], "error": f"{type(exc).__name__}: {exc}"}
    return {"text": text, "ids": [cid for cid, _, _ in rollouts], "error": None}


# ------------------------------------------------------------------ transcript
def steps_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Chronological transcript: assistant text, tool calls, and each call's paired output."""
    tool_log = as_list(record.get("tool_log"))
    steps: list[dict[str, Any]] = []
    ti = 0
    for turn in as_list(record.get("turns")):
        turn = as_dict(turn)
        if as_text(turn.get("role")) == "tool":
            continue
        think = clip(as_text(turn.get("content")).strip(), THINK_CAP)
        calls = [as_dict(c) for c in as_list(turn.get("tool_calls"))]
        if not calls:
            if think and think != "...":
                steps.append({"think": think, "name": None, "args": "", "output": "", "meta": ""})
            continue
        for call in calls:
            logged = as_dict(tool_log[ti]) if ti < len(tool_log) else {}
            tool_index = ti
            ti += 1
            name = as_text(logged.get("name")) or as_text(call.get("name")) or "tool"
            args = as_dict(logged.get("args")) or as_dict(call.get("args"))
            cells = as_int(logged.get("cells_served")) or 0
            tok = as_int(turn.get("output_tokens")) or 0
            meta = " · ".join(
                p for p in (f"{cells} cells" if cells else "", f"{tok} tok" if tok else "") if p
            )
            steps.append(
                {
                    "think": think if think and think != "..." else "",
                    "name": name,
                    "args": compact_args(args),
                    "user": as_text(args.get("user")) if name == "chat" else "",
                    "system": as_text(args.get("system")) if name == "chat" else "",
                    "prefill": as_text(args.get("prefill")) if name == "chat" else "",
                    "note": as_text(args.get("text")) if name == "note" else "",
                    "conversation": as_text(args.get("conversation")),
                    "tool_index": tool_index,
                    "output": clip(as_text(logged.get("output")), TOOL_OUTPUT_CAP),
                    "meta": meta,
                }
            )
            think = ""
    return steps


def compact_args(args: dict[str, Any]) -> str:
    """Tool args as one line, long values truncated — display only."""
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        text = as_text(value)
        parts.append(f"{key}={clip(text, 200)!r}" if isinstance(value, str) else f"{key}={text}")
    return ", ".join(parts)


def mechanisms_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for mech in as_list(as_dict(record.get("result")).get("mechanisms")):
        mech = as_dict(mech)
        out.append(
            {
                "mechanism": as_text(mech.get("mechanism")),
                "evidence": as_text(mech.get("evidence")),
                "readout_cells": as_text(mech.get("readout_cells")),
                "confidence": as_float(mech.get("confidence")),
                "would_test_by": as_text(mech.get("would_test_by")),
            }
        )
    return out


def run_of(path: Path, auditor_dir: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One agent run file as the viewer's run object, plus its raw tool_log for read parsing."""
    blob = read_json(path) or {}
    record = as_dict(blob.get("record"))
    extras = as_dict(blob.get("extras"))
    result = as_dict(record.get("result"))
    run = {
        "auditor": as_text(record.get("auditor")) or auditor_dir,
        "auditor_dir": auditor_dir,
        "seed": as_int(record.get("seed")),
        "arm": as_text(record.get("arm")),
        "condition": as_text(record.get("condition")),
        "stopped_by": as_text(record.get("stopped_by")),
        "output_tokens": as_int(record.get("output_tokens")),
        "cells_served": as_int(record.get("cells_served")),
        "server_calls": as_int(extras.get("server_calls")),
        "server_seconds": as_float(extras.get("server_seconds")),
        "notes": [as_text(n) for n in as_list(record.get("notes"))],
        "summary": as_text(result.get("summary")),
        "mechanisms": mechanisms_of(record),
        "steps": steps_of(record),
        "file": path.name,
    }
    tool_log = [as_dict(t) for t in as_list(record.get("tool_log"))]
    run["is_lens"] = as_text(record.get("arm")) != "blackbox"
    run["n_tool_calls"] = len(tool_log)
    run["n_readouts"] = sum(1 for t in tool_log if as_text(t.get("name")) == "readouts")
    run["n_chat"] = sum(1 for t in tool_log if as_text(t.get("name")) == "chat")
    return run, tool_log


# ---------------------------------------------------------------- collection
def pattern_of(path: Path, out_root: Path) -> dict[str, Any]:
    """One pattern plus whatever diagnostics and agent runs exist beside it."""
    meta = read_json(path) or {}
    key = as_text(meta.get("pattern_key")) or path.stem
    diag = read_json(out_root / "diag" / f"{key}.json") or {}
    fork = as_dict(diag.get("fork"))
    reads = [diag_read(as_dict(r)) for r in as_list(diag.get("reads"))]
    diag_reads = list(reads)
    used_ids = {r["id"] for r in reads}
    runs = []
    run_root = out_root / "runs" / key
    run_paths = sorted(run_root.glob("*/seed_*.json")) if run_root.is_dir() else []
    for run_path in run_paths:
        run, tool_log = run_of(run_path, run_path.parent.name)
        run["run_index"] = len(runs)
        runs.append(run)
        if run["arm"] == "blackbox":  # chat tools only: a black-box run must add no phantom reads
            continue
        for ti, logged in enumerate(tool_log):
            if as_text(logged.get("name")) != "readouts":
                continue
            read = agent_read(ti, logged, used_ids, diag_reads, run["arm"] or "olens")
            read["run_index"] = run["run_index"]
            read["arm"] = run["arm"]
            reads.append(read)
    for read in reads:
        if read["source"] == "diag" or STUDY_CONV_RE.match(read["conv_id"] or ""):
            read["fork_pos"] = fork_position(read, fork) if fork else None
    samples = [
        {
            "sample_index": as_int(s.get("sample_index")),
            "matched": bool(s.get("matched")),
            "text": as_text(s.get("text")),
        }
        for s in (as_dict(x) for x in as_list(meta.get("samples")))
    ]
    return {
        "key": key,
        "entry_id": as_text(meta.get("entry_id")),
        "behavior_id": as_text(meta.get("behavior_id")) or "—",
        "behavior_name": as_text(meta.get("behavior_name")) or as_text(meta.get("behavior_id")),
        "group_id": as_text(meta.get("group_id")),
        "group_summary": as_text(meta.get("group_summary")),
        "prompt": as_text(meta.get("prompt")),
        "published_match_rate": as_float(meta.get("published_match_rate")),
        "elo": as_float(meta.get("elo")),
        "elo_axes": {k: as_float(v) for k, v in as_dict(meta.get("elo_axes")).items()},
        "n_group_members": as_int(meta.get("n_group_members")),
        "rubric": as_text(meta.get("transcript_rubric")),
        "weirdchat_url": as_text(meta.get("weirdchat_url")),
        "samples": samples,
        "reads": reads,
        "fork": {
            "prefix_chars": as_int(fork.get("prefix_chars")),
            "prefix_words": as_int(fork.get("prefix_words")),
            "note": as_text(fork.get("note")),
        }
        if fork
        else None,
        "runs": runs,
        "n_mechanisms": sum(len(r["mechanisms"]) for r in runs),
    }


def behaviors_of(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per behavior: how many patterns it has and their mean published match rate."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for pat in patterns:
        groups.setdefault(pat["behavior_id"], []).append(pat)
    rows = []
    for bid, members in groups.items():
        rates = [m["published_match_rate"] for m in members]
        rates = [r for r in rates if r is not None]
        name = next((m["behavior_name"] for m in members if m["behavior_name"]), bid)
        rows.append(
            {
                "behavior_id": bid,
                "behavior_name": name,
                "n_patterns": len(members),
                "mean_match_rate": (sum(rates) / len(rates)) if rates else None,
                "n_runs": sum(len(m["runs"]) for m in members),
            }
        )
    rows.sort(key=lambda r: (-r["n_patterns"], r["behavior_name"]))
    return rows


def clusters_of(out_root: Path) -> list[dict[str, Any]]:
    synth = read_json(out_root / "synth.json") or {}
    clusters = []
    for entry in as_list(synth.get("clusters")):
        entry = as_dict(entry)
        clusters.append(
            {
                "name": as_text(entry.get("name")) or "unnamed cluster",
                "description": as_text(entry.get("description")),
                "behavior_ids": [as_text(b) for b in as_list(entry.get("behavior_ids"))],
                "members": [
                    {
                        "pattern_key": as_text(m.get("pattern_key")),
                        "mechanism": as_text(m.get("mechanism")),
                        "evidence": as_text(m.get("evidence")),
                    }
                    for m in (as_dict(x) for x in as_list(entry.get("members")))
                ],
            }
        )
    return clusters


def agreements_of(out_root: Path) -> list[dict[str, Any]]:
    """agreement.json (olens vs blackbox) plus every agreement_<arm>_vs_<arm_b>.json."""
    paths = [out_root / "agreement.json"] + sorted(out_root.glob("agreement_*_vs_*.json"))
    out = []
    for path in paths:
        entry = agreement_of(path)
        if entry is not None:
            m = re.match(r"agreement_(.+)_vs_(.+)\.json$", path.name)
            entry["arm"] = entry["arm"] or (m.group(1) if m else "olens")
            entry["arm_b"] = entry["arm_b"] or (m.group(2) if m else "blackbox")
            entry["file"] = path.name
            out.append(entry)
    return out


def predictions_of(out_root: Path) -> list[dict[str, Any]]:
    """predictions_<arm>.json: did the arm's mechanisms predict each intervention's direction."""
    out = []
    for path in sorted(out_root.glob("predictions_*.json")):
        blob = read_json(path)
        if blob is None:
            continue
        pats: dict[str, dict[str, Any]] = {}
        for entry in (as_dict(x) for x in as_list(blob.get("patterns"))):
            key = as_text(entry.get("pattern_key"))
            if not key:
                continue
            pats[key] = {
                "n_arms": as_int(entry.get("n_arms")),
                "right": as_int(entry.get("right")),
                "wrong": as_int(entry.get("wrong")),
                "not_predicted": as_int(entry.get("not_predicted")),
                "arms": [
                    {
                        "arm": as_text(a.get("arm")),
                        "predicted": bool(a.get("predicted")),
                        "direction": as_text(a.get("direction")),
                        "mechanism": as_text(a.get("mechanism")),
                        "measured": as_float(a.get("measured")),
                        "delta": as_float(a.get("delta")),
                        "p": as_float(a.get("p")),
                        "verdict": as_text(a.get("verdict")) or "not_predicted",
                    }
                    for a in (as_dict(x) for x in as_list(entry.get("arms")))
                ],
            }
        out.append(
            {
                "arm": as_text(blob.get("arm")) or path.stem.replace("predictions_", ""),
                "model": as_text(blob.get("model")),
                "summary": {k: as_int(v) for k, v in as_dict(blob.get("summary")).items()},
                "patterns": pats,
                "file": path.name,
            }
        )
    return out


def agreement_of(path: Path) -> dict[str, Any] | None:
    """One agreement file: the reader's summary plus one entry per pattern."""
    blob = read_json(path)
    if blob is None:
        return None
    entries: dict[str, dict[str, Any]] = {}
    for entry in (as_dict(x) for x in as_list(blob.get("patterns"))):
        key = as_text(entry.get("pattern_key"))
        if not key:
            continue
        pick = lambda lst: [  # noqa: E731
            {"mechanism": as_text(m.get("mechanism")), "confidence": as_float(m.get("confidence"))}
            for m in (as_dict(x) for x in as_list(lst))
        ]
        entries[key] = {
            "n_lens": as_int(entry.get("n_lens")),
            "n_blackbox": as_int(entry.get("n_blackbox")),
            "lens_with_counterpart": as_int(entry.get("lens_with_counterpart")),
            "blackbox_with_counterpart": as_int(entry.get("blackbox_with_counterpart")),
            "lens_only": pick(entry.get("lens_only")),
            "blackbox_only": pick(entry.get("blackbox_only")),
            "top_match": entry.get("top_match")
            if isinstance(entry.get("top_match"), bool)
            else None,
            "lens_only_summary": as_text(entry.get("lens_only_summary")),
            "blackbox_only_summary": as_text(entry.get("blackbox_only_summary")),
        }
    summary = {k: as_int(v) for k, v in as_dict(blob.get("summary")).items()}
    return {
        "summary": summary,
        "patterns": entries,
        "model": as_text(blob.get("model")),
        "arm": as_text(blob.get("arm")),
        "arm_b": as_text(blob.get("arm_b")),
    }


def calibrations_of(out_root: Path) -> list[dict[str, Any]]:
    """interventions/calibration.json and every calibration_<judge>.json: each judge vs the study's labels."""
    folder = out_root / "interventions"
    if not folder.is_dir():
        return []
    out = []
    for path in [folder / "calibration.json"] + sorted(folder.glob("calibration_*.json")):
        blob = read_json(path)
        if blob is None:
            continue
        conf = as_dict(blob.get("confusion"))
        out.append(
            {
                "n": as_int(blob.get("n")),
                "judge_failures": as_int(blob.get("judge_failures")),
                "agreement": as_float(blob.get("agreement")),
                "kappa": as_float(blob.get("kappa")),
                "confusion": {k: as_int(conf.get(k)) for k in ("tp", "tn", "fp", "fn")},
                "model": as_text(blob.get("model")) or path.stem.replace("calibration_", ""),
                "file": path.name,
            }
        )
    return out


def calibration_for(judge: str, calibrations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The calibration of the judge that scored an intervention file (model names may carry a vendor prefix)."""
    j = (judge or "").lower()
    for c in calibrations:
        m = (c["model"] or "").lower()
        if (
            j
            and m
            and (
                m == j
                or m.endswith("/" + j)
                or j.endswith("/" + m)
                or m.split("/")[-1] == j.split("/")[-1]
            )
        ):
            return c
    return None


def interventions_of(out_root: Path, key: str) -> dict[str, Any] | None:
    """interventions/<key>.json: one row per arm; the first arm is the unchanged baseline."""
    blob = read_json(out_root / "interventions" / f"{key}.json")
    if blob is None:
        return None
    arms = []
    for entry in (as_dict(x) for x in as_list(blob.get("arms"))):
        arm = as_dict(entry.get("arm"))
        ci = [as_float(v) for v in as_list(entry.get("ci95"))[:2]]
        arms.append(
            {
                "name": as_text(arm.get("name")) or f"arm {len(arms)}",
                "prompt": as_text(arm.get("prompt")),
                "system": as_text(arm.get("system")),
                "prefill": as_text(arm.get("prefill")),
                "note": as_text(arm.get("note")),
                "n": as_int(entry.get("n")),
                "k": as_int(entry.get("k")),
                "rate": as_float(entry.get("rate")),
                "ci95": ci if len(ci) == 2 else None,
                "judge_failures": as_int(entry.get("judge_failures")),
                "replies": [as_text(r) for r in as_list(entry.get("replies"))],
                "verdicts": [
                    v if isinstance(v, bool) else None for v in as_list(entry.get("verdicts"))
                ],
                "explanations": [as_text(e) for e in as_list(entry.get("explanations"))],
                "delta_vs_baseline": as_float(entry.get("delta_vs_baseline")),
                "fisher_p_vs_baseline": as_float(entry.get("fisher_p_vs_baseline")),
            }
        )
    return {
        "n_per_arm": as_int(blob.get("n_per_arm")),
        "judge_model": as_text(blob.get("judge_model")),
        "arms": arms,
    }


FLAG_CATEGORIES = (
    "role_adoption",
    "both_branches",
    "hidden_referent",
    "disclaimer_present",
    "contradicts_text",
    "commitment_point",
    "other",
)


def flags_of(out_root: Path, pat: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """flags/<key>.json re-verified: a flag stays only if its quote is verbatim in the cited cell."""
    blob = read_json(out_root / "flags" / f"{pat['key']}.json")
    stats = {"kept": 0, "dropped": 0, "proposed": 0, "unverified": 0}
    if blob is None:
        return None, stats
    by_id = {r["id"]: r for r in pat["reads"]}
    out: dict[str, Any] = {}
    for rid, entry in as_dict(blob.get("reads")).items():
        entry = as_dict(entry)
        read = by_id.get(rid)
        kept, dropped = [], 0
        for fl in (as_dict(x) for x in as_list(entry.get("flags"))):
            pos, layer, quote = (
                as_int(fl.get("position")),
                as_int(fl.get("layer")),
                as_text(fl.get("quote")),
            )
            row = (
                next((r for r in (read["rows"] or []) if r["pos"] == pos), None)
                if read and read.get("rows")
                else None
            )
            li = read["layers"].index(layer) if (read and layer in read["layers"]) else -1
            samples = row["samples"][li] if (row and li >= 0 and li < len(row["samples"])) else []
            if quote and any(quote in smp for smp in samples):
                kept.append(
                    {
                        "position": pos,
                        "layer": layer,
                        "category": as_text(fl.get("category"))
                        if as_text(fl.get("category")) in FLAG_CATEGORIES
                        else "other",
                        "quote": quote,
                        "why": as_text(fl.get("why")),
                    }
                )
            else:
                dropped += 1
        kept.sort(key=lambda f: (f["position"] or 0, f["layer"] or 0))
        out[rid] = {
            "flags": kept,
            "n_windows": as_int(entry.get("n_windows")),
            "failed_calls": as_int(entry.get("failed_calls")),
            "proposed": as_int(entry.get("proposed")),
            "unverified": as_int(entry.get("unverified")),
            "dropped_here": dropped,
            "model": as_text(entry.get("model")),
            "in_build": read is not None,
        }
        stats["kept"] += len(kept)
        stats["dropped"] += dropped
        stats["proposed"] += as_int(entry.get("proposed")) or 0
        stats["unverified"] += as_int(entry.get("unverified")) or 0
    return out, stats


def synth_meta(out_root: Path) -> dict[str, Any]:
    synth = read_json(out_root / "synth.json") or {}
    return {"model": as_text(synth.get("model")), "n_records": as_int(synth.get("n_records"))}


# ------------------------------------------------------------------- split
HEAVY_PATTERN_KEYS = ("samples", "reads", "fork", "brief", "interventions", "predictions", "flags")
HEAVY_RUN_KEYS = ("steps", "notes")


def grade_of(pat: dict[str, Any]) -> str:
    """run = an agent run reported ≥1 mechanism; diag = diagnostics only; data = neither."""
    if any(r["mechanisms"] for r in pat["runs"]):
        return "run"
    if any(r["source"] == "diag" for r in pat["reads"]):
        return "diag"
    return "data"


def light_pattern(pat: dict[str, Any]) -> dict[str, Any]:
    """What index.html carries per pattern: metadata, flags, and every run's mechanisms."""
    light = {k: v for k, v in pat.items() if k not in HEAVY_PATTERN_KEYS}
    light["runs"] = [{k: v for k, v in r.items() if k not in HEAVY_RUN_KEYS} for r in pat["runs"]]
    light["has_diag"] = any(r["source"] == "diag" for r in pat["reads"])
    light["n_reads"] = len(pat["reads"])
    light["n_samples"] = len(pat["samples"])
    light["n_runs"] = len(pat["runs"])
    light["grade"] = grade_of(pat)
    light["has_blackbox"] = any(r["arm"] == "blackbox" for r in pat["runs"])
    light["arms"] = sorted({r["arm"] or "olens" for r in pat["runs"]})
    light["agreements"] = pat.get("agreements") or []
    fl = pat.get("flags") or {}
    by_id = {r["id"]: r for r in pat["reads"]}
    counts = {"matched": 0, "unmatched": 0, "other": 0}
    for rid, entry in fl.items():
        side = by_id[rid]["label"] if rid in by_id and by_id[rid]["source"] == "diag" else "other"
        counts[side if side in counts else "other"] += len(entry["flags"])
    light["flags_meta"] = (
        {
            "matched": counts["matched"],
            "unmatched": counts["unmatched"],
            "proposed": sum(e["proposed"] or 0 for e in fl.values()),
            "dropped": sum((e["unverified"] or 0) + e["dropped_here"] for e in fl.values()),
            "model": next((e["model"] for e in fl.values() if e["model"]), ""),
        }
        if fl
        else None
    )
    light["flag_whys"] = [
        {"read": rid, "position": f["position"], "category": f["category"], "why": f["why"]}
        for rid, e in fl.items()
        for f in e["flags"]
        if f["why"]
    ]
    iv = pat.get("interventions")
    light["interventions_meta"] = (
        {"n_per_arm": iv["n_per_arm"], "judge_model": iv["judge_model"], "n_arms": len(iv["arms"])}
        if iv
        else None
    )
    light["intervention_notes"] = [
        {"name": a["name"], "note": a["note"]} for a in (iv["arms"] if iv else [])
    ]
    return light


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def read_stub(read: dict[str, Any]) -> dict[str, Any]:
    """An agent read without its rows — the page fetches them from the .agent.json file."""
    stub = {k: v for k, v in read.items() if k != "rows"}
    stub["rows"] = None
    stub["deferred"] = True
    return stub


def write_pattern_files(pat: dict[str, Any], data_dir: Path) -> list[tuple[str, int]]:
    """data/<key>.json, splitting the agent reads into data/<key>.agent.json when over budget."""
    main_path = data_dir / f"{pat['key']}.json"
    agent_path = data_dir / f"{pat['key']}.agent.json"
    text = dumps(pat)
    agent_reads = [r for r in pat["reads"] if r["source"] == "agent" and r["rows"]]
    written: list[tuple[str, int]] = []
    if len(text.encode()) > DATA_FILE_BUDGET and agent_reads:
        deferred = {r["id"] for r in agent_reads}
        split = dict(pat)
        split["reads"] = [read_stub(r) if r["id"] in deferred else r for r in pat["reads"]]
        split["agent_file"] = f"data/{agent_path.name}"
        text = dumps(split)
        agent_path.write_text(dumps({"reads": agent_reads}))
        written.append((f"data/{agent_path.name}", agent_path.stat().st_size))
    elif agent_path.exists():
        agent_path.unlink()
    main_path.write_text(text)
    written.insert(0, (f"data/{main_path.name}", main_path.stat().st_size))
    return written


def mb(n_bytes: int) -> str:
    return f"{n_bytes / 1048576:.2f} MB"


def report_reads(patterns: list[dict[str, Any]]) -> None:
    """Print how every agent readout page parsed against the header's own claim."""
    agent = [(p["key"], r) for p in patterns for r in p["reads"] if r["source"] == "agent"]
    if not agent:
        return
    clean = 0
    print(f"agent readout pages: {len(agent)}")
    for key, r in agent:
        claim = r["claimed"]
        got = f"{len(r['rows'] or [])} pos x {len(r['layers'])} layers"
        want = f"{claim.get('positions')} pos x {claim.get('layers')} layers"
        if r["parse_error"]:
            print(f"  FAILED  {key} {r['id']} (tool #{r['tool_index']}): {r['parse_error'][:140]}")
        elif (len(r["rows"]), len(r["layers"])) != (claim.get("positions"), claim.get("layers")):
            print(f"  PARTIAL {key} {r['id']}: parsed {got}, header claims {want}")
        else:
            clean += 1
    print(f"  {clean}/{len(agent)} parsed cleanly (positions and layers match the header)")


# -------------------------------------------------------------- highlights
# a highlight = one lens cell a mechanism quotes that verifies verbatim against the read
QUOTE_RES = (
    re.compile(r"“([^”]{20,})”"),
    re.compile(r'"([^"]{20,})"'),
    re.compile(r"(?<!\w)'(.{20,}?)'(?!\w)"),
)
CID_RE = re.compile(r"\b([wc]\d+[mu]?)\b")
POSREF_RE = re.compile(r"\bpos\s*~?\s*(\d+)")
LAYERREF_RE = re.compile(r"\bL(\d+)(?:\s*[-–]\s*L?(\d+))?")
WS_RE = re.compile(r"\s+")


STRAY_QUOTE_RE = re.compile(r"[\"“”]|(?<!\w)'|'(?!\w)")
CITATION_RE = re.compile(r"\bpos\s*~?\s*\d+|\bL\d\d\b")
ELLIPSIS_RE = re.compile(r"…|\.\.\.")


def quoted_fragments(text: str) -> list[tuple[int, str]]:
    """(offset, fragment) for every quoted run of ≥ 20 chars, in text order.

    A span that itself contains a stray quote mark or citation syntax (pos N / LNN) is the
    scanner bridging two mismatched quotes, not something the agent quoted — dropped."""
    found: set[tuple[int, str]] = set()
    for rx in QUOTE_RES:
        for m in rx.finditer(text):
            q = m.group(1).strip()
            if len(q) >= 20 and not STRAY_QUOTE_RE.search(q) and not CITATION_RE.search(q):
                found.add((m.start(1), q))
    return sorted(found)


def last_match(rx: re.Pattern[str], text: str) -> re.Match[str] | None:
    last = None
    for m in rx.finditer(text):
        last = m
    return last


def reads_by_conv(pat: dict[str, Any], run_index: int | None = None) -> dict[str, dict[str, Any]]:
    """conv id → the read that actually parsed, from one run when given (each arm reads the same
    conversations; a failed retry of the same id never wins)."""
    out: dict[str, dict[str, Any]] = {}
    for r in pat["reads"]:
        if run_index is not None and r.get("run_index") != run_index:
            continue
        if r["conv_id"] and r["rows"] and r["conv_id"] not in out:
            out[r["conv_id"]] = r
    return out


def local_text(read: dict[str, Any], pos: int, n_tokens: int = 22) -> str:
    """The read tokens up to and including `pos` (thinned, so approximate), last n_tokens."""
    toks = [row["tok"] for row in read["rows"] if row["pos"] <= pos]
    return "".join(toks[-n_tokens:])


def find_sample(
    read: dict[str, Any], row: dict[str, Any], layers: list[int] | None, quote: str
) -> tuple[int, str] | None:
    """(layer, sample) of the first sample at this position containing `quote` verbatim."""
    for li, layer in enumerate(read["layers"]):
        if layers is not None and layer not in layers:
            continue
        for sample in row["samples"][li] if li < len(row["samples"]) else []:
            if quote in sample:
                return layer, sample
    return None


def layer_range(m: re.Match[str] | None) -> list[int] | None:
    if m is None:
        return None
    lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
    return list(range(min(lo, hi), max(lo, hi) + 1))


def card_of(
    pat: dict[str, Any],
    read: dict[str, Any],
    row: dict[str, Any],
    layer: int,
    sample: str,
    quote: str,
    note: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "pattern_key": pat["key"],
        "behavior": pat["behavior_name"],
        "summary": pat["group_summary"],
        "read": read["id"],
        "pos": row["pos"],
        "layer": layer,
        "region": row["region"],
        "token": row["tok"],
        "local": local_text(read, row["pos"]),
        "sample": sample,
        "quote": quote,
        "note": clip(note, 200),
        "kind": kind,
    }


def auto_highlights(patterns: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every quoted fragment in every mechanism's readout_cells, verified against the cited cell."""
    cards: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "fragments": 0,
        "verified": 0,
        "failed": 0,
        "ws_only": 0,
        "unresolved": 0,
        "ellipsis": 0,
        "failures": [],
        "per_pattern": {},
    }
    seen: set[tuple[str, str, int, int, str]] = set()
    for pat in patterns:
        for run in pat["runs"]:
            by_conv = reads_by_conv(pat, run["run_index"]) or reads_by_conv(pat)
            for mi, mech in enumerate(run["mechanisms"]):
                text = mech["readout_cells"]
                for start, quote in quoted_fragments(text):
                    stats["fragments"] += 1
                    per = stats["per_pattern"].setdefault(pat["key"], [0, 0])
                    per[1] += 1
                    prefix = text[:start]
                    cid, posm, laym = (
                        last_match(CID_RE, prefix),
                        last_match(POSREF_RE, prefix),
                        last_match(LAYERREF_RE, prefix),
                    )
                    read = by_conv.get(cid.group(1)) if cid else None
                    row = None
                    if read is not None and posm is not None:
                        pos = int(posm.group(1))
                        row = next((r for r in read["rows"] if r["pos"] == pos), None)
                    where = f"{pat['key']} run{run['run_index']} mech{mi + 1}: {cid.group(1) if cid else '?'} pos {posm.group(1) if posm else '?'} {laym.group(0) if laym else ''}"
                    if row is None:
                        stats["failed"] += 1
                        stats["unresolved"] += 1
                        stats["failures"].append(
                            f"{where} — cited read/position not in the build: {quote[:80]!r}"
                        )
                        continue
                    hit = find_sample(read, row, layer_range(laym), quote)
                    if hit is None and laym is not None:
                        hit = find_sample(
                            read, row, None, quote
                        )  # the layer was wrong, the position right
                        if hit is not None:
                            where += f" (found at L{hit[0]}, not the cited layer)"
                    if hit is None:
                        stats["failed"] += 1
                        if ELLIPSIS_RE.search(quote):
                            stats["ellipsis"] += 1
                            where += " (the quote contains an ellipsis — agent-side truncation)"
                        norm_q = WS_RE.sub(" ", quote)
                        if any(norm_q in WS_RE.sub(" ", s) for ss in row["samples"] for s in ss):
                            stats["ws_only"] += 1
                            where += " (matches after whitespace normalisation)"
                        stats["failures"].append(
                            f"{where} — not verbatim in any sample at that position: {quote[:80]!r}"
                        )
                        continue
                    stats["verified"] += 1
                    per[0] += 1
                    layer, sample = hit
                    sig = (pat["key"], read["id"], row["pos"], layer, quote)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    cards.append(
                        card_of(pat, read, row, layer, sample, quote, mech["mechanism"], "auto")
                    )
    return cards, stats


def hand_highlights(patterns: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    """Hand-picked items; one that does not verify verbatim is a build error."""
    by_key = {p["key"]: p for p in patterns}
    cards = []
    for item in as_list((read_json(path) or {}).get("items")):
        item = as_dict(item)
        pat = by_key.get(as_text(item.get("pattern_key")))
        if pat is None:
            raise ValueError(f"highlight pattern missing: {item}")
        read = next((r for r in pat["reads"] if r["id"] == as_text(item.get("read"))), None)
        if read is None or not read["rows"]:
            raise ValueError(f"highlight read missing or unparsed: {item}")
        pos = as_int(item.get("pos"))
        row = next((r for r in read["rows"] if r["pos"] == pos), None)
        if row is None:
            raise ValueError(f"highlight position not read: {item}")
        layer = as_int(item.get("layer"))
        quote = as_text(item.get("quote"))
        hit = find_sample(read, row, [layer] if layer is not None else None, quote)
        if hit is None:
            raise ValueError(f"highlight quote not found verbatim: {item}")
        cards.append(
            card_of(pat, read, row, hit[0], hit[1], quote, as_text(item.get("note")), "hand")
        )
    return cards


# ------------------------------------------------------------ translations
NONLATIN_RE = re.compile("[Ѐ-ӿ؀-ۿऀ-ॿ฀-๿ᄀ-ᇿ　-ヿ㐀-䶿一-鿿가-힯＀-￯]")


def all_samples(patterns: list[dict[str, Any]]) -> set[str]:
    return {
        s
        for p in patterns
        for r in p["reads"]
        for row in r["rows"] or []
        for ss in row["samples"]
        for s in ss
    }


# ------------------------------------------------------------------------ main
def build_site(
    out_root: Path,
    site_dir: Path,
    *,
    single: bool = False,
    translations: Path | None = None,
    highlights: Path | None = None,
) -> Path:
    """Collect everything under ``out_root``; write ``site_dir/index.html`` (+ ``data/*.json``)."""
    pattern_dir = out_root / "patterns"
    paths = sorted(pattern_dir.glob("*.json")) if pattern_dir.is_dir() else []
    patterns = [pattern_of(p, out_root) for p in paths]
    patterns.sort(key=lambda p: (p["behavior_name"], p["key"]))
    brief_fn, system_fn, brief_how = load_prompts()
    agreements = agreements_of(out_root)
    predictions = predictions_of(out_root)
    calibrations = calibrations_of(out_root)
    judges = [
        p_iv["judge_model"]
        for p_iv in (interventions_of(out_root, p["key"]) for p in patterns)
        if p_iv
    ]
    main_judge = max(set(judges), key=judges.count) if judges else ""
    calibration = calibration_for(main_judge, calibrations) or (
        calibrations[0] if calibrations else None
    )
    flag_stats = {"kept": 0, "dropped": 0, "proposed": 0, "unverified": 0, "files": 0}
    for pat in patterns:
        pat["brief"] = brief_of(pat, brief_fn)
        pat["flags"], fstats = flags_of(out_root, pat)
        if pat["flags"] is not None:
            flag_stats["files"] += 1
            for k in ("kept", "dropped", "proposed", "unverified"):
                flag_stats[k] += fstats[k]
        pat["interventions"] = interventions_of(out_root, pat["key"])
        pat["agreements"] = [
            {"arm": a["arm"], "arm_b": a["arm_b"], **a["patterns"][pat["key"]]}
            for a in agreements
            if pat["key"] in a["patterns"]
        ]
        pat["agreement"] = pat["agreements"][0] if pat["agreements"] else None
        pat["predictions"] = {
            pr["arm"]: pr["patterns"][pat["key"]]
            for pr in predictions
            if pat["key"] in pr["patterns"]
        }
    print(f"brief: {brief_how}")
    print(
        f"reader flags: {flag_stats['files']} files · {flag_stats['kept']} kept (verbatim in the cited cell) · "
        f"{flag_stats['dropped']} dropped at build time as non-verbatim · file says {flag_stats['proposed']} proposed, "
        f"{flag_stats['unverified']} unverified"
    )
    arm_counts = {
        a: sum(1 for p in patterns if any((r["arm"] or "olens") == a for r in p["runs"]))
        for a in ("olens", "jlens", "nla", "blackbox")
    }
    n_iv = sum(1 for p in patterns if p["interventions"])
    print(
        "runs per arm: "
        + " · ".join(f"{a} {n}" for a, n in arm_counts.items())
        + " · agreement files: "
        + (", ".join(f"{a['file']} ({len(a['patterns'])} patterns)" for a in agreements) or "none")
        + " · predictions: "
        + (", ".join(f"{p['file']} ({len(p['patterns'])})" for p in predictions) or "none")
        + f" · interventions on {n_iv} patterns (judge: {main_judge or '—'}) · calibrations: "
        + (", ".join(f"{c['file']} κ={c['kappa']}" for c in calibrations) or "absent")
    )

    cards, hl_stats = auto_highlights(patterns)
    hand = hand_highlights(patterns, highlights) if highlights else []
    cards = hand + cards
    for pat in patterns:  # per-pattern "cited cells verified k/n", shown in the agent-summary pane
        k, n = hl_stats["per_pattern"].get(pat["key"], (0, 0))
        pat["cited_verified"], pat["cited_fragments"] = k, n
    samples = all_samples(patterns)
    if translations is None and (out_root / "translations.json").is_file():
        translations = out_root / "translations.json"
    cache = read_json(translations) if translations else None
    en = {s: as_text(e) for s, e in (cache or {}).items() if s in samples and as_text(e)}
    to_translate = sorted(s for s in samples if NONLATIN_RE.search(s))

    data = {
        "patterns": [light_pattern(p) for p in patterns],
        "behaviors": behaviors_of(patterns),
        "clusters": clusters_of(out_root),
        "synth": synth_meta(out_root),
        "highlights": cards,
        "verify": {"verified": hl_stats["verified"], "fragments": hl_stats["fragments"]},
        "system_prompt": system_fn() if system_fn else None,
        "agreement_summary": agreements[0]["summary"] if agreements else None,
        "agreement_model": agreements[0]["model"] if agreements else None,
        "agreements": [
            {k: a[k] for k in ("arm", "arm_b", "summary", "model", "file")} for a in agreements
        ],
        "predictions": [
            {k: p[k] for k in ("arm", "summary", "model", "file")} for p in predictions
        ],
        "calibration": calibration,
        "calibrations": calibrations,
        "brief_how": brief_how,
        "n_side": N_SIDE,
        "en": en,
        "counts": {
            "patterns": len(patterns),
            "behaviors": len({p["behavior_id"] for p in patterns}),
            "runs": sum(len(p["runs"]) for p in patterns),
            "with_diag": sum(1 for p in patterns if any(r["source"] == "diag" for r in p["reads"])),
            "reads": sum(len(p["reads"]) for p in patterns),
            "mechanisms": sum(p["n_mechanisms"] for p in patterns),
        },
        "source": str(out_root),
        "single": single,
    }
    site_dir.mkdir(parents=True, exist_ok=True)
    sizes: list[tuple[str, int]] = []
    warnings: list[str] = []
    if single:
        data["heavy"] = {p["key"]: p for p in patterns}
    else:
        data_dir = site_dir / "data"
        data_dir.mkdir(exist_ok=True)
        for pat in patterns:
            for name, size in write_pattern_files(pat, data_dir):
                sizes.append((name, size))
                if size > DATA_FILE_BUDGET:
                    warnings.append(f"{name} is {mb(size)} > {mb(DATA_FILE_BUDGET)} budget")

    payload = dumps(data).replace("</", "<\\/")
    out = site_dir / "index.html"
    out.write_text(TEMPLATE.replace("/*__DATA__*/", payload))
    sizes.insert(0, ("index.html", out.stat().st_size))
    cells_path = site_dir / "cells_to_translate.json"
    cells_path.write_text(json.dumps(to_translate, ensure_ascii=False, indent=0))
    cells_size = cells_path.stat().st_size
    total = sum(s for _, s in sizes)
    if total > TOTAL_BUDGET:
        warnings.append(f"site total is {mb(total)} > {mb(TOTAL_BUDGET)} budget (informational)")

    for name, size in sizes:
        print(f"{mb(size):>10}  {name}")
    report_reads(patterns)
    print(
        f"quoted cell fragments in mechanisms: {hl_stats['fragments']} · verified verbatim "
        f"{hl_stats['verified']} · FAILED {hl_stats['failed']} "
        f"({hl_stats['unresolved']} cite a read/position not in the build, "
        f"{hl_stats['ellipsis']} contain an ellipsis = agent-side truncation, "
        f"{hl_stats['ws_only']} match only after whitespace normalisation)"
    )
    for line in hl_stats["failures"]:
        print(f"  unverified: {line}")
    print(f"{len(cards)} highlights ({len(cards) - len(hand)} auto, {len(hand)} hand-picked)")
    print(
        f"{len(to_translate)} cells to translate (non-Latin script) → {cells_path.name} "
        f"({mb(cells_size)}, a work file — not counted in the site total); "
        f"{len(en)} translations attached from the cache"
        f"{' (' + str(translations) + ')' if translations else ''}"
    )
    c = data["counts"]
    print(
        f"{len(patterns)} patterns · {c['behaviors']} behaviors · {c['runs']} agent runs · "
        f"{c['reads']} reads · {c['mechanisms']} mechanisms · {len(data['clusters'])} clusters "
        f"· {'single self-contained file' if single else 'split'} · total {mb(total)} in {len(sizes)} files"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(out.resolve())
    return out


def main() -> None:
    args: dict[str, str] = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out_root = Path(args.get("out", "outputs/weirdchat"))
    site_dir = Path(args["site"]) if "site" in args else out_root / "site"
    build_site(
        out_root,
        site_dir,
        single=args.get("single", "").lower() in ("1", "true", "yes"),
        translations=Path(args["translations"]) if args.get("translations") else None,
        highlights=Path(args["highlights"]) if args.get("highlights") else None,
    )


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WeirdChat × OLens</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;1,400&display=swap">
<style>
:root{
  --ground:#f3f5f7; --surface:#ffffff; --surface-2:#e9edf0; --line:#cfd8de; --line-soft:#e1e7eb;
  --text:#152230; --text-dim:#5a6b78; --text-faint:#8695a2;
  --accent:#1d6f8b; --accent-soft:#dbeaf0; --accent-ink:#0d4c62;
  --hit:#20714c; --hit-soft:#d9eee2; --miss:#a8431c; --miss-soft:#f6e0d5;
  --hold:#8a6410; --hold-soft:#f5e9cf;
  --sample:#8a6d00; --sample-soft:#fff3a8; --sample-line:#b89a1e;
  --find:#5b3fa8; --find-soft:#e6ddff; --find-ink:#3d2a80;
  --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif; --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --serif:"IBM Plex Serif",Georgia,serif;
  --col:300px; --rowmax:none; --split:46%;
}
:root[data-theme="dark"]{
  --ground:#0e161d; --surface:#152029; --surface-2:#1b2833; --line:#2a3d4a; --line-soft:#223341;
  --text:#e6edf2; --text-dim:#9fb2bf; --text-faint:#6d8393;
  --accent:#57b6d0; --accent-soft:#16333f; --accent-ink:#9ad8e8; --hit:#5cc294; --hit-soft:#12332a;
  --miss:#e08b62; --miss-soft:#3a2118; --hold:#d9b45e; --hold-soft:#33290f;
  --sample:#f0d24a; --sample-soft:#4a3d05; --sample-line:#e2c53a; --find:#b9a3ff; --find-soft:#2c1f5c; --find-ink:#d6c8ff;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--ground);color:var(--text);font-family:var(--sans);font-size:13px;line-height:1.45;display:flex;flex-direction:column;overflow:hidden}
button,input,select{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
a{color:var(--accent)}
.kbd{font-family:var(--mono);font-size:11px;background:var(--surface-2);border:1px solid var(--line);border-radius:3px;padding:0 5px;color:var(--text-dim)}
/* top bar */
#bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:8px 14px;padding-top:calc(8px + env(safe-area-inset-top,0px));background:var(--surface);border-bottom:1px solid var(--line);flex:none}
#bar h1{margin:0 8px 0 0;font-size:14px;font-weight:600}
#bar h1 a{color:inherit;text-decoration:none}
.pick{display:flex;align-items:center;gap:5px;font-size:10.5px;color:var(--text-faint);text-transform:uppercase;letter-spacing:.1em}
.pick select{font-family:var(--mono);font-size:12px;text-transform:none;letter-spacing:0;padding:3px 6px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--text);max-width:300px}
.btn{font-size:12px;line-height:1;padding:4px 8px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--text-dim);cursor:pointer}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent);color:var(--accent-ink);font-weight:600}
.btn:disabled{opacity:.5;cursor:default}
.btn.cmp{font-family:var(--mono);font-size:11px;padding:3px 6px}
.btn.cmp i{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px;vertical-align:0}
i.matched{background:var(--miss)} i.unmatched{background:var(--hit)} i.agent{background:var(--text-faint)} i.err{background:var(--hold)}
#search,#find{padding:4px 8px;border:1px solid var(--line);border-radius:4px;background:var(--ground);font-family:var(--mono);font-size:12px;width:230px}
#find{border-color:var(--find)}
.spacer{flex:1}
.sub{font-size:11.5px;color:var(--text-dim)}
#primer{flex:none;border-bottom:1px solid var(--line)}
#primer .p1{background:var(--surface);color:var(--text);font-size:11.5px;padding:4px 14px;border-bottom:1px solid var(--line-soft)}
#primer .p2{background:var(--miss-soft);color:var(--miss);font-size:11.5px;padding:3px 14px}
#results{flex:none;background:var(--surface);border-bottom:1px solid var(--line);max-height:170px;overflow:auto;padding:4px 14px}
#results[hidden]{display:none}
.res{display:flex;gap:10px;align-items:baseline;width:100%;text-align:left;border:0;background:none;padding:3px 0;cursor:pointer;font-size:11.5px;color:var(--text)}
.res:hover{background:var(--surface-2)}
.res .where{font-family:var(--mono);font-size:11px;color:var(--find);flex:none;width:220px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.res .snip{font-family:var(--mono);font-size:11px;color:var(--text-dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.res .fam{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-faint);flex:none;width:200px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#results h4{margin:4px 0;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-faint);font-weight:600}
/* context panes */
#ctx{flex:none;background:var(--surface);border-bottom:1px solid var(--line);display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1.3fr) minmax(0,1fr) minmax(0,1fr)}
#ctx[hidden]{display:none}
#ctx .pane{padding:8px 14px;border-right:1px solid var(--line-soft);min-width:0;max-height:170px;overflow:auto}
#ctx .pane:last-child{border-right:0}
#ctx h3{margin:0 0 4px;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-faint);font-weight:600}
.prompt{font-family:var(--serif);font-size:13px;white-space:pre-wrap;word-break:break-word}
.mono{font-family:var(--mono);font-size:11.5px;white-space:pre-wrap;word-break:break-word;color:var(--text)}
.tags{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px}
.tag{font-family:var(--mono);font-size:11px;border-radius:3px;padding:1px 6px;background:var(--hold-soft);color:var(--text)}
.vrow{display:flex;gap:8px;align-items:baseline;font-size:11.5px;margin-bottom:3px}
.vrow .name{width:110px;flex:none;color:var(--text-dim)}
.badge{font-family:var(--mono);font-size:10.5px;border-radius:3px;padding:0 5px;border:1px solid transparent;white-space:nowrap}
.badge.hit{background:var(--hit-soft);color:var(--hit);border-color:var(--hit)}
.badge.miss{background:var(--miss-soft);color:var(--miss);border-color:var(--miss)}
.badge.hold{background:var(--hold-soft);color:var(--hold);border-color:var(--hold)}
.badge.dim{background:var(--surface-2);color:var(--text-dim);border-color:var(--line)}
.quote{font-family:var(--mono);font-size:11px;color:var(--text-dim);border-left:2px solid var(--line);padding-left:6px;margin:2px 0 4px;white-space:pre-wrap;word-break:break-word}
.empty{color:var(--text-faint);font-style:italic}
.dim{color:var(--text-dim)}
.clipbox{max-height:96px;overflow:hidden;position:relative}
.clipbox.open{max-height:none}
.showall{font-size:11px;color:var(--accent);background:none;border:0;padding:0;cursor:pointer}
/* hits row */
#hits{flex:none;background:var(--surface);border-bottom:1px solid var(--line);padding:4px 14px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;min-height:28px}
#hits[hidden]{display:none}
#hits .lbl{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);margin-right:4px}
.hitchip{font-family:var(--mono);font-size:11px;border:1px solid var(--find);background:var(--find-soft);color:var(--find-ink);border-radius:3px;padding:1px 6px;cursor:pointer;white-space:pre}
.hitchip:hover{filter:brightness(.95)}
/* main: two panes + splitter */
#main{flex:1;display:flex;min-height:0}
#text{flex:none;width:var(--split);min-width:240px;max-width:85%;overflow:auto;background:var(--surface);padding:10px 14px 30px;border-right:1px solid var(--line)}
#split{flex:none;width:7px;cursor:col-resize;background:var(--ground);touch-action:none}
#split:hover,#split.on{background:var(--accent)}
#grid{flex:1;min-width:0;display:flex;flex-direction:column;background:var(--surface)}
.blk{margin:0 0 10px}
.blk .role{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-faint);font-weight:600;margin:0 0 4px;display:flex;gap:8px;align-items:baseline}
.blk .role .lbl2{letter-spacing:0;text-transform:none;font-weight:400;color:var(--text-dim);font-size:11px}
.toks{display:flex;flex-wrap:wrap;gap:2px;padding:6px;background:var(--ground);border:1px solid var(--line-soft);border-radius:5px}
.blk.reply .toks{border-color:var(--line)}
.tok{font-family:var(--mono);font-size:12.5px;padding:2px 4px;border-radius:3px;border:1px solid transparent;cursor:pointer;white-space:pre;color:var(--text-dim);background:var(--surface)}
.tok:hover{border-color:var(--line)}
.tok.read{background:var(--sample-soft);color:var(--text);border-color:var(--sample-line)}
.tok.hit{border-top:3px solid var(--hit)}
.tok.found{box-shadow:inset 0 -3px 0 var(--find)}
.tok.mark{border-bottom:2px dotted var(--text)}
.tok.flagged{border-top:3px solid var(--hold);position:relative;margin-top:9px}
.tok.flagged::before{content:"⚑";position:absolute;top:-13px;left:1px;font-size:9px;line-height:1;color:var(--hold)}
.tok.flagged.hit{border-top-color:var(--hold)}
#flagbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:11px;color:var(--text-dim);margin:0 0 8px}
#flagbar select{font-family:var(--mono);font-size:11px;padding:1px 4px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--text)}
.fchip{font-family:var(--mono);font-size:11px;border:1px solid var(--hold);background:var(--hold-soft);color:var(--text);border-radius:3px;padding:1px 6px;cursor:pointer;white-space:pre}
.fchip.cur{outline:2px solid var(--accent)}
.fcard{border:1px solid var(--line-soft);border-left:3px solid var(--hold);border-radius:4px;background:var(--surface);padding:5px 8px;font-size:12px}
.fcard .q{font-family:var(--mono);font-size:11.5px;color:var(--text);margin:2px 0;white-space:pre-wrap;word-break:break-word}
.fcard .why{color:var(--text-dim)} .fcard .L{font-family:var(--mono);font-size:11px;color:var(--text-faint);margin-left:6px}
.cat{font-family:var(--mono);font-size:10.5px;border-radius:3px;padding:0 5px;border:1px solid transparent;text-transform:none}
.cat.c1{background:var(--miss-soft);color:var(--miss);border-color:var(--miss)} .cat.c2{background:var(--hold-soft);color:var(--hold);border-color:var(--hold)} .cat.c3{background:var(--find-soft);color:var(--find-ink);border-color:var(--find)} .cat.c0{background:var(--surface-2);color:var(--text-dim);border-color:var(--line)}
td.flagged .cell{box-shadow:inset 3px 0 0 var(--hold)} td.flagged.hit .cell{box-shadow:inset 3px 0 0 var(--hold), inset 6px 0 0 var(--hit)}
mark.flag{background:var(--hold-soft);color:var(--text);border-bottom:2px solid var(--hold);font-weight:600}
.tok.cur{background:var(--accent)!important;color:#fff;border-color:var(--accent)}
.tok.nodata{opacity:.4;cursor:default;background:transparent}
.tok .nl{font-size:10px;color:var(--text-faint)} .tok.cur .nl{color:#fff}
.tlegend{display:flex;gap:12px;flex-wrap:wrap;align-items:center;font-size:11px;color:var(--text-dim);margin:0 0 8px}
.tlegend .sw{width:11px;height:11px}
.prose{font-family:var(--sans);font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-word;background:var(--ground);border:1px solid var(--line-soft);border-radius:5px;padding:8px 10px;color:var(--text)}
.cmp2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}
.cmp2 .col{min-width:0}
.cmp2 .col h4{margin:0 0 4px;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-faint);font-weight:600}
.cmp2 .div.flagged{background:var(--miss-soft)} .cmp2 .div.clean{background:var(--hit-soft)}
.caption{font-size:11.5px;color:var(--text-dim);margin:8px 0 4px}
pre.brief{font-family:var(--mono);font-size:11.5px;line-height:1.5;white-space:pre-wrap;word-break:break-word;background:var(--ground);border:1px solid var(--line-soft);border-radius:4px;padding:8px 10px;margin:6px 0 0}
.qa{margin-top:4px} .qa .lab{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);margin:6px 0 2px} .qa .lab b{letter-spacing:0;text-transform:none;font-family:var(--mono);color:var(--text-dim);font-weight:400}
.qa .body{font-family:var(--sans);font-size:12.5px;white-space:pre-wrap;word-break:break-word;background:var(--ground);border:1px solid var(--line-soft);border-radius:4px;padding:6px 8px}
table.iv{table-layout:auto;width:100%;border:1px solid var(--line-soft);border-radius:4px;margin:6px 0}
table.iv th{position:static;width:auto;text-transform:uppercase;letter-spacing:.06em;font-size:10.5px;padding:5px 8px;color:var(--text-faint);background:var(--surface-2)}
table.iv td{padding:5px 8px;font-size:12px;vertical-align:top}
table.iv tr.base td{background:color-mix(in srgb, var(--sample-soft) 35%, var(--surface))}
table.iv td.up{background:var(--miss-soft);color:var(--miss);font-weight:600} table.iv td.down{background:var(--hit-soft);color:var(--hit);font-weight:600}
table.iv .chg{font-family:var(--mono);font-size:11px;white-space:pre-wrap;word-break:break-word;max-width:320px} table.iv .chg ins{background:var(--hit-soft);text-decoration:none} table.iv .chg del{background:var(--miss-soft);text-decoration:line-through}
.cibar{position:relative;height:6px;background:var(--surface-2);border-radius:3px;width:120px;margin-top:4px}
.cibar i{position:absolute;top:0;height:100%;background:var(--accent-soft);border:1px solid var(--accent);border-radius:3px}
.cibar b{position:absolute;top:-2px;width:2px;height:10px;background:var(--accent)}
.ex{border:1px solid var(--line-soft);border-radius:4px;padding:5px 8px;margin:4px 0;font-size:12px}
.ex .rep{font-family:var(--sans);white-space:pre-wrap;word-break:break-word} .ex .why{color:var(--text-dim);font-size:11.5px;margin-top:2px}
table.sum{border:1px solid var(--line-soft);table-layout:auto} table.sum th{position:static;width:auto;text-transform:none;letter-spacing:0;font-size:12px;padding:4px 10px;color:var(--text-dim)} table.sum td{padding:4px 10px;font-family:var(--mono);font-size:12px}
.armsw{display:flex;gap:4px;align-items:center;margin:0 0 8px;flex-wrap:wrap}
td .bag{display:flex;flex-wrap:wrap;gap:3px;padding:8px 10px}
td .bag span{font-family:var(--mono);font-size:11px;background:var(--ground);border:1px solid var(--line-soft);border-radius:3px;padding:0 4px;white-space:pre}
td .bag span.m{background:var(--hit-soft);border-color:var(--hit)} td .bag span.f{background:var(--find-soft);border-color:var(--find)}
.pred{font-family:var(--mono);font-size:12px} .pred.right{color:var(--hit);font-weight:600} .pred.wrong{color:var(--miss);font-weight:600} .pred.np{color:var(--text-faint)}
.score{display:flex;gap:10px;flex-wrap:wrap;font-size:11.5px;margin:0 0 8px} .score span{border:1px solid var(--line-soft);border-radius:4px;padding:2px 8px;background:var(--ground)}
.rolls{margin-top:14px}
.rolls h3{margin:0 0 4px;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-faint);font-weight:600}
.roll{display:flex;gap:8px;align-items:baseline;width:100%;text-align:left;border:1px solid var(--line-soft);border-left:3px solid var(--line);background:var(--surface);border-radius:4px;padding:4px 8px;margin:0 0 4px;cursor:pointer;font-size:11.5px}
.roll:hover{border-color:var(--accent)} .roll:disabled{cursor:default;opacity:.7}
.roll.matched{border-left-color:var(--miss)} .roll.unmatched{border-left-color:var(--hit)}
.roll .lab{font-family:var(--mono);font-size:10.5px;color:var(--text-dim);flex:none;width:130px}
.roll .snip{font-family:var(--mono);font-size:11px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
/* position header + grid */
#posbar{flex:none;padding:8px 14px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:6px}
#posbar .row{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
#posbar h2{margin:0;font-size:14px;font-weight:600}
.tokbox{font-family:var(--mono);background:var(--accent);color:#fff;padding:1px 6px;border-radius:3px;font-size:12px}
.wherenote{font-size:12px;color:var(--text-dim)}
.mcards{display:flex;flex-direction:column;gap:4px}
.mcard{border:1px solid var(--line-soft);border-left:3px solid var(--miss);border-radius:4px;background:var(--surface);padding:4px 8px;font-size:12px;cursor:pointer;text-align:left;width:100%}
.mcard:hover{border-color:var(--accent)}
.mcard .hd{display:flex;gap:8px;align-items:baseline}
.mcard .txt{flex:1}
.mcard .more{display:none;margin-top:4px}
.mcard.open .more{display:block}
.mcard.other{border-left-color:var(--line)}
.notice{font-size:11.5px;color:var(--text-dim);padding:4px 14px;background:var(--surface-2);border-bottom:1px solid var(--line-soft)}
#gridwrap{flex:1;overflow:auto;min-height:0;background:var(--surface)}
table{border-collapse:separate;border-spacing:0;table-layout:fixed}
th,td{border-bottom:1px solid var(--line-soft);border-right:1px solid var(--line-soft);vertical-align:top;text-align:left;padding:0}
thead th{position:sticky;top:0;z-index:3;background:var(--surface);padding:7px 10px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-dim);font-weight:600;border-bottom:1px solid var(--line);width:var(--col)}
thead th.layer{width:58px;left:0;z-index:4}
thead th .rate{display:block;font-family:var(--mono);font-size:10.5px;letter-spacing:0;text-transform:none;color:var(--text-faint);font-weight:400;margin-top:2px}
thead th .own{display:block;font-family:var(--mono);font-size:11px;letter-spacing:0;text-transform:none;color:var(--text);font-weight:400;margin-top:2px;white-space:pre-wrap}
thead th.matched{color:var(--miss)} thead th.unmatched{color:var(--hit)}
tbody th.layer{position:sticky;left:0;z-index:2;background:var(--surface);font-family:var(--mono);font-size:12px;color:var(--text-dim);padding:8px 10px;font-variant-numeric:tabular-nums;cursor:pointer;width:58px}
tbody tr.focus th.layer{background:var(--accent-soft);color:var(--accent-ink);font-weight:600}
tbody tr.focus td{background:color-mix(in srgb, var(--accent-soft) 45%, var(--surface))}
.grip-x{position:absolute;top:0;right:-3px;width:7px;height:100%;cursor:col-resize;z-index:5}
.grip-x:hover{background:var(--accent)}
.grip-y{position:absolute;left:0;bottom:-3px;height:7px;width:100%;cursor:row-resize;z-index:5}
.grip-y:hover{background:var(--accent)}
td .cell{padding:8px 10px;font-family:var(--mono);font-size:11.5px;line-height:1.45;white-space:pre-wrap;word-break:break-word;color:var(--text);max-height:var(--rowmax);overflow:auto}
tbody tr{--rowmax:inherit}
td.hit .cell{background:var(--hit-soft);box-shadow:inset 3px 0 0 var(--hit)}
td .none{padding:8px 10px;color:var(--text-faint);font-style:italic;font-size:11px}
.samp+.samp{border-top:1px dashed var(--line-soft);margin-top:4px;padding-top:4px}
.en{color:var(--text-dim);font-style:italic;font-family:var(--sans);font-size:11.5px;margin-top:2px}
.en::before{content:"EN  ";font-style:normal;font-size:10px;letter-spacing:.06em;color:var(--text-faint)}
mark{background:var(--hit-soft);color:inherit;font-weight:600;border-radius:2px}
mark.find{background:var(--find-soft);color:var(--find-ink);font-weight:700;border-bottom:2px solid var(--find);border-radius:2px}
.legend{display:flex;gap:14px;align-items:center;font-size:11px;color:var(--text-dim);flex-wrap:wrap}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-2px;margin-right:4px;border:1px solid transparent}
/* drawer under the panes */
#below{flex:none;background:var(--surface);border-top:1px solid var(--line);display:flex;flex-direction:column;max-height:42vh;min-height:0}
#below[hidden]{display:none}
#tabs{flex:none;display:flex;gap:4px;padding:6px 14px;border-bottom:1px solid var(--line-soft);align-items:center}
#drawer{overflow:auto;padding:10px 14px 20px;min-height:0}
.mech{border:1px solid var(--line-soft);border-left:3px solid var(--accent);border-radius:4px;background:var(--surface);padding:8px 10px;margin:0 0 8px;font-size:12.5px}
.mech .n{font-family:var(--mono);color:var(--text-faint);font-size:11px;margin-right:6px}
.mech .ev{color:var(--text-dim);margin-top:4px;font-size:12px}
.mech .cells{font-family:var(--mono);font-size:11px;color:var(--text-dim);margin-top:4px;word-break:break-word;white-space:pre-wrap}
.mech .test{font-size:11.5px;color:var(--text-dim);font-style:italic;margin-top:5px;border-top:1px dashed var(--line-soft);padding-top:4px}
.mech .test b{font-style:normal;font-weight:600;letter-spacing:.06em;text-transform:uppercase;font-size:10px;color:var(--hold)}
.step{border:1px solid var(--line-soft);border-radius:4px;background:var(--surface);margin:0 0 6px;padding:6px 10px;font-size:12px}
.step .hd{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-faint)}
.step .hd .nm{font-family:var(--mono);letter-spacing:0;text-transform:none;color:var(--text);background:var(--surface-2);border-radius:3px;padding:0 6px;font-weight:600}
.step .hd a{letter-spacing:0;text-transform:none;margin-left:auto;font-size:11.5px}
.step .think{color:var(--text-dim);white-space:pre-wrap;word-break:break-word;margin-top:4px}
.step .call{font-family:var(--mono);font-size:11px;color:var(--text-dim);margin-top:4px;white-space:pre-wrap;word-break:break-word}
.step .call .usr{color:var(--text);background:var(--accent-soft);border-radius:3px;padding:0 3px}
.step.finish{border-color:var(--accent)}
pre.block{font-family:var(--mono);font-size:11.5px;line-height:1.5;white-space:pre-wrap;word-break:break-word;margin:4px 0 0;background:var(--ground);border:1px solid var(--line-soft);border-radius:4px;padding:6px 8px;color:var(--text)}
details{margin:4px 0} summary{cursor:pointer;color:var(--accent);font-size:11.5px}
.hlgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}
.hlc{background:var(--surface);border:1px solid var(--line-soft);border-left:4px solid var(--miss);border-radius:5px;padding:8px 10px;cursor:pointer;display:flex;flex-direction:column;gap:5px;text-align:left}
.hlc:hover{border-color:var(--accent)}
.hlc.header{border-left-color:var(--hold)} .hlc.user{border-left-color:var(--hit)}
.hlc .where{font-family:var(--mono);font-size:11px;color:var(--text-faint);display:flex;gap:8px;flex-wrap:wrap}
.hlc .where b{color:var(--text-dim);font-weight:600}
.hlc .q{font-family:var(--mono);font-size:11.5px;color:var(--text);line-height:1.45;white-space:pre-wrap;word-break:break-word}
.hlc .loc{font-family:var(--mono);font-size:11px;color:var(--text-dim);white-space:pre-wrap;word-break:break-word}
.hlc .loc b{background:var(--sample-soft);font-weight:500}
.hlc .n{font-size:12px;color:var(--text-dim)}
.hlc .hand{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}
#note{flex:none;padding:5px 14px;font-size:11px;color:var(--text-dim);background:var(--surface);border-top:1px solid var(--line);display:flex;gap:16px;flex-wrap:wrap;align-items:center}
dialog{border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--text);padding:0;max-width:560px;width:92vw}
dialog::backdrop{background:rgba(10,18,24,.45)}
dialog .body{padding:16px;max-height:80vh;overflow:auto}
dialog h2{margin:0 0 10px;font-size:15px}
.man p{margin:0 0 10px;font-size:13px;line-height:1.5}
.man .sub{font-size:12px}
.man .sw{width:13px;height:13px;vertical-align:-2px;margin-right:6px}
.keys{display:grid;grid-template-columns:auto 1fr;gap:6px 12px;font-size:12px;align-items:baseline}
.status{padding:24px;color:var(--text-dim);font-family:var(--mono);font-size:12px}
.theme{border:1px solid var(--line-soft);border-radius:5px;padding:8px 10px;margin:0 0 8px}
.theme .mem{font-size:12px;margin:3px 0} .theme .mem a{font-family:var(--mono);font-size:11px}
@media (max-width:900px){ #ctx{grid-template-columns:1fr 1fr} :root{--col:240px} }
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
<script>
  try{ document.documentElement.setAttribute("data-theme", localStorage.getItem("wc-theme")==="dark" ? "dark" : "light"); }
  catch(e){ document.documentElement.setAttribute("data-theme","light"); }
</script>

<div id="bar">
  <h1><a href="#/">WeirdChat × OLens</a></h1>
  <label class="pick"><span>behavior</span><select id="beh-select"></select></label>
  <label class="pick"><span>pattern</span><select id="pat-select"></select></label>
  <button class="btn" id="prev-item" title="previous pattern (k)">‹</button>
  <button class="btn" id="next-item" title="next pattern (j)">›</button>
  <label class="pick"><span>read</span><select id="read-select"></select></label>
  <span id="cmp-toggles" style="display:flex;gap:4px;flex-wrap:wrap" title="compare columns (1…9)"></span>
  <input id="find" placeholder="find in this pattern's readouts  ( / )" autocomplete="off" spellcheck="false">
  <input id="search" placeholder="search patterns and mechanisms  ( ; )" autocomplete="off" spellcheck="false">
  <span class="spacer"></span>
  <a class="btn" id="themes-btn" href="#/themes" title="mechanism clusters">themes</a>
  <button class="btn" id="agree-btn" title="how the lens arm and a black-box arm compare" hidden>lens vs black-box</button>
  <button class="btn" id="text-toggle" aria-pressed="false" title="read the replies as plain text instead of tokens (x)">text</button>
  <button class="btn" id="ctx-toggle" aria-pressed="true" title="case / hypothesis / summary / rubric (c)">context</button>
  <button class="btn" id="wrap-toggle" aria-pressed="false" title="cap row height (w)">compact rows</button>
  <button class="btn" id="below-toggle" aria-pressed="false" title="mechanisms · transcript · highlights">details</button>
  <button class="btn" id="theme" title="light / dark (t)">◐</button>
  <button class="btn" id="manual-btn" title="what the colours mean (m)">m</button>
  <button class="btn" id="keys" title="keyboard (?)">?</button>
</div>
<div id="primer"><div class="p1" id="primer1"></div><div class="p2" id="banner"></div></div>
<div id="results" hidden></div>
<div id="ctx"></div>
<div id="hits" hidden></div>
<div id="main">
  <section id="text"><div class="status">Loading…</div></section>
  <div id="split" title="drag to resize (double-click resets)"></div>
  <section id="grid"><div id="posbar"></div><div id="gridwrap"></div></section>
</div>
<div id="below" hidden>
  <div id="tabs"></div>
  <div id="drawer"></div>
</div>
<div id="note"></div>

<dialog id="manual"><div class="body">
  <h2>Reading the page</h2>
  <div class="man">
    <h3 style="margin:0 0 6px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--text-faint)">What am I looking at?</h3>
    <p id="manual-primer"></p>
    <p><b>Flagged reply</b> — the judge said this reply SHOWS the behavior. <b>Clean reply</b> — the judge said this reply does NOT show it. Same prompt, same model, same settings; the two are read token by token through the lens so the internals can be compared where the text diverges.</p>
    <h3 style="margin:10px 0 6px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--text-faint)">Colours</h3>
    <p><i class="sw" style="background:var(--accent)"></i><b>Teal</b> is the token in view: the grid on the right shows what the lens decoded at that position, one row per layer, one column per compared read.</p>
    <p><i class="sw" style="background:var(--sample-soft);border-color:var(--sample-line)"></i><b>Yellow</b> is the position of record — the last header token, where the model has read the request and written nothing. It is identical for every reply to a prompt, so a readout difference there is lens sampling, not the model. <span class="kbd">p</span> jumps to it.</p>
    <p><i class="sw" style="background:var(--hit-soft);border-color:var(--hit)"></i><b>Green</b> means a phrase the agent quoted from a readout cell was found verbatim at build time: a green bar on a token says one of that token's cells carries such a phrase; the cell itself gets a green inset bar and the phrase is highlighted. Quotes that did not verify are listed in the build log, not here.</p>
    <p><i class="sw" style="border-bottom:2px dotted var(--text);background:var(--surface)"></i><b>Dotted</b> underline is the fork: the first reply word where the flagged and clean replies diverge, computed from the read tokens (≈, positions were thinned).</p>
    <p><i class="sw" style="background:var(--find-soft);border-color:var(--find)"></i><b>Violet</b> is your search: <span class="kbd">/</span> filters this pattern's readouts (tokens whose cells match get a violet underline and are listed under the bar); <span class="kbd">;</span> searches summaries, prompts and mechanisms across every pattern.</p>
    <p><i class="sw" style="border-top:3px solid var(--hold);background:var(--surface)"></i><b>Amber ⚑</b> is a reader flag: a second model (Gemini) read each cell of the study reads and flagged cells that say something the reply's text does not — a role being adopted, both branches present at once, a hidden referent, a disclaimer that never surfaces, a contradiction, a commitment point. Every quote was re-checked verbatim at build time. Three marks, three sources: <b>yellow</b> = a fixed position (about to speak), <b>green</b> = a phrase the investigator quoted, <b>amber ⚑</b> = a phrase the reader model flagged. The flagger saw the reply and its label — an attention pass, not a blind judge.</p>
    <p id="manual-cal" class="sub"></p>
    <p><b>Faded</b> tokens (‥) stand for positions the read thinned away — the lens read every 4th token plus punctuation and boundaries.</p>
    <p><b>Resizing.</b> Drag the splitter between the panes, the right edge of a column header, or the bottom edge of a layer label; double-click any of them to reset.</p>
    <p class="sub">Reads: <i>diag</i> = the diagnostic pass over the study's flagged / clean replies; <i>agent</i> = the investigator's own readouts, parsed back from its tool pages (w###m/u = the study rollouts it was seeded with — same text, a different lens sample; c### = conversations it created). Highlight cards: rust = inside the reply, amber = about to answer, green = inside the user turn.</p>
  </div>
</div></dialog>

<dialog id="help"><div class="body">
  <h2>Keyboard</h2>
  <div class="keys">
    <span class="kbd">← →</span><span>previous / next token</span>
    <span class="kbd">shift ← →</span><span>jump 10 tokens</span>
    <span class="kbd">home / end</span><span>first / last token</span>
    <span class="kbd">p</span><span>the position of record (about to speak)</span>
    <span class="kbd">↑ ↓</span><span>move the highlighted layer</span>
    <span class="kbd">j k</span><span>next / previous pattern</span>
    <span class="kbd">[ ]</span><span>previous / next behavior</span>
    <span class="kbd">1…9</span><span>show / hide a read as a column</span>
    <span class="kbd">c</span><span>show / hide the context panes</span>
    <span class="kbd">w</span><span>compact rows on / off</span>
    <span class="kbd">x</span><span>replies as plain text / as tokens</span>
    <span class="kbd">f</span><span>next reader-flagged position</span>
    <span class="kbd">/</span><span>find in this pattern's readouts (violet)</span>
    <span class="kbd">;</span><span>search patterns and mechanisms</span>
    <span class="kbd">t</span><span>light / dark</span>
    <span class="kbd">m</span><span>what the colours mean</span>
    <span class="kbd">?</span><span>this list</span>
  </div>
</div></dialog>

<dialog id="agree"><div class="body"><h2>Arms compared</h2><p class="sub" style="margin:0 0 10px">agreement means the two arms told the same story, not that either is right</p><div id="agree-body"></div></div></dialog>
<dialog id="themes"><div class="body"><h2>Recurring hypotheses across patterns</h2><p class="sub" style="margin:0 0 10px">how often each was proposed — not evidence it is right</p><div id="themes-body"></div></div></dialog>

<script>
const D = /*__DATA__*/;
const $ = s => document.querySelector(s);
const esc = s => (s==null?"":String(s)).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const rxEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const num = (v,d) => v==null ? "—" : Number(v).toFixed(d==null?2:d);
const pct = v => v==null ? "—" : Math.round(Number(v)*100) + "%";
const cut = (s,n) => { s = s==null ? "" : String(s); return s.length<=n ? s : s.slice(0,n) + "…"; };
const byKey = {}; (D.patterns||[]).forEach(p => byKey[p.key] = p);
const GLABEL = {run: "agent run with mechanisms", diag: "diagnostics only", data: "data only"};
function el(tag, cls, text){ const e = document.createElement(tag); if (cls) e.className = cls; if (text!=null) e.textContent = text; return e; }
function store(k, v){ try { v==null ? localStorage.removeItem(k) : localStorage.setItem(k, String(v)); } catch(e){} }
function load(k){ try { return localStorage.getItem(k); } catch(e){ return null; } }
// the primer: what WeirdChat is, what is ground truth here and what is not
const PRIMER1 = "WeirdChat (Transluce) sampled the plain Qwen3.6-27B ~64 times per prompt and had a judge label every reply: does it show the behavior or not. That label is the ground truth for WHAT the model does. Nobody has ground truth for WHY — this page collects one investigator's hypotheses about why, read off the model's internals with OLens.";
const VPCT = D.verify && D.verify.fragments ? Math.round(100 * D.verify.verified / D.verify.fragments) : null;
$("#primer1").textContent = PRIMER1; if ($("#manual-primer")) $("#manual-primer").textContent = PRIMER1;
const AGS = D.agreement_summary;
const AGL = D.agreements || [];
function calFor(judge){ const j = (judge||"").toLowerCase(); if (!j) return null; return (D.calibrations||[]).find(c => { const m = (c.model||"").toLowerCase(); return m && (m === j || m.endsWith("/" + j) || j.endsWith("/" + m) || m.split("/").pop() === j.split("/").pop()); }) || null; }
const armLabel0 = a => ({olens: "OLens", jlens: "J-lens", nla: "NLA (L42)", blackbox: "black-box"})[a] || a;
$("#banner").textContent = "Hypotheses, unverified: no intervention was run under the judge's rubric; " + (VPCT==null ? "none of" : VPCT + "% of") + " the lens cells the investigator quoted check out verbatim (build-time figure)" +
  AGL.filter(a => a.summary && a.summary.n_patterns).map(a => `; ${armLabel0(a.arm)} and ${armLabel0(a.arm_b)} agreed on the top hypothesis in ${a.summary.top_match||0} of ${a.summary.n_patterns} patterns`).join("") +
  ((D.predictions||[]).length ? "; predictions right " + D.predictions.map(pr => `${(pr.summary||{}).right==null?"?":pr.summary.right}/${(pr.summary||{}).arms==null?"?":pr.summary.arms} (${armLabel0(pr.arm)})`).join(", ") : "") + ".";
if ((D.calibrations||[]).length){ $("#manual-cal").innerHTML = `<b>Intervention judges.</b> ` + D.calibrations.filter(c => c.n).map(c => `${esc(c.model)} agreed with WeirdChat's judge on ${Math.round((c.agreement||0)*100)}% of ${c.n} labelled replies, κ=${num(c.kappa)}${c.confusion && c.confusion.tp!=null ? ` (tp ${c.confusion.tp} · tn ${c.confusion.tn} · fp ${c.confusion.fp} · fn ${c.confusion.fn})` : ""}${c.judge_failures ? ` · ${c.judge_failures} judge failures` : ""}`).join("; ") + `. The κ shown on a pattern is its own judge's.`; }
if (AGL.length || (D.predictions||[]).length){ $("#agree-btn").hidden = false; $("#agree-btn").textContent = "arms compared";
  const q = v => v==null ? "?" : v;
  $("#agree-body").innerHTML = (AGL.length ? `<table class="sum"><thead><tr><th>pair</th><th>patterns</th><th>top agree</th><th>counterparts</th><th>only one arm</th></tr></thead><tbody>` +
    AGL.map(a => { const s = a.summary || {}, A = armLabel0(a.arm), B = armLabel0(a.arm_b); return `<tr><th>${esc(A)} vs ${esc(B)}</th><td>${q(s.n_patterns)}</td><td>${q(s.top_match)}</td><td>${esc(A)} ${q(s.lens_with_counterpart)}/${q(s.lens_mechanisms)} · ${esc(B)} ${q(s.blackbox_with_counterpart)}/${q(s.blackbox_mechanisms)}</td><td>${esc(A)}-only ${q(s.lens_only_total)} · ${esc(B)}-only ${q(s.blackbox_only_total)}</td></tr>`; }).join("") + `</tbody></table>` +
    AGL.map(a => { const s = a.summary || {}, A = armLabel0(a.arm), B = armLabel0(a.arm_b); return `<p class="sub" style="margin-top:6px">${esc(A)} vs ${esc(B)}: ${q(s.n_patterns)} patterns · top agree ${q(s.top_match)} · ${esc(A)} mechanisms with a counterpart ${q(s.lens_with_counterpart)}/${q(s.lens_mechanisms)} · ${esc(B)} with a counterpart ${q(s.blackbox_with_counterpart)}/${q(s.blackbox_mechanisms)} · ${esc(A)}-only ${q(s.lens_only_total)} · ${esc(B)}-only ${q(s.blackbox_only_total)}${a.model ? ` · read by ${esc(a.model)}` : ""}</p>`; }).join("") : `<p class="sub">no agreement files yet.</p>`) +
    ((D.predictions||[]).length ? `<h3 style="margin:12px 0 6px;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-faint)">prediction scoreboard</h3>` + D.predictions.map(pr => { const s = pr.summary || {}; return `<p class="sub" style="margin:2px 0"><b>${esc(armLabel0(pr.arm))}</b>: right ${q(s.right)} · wrong ${q(s.wrong)} · not predicted ${q(s.not_predicted)} of ${q(s.arms)} arms across ${q(s.n_patterns)} patterns${pr.model ? ` · scored by ${esc(pr.model)}` : ""}</p>`; }).join("") : "");
  $("#agree-btn").onclick = () => $("#agree").showModal(); }
// the two sides of every pattern, in plain words (internal ids keep matched/unmatched)
const SIDE = {matched: "flagged reply", unmatched: "clean reply"};
const SIDE_TIP = {matched: "the judge said this reply SHOWS the behavior", unmatched: "the judge said this reply does NOT show it"};
const side = l => SIDE[l] || l;
// "of 64": the pattern's own sample count when it is the study total, else WeirdChat's ~64
function sampleTotal(){ const p = byKey[S.key]; return p && p.n_samples >= 32 ? `of ${p.n_samples}` : "of ~64"; }
function nameOfId(id){
  const m = String(id).match(/^diag:(\w+):(.*)$/); if (m) return `${side(m[1])} · sample ${m[2]}`;
  const a = String(id).match(/^agent:(?:(\w+):)?([wc]\d+[mu]?)/); if (!a) return String(id);
  const lens = a[1] && a[1] !== "olens" ? ` (${LENS_SHORT[a[1]] || a[1]})` : "";
  const w = a[2].match(/^w\d+([mu])$/); return w ? `${side(w[1]==="m"?"matched":"unmatched")} · ${a[2]} (investigator's own lens sample${lens ? ", " + (LENS_SHORT[a[1]] || a[1]) : ""})` : `investigator probe ${a[2]}${lens}`;
}
function lensTag(r){ const l = r.lens || "olens"; return l === "olens" ? "" : ` (${LENS_SHORT[l] || l})`; }
function readName(r){
  if (r.source === "diag") return `${side(r.label)} · sample ${r.sample_index} ${sampleTotal()}`;
  if (/^w\d+[mu]$/.test(r.conv_id||"")){ const twin = S.data && r.same_rollout_as ? S.data.byId[r.same_rollout_as] : null; return `${side(r.label)} · sample ${twin ? twin.sample_index : r.conv_id} (investigator's own lens sample${r.lens && r.lens !== "olens" ? ", " + (LENS_SHORT[r.lens] || r.lens) : ""})`; }
  return `investigator probe ${r.conv_id}${lensTag(r)}`;
}

// ------------------------------------------------------------------ state
const S = {key:null, data:null, read:null, pos:null, compare:[], layer:null, find:"", query:"", ctx:true, compact:false,
           colw:{}, rowh:{}, below:false, tab:"brief", arm:"lens", textView: load("wc-text") === "1", agentPromise:null, flagCat:"all"};
const CAT_CLASS = {role_adoption: "c1", contradicts_text: "c1", both_branches: "c2", commitment_point: "c2", hidden_referent: "c3", disclaimer_present: "c3", other: "c0"};
const catPill = c => `<span class="cat ${CAT_CLASS[c]||"c0"}">${esc((c||"other").replace(/_/g, " "))}</span>`;
const ARM_ORDER = ["olens", "jlens", "nla", "blackbox"];
const ARM_LABEL = {olens: "OLens", jlens: "J-lens", nla: "NLA (L42)", blackbox: "black-box (no lens)"};
const LENS_SHORT = {olens: "OLens", jlens: "J-lens", nla: "NLA L42", logit: "logit lens"};
const armOf = r => (r && r.arm) || "olens";
const armRank = a => { const i = ARM_ORDER.indexOf(a); return i < 0 ? 99 : i; };
const isLens = r => r && armOf(r) !== "blackbox";
function runsInOrder(runs){ return (runs||[]).slice().sort((a, b) => armRank(armOf(a)) - armRank(armOf(b))); }
function lensRun(runs){ return runsInOrder(runs).find(isLens) || null; }
function bbRun(runs){ return (runs||[]).find(r => armOf(r) === "blackbox") || null; }
function armLabel(a){ return ARM_LABEL[a] || a; }
function topMech(run){ return run ? (run.mechanisms||[]).slice().sort((a,b) => (b.confidence||0) - (a.confidence||0))[0] : null; }
function confBadge(c){ return `<span class="badge ${c>=0.7?"miss":(c>=0.4?"hold":"dim")}">conf ${num(c)}</span>`; }
const heavyCache = {};
function dataUrl(key){ return "data/" + encodeURIComponent(key) + ".json"; }
function patUrl(key, read, pos){ let h = "#/pattern/" + encodeURIComponent(key); if (read) h += "?read=" + encodeURIComponent(read) + (pos==null ? "" : "&pos=" + pos); return h; }
function patternsOf(bid){ return (D.patterns||[]).filter(p => p.behavior_id === bid); }
function firstKey(){ const b = (D.behaviors||[])[0]; if (!b) return null; const p = patternsOf(b.behavior_id)[0] || D.patterns[0]; return p ? p.key : null; }
function curRead(){ return S.data && S.read ? S.data.byId[S.read] : null; }

// ------------------------------------------------------------------- theme
function toggleTheme(){ const dark = document.documentElement.getAttribute("data-theme") !== "dark"; document.documentElement.setAttribute("data-theme", dark ? "dark" : "light"); store("wc-theme", dark ? "dark" : "light"); }

// ---------------------------------------------------------------- splitter
(function splitter(){
  const sp = $("#split"), text = $("#text"), root = document.documentElement;
  const saved = load("wc-split"); if (saved) root.style.setProperty("--split", saved);
  sp.onpointerdown = e => { e.preventDefault(); sp.classList.add("on"); const main = $("#main").getBoundingClientRect();
    const mv = ev => { const pctw = Math.min(85, Math.max(15, (ev.clientX - main.left) / main.width * 100)); root.style.setProperty("--split", pctw.toFixed(1) + "%"); };
    const up = () => { sp.classList.remove("on"); store("wc-split", root.style.getPropertyValue("--split")); window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); };
    window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up); };
  sp.ondblclick = () => { root.style.setProperty("--split", "46%"); store("wc-split", null); };
  if (text) text.style.width = "var(--split)";
})();

// -------------------------------------------------------------------- data
function fetchJson(url){ if (typeof fetch !== "function") return Promise.reject(new Error("no fetch()")); return fetch(url).then(r => { if (!r.ok) throw new Error(r.status + " " + r.statusText); return r.json(); }); }
function indexRead(r){ r.rowByPos = {}; r.positions = []; r.aboutPos = null; for (const row of r.rows || []){ r.rowByPos[row.pos] = row; r.positions.push(row.pos); if (row.region === "header") r.aboutPos = row.pos; } }
function prepareData(key, data){
  data.reads = data.reads || []; data.runs = data.runs || []; data.samples = data.samples || [];
  data.reads.forEach(indexRead);
  data.byId = {}; data.byConv = {}; data.byTool = {}; data.bySample = {};
  data.byConvRun = {};
  for (const r of data.reads){ data.byId[r.id] = r;
    if (r.conv_id){ const usable = r.rows || r.deferred; const cur = data.byConv[r.conv_id];
      if (!cur || ((!cur.rows && !cur.deferred) && usable) || (usable && (r.lens||"olens") === "olens" && (cur.lens||"olens") !== "olens")) data.byConv[r.conv_id] = r; /* the OLens read is the generic target */
      const ri = r.run_index==null ? 0 : r.run_index; data.byConvRun[ri] = data.byConvRun[ri] || {}; if (!data.byConvRun[ri][r.conv_id] || usable) data.byConvRun[ri][r.conv_id] = r; }
    if (r.tool_index!=null) data.byTool[(r.run_index==null?0:r.run_index) + ":" + r.tool_index] = r; /* tool indices are per run */
    if (r.source==="diag" && r.sample_index!=null) data.bySample[r.sample_index] = r; }
  data.convFor = (cid, ri) => (ri!=null && data.byConvRun[ri] && data.byConvRun[ri][cid]) || data.byConv[cid] || null;
  for (const r of data.reads){ r.mention_ids = r.conv_id ? [r.conv_id] : []; for (const o of data.reads) if (o !== r && o.same_rollout_as === r.id && o.conv_id) r.mention_ids.push(o.conv_id); }
  data.mechs = []; data.runs.forEach((run, ri) => (run.mechanisms||[]).forEach((m, mi) => data.mechs.push({...m, run: ri, i: mi, auditor: run.auditor, seed: run.seed, arm: armOf(run)})));
  // verified highlight cells for this pattern: read → pos → [{layer, quote}]
  data.hl = {}; for (const h of (D.highlights||[])) if (h.pattern_key === key){ (data.hl[h.read] = data.hl[h.read] || {}); (data.hl[h.read][h.pos] = data.hl[h.read][h.pos] || []).push(h); }
  // reader flags (build-verified): read id → position → [flags]
  data.fl = {}; for (const [rid, e] of Object.entries(data.flags || {})) for (const f of (e.flags||[])){ (data.fl[rid] = data.fl[rid] || {}); (data.fl[rid][f.position] = data.fl[rid][f.position] || []).push(f); }
  data.key = key;
}
function ensureAgentRows(read){
  const data = S.data;
  if (!read.deferred || read.rows) return Promise.resolve();
  if (!data.agentPromise) data.agentPromise = fetchJson(data.agent_file).then(blob => { for (const full of (blob.reads||[])){ const stub = data.byId[full.id]; if (stub){ stub.rows = full.rows; stub.layers = full.layers || stub.layers; stub.deferred = false; indexRead(stub); } } });
  return data.agentPromise;
}

// ------------------------------------------------------------------ opening
function openPattern(key, want){
  if (!byKey[key]){ $("#text").innerHTML = `<div class="status">no pattern ${esc(key)}</div>`; return; }
  if (S.key === key && S.data){ if (want && want.read && S.data.byId[want.read]) selectRead(want.read, want.pos); return; }
  S.key = key; S.data = null; S.read = null; S.pos = null; S.compare = []; S.find = ""; $("#find").value = "";
  renderBar(); renderCtx(); renderHits(); renderNote();
  $("#text").innerHTML = `<div class="status">loading ${esc(dataUrl(key))} …</div>`; $("#posbar").innerHTML = ""; $("#gridwrap").innerHTML = "";
  const go = data => { if (S.key !== key) return; if (!data.byId) prepareData(key, data); S.data = data;
    const dm = data.reads.find(r => r.source==="diag" && r.label==="matched"), du = data.reads.find(r => r.source==="diag" && r.label==="unmatched");
    if (dm && du) S.compare = [dm.id, du.id];
    if (!data.reads.length){ renderAll(); $("#text").innerHTML = `<div class="status">no lens reads for this pattern yet (no diag file, no agent readouts)</div>`; return; }
    const start = (want && want.read && data.byId[want.read]) ? data.byId[want.read] : (dm || data.reads.find(r => r.rows && r.rows.length) || data.reads[0]);
    selectRead(start.id, want && want.pos!=null ? +want.pos : null); };
  if (D.heavy && D.heavy[key]) return go(D.heavy[key]);
  if (heavyCache[key]) return go(heavyCache[key]);
  fetchJson(dataUrl(key)).then(d => { heavyCache[key] = d; return d; }, err => { if (S.key === key) $("#text").innerHTML = `<div class="status">could not load ${esc(dataUrl(key))} (${esc(err && err.message || err)}). Browsers block fetch() from file:// — serve the site dir with <span class="kbd">python -m http.server</span>, or build with single=true.</div>`; return null; })
    .then(d => { if (d) go(d); });
}
function defaultPos(r){ if (!r.positions || !r.positions.length) return null; if (r.fork_pos!=null && r.rowByPos[r.fork_pos]) return r.fork_pos; if (r.aboutPos!=null) return r.aboutPos; return r.positions[0]; }
function selectRead(id, pos){
  const data = S.data; if (!data) return; const r = data.byId[id]; if (!r) return;
  S.read = id; if (!S.compare.includes(id)) S.compare = [id, ...S.compare];
  const needs = [r, ...S.compare.map(c => data.byId[c]).filter(Boolean)].filter(x => x.deferred && !x.rows);
  if (needs.length){ S.pos = pos; renderBar(); $("#text").innerHTML = `<div class="status">loading the agent's reads (${esc(data.agent_file)}) …</div>`; const key = S.key;
    Promise.all(needs.map(ensureAgentRows)).then(() => { if (S.key === key && S.read === id) selectRead(id, S.pos); })
      .catch(err => { if (S.key === key) $("#text").innerHTML = `<div class="status">could not load ${esc(data.agent_file)} (${esc(err && err.message || err)})</div>`; }); return; }
  if (pos==null || !r.rowByPos[pos]) pos = defaultPos(r);
  S.pos = pos;
  if (S.layer==null || !(r.layers||[]).includes(S.layer)) S.layer = (r.layers||[])[0];
  renderAll();
  try { history.replaceState(null, "", patUrl(S.key, id, pos)); } catch(e){}
}
function toggleCompare(id){ if (!S.data || id === S.read) return; S.compare = S.compare.includes(id) ? S.compare.filter(x => x !== id) : [...S.compare, id]; const r = S.data.byId[id]; if (r && r.deferred && !r.rows) return selectRead(S.read, S.pos); renderBar(); renderGrid(); renderText(); renderHits(); }
function renderAll(){ renderBar(); renderCtx(); renderText(); renderGrid(); renderHits(); renderBelow(); renderNote(); }

// --------------------------------------------------------------------- bar
function readDot(r){ return r.parse_error ? "err" : (r.source==="diag" ? r.label : (r.label==="agent" ? "agent" : r.label)); }
function readLabel(r){ return readName(r) + (r.parse_error ? " ⚠ unparsed" : ""); }
function renderBar(){
  const p = byKey[S.key], bs = D.behaviors || [], bid = p ? p.behavior_id : (bs[0] ? bs[0].behavior_id : null);
  $("#beh-select").innerHTML = bs.map(b => `<option value="${esc(b.behavior_id)}" ${b.behavior_id===bid?"selected":""}>${esc(b.behavior_name)} (${b.n_patterns})</option>`).join("");
  const pats = patternsOf(bid);
  $("#pat-select").innerHTML = pats.map(q => `<option value="${esc(q.key)}" ${q.key===S.key?"selected":""}>${esc(cut(q.group_summary || q.key, 60))} · ${pct(q.published_match_rate)} flagged</option>`).join("");
  const idx = pats.findIndex(q => q.key === S.key); $("#prev-item").disabled = idx <= 0; $("#next-item").disabled = idx < 0 || idx >= pats.length-1;
  const reads = S.data ? S.data.reads : [];
  const groups = [["study (flagged / clean)", r => r.source === "diag"], ["OLens probes", r => r.source !== "diag" && (r.lens||"olens") === "olens"], ["J-lens probes", r => r.lens === "jlens"], ["NLA probes", r => r.lens === "nla"], ["other probes", r => r.source !== "diag" && !["olens","jlens","nla"].includes(r.lens||"olens")]];
  $("#read-select").innerHTML = reads.length ? groups.map(([g, f]) => { const rs = reads.filter(f); return rs.length ? `<optgroup label="${esc(g)}">` + rs.map(r => `<option value="${esc(r.id)}" ${r.id===S.read?"selected":""}>${esc(readLabel(r))}</option>`).join("") + `</optgroup>` : ""; }).join("") : `<option>—</option>`;
  $("#cmp-toggles").innerHTML = reads.filter(r => !r.parse_error).map((r, i) => `<button class="btn cmp" data-cmp="${esc(r.id)}" aria-pressed="${S.compare.includes(r.id)}" ${r.id===S.read?"disabled":""} title="${esc((SIDE_TIP[r.label] ? SIDE_TIP[r.label] + " — " : "") + readName(r) + " as a grid column (" + (i+1) + ")")}"><i class="${readDot(r)}"></i>${esc(readName(r))}</button>`).join("");
  $("#cmp-toggles").querySelectorAll("[data-cmp]").forEach(b => b.onclick = () => toggleCompare(b.dataset.cmp));
  $("#ctx-toggle").setAttribute("aria-pressed", String(S.ctx)); $("#text-toggle").setAttribute("aria-pressed", String(S.textView)); $("#wrap-toggle").setAttribute("aria-pressed", String(S.compact)); $("#below-toggle").setAttribute("aria-pressed", String(S.below));
}

// --------------------------------------------------------------------- ctx
function clipbox(text, n){ const t = text || ""; if (t.length <= n) return `<div class="mono">${esc(t)}</div>`; return `<div class="clipbox"><div class="mono">${esc(t)}</div></div><button class="showall" data-showall>show all (${t.length} chars)</button>`; }
function matchLine(rubric){
  // the judge's one-line criterion: the rubric line naming match=true, else its first sentence
  const t = rubric || ""; if (!t) return "";
  const line = t.split(/\n+/).find(l => /match\s*=\s*true/i.test(l));
  const clean = x => x.replace(/\*\*/g, "").replace(/^\s*match\s*=\s*true\s*(if|when|:)?\s*/i, "").trim();
  if (line) return cut(clean(line), 260);
  const first = t.split(/(?<=\.)\s+/)[0] || t; return cut(first.trim(), 260);
}
function verifiedBadge(p){ if (!p || !p.cited_fragments) return `<span class="badge dim">no quoted cells</span>`; const k = p.cited_verified, n = p.cited_fragments; return `<span class="badge ${k===n?"hit":(k?"hold":"miss")}">cited cells verified ${k}/${n}</span>`; }
function renderCtx(){
  const box = $("#ctx"); box.hidden = !S.ctx; const p = byKey[S.key]; if (!p){ box.innerHTML = ""; return; }
  const data = S.data, runs = runsInOrder(p.runs), run = lensRun(runs) || runs[0] || null, bb = bbRun(runs);
  const mechs = run ? (run.mechanisms||[]) : [];
  const top = topMech(run), others = runs.filter(r => r !== run);
  const ags = p.agreements || [], ivm = p.interventions_meta, cal = ivm ? (calFor(ivm.judge_model) || D.calibration) : D.calibration;
  const nrep = 64;  // WeirdChat samples ~64 replies per prompt; the published rate is over those
  let h = `<div class="pane"><h3>the case</h3><div><b>Behavior:</b> ${esc(p.behavior_name)}${matchLine(p.rubric) ? ` <span class="dim">— ${esc(matchLine(p.rubric))}</span>` : ""}</div>` +
    `<div style="margin-top:3px"><b>What WeirdChat found:</b> on this prompt, ${pct(p.published_match_rate)} of ${nrep} replies were judged to show it. Same prompt, same model, same settings — it went both ways.</div>` +
    `<div style="margin-top:3px"><b>What you see here:</b> one <span title="${esc(SIDE_TIP.matched)}">flagged</span> and one <span title="${esc(SIDE_TIP.unmatched)}">clean</span> reply, read token by token through the lens (layers 20–60), plus the investigator's probes.</div>` +
    ags.map(ag => `<div style="margin-top:3px"><b>${esc(armLabel(ag.arm))} vs ${esc(armLabel(ag.arm_b))}:</b> top hypotheses agree: ${ag.top_match==null?"?":(ag.top_match?"yes":"no")} · ${esc(armLabel(ag.arm))} mechanisms with a ${esc(armLabel(ag.arm_b))} counterpart ${ag.lens_with_counterpart==null?"?":ag.lens_with_counterpart}/${ag.n_lens==null?"?":ag.n_lens} · ${esc(armLabel(ag.arm_b))} with a ${esc(armLabel(ag.arm))} counterpart ${ag.blackbox_with_counterpart==null?"?":ag.blackbox_with_counterpart}/${ag.n_blackbox==null?"?":ag.n_blackbox}</div>`).join("") +
    (p.flags_meta ? `<div style="margin-top:3px"><b>Reader flags:</b> ${p.flags_meta.matched} cells on the flagged reply, ${p.flags_meta.unmatched} on the clean reply (of ${p.flags_meta.proposed} proposed, ${p.flags_meta.dropped} dropped as non-verbatim${p.flags_meta.model ? `; reader ${esc(p.flags_meta.model)}` : ""})</div>` : "") +
    (ivm ? `<div style="margin-top:3px"><b>Interventions run:</b> ${ivm.n_per_arm==null?"?":ivm.n_per_arm} samples/arm over ${ivm.n_arms} arms, judged by ${esc(ivm.judge_model||"?")}${cal && cal.kappa!=null ? ` (κ=${num(cal.kappa)} vs the study's labels)` : ""} — see the interventions tab</div>` : "") +
    `<div class="tags" style="margin-top:5px"><span class="tag">${pct(p.published_match_rate)} flagged</span><span class="tag">elo ${num(p.elo,0)}</span><span class="tag">${p.n_reads||0} lens reads</span>${p.weirdchat_url?`<a class="tag" href="${esc(p.weirdchat_url)}" target="_blank" rel="noopener">WeirdChat ↗</a>`:""}</div>` +
    `<div class="prompt">${esc(p.prompt)}</div></div>`;
  h += `<div class="pane"><h3>detective-joracle's hypothesis (unverified)</h3>` + (top ? `<div><span class="badge dim">${esc(armLabel(armOf(run)))}</span> ${esc(top.mechanism)} <span class="badge ${top.confidence>=0.7?"miss":(top.confidence>=0.4?"hold":"dim")}">confidence ${num(top.confidence)}</span></div>` +
    (top.would_test_by ? `<div class="dim" style="margin-top:4px">How it would be tested: ${esc(top.would_test_by)} <span class="badge hold">not run</span></div>` : `<div class="dim" style="margin-top:4px"><span class="badge hold">not run</span> no test proposed</div>`) +
    (mechs.length > 1 ? `<div style="margin-top:4px"><a href="#" data-more>+ ${mechs.length-1} more in details ▸</a></div>` : "") : `<div class="empty">no run for this pattern yet — no hypothesis.</div>`) +
    others.map((o, i) => { const t = topMech(o); return `<div class="dim" style="margin-top:${i?4:6}px;${i?"":"border-top:1px dashed var(--line-soft);padding-top:4px"}"><b>${esc(armLabel(armOf(o)))}:</b> ${t ? esc(cut(t.mechanism, 200)) + " · " + confBadge(t.confidence) : "run present, no mechanisms reported"}</div>`; }).join("") + `</div>`;
  h += `<div class="pane"><h3>agent summary</h3>` + (run ? `<div>${esc(run.summary || "(no summary)")}</div>` +
    `<div class="vrow" style="margin-top:6px"><span class="name">tool calls</span><span>${run.n_tool_calls||0} · ${run.n_readouts||0} readouts · ${run.n_chat||0} chat probes</span></div>` +
    `<div class="vrow"><span class="name">mechanisms</span><span>${mechs.length}</span></div>` +
    `<div class="vrow"><span class="name">verification</span>${verifiedBadge(p)}</div>` +
    `<div class="vrow"><span class="name">${esc(armLabel(armOf(run)))} run</span><span class="dim">${esc(run.auditor)} s${run.seed} · stopped: ${esc(run.stopped_by||"?")}</span></div>` +
    others.map(o => `<div class="vrow"><span class="name">${esc(armLabel(armOf(o)))} run</span><span>${o.n_tool_calls||0} calls${isLens(o)?` · ${o.n_readouts||0} readouts`:""} · ${o.n_chat||0} chat probes · ${(o.mechanisms||[]).length} mechanisms</span></div>`).join("") : `<div class="empty">no agent run for this pattern yet.</div>`) + `</div>`;
  h += `<div class="pane"><h3>rubric</h3>${p.rubric ? clipbox(p.rubric, 320) : `<div class="empty">no rubric.</div>`}</div>`;
  box.innerHTML = h;
  box.querySelectorAll("[data-showall]").forEach(b => b.onclick = () => { b.previousElementSibling.classList.add("open"); b.remove(); });
  box.querySelectorAll("[data-more]").forEach(a => a.onclick = e => { e.preventDefault(); S.below = true; S.tab = "mechanisms"; renderBar(); renderBelow(); });
}

// -------------------------------------------------------------------- text
const REGION_LABEL = {user: "the model is reading the request", header: "about to answer — identical for every reply to this prompt", reply: "the model is writing its answer", system: ""};
function replyLabel(r){ if (r.source !== "diag") return "the model's reply (investigator's probe, unjudged)"; return `${side(r.label)} — judge: ${r.label==="matched" ? "shows the behavior" : "does not"}`; }
function tokHtml(s){ if (s === "") return `<span class="nl">∅</span>`; return esc(s).replace(/\n/g, `<span class="nl">⏎</span>`).replace(/\t/g, `<span class="nl">⇥</span>`); }
function findRx(){ return S.find ? new RegExp(rxEsc(S.find), "i") : null; }
function cellsAt(r, pos){ const row = r.rowByPos && r.rowByPos[pos]; return row ? (row.samples||[]).reduce((a, ss) => a.concat(ss), []) : []; }
function foundPositions(){
  const rx = findRx(), set = new Set(); if (!rx || !S.data) return set;
  const reads = S.compare.map(id => S.data.byId[id]).filter(r => r && r.rows);
  const r0 = curRead(); for (const p of (r0.positions||[])) for (const r of reads) if (cellsAt(r, p).some(s => rx.test(s))){ set.add(p); break; }
  return set;
}
const LEGEND = `<div class="tlegend"><span><i class="sw" style="background:var(--accent)"></i>token in view</span>` +
  `<span title="about to speak: the last token before the model writes — identical for every reply"><i class="sw" style="background:var(--sample-soft);border-color:var(--sample-line)"></i>yellow = about to speak: the last token before the model writes — identical for every reply</span>` +
  `<span><i class="sw" style="border-top:3px solid var(--hit);background:var(--surface)"></i>green top bar = a cell here contains a phrase the investigator quoted (verified)</span>` +
  `<span><i class="sw" style="box-shadow:inset 0 -3px 0 var(--find);background:var(--surface)"></i>violet = search hit</span>` +
  `<span><i class="sw" style="border-bottom:2px dotted var(--text);background:var(--surface)"></i>dotted = ≈ where the flagged and clean replies diverge</span>` +
  `<span><i class="sw" style="border-top:3px solid var(--hold);background:var(--surface)"></i>⚑ flagged by the reader model (Gemini): a cell here says something the text does not</span>` +
  `<span><i class="sw" style="opacity:.4;background:var(--text-faint)"></i>faded ‥ = position not read</span></div>`;
function flagsOf(r){ return (S.data && S.data.fl[r.id]) || {}; }
function flagList(r){ const fl = flagsOf(r); return Object.keys(fl).map(Number).sort((a, b) => a - b).map(p => ({pos: p, flags: fl[p]})).filter(x => S.flagCat === "all" || x.flags.some(f => f.category === S.flagCat)); }
function flagBar(r){
  const all = flagsOf(r), positions = Object.keys(all); if (!positions.length) return "";
  const cats = [...new Set(Object.values(all).reduce((a, fs) => a.concat(fs.map(f => f.category)), []))].sort();
  const list = flagList(r), n = Object.values(all).reduce((a, fs) => a + fs.length, 0);
  return `<div id="flagbar"><span>⚑ ${n} reader flag${n===1?"":"s"} at ${positions.length} position${positions.length===1?"":"s"}</span><select id="flagcat" title="filter by category"><option value="all" ${S.flagCat==="all"?"selected":""}>all categories</option>` + cats.map(c => `<option value="${esc(c)}" ${S.flagCat===c?"selected":""}>${esc(c.replace(/_/g, " "))}</option>`).join("") + `</select>` +
    list.slice(0, 60).map(x => `<button class="fchip ${x.pos===S.pos?"cur":""}" data-fpos="${x.pos}" title="${esc(x.flags.map(f => f.category + ": " + f.why).join("\n"))}">pos ${x.pos} · ${esc([...new Set(x.flags.map(f => f.category.replace(/_/g, " ")))].join(", "))}</button>`).join("") + (list.length > 60 ? `<span>+${list.length-60} more</span>` : "") + `<span class="kbd">f</span><span>next flag</span></div>`;
}
function proseBlock(label, text, cls, tip){ return `<div class="blk ${cls||""}"><div class="role"><span${tip?` title="${esc(tip)}"`:""}>${esc(label)}</span></div><div class="prose">${esc(text||"(empty)")}</div></div>`; }
function renderProse(r, box){
  // the same read as readable text: prompt, then the reply, labelled exactly as the strip labels it
  const data = S.data, p = byKey[S.key];
  const user = (r.messages||[]).filter(m => m.role === "user").map(m => m.content).join("\n\n") || p.prompt;
  let h = `<div class="tlegend"><span>text view — the replies as prose; press <span class="kbd">x</span> for the clickable tokens</span></div>`;
  h += proseBlock("user prompt", user, "user");
  h += proseBlock(replyLabel(r), r.completion + (r.text_source === "tool page" && r.completion && r.completion.length >= 900 ? "\n\n[clipped to 900 chars on the investigator's page]" : ""), "reply", SIDE_TIP[r.label]);
  const dm = data.reads.find(x => x.source==="diag" && x.label==="matched"), du = data.reads.find(x => x.source==="diag" && x.label==="unmatched");
  if (dm && du && data.fork && data.fork.prefix_chars != null){
    const n = data.fork.prefix_chars, cutAt = (t) => [t.slice(0, n), t.slice(n)];
    const [pm, rm] = cutAt(dm.completion||""), [pu, ru] = cutAt(du.completion||"");
    h += `<div class="caption">same prompt, same settings — these two replies diverge here${n ? ` (after ${n} shared characters)` : " (from the first word)"}</div>` +
      `<div class="cmp2"><div class="col"><h4 title="${esc(SIDE_TIP.matched)}">flagged reply · sample ${dm.sample_index}</h4><div class="prose">${esc(pm)}<span class="div flagged">${esc(rm)}</span></div></div>` +
      `<div class="col"><h4 title="${esc(SIDE_TIP.unmatched)}">clean reply · sample ${du.sample_index}</h4><div class="prose">${esc(pu)}<span class="div clean">${esc(ru)}</span></div></div></div>`;
  }
  box.innerHTML = h;
}
function renderText(){
  const r = curRead(), box = $("#text"); if (!r){ return; }
  if (S.textView) return renderProse(r, box);
  const hl = S.data.hl[r.id] || {}, found = foundPositions();
  let h = LEGEND + flagBar(r);
  if (r.parse_error) h += `<div class="notice">this readout page did not parse: ${esc(r.parse_error)}</div>`;
  const rows = r.rows || []; let i = 0;
  if (!rows.length) h += `<div class="status">no positions in this read.</div>`;
  while (i < rows.length){
    const region = rows[i].region; let inner = ""; let prev = null;
    for (; i < rows.length && rows[i].region === region; i++){
      const row = rows[i];
      if (prev!=null && row.pos - prev > 1) inner += `<span class="tok nodata" title="${row.pos-prev-1} positions not read">‥</span>`;
      prev = row.pos;
      const fls = flagsOf(r)[row.pos];
      const cls = ["tok", row.pos===S.pos?"cur":"", row.pos===r.aboutPos?"read":"", hl[row.pos]?"hit":"", fls?"flagged":"", found.has(row.pos)?"found":"", row.pos===r.fork_pos?"mark":""].join(" ");
      inner += `<span class="${cls}" data-pos="${row.pos}" title="position ${row.pos} · ${esc(row.kind)}${hl[row.pos]?" · a quoted phrase verifies here":""}${fls?" · ⚑ " + esc(fls.map(f => f.category.replace(/_/g, " ")).join(", ")):""}${row.pos===r.aboutPos?" · about to speak":""}${row.pos===r.fork_pos?" · fork":""}">${tokHtml(row.tok)}</span>`;
    }
    const lbl2 = region === "reply" ? replyLabel(r) : (REGION_LABEL[region]||"");
    const tip = region === "reply" && r.source === "diag" ? ` title="${esc(SIDE_TIP[r.label]||"")}"` : "";
    h += `<div class="blk ${esc(region)}"><div class="role"><span${tip}>${esc(region==="header"?"about to answer":region)}</span><span class="lbl2"${tip}>${esc(lbl2)}</span></div><div class="toks">${inner}</div></div>`;
  }
  // the other rollouts of the contrast set, compact, clickable when a read exists for them
  const others = (S.data.samples||[]).filter(s => !(r.source==="diag" && s.sample_index===r.sample_index));
  if (others.length){
    h += `<div class="rolls"><h3>other replies in the contrast set (${others.length}) — click one to read it</h3>` + others.map(s => { const rd = S.data.bySample[s.sample_index];
      return `<button class="roll ${s.matched?"matched":"unmatched"}" ${rd?`data-read="${esc(rd.id)}"`:"disabled"} title="${rd?"open this reply's read":"no lens read of this reply"}"><span class="lab" title="${esc(SIDE_TIP[s.matched?"matched":"unmatched"])}">${s.matched?"flagged reply":"clean reply"} · sample ${s.sample_index}${rd?"":" · no read"}</span><span class="snip">${esc(cut((s.text||"").replace(/\s+/g," "), 160))}</span></button>`; }).join("") + `</div>`;
  }
  box.innerHTML = h;
  box.querySelectorAll(".tok[data-pos]").forEach(t => t.onclick = () => selectRead(S.read, +t.dataset.pos));
  box.querySelectorAll("[data-fpos]").forEach(b => b.onclick = () => selectRead(S.read, +b.dataset.fpos));
  const sel = box.querySelector("#flagcat"); if (sel) sel.onchange = e => { S.flagCat = e.target.value; renderText(); };
  box.querySelectorAll(".roll[data-read]").forEach(b => b.onclick = () => selectRead(b.dataset.read, null));
  const cur = box.querySelector(".tok.cur"); if (cur && cur.scrollIntoView) cur.scrollIntoView({block:"nearest"});
}

// -------------------------------------------------------------------- grid
function citations(text, ri){ const out = []; let cid = null; const re = /\b([wc]\d+[mu]?)\b|\bpos\s*~?\s*(\d+)/g; let m; text = text || "";
  while ((m = re.exec(text))){ if (m[1]){ const r = S.data.convFor(m[1], ri); if (r) cid = m[1]; out.push({cid: r ? m[1] : null, pos: null, index: m.index, len: m[0].length}); } else out.push({cid, pos: +m[2], index: m.index, len: m[0].length}); }
  return out; }
function mechsAt(r, pos){ const out = [];
  for (const m of S.data.mechs){ if (r.source !== "diag" && r.run_index != null && m.run !== r.run_index) continue; /* another arm's citation names its own read of this conversation */
    const here = citations(m.readout_cells, m.run).filter(c => c.pos === pos && c.cid && r.mention_ids.includes(c.cid)); if (here.length) out.push({m, via: here[0].cid}); } return out; }
function whereText(r, row){
  if (row.region === "header") return row.pos === r.aboutPos ? "about to speak: the model has read the request and written nothing — identical for every reply to this prompt" : "the chat boundary before the reply — identical for every reply to this prompt";
  if (row.region === "user") return "inside the user turn: the model is reading the request and has written nothing";
  if (row.region === "reply") return (row.pos === r.fork_pos ? "on the fork: " : "inside the reply: ") + "the model is writing its answer" + (row.pos === r.fork_pos ? " — the first word where the flagged and clean replies diverge (≈)" : "");
  return row.region;
}
function markCell(text, quotes, qrx, fquotes){
  let html = esc(text);
  for (const q of quotes){ const e = esc(q); if (html.includes(e)) html = html.split(e).join(`<mark>${e}</mark>`); }
  for (const q of (fquotes||[])){ const e = esc(q); if (html.includes(e) && !html.includes(`<mark class="flag">${e}`)) html = html.split(e).join(`<mark class="flag">${e}</mark>`); }
  if (qrx) html = html.replace(new RegExp("(" + rxEsc(esc(S.find)) + ")(?![^<]*>)", "gi"), `<mark class="find">$1</mark>`);
  return html;
}
function renderGrid(){
  const r = curRead(), posbar = $("#posbar"), wrap = $("#gridwrap"); if (!r) return;
  const p = S.pos, row = r.rowByPos[p];
  if (!row){ posbar.innerHTML = `<div class="status">${r.parse_error ? "nothing to show — this page did not parse." : "no position selected."}</div>`; wrap.innerHTML = ""; return; }
  const fl = mechsAt(r, p);
  let h = `<div class="row"><h2>position ${p}</h2><span class="tokbox">${esc(JSON.stringify(row.tok))}</span><span class="wherenote">${esc(whereText(r, row))} · ${esc(row.kind)}</span></div>`;
  const here = flagsOf(r)[p] || [];
  if (here.length){ h += `<div class="flags"><h3>⚑ reader flags at this position (${here.length}) — verbatim in the cell, seen with the label</h3><div class="mcards">` + here.map(f => { const smp = cellsAt(r, p).find(s => s.includes(f.quote)); const en = smp && D.en && D.en[smp]; return `<div class="fcard">${catPill(f.category)}<span class="L">L${f.layer}</span><div class="q">“${esc(f.quote)}”</div>${en ? `<div class="en">${esc(en)}</div>` : ""}<div class="why">${esc(f.why)}</div></div>`; }).join("") + `</div></div>`; }
  if (fl.length) h += `<div class="mcards">` + fl.map((f, i) => `<button class="mcard ${f.via===r.conv_id?"":"other"}" data-mc="${i}"><div class="hd"><span class="txt">${esc(cut(f.m.mechanism, 160))}</span><span class="badge ${f.m.confidence>=0.7?"miss":(f.m.confidence>=0.4?"hold":"dim")}">conf ${num(f.m.confidence)}</span></div><div class="more"><div>${esc(f.m.mechanism)}</div><div class="quote">${esc(f.m.readout_cells)}</div>${f.via!==r.conv_id?`<div class="dim">cited on ${esc(f.via)} — same reply, different lens sample</div>`:""}</div></button>`).join("") + `</div>`;
  else h += `<div class="dim" style="font-size:11.5px">no reported mechanism cites ${esc(r.mention_ids.join(" / ") || r.id)} at pos ${p}.</div>`;
  posbar.innerHTML = h;
  posbar.querySelectorAll("[data-mc]").forEach(b => b.onclick = () => b.classList.toggle("open"));
  // the table
  wrap.innerHTML = "";
  document.documentElement.style.setProperty("--rowmax", S.compact ? "150px" : "none");
  const cols = S.compare.map(id => S.data.byId[id]).filter(c => c && c.rows);
  const layers = [...new Set(cols.reduce((a, c) => a.concat(c.layers||[]), []))].sort((a,b) => a-b);
  const toks = cols.map(c => c.rowByPos[p] ? c.rowByPos[p].tok : null), present = toks.filter(t => t!=null);
  const identical = cols.length > 1 && present.length === cols.length && present.every(t => t === present[0]);
  if (identical) wrap.appendChild(el("div", "notice", "identical prefix — same activation" + (new Set(cols.map(c => c.lens||"olens")).size > 1 ? ", read through different lenses" : ", different lens samples") + (row.region==="reply" ? " (the replies still agree at this position)" : "")));
  const colw = Math.max(260, Math.floor((wrap.clientWidth - 58) / Math.max(1, cols.length)) - 1); wrap.style.setProperty("--col", colw + "px");
  const table = el("table"), thead = el("thead"), hr = el("tr"); hr.appendChild(el("th", "layer", "layer"));
  const lensesDiffer = new Set(cols.map(c => c.lens || "olens")).size > 1;
  cols.forEach((c, j) => { const th = el("th", readDot(c), readName(c)); th.title = (SIDE_TIP[c.label] ? SIDE_TIP[c.label] + " · " : "") + c.id;
    if (lensesDiffer) th.appendChild(el("span", "rate", (LENS_SHORT[c.lens||"olens"] || c.lens) + (c.source==="diag" ? " · study read" : "")));
    if (!identical && cols.length > 1) th.appendChild(el("span", "own", toks[j]==null ? "not read at pos " + p : JSON.stringify(toks[j])));
    if (S.colw[c.id]) th.style.width = S.colw[c.id] + "px";
    const grip = el("div", "grip-x"); grip.title = "drag to resize this column (double-click resets)";
    grip.onpointerdown = e => { e.preventDefault(); const x0 = e.clientX, w0 = th.getBoundingClientRect().width; const mv = ev => { S.colw[c.id] = Math.max(140, w0 + ev.clientX - x0); th.style.width = S.colw[c.id] + "px"; }; const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); }; window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up); };
    grip.ondblclick = () => { delete S.colw[c.id]; renderGrid(); }; th.appendChild(grip); hr.appendChild(th); });
  thead.appendChild(hr); table.appendChild(thead);
  const tbody = el("tbody"), qrx = findRx();
  for (const L of layers){
    const tr = el("tr"); if (L === S.layer) tr.className = "focus";
    const th = el("th", "layer", `L${L}`); th.onclick = () => { S.layer = L; renderGrid(); };
    if (S.rowh[L]) tr.style.setProperty("--rowmax", S.rowh[L] + "px");
    const grip = el("div", "grip-y"); grip.title = "drag to resize this layer's row (double-click resets)";
    grip.onpointerdown = e => { e.preventDefault(); e.stopPropagation(); const y0 = e.clientY, h0 = tr.getBoundingClientRect().height; const mv = ev => { S.rowh[L] = Math.max(40, h0 + ev.clientY - y0); tr.style.setProperty("--rowmax", S.rowh[L] + "px"); }; const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); }; window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up); };
    grip.ondblclick = e => { e.stopPropagation(); delete S.rowh[L]; renderGrid(); }; th.appendChild(grip); tr.appendChild(th);
    for (const c of cols){
      const td = el("td"), li = (c.layers||[]).indexOf(L), crow = c.rowByPos[p];
      const samples = (crow && li >= 0 && crow.samples) ? (crow.samples[li] || []) : null;
      if (samples === null) td.appendChild(el("div", "none", crow ? "—" : "not read at this position"));
      else if (!samples.length) td.appendChild(el("div", "none", "—"));
      else if ((c.lens||"olens") === "jlens"){ /* a J-lens cell is a bag of tokens */
        const quotes = ((S.data.hl[c.id]||{})[p]||[]).filter(x => x.layer === L).map(x => x.quote); if (quotes.length) td.className = "hit";
        const d = el("div", "bag"); d.innerHTML = samples.map(s => { const hit = quotes.some(q => s.includes(q) || q.includes(s)), f = qrx && qrx.test(s); return `<span class="${hit?"m":""}${f?" f":""}" title="J-lens token">${esc(s)}</span>`; }).join(""); td.appendChild(d); }
      else { const quotes = ((S.data.hl[c.id]||{})[p]||[]).filter(x => x.layer === L).map(x => x.quote); if (quotes.length) td.className = "hit";
        const fq = ((S.data.fl[c.id]||{})[p]||[]).filter(f => f.layer === L).map(f => f.quote); if (fq.length) td.className = (td.className ? td.className + " " : "") + "flagged";
        const d = el("div", "cell"); d.innerHTML = samples.map(s => { const en = D.en && D.en[s]; return `<div class="samp">${markCell(s.trim(), quotes, qrx, fq)}${en ? `<div class="en">${esc(en)}</div>` : ""}</div>`; }).join(""); td.appendChild(d); }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody); wrap.appendChild(table);
}

// ------------------------------------------------------------------- search
function renderHits(){
  const box = $("#hits"), r = curRead(); if (!r || !S.find){ box.hidden = true; box.innerHTML = ""; return; }
  const found = [...foundPositions()].sort((a,b) => a-b);
  box.hidden = false;
  box.innerHTML = `<span class="lbl">${found.length} token${found.length===1?"":"s"} with “${esc(S.find)}” in their cells</span>` + found.slice(0, 80).map(p => `<button class="hitchip" data-pos="${p}" title="position ${p}">${p} ${esc(JSON.stringify(r.rowByPos[p].tok))}</button>`).join("") + (found.length > 80 ? `<span class="dim">+${found.length-80} more</span>` : "");
  box.querySelectorAll("[data-pos]").forEach(b => b.onclick = () => selectRead(S.read, +b.dataset.pos));
}
function firstCell(text){ let cid = null; const re = /\b([wc]\d+[mu]?)\b|\bpos\s*~?\s*(\d+)/g; let m; while ((m = re.exec(text||""))){ if (m[1]) cid = m[1]; else if (cid) return {read: "agent:" + cid, pos: +m[2]}; } return null; }
function renderResults(){
  const box = $("#results"); const q = S.query.trim(); if (!q){ box.hidden = true; box.innerHTML = ""; return; }
  const rx = new RegExp(rxEsc(q), "i"), rows = [];
  const snip = t => { const i = t.search(rx); return cut(t.slice(Math.max(0, i - 40)).replace(/\s+/g, " "), 160); };
  for (const p of D.patterns || []){
    if (rx.test(p.group_summary||"")) rows.push({p, where: "summary", snip: snip(p.group_summary)});
    if (rx.test(p.prompt||"")) rows.push({p, where: "prompt", snip: snip(p.prompt)});
    (p.flag_whys||[]).forEach(f => { if (rx.test(f.why||"")) rows.push({p, where: `⚑ flag · pos ${f.position} · ${f.category.replace(/_/g, " ")}`, snip: snip(f.why), cell: {read: f.read, pos: f.position}}); });
    (p.intervention_notes||[]).forEach(a => { if (rx.test(a.note||"") || rx.test(a.name||"")) rows.push({p, where: `intervention arm ${a.name}`, snip: snip(a.name + ": " + a.note)}); });
    (p.runs||[]).forEach((run, ri) => { if (rx.test(run.summary||"")) rows.push({p, where: `${run.arm==="blackbox"?"black-box":"lens"} run summary`, snip: snip(run.summary)});
      (run.mechanisms||[]).forEach((m, mi) => { const t = [m.mechanism, m.evidence, m.readout_cells].join(" · "); if (rx.test(t)) rows.push({p, where: `mechanism ${mi+1}`, snip: snip(t), cell: firstCell(m.readout_cells)}); }); });
  }
  box.hidden = false;
  box.innerHTML = `<h4>${rows.length} hit${rows.length===1?"":"s"} for “${esc(q)}”</h4>` + rows.slice(0, 60).map((x, i) => `<button class="res" data-i="${i}"><span class="fam">${esc(x.p.behavior_name)}</span><span class="where">${esc(cut(x.p.group_summary||x.p.key, 34))} · ${esc(x.where)}</span><span class="snip">${esc(x.snip)}</span></button>`).join("") + (rows.length > 60 ? `<div class="dim">+${rows.length-60} more</div>` : "");
  box.querySelectorAll("[data-i]").forEach(b => b.onclick = () => { const x = rows[+b.dataset.i]; location.hash = x.cell ? patUrl(x.p.key, x.cell.read, x.cell.pos) : patUrl(x.p.key); });
}

// ------------------------------------------------------------------- drawer
function linkCells(text, data, ri){ let out = "", last = 0, cid = null; const re = /\b([wc]\d+[mu]?)\b|\bpos\s*~?\s*(\d+)/g; let m; text = text || "";
  while ((m = re.exec(text))){ out += esc(text.slice(last, m.index));
    if (m[1]){ const r = data.convFor(m[1], ri); if (r) cid = m[1]; out += r ? `<a href="${patUrl(data.key, r.id, null)}">${esc(m[0])}</a>` : esc(m[0]); }
    else { const r = cid ? data.convFor(cid, ri) : null, pos = +m[2]; const ok = r && (r.deferred || (r.rowByPos && r.rowByPos[pos])); out += ok ? `<a href="${patUrl(data.key, r.id, pos)}" title="open ${esc(r.id)} at position ${pos}">${esc(m[0])}</a>` : esc(m[0]); }
    last = m.index + m[0].length; }
  return out + esc(text.slice(last)); }
function mechBlock(m, i, data, ri){ return `<div class="mech"><div><span class="n">${i+1}.</span>${esc(m.mechanism)}${m.confidence==null?"":` <span class="badge ${m.confidence>=0.7?"miss":(m.confidence>=0.4?"hold":"dim")}">confidence ${num(m.confidence)}</span>`}</div>` + (m.evidence ? `<div class="ev">${esc(m.evidence)}</div>` : "") + (m.readout_cells ? `<div class="cells">cells: ${linkCells(m.readout_cells, data, ri)}</div>` : "") + (m.would_test_by ? `<div class="test"><b>would test by — not run in this pass:</b> ${esc(m.would_test_by)}</div>` : "") + `</div>`; }
function showAll(text, n){ const t = text || ""; if (t.length <= n) return `<div class="body">${esc(t)}</div>`; return `<div class="body">${esc(t.slice(0, n))} …</div><details><summary>show all (${t.length} chars)</summary><div class="body">${esc(t)}</div></details>`; }
function parseReplies(out){
  // the tool's _record format: "[c005] organism reply (N tokens)[ [truncated]]:\n<text>" blocks separated by blank lines
  const re = /^\[(c\d+)\] (\S+) (reply|sampled USER turn|continuation) \((\d+) tokens\)( \[truncated\])?:\n/gm; const heads = []; let m;
  while ((m = re.exec(out||""))) heads.push({cid: m[1], model: m[2], kind: m[3], tokens: m[4], trunc: !!m[5], start: m.index, end: m.index + m[0].length});
  if (!heads.length) return null;
  return heads.map((h, i) => ({...h, text: (out||"").slice(h.end, i+1 < heads.length ? heads[i+1].start : undefined).replace(/\n+$/, "")}));
}
function stepBlock(s, i, data, run){ const rd = s.tool_index!=null ? data.byTool[((run && run.run_index!=null) ? run.run_index : 0) + ":" + s.tool_index] : null;
  let h = `<div class="step ${esc(s.name||"")}"><div class="hd"><span class="nm">${esc(s.name||"assistant")}</span><span>#${i+1}${s.meta?" · "+esc(s.meta):""}</span>${rd ? `<a href="${patUrl(data.key, rd.id, null)}" title="${esc(rd.id)}">open this read → ${esc(readName(rd))}${rd.parse_error?" (unparsed)":""}</a>` : ""}</div>`;
  if (s.think) h += `<div class="think">${esc(s.think)}</div>`;
  if (s.name === "chat"){
    h += `<div class="qa"><div class="lab">investigator asked:</div>${showAll(s.user, 600)}`;
    if (s.system) h += `<div class="lab">with system prompt:</div>${showAll(s.system, 600)}`;
    if (s.prefill) h += `<div class="lab">with the reply prefilled:</div>${showAll(s.prefill, 600)}`;
    const reps = parseReplies(s.output);
    if (reps) reps.forEach(r => { h += `<div class="lab">model answered: <b>${esc(r.cid)} · ${esc(r.model)} · ${r.tokens} tokens${r.trunc?" · truncated":""}</b></div>${showAll(r.text, 600)}`; });
    else if (s.output) h += `<div class="lab">model answered:</div>${showAll(s.output, 600)}`;
    h += `</div>`;
  } else if (s.name === "note"){ h += `<div class="qa"><div class="lab">note to self:</div>${showAll(s.note || s.args, 600)}</div>`; }
  else if (s.name === "finish"){ h += `<div class="call">finished with ${(run && run.mechanisms ? run.mechanisms.length : 0)} mechanisms — see the mechanisms tab</div>`; }
  else { if (s.name) h += `<div class="call">${esc(s.name)}(${esc(s.args)})</div>`;
    if (s.output && s.name !== "readouts"){ const o = s.output; h += `<pre class="block">${esc(o.slice(0,500))}${o.length>500?" …":""}</pre>`; if (o.length > 500) h += `<details><summary>show full result (${o.length} chars)</summary><pre class="block">${esc(o)}</pre></details>`; }
    else if (s.output) h += `<details><summary>raw readout page (${s.output.length} chars, clipped; the structured read is in the strip)</summary><pre class="block">${esc(s.output)}</pre></details>`; }
  return h + `</div>`; }
function agreeTab(ag){
  const A = armLabel(ag.arm || "olens"), B = armLabel(ag.arm_b || "blackbox");
  const list = (items, label) => items.length ? items.map(m => `<div class="mech"><div><span class="badge dim">${esc(label)}</span> ${esc(m.mechanism)}${m.confidence==null?"":" "+confBadge(m.confidence)}</div></div>`).join("") : `<div class="empty">none</div>`;
  return `<div class="sub" style="margin:0 0 8px"><b>${esc(A)} vs ${esc(B)}</b> — top hypotheses agree: <b>${ag.top_match==null?"?":(ag.top_match?"yes":"no")}</b> · ${esc(A)} mechanisms with a ${esc(B)} counterpart ${ag.lens_with_counterpart==null?"?":ag.lens_with_counterpart}/${ag.n_lens==null?"?":ag.n_lens} · ${esc(B)} with a ${esc(A)} counterpart ${ag.blackbox_with_counterpart==null?"?":ag.blackbox_with_counterpart}/${ag.n_blackbox==null?"?":ag.n_blackbox} — agreement means the two arms told the same story, not that either is right</div>` +
    `<h3 style="margin:8px 0 4px;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-faint)">only ${esc(A)} proposed (${ag.lens_only.length})</h3>` + (ag.lens_only_summary ? `<div class="sub" style="margin:0 0 6px">${esc(ag.lens_only_summary)}</div>` : "") + list(ag.lens_only, `only ${A} proposed`) +
    `<h3 style="margin:12px 0 4px;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-faint)">only ${esc(B)} proposed (${ag.blackbox_only.length})</h3>` + (ag.blackbox_only_summary ? `<div class="sub" style="margin:0 0 6px">${esc(ag.blackbox_only_summary)}</div>` : "") + list(ag.blackbox_only, `only ${B} proposed`);
}
function scoreboard(){
  const preds = D.predictions || []; if (!preds.length) return "";
  return `<div class="score"><span class="sub" style="border:0;background:none;padding:0">prediction scoreboard:</span>` + preds.slice().sort((a, b) => armRank(a.arm) - armRank(b.arm)).map(pr => { const s = pr.summary || {}; return `<span><b>${esc(armLabel(pr.arm))}</b> right ${s.right==null?"?":s.right} · wrong ${s.wrong==null?"?":s.wrong} · not predicted ${s.not_predicted==null?"?":s.not_predicted} of ${s.arms==null?"?":s.arms} arms across ${s.n_patterns==null?"?":s.n_patterns} patterns</span>`; }).join("") + `</div>`;
}
const DIR = {down: "↓", up: "↑", none: "–", flat: "–"};
function predCell(row){ if (!row || !row.predicted || row.verdict === "not_predicted") return `<span class="pred np">not predicted</span>`; const v = row.verdict === "right" ? "right" : (row.verdict === "wrong" ? "wrong" : "np"); return `<span class="pred ${v}" title="${esc((row.mechanism||"") + (row.direction ? " · predicted " + row.direction : ""))}">${esc(DIR[(row.direction||"").toLowerCase()] || row.direction || "–")} ${v === "np" ? "" : v}</span>`; }
function diffSpan(base, other){
  // the changed span of `other` against `base`: common prefix/suffix stripped, a little context kept
  if (other === base) return null;
  let a = 0; while (a < base.length && a < other.length && base[a] === other[a]) a++;
  let b = 0; while (b < base.length - a && b < other.length - a && base[base.length-1-b] === other[other.length-1-b]) b++;
  const ctx = 30, pre = other.slice(Math.max(0, a - ctx), a), post = other.slice(other.length - b, other.length - b + ctx);
  return {pre: (a > ctx ? "…" : "") + pre, removed: base.slice(a, base.length - b), added: other.slice(a, other.length - b), post: post + (other.length - b + ctx < other.length ? "…" : "")};
}
function interventionsTab(iv, preds){
  const arms = iv.arms || [], base = arms[0];
  if (!arms.length) return `<div class="empty">no arms recorded.</div>`;
  const parms = Object.keys(preds||{}).sort((a, b) => armRank(a) - armRank(b)); /* investigator arms with a predictions entry for this pattern */
  let h = scoreboard();
  if (parms.length) h += `<div class="sub" style="margin:0 0 6px">this pattern: ` + parms.map(a => { const e = preds[a]; return `<b>${esc(armLabel(a))}</b> right ${e.right==null?"?":e.right} · wrong ${e.wrong==null?"?":e.wrong} · not predicted ${e.not_predicted==null?"?":e.not_predicted}`; }).join(" · ") + `</div>`;
  h += `<div class="sub" style="margin:0 0 6px">${iv.n_per_arm==null?"?":iv.n_per_arm} samples per arm, judged by ${esc(iv.judge_model||"?")} under the study's rubric; the first row is the unchanged baseline. Green Δ = flagged rate went DOWN with p&lt;0.05, rust = UP with p&lt;0.05.${parms.length ? " Prediction columns: the arm's mechanisms predicted this direction beforehand — green right, rust wrong, faint when it made no prediction." : ""}</div>`;
  h += `<table class="iv"><thead><tr><th>arm</th><th>what changed</th><th>tests</th><th>flagged rate · 95% CI</th><th>Δ vs baseline</th><th>Fisher p</th>` + parms.map(a => `<th>${esc(armLabel(a))} predicted</th>`).join("") + `</tr></thead><tbody>`;
  arms.forEach((a, i) => {
    const isBase = i === 0, d = isBase || !base ? null : diffSpan(base.prompt||"", a.prompt||"");
    let chg = isBase ? `<span class="dim">unchanged prompt</span>` : (d ? `<div class="chg">${esc(d.pre)}<del>${esc(d.removed)}</del><ins>${esc(d.added)}</ins>${esc(d.post)}</div>` : `<span class="dim">same prompt</span>`);
    if (a.system && (!base || a.system !== base.system)) chg += `<div class="chg">system: ${esc(cut(a.system, 200))}</div>`;
    if (a.prefill && (!base || a.prefill !== base.prefill)) chg += `<div class="chg">prefill: ${esc(cut(a.prefill, 200))}</div>`;
    const ci = a.ci95, lo = ci ? Math.max(0, Math.min(1, ci[0])) : null, hi = ci ? Math.max(0, Math.min(1, ci[1])) : null;
    const bar = ci ? `<div class="cibar" title="95% CI ${pct(lo)}–${pct(hi)}"><i style="left:${(lo*100).toFixed(1)}%;width:${((hi-lo)*100).toFixed(1)}%"></i>${a.rate!=null?`<b style="left:${(Math.max(0,Math.min(1,a.rate))*100).toFixed(1)}%"></b>`:""}</div>` : "";
    const sig = a.fisher_p_vs_baseline!=null && a.fisher_p_vs_baseline < 0.05, cls = !isBase && sig && a.delta_vs_baseline!=null ? (a.delta_vs_baseline < 0 ? "down" : (a.delta_vs_baseline > 0 ? "up" : "")) : "";
    h += `<tr class="${isBase?"base":""}"><td><b>${esc(a.name)}</b>${isBase?' <span class="badge hold">baseline</span>':""}${a.judge_failures?`<div class="dim" style="font-size:11px">${a.judge_failures} judge failure${a.judge_failures===1?"":"s"}</div>`:""}</td><td>${chg}</td><td class="dim">${esc(a.note||"")}</td>` +
      `<td><span class="mono">${a.k==null?"?":a.k}/${a.n==null?"?":a.n} = ${pct(a.rate)}</span>${bar}</td><td class="${cls}">${isBase?"—":(a.delta_vs_baseline==null?"—":(a.delta_vs_baseline>0?"+":"")+(a.delta_vs_baseline*100).toFixed(1)+" pp")}</td><td class="mono">${isBase?"—":(a.fisher_p_vs_baseline==null?"—":a.fisher_p_vs_baseline<0.001?"<0.001":a.fisher_p_vs_baseline.toFixed(3))}</td>` +
      parms.map(pa => `<td>${isBase ? "—" : predCell((preds[pa].arms||[]).find(x => x.arm === a.name))}</td>`).join("") + `</tr>`;
  });
  h += `</tbody></table>`;
  arms.forEach((a, i) => {
    const n = Math.min(3, (a.replies||[]).length);
    h += `<button class="btn" data-ex="${i}" style="margin:4px 4px 4px 0">${esc(a.name)}: ${n} example repl${n===1?"y":"ies"} ▸</button>`;
  });
  arms.forEach((a, i) => {
    h += `<div id="ex-${i}" hidden>` + (a.replies||[]).slice(0, 3).map((r, j) => { const v = (a.verdicts||[])[j]; return `<div class="ex"><span class="badge ${v===true?"miss":(v===false?"hit":"dim")}" title="${esc(v===true?SIDE_TIP.matched:(v===false?SIDE_TIP.unmatched:"the judge failed on this reply"))}">${v===true?"flagged":(v===false?"clean":"no verdict")}</span> <span class="dim">${esc(a.name)} · reply ${j+1}</span><div class="rep">${esc(cut(r||"(empty)", 900))}</div>${(a.explanations||[])[j]?`<div class="why">judge: ${esc((a.explanations||[])[j])}</div>`:""}</div>`; }).join("") + `</div>`;
  });
  return h;
}
function briefTab(data){
  const b = data.brief;
  if (!b || !b.text) return `<div class="empty">the brief could not be rebuilt — ${esc(b && b.error ? b.error : (D.brief_how || "prompts unavailable"))}</div>`;
  let h = `<div class="sub" style="margin:0 0 6px">what the investigator was handed: the behavior, the judge's rubric, the prompt, and ${D.n_side||2} flagged + ${D.n_side||2} clean replies, pre-loaded as conversations ${esc((b.ids||[]).join("/"))}</div>`;
  if (D.system_prompt) h += `<details><summary>system prompt (${D.system_prompt.length} chars)</summary><pre class="brief">${esc(D.system_prompt)}</pre></details>`;
  h += `<pre class="brief">${esc(b.text)}</pre><div class="sub" style="margin-top:6px">rebuilt at build time from the pattern with the repo's prompts module (${esc(D.brief_how||"")})</div>`;
  return h;
}
function renderBelow(){
  const box = $("#below"); box.hidden = !S.below; if (!S.below) return;
  const data = S.data, p = byKey[S.key];
  const tabs = [["brief", "brief"], ["mechanisms", "ranked mechanisms"], ["transcript", "agent transcript"]];
  if (p && (p.agreements||[]).length) tabs.push(["agree", "arms compared"]);
  if (data && data.interventions) tabs.push(["interventions", "interventions"]);
  tabs.push(["highlights", "highlights"]);
  if (!tabs.some(t => t[0] === S.tab)) S.tab = "brief";
  $("#tabs").innerHTML = tabs.map(([k, l]) => `<button class="btn" data-tab="${k}" aria-pressed="${S.tab===k}">${l}</button>`).join("") + `<span class="sub" style="margin-left:8px">unverified hypotheses — nothing here was tested${data && data.interventions ? ", except where the interventions tab says so" : ""}</span>`;
  $("#tabs").querySelectorAll("[data-tab]").forEach(b => b.onclick = () => { S.tab = b.dataset.tab; renderBelow(); });
  // the transcript / mechanisms tabs follow one arm; a switch appears when both arms ran
  const allRuns = data ? runsInOrder(data.runs) : [], arms = [...new Set(allRuns.map(armOf))].sort((a, b) => armRank(a) - armRank(b));
  if (!arms.includes(S.arm)) S.arm = arms[0] || "olens";
  const armRuns = allRuns.filter(r => armOf(r) === S.arm);
  const armSwitch = arms.length > 1 ? `<div class="armsw"><span class="sub">arm:</span>` + arms.map(a => `<button class="btn" data-arm="${esc(a)}" aria-pressed="${S.arm===a}">${esc(armLabel(a))}</button>`).join("") + `<span class="sub">— same brief for every arm; the black-box arm had chat tools only</span></div>` : "";
  let h = "";
  if (!data) h = `<div class="status">no pattern loaded.</div>`;
  else if (S.tab === "brief") h = briefTab(data);
  else if (S.tab === "agree") h = (p.agreements||[]).map(agreeTab).join(`<hr style="border:0;border-top:1px solid var(--line-soft);margin:12px 0">`);
  else if (S.tab === "interventions") h = interventionsTab(data.interventions, data.predictions || {});
  else if (S.tab === "mechanisms"){ h = armSwitch; if (!armRuns.length) h += `<div class="empty">no ${esc(armLabel(S.arm))} run for this pattern yet.</div>`;
    armRuns.forEach((run, ri) => { h += `<div class="sub" style="margin:${ri?"12px":"0"} 0 6px"><b>${esc(run.auditor)}</b> seed ${run.seed} · ${esc(run.arm||"")} · ${esc(run.summary||"")}</div>`; const ms = (run.mechanisms||[]).slice().sort((a,b) => (b.confidence||0)-(a.confidence||0)); if (!ms.length) h += `<div class="empty">no mechanisms reported.</div>`; ms.forEach((m, i) => { h += mechBlock(m, i, data, run.run_index); }); }); }
  else if (S.tab === "transcript"){ h = armSwitch; if (!armRuns.length) h += `<div class="empty">no ${esc(armLabel(S.arm))} run for this pattern yet.</div>`;
    armRuns.forEach(run => { h += `<div class="sub" style="margin:0 0 6px"><b>${esc(run.auditor)}</b> seed ${run.seed} · ${(run.steps||[]).length} steps · stopped: ${esc(run.stopped_by||"?")}</div>`; (run.steps||[]).forEach((s, i) => { h += stepBlock(s, i, data, run); }); if ((run.notes||[]).length){ h += `<div class="sub" style="margin:8px 0 4px">scratch notes</div>`; run.notes.forEach(n => { h += `<pre class="block">${esc(n)}</pre>`; }); } }); }
  else { const items = D.highlights || []; const mine = items.filter(x => x.pattern_key === S.key), rest = items.filter(x => x.pattern_key !== S.key);
    h = `<div class="sub" style="margin:0 0 6px">${mine.length} on this pattern · ${rest.length} elsewhere — lens cells a mechanism quotes that verify verbatim at build time; click to jump</div><div class="hlgrid">` + [...mine, ...rest].map(x => { const idx = items.indexOf(x);
      const smp = esc(x.sample.trim()).split(esc(x.quote)).join(`<mark>${esc(x.quote)}</mark>`), loc = esc(x.local.slice(0, x.local.length - x.token.length)) + `<b>${esc(x.token)}</b>`;
      return `<button class="hlc ${esc(x.region)}" data-hl="${idx}"><div class="where"><b>${esc(x.behavior)}</b> ${esc(cut(x.summary, 50))} · <span title="${esc(SIDE_TIP[(x.read.match(/^diag:(\w+):/)||[])[1]]||"")}">${esc(nameOfId(x.read))}</span> · pos ${x.pos} · L${x.layer}${x.kind==="hand"?' · <span class="hand">hand-picked</span>':""}</div><div class="loc">${loc}</div><div class="q">${smp}</div><div class="n">${esc(x.note)}</div></button>`; }).join("") + `</div>`; }
  $("#drawer").innerHTML = h;
  $("#drawer").querySelectorAll("[data-hl]").forEach(b => b.onclick = () => { const x = (D.highlights||[])[+b.dataset.hl]; location.hash = patUrl(x.pattern_key, x.read, x.pos); });
  $("#drawer").querySelectorAll("[data-arm]").forEach(b => b.onclick = () => { S.arm = b.dataset.arm; renderBelow(); });
  $("#drawer").querySelectorAll("[data-ex]").forEach(b => b.onclick = () => { const box = document.getElementById("ex-" + b.dataset.ex); if (box) box.hidden = !box.hidden; });
}
function renderNote(){
  const r = curRead(), p = byKey[S.key];
  $("#note").innerHTML = (p ? `<span class="mono" style="font-size:11px">${esc(p.key)}</span>` : "") + (r ? `<span title="${esc(r.id)}">${esc(readName(r))} · ${(r.positions||[]).length} positions read${r.claimed && r.claimed.positions!=null && r.claimed.positions!==(r.positions||[]).length ? ` (page header says ${r.claimed.positions})` : ""} · ${(r.layers||[]).length} layers · positions were thinned (every 4th token + punctuation + boundaries; ‥ marks gaps)</span>` : "") +
    `<span class="spacer"></span><span><span class="kbd">← →</span> token · <span class="kbd">p</span> about to speak · <span class="kbd">/</span> find · <span class="kbd">?</span> keys</span>`;
}

// ------------------------------------------------------------------- themes
function renderThemes(focus){
  const cl = D.clusters || [];
  $("#themes-body").innerHTML = cl.length ? cl.map((k, i) => `<div class="theme" id="theme-${i}" style="${i===focus?"border-color:var(--accent)":""}"><div style="font-weight:600">${esc(k.name)}</div><div class="dim">${k.members.length} pattern${k.members.length===1?"":"s"} · ${k.behavior_ids.map(esc).join(", ")}</div>${k.description?`<div style="margin:4px 0">${esc(k.description)}</div>`:""}` +
    k.members.map(m => { const p = byKey[m.pattern_key]; return `<div class="mem"><a href="${patUrl(m.pattern_key)}">${esc(p ? p.behavior_name + " · " + cut(p.group_summary, 40) : m.pattern_key)}</a>${p?"":' <span class="badge dim">not in this build</span>'} — ${esc(m.mechanism)}</div>`; }).join("") + `</div>`).join("")
    : `<div class="empty">no synth.json yet — clusters appear once the synthesis pass runs.</div>`;
  const d = $("#themes"); if (d && !d.open && d.showModal) d.showModal();
  if (focus!=null){ const t = document.getElementById("theme-" + focus); if (t && t.scrollIntoView) t.scrollIntoView({block:"start"}); }
}

// ------------------------------------------------------------------- route
function parseHash(){ const raw = (location.hash || "#/").replace(/^#\/?/, ""); const [pathPart, query] = raw.split("?"); const parts = pathPart.split("/"); const q = {};
  (query||"").split("&").forEach(kv => { if (!kv) return; const [k, v] = kv.split("="); q[decodeURIComponent(k)] = v==null ? "" : decodeURIComponent(v); }); return {parts, q}; }
function route(){
  const {parts, q} = parseHash();
  if (parts[0] === "pattern" && parts.length > 1){ const d = $("#themes"); if (d && d.open) d.close(); openPattern(decodeURIComponent(parts.slice(1).join("/")), {read: q.read || null, pos: q.pos!=null && q.pos!=="" ? +q.pos : null}); return; }
  if (parts[0] === "themes" || parts[0] === "cluster"){ if (!S.key){ const k = firstKey(); if (k) openPattern(k, {}); } renderThemes(parts[0] === "cluster" ? parseInt(parts[1], 10) : null); return; }
  const key = firstKey();
  if (!key){ $("#text").innerHTML = `<div class="status">no patterns found under ${esc(D.source)}.</div>`; renderBar(); return; }
  openPattern(key, {read: null, pos: null});
}

// -------------------------------------------------------------------- wiring
function stepPattern(d){ const p = byKey[S.key]; if (!p) return; const pats = patternsOf(p.behavior_id); const i = pats.findIndex(x => x.key === S.key) + d; if (i >= 0 && i < pats.length) location.hash = patUrl(pats[i].key); }
function stepBehavior(d){ const p = byKey[S.key], bs = D.behaviors || []; const i = bs.findIndex(b => b.behavior_id === (p ? p.behavior_id : null)) + d; if (i >= 0 && i < bs.length){ const q = patternsOf(bs[i].behavior_id)[0]; if (q) location.hash = patUrl(q.key); } }
function stepPos(d){ const r = curRead(); if (!r || !r.positions.length) return; const i = Math.max(0, Math.min(r.positions.length-1, r.positions.indexOf(S.pos) + d)); selectRead(r.id, r.positions[i]); }
function stepLayer(d){ const r = curRead(); if (!r) return; const ls = r.layers || []; const i = Math.max(0, Math.min(ls.length-1, ls.indexOf(S.layer) + d)); S.layer = ls[i]; renderGrid(); }
$("#beh-select").onchange = e => { const q = patternsOf(e.target.value)[0]; if (q) location.hash = patUrl(q.key); e.target.blur(); };
$("#pat-select").onchange = e => { location.hash = patUrl(e.target.value); e.target.blur(); };
$("#read-select").onchange = e => { selectRead(e.target.value, null); e.target.blur(); };
$("#prev-item").onclick = () => stepPattern(-1); $("#next-item").onclick = () => stepPattern(1);
$("#keys").onclick = () => $("#help").showModal(); $("#manual-btn").onclick = () => $("#manual").showModal();
$("#ctx-toggle").onclick = () => { S.ctx = !S.ctx; renderBar(); renderCtx(); };
$("#text-toggle").onclick = () => { S.textView = !S.textView; store("wc-text", S.textView ? "1" : "0"); renderBar(); renderText(); };
$("#wrap-toggle").onclick = () => { S.compact = !S.compact; renderBar(); renderGrid(); };
$("#below-toggle").onclick = () => { S.below = !S.below; renderBar(); renderBelow(); };
$("#theme").onclick = toggleTheme;
$("#search").addEventListener("input", e => { S.query = e.target.value; renderResults(); });
$("#find").addEventListener("input", e => { S.find = e.target.value.trim(); renderText(); renderGrid(); renderHits(); });
for (const id of ["search", "find"]) $("#" + id).addEventListener("keydown", e => { if (e.key === "Escape"){ e.target.value = ""; if (id === "find"){ S.find = ""; renderText(); renderGrid(); renderHits(); } else { S.query = ""; renderResults(); } e.target.blur(); } });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA"){ if (e.key === "Escape") e.target.blur(); return; }
  const k = e.key;
  if (k === "ArrowLeft"){ stepPos(e.shiftKey ? -10 : -1); e.preventDefault(); }
  else if (k === "ArrowRight"){ stepPos(e.shiftKey ? 10 : 1); e.preventDefault(); }
  else if (k === "Home"){ const r = curRead(); if (r && r.positions.length) selectRead(r.id, r.positions[0]); e.preventDefault(); }
  else if (k === "End"){ const r = curRead(); if (r && r.positions.length) selectRead(r.id, r.positions[r.positions.length-1]); e.preventDefault(); }
  else if (k === "ArrowUp"){ stepLayer(-1); e.preventDefault(); }
  else if (k === "ArrowDown"){ stepLayer(1); e.preventDefault(); }
  else if (k === "j") stepPattern(1); else if (k === "k") stepPattern(-1);
  else if (k === "]") stepBehavior(1); else if (k === "[") stepBehavior(-1);
  else if (k === "p"){ const r = curRead(); if (r && r.aboutPos!=null) selectRead(r.id, r.aboutPos); }
  else if (k === "f"){ const r = curRead(); if (r){ const list = flagList(r).map(x => x.pos); if (list.length){ const i = list.findIndex(p => p > S.pos); selectRead(r.id, list[i < 0 ? 0 : i]); } } }
  else if (k === "c") $("#ctx-toggle").click(); else if (k === "w") $("#wrap-toggle").click(); else if (k === "x") $("#text-toggle").click();
  else if (k === "/"){ $("#find").focus(); e.preventDefault(); }
  else if (k === ";"){ $("#search").focus(); e.preventDefault(); }
  else if (k === "?") $("#help").showModal(); else if (k === "m") $("#manual").showModal(); else if (k === "t") toggleTheme();
  else if (/^[1-9]$/.test(k) && S.data){ const r = S.data.reads.filter(x => !x.parse_error)[+k-1]; if (r) toggleCompare(r.id); }
});
for (const d of document.querySelectorAll("dialog")) d.addEventListener("click", e => { if (e.target === d) d.close(); });
window.addEventListener("resize", () => { if (S.data) renderGrid(); });
window.addEventListener("hashchange", route);
route();
</script>
"""


if __name__ == "__main__":
    main()
