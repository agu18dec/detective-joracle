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
            parts = raw.split(" | ") if (k > 1 and raw) else ([raw] if raw else [])
            r["samples"].append([clip(p, CELL_CAP) for p in parts if p])
    rows.sort(key=lambda r: r["pos"])
    out["layers"] = layers
    out["rows"] = rows


def agent_read(
    tool_index: int, logged: dict[str, Any], used_ids: set[str], diag_reads: list[dict[str, Any]]
) -> dict[str, Any]:
    """One `readouts` tool call as a read; borrows the full text from a matching diag read."""
    args = as_dict(logged.get("args"))
    parsed = parse_readout_page(as_text(logged.get("output")))
    conv_id = as_text(args.get("conversation")) or parsed["conv_id"] or f"call{tool_index}"
    rid = f"agent:{conv_id}"
    if rid in used_ids:
        rid = f"agent:{conv_id}#{tool_index}"
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
        "lens": "olens",
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
        for ti, logged in enumerate(tool_log):
            if as_text(logged.get("name")) != "readouts":
                continue
            read = agent_read(ti, logged, used_ids, diag_reads)
            read["run_index"] = run["run_index"]
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


def synth_meta(out_root: Path) -> dict[str, Any]:
    synth = read_json(out_root / "synth.json") or {}
    return {"model": as_text(synth.get("model")), "n_records": as_int(synth.get("n_records"))}


# ------------------------------------------------------------------- split
HEAVY_PATTERN_KEYS = ("samples", "reads", "fork")
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


def reads_by_conv(pat: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """conv id → the read that actually parsed (a failed retry of the same id never wins)."""
    out: dict[str, dict[str, Any]] = {}
    for r in pat["reads"]:
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
        by_conv = reads_by_conv(pat)
        for run in pat["runs"]:
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

    cards, hl_stats = auto_highlights(patterns)
    hand = hand_highlights(patterns, highlights) if highlights else []
    cards = hand + cards
    for pat in patterns:  # per-pattern "cited cells verified k/n", shown in the agent-summary pane
        k, n = hl_stats["per_pattern"].get(pat["key"], (0, 0))
        pat["cited_verified"], pat["cited_fragments"] = k, n
    samples = all_samples(patterns)
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
.tok.cur{background:var(--accent)!important;color:#fff;border-color:var(--accent)}
.tok.nodata{opacity:.4;cursor:default;background:transparent}
.tok .nl{font-size:10px;color:var(--text-faint)} .tok.cur .nl{color:#fff}
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
    <span class="kbd">/</span><span>find in this pattern's readouts (violet)</span>
    <span class="kbd">;</span><span>search patterns and mechanisms</span>
    <span class="kbd">t</span><span>light / dark</span>
    <span class="kbd">m</span><span>what the colours mean</span>
    <span class="kbd">?</span><span>this list</span>
  </div>
</div></dialog>

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
$("#banner").textContent = "Hypotheses, unverified: no intervention was run under the judge's rubric; " + (VPCT==null ? "none of" : VPCT + "% of") + " the lens cells the investigator quoted check out verbatim (build-time figure).";
// the two sides of every pattern, in plain words (internal ids keep matched/unmatched)
const SIDE = {matched: "flagged reply", unmatched: "clean reply"};
const SIDE_TIP = {matched: "the judge said this reply SHOWS the behavior", unmatched: "the judge said this reply does NOT show it"};
const side = l => SIDE[l] || l;
function nameOfId(id){ const m = String(id).match(/^diag:(\w+):(.*)$/); return m ? `${side(m[1])} s${m[2]}` : String(id).replace(/^agent:/, "agent "); }
function readName(r){ return r.source==="diag" ? `${side(r.label)} · s${r.sample_index}` : `agent ${r.conv_id}${r.same_rollout_as ? " (= " + nameOfId(r.same_rollout_as) + ", other lens sample)" : ""}`; }

// ------------------------------------------------------------------ state
const S = {key:null, data:null, read:null, pos:null, compare:[], layer:null, find:"", query:"", ctx:true, compact:false,
           colw:{}, rowh:{}, below:false, tab:"mechanisms", agentPromise:null};
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
  for (const r of data.reads){ data.byId[r.id] = r; if (r.conv_id && (!data.byConv[r.conv_id] || (!data.byConv[r.conv_id].rows && !data.byConv[r.conv_id].deferred))) data.byConv[r.conv_id] = r; if (r.tool_index!=null) data.byTool[r.tool_index] = r; if (r.source==="diag" && r.sample_index!=null) data.bySample[r.sample_index] = r; }
  for (const r of data.reads){ r.mention_ids = r.conv_id ? [r.conv_id] : []; for (const o of data.reads) if (o !== r && o.same_rollout_as === r.id && o.conv_id) r.mention_ids.push(o.conv_id); }
  data.mechs = []; data.runs.forEach((run, ri) => (run.mechanisms||[]).forEach((m, mi) => data.mechs.push({...m, run: ri, i: mi, auditor: run.auditor, seed: run.seed})));
  // verified highlight cells for this pattern: read → pos → [{layer, quote}]
  data.hl = {}; for (const h of (D.highlights||[])) if (h.pattern_key === key){ (data.hl[h.read] = data.hl[h.read] || {}); (data.hl[h.read][h.pos] = data.hl[h.read][h.pos] || []).push(h); }
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
  $("#read-select").innerHTML = reads.length ? reads.map(r => `<option value="${esc(r.id)}" ${r.id===S.read?"selected":""}>${esc(readLabel(r))}</option>`).join("") : `<option>—</option>`;
  $("#cmp-toggles").innerHTML = reads.filter(r => !r.parse_error).map((r, i) => `<button class="btn cmp" data-cmp="${esc(r.id)}" aria-pressed="${S.compare.includes(r.id)}" ${r.id===S.read?"disabled":""} title="${esc((SIDE_TIP[r.label] ? SIDE_TIP[r.label] + " — " : "") + readName(r) + " as a grid column (" + (i+1) + ")")}"><i class="${readDot(r)}"></i>${esc(readName(r))}</button>`).join("");
  $("#cmp-toggles").querySelectorAll("[data-cmp]").forEach(b => b.onclick = () => toggleCompare(b.dataset.cmp));
  $("#ctx-toggle").setAttribute("aria-pressed", String(S.ctx)); $("#wrap-toggle").setAttribute("aria-pressed", String(S.compact)); $("#below-toggle").setAttribute("aria-pressed", String(S.below));
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
  const data = S.data, runs = p.runs || [], run = runs[0];
  const mechs = run ? (run.mechanisms||[]) : [];
  const top = mechs.slice().sort((a,b) => (b.confidence||0) - (a.confidence||0))[0];
  const nrep = 64;  // WeirdChat samples ~64 replies per prompt; the published rate is over those
  let h = `<div class="pane"><h3>the case</h3><div><b>Behavior:</b> ${esc(p.behavior_name)}${matchLine(p.rubric) ? ` <span class="dim">— ${esc(matchLine(p.rubric))}</span>` : ""}</div>` +
    `<div style="margin-top:3px"><b>What WeirdChat found:</b> on this prompt, ${pct(p.published_match_rate)} of ${nrep} replies were judged to show it. Same prompt, same model, same settings — it went both ways.</div>` +
    `<div style="margin-top:3px"><b>What you see here:</b> one <span title="${esc(SIDE_TIP.matched)}">flagged</span> and one <span title="${esc(SIDE_TIP.unmatched)}">clean</span> reply, read token by token through the lens (layers 20–60), plus the investigator's probes.</div>` +
    `<div class="tags" style="margin-top:5px"><span class="tag">${pct(p.published_match_rate)} flagged</span><span class="tag">elo ${num(p.elo,0)}</span><span class="tag">${p.n_reads||0} lens reads</span>${p.weirdchat_url?`<a class="tag" href="${esc(p.weirdchat_url)}" target="_blank" rel="noopener">WeirdChat ↗</a>`:""}</div>` +
    `<div class="prompt">${esc(p.prompt)}</div></div>`;
  h += `<div class="pane"><h3>detective-joracle's hypothesis (unverified)</h3>` + (top ? `<div>${esc(top.mechanism)} <span class="badge ${top.confidence>=0.7?"miss":(top.confidence>=0.4?"hold":"dim")}">confidence ${num(top.confidence)}</span></div>` +
    (top.would_test_by ? `<div class="dim" style="margin-top:4px">How it would be tested: ${esc(top.would_test_by)} <span class="badge hold">not run</span></div>` : `<div class="dim" style="margin-top:4px"><span class="badge hold">not run</span> no test proposed</div>`) +
    (mechs.length > 1 ? `<div style="margin-top:4px"><a href="#" data-more>+ ${mechs.length-1} more in details ▸</a></div>` : "") : `<div class="empty">no agent run for this pattern yet — no hypothesis.</div>`) + `</div>`;
  h += `<div class="pane"><h3>agent summary</h3>` + (run ? `<div>${esc(run.summary || "(no summary)")}</div>` +
    `<div class="vrow" style="margin-top:6px"><span class="name">tool calls</span><span>${run.n_tool_calls||0} · ${run.n_readouts||0} readouts · ${run.n_chat||0} chat probes</span></div>` +
    `<div class="vrow"><span class="name">mechanisms</span><span>${mechs.length}</span></div>` +
    `<div class="vrow"><span class="name">verification</span>${verifiedBadge(p)}</div>` +
    `<div class="vrow"><span class="name">run</span><span class="dim">${esc(run.auditor)} s${run.seed} · stopped: ${esc(run.stopped_by||"?")}${runs.length>1?` · +${runs.length-1} more run(s)`:""}</span></div>` : `<div class="empty">no agent run for this pattern yet.</div>`) + `</div>`;
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
function renderText(){
  const r = curRead(), box = $("#text"); if (!r){ return; }
  const hl = S.data.hl[r.id] || {}, found = foundPositions();
  let h = "";
  if (r.parse_error) h += `<div class="notice">this readout page did not parse: ${esc(r.parse_error)}</div>`;
  const rows = r.rows || []; let i = 0;
  if (!rows.length) h += `<div class="status">no positions in this read.</div>`;
  while (i < rows.length){
    const region = rows[i].region; let inner = ""; let prev = null;
    for (; i < rows.length && rows[i].region === region; i++){
      const row = rows[i];
      if (prev!=null && row.pos - prev > 1) inner += `<span class="tok nodata" title="${row.pos-prev-1} positions not read">‥</span>`;
      prev = row.pos;
      const cls = ["tok", row.pos===S.pos?"cur":"", row.pos===r.aboutPos?"read":"", hl[row.pos]?"hit":"", found.has(row.pos)?"found":"", row.pos===r.fork_pos?"mark":""].join(" ");
      inner += `<span class="${cls}" data-pos="${row.pos}" title="position ${row.pos} · ${esc(row.kind)}${hl[row.pos]?" · a quoted phrase verifies here":""}${row.pos===r.aboutPos?" · about to speak":""}${row.pos===r.fork_pos?" · fork":""}">${tokHtml(row.tok)}</span>`;
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
  box.querySelectorAll(".roll[data-read]").forEach(b => b.onclick = () => selectRead(b.dataset.read, null));
  const cur = box.querySelector(".tok.cur"); if (cur && cur.scrollIntoView) cur.scrollIntoView({block:"nearest"});
}

// -------------------------------------------------------------------- grid
function citations(text){ const out = []; let cid = null; const re = /\b([wc]\d+[mu]?)\b|\bpos\s*~?\s*(\d+)/g; let m; text = text || "";
  while ((m = re.exec(text))){ if (m[1]){ if (S.data.byConv[m[1]]) cid = m[1]; out.push({cid: S.data.byConv[m[1]] ? m[1] : null, pos: null, index: m.index, len: m[0].length}); } else out.push({cid, pos: +m[2], index: m.index, len: m[0].length}); }
  return out; }
function mechsAt(r, pos){ const out = []; for (const m of S.data.mechs){ const here = citations(m.readout_cells).filter(c => c.pos === pos && c.cid && r.mention_ids.includes(c.cid)); if (here.length) out.push({m, via: here[0].cid}); } return out; }
function whereText(r, row){
  if (row.region === "header") return row.pos === r.aboutPos ? "about to speak: the model has read the request and written nothing — identical for every reply to this prompt" : "the chat boundary before the reply — identical for every reply to this prompt";
  if (row.region === "user") return "inside the user turn: the model is reading the request and has written nothing";
  if (row.region === "reply") return (row.pos === r.fork_pos ? "on the fork: " : "inside the reply: ") + "the model is writing its answer" + (row.pos === r.fork_pos ? " — the first word where the flagged and clean replies diverge (≈)" : "");
  return row.region;
}
function markCell(text, quotes, qrx){
  let html = esc(text);
  for (const q of quotes){ const e = esc(q); if (html.includes(e)) html = html.split(e).join(`<mark>${e}</mark>`); }
  if (qrx) html = html.replace(new RegExp("(" + rxEsc(esc(S.find)) + ")(?![^<]*>)", "gi"), `<mark class="find">$1</mark>`);
  return html;
}
function renderGrid(){
  const r = curRead(), posbar = $("#posbar"), wrap = $("#gridwrap"); if (!r) return;
  const p = S.pos, row = r.rowByPos[p];
  if (!row){ posbar.innerHTML = `<div class="status">${r.parse_error ? "nothing to show — this page did not parse." : "no position selected."}</div>`; wrap.innerHTML = ""; return; }
  const fl = mechsAt(r, p);
  let h = `<div class="row"><h2>position ${p}</h2><span class="tokbox">${esc(JSON.stringify(row.tok))}</span><span class="wherenote">${esc(whereText(r, row))} · ${esc(row.kind)}</span></div>`;
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
  if (identical) wrap.appendChild(el("div", "notice", "identical prefix — same activation, different lens samples" + (row.region==="reply" ? " (the replies still agree at this position)" : "")));
  const colw = Math.max(260, Math.floor((wrap.clientWidth - 58) / Math.max(1, cols.length)) - 1); wrap.style.setProperty("--col", colw + "px");
  const table = el("table"), thead = el("thead"), hr = el("tr"); hr.appendChild(el("th", "layer", "layer"));
  cols.forEach((c, j) => { const th = el("th", readDot(c), readName(c)); th.title = (SIDE_TIP[c.label] ? SIDE_TIP[c.label] + " · " : "") + c.id; if (!identical && cols.length > 1) th.appendChild(el("span", "own", toks[j]==null ? "not read at pos " + p : JSON.stringify(toks[j])));
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
      else { const quotes = ((S.data.hl[c.id]||{})[p]||[]).filter(x => x.layer === L).map(x => x.quote); if (quotes.length) td.className = "hit";
        const d = el("div", "cell"); d.innerHTML = samples.map(s => { const en = D.en && D.en[s]; return `<div class="samp">${markCell(s.trim(), quotes, qrx)}${en ? `<div class="en">${esc(en)}</div>` : ""}</div>`; }).join(""); td.appendChild(d); }
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
    (p.runs||[]).forEach((run, ri) => { if (rx.test(run.summary||"")) rows.push({p, where: `run ${ri} summary`, snip: snip(run.summary)});
      (run.mechanisms||[]).forEach((m, mi) => { const t = [m.mechanism, m.evidence, m.readout_cells].join(" · "); if (rx.test(t)) rows.push({p, where: `mechanism ${mi+1}`, snip: snip(t), cell: firstCell(m.readout_cells)}); }); });
  }
  box.hidden = false;
  box.innerHTML = `<h4>${rows.length} hit${rows.length===1?"":"s"} for “${esc(q)}”</h4>` + rows.slice(0, 60).map((x, i) => `<button class="res" data-i="${i}"><span class="fam">${esc(x.p.behavior_name)}</span><span class="where">${esc(cut(x.p.group_summary||x.p.key, 34))} · ${esc(x.where)}</span><span class="snip">${esc(x.snip)}</span></button>`).join("") + (rows.length > 60 ? `<div class="dim">+${rows.length-60} more</div>` : "");
  box.querySelectorAll("[data-i]").forEach(b => b.onclick = () => { const x = rows[+b.dataset.i]; location.hash = x.cell ? patUrl(x.p.key, x.cell.read, x.cell.pos) : patUrl(x.p.key); });
}

// ------------------------------------------------------------------- drawer
function linkCells(text, data){ let out = "", last = 0, cid = null; const re = /\b([wc]\d+[mu]?)\b|\bpos\s*~?\s*(\d+)/g; let m; text = text || "";
  while ((m = re.exec(text))){ out += esc(text.slice(last, m.index));
    if (m[1]){ const r = data.byConv[m[1]]; if (r) cid = m[1]; out += r ? `<a href="${patUrl(data.key, r.id, null)}">${esc(m[0])}</a>` : esc(m[0]); }
    else { const r = cid ? data.byConv[cid] : null, pos = +m[2]; const ok = r && (r.deferred || (r.rowByPos && r.rowByPos[pos])); out += ok ? `<a href="${patUrl(data.key, r.id, pos)}" title="open ${esc(r.id)} at position ${pos}">${esc(m[0])}</a>` : esc(m[0]); }
    last = m.index + m[0].length; }
  return out + esc(text.slice(last)); }
function mechBlock(m, i, data){ return `<div class="mech"><div><span class="n">${i+1}.</span>${esc(m.mechanism)}${m.confidence==null?"":` <span class="badge ${m.confidence>=0.7?"miss":(m.confidence>=0.4?"hold":"dim")}">confidence ${num(m.confidence)}</span>`}</div>` + (m.evidence ? `<div class="ev">${esc(m.evidence)}</div>` : "") + (m.readout_cells ? `<div class="cells">cells: ${linkCells(m.readout_cells, data)}</div>` : "") + (m.would_test_by ? `<div class="test"><b>would test by — not run in this pass:</b> ${esc(m.would_test_by)}</div>` : "") + `</div>`; }
function stepBlock(s, i, data){ const rd = s.tool_index!=null ? data.byTool[s.tool_index] : null;
  let h = `<div class="step ${esc(s.name||"")}"><div class="hd"><span class="nm">${esc(s.name||"assistant")}</span><span>#${i+1}${s.meta?" · "+esc(s.meta):""}</span>${rd ? `<a href="${patUrl(data.key, rd.id, null)}" title="${esc(rd.id)}">open this read → ${esc(readName(rd))}${rd.parse_error?" (unparsed)":""}</a>` : ""}</div>`;
  if (s.think) h += `<div class="think">${esc(s.think)}</div>`;
  if (s.name === "chat") h += `<div class="call">user → <span class="usr">${esc(s.user)}</span></div>`; else if (s.name) h += `<div class="call">${esc(s.name)}(${esc(s.args)})</div>`;
  if (s.output){ const o = s.output; h += `<pre class="block">${esc(o.slice(0,500))}${o.length>500?" …":""}</pre>`; if (o.length > 500) h += `<details><summary>show full ${s.name==="chat"?"reply":"result"} (${o.length} chars)</summary><pre class="block">${esc(o)}</pre></details>`; }
  return h + `</div>`; }
function renderBelow(){
  const box = $("#below"); box.hidden = !S.below; if (!S.below) return;
  const data = S.data, tabs = [["mechanisms", "ranked mechanisms"], ["transcript", "agent transcript"], ["highlights", "highlights"]];
  $("#tabs").innerHTML = tabs.map(([k, l]) => `<button class="btn" data-tab="${k}" aria-pressed="${S.tab===k}">${l}</button>`).join("") + `<span class="sub" style="margin-left:8px">unverified hypotheses — nothing here was tested</span>`;
  $("#tabs").querySelectorAll("[data-tab]").forEach(b => b.onclick = () => { S.tab = b.dataset.tab; renderBelow(); });
  let h = "";
  if (!data) h = `<div class="status">no pattern loaded.</div>`;
  else if (S.tab === "mechanisms"){ if (!data.runs.length) h = `<div class="empty">no agent run for this pattern yet.</div>`;
    data.runs.forEach((run, ri) => { h += `<div class="sub" style="margin:${ri?"12px":"0"} 0 6px"><b>${esc(run.auditor)}</b> seed ${run.seed} · ${esc(run.summary||"")}</div>`; const ms = (run.mechanisms||[]).slice().sort((a,b) => (b.confidence||0)-(a.confidence||0)); if (!ms.length) h += `<div class="empty">no mechanisms reported.</div>`; ms.forEach((m, i) => { h += mechBlock(m, i, data); }); }); }
  else if (S.tab === "transcript"){ if (!data.runs.length) h = `<div class="empty">no agent run for this pattern yet.</div>`;
    data.runs.forEach(run => { h += `<div class="sub" style="margin:0 0 6px"><b>${esc(run.auditor)}</b> seed ${run.seed} · ${(run.steps||[]).length} steps · stopped: ${esc(run.stopped_by||"?")}</div>`; (run.steps||[]).forEach((s, i) => { h += stepBlock(s, i, data); }); if ((run.notes||[]).length){ h += `<div class="sub" style="margin:8px 0 4px">scratch notes</div>`; run.notes.forEach(n => { h += `<pre class="block">${esc(n)}</pre>`; }); } }); }
  else { const items = D.highlights || []; const mine = items.filter(x => x.pattern_key === S.key), rest = items.filter(x => x.pattern_key !== S.key);
    h = `<div class="sub" style="margin:0 0 6px">${mine.length} on this pattern · ${rest.length} elsewhere — lens cells a mechanism quotes that verify verbatim at build time; click to jump</div><div class="hlgrid">` + [...mine, ...rest].map(x => { const idx = items.indexOf(x);
      const smp = esc(x.sample.trim()).split(esc(x.quote)).join(`<mark>${esc(x.quote)}</mark>`), loc = esc(x.local.slice(0, x.local.length - x.token.length)) + `<b>${esc(x.token)}</b>`;
      return `<button class="hlc ${esc(x.region)}" data-hl="${idx}"><div class="where"><b>${esc(x.behavior)}</b> ${esc(cut(x.summary, 50))} · <span title="${esc(SIDE_TIP[(x.read.match(/^diag:(\w+):/)||[])[1]]||"")}">${esc(nameOfId(x.read))}</span> · pos ${x.pos} · L${x.layer}${x.kind==="hand"?' · <span class="hand">hand-picked</span>':""}</div><div class="loc">${loc}</div><div class="q">${smp}</div><div class="n">${esc(x.note)}</div></button>`; }).join("") + `</div>`; }
  $("#drawer").innerHTML = h;
  $("#drawer").querySelectorAll("[data-hl]").forEach(b => b.onclick = () => { const x = (D.highlights||[])[+b.dataset.hl]; location.hash = patUrl(x.pattern_key, x.read, x.pos); });
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
  else if (k === "c") $("#ctx-toggle").click(); else if (k === "w") $("#wrap-toggle").click();
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
