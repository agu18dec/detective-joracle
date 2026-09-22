"""Build the WeirdChat explanation viewer — a small index page plus per-pattern data files.

Walks ``<out_root>/{patterns,diag,runs}`` plus an optional ``synth.json`` and writes into
``<site_dir>`` (default ``<out_root>/site``):

    index.html                 page + JS + the small payload (pattern metadata, per-run
                               mechanisms, behaviors table, synth clusters) — no samples/grids
    data/<pattern_key>.json    the heavy payload (samples, lens reads, full agent transcripts),
                               fetched on demand when a pattern page opens
    data/<pattern_key>.agent.json   only when the pattern's file would blow the per-file budget:
                               the agent's own lens reads, fetched when one is first selected

    python scripts/weirdchat/build_site.py [out=outputs/weirdchat] [site=<dir>]
    cd outputs/weirdchat/site && python -m http.server 8905

The pattern page follows the lie-detection sweep viewer: left, the read tokens of one lens
read grouped by region, every token clickable; right, that position's readout at every layer,
one column per compared read. A "read" is one OLens readout over one conversation — from the
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
    return run, [as_dict(t) for t in as_list(record.get("tool_log"))]


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


def light_pattern(pat: dict[str, Any]) -> dict[str, Any]:
    """What index.html carries per pattern: metadata, flags, and every run's mechanisms."""
    light = {k: v for k, v in pat.items() if k not in HEAVY_PATTERN_KEYS}
    light["runs"] = [{k: v for k, v in r.items() if k not in HEAVY_RUN_KEYS} for r in pat["runs"]]
    light["has_diag"] = any(r["source"] == "diag" for r in pat["reads"])
    light["n_reads"] = len(pat["reads"])
    light["n_samples"] = len(pat["samples"])
    light["n_runs"] = len(pat["runs"])
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


# ------------------------------------------------------------------------ main
def build_site(out_root: Path, site_dir: Path) -> Path:
    """Collect everything under ``out_root``; write ``site_dir/index.html`` + ``data/*.json``."""
    pattern_dir = out_root / "patterns"
    paths = sorted(pattern_dir.glob("*.json")) if pattern_dir.is_dir() else []
    patterns = [pattern_of(p, out_root) for p in paths]
    patterns.sort(key=lambda p: (p["behavior_name"], p["key"]))
    data = {
        "patterns": [light_pattern(p) for p in patterns],
        "behaviors": behaviors_of(patterns),
        "clusters": clusters_of(out_root),
        "synth": synth_meta(out_root),
        "counts": {
            "patterns": len(patterns),
            "behaviors": len({p["behavior_id"] for p in patterns}),
            "runs": sum(len(p["runs"]) for p in patterns),
            "with_diag": sum(1 for p in patterns if any(r["source"] == "diag" for r in p["reads"])),
            "reads": sum(len(p["reads"]) for p in patterns),
            "mechanisms": sum(p["n_mechanisms"] for p in patterns),
        },
        "source": str(out_root),
    }
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir = site_dir / "data"
    data_dir.mkdir(exist_ok=True)
    sizes: list[tuple[str, int]] = []
    warnings: list[str] = []
    for pat in patterns:
        for name, size in write_pattern_files(pat, data_dir):
            sizes.append((name, size))
            if size > DATA_FILE_BUDGET:
                warnings.append(f"{name} is {mb(size)} > {mb(DATA_FILE_BUDGET)} budget")

    payload = dumps(data).replace("</", "<\\/")
    out = site_dir / "index.html"
    out.write_text(TEMPLATE.replace("/*__DATA__*/", payload))
    sizes.insert(0, ("index.html", out.stat().st_size))
    total = sum(s for _, s in sizes)
    if total > TOTAL_BUDGET:
        warnings.append(f"site total is {mb(total)} > {mb(TOTAL_BUDGET)} budget")

    for name, size in sizes:
        print(f"{mb(size):>10}  {name}")
    report_reads(patterns)
    c = data["counts"]
    print(
        f"{len(patterns)} patterns · {c['behaviors']} behaviors · {c['runs']} agent runs · "
        f"{c['reads']} reads · {c['mechanisms']} mechanisms · {len(data['clusters'])} clusters "
        f"· total {mb(total)} in {len(sizes)} files"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(out.resolve())
    return out


def main() -> None:
    args: dict[str, str] = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out_root = Path(args.get("out", "outputs/weirdchat"))
    site_dir = Path(args["site"]) if "site" in args else out_root / "site"
    build_site(out_root, site_dir)


# plain text (no quotes/backslashes): it is injected into a JS string and set via textContent
BANNER = (
    "No ground truth, no interventions. The agent was told the behavior and asked why the model "
    "does it; every mechanism below is an UNVERIFIED HYPOTHESIS read off chat probes and OLens "
    "readouts. Nothing on this page has been causally tested."
)

TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WeirdChat — why does it do that?</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  color-scheme:light dark;
  --paper:#F7F7F4; --card:#FFFFFF; --ink:#1B2230; --ink2:#4A5262; --mute:#7A8394; --line:#DDDFE3; --line2:#ECEDEF;
  --s3d:#0E7C7B; --s3d-bg:#E6F4F3; --ddp:#B8531A; --ddp-bg:#FBEEE4; --jl:#4B5563; --jl-bg:#EEF0F3;
  --claims:#B42318; --claims-bg:#FCE9E6; --discl:#15803D; --discl-bg:#E5F4EA; --silent:#B45309; --silent-bg:#FCF0DF;
  --sel:#1B2230; --selfg:#FFFFFF; --swept:#B9BEC7; --mark:#FFF1A8; --flagdot:#B42318;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif; --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif; --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#12151B; --card:#1A1E26; --ink:#E7EAF0; --ink2:#B4BCCB; --mute:#8A93A5; --line:#2C323D; --line2:#232832;
    --s3d:#4FD1C5; --s3d-bg:#14312F; --ddp:#F0A070; --ddp-bg:#3A2418; --jl:#B4BCCB; --jl-bg:#232832;
    --claims:#FF8A80; --claims-bg:#3A1D1C; --discl:#6EE7A0; --discl-bg:#16301F; --silent:#F0B25F; --silent-bg:#3A2E14;
    --sel:#E7EAF0; --selfg:#12151B; --swept:#4A5262; --mark:#6B5A10; --flagdot:#FF8A80;
  }
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.45;margin:0}
a{color:inherit}
header{padding:16px 28px 12px;border-bottom:1px solid var(--line);background:var(--card)}
h1{font-family:var(--serif);font-weight:600;font-size:24px;margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}
h1 a{text-decoration:none}
h2{font-family:var(--serif);font-weight:600;font-size:19px;margin:22px 0 8px;letter-spacing:-.01em}
h3{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:16px 0 6px;font-weight:500}
.crumb{font-size:12.5px;color:var(--mute)}
.crumb a{color:var(--ink2)}
.banner{background:var(--claims-bg);color:var(--claims);border:1px solid var(--claims);border-radius:6px;padding:8px 12px;margin:10px 0 0;font-size:12.5px;max-width:120ch}
.banner b{letter-spacing:.04em}
.page{padding:18px 28px 80px;max-width:1500px}
.counts{color:var(--ink2);font-size:13px;margin:0 0 4px}
.counts b{color:var(--ink)}
.kkey{font-family:var(--mono);font-size:11.5px;color:var(--ink2)}
.hint{color:var(--mute);font-size:12.5px}
.none{color:var(--mute);font-size:13px}
kbd{font-family:var(--mono);font-size:11px;border:1px solid var(--line);border-radius:3px;padding:0 4px;background:var(--card)}
table.t{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden}
table.t th{text-align:left;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute);font-weight:500;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
table.t td{padding:7px 10px;border-bottom:1px solid var(--line2);vertical-align:top;font-size:13px}
table.t tr:last-child td{border-bottom:none}
table.t tr.click{cursor:pointer} table.t tr.click:hover td{background:var(--s3d-bg)}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:12px;white-space:nowrap}
.tag{font-size:11px;border-radius:4px;padding:1px 6px;font-family:var(--mono)}
.tag.y{background:var(--discl-bg);color:var(--discl)} .tag.n{background:var(--line2);color:var(--mute)} .tag.w{background:var(--silent-bg);color:var(--silent)}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;margin:0 0 10px}
pre.block{font-family:var(--mono);font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;color:var(--ink)}
pre.small{font-size:11.5px;color:var(--ink2);border:none;padding:0;background:none}
details{margin:6px 0} summary{cursor:pointer;color:var(--s3d);font-size:12.5px} summary::marker{color:var(--mute)}
.pill{display:inline-block;font-size:11.5px;color:var(--ink2);background:var(--line2);border-radius:999px;padding:1px 9px;margin:0 5px 5px 0;font-family:var(--mono)}
/* --- read picker (chips), as in the sweep viewer --- */
.recs{display:flex;flex-direction:column;gap:8px;margin:12px 0 6px}
.pick{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.pick .lbl{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);min-width:64px}
.tab{font:inherit;font-family:var(--mono);font-size:12px;border:1px solid var(--line);background:var(--card);border-radius:5px;padding:3px 9px;cursor:pointer;color:var(--ink2)}
.tab:hover{border-color:var(--ink2)} .tab[aria-pressed="true"]{border-color:var(--ink);color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
.tab:disabled{opacity:.55;cursor:default}
.chip{border:1px solid var(--line);background:var(--card);border-radius:999px;padding:3px 10px;font-size:12.5px;cursor:pointer;display:inline-flex;gap:6px;align-items:center;font-family:var(--sans);color:var(--ink2)}
.chip:hover{border-color:var(--ink2)} .chip[aria-pressed="true"]{border-color:var(--ink);color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
.chip .g{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.g.matched{background:var(--claims)} .g.unmatched{background:var(--discl)} .g.agent{background:var(--swept)} .g.err{background:var(--silent)}
.chip .n{color:var(--mute);font-family:var(--mono);font-size:11.5px}
/* --- two-column viewer --- */
main.two{display:grid;grid-template-columns:minmax(380px,46%) 1fr;gap:0;border:1px solid var(--line);border-radius:6px;background:var(--paper);min-height:60vh}
@media (max-width:980px){main.two{grid-template-columns:1fr}}
#left{border-right:1px solid var(--line);padding:14px 18px 30px;overflow-x:hidden;min-width:0}
#right{padding:14px 20px 30px;position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;min-width:0}
.recmeta{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.badge{font-size:11.5px;letter-spacing:.04em;padding:2px 8px;border-radius:4px;font-weight:500}
.badge.matched{background:var(--claims-bg);color:var(--claims)} .badge.unmatched{background:var(--discl-bg);color:var(--discl)} .badge.agent{background:var(--jl-bg);color:var(--jl)} .badge.err{background:var(--silent-bg);color:var(--silent)}
.block{margin:0 0 10px;border:1px solid var(--line2);border-radius:6px;background:var(--card)}
.block .role{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);padding:6px 10px 0;display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.block .role .rl{letter-spacing:0;text-transform:none;color:var(--ink2);font-size:11.5px}
.block .role button{font:inherit;letter-spacing:0;text-transform:none;background:none;border:1px solid var(--line);border-radius:4px;padding:0 6px;cursor:pointer;color:var(--ink2)}
.block pre{margin:0;padding:6px 10px 10px;font-family:var(--mono);font-size:12px;line-height:1.7;white-space:pre-wrap;word-break:break-word;color:var(--ink2)}
.block pre.clip{max-height:120px;overflow:hidden;position:relative}
.block pre.clip::after{content:"";position:absolute;left:0;right:0;bottom:0;height:40px;background:linear-gradient(transparent,var(--card))}
.block.reply{border-color:var(--ink2)} .block.reply pre{color:var(--ink)}
.tok{border-radius:3px;padding:0 1px;margin:0 -1px}
.tok.swept{outline:1px solid var(--swept);outline-offset:-1px;cursor:pointer;background:var(--card)}
.tok.swept:hover{outline-color:var(--ink)}
.tok.swept.flag{background:var(--mark)}
.tok.sel{background:var(--sel)!important;color:var(--selfg);outline-color:var(--sel)}
.tok.r1{box-shadow:0 2px 0 var(--ink)}
.special{color:var(--mute)}
.nl{color:var(--mute);font-size:10.5px}
.gap{color:var(--mute);font-size:10px;letter-spacing:-1px;padding:0 2px;cursor:default}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:6px 0 12px}
.legend span i{display:inline-block;width:14px;height:12px;vertical-align:-2px;margin-right:5px;border-radius:2px;background:var(--card)}
.poshdr{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.poshdr h2{margin:0 0 4px}
.poshdr .navb{margin-left:auto;display:flex;gap:4px}
.navb button{font:inherit;background:var(--card);border:1px solid var(--line);border-radius:4px;padding:2px 9px;cursor:pointer;color:var(--ink)}
.navb button:hover{border-color:var(--ink2)} .navb button:disabled{opacity:.4;cursor:default}
.tokbox{font-family:var(--mono);background:var(--sel);color:var(--selfg);padding:1px 6px;border-radius:3px;font-size:12.5px}
.local{font-family:var(--mono);font-size:12px;color:var(--ink2);background:var(--card);border:1px solid var(--line2);border-radius:6px;padding:8px 10px;margin:10px 0 12px;white-space:pre-wrap;word-break:break-word}
.local b{color:var(--ink);background:var(--mark);font-weight:500}
.flags{margin:0 0 14px}
.flag{border-left:3px solid var(--flagdot);background:var(--card);padding:6px 10px;margin:0 0 6px;border-radius:0 6px 6px 0;font-size:13px}
.flag .cat{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--claims);margin-right:8px}
.flag .cell{font-family:var(--mono);font-size:11.5px;color:var(--mute)}
.flag q{font-family:var(--mono);font-size:12px;quotes:"“" "”";display:block;margin:3px 0;color:var(--ink2)}
.flag .why{color:var(--ink2)}
.flag.other{border-left-color:var(--swept);opacity:.8}
table.reads{border-collapse:separate;border-spacing:0;width:100%;table-layout:fixed}
table.reads th{text-align:left;font-weight:500;font-size:12px;padding:6px 8px;border-bottom:1px solid var(--line);color:var(--ink2);vertical-align:bottom}
table.reads th .own{display:block;font-family:var(--mono);font-size:11.5px;color:var(--ink);font-weight:400;margin-top:2px}
table.reads th.matched{color:var(--claims)} table.reads th.unmatched{color:var(--discl)}
table.reads td{vertical-align:top;padding:8px 8px 10px;border-bottom:1px solid var(--line2);font-family:var(--mono);font-size:11.8px;line-height:1.5;white-space:pre-wrap;word-break:break-word;color:var(--ink)}
table.reads td.L{font-family:var(--sans);font-weight:500;color:var(--ink2);width:52px;white-space:nowrap}
.samp{padding:2px 0 6px;border-left:2px solid var(--line2);padding-left:8px;margin:0 0 6px}
.samp.hit{border-color:var(--flagdot)}
mark{background:var(--mark);color:inherit;padding:0 1px}
.notice{font-size:12.5px;color:var(--ink2);background:var(--card);border:1px solid var(--line2);border-radius:6px;padding:6px 10px;margin:0 0 8px}
/* --- below the viewer --- */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:980px){.cols{grid-template-columns:1fr}}
.roll{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:9px 11px;margin:0 0 9px}
.roll .hd{font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--mute);margin-bottom:5px;display:flex;gap:8px;align-items:baseline}
.roll.matched{border-left:3px solid var(--claims)} .roll.unmatched{border-left:3px solid var(--discl)}
.step{border:1px solid var(--line2);border-radius:6px;background:var(--card);margin:0 0 8px;padding:7px 10px}
.step .hd{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.step .hd .nm{font-family:var(--mono);letter-spacing:0;text-transform:none;color:var(--ink);background:var(--line2);border-radius:4px;padding:0 6px;font-weight:500}
.step .hd a{letter-spacing:0;text-transform:none;color:var(--s3d);margin-left:auto;font-size:12px}
.step .think{font-size:12.5px;color:var(--ink2);white-space:pre-wrap;word-break:break-word;margin-top:5px}
.step .call{font-family:var(--mono);font-size:11.5px;color:var(--ink2);margin-top:5px;white-space:pre-wrap;word-break:break-word}
.step .call .k{color:var(--mute)} .step .call .usr{color:var(--ink);background:var(--s3d-bg);border-radius:3px;padding:0 3px}
.step.readouts{border-color:var(--jl-bg)} .step.finish{border-color:var(--ink)} .step.finish .nm{background:var(--ink);color:var(--selfg)}
.mech{border:1px solid var(--line2);border-left:3px solid var(--s3d);border-radius:0 6px 6px 0;background:var(--card);padding:9px 11px;margin:0 0 8px}
.mech .n{font-family:var(--mono);color:var(--mute);font-size:11.5px;margin-right:6px}
.mech .txt{font-size:13.5px;color:var(--ink)}
.mech .ev{font-size:12.5px;color:var(--ink2);margin-top:5px}
.mech .cells{font-family:var(--mono);font-size:11.5px;color:var(--mute);margin-top:4px;word-break:break-word;white-space:pre-wrap}
.mech .cells a{color:var(--s3d)}
.mech .test{font-size:12px;color:var(--mute);font-style:italic;margin-top:6px;border-top:1px dashed var(--line);padding-top:5px}
.mech .test b{font-style:normal;font-weight:500;letter-spacing:.04em;text-transform:uppercase;font-size:10.5px;color:var(--silent)}
.conf{font-family:var(--mono);font-size:11.5px;color:var(--ink2);background:var(--line2);border-radius:4px;padding:1px 6px}
</style>

<header>
  <h1><a href="#/">WeirdChat — why does it do that?</a></h1>
  <div class="crumb" id="crumb"></div>
  <div class="banner"><b>UNVERIFIED HYPOTHESES.</b> <span id="banner"></span></div>
</header>
<div class="page" id="app"></div>

<script>
const D = /*__DATA__*/;
const BANNER = "__BANNER__";
const esc = s => (s==null?"":String(s)).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const num = (v,d) => v==null ? "—" : Number(v).toFixed(d==null?2:d);
const byKey = {}; (D.patterns||[]).forEach(p => byKey[p.key] = p);
document.getElementById("banner").textContent = BANNER;

function cut(s, n){ s = s==null ? "" : String(s); return s.length<=n ? s : s.slice(0,n) + " …"; }
function dataUrl(key){ return "data/" + encodeURIComponent(key) + ".json"; }
function patUrl(key, read, pos){
  let h = "#/pattern/" + encodeURIComponent(key);
  if (read) h += "?read=" + encodeURIComponent(read) + (pos==null ? "" : "&pos=" + pos);
  return h;
}

// text shown up to `n` chars, with the remainder behind a disclosure
function expandable(text, n, label){
  const t = text==null ? "" : String(text);
  if (!t) return `<p class="none">(empty)</p>`;
  if (t.length <= n) return `<pre class="block">${esc(t)}</pre>`;
  return `<pre class="block">${esc(t.slice(0,n))} …</pre>` +
    `<details><summary>${esc(label||"show full")} (${t.length} chars)</summary><pre class="block">${esc(t)}</pre></details>`;
}

// ------------------------------------------------------------------ overview
function renderOverview(){
  const c = D.counts || {};
  let h = `<p class="counts"><b>${c.patterns||0}</b> patterns · <b>${c.behaviors||0}</b> behaviors · ` +
    `<b>${c.runs||0}</b> agent runs · <b>${c.with_diag||0}</b> with OLens diagnostics · <b>${c.reads||0}</b> lens reads · ` +
    `<b>${c.mechanisms||0}</b> reported mechanisms` +
    (D.synth && D.synth.model ? ` · synthesis by <span class="kkey">${esc(D.synth.model)}</span>` : "") + `</p>`;
  h += `<p class="crumb">source: <span class="kkey">${esc(D.source)}</span></p>`;

  h += `<h2>Behaviors</h2>`;
  const bs = D.behaviors || [];
  if (!bs.length) h += `<p class="none">no patterns found.</p>`;
  else {
    h += `<table class="t"><thead><tr><th>behavior</th><th>id</th><th class="num">patterns</th>` +
      `<th class="num">mean published match</th><th class="num">agent runs</th></tr></thead><tbody>`;
    for (const b of bs)
      h += `<tr><td>${esc(b.behavior_name)}</td><td class="kkey">${esc(b.behavior_id)}</td>` +
        `<td class="num">${b.n_patterns}</td><td class="num">${num(b.mean_match_rate)}</td>` +
        `<td class="num">${b.n_runs}</td></tr>`;
    h += `</tbody></table>`;
  }

  h += `<h2>Patterns</h2>`;
  if ((D.patterns||[]).length){
    h += `<table class="t"><thead><tr><th>behavior</th><th>group summary</th><th class="num">match</th>` +
      `<th class="num">elo</th><th>diag</th><th>reads</th><th>runs</th><th class="num">mechanisms</th></tr></thead><tbody>`;
    for (const p of D.patterns)
      h += `<tr class="click" data-go="${patUrl(p.key)}">` +
        `<td>${esc(p.behavior_name)}<div class="kkey">${esc(p.key)}</div></td>` +
        `<td>${esc(cut(p.group_summary, 160))}</td>` +
        `<td class="num">${num(p.published_match_rate)}</td>` +
        `<td class="num">${num(p.elo, 0)}</td>` +
        `<td><span class="tag ${p.has_diag?"y":"n"}">${p.has_diag?"yes":"none"}</span></td>` +
        `<td><span class="tag ${p.n_reads?"y":"n"}">${p.n_reads||"none"}</span></td>` +
        `<td><span class="tag ${p.n_runs?"y":"n"}">${p.n_runs||"none"}</span></td>` +
        `<td class="num">${p.n_mechanisms}</td></tr>`;
    h += `</tbody></table>`;
  }

  h += `<h2>Mechanism clusters</h2>`;
  const cl = D.clusters || [];
  if (!cl.length) h += `<p class="none">no synth.json yet — clusters appear once the synthesis pass runs.</p>`;
  cl.forEach((k,i) => {
    h += `<div class="card"><a href="#/cluster/${i}"><b>${esc(k.name)}</b></a>` +
      `<div class="crumb">${k.members.length} pattern${k.members.length===1?"":"s"} · ${k.behavior_ids.length} behavior${k.behavior_ids.length===1?"":"s"}</div>` +
      (k.description ? `<div style="margin-top:5px">${esc(k.description)}</div>` : "") + `</div>`;
  });
  return h;
}

// ------------------------------------------------------------------- pattern
// V = the open pattern page: its heavy data, the selected read/position and the comparison set
let V = null;
const heavyCache = {};

function renderPattern(key){
  const p = byKey[key];
  if (!p) return `<p class="none">no pattern <span class="kkey">${esc(key)}</span>.</p>`;
  let h = `<h2 style="margin-top:4px">${esc(p.behavior_name)}</h2>` +
    `<p class="crumb"><span class="kkey">${esc(p.key)}</span>` +
    (p.entry_id ? ` · entry ${esc(p.entry_id)}` : "") +
    (p.group_id ? ` · group ${esc(p.group_id)}` : "") +
    (p.n_group_members!=null ? ` · ${p.n_group_members} group members` : "") + `</p>`;
  h += `<div>` +
    `<span class="pill">published match ${num(p.published_match_rate)}</span>` +
    `<span class="pill">elo ${num(p.elo,0)}</span>` +
    Object.keys(p.elo_axes||{}).map(k => `<span class="pill">${esc(k)} ${num(p.elo_axes[k])}</span>`).join("") +
    (p.weirdchat_url ? `<a class="pill" href="${esc(p.weirdchat_url)}" target="_blank" rel="noopener">WeirdChat ↗</a>` : "") +
    `</div>`;
  if (p.group_summary) h += `<div class="card">${esc(p.group_summary)}</div>`;
  h += `<h3>prompt (verbatim)</h3><pre class="block">${esc(p.prompt)}</pre>`;
  if (p.rubric) h += `<details><summary>transcript rubric</summary><pre class="block">${esc(p.rubric)}</pre></details>`;
  h += `<div id="heavy" data-key="${esc(key)}"><p class="none">loading <span class="kkey">${esc(dataUrl(key))}</span> …</p></div>`;
  return h;
}

function fillHeavy(key, html){
  const slot = document.getElementById("heavy");
  if (slot && slot.dataset.key === key) slot.innerHTML = html;
  return !!(slot && slot.dataset.key === key);
}
function failNotice(url, why){
  return `<p class="none">could not load <span class="kkey">${esc(url)}</span> (${esc(why)}). ` +
    `Browsers block fetch() from a file:// page — serve the site dir instead: <span class="kkey">python -m http.server</span> then open the printed URL.</p>`;
}
function fetchJson(url){
  if (typeof fetch !== "function") return Promise.reject(new Error("no fetch()"));
  return fetch(url).then(r => { if (!r.ok) throw new Error(r.status + " " + r.statusText); return r.json(); });
}
function loadPattern(key, want){
  const url = dataUrl(key);
  const go = data => { if (document.getElementById("heavy") && document.getElementById("heavy").dataset.key === key) openViewer(key, data, want); };
  if (heavyCache[key]) return go(heavyCache[key]);
  // only the fetch is caught: a render bug must surface in the console, not as a "could not load"
  fetchJson(url).then(data => { heavyCache[key] = data; return data; },
                      err => { fillHeavy(key, failNotice(url, err && err.message ? err.message : String(err))); return null; })
    .then(data => { if (data) go(data); });
}

// ---- read indexing
function indexRead(r){
  r.rowByPos = {}; r.positions = [];
  for (const row of r.rows || []){ r.rowByPos[row.pos] = row; r.positions.push(row.pos); }
}
function prepareData(key, data){
  data.reads = data.reads || []; data.runs = data.runs || []; data.samples = data.samples || [];
  data.reads.forEach(indexRead);
  data.byId = {}; data.byConv = {}; data.byTool = {};
  for (const r of data.reads){ data.byId[r.id] = r; if (r.conv_id && !data.byConv[r.conv_id]) data.byConv[r.conv_id] = r; if (r.tool_index!=null) data.byTool[r.tool_index] = r; }
  for (const r of data.reads){
    // ids a mechanism can cite to mean this read: its conv id, or the agent's own read of the same rollout
    r.mention_ids = r.conv_id ? [r.conv_id] : [];
    for (const o of data.reads) if (o !== r && o.same_rollout_as === r.id && o.conv_id) r.mention_ids.push(o.conv_id);
  }
  data.mechs = [];
  data.runs.forEach((run, ri) => (run.mechanisms||[]).forEach((m, mi) => data.mechs.push({...m, run: ri, i: mi, auditor: run.auditor, seed: run.seed})));
  // fragments the agent quoted: split its cell citations / evidence on quotes, keep >= 25 chars
  const frags = new Set();
  for (const m of data.mechs) for (const f of ((m.readout_cells||"") + "\n" + (m.evidence||"")).split(/["'“”‘’;\n]/)){ const t = f.trim(); if (t.length >= 25) frags.add(t); }
  data.quotes = [...frags].sort((a,b) => b.length - a.length);
  data.key = key;
}

// ---- opening the viewer
function openViewer(key, data, want){
  if (!data.byId) prepareData(key, data);
  V = {key, data, readId: null, pos: null, compare: []};
  const reads = data.reads;
  let h = `<h2>Lens reads</h2>`;
  if (!reads.length){
    h += `<p class="none">no lens reads for this pattern yet (no diag file, no agent readouts).</p>`;
  } else {
    h += `<p class="hint">A read is one OLens readout over one conversation: from the diagnostic pass (matched / unmatched study rollouts), or parsed back out of the agent's own <span class="kkey">readouts</span> tool pages (w###m/u = the study rollouts it was seeded with — same text, a different lens sample; c### = conversations it created via chat). Left: the tokens that were read. Right: what the lens decoded at the selected token, every layer, one column per compared read.</p>`;
    h += `<div class="recs" id="recs"></div><main class="two"><section id="left"></section><section id="right"></section></main>`;
  }
  h += belowViewer(data);
  fillHeavy(key, h);
  wireBelow(data);
  if (!reads.length) return;
  const diagM = reads.find(r => r.source==="diag" && r.label==="matched"), diagU = reads.find(r => r.source==="diag" && r.label==="unmatched");
  if (diagM && diagU) V.compare = [diagM.id, diagU.id];
  let start = (want && want.read && data.byId[want.read]) ? data.byId[want.read] : (diagM || reads.find(r => r.rows && r.rows.length) || reads[0]);
  selectRead(start.id, want && want.pos!=null ? +want.pos : null);
}

function defaultPos(r){
  if (!r.positions || !r.positions.length) return null;
  if (r.fork_pos!=null && r.rowByPos[r.fork_pos]) return r.fork_pos;
  const reply = (r.rows||[]).find(row => row.region === "reply");
  return reply ? reply.pos : r.positions[0];
}

function ensureAgentRows(read){
  // deferred agent reads live in data/<key>.agent.json; fetch it once and merge rows by id
  const data = V.data;
  if (!read.deferred || read.rows) return Promise.resolve();
  if (!data.agentPromise){
    data.agentPromise = fetchJson(data.agent_file).then(blob => {
      for (const full of (blob.reads||[])){ const stub = data.byId[full.id]; if (stub){ stub.rows = full.rows; stub.layers = full.layers || stub.layers; stub.deferred = false; indexRead(stub); } }
    });
  }
  return data.agentPromise;
}

function selectRead(id, pos){
  if (!V) return;
  const r = V.data.byId[id]; if (!r) return;
  V.readId = id;
  if (!V.compare.includes(id)) V.compare = [id, ...V.compare];
  const needs = [r, ...V.compare.map(c => V.data.byId[c]).filter(Boolean)].filter(x => x.deferred && !x.rows);
  if (needs.length){
    V.pos = pos;
    renderChips();
    document.getElementById("left").innerHTML = `<p class="none">loading the agent's reads (<span class="kkey">${esc(V.data.agent_file)}</span>) …</p>`;
    document.getElementById("right").innerHTML = "";
    const key = V.key;
    Promise.all(needs.map(ensureAgentRows)).then(() => { if (V && V.key === key && V.readId === id) selectRead(id, V.pos); })
      .catch(err => { if (V && V.key === key) document.getElementById("left").innerHTML = failNotice(V.data.agent_file, err && err.message ? err.message : String(err)); });
    return;
  }
  if (pos==null || !r.rowByPos || !r.rowByPos[pos]) pos = defaultPos(r);
  V.pos = pos;
  renderChips(); renderLeft(); renderRight();
  try { history.replaceState(null, "", patUrl(V.key, id, pos)); } catch(e){}
}

function toggleCompare(id){
  if (id === V.readId) return;
  V.compare = V.compare.includes(id) ? V.compare.filter(x => x !== id) : [...V.compare, id];
  const r = V.data.byId[id];
  if (r && r.deferred && !r.rows) return selectRead(V.readId, V.pos);  // loads, then re-renders
  renderChips(); renderRight();
}

function readDot(r){ return r.parse_error ? "err" : (r.source==="diag" ? r.label : (r.label==="agent" ? "agent" : r.label)); }
function readTitle(r){
  const parts = [];
  if (r.source==="diag") parts.push(`diagnostic pass · ${r.label} rollout · sample ${r.sample_index}`);
  else parts.push(`agent read · tool call #${r.tool_index} · conversation ${r.conv_id}` + (r.same_rollout_as ? ` (same rollout as ${r.same_rollout_as}, different lens sample)` : ""));
  if (r.parse_error) parts.push("could not parse: " + r.parse_error);
  return parts.join(" · ");
}

function renderChips(){
  const box = document.getElementById("recs"); if (!box) return;
  const reads = V.data.reads;
  let h = `<div class="pick"><span class="lbl">read</span>` + reads.map(r =>
    `<button class="chip" data-read="${esc(r.id)}" aria-pressed="${r.id===V.readId}" title="${esc(readTitle(r))}"><span class="g ${readDot(r)}"></span>${esc(r.id)}` +
    `<span class="n">${r.parse_error ? "⚠ unparsed" : (r.positions && r.positions.length ? r.positions.length + " pos" : (r.deferred ? "lazy" : "0 pos"))}${r.tool_index!=null ? " · #"+r.tool_index : ""}</span></button>`).join("") + `</div>`;
  h += `<div class="pick"><span class="lbl">compare</span>` + reads.filter(r => !r.parse_error).map(r =>
    `<button class="tab" data-cmp="${esc(r.id)}" aria-pressed="${V.compare.includes(r.id)}" ${r.id===V.readId?"disabled":""} title="${r.id===V.readId?"the selected read is always a column":"toggle this read as a column in the layer table"}">${esc(r.id)}</button>`).join("") +
    `<span class="hint" style="margin-left:6px">columns of the layer table, side by side at the selected position; the selected read is always included</span></div>`;
  box.innerHTML = h;
  box.querySelectorAll("[data-read]").forEach(b => b.onclick = () => selectRead(b.dataset.read, null));
  box.querySelectorAll("[data-cmp]").forEach(b => b.onclick = () => toggleCompare(b.dataset.cmp));
}

// ---- left: the token strip of the selected read
const REGION_LABEL = {user: "user — the model is reading the request", header: "about to answer — identical for every rollout of this prompt", reply: "reply — the model is writing its answer", system: "system"};
function tokHtml(s){
  if (s === "") return `<span class="special">∅</span>`;
  if (/^<\|.*\|>$/.test(s)) return `<span class="special">${esc(s)}</span>`;
  return esc(s).replace(/\n/g, `<span class="nl">⏎</span>`).replace(/\t/g, `<span class="nl">⇥</span>`);
}
function flagPositions(r){
  const set = new Set();
  for (const m of V.data.mechs) for (const c of citations(m.readout_cells)) if (c.pos!=null && c.cid && r.mention_ids.includes(c.cid)) set.add(c.pos);
  return set;
}
function renderLeft(){
  const r = V.data.byId[V.readId], left = document.getElementById("left"); if (!left) return;
  let h = `<div class="recmeta"><span class="badge ${readDot(r)}">${esc(r.source==="diag" ? r.label + " rollout" : "agent read · " + r.label)}</span>` +
    `<span class="hint">${esc(r.id)} · ${(r.positions||[]).length} positions read · ${(r.layers||[]).length} layers` +
    (r.claimed && r.claimed.positions!=null && r.claimed.positions !== (r.positions||[]).length ? ` (page header says ${r.claimed.positions})` : "") +
    (r.tool_index!=null ? ` · tool call #${r.tool_index}` : "") +
    (r.same_rollout_as ? ` · same rollout as <span class="kkey">${esc(r.same_rollout_as)}</span>, different lens sample` : "") + `</span></div>`;
  if (r.parse_error) h += `<div class="notice"><b>could not parse this readout page:</b> ${esc(r.parse_error)}</div>`;
  h += `<div class="legend"><span><i style="outline:1px solid var(--swept);outline-offset:-1px"></i>read position (click)</span><span><i style="background:var(--mark)"></i>cited by a mechanism</span><span><i style="background:var(--sel)"></i>selected</span>` +
    (r.fork_pos!=null ? `<span><i style="box-shadow:0 2px 0 var(--ink)"></i>fork: first word where the matched and unmatched replies diverge (≈, from the read tokens)</span>` : "") +
    `<span><i></i>‥ = positions not read</span><span>keys <kbd>←</kbd> <kbd>→</kbd></span></div>`;
  // the exact text that was read, clipped
  const msgs = (r.messages||[]).map(m => `<div class="block"><div class="role"><span>${esc(m.role)}</span></div><pre class="clip">${esc(m.content)}</pre></div>`).join("");
  h += `<details><summary>text that was read (${r.text_source==="tool page" ? "from the agent's page, clipped to 900 chars" : "full, " + esc(r.text_source)})</summary>${msgs}` +
    `<div class="block reply"><div class="role"><span>assistant reply</span></div><pre class="clip">${esc(r.completion||"(empty)")}</pre></div></details>`;
  // blocks by region, in position order
  const rows = r.rows || [], flags = flagPositions(r);
  if (!rows.length) h += `<p class="none">no positions in this read.</p>`;
  let i = 0;
  while (i < rows.length){
    const region = rows[i].region; let inner = ""; let prev = null;
    for (; i < rows.length && rows[i].region === region; i++){
      const row = rows[i];
      if (prev!=null && row.pos - prev > 1) inner += `<span class="gap" title="${row.pos - prev - 1} positions not read">‥</span>`;
      prev = row.pos;
      const cls = ["tok","swept", flags.has(row.pos)?"flag":"", row.pos===V.pos?"sel":"", row.pos===r.fork_pos?"r1":""].join(" ");
      inner += `<span class="${cls}" data-pos="${row.pos}" title="position ${row.pos} · ${esc(row.kind)}">${tokHtml(row.tok)}</span>`;
    }
    h += `<div class="block ${esc(region)}"><div class="role"><span>${esc(region)}</span><span class="rl">${esc(REGION_LABEL[region]||"")}</span></div><pre>${inner}</pre></div>`;
  }
  left.innerHTML = h;
  left.querySelectorAll(".tok.swept").forEach(el => el.onclick = () => selectRead(V.readId, +el.dataset.pos));
  const sel = left.querySelector(".tok.sel"); if (sel && sel.scrollIntoView) sel.scrollIntoView({block:"nearest"});
}

// ---- citations inside a mechanism's readout_cells: "w001u L36 pos 87 …" → [{cid, pos}]
function citations(text){
  const out = []; let cid = null;
  const re = /\b([wc]\d+[mu]?)\b|\bpos\s*(\d+)/g; let m;
  text = text || "";
  while ((m = re.exec(text))){
    if (m[1]){ if (V.data.byConv[m[1]]) cid = m[1]; out.push({cid: V.data.byConv[m[1]] ? m[1] : null, pos: null, index: m.index, len: m[0].length}); }
    else out.push({cid, pos: +m[2], index: m.index, len: m[0].length});
  }
  return out;
}
function mechsAt(r, pos){
  const out = [];
  for (const m of V.data.mechs){
    const here = citations(m.readout_cells).filter(c => c.pos === pos && c.cid && r.mention_ids.includes(c.cid));
    if (here.length) out.push({m, via: here[0].cid});
  }
  return out;
}
function markQuotes(text, quotes){
  let html = esc(text), hit = false;
  for (const q of quotes){ const e = esc(q); if (html.includes(e)){ html = html.split(e).join(`<mark>${e}</mark>`); hit = true; } }
  return [html, hit];
}

// ---- right: the selected position, every layer, one column per compared read
function renderRight(){
  const r = V.data.byId[V.readId], right = document.getElementById("right"); if (!right) return;
  const p = V.pos, row = r.rowByPos ? r.rowByPos[p] : null;
  if (row == null){ right.innerHTML = `<p class="none">${r.parse_error ? "nothing to show — this page did not parse." : "no position selected."}</p>`; return; }
  const idx = r.positions.indexOf(p);
  const where = row.region === "header" ? "about to answer — the chat boundary, identical for every rollout" : row.region === "user" ? "reading the request; nothing written yet" : row.region === "reply" ? "writing the answer" : row.region;
  let h = `<div class="poshdr"><h2>position ${p}</h2><span class="tokbox">${esc(JSON.stringify(row.tok))}</span><span class="hint">${esc(where)} · ${esc(row.kind)}${p===r.fork_pos?" · fork":""}</span>` +
    `<span class="navb"><button id="prev" ${idx<=0?"disabled":""}>← prev</button><button id="next" ${idx>=r.positions.length-1?"disabled":""}>next →</button></span></div>`;
  // text so far: the read tokens up to and including this one (thinned, so approximate)
  let sofar = ""; for (const x of r.rows){ if (x.pos > p) break; if (x.pos < p) sofar += x.tok; }
  h += `<div class="local">${esc(sofar.slice(-200))}<b>${esc(row.tok)}</b><span class="hint">   ← read tokens so far (this token last; only read positions, so gaps are elided)</span></div>`;
  // mechanisms citing this read at this position (the flags analogue)
  const fl = mechsAt(r, p);
  h += `<div class="flags"><h3>mechanisms citing this position (${fl.length}) — unverified hypotheses</h3>`;
  if (!fl.length) h += `<p class="none">none — no reported mechanism cites <span class="kkey">${esc(r.mention_ids.join(" / ")||r.id)}</span> at pos ${p}.</p>`;
  for (const f of fl) h += `<div class="flag ${f.via===r.conv_id?"":"other"}"><span class="cat">mechanism ${f.m.i+1}</span><span class="cell">${esc(f.m.auditor)} s${f.m.seed}${f.m.confidence==null?"":" · confidence "+num(f.m.confidence)}${f.via!==r.conv_id?" · cited on "+esc(f.via)+" (same rollout, different lens sample)":""}</span><div class="why">${esc(f.m.mechanism)}</div><q>${esc(cut(f.m.readout_cells, 400))}</q></div>`;
  h += `</div>`;
  // the layer table
  const cols = V.compare.map(id => V.data.byId[id]).filter(Boolean);
  const layers = [...new Set(cols.reduce((acc, c) => acc.concat(c.layers||[]), []))].sort((a,b) => a-b);
  const toks = cols.map(c => c.rowByPos && c.rowByPos[p] ? c.rowByPos[p].tok : null);
  const present = toks.filter(t => t!=null);
  const identical = cols.length > 1 && present.length === cols.length && present.every(t => t === present[0]);
  const differ = cols.length > 1 && !identical;
  h += `<div class="grid"><h3>every readout at this token (verbatim, unedited) — ${cols.length} read${cols.length===1?"":"s"}</h3>`;
  if (identical) h += `<div class="notice">identical prefix — same activation, different lens samples${row.region==="reply"?" (the replies still agree at this position)":""}</div>`;
  h += `<table class="reads"><thead><tr><th>layer</th>`;
  cols.forEach((c, j) => { h += `<th class="${esc(readDot(c))}">${esc(c.id)}${differ ? `<span class="own">${toks[j]==null ? "not read at pos "+p : esc(JSON.stringify(toks[j]))}</span>` : ""}</th>`; });
  h += `</tr></thead><tbody>`;
  for (const L of layers){
    h += `<tr><td class="L">L${L}</td>`;
    for (const c of cols){
      const li = (c.layers||[]).indexOf(L), crow = c.rowByPos ? c.rowByPos[p] : null;
      const samples = (crow && li >= 0 && crow.samples) ? (crow.samples[li] || []) : null;
      h += `<td>`;
      if (samples === null) h += `<span class="none">${crow ? "—" : "not read at this position"}</span>`;
      else if (!samples.length) h += `<span class="none">—</span>`;
      else for (const s of samples){ const [sh, hit] = markQuotes(s.trim(), V.data.quotes); h += `<div class="samp ${hit?"hit":""}">${sh}</div>`; }
      h += `</td>`;
    }
    h += `</tr>`;
  }
  h += `</tbody></table></div>`;
  if (V.data.quotes.length) h += `<p class="hint" style="margin-top:8px"><mark>marked</mark> = text the agent quoted in a mechanism's evidence or cell citation (fragments of ≥ 25 chars, verbatim).</p>`;
  right.innerHTML = h;
  right.scrollTop = 0;
  const pv = document.getElementById("prev"), nx = document.getElementById("next");
  if (pv) pv.onclick = () => selectRead(r.id, r.positions[idx-1]);
  if (nx) nx.onclick = () => selectRead(r.id, r.positions[idx+1]);
}

document.addEventListener("keydown", e => {
  if (!V || !V.readId) return;
  if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  const r = V.data.byId[V.readId]; if (!r || !r.positions) return;
  const idx = r.positions.indexOf(V.pos);
  if (e.key === "ArrowLeft" && idx > 0){ e.preventDefault(); selectRead(r.id, r.positions[idx-1]); }
  if (e.key === "ArrowRight" && idx >= 0 && idx < r.positions.length-1){ e.preventDefault(); selectRead(r.id, r.positions[idx+1]); }
});

// ---- below the viewer: contrast rollouts, transcript, ranked mechanisms
function rollout(s){
  const cls = s.matched ? "matched" : "unmatched";
  return `<div class="roll ${cls}"><div class="hd"><span>${s.matched?"matched":"unmatched"}</span>` +
    `<span class="kkey">sample ${s.sample_index==null?"?":s.sample_index}</span></div>` +
    expandable(s.text, 800, "show full rollout") + `</div>`;
}
function linkCells(text, data){
  // every "pos N" that resolves to a read of this pattern becomes a link selecting read@position
  let out = "", last = 0, cid = null;
  const re = /\b([wc]\d+[mu]?)\b|\bpos\s*(\d+)/g; let m;
  text = text || "";
  while ((m = re.exec(text))){
    out += esc(text.slice(last, m.index));
    if (m[1]){
      const r = data.byConv[m[1]]; if (r) cid = m[1];
      out += r ? `<a href="${patUrl(data.key, r.id, null)}" title="${esc(readTitle(r))}">${esc(m[0])}</a>` : esc(m[0]);
    } else {
      const r = cid ? data.byConv[cid] : null, pos = +m[2];
      const ok = r && (r.deferred || (r.rowByPos && r.rowByPos[pos]));
      out += ok ? `<a href="${patUrl(data.key, r.id, pos)}" title="open ${esc(r.id)} at position ${pos}">${esc(m[0])}</a>` : esc(m[0]);
    }
    last = m.index + m[0].length;
  }
  return out + esc(text.slice(last));
}
function stepBlock(s, i, data){
  const rd = s.tool_index!=null ? data.byTool[s.tool_index] : null;
  let h = `<div class="step ${esc(s.name||"")}"><div class="hd"><span class="nm">${esc(s.name||"assistant")}</span><span>#${i+1}${s.meta?" · "+esc(s.meta):""}</span>` +
    (rd ? `<a href="${patUrl(data.key, rd.id, null)}" data-open="${esc(rd.id)}">open this read → ${esc(rd.id)}${rd.parse_error?" (unparsed)":""}</a>` : "") + `</div>`;
  if (s.think) h += `<div class="think">${esc(s.think)}</div>`;
  if (s.name === "chat") h += `<div class="call"><span class="k">user →</span> <span class="usr">${esc(s.user)}</span></div>` + (s.args ? `<div class="call"><span class="k">${esc(s.args)}</span></div>` : "");
  else if (s.name) h += `<div class="call">${esc(s.name)}(${esc(s.args)})</div>`;
  if (s.output){
    const o = s.output, isRead = s.name === "readouts";
    const label = s.name === "chat" ? "reply" : "result";
    h += `<pre class="block" style="margin-top:6px">${esc(o.slice(0,600))}${o.length>600?" …":""}</pre>`;
    if (o.length > 600) h += `<details><summary>show full ${label} (${o.length} chars${isRead?", clipped for the viewer; the structured read is in the strip above":""})</summary><pre class="block">${esc(o)}</pre></details>`;
  }
  return h + `</div>`;
}
function mechBlock(m, i, data){
  let h = `<div class="mech"><div class="txt"><span class="n">${i+1}.</span>${esc(m.mechanism)}` +
    (m.confidence==null ? "" : ` <span class="conf">confidence ${num(m.confidence)}</span>`) + `</div>`;
  if (m.evidence) h += `<div class="ev">${esc(m.evidence)}</div>`;
  if (m.readout_cells) h += `<div class="cells">cells: ${linkCells(m.readout_cells, data)}</div>`;
  if (m.would_test_by) h += `<div class="test"><b>would test by — not run in this pass:</b> ${esc(m.would_test_by)}</div>`;
  return h + `</div>`;
}
function runBlock(r, data){
  let h = `<div class="card"><b>${esc(r.auditor)}</b> <span class="kkey">seed ${r.seed==null?"?":r.seed}</span>` +
    ` <span class="crumb">· ${(r.steps||[]).length} steps` +
    (r.cells_served ? ` · ${r.cells_served} cells served` : "") + (r.output_tokens ? ` · ${r.output_tokens} tok` : "") +
    (r.server_calls ? ` · ${r.server_calls} server calls` : "") + (r.server_seconds ? ` · ${num(r.server_seconds,1)}s server` : "") +
    (r.stopped_by ? ` · stopped: ${esc(r.stopped_by)}` : "") + `</span>`;
  if (r.summary) h += `<div style="margin-top:6px">${esc(r.summary)}</div>`;
  h += `</div>`;
  h += `<h3>transcript</h3>`;
  if (!(r.steps||[]).length) h += `<p class="none">no turns recorded.</p>`;
  (r.steps||[]).forEach((s,i) => { h += stepBlock(s, i, data); });
  if ((r.notes||[]).length){
    h += `<h3>scratch notes (${r.notes.length})</h3>`;
    r.notes.forEach(n => { h += `<pre class="block small" style="margin-bottom:6px">${esc(n)}</pre>`; });
  }
  h += `<h3>ranked mechanisms (${(r.mechanisms||[]).length}) — unverified hypotheses</h3>`;
  if (!(r.mechanisms||[]).length) h += `<p class="none">this run reported no mechanisms.</p>`;
  (r.mechanisms||[]).forEach((m,i) => { h += mechBlock(m, i, data); });
  return h;
}
function belowViewer(data){
  let h = `<h2>Contrast rollouts</h2>`;
  const mt = data.samples.filter(s => s.matched), um = data.samples.filter(s => !s.matched);
  if (!data.samples.length) h += `<p class="none">no samples recorded.</p>`;
  else h += `<div class="cols"><div><h3>matched (${mt.length})</h3>${mt.length ? mt.map(rollout).join("") : `<p class="none">none</p>`}</div>` +
    `<div><h3>unmatched (${um.length})</h3>${um.length ? um.map(rollout).join("") : `<p class="none">none</p>`}</div></div>`;
  if (data.fork && (data.fork.prefix_chars!=null || data.fork.note))
    h += `<div class="card"><b>fork</b> <span class="kkey">shared prefix ${data.fork.prefix_words==null?"?":data.fork.prefix_words} words / ${data.fork.prefix_chars==null?"?":data.fork.prefix_chars} chars</span>` +
      (data.fork.note ? `<div style="margin-top:4px">${esc(data.fork.note)}</div>` : "") + `</div>`;
  h += `<h2>Agent runs</h2>`;
  if (!data.runs.length) h += `<p class="none">no agent runs for this pattern yet.</p>`;
  data.runs.forEach(r => { h += runBlock(r, data); });
  return h;
}
function wireBelow(data){
  // "open this read" jumps to the viewer as well as selecting the read
  document.querySelectorAll("#heavy [data-open]").forEach(a => a.onclick = ev => {
    ev.preventDefault(); selectRead(a.dataset.open, null);
    const m = document.querySelector("main.two"); if (m && m.scrollIntoView) m.scrollIntoView({block:"start", behavior:"smooth"});
  });
}

// ------------------------------------------------------------------- cluster
function renderCluster(i){
  const k = (D.clusters||[])[i];
  if (!k) return `<p class="none">no cluster ${esc(i)}.</p>`;
  let h = `<h2 style="margin-top:4px">${esc(k.name)}</h2>`;
  if (k.description) h += `<div class="card">${esc(k.description)}</div>`;
  h += `<h3>behaviors spanned (${k.behavior_ids.length})</h3><div>` +
    (k.behavior_ids.length ? k.behavior_ids.map(b => `<span class="pill">${esc(b)}</span>`).join("") : `<span class="none">none listed</span>`) + `</div>`;
  h += `<h3>members (${k.members.length}) — unverified mechanisms</h3>`;
  if (!k.members.length) h += `<p class="none">no members listed.</p>`;
  for (const m of k.members){
    const p = byKey[m.pattern_key];
    h += `<div class="mech"><div class="txt">` +
      `<a href="${patUrl(m.pattern_key)}">${esc(p ? p.behavior_name : m.pattern_key)}</a>` +
      ` <span class="kkey">${esc(m.pattern_key)}</span>${p?"":` <span class="tag n">not in this build</span>`}</div>` +
      (m.mechanism ? `<div class="ev">${esc(m.mechanism)}</div>` : "") +
      (m.evidence ? `<div class="cells">${esc(m.evidence)}</div>` : "") + `</div>`;
  }
  return h;
}

// --------------------------------------------------------------------- route
function parseHash(){
  const raw = (location.hash || "#/").replace(/^#\/?/, "");
  const [pathPart, query] = raw.split("?");
  const parts = pathPart.split("/");
  const q = {};
  (query||"").split("&").forEach(kv => { if (!kv) return; const [k, v] = kv.split("="); q[decodeURIComponent(k)] = v==null ? "" : decodeURIComponent(v); });
  return {parts, q};
}
function route(){
  const {parts, q} = parseHash();
  const app = document.getElementById("app"), crumb = document.getElementById("crumb");
  if (parts[0] === "pattern" && parts.length > 1){
    const key = decodeURIComponent(parts.slice(1).join("/"));
    const want = {read: q.read || null, pos: q.pos!=null && q.pos!=="" ? +q.pos : null};
    if (V && V.key === key && document.getElementById("heavy") && document.getElementById("heavy").dataset.key === key){
      // same page: a deep link from a mechanism or transcript — just move the selection
      if (want.read && V.data.byId[want.read]){ selectRead(want.read, want.pos); const m = document.querySelector("main.two"); if (m && m.scrollIntoView) m.scrollIntoView({block:"start"}); }
      return;
    }
    V = null;
    crumb.innerHTML = `<a href="#/">overview</a> › pattern <span class="kkey">${esc(key)}</span>`;
    app.innerHTML = renderPattern(key);
    window.scrollTo(0, 0);
    if (byKey[key]) loadPattern(key, want);
    return;
  }
  V = null;
  if (parts[0] === "cluster" && parts.length > 1){
    const i = parseInt(parts[1], 10), k = (D.clusters||[])[i];
    crumb.innerHTML = `<a href="#/">overview</a> › cluster ${k ? esc(k.name) : esc(parts[1])}`;
    app.innerHTML = renderCluster(i);
  } else {
    crumb.innerHTML = "overview";
    app.innerHTML = renderOverview();
  }
  app.querySelectorAll("tr[data-go]").forEach(tr => { tr.onclick = () => { location.hash = tr.dataset.go; }; });
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);
route();
</script>
"""
TEMPLATE = TEMPLATE.replace("__BANNER__", BANNER)


if __name__ == "__main__":
    main()
