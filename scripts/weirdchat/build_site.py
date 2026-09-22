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
    }
    seen: set[tuple[str, str, int, int, str]] = set()
    for pat in patterns:
        by_conv = reads_by_conv(pat)
        for run in pat["runs"]:
            for mi, mech in enumerate(run["mechanisms"]):
                text = mech["readout_cells"]
                for start, quote in quoted_fragments(text):
                    stats["fragments"] += 1
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


# plain text (no quotes/backslashes): it is injected into a JS string and set via textContent
BANNER = (
    "No ground truth, no interventions. The agent was told the behavior and asked why the model "
    "does it; every mechanism below is an UNVERIFIED HYPOTHESIS read off chat probes and OLens "
    "readouts. Nothing on this page has been causally tested."
)

TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WeirdChat × OLens: why Qwen3.6-27B does it</title>
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
header{padding:18px 28px 12px;border-bottom:1px solid var(--line);background:var(--card)}
h1{font-family:var(--serif);font-weight:600;font-size:24px;margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}
h1 a{text-decoration:none}
.sub{color:var(--ink2);max-width:100ch;margin:0}
.sub code{font-family:var(--mono);font-size:12.5px;background:var(--line2);padding:0 4px;border-radius:3px}
h2{font-family:var(--serif);font-weight:600;font-size:19px;margin:22px 0 8px;letter-spacing:-.01em}
h3{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:16px 0 6px;font-weight:500}
.crumb{font-size:12.5px;color:var(--mute)} .crumb a{color:var(--ink2)}
.banner{background:var(--claims-bg);color:var(--claims);border:1px solid var(--claims);border-radius:6px;padding:8px 12px;margin:10px 0 0;font-size:12.5px;max-width:120ch}
.banner b{letter-spacing:.04em}
.page{padding:18px 28px 80px;max-width:1500px}
.kkey{font-family:var(--mono);font-size:11.5px;color:var(--ink2)}
.hint{color:var(--mute);font-size:12.5px}
.none{color:var(--mute);font-size:13px}
kbd{font-family:var(--mono);font-size:11px;border:1px solid var(--line);border-radius:3px;padding:0 4px;background:var(--card)}
table.t{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:6px 0 10px}
table.t th{text-align:left;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute);font-weight:500;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
table.t td{padding:7px 10px;border-bottom:1px solid var(--line2);vertical-align:top;font-size:13px}
table.t tr:last-child td{border-bottom:none}
table.t tr.click{cursor:pointer} table.t tr.click:hover td{background:var(--s3d-bg)}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:12px;white-space:nowrap}
.tag{font-size:11px;border-radius:4px;padding:1px 6px;font-family:var(--mono)}
.tag.y{background:var(--discl-bg);color:var(--discl)} .tag.n{background:var(--line2);color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;margin:0 0 10px}
pre.block{font-family:var(--mono);font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;color:var(--ink)}
pre.small{font-size:11.5px;color:var(--ink2);border:none;padding:0;background:none}
details{margin:6px 0} summary{cursor:pointer;color:var(--s3d);font-size:12.5px} summary::marker{color:var(--mute)}
details.alltbl{margin:10px 0 0} details.alltbl summary{color:var(--mute)}
.pill{display:inline-block;font-size:11.5px;color:var(--ink2);background:var(--line2);border-radius:999px;padding:1px 9px;margin:0 5px 5px 0;font-family:var(--mono)}
/* --- pickers, as in the sweep viewer --- */
.recs{display:flex;flex-direction:column;gap:8px;margin-top:12px}
.pick{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.pick .lbl{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);min-width:64px}
.tab{font:inherit;font-family:var(--mono);font-size:12px;border:1px solid var(--line);background:var(--card);border-radius:5px;padding:3px 9px;cursor:pointer;color:var(--ink2)}
.tab:hover{border-color:var(--ink2)} .tab[aria-pressed="true"]{border-color:var(--ink);color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
.tab:disabled{opacity:.55;cursor:default}
.tab .cnt{color:var(--mute);margin-left:4px}
.chip{border:1px solid var(--line);background:var(--card);border-radius:999px;padding:3px 10px;font-size:12.5px;cursor:pointer;display:inline-flex;gap:6px;align-items:center;font-family:var(--sans);color:var(--ink2);max-width:100%}
.chip:hover{border-color:var(--ink2)} .chip[aria-pressed="true"]{border-color:var(--ink);color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
.chip .g{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.g.run{background:var(--claims)} .g.diag{background:var(--silent)} .g.data{background:var(--swept)}
.g.matched{background:var(--claims)} .g.unmatched{background:var(--discl)} .g.agent{background:var(--swept)} .g.err{background:var(--silent)}
.chip .n{color:var(--mute);font-family:var(--mono);font-size:11.5px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:6px 0 12px}
.legend span i{display:inline-block;width:14px;height:12px;vertical-align:-2px;margin-right:5px;border-radius:2px;background:var(--card)}
.legend span i.dot{width:8px;height:8px;border-radius:50%;vertical-align:0}
/* --- highlights --- */
#hl{padding:14px 28px 6px;border-bottom:1px solid var(--line);background:var(--paper)}
.hlhead{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}
.hlhead h2{font-family:var(--serif);font-weight:600;font-size:18px;margin:0}
.hlhead button{margin-left:auto}
.hlgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}
.hlgrid[hidden]{display:none}
.hlc{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--claims);border-radius:6px;padding:10px 12px;cursor:pointer;display:flex;flex-direction:column;gap:6px}
.hlc:hover{border-color:var(--ink2);border-left-color:inherit}
.hlc.header{border-left-color:var(--silent)} .hlc.user{border-left-color:var(--discl)}
.hlc .where{font-family:var(--mono);font-size:11.5px;color:var(--mute);display:flex;gap:8px;flex-wrap:wrap}
.hlc .where b{color:var(--ink2);font-weight:500}
.hlc .q{font-family:var(--mono);font-size:12.5px;color:var(--ink);line-height:1.45;white-space:pre-wrap;word-break:break-word}
.hlc .loc{font-family:var(--mono);font-size:11.5px;color:var(--ink2);white-space:pre-wrap;word-break:break-word}
.hlc .loc b{background:var(--mark);font-weight:500}
.hlc .n{font-size:12.5px;color:var(--ink2)}
.hlc .hand{font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--s3d)}
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
.tok.fork{border-bottom:2px dotted var(--ink)}
.special{color:var(--mute)}
.nl{color:var(--mute);font-size:10.5px}
.gap{color:var(--mute);font-size:10px;letter-spacing:-1px;padding:0 2px;cursor:default}
.poshdr{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.poshdr h2{margin:0 0 4px}
.poshdr .navb{margin-left:auto;display:flex;gap:4px}
.navb button{font:inherit;background:var(--card);border:1px solid var(--line);border-radius:4px;padding:2px 9px;cursor:pointer;color:var(--ink)}
.navb button:hover{border-color:var(--ink2)} .navb button:disabled{opacity:.4;cursor:default}
.tokbox{font-family:var(--mono);background:var(--sel);color:var(--selfg);padding:1px 6px;border-radius:3px;font-size:12.5px}
.where{font-size:13px;color:var(--ink2);margin:4px 0 0}
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
.en{color:var(--ink2);font-style:italic;margin-top:2px;font-family:var(--sans);font-size:12px}
.en::before{content:"EN  ";font-style:normal;font-size:10px;letter-spacing:.06em;color:var(--mute)}
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
  <h1><a href="#/">WeirdChat × OLens: why Qwen3.6-27B does it</a></h1>
  <p class="sub">An investigator agent (Claude Opus 5) was told each WeirdChat-catalogued behavior of Qwen3.6-27B and asked <em>why</em> the model does it, with chat probes plus an OLens readout of the model's own activations. Pick a behavior, then one of its prompts; click any read token to see what the lens decoded there at every layer, side by side across reads.</p>
  <div class="banner"><b>UNVERIFIED HYPOTHESES.</b> <span id="banner"></span></div>
  <div class="recs" id="recs"></div>
  <details class="alltbl"><summary>every pattern as a table</summary><div id="alltbl"></div></details>
</header>
<section id="hl">
  <div class="hlhead"><h2>Highlights</h2><span class="hint">lens cells a mechanism quotes that verify verbatim against the read at build time — click one to jump to that pattern, read and position. Red = inside the reply, amber = about to answer (header), green = inside the user turn.</span><button id="hltoggle" class="tab">collapse</button></div>
  <div id="hlgrid" class="hlgrid"></div>
</section>
<div class="page" id="app"></div>

<script>
const D = /*__DATA__*/;
const BANNER = "__BANNER__";
const esc = s => (s==null?"":String(s)).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const num = (v,d) => v==null ? "—" : Number(v).toFixed(d==null?2:d);
const pct = v => v==null ? "—" : Math.round(Number(v)*100) + "%";
const byKey = {}; (D.patterns||[]).forEach(p => byKey[p.key] = p);
const GLABEL = {run: "agent run with mechanisms", diag: "diagnostics only", data: "data only"};
document.getElementById("banner").textContent = BANNER;

function cut(s, n){ s = s==null ? "" : String(s); return s.length<=n ? s : s.slice(0,n) + " …"; }
function dataUrl(key){ return "data/" + encodeURIComponent(key) + ".json"; }
function patUrl(key, read, pos){
  let h = "#/pattern/" + encodeURIComponent(key);
  if (read) h += "?read=" + encodeURIComponent(read) + (pos==null ? "" : "&pos=" + pos);
  return h;
}
function firstKey(){
  const b = (D.behaviors||[])[0]; if (!b) return null;
  const p = (D.patterns||[]).find(p => p.behavior_id === b.behavior_id) || D.patterns[0];
  return p ? p.key : null;
}

// text shown up to `n` chars, with the remainder behind a disclosure
function expandable(text, n, label){
  const t = text==null ? "" : String(text);
  if (!t) return `<p class="none">(empty)</p>`;
  if (t.length <= n) return `<pre class="block">${esc(t)}</pre>`;
  return `<pre class="block">${esc(t.slice(0,n))} …</pre>` +
    `<details><summary>${esc(label||"show full")} (${t.length} chars)</summary><pre class="block">${esc(t)}</pre></details>`;
}

// ------------------------------------------------------------ header pickers
function renderPickers(curKey){
  const box = document.getElementById("recs"); if (!box) return;
  const cur = curKey ? byKey[curKey] : null;
  const bs = D.behaviors || [];
  const curB = cur ? cur.behavior_id : (bs[0] ? bs[0].behavior_id : null);
  let h = `<div class="pick"><span class="lbl">behavior</span>` + bs.map(b =>
    `<button class="tab" data-beh="${esc(b.behavior_id)}" aria-pressed="${b.behavior_id===curB}">${esc(b.behavior_name)}<span class="cnt">${b.n_patterns}</span></button>`).join("") +
    `<a class="hint" href="#/themes" style="margin-left:auto">themes (${(D.clusters||[]).length} mechanism clusters) →</a></div>`;
  const pats = (D.patterns||[]).filter(p => p.behavior_id === curB);
  h += `<div class="pick"><span class="lbl">prompt</span>` + pats.map(p =>
    `<button class="chip" data-key="${esc(p.key)}" aria-pressed="${cur && p.key===cur.key}" title="${esc(p.key)} · ${esc(GLABEL[p.grade]||p.grade)}"><span class="g ${esc(p.grade)}"></span>${esc(cut(p.group_summary || p.key, 70))}<span class="n">${pct(p.published_match_rate)}</span></button>`).join("") + `</div>`;
  h += `<div class="legend"><span><i class="dot" style="background:var(--claims)"></i>agent run with ≥1 mechanism</span><span><i class="dot" style="background:var(--silent)"></i>diagnostics only</span><span><i class="dot" style="background:var(--swept)"></i>data only</span><span class="hint">% = WeirdChat's published match rate for the prompt</span></div>`;
  box.innerHTML = h;
  box.querySelectorAll("[data-beh]").forEach(b => b.onclick = () => { const p = (D.patterns||[]).find(p => p.behavior_id === b.dataset.beh); if (p) location.hash = patUrl(p.key); });
  box.querySelectorAll("[data-key]").forEach(b => b.onclick = () => { location.hash = patUrl(b.dataset.key); });
}

function renderOverviewTables(){
  const box = document.getElementById("alltbl"); if (!box) return;
  const c = D.counts || {};
  let h = `<p class="hint"><b>${c.patterns||0}</b> patterns · <b>${c.behaviors||0}</b> behaviors · <b>${c.runs||0}</b> agent runs · <b>${c.with_diag||0}</b> with OLens diagnostics · <b>${c.reads||0}</b> lens reads · <b>${c.mechanisms||0}</b> reported mechanisms · source <span class="kkey">${esc(D.source)}</span></p>`;
  const bs = D.behaviors || [];
  if (!bs.length) return box.innerHTML = h + `<p class="none">no patterns found.</p>`;
  h += `<table class="t"><thead><tr><th>behavior</th><th>id</th><th class="num">patterns</th><th class="num">mean published match</th><th class="num">agent runs</th></tr></thead><tbody>`;
  for (const b of bs) h += `<tr><td>${esc(b.behavior_name)}</td><td class="kkey">${esc(b.behavior_id)}</td><td class="num">${b.n_patterns}</td><td class="num">${num(b.mean_match_rate)}</td><td class="num">${b.n_runs}</td></tr>`;
  h += `</tbody></table>`;
  h += `<table class="t"><thead><tr><th>behavior</th><th>group summary</th><th class="num">match</th><th class="num">elo</th><th>grade</th><th>reads</th><th>runs</th><th class="num">mechanisms</th></tr></thead><tbody>`;
  for (const p of D.patterns)
    h += `<tr class="click" data-go="${patUrl(p.key)}"><td>${esc(p.behavior_name)}<div class="kkey">${esc(p.key)}</div></td><td>${esc(cut(p.group_summary, 160))}</td>` +
      `<td class="num">${num(p.published_match_rate)}</td><td class="num">${num(p.elo, 0)}</td><td><span class="g ${esc(p.grade)}" style="display:inline-block;width:8px;height:8px;border-radius:50%"></span> ${esc(GLABEL[p.grade]||p.grade)}</td>` +
      `<td><span class="tag ${p.n_reads?"y":"n"}">${p.n_reads||"none"}</span></td><td><span class="tag ${p.n_runs?"y":"n"}">${p.n_runs||"none"}</span></td><td class="num">${p.n_mechanisms}</td></tr>`;
  h += `</tbody></table>`;
  box.innerHTML = h;
  box.querySelectorAll("tr[data-go]").forEach(tr => { tr.onclick = () => { location.hash = tr.dataset.go; }; });
}

// ---------------------------------------------------------------- highlights
function renderHighlights(){
  const g = document.getElementById("hlgrid"), sec = document.getElementById("hl"); if (!g) return;
  const items = D.highlights || [];
  if (!items.length){ sec.querySelector(".hint").textContent = "none yet — a highlight is a lens cell a mechanism quotes that verifies verbatim against the read; none of the current citations do."; g.innerHTML = ""; }
  g.innerHTML = items.map((h, i) => {
    const smp = esc(h.sample.trim()).split(esc(h.quote)).join(`<mark>${esc(h.quote)}</mark>`);
    const loc = esc(h.local.slice(0, h.local.length - h.token.length)) + `<b>${esc(h.token)}</b>`;
    const en = D.en && D.en[h.sample];
    return `<div class="hlc ${esc(h.region)}" data-i="${i}">
      <div class="where"><b>${esc(h.behavior)}</b> ${esc(cut(h.summary, 60))} · <span>${esc(h.read)}</span> · pos ${h.pos} · L${h.layer}${h.kind==="hand"?' · <span class="hand">hand-picked</span>':""}</div>
      <div class="loc">${loc}</div>
      <div class="q">${smp}${en ? `<div class="en">${esc(en)}</div>` : ""}</div>
      <div class="n">${esc(h.note)}</div></div>`;
  }).join("");
  g.querySelectorAll(".hlc").forEach(c => c.onclick = () => { const h = items[+c.dataset.i]; location.hash = patUrl(h.pattern_key, h.read, h.pos); const m = document.getElementById("app"); if (m && m.scrollIntoView) m.scrollIntoView({behavior:"smooth", block:"start"}); });
  const tg = document.getElementById("hltoggle");
  if (tg) tg.onclick = () => { g.hidden = !g.hidden; tg.textContent = g.hidden ? "show" : "collapse"; };
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
    (p.entry_id ? ` · entry ${esc(p.entry_id)}` : "") + (p.group_id ? ` · group ${esc(p.group_id)}` : "") +
    (p.n_group_members!=null ? ` · ${p.n_group_members} group members` : "") + ` · ${esc(GLABEL[p.grade]||p.grade)}</p>`;
  h += `<div><span class="pill">published match ${pct(p.published_match_rate)}</span><span class="pill">elo ${num(p.elo,0)}</span>` +
    Object.keys(p.elo_axes||{}).map(k => `<span class="pill">${esc(k)} ${num(p.elo_axes[k])}</span>`).join("") +
    (p.weirdchat_url ? `<a class="pill" href="${esc(p.weirdchat_url)}" target="_blank" rel="noopener">WeirdChat ↗</a>` : "") + `</div>`;
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
    `Browsers block fetch() from a file:// page — serve the site dir instead: <span class="kkey">python -m http.server</span> then open the printed URL (or build with <span class="kkey">single=true</span> for a self-contained file).</p>`;
}
function fetchJson(url){
  if (typeof fetch !== "function") return Promise.reject(new Error("no fetch()"));
  return fetch(url).then(r => { if (!r.ok) throw new Error(r.status + " " + r.statusText); return r.json(); });
}
function loadPattern(key, want){
  const url = dataUrl(key);
  const go = data => { const s = document.getElementById("heavy"); if (s && s.dataset.key === key) openViewer(key, data, want); };
  if (D.heavy && D.heavy[key]) return go(D.heavy[key]);   // single-file build: everything inlined
  if (heavyCache[key]) return go(heavyCache[key]);
  // only the fetch is caught: a render bug must surface in the console, not as a "could not load"
  fetchJson(url).then(data => { heavyCache[key] = data; return data; },
                      err => { fillHeavy(key, failNotice(url, err && err.message ? err.message : String(err))); return null; })
    .then(data => { if (data) go(data); });
}

// ---- read indexing
function indexRead(r){
  r.rowByPos = {}; r.positions = []; r.aboutPos = null;
  for (const row of r.rows || []){ r.rowByPos[row.pos] = row; r.positions.push(row.pos); if (row.region === "header") r.aboutPos = row.pos; }
}
function prepareData(key, data){
  data.reads = data.reads || []; data.runs = data.runs || []; data.samples = data.samples || [];
  data.reads.forEach(indexRead);
  data.byId = {}; data.byConv = {}; data.byTool = {};
  for (const r of data.reads){ data.byId[r.id] = r; if (r.conv_id && (!data.byConv[r.conv_id] || (!data.byConv[r.conv_id].rows && !data.byConv[r.conv_id].deferred))) data.byConv[r.conv_id] = r; if (r.tool_index!=null) data.byTool[r.tool_index] = r; }
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
  for (const h of (D.highlights||[])) if (h.pattern_key === key && h.quote) frags.add(h.quote);   // build-verified quotes, so cards and cells agree
  data.quotes = [...frags].sort((a,b) => b.length - a.length);
  data.key = key;
}

// ---- opening the viewer
function openViewer(key, data, want){
  if (!data.byId) prepareData(key, data);
  V = {key, data, readId: null, pos: null, compare: []};
  const reads = data.reads;
  let h = `<h2>Lens reads</h2>`;
  if (!reads.length) h += `<p class="none">no lens reads for this pattern yet (no diag file, no agent readouts).</p>`;
  else {
    h += `<p class="hint">A read is one OLens readout over one conversation: from the diagnostic pass (matched / unmatched study rollouts), or parsed back out of the agent's own <span class="kkey">readouts</span> tool pages (w###m/u = study rollouts it was seeded with; c### = conversations it created via chat). Left: the tokens that were read. Right: what the lens decoded at the selected token, every layer, one column per compared read.</p>`;
    h += `<div class="recs" id="reads" style="margin:0 0 10px"></div><main class="two"><section id="left"></section><section id="right"></section></main>`;
  }
  h += belowViewer(data);
  fillHeavy(key, h);
  wireBelow(data);
  if (!reads.length) return;
  const diagM = reads.find(r => r.source==="diag" && r.label==="matched"), diagU = reads.find(r => r.source==="diag" && r.label==="unmatched");
  if (diagM && diagU) V.compare = [diagM.id, diagU.id];
  const start = (want && want.read && data.byId[want.read]) ? data.byId[want.read] : (diagM || reads.find(r => r.rows && r.rows.length) || reads[0]);
  selectRead(start.id, want && want.pos!=null ? +want.pos : null);
}

function defaultPos(r){
  if (!r.positions || !r.positions.length) return null;
  if (r.fork_pos!=null && r.rowByPos[r.fork_pos]) return r.fork_pos;
  if (r.aboutPos!=null) return r.aboutPos;
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
    const left = document.getElementById("left"), right = document.getElementById("right");
    if (left) left.innerHTML = `<p class="none">loading the agent's reads (<span class="kkey">${esc(V.data.agent_file)}</span>) …</p>`;
    if (right) right.innerHTML = "";
    const key = V.key;
    Promise.all(needs.map(ensureAgentRows)).then(() => { if (V && V.key === key && V.readId === id) selectRead(id, V.pos); })
      .catch(err => { const l = document.getElementById("left"); if (V && V.key === key && l) l.innerHTML = failNotice(V.data.agent_file, err && err.message ? err.message : String(err)); });
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
  const box = document.getElementById("reads"); if (!box) return;
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
    (r.same_rollout_as ? ` · same rollout as <span class="kkey">${esc(r.same_rollout_as)}</span>, different lens sample` : "") +
    (r.aboutPos!=null ? ` · about-to-speak = ${r.aboutPos}` : "") + `</span></div>`;
  if (r.parse_error) h += `<div class="notice"><b>could not parse this readout page:</b> ${esc(r.parse_error)}</div>`;
  h += `<div class="legend"><span><i style="outline:1px solid var(--swept);outline-offset:-1px"></i>read position (click)</span><span><i style="background:var(--mark)"></i>cited by a mechanism</span><span><i style="background:var(--sel)"></i>selected</span>` +
    `<span><i style="box-shadow:0 2px 0 var(--ink)"></i>about to speak (last header token)</span>` +
    (r.fork_pos!=null ? `<span><i style="border-bottom:2px dotted var(--ink)"></i>fork: first word where the matched and unmatched replies diverge (≈)</span>` : "") +
    `<span><i></i>‥ = positions not read</span><span>keys <kbd>←</kbd> <kbd>→</kbd></span></div>`;
  const msgs = (r.messages||[]).map(m => `<div class="block"><div class="role"><span>${esc(m.role)}</span></div><pre class="clip">${esc(m.content)}</pre></div>`).join("");
  h += `<details><summary>text that was read (${r.text_source==="tool page" ? "from the agent's page, clipped to 900 chars" : "full, " + esc(r.text_source)})</summary>${msgs}` +
    `<div class="block reply"><div class="role"><span>assistant reply</span></div><pre class="clip">${esc(r.completion||"(empty)")}</pre></div></details>`;
  const rows = r.rows || [], flags = flagPositions(r);
  if (!rows.length) h += `<p class="none">no positions in this read.</p>`;
  let i = 0;
  while (i < rows.length){
    const region = rows[i].region; let inner = ""; let prev = null;
    for (; i < rows.length && rows[i].region === region; i++){
      const row = rows[i];
      if (prev!=null && row.pos - prev > 1) inner += `<span class="gap" title="${row.pos - prev - 1} positions not read">‥</span>`;
      prev = row.pos;
      const cls = ["tok","swept", flags.has(row.pos)?"flag":"", row.pos===V.pos?"sel":"", row.pos===r.aboutPos?"r1":"", row.pos===r.fork_pos?"fork":""].join(" ");
      inner += `<span class="${cls}" data-pos="${row.pos}" title="position ${row.pos} · ${esc(row.kind)}${flags.has(row.pos)?" · cited by a mechanism":""}">${tokHtml(row.tok)}</span>`;
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
  const re = /\b([wc]\d+[mu]?)\b|\bpos\s*~?\s*(\d+)/g; let m;
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
function whereText(r, row){
  if (row.region === "header") return row.pos === r.aboutPos
    ? "about to speak: the model has read the request and written nothing — identical for every rollout"
    : "the chat boundary before the reply — identical for every rollout of this prompt";
  if (row.region === "user") return "inside the user turn: the model is reading the request and has written nothing";
  if (row.region === "reply") return "inside the reply: the model is writing its answer" + (row.pos === r.fork_pos ? " — the fork, where the matched and unmatched replies first diverge" : "");
  return row.region;
}

// ---- right: the selected position, every layer, one column per compared read
function renderRight(){
  const r = V.data.byId[V.readId], right = document.getElementById("right"); if (!right) return;
  const p = V.pos, row = r.rowByPos ? r.rowByPos[p] : null;
  if (row == null){ right.innerHTML = `<p class="none">${r.parse_error ? "nothing to show — this page did not parse." : "no position selected."}</p>`; return; }
  const idx = r.positions.indexOf(p);
  let h = `<div class="poshdr"><h2>position ${p}</h2><span class="tokbox">${esc(JSON.stringify(row.tok))}</span><span class="hint">${esc(row.region)} · ${esc(row.kind)}</span>` +
    `<span class="navb"><button id="prev" ${idx<=0?"disabled":""}>← prev</button><button id="next" ${idx>=r.positions.length-1?"disabled":""}>next →</button></span></div>`;
  h += `<div class="where">${esc(whereText(r, row))}</div>`;
  let sofar = ""; for (const x of r.rows){ if (x.pos > p) break; if (x.pos < p) sofar += x.tok; }
  h += `<div class="local">${esc(sofar.slice(-200))}<b>${esc(row.tok)}</b><span class="hint">   ← read tokens so far (this token last; only read positions, so gaps are elided)</span></div>`;
  const fl = mechsAt(r, p);
  h += `<div class="flags"><h3>mechanisms citing this position (${fl.length}) — unverified hypotheses</h3>`;
  if (!fl.length) h += `<p class="none">none — no reported mechanism cites <span class="kkey">${esc(r.mention_ids.join(" / ")||r.id)}</span> at pos ${p}.</p>`;
  for (const f of fl) h += `<div class="flag ${f.via===r.conv_id?"":"other"}"><span class="cat">mechanism ${f.m.i+1}</span><span class="cell">${esc(f.m.auditor)} s${f.m.seed}${f.m.confidence==null?"":" · confidence "+num(f.m.confidence)}${f.via!==r.conv_id?" · cited on "+esc(f.via)+" (same rollout, different lens sample)":""}</span><div class="why">${esc(f.m.mechanism)}</div><q>${esc(cut(f.m.readout_cells, 400))}</q></div>`;
  h += `</div>`;
  const cols = V.compare.map(id => V.data.byId[id]).filter(Boolean);
  const layers = [...new Set(cols.reduce((acc, c) => acc.concat(c.layers||[]), []))].sort((a,b) => a-b);
  const toks = cols.map(c => c.rowByPos && c.rowByPos[p] ? c.rowByPos[p].tok : null);
  const present = toks.filter(t => t!=null);
  const identical = cols.length > 1 && present.length === cols.length && present.every(t => t === present[0]);
  const differ = cols.length > 1 && !identical;
  h += `<div class="grid"><h3>every readout at this token (verbatim, unedited; italic EN lines are translations) — ${cols.length} read${cols.length===1?"":"s"}</h3>`;
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
      else for (const s of samples){ const [sh, hit] = markQuotes(s.trim(), V.data.quotes); const en = D.en && D.en[s]; h += `<div class="samp ${hit?"hit":""}">${sh}${en ? `<div class="en">${esc(en)}</div>` : ""}</div>`; }
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
    `<span class="kkey">sample ${s.sample_index==null?"?":s.sample_index}</span></div>` + expandable(s.text, 800, "show full rollout") + `</div>`;
}
function linkCells(text, data){
  // every "pos N" that resolves to a read of this pattern becomes a link selecting read@position
  let out = "", last = 0, cid = null;
  const re = /\b([wc]\d+[mu]?)\b|\bpos\s*~?\s*(\d+)/g; let m;
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
    const o = s.output, isRead = s.name === "readouts", label = s.name === "chat" ? "reply" : "result";
    h += `<pre class="block" style="margin-top:6px">${esc(o.slice(0,600))}${o.length>600?" …":""}</pre>`;
    if (o.length > 600) h += `<details><summary>show full ${label} (${o.length} chars${isRead?", clipped for the viewer; the structured read is in the strip above":""})</summary><pre class="block">${esc(o)}</pre></details>`;
  }
  return h + `</div>`;
}
function mechBlock(m, i, data){
  let h = `<div class="mech"><div class="txt"><span class="n">${i+1}.</span>${esc(m.mechanism)}` + (m.confidence==null ? "" : ` <span class="conf">confidence ${num(m.confidence)}</span>`) + `</div>`;
  if (m.evidence) h += `<div class="ev">${esc(m.evidence)}</div>`;
  if (m.readout_cells) h += `<div class="cells">cells: ${linkCells(m.readout_cells, data)}</div>`;
  if (m.would_test_by) h += `<div class="test"><b>would test by — not run in this pass:</b> ${esc(m.would_test_by)}</div>`;
  return h + `</div>`;
}
function runBlock(r, data){
  let h = `<div class="card"><b>${esc(r.auditor)}</b> <span class="kkey">seed ${r.seed==null?"?":r.seed}</span>` +
    ` <span class="crumb">· ${(r.steps||[]).length} steps` + (r.cells_served ? ` · ${r.cells_served} cells served` : "") + (r.output_tokens ? ` · ${r.output_tokens} tok` : "") +
    (r.server_calls ? ` · ${r.server_calls} server calls` : "") + (r.server_seconds ? ` · ${num(r.server_seconds,1)}s server` : "") + (r.stopped_by ? ` · stopped: ${esc(r.stopped_by)}` : "") + `</span>`;
  if (r.summary) h += `<div style="margin-top:6px">${esc(r.summary)}</div>`;
  h += `</div><h3>transcript</h3>`;
  if (!(r.steps||[]).length) h += `<p class="none">no turns recorded.</p>`;
  (r.steps||[]).forEach((s,i) => { h += stepBlock(s, i, data); });
  if ((r.notes||[]).length){ h += `<h3>scratch notes (${r.notes.length})</h3>`; r.notes.forEach(n => { h += `<pre class="block small" style="margin-bottom:6px">${esc(n)}</pre>`; }); }
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
  document.querySelectorAll("#heavy [data-open]").forEach(a => a.onclick = ev => {
    ev.preventDefault(); selectRead(a.dataset.open, null);
    const m = document.querySelector("main.two"); if (m && m.scrollIntoView) m.scrollIntoView({block:"start", behavior:"smooth"});
  });
}

// ------------------------------------------------------------------- themes
function renderThemes(){
  const cl = D.clusters || [];
  let h = `<h2 style="margin-top:4px">Themes — mechanism clusters</h2>`;
  if (!cl.length) return h + `<p class="none">no synth.json yet — clusters appear once the synthesis pass runs.</p>`;
  cl.forEach((k,i) => {
    h += `<div class="card"><a href="#/cluster/${i}"><b>${esc(k.name)}</b></a><div class="crumb">${k.members.length} pattern${k.members.length===1?"":"s"} · ${k.behavior_ids.length} behavior${k.behavior_ids.length===1?"":"s"}</div>` +
      (k.description ? `<div style="margin-top:5px">${esc(k.description)}</div>` : "") + `</div>`;
  });
  return h;
}
function renderCluster(i){
  const k = (D.clusters||[])[i];
  if (!k) return `<p class="none">no cluster ${esc(i)}.</p>`;
  let h = `<h2 style="margin-top:4px">${esc(k.name)}</h2>`;
  if (k.description) h += `<div class="card">${esc(k.description)}</div>`;
  h += `<h3>behaviors spanned (${k.behavior_ids.length})</h3><div>` + (k.behavior_ids.length ? k.behavior_ids.map(b => `<span class="pill">${esc(b)}</span>`).join("") : `<span class="none">none listed</span>`) + `</div>`;
  h += `<h3>members (${k.members.length}) — unverified mechanisms</h3>`;
  if (!k.members.length) h += `<p class="none">no members listed.</p>`;
  for (const m of k.members){
    const p = byKey[m.pattern_key];
    h += `<div class="mech"><div class="txt"><a href="${patUrl(m.pattern_key)}">${esc(p ? p.behavior_name : m.pattern_key)}</a> <span class="kkey">${esc(m.pattern_key)}</span>${p?"":` <span class="tag n">not in this build</span>`}</div>` +
      (m.mechanism ? `<div class="ev">${esc(m.mechanism)}</div>` : "") + (m.evidence ? `<div class="cells">${esc(m.evidence)}</div>` : "") + `</div>`;
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
function openPatternPage(key, want){
  const app = document.getElementById("app");
  const slot = document.getElementById("heavy");
  if (V && V.key === key && slot && slot.dataset.key === key){
    // same page: a deep link from a highlight, mechanism or transcript — just move the selection
    if (want.read && V.data.byId[want.read]){ selectRead(want.read, want.pos); const m = document.querySelector("main.two"); if (m && m.scrollIntoView) m.scrollIntoView({block:"start"}); }
    return;
  }
  V = null;
  renderPickers(key);
  app.innerHTML = renderPattern(key);
  if (byKey[key]) loadPattern(key, want);
}
function route(){
  const {parts, q} = parseHash();
  const app = document.getElementById("app");
  if (parts[0] === "pattern" && parts.length > 1){
    openPatternPage(decodeURIComponent(parts.slice(1).join("/")), {read: q.read || null, pos: q.pos!=null && q.pos!=="" ? +q.pos : null});
    return;
  }
  if (parts[0] === "cluster" && parts.length > 1){
    V = null; renderPickers(null);
    app.innerHTML = `<p class="crumb"><a href="#/themes">themes</a> › cluster</p>` + renderCluster(parseInt(parts[1], 10));
    window.scrollTo(0, 0); return;
  }
  if (parts[0] === "themes"){ V = null; renderPickers(null); app.innerHTML = renderThemes(); window.scrollTo(0, 0); return; }
  // "#/" = the first behavior's first pattern
  const key = firstKey();
  if (!key){ V = null; renderPickers(null); app.innerHTML = `<p class="none">no patterns found under <span class="kkey">${esc(D.source)}</span>.</p>`; return; }
  openPatternPage(key, {read: null, pos: null});
}
renderOverviewTables();
renderHighlights();
window.addEventListener("hashchange", route);
route();
</script>
"""
TEMPLATE = TEMPLATE.replace("__BANNER__", BANNER)


if __name__ == "__main__":
    main()
