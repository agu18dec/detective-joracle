"""Build the WeirdChat explanation viewer — one self-contained static page.

Walks ``<out_root>/{patterns,diag,runs}`` plus an optional ``synth.json``, embeds everything as
JSON in a single HTML file (CSS + JS inlined, no external assets) and writes
``<site_dir>/index.html`` (default ``<out_root>/site``).

    python scripts/weirdchat/build_site.py [out=outputs/weirdchat] [site=<dir>]
    cd outputs/weirdchat/site && python -m http.server 8905

The experiment tells an investigator agent about a behavior WeirdChat catalogued in Qwen3.6-27B
and asks it to explain *why* the model does it, using chat probes plus an OLens readout of the
model's own activations. There is no ground truth and no intervention testing in this pass, so
every mechanism the agent reports is an unverified hypothesis — the page says so on every view.

Routing is client-side by hash: ``#/`` overview, ``#/pattern/<key>``, ``#/cluster/<i>``. Every
field is treated as optional: partial output is the normal state mid-run.
"""

import json
import sys
from pathlib import Path
from typing import Any

# a single OLens cell keeps at most this much text (the title attribute holds the rest)
CELL_CAP = 600
# a tool result in the transcript keeps at most this much text, so the one file stays sane
TOOL_OUTPUT_CAP = 20000


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


# ----------------------------------------------------------------- olens grid
def show_token(token: Any) -> str:
    """Token text with whitespace made visible, for the row label."""
    text = as_text(token).replace("\\", "\\\\")
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def sorted_keys(mapping: dict[str, Any]) -> list[str]:
    """Numeric-looking keys in numeric order; anything else after, lexically."""
    nums = sorted((k for k in mapping if as_int(k) is not None), key=lambda k: as_int(k) or 0)
    rest = sorted(k for k in mapping if as_int(k) is None)
    return nums + rest


def grid_of(readout: dict[str, Any]) -> dict[str, Any]:
    """One read's OLens table: layers ascending, one row per token position, region-grouped."""
    tokens = as_dict(readout.get("tokens"))
    tags = as_dict(readout.get("tags"))
    layers_raw = as_dict(readout.get("readouts"))
    layers = sorted_keys(layers_raw)
    positions: set[str] = set(tokens) | set(tags)
    for layer in layers:
        positions |= set(as_dict(layers_raw.get(layer)))
    rows: list[dict[str, Any]] = []
    for pos in sorted_keys(dict.fromkeys(positions)):
        tag = as_dict(tags.get(pos))
        cells: list[str] = []
        for layer in layers:
            value = as_dict(layers_raw.get(layer)).get(pos)
            if isinstance(value, list):
                text = " · ".join(as_text(v) for v in value if as_text(v))
            else:
                text = as_text(value)
            cells.append(clip(text, CELL_CAP))
        rows.append(
            {
                "pos": pos,
                "token": show_token(tokens.get(pos)),
                "region": as_text(tag.get("region")) or "—",
                "kind": as_text(tag.get("kind")),
                "cells": cells,
            }
        )
    return {
        "layers": layers,
        "rows": rows,
        "lens": as_text(readout.get("lens")) or "olens",
        "n_tokens": as_int(readout.get("n_tokens")),
    }


def read_of(entry: dict[str, Any]) -> dict[str, Any]:
    """One diagnostic read: its label, the exact conversation read, and its grid."""
    conv = as_dict(entry.get("conversation"))
    messages = [
        {"role": as_text(m.get("role")), "content": as_text(m.get("content"))}
        for m in as_list(conv.get("messages"))
        if isinstance(m, dict)
    ]
    return {
        "label": as_text(entry.get("label")) or "read",
        "sample_index": as_int(entry.get("sample_index")),
        "messages": messages,
        "completion": as_text(conv.get("completion")),
        "grid": grid_of(as_dict(entry.get("readout"))),
    }


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
        think = as_text(turn.get("content")).strip()
        calls = [as_dict(c) for c in as_list(turn.get("tool_calls"))]
        if not calls:
            if think and think != "...":
                steps.append({"think": think, "name": None, "args": "", "output": "", "meta": ""})
            continue
        for call in calls:
            logged = as_dict(tool_log[ti]) if ti < len(tool_log) else {}
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


def run_of(path: Path, auditor_dir: str) -> dict[str, Any]:
    """One agent run file as the viewer's run object (tolerant of every missing key)."""
    blob = read_json(path) or {}
    record = as_dict(blob.get("record"))
    extras = as_dict(blob.get("extras"))
    result = as_dict(record.get("result"))
    return {
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


# ---------------------------------------------------------------- collection
def pattern_of(path: Path, out_root: Path) -> dict[str, Any]:
    """One pattern plus whatever diagnostics and agent runs exist beside it."""
    meta = read_json(path) or {}
    key = as_text(meta.get("pattern_key")) or path.stem
    diag = read_json(out_root / "diag" / f"{key}.json") or {}
    fork = as_dict(diag.get("fork")) if diag.get("fork") else {}
    runs = []
    run_root = out_root / "runs" / key
    if run_root.is_dir():
        for run_path in sorted(run_root.glob("*/seed_*.json")):
            runs.append(run_of(run_path, run_path.parent.name))
    runs.sort(key=lambda r: (r["auditor_dir"], r["seed"] if r["seed"] is not None else -1))
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
        "reads": [read_of(as_dict(r)) for r in as_list(diag.get("reads"))],
        "fork": {"position": as_int(fork.get("position")), "note": as_text(fork.get("note"))}
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


# ------------------------------------------------------------------------ main
def build_site(out_root: Path, site_dir: Path) -> Path:
    """Collect everything under ``out_root`` and write ``site_dir/index.html``."""
    pattern_dir = out_root / "patterns"
    paths = sorted(pattern_dir.glob("*.json")) if pattern_dir.is_dir() else []
    patterns = [pattern_of(p, out_root) for p in paths]
    patterns.sort(key=lambda p: (p["behavior_name"], p["key"]))
    data = {
        "patterns": patterns,
        "behaviors": behaviors_of(patterns),
        "clusters": clusters_of(out_root),
        "synth": synth_meta(out_root),
        "counts": {
            "patterns": len(patterns),
            "behaviors": len({p["behavior_id"] for p in patterns}),
            "runs": sum(len(p["runs"]) for p in patterns),
            "with_diag": sum(1 for p in patterns if p["reads"]),
            "mechanisms": sum(p["n_mechanisms"] for p in patterns),
        },
        "source": str(out_root),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.replace("/*__DATA__*/", payload)
    site_dir.mkdir(parents=True, exist_ok=True)
    out = site_dir / "index.html"
    out.write_text(page)
    size_mb = out.stat().st_size / 1048576
    print(
        f"{len(patterns)} patterns · {data['counts']['behaviors']} behaviors · "
        f"{data['counts']['runs']} agent runs · {data['counts']['mechanisms']} mechanisms "
        f"· {len(data['clusters'])} clusters ({size_mb:.1f} MB)"
    )
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
<style>
:root{
  color-scheme:light dark;
  --paper:#F7F7F4; --card:#FFFFFF; --ink:#1B2230; --ink2:#4A5262; --mute:#7A8394;
  --line:#DDDFE3; --line2:#ECEDEF; --accent:#0E7C7B; --accent-bg:#E6F4F3;
  --warn:#B42318; --warn-bg:#FCE9E6; --ok:#15803D; --ok-bg:#E5F4EA; --amber:#B45309;
  --sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#12151B; --card:#1A1E26; --ink:#E7EAF0; --ink2:#B4BCCB; --mute:#8A93A5;
    --line:#2C323D; --line2:#232832; --accent:#4FD1C5; --accent-bg:#14312F;
    --warn:#FF8A80; --warn-bg:#3A1D1C; --ok:#6EE7A0; --ok-bg:#16301F; --amber:#F0B25F;
  }
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.45;margin:0}
a{color:var(--accent)}
header{padding:16px 24px 12px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:30}
h1{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
h1 a{color:inherit;text-decoration:none}
h2{font-size:17px;margin:22px 0 8px}
h3{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:18px 0 7px;font-weight:600}
.crumb{font-size:12.5px;color:var(--mute)}
.banner{background:var(--warn-bg);color:var(--warn);border:1px solid var(--warn);border-radius:6px;padding:8px 12px;margin:10px 0 0;font-size:12.5px;max-width:120ch}
.banner b{letter-spacing:.04em}
main{padding:18px 24px 80px;max-width:1500px}
.counts{color:var(--ink2);font-size:13px;margin:0 0 4px}
.counts b{color:var(--ink)}
table.t{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden}
table.t th{text-align:left;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute);font-weight:600;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
table.t td{padding:7px 10px;border-bottom:1px solid var(--line2);vertical-align:top;font-size:13px}
table.t tr:last-child td{border-bottom:none}
table.t tr.click{cursor:pointer}
table.t tr.click:hover td{background:var(--accent-bg)}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:12px;white-space:nowrap}
.kkey{font-family:var(--mono);font-size:11.5px;color:var(--ink2)}
.tag{font-size:11px;border-radius:4px;padding:1px 6px;font-family:var(--mono)}
.tag.y{background:var(--ok-bg);color:var(--ok)} .tag.n{background:var(--line2);color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;margin:0 0 10px}
pre.block{font-family:var(--mono);font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;color:var(--ink)}
pre.small{font-size:11.5px;color:var(--ink2);border:none;padding:0;background:none}
details{margin:6px 0}
summary{cursor:pointer;color:var(--accent);font-size:12.5px}
summary::marker{color:var(--mute)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:980px){.cols{grid-template-columns:1fr}}
.roll{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:9px 11px;margin:0 0 9px}
.roll .hd{font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--mute);margin-bottom:5px;display:flex;gap:8px;align-items:baseline}
.roll.matched{border-left:3px solid var(--warn)} .roll.unmatched{border-left:3px solid var(--mute)}
.gridwrap{overflow:auto;max-height:72vh;border:1px solid var(--line);border-radius:6px;background:var(--card)}
table.grid{border-collapse:separate;border-spacing:0;font-size:11.5px}
table.grid th,table.grid td{border-bottom:1px solid var(--line2);border-right:1px solid var(--line2);padding:3px 6px;vertical-align:top}
table.grid thead th{position:sticky;top:0;z-index:2;background:var(--card);color:var(--mute);font-size:11px;letter-spacing:.04em;text-transform:uppercase;white-space:nowrap}
table.grid th.tok{position:sticky;left:0;z-index:1;background:var(--card);font-family:var(--mono);font-weight:500;color:var(--ink);text-align:left;white-space:pre;max-width:190px;overflow:hidden;text-overflow:ellipsis}
table.grid thead th.tok{z-index:3}
table.grid td{max-width:230px;color:var(--ink2)}
table.grid td:empty::after{content:"·";color:var(--line)}
table.grid tr.region th{position:sticky;left:0;background:var(--accent-bg);color:var(--accent);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;text-align:left;font-weight:600}
table.grid tr.region td{background:var(--accent-bg)}
.pos{color:var(--mute);font-family:var(--mono);font-size:10px;margin-right:5px}
.step{border:1px solid var(--line2);border-radius:6px;background:var(--card);margin:0 0 8px;padding:7px 10px}
.step .hd{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);display:flex;gap:8px;align-items:baseline}
.step .hd .nm{font-family:var(--mono);letter-spacing:0;text-transform:none;color:var(--ink);background:var(--line2);border-radius:4px;padding:0 6px;font-weight:600}
.step .think{font-size:12.5px;color:var(--ink2);white-space:pre-wrap;word-break:break-word;margin-top:5px}
.step .call{font-family:var(--mono);font-size:11.5px;color:var(--accent);margin-top:5px;white-space:pre-wrap;word-break:break-word}
.mech{border:1px solid var(--line2);border-left:3px solid var(--accent);border-radius:0 6px 6px 0;background:var(--card);padding:9px 11px;margin:0 0 8px}
.mech .n{font-family:var(--mono);color:var(--mute);font-size:11.5px;margin-right:6px}
.mech .txt{font-size:13.5px;color:var(--ink)}
.mech .ev{font-size:12.5px;color:var(--ink2);margin-top:5px}
.mech .cells{font-family:var(--mono);font-size:11.5px;color:var(--mute);margin-top:4px;word-break:break-word}
.mech .test{font-size:12px;color:var(--mute);font-style:italic;margin-top:6px;border-top:1px dashed var(--line);padding-top:5px}
.mech .test b{font-style:normal;font-weight:600;letter-spacing:.04em;text-transform:uppercase;font-size:10.5px;color:var(--amber)}
.conf{font-family:var(--mono);font-size:11.5px;color:var(--ink2);background:var(--line2);border-radius:4px;padding:1px 6px}
.none{color:var(--mute);font-size:13px;font-style:italic}
.pill{display:inline-block;font-size:11.5px;color:var(--ink2);background:var(--line2);border-radius:999px;padding:1px 9px;margin:0 5px 5px 0;font-family:var(--mono)}
</style>

<header>
  <h1><a href="#/">WeirdChat — why does it do that?</a></h1>
  <div class="crumb" id="crumb"></div>
  <div class="banner"><b>UNVERIFIED HYPOTHESES.</b> <span id="banner"></span></div>
</header>
<main id="app"></main>

<script>
const D = /*__DATA__*/;
const BANNER = "__BANNER__";
const esc = s => (s==null?"":String(s)).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const num = (v,d) => v==null ? "—" : Number(v).toFixed(d==null?2:d);
const byKey = {}; (D.patterns||[]).forEach(p => byKey[p.key] = p);
document.getElementById("banner").textContent = BANNER;

function cut(s, n){ s = s==null ? "" : String(s); return s.length<=n ? s : s.slice(0,n) + " …"; }

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
    `<b>${c.runs||0}</b> agent runs · <b>${c.with_diag||0}</b> with OLens diagnostics · ` +
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
      `<th class="num">elo</th><th>diag</th><th>runs</th><th class="num">mechanisms</th></tr></thead><tbody>`;
    for (const p of D.patterns)
      h += `<tr class="click" data-go="#/pattern/${encodeURIComponent(p.key)}">` +
        `<td>${esc(p.behavior_name)}<div class="kkey">${esc(p.key)}</div></td>` +
        `<td>${esc(cut(p.group_summary, 160))}</td>` +
        `<td class="num">${num(p.published_match_rate)}</td>` +
        `<td class="num">${num(p.elo, 0)}</td>` +
        `<td><span class="tag ${p.reads.length?"y":"n"}">${p.reads.length?p.reads.length+" reads":"none"}</span></td>` +
        `<td><span class="tag ${p.runs.length?"y":"n"}">${p.runs.length?p.runs.length:"none"}</span></td>` +
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
function rollout(s){
  const cls = s.matched ? "matched" : "unmatched";
  return `<div class="roll ${cls}"><div class="hd"><span>${s.matched?"matched":"unmatched"}</span>` +
    `<span class="kkey">sample ${s.sample_index==null?"?":s.sample_index}</span></div>` +
    expandable(s.text, 800, "show full rollout") + `</div>`;
}

function gridBlock(r){
  const g = r.grid || {layers:[], rows:[]};
  let h = `<div class="card"><b>read: ${esc(r.label)}</b> <span class="kkey">sample ${r.sample_index==null?"?":r.sample_index}</span>` +
    ` <span class="crumb">· lens ${esc(g.lens)}${g.n_tokens!=null?" · "+g.n_tokens+" tokens":""} · ${g.layers.length} layers · ${g.rows.length} positions</span>`;
  h += `<h3>exact text read</h3>`;
  for (const m of r.messages||[])
    h += `<div class="crumb" style="margin-top:4px">${esc(m.role)}</div>` + expandable(m.content, 800, "show full message");
  if (r.completion){ h += `<div class="crumb" style="margin-top:4px">completion</div>` + expandable(r.completion, 800, "show full completion"); }
  h += `</div>`;
  if (!g.rows.length){ return h + `<p class="none">no readout grid for this read.</p>`; }

  const ncol = g.layers.length + 1;
  let t = `<div class="gridwrap"><table class="grid"><thead><tr><th class="tok">token</th>` +
    g.layers.map(l => `<th>L${esc(l)}</th>`).join("") + `</tr></thead><tbody>`;
  let region = null;
  for (const row of g.rows){
    if (row.region !== region){
      region = row.region;
      t += `<tr class="region"><th class="tok">${esc(region)}</th><td colspan="${ncol-1}">region: ${esc(region)}</td></tr>`;
    }
    const kind = row.kind ? ` title="kind: ${esc(row.kind)}"` : "";
    t += `<tr><th class="tok"${kind}><span class="pos">${esc(row.pos)}</span>${esc(row.token)}</th>` +
      row.cells.map(c => `<td title="${esc(c)}">${esc(cut(c, 90))}</td>`).join("") + `</tr>`;
  }
  return h + t + `</tbody></table></div>`;
}

function stepBlock(s, i){
  let h = `<div class="step"><div class="hd"><span class="nm">${esc(s.name||"assistant")}</span>` +
    `<span>#${i+1}${s.meta?" · "+esc(s.meta):""}</span></div>`;
  if (s.think) h += `<div class="think">${esc(s.think)}</div>`;
  if (s.name) h += `<div class="call">${esc(s.name)}(${esc(s.args)})</div>`;
  if (s.output){
    const o = s.output;
    h += `<pre class="block" style="margin-top:6px">${esc(o.slice(0,600))}${o.length>600?" …":""}</pre>`;
    if (o.length > 600)
      h += `<details><summary>show full result (${o.length} chars)</summary><pre class="block">${esc(o)}</pre></details>`;
  }
  return h + `</div>`;
}

function runBlock(r){
  let h = `<div class="card"><b>${esc(r.auditor)}</b> <span class="kkey">seed ${r.seed==null?"?":r.seed}</span>` +
    ` <span class="crumb">· ${r.steps.length} steps` +
    (r.cells_served ? ` · ${r.cells_served} cells served` : "") +
    (r.output_tokens ? ` · ${r.output_tokens} tok` : "") +
    (r.server_calls ? ` · ${r.server_calls} server calls` : "") +
    (r.server_seconds ? ` · ${num(r.server_seconds,1)}s server` : "") +
    (r.stopped_by ? ` · stopped: ${esc(r.stopped_by)}` : "") + `</span>`;
  if (r.summary) h += `<div style="margin-top:6px">${esc(r.summary)}</div>`;
  h += `</div>`;

  h += `<h3>transcript</h3>`;
  if (!r.steps.length) h += `<p class="none">no turns recorded.</p>`;
  r.steps.forEach((s,i) => { h += stepBlock(s,i); });

  if ((r.notes||[]).length){
    h += `<h3>scratch notes (${r.notes.length})</h3>`;
    r.notes.forEach(n => { h += `<pre class="block small" style="margin-bottom:6px">${esc(n)}</pre>`; });
  }

  h += `<h3>reported mechanisms (${r.mechanisms.length}) — unverified</h3>`;
  if (!r.mechanisms.length) h += `<p class="none">this run reported no mechanisms.</p>`;
  r.mechanisms.forEach((m,i) => {
    h += `<div class="mech"><div class="txt"><span class="n">${i+1}.</span>${esc(m.mechanism)}` +
      (m.confidence==null ? "" : ` <span class="conf">confidence ${num(m.confidence)}</span>`) + `</div>`;
    if (m.evidence) h += `<div class="ev">${esc(m.evidence)}</div>`;
    if (m.readout_cells) h += `<div class="cells">cells: ${esc(m.readout_cells)}</div>`;
    if (m.would_test_by) h += `<div class="test"><b>would test by — not run in this pass:</b> ${esc(m.would_test_by)}</div>`;
    h += `</div>`;
  });
  return h;
}

function renderPattern(key){
  const p = byKey[key];
  if (!p) return `<p class="none">no pattern <span class="kkey">${esc(key)}</span>.</p>`;
  let h = `<h2>${esc(p.behavior_name)}</h2>` +
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

  h += `<h3>rollouts</h3>`;
  const mt = p.samples.filter(s => s.matched), um = p.samples.filter(s => !s.matched);
  if (!p.samples.length) h += `<p class="none">no samples recorded.</p>`;
  else h += `<div class="cols"><div><h3>matched (${mt.length})</h3>` +
    (mt.length ? mt.map(rollout).join("") : `<p class="none">none</p>`) +
    `</div><div><h3>unmatched (${um.length})</h3>` +
    (um.length ? um.map(rollout).join("") : `<p class="none">none</p>`) + `</div></div>`;

  h += `<h2>OLens readouts</h2>`;
  if (p.fork && (p.fork.position!=null || p.fork.note))
    h += `<div class="card"><b>fork</b> <span class="kkey">position ${p.fork.position==null?"—":p.fork.position}</span>` +
      (p.fork.note ? `<div style="margin-top:4px">${esc(p.fork.note)}</div>` : "") + `</div>`;
  if (!p.reads.length) h += `<p class="none">no diagnostics for this pattern yet.</p>`;
  p.reads.forEach(r => { h += gridBlock(r); });

  h += `<h2>Agent runs</h2>`;
  if (!p.runs.length) h += `<p class="none">no agent runs for this pattern yet.</p>`;
  p.runs.forEach(r => { h += runBlock(r); });
  return h;
}

// ------------------------------------------------------------------- cluster
function renderCluster(i){
  const k = (D.clusters||[])[i];
  if (!k) return `<p class="none">no cluster ${esc(i)}.</p>`;
  let h = `<h2>${esc(k.name)}</h2>`;
  if (k.description) h += `<div class="card">${esc(k.description)}</div>`;
  h += `<h3>behaviors spanned (${k.behavior_ids.length})</h3><div>` +
    (k.behavior_ids.length ? k.behavior_ids.map(b => `<span class="pill">${esc(b)}</span>`).join("") : `<span class="none">none listed</span>`) + `</div>`;
  h += `<h3>members (${k.members.length}) — unverified mechanisms</h3>`;
  if (!k.members.length) h += `<p class="none">no members listed.</p>`;
  for (const m of k.members){
    const p = byKey[m.pattern_key];
    h += `<div class="mech"><div class="txt">` +
      `<a href="#/pattern/${encodeURIComponent(m.pattern_key)}">${esc(p ? p.behavior_name : m.pattern_key)}</a>` +
      ` <span class="kkey">${esc(m.pattern_key)}</span>${p?"":` <span class="tag n">not in this build</span>`}</div>` +
      (m.mechanism ? `<div class="ev">${esc(m.mechanism)}</div>` : "") +
      (m.evidence ? `<div class="cells">${esc(m.evidence)}</div>` : "") + `</div>`;
  }
  return h;
}

// --------------------------------------------------------------------- route
function route(){
  const raw = (location.hash || "#/").replace(/^#\/?/, "");
  const parts = raw.split("/");
  const app = document.getElementById("app"), crumb = document.getElementById("crumb");
  let html = "", trail = "overview";
  if (parts[0] === "pattern" && parts.length > 1){
    const key = decodeURIComponent(parts.slice(1).join("/"));
    html = renderPattern(key);
    trail = `<a href="#/">overview</a> › pattern <span class="kkey">${esc(key)}</span>`;
  } else if (parts[0] === "cluster" && parts.length > 1){
    const i = parseInt(parts[1], 10);
    html = renderCluster(i);
    const k = (D.clusters||[])[i];
    trail = `<a href="#/">overview</a> › cluster ${k ? esc(k.name) : esc(parts[1])}`;
  } else {
    html = renderOverview();
  }
  crumb.innerHTML = trail;
  app.innerHTML = html;
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
