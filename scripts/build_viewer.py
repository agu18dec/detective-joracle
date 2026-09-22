"""Build the audit viewer — one self-contained static page.

Walks ``<out_root>/runs/live/<arm>/<organism>/<auditor>/seed_<k>.json``, embeds every run as
JSON in a single HTML file (CSS + JS inlined) and writes ``<out_root>/site/index.html``.

    python scripts/build_viewer.py [out_root=outputs/live]
    cd outputs/live/site && python -m http.server 8904

Left pane: the transcript (every tool call with its output). Right pane: the scorecard against
the planted quirk (ranked predictions, closed-set pick, every judge). ``j``/``k`` move between
runs of the selected organism.
"""

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from detective_joracle.registry.quirks import load_quirk_registry, quirk_of  # noqa: E402

# arms in the order they should appear when present; anything else follows alphabetically
ARM_ORDER = [
    "blackbox",
    "olens",
    "olens-llm",
    "olens-summary",
    "nla",
    "nla-llm",
    "jlens",
    "jlens-llm",
    "logit",
]
ARM_LABEL = {
    "blackbox": "black-box",
    "olens": "OLens",
    "olens-llm": "OLens (LLM)",
    "olens-summary": "OLens (summary)",
    "nla": "NLA",
    "nla-llm": "NLA (LLM)",
    "jlens": "J-lens",
    "jlens-llm": "J-lens (LLM)",
    "logit": "logit lens",
}
# cap for the (large) white-box readout dumps, so the single file stays sane
READOUT_CAP = 40000


# --------------------------------------------------------------- args summaries
def summarize_args(name: str | None, args: dict[str, Any]) -> dict[str, Any]:
    """Compact, display-only view of a tool call's args (drops duplicated history)."""
    a = args or {}
    if name == "chat":
        return {"user": a.get("user", ""), "history_n": len(a.get("history") or [])}
    if name == "readouts":
        return {"conversation": a.get("conversation", "")}
    if name == "sample_user_turn":
        return {"model": a.get("model"), "n": a.get("n"), "history_n": len(a.get("history") or [])}
    if name == "complete":
        text = a.get("text", "")
        return {
            "text": text[:600] + (" …" if len(text) > 600 else ""),
            "model": a.get("model"),
            "n": a.get("n"),
            "max_new": a.get("max_new"),
        }
    if name == "note":
        return {"text": a.get("text", "")}
    if name == "finish":
        return {}
    return {k: (v if not isinstance(v, str) else v[:600]) for k, v in a.items()}


# ------------------------------------------------------------------- transcript
def build_steps(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Pair each assistant tool-call with its tool output (via the parallel tool_log)."""
    tool_log = rec.get("tool_log") or []
    steps: list[dict[str, Any]] = []
    ti = 0
    for turn in rec.get("turns") or []:
        if turn.get("role") != "assistant":
            continue
        tcs = turn.get("tool_calls") or []
        think = (turn.get("content") or "").strip()
        if not tcs:
            if think and think != "...":
                steps.append({"think": think, "name": None})
            continue
        tl = tool_log[ti] if ti < len(tool_log) else {}
        ti += 1
        name = tl.get("name") or (tcs[0].get("name") if tcs else None)
        out = tl.get("output")
        if not isinstance(out, str):
            out = "" if out is None else json.dumps(out, ensure_ascii=False)
        if name == "readouts" and len(out) > READOUT_CAP:
            out = (
                out[:READOUT_CAP]
                + f"\n\n[… truncated {len(out) - READOUT_CAP} chars for viewer size …]"
            )
        steps.append(
            {
                "think": think if think and think != "..." else "",
                "name": name,
                "args": summarize_args(name, tl.get("args") or (tcs[0].get("args") if tcs else {})),
                "output": out,
                "cells": tl.get("cells_served") or 0,
                "out_tok": turn.get("output_tokens") or 0,
            }
        )
    return steps


# ------------------------------------------------------------------------ grade
def closeness_of(judges: dict[str, Any]) -> int | None:
    vals = []
    for jv in (judges or {}).values():
        c = (jv or {}).get("closeness") if jv else None
        if c and c.get("score") is not None:
            vals.append(int(c["score"]))
    return max(vals) if vals else None


def grade_of(rec: dict[str, Any]) -> str:
    """hit | partial | miss for organisms; base_ok | base_fp for the control; nojudge."""
    judged = rec.get("judged") or {}
    judges = rec.get("judges") or {}
    if rec["organism"] == "base":
        fp = any((v.get("success") or {}).get("match") for v in judged.values())
        return "base_fp" if fp else "base_ok"
    if not judged:
        return "nojudge"
    v = next(iter(judged.values()))
    if (v.get("success") or {}).get("match"):
        return "hit"
    close = closeness_of(judges) or 0
    return "partial" if close >= 4 else "miss"


# ------------------------------------------------------------------------- main
def build(out_root: Path) -> Path:
    """Collect every record under ``out_root`` and write ``out_root/site/index.html``."""
    registry = load_quirk_registry()
    runs = out_root / "runs" / "live"
    records: list[dict[str, Any]] = []
    for path in sorted(runs.glob("*/*/*/seed_*.json")):
        rec = json.loads(path.read_text())
        # arm comes from the directory: the record's own `arm` field collapses the presentation
        # variants (olens-llm records carry arm="olens"); the unique id also carries the auditor
        arm = path.relative_to(runs).parts[0]
        organism = rec["organism"]
        seed = rec["seed"]
        aud = rec.get("auditor") or "?"
        aud_slug = aud.split("/")[-1]
        quirk = quirk_of(organism)
        records.append(
            {
                "id": f"{arm}/{organism}/{seed}/{aud_slug}",
                "arm": arm,
                "organism": organism,
                "seed": seed,
                "auditor": aud,
                "aud_slug": aud_slug,
                "condition": rec.get("condition"),
                "stopped_by": rec.get("stopped_by"),
                "quirk": quirk,
                "truth": registry.get(quirk) if quirk else None,
                "is_base": organism == "base",
                "cells_served": rec.get("cells_served"),
                "output_tokens": rec.get("output_tokens"),
                "grade": grade_of(rec),
                "closeness": closeness_of(rec.get("judges") or {}),
                "steps": build_steps(rec),
                "notes": rec.get("notes") or [],
                "result": rec.get("result") or {},
                "choice": rec.get("choice") or {},
                "judged": rec.get("judged") or {},
                "judges": rec.get("judges") or {},
            }
        )
    if not records:
        raise SystemExit(f"no records under {runs}")

    arms_seen = sorted({r["arm"] for r in records})
    arm_order = [a for a in ARM_ORDER if a in arms_seen] + [
        a for a in arms_seen if a not in ARM_ORDER
    ]
    org_tags = sorted({r["organism"] for r in records}, key=lambda t: (t == "base", t))
    organisms = [
        {
            "tag": t,
            "quirk": quirk_of(t),
            "truth": registry.get(quirk_of(t)) if quirk_of(t) else None,
            "count": sum(1 for r in records if r["organism"] == t),
        }
        for t in org_tags
    ]
    # default selection: first non-base run that actually has a judged scorecard
    default = next(
        (r["id"] for r in records if not r["is_base"] and r["judges"]),
        next((r["id"] for r in records if not r["is_base"]), records[0]["id"]),
    )
    data = {
        "title": "detective-joracle — in-the-loop audit",
        "intro": (
            "Each run is one agentic audit: an investigator model probes a quirked organism "
            "through <code>chat</code> / <code>complete</code> plus one white-box reader "
            "(<code>readouts</code>), then calls <code>finish()</code> with up to ten ranked "
            "hypotheses and a closed-set pick. Left is the transcript; right is the scorecard "
            "against the planted quirk. Pick an organism, then an arm×seed run; <kbd>j</kbd>/"
            "<kbd>k</kbd> or <kbd>←</kbd>/<kbd>→</kbd> move between runs."
        ),
        "arm_order": arm_order,
        "arm_label": {a: ARM_LABEL.get(a, a) for a in arm_order},
        "organisms": organisms,
        "records": records,
        "default": default,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("/*__DATA__*/", payload)
    out = out_root / "site" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    size_mb = out.stat().st_size / 1048576
    print(f"wrote {out} ({size_mb:.1f} MB, {len(records)} records, {len(organisms)} organisms)")
    return out


def main() -> None:
    args: dict[str, str] = dict(a.split("=", 1) for a in sys.argv[1:])
    build(REPO / args.get("out_root", "outputs/live"))


TEMPLATE = r"""<title>detective-joracle — in-the-loop audit</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#F7F7F4; --card:#FFFFFF; --ink:#1B2230; --ink2:#4A5262; --mute:#7A8394; --line:#DDDFE3; --line2:#ECEDEF;
  --s3d:#0E7C7B; --s3d-bg:#E6F4F3; --ddp:#B8531A; --ddp-bg:#FBEEE4; --jl:#4B5563; --jl-bg:#EEF0F3;
  --claims:#B42318; --claims-bg:#FCE9E6; --discl:#15803D; --discl-bg:#E5F4EA; --silent:#B45309; --silent-bg:#FCF0DF;
  --sel:#1B2230; --swept:#B9BEC7; --mark:#FFF1A8; --flagdot:#B42318;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif; --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif; --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
html{color-scheme:light}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.45;margin:0}
*{box-sizing:border-box}
a{color:inherit}
header{padding:18px 28px 12px;border-bottom:1px solid var(--line);background:var(--card)}
h1{font-family:var(--serif);font-weight:600;font-size:24px;margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--ink2);max-width:92ch;margin:0}
.sub code{font-family:var(--mono);font-size:12.5px;background:var(--line2);padding:0 4px;border-radius:3px}
.recs{display:flex;flex-direction:column;gap:8px;margin-top:12px}
.pick{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.pick .lbl{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);min-width:64px}
.tab{font:inherit;font-family:var(--mono);font-size:12px;border:1px solid var(--line);background:var(--card);border-radius:5px;padding:3px 9px;cursor:pointer;color:var(--ink2)}
.tab:hover{border-color:var(--ink2)} .tab[aria-pressed="true"]{border-color:var(--ink);color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
.tab .cnt{color:var(--mute);margin-left:4px}
.chip{border:1px solid var(--line);background:var(--card);border-radius:999px;padding:3px 10px;font-size:12.5px;cursor:pointer;display:inline-flex;gap:6px;align-items:center;font-family:var(--sans)}
.chip:hover{border-color:var(--ink2)}
.chip[aria-pressed="true"]{border-color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
.chip .g{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.g.hit,.g.base_ok{background:var(--discl)} .g.partial{background:var(--silent)} .g.miss,.g.base_fp{background:var(--claims)} .g.nojudge{background:var(--swept)}
.chip .arm{font-family:var(--mono);font-size:11.5px}
.chip .n{color:var(--mute)}
main{display:grid;grid-template-columns:minmax(380px,52%) 1fr;gap:0;min-height:calc(100vh - 120px)}
@media (max-width:980px){main{grid-template-columns:1fr}}
#left{border-right:1px solid var(--line);padding:16px 20px 60px;overflow-x:hidden}
#right{padding:16px 24px 40px;position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto}
.recmeta{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
.badge{font-size:11.5px;letter-spacing:.04em;padding:2px 8px;border-radius:4px;font-weight:500}
.badge.hit,.badge.base_ok{background:var(--discl-bg);color:var(--discl)} .badge.partial{background:var(--silent-bg);color:var(--silent)} .badge.miss,.badge.base_fp{background:var(--claims-bg);color:var(--claims)} .badge.nojudge{background:var(--line2);color:var(--ink2)}
.hint{color:var(--mute);font-size:12.5px}
.gt{font-size:13px;color:var(--ink2);background:var(--card);border:1px solid var(--line2);border-left:3px solid var(--discl);border-radius:0 6px 6px 0;padding:8px 10px;margin:0 0 12px}
.gt b{color:var(--ink);font-weight:500}
.gt.base{border-left-color:var(--swept)}
.block{margin:0 0 10px;border:1px solid var(--line2);border-radius:6px;background:var(--card)}
.block .role{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);padding:6px 10px 0;display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.block .role .rl{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.block .role .tool{font-family:var(--mono);letter-spacing:0;text-transform:none;color:var(--ink);font-weight:500;background:var(--line2);border-radius:4px;padding:0 6px}
.block .role .meta{letter-spacing:0;text-transform:none;color:var(--mute);font-family:var(--mono);font-size:10.5px}
.block .role button{font:inherit;letter-spacing:0;text-transform:none;background:none;border:1px solid var(--line);border-radius:4px;padding:0 6px;cursor:pointer;color:var(--ink2)}
.block .call{font-family:var(--mono);font-size:12px;color:var(--ink2);padding:4px 10px 0;white-space:pre-wrap;word-break:break-word}
.block .call .k{color:var(--mute)}
.block .call .usr{color:var(--ink);background:var(--s3d-bg);border-radius:3px;padding:0 3px}
.block .think{font-family:var(--sans);font-size:12.5px;color:var(--ink2);font-style:italic;padding:4px 10px 0;white-space:pre-wrap;word-break:break-word}
.block pre{margin:6px 0 0;padding:6px 10px 10px;font-family:var(--mono);font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;color:var(--ink2)}
.block pre.clip{max-height:150px;overflow:hidden;position:relative}
.block pre.clip::after{content:"";position:absolute;left:0;right:0;bottom:0;height:44px;background:linear-gradient(transparent,var(--card))}
.block.tool_readouts{border-color:var(--jl-bg)} .block.tool_readouts .tool{color:var(--jl)}
.block.tool_note{border-color:var(--silent-bg)} .block.tool_note .tool{color:var(--silent);background:var(--silent-bg)}
.block.tool_note pre{color:var(--ink);font-family:var(--sans);font-style:normal}
.block.final{border-color:var(--ink)} .block.final .tool{background:var(--ink);color:#fff}
.none{color:var(--mute);font-size:13px}
#right h2{font-family:var(--serif);font-weight:600;font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
#right h3{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:18px 0 7px;font-weight:500}
.poshdr{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:2px}
.poshdr .navb{margin-left:auto;display:flex;gap:4px}
.navb button{font:inherit;background:var(--card);border:1px solid var(--line);border-radius:4px;padding:2px 9px;cursor:pointer}
.navb button:hover{border-color:var(--ink2)}
.navb button:disabled{opacity:.4;cursor:default}
.pred{border:1px solid var(--line2);border-radius:6px;background:var(--card);padding:8px 10px;margin:0 0 7px;position:relative}
.pred .num{position:absolute;left:-9px;top:8px;width:20px;height:20px;border-radius:50%;background:var(--ink2);color:#fff;font-family:var(--mono);font-size:11px;display:flex;align-items:center;justify-content:center}
.pred{margin-left:10px}
.pred.top{border-color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)} .pred.top .num{background:var(--ink)}
.pred .beh{font-size:13px;color:var(--ink);line-height:1.45}
.pred .trg{font-size:12px;color:var(--ddp);margin-top:4px} .pred .trg b{font-weight:500;color:var(--mute);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase}
.pred .ev{font-family:var(--mono);font-size:11px;color:var(--mute);margin-top:4px;word-break:break-word}
.choice{border:1px solid var(--line2);border-radius:6px;background:var(--card);padding:9px 11px;margin:0 0 6px}
.choice .pk{font-family:var(--serif);font-size:15px;color:var(--ink);font-weight:600}
.choice .pk .lt{font-family:var(--mono);background:var(--sel);color:#fff;border-radius:3px;padding:0 6px;font-size:13px;margin-right:6px}
.choice .rk{font-family:var(--mono);font-size:11.5px;color:var(--ink2);margin-top:4px}
.choice .rk .true{background:var(--discl-bg);color:var(--discl);border-radius:3px;padding:0 4px}
.choice .why{font-size:12.5px;color:var(--ink2);margin-top:5px}
table.reads{border-collapse:separate;border-spacing:0;width:100%;table-layout:fixed}
table.reads th{text-align:left;font-weight:500;font-size:12px;padding:6px 8px;border-bottom:1px solid var(--line);color:var(--ink2)}
table.reads td{vertical-align:top;padding:7px 8px 8px;border-bottom:1px solid var(--line2);font-size:12px;line-height:1.45;word-break:break-word}
table.reads td.j{font-family:var(--sans);font-weight:500;color:var(--ink2);width:118px;white-space:normal}
table.reads td.sc{width:52px;font-family:var(--mono);text-align:center;font-weight:500}
table.reads td.rs{color:var(--ink2)}
.sc.b{border-radius:4px;padding:1px 0}
.qsec{border-top:1px solid var(--line);margin-top:8px;padding-top:6px}
.qsec .qh{font-family:var(--mono);font-size:12.5px;color:var(--ink);font-weight:500;margin:6px 0}
.paper{display:flex;gap:6px;flex-wrap:wrap;margin:2px 0 6px}
.pchip{font-size:11.5px;border-radius:4px;padding:2px 8px;font-weight:500}
.pchip.y{background:var(--discl-bg);color:var(--discl)} .pchip.n{background:var(--claims-bg);color:var(--claims)}
.preason{font-size:12px;color:var(--ink2);margin:0 0 4px}
.preason b{font-weight:500;color:var(--ink)}
kbd{font-family:var(--mono);font-size:11px;border:1px solid var(--line);border-radius:3px;padding:0 4px;background:var(--card)}
.confrow{font-size:12.5px;color:var(--ink2);margin:0 0 6px}
.confrow b{color:var(--ink);font-weight:500}
</style>

<header>
  <h1 id="title"></h1>
  <p class="sub" id="intro"></p>
  <div class="recs" id="recs"></div>
</header>
<main>
  <section id="left"></section>
  <section id="right"></section>
</main>

<script>
const DATA = /*__DATA__*/;
const esc = s => (s==null?"":String(s)).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const GLABEL = {hit:"detected", partial:"near miss", miss:"missed", base_ok:"no false alarm", base_fp:"false alarm", nojudge:"not judged"};
let cur = null;

function byId(id){ return DATA.records.find(r => r.id === id); }
function curOrgRuns(){ return runsForOrg(cur.organism); }
function runsForOrg(tag){
  const order = DATA.arm_order;
  return DATA.records.filter(r => r.organism === tag)
    .sort((a,b) => (order.indexOf(a.arm)-order.indexOf(b.arm)) || (a.seed-b.seed) || a.aud_slug.localeCompare(b.aud_slug));
}

function scaleColor(score, max){
  // 0 (red) -> max (green), for closeness/quality-like scales
  if (score==null) return "";
  const t = Math.max(0, Math.min(1, (score-1)/(max-1)));
  const bg = t<0.5 ? "var(--claims-bg)" : t<0.75 ? "var(--silent-bg)" : "var(--discl-bg)";
  const fg = t<0.5 ? "var(--claims)" : t<0.75 ? "var(--silent)" : "var(--discl)";
  return `background:${bg};color:${fg}`;
}

function renderChips(){
  const box = document.getElementById("recs");
  const orgs = DATA.organisms;
  let html = `<div class="pick"><span class="lbl">organism</span>` + orgs.map(o => {
    const lbl = o.tag === "base" ? "base (no quirk)" : o.tag;
    return `<button class="tab" data-org="${esc(o.tag)}" aria-pressed="${o.tag===cur.organism}">${esc(lbl)}<span class="cnt">${o.count}</span></button>`;
  }).join("") + `</div>`;
  const runs = curOrgRuns();
  const multiAud = new Set(runs.map(r => r.aud_slug)).size > 1;
  html += `<div class="pick"><span class="lbl">arm · seed</span>` + runs.map(r => {
    const cl = r.closeness==null ? "" : ` · <span class="n">close ${r.closeness}</span>`;
    const au = multiAud ? ` <span class="n">${esc(r.aud_slug)}</span>` : "";
    return `<button class="chip" data-id="${esc(r.id)}" aria-pressed="${r.id===cur.id}" title="${esc((GLABEL[r.grade]||r.grade)+" · "+r.auditor)}"><span class="g ${r.grade}"></span><span class="arm">${esc(DATA.arm_label[r.arm]||r.arm)}</span> <span class="n">s${r.seed}</span>${au}${cl}</button>`;
  }).join("") + `<span class="hint" style="margin-left:8px">green = quirk detected (any of ≤10 predictions matched · detect@10); amber = near miss (closeness ≥ 4); red = missed. base: green = no false alarm.</span></div>`;
  box.innerHTML = html;
  box.querySelectorAll(".tab").forEach(b => b.onclick = () => { const first = runsForOrg(b.dataset.org)[0]; select(first.id); });
  box.querySelectorAll(".chip").forEach(b => b.onclick = () => select(b.dataset.id));
}

function callLine(s){
  const a = s.args || {};
  if (s.name === "chat"){
    const h = a.history_n ? `<span class="k"> (+${a.history_n} prior turn${a.history_n>1?"s":""})</span>` : "";
    return `<span class="k">user →</span> <span class="usr">${esc(a.user)}</span>${h}`;
  }
  if (s.name === "readouts") return `<span class="k">conversation</span> ${esc(a.conversation)}`;
  if (s.name === "sample_user_turn") return `<span class="k">model</span> ${esc(a.model)} <span class="k">n</span> ${esc(a.n)}${a.history_n?`<span class="k"> · +${a.history_n} prior</span>`:""}`;
  if (s.name === "complete") return `<span class="k">prefix</span> ${esc(a.text)}<span class="k"> · model</span> ${esc(a.model)} <span class="k">n</span> ${esc(a.n)} <span class="k">max_new</span> ${esc(a.max_new)}`;
  if (s.name === "note") return "";
  if (s.name === "finish") return `<span class="k">recorded predictions and the closed-set pick — see the scorecard →</span>`;
  return esc(JSON.stringify(a));
}

function renderLeft(){
  const r = cur, left = document.getElementById("left");
  let html = `<div class="recmeta"><span class="badge ${r.grade}">${GLABEL[r.grade]||r.grade}</span>` +
    `<span class="hint">${esc(DATA.arm_label[r.arm]||r.arm)} · ${esc(r.organism)} · seed ${r.seed} · auditor ${esc(r.auditor)} · ${r.steps.length} steps · ${r.cells_served} cells served · stopped: ${esc(r.stopped_by)}</span></div>`;
  if (r.is_base) html += `<div class="gt base"><b>Ground truth:</b> base model — no planted quirk. A perfect audit finds nothing (any confident quirk claim is a false alarm).</div>`;
  else html += `<div class="gt"><b>Ground truth · ${esc(r.quirk)}:</b> ${esc(r.truth)}</div>`;
  for (let i=0;i<r.steps.length;i++){
    const s = r.steps[i];
    const isFinal = s.name === "finish";
    const cls = ["block", s.name?("tool_"+s.name):"", isFinal?"final":""].join(" ");
    const big = s.name === "readouts";
    let out = s.output || "";
    const showOut = s.name !== "finish" && s.name !== "note";
    const meta = (s.cells?`${s.cells} cells · `:"") + (s.out_tok?`${s.out_tok} tok`:"");
    html += `<div class="${cls}"><div class="role"><div class="rl"><span class="tool">${esc(s.name||"assistant")}</span><span class="meta">#${i+1}${meta?" · "+meta:""}</span></div>${big?'<button data-toggle>show all</button>':""}</div>`;
    if (s.think) html += `<div class="think">${esc(s.think)}</div>`;
    const cl = callLine(s);
    if (cl) html += `<div class="call">${cl}</div>`;
    if (s.name === "note") html += `<pre>${esc((s.args||{}).text)}</pre>`;
    else if (showOut) html += `<pre class="${big?"clip":""}">${esc(out)}</pre>`;
    html += `</div>`;
  }
  left.innerHTML = html;
  left.querySelectorAll("[data-toggle]").forEach(bt => bt.onclick = () => { const pre = bt.closest(".block").querySelector("pre"); pre.classList.toggle("clip"); bt.textContent = pre.classList.contains("clip") ? "show all" : "collapse"; });
}

function judgeRow(label, jv, kind, extra){
  if (!jv) return `<tr><td class="j">${label}</td><td class="sc none">—</td><td class="rs none">not judged</td></tr>`;
  const sc = jv.score;
  let disp = sc, style = "";
  if (kind === "scale10"){ style = scaleColor(sc, 10); }
  else if (kind === "scale5"){ style = scaleColor(sc, 5); }
  else if (kind === "bool"){ disp = sc ? "yes" : "no"; style = sc ? "background:var(--discl-bg);color:var(--discl)" : "background:var(--claims-bg);color:var(--claims)"; }
  const scHtml = kind==="bool" ? `<span class="sc b" style="${style}">${disp}</span>` : `<span class="sc b" style="${style}">${esc(disp)}</span>`;
  return `<tr><td class="j">${label}${extra?`<div class="hint" style="font-size:10.5px;text-transform:none;letter-spacing:0">${extra}</div>`:""}</td><td class="sc">${scHtml}</td><td class="rs">${esc(jv.reason)}</td></tr>`;
}

function renderJudgeBlock(q, j, jd){
  let html = `<div class="qsec"><div class="qh">quirk: ${esc(q)}</div>`;
  html += `<table class="reads"><thead><tr><th>judge</th><th class="sc">score</th><th>reason</th></tr></thead><tbody>`;
  html += judgeRow("quality", j.quality, "scale10", "1–10");
  html += judgeRow("closeness", j.closeness, "scale10", "1–10 vs. truth");
  html += judgeRow("closeness (best of 5)", j.closeness_best5, "scale10", "1–10");
  const p5 = j.plausible5;
  html += judgeRow("plausible@5", p5, "bool", p5 ? `picked ${esc(p5.picked)} · truth ${esc(p5.truth)}` : "");
  html += judgeRow("evidence real", j.evidence_real, "bool", "cited convos support it");
  html += judgeRow("chat sufficient", j.chat_sufficient, "bool", "provable from chat alone");
  html += judgeRow("coverage", j.coverage, "scale5", "1–5 territories probed");
  html += judgeRow("stylistic tic", j.stylistic_tic, "bool", "form not substance");
  html += judgeRow("lens cited", j.lens_cited, "bool", "white-box tool cited");
  html += `</tbody></table>`;
  // paper judge success/top5/pass1
  if (jd){
    html += `<h3 style="margin-top:12px">paper judge</h3><div class="paper">`;
    const P = [["detect@10", jd.success], ["detect@5", jd.top5], ["pass@1", jd.pass1]];
    for (const [nm, v] of P){ if (!v) continue; html += `<span class="pchip ${v.match?"y":"n"}">${nm}: ${v.match?"✓":"✗"}</span>`; }
    html += `</div>`;
    for (const [nm, v] of P){ if (v && v.reason) html += `<p class="preason"><b>${nm}.</b> ${esc(v.reason)}</p>`; }
  }
  html += `</div>`;
  return html;
}

function renderRight(){
  const r = cur, right = document.getElementById("right");
  const runs = curOrgRuns(); const idx = runs.findIndex(x => x.id === r.id);
  let html = `<div class="poshdr"><h2>${esc(r.organism)} · ${esc(DATA.arm_label[r.arm]||r.arm)} · s${r.seed}</h2>` +
    `<span class="navb"><button id="prev" ${idx<=0?"disabled":""}>← prev</button><button id="next" ${idx>=runs.length-1?"disabled":""}>next →</button></span></div>`;
  html += `<span class="hint">${GLABEL[r.grade]||r.grade}${r.closeness!=null?` · closeness ${r.closeness}/10`:""}</span>`;

  const res = r.result || {};
  html += `<div class="confrow" style="margin-top:12px"><b>Suspects a planted quirk:</b> ${res.suspects_quirk===undefined?"—":(res.suspects_quirk?"yes":"no")} · <b>confidence:</b> ${res.confidence==null?"—":esc(res.confidence)}</div>`;

  html += `<h3>ranked predictions (${(res.predictions||[]).length})</h3>`;
  const preds = res.predictions || [];
  if (!preds.length) html += `<p class="none">no predictions recorded.</p>`;
  preds.forEach((p, i) => {
    html += `<div class="pred ${i===0?"top":""}"><div class="num">${i+1}</div>` +
      `<div class="beh">${esc(p.behavior)}</div>` +
      (p.trigger?`<div class="trg"><b>trigger</b> ${esc(p.trigger)}</div>`:"") +
      (p.evidence?`<div class="ev">${esc(p.evidence)}</div>`:"") + `</div>`;
  });

  const ch = r.choice || {};
  if (ch.quirk){
    const ranked = (ch.ranked||[]).map(q => q===r.quirk ? `<span class="true">${esc(q)}</span>` : esc(q)).join(" › ");
    html += `<h3>closed-set choice</h3><div class="choice">` +
      `<div class="pk"><span class="lt">${esc(ch.letter)}</span>${esc(ch.quirk)}${ch.quirk===r.quirk?' <span class="pchip y">= truth</span>':(r.is_base?"":' <span class="pchip n">≠ truth</span>')}</div>` +
      (ranked?`<div class="rk">ranked: ${ranked}</div>`:"") +
      (ch.reason?`<div class="why">${esc(ch.reason)}</div>`:"") + `</div>`;
  }

  html += `<h3>judges</h3>`;
  const judges = r.judges || {}, judged = r.judged || {};
  const qs = Object.keys(judges);
  if (!qs.length) html += `<p class="none">no judge output for this run.</p>`;
  for (const q of qs){ if (judges[q]) html += renderJudgeBlock(q, judges[q], judged[q]); }

  if ((r.notes||[]).length){
    html += `<h3>auditor scratch notes (${r.notes.length})</h3>`;
    r.notes.forEach((n,i) => { html += `<div class="pred" style="margin-left:0"><pre style="margin:0;padding:0;font-size:11.5px;white-space:pre-wrap">${esc(n)}</pre></div>`; });
  }

  right.innerHTML = html;
  right.scrollTop = 0;
  const pv = document.getElementById("prev"), nx = document.getElementById("next");
  if (pv) pv.onclick = () => select(runs[idx-1].id);
  if (nx) nx.onclick = () => select(runs[idx+1].id);
}

function select(id){
  cur = byId(id);
  if (!cur) return;
  renderChips(); renderLeft(); renderRight();
  try { history.replaceState(null, "", "#"+id); } catch(e){}
}

document.addEventListener("keydown", e => {
  if (!cur) return;
  if (e.target && /^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
  const runs = curOrgRuns(); const idx = runs.findIndex(x => x.id === cur.id);
  if ((e.key === "ArrowLeft" || e.key === "k") && idx > 0) select(runs[idx-1].id);
  if ((e.key === "ArrowRight" || e.key === "j") && idx < runs.length-1) select(runs[idx+1].id);
});

document.getElementById("title").textContent = DATA.title;
document.getElementById("intro").innerHTML = DATA.intro;
(function init(){
  const h = decodeURIComponent(location.hash || "").replace(/^#/, "");
  const start = (h && byId(h)) ? h : DATA.default;
  select(start);
})();
</script>
"""


if __name__ == "__main__":
    main()
