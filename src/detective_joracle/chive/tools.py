"""The investigator's tools for one CHIVE incident: tokens, search, inspect, notebook, experiment,
submit. ``ChiveTools`` is the tools object ``agent.loop.run_tool_loop_async`` drives.

Every call is logged (``log``: what the agent saw; ``events``: the positions each call read,
for the viewer). A malformed call raises ``ValueError`` (the loop shows it to the agent as
``ERROR: ...``); any other failure is caught, logged and returned as text, never crashing a run.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..agent.loop import ToolLog
from .data import Incident
from .executor import Executor
from .readers import Reader

WINDOW = {"olens": 2, "jlens": 2, "nla": 4}  # neighbouring positions an inspect shows by default
MAX_WINDOW = 8
SEARCH_K = 20
SAMPLES_PER_HIT = 3
CLIP = 700  # characters of one readout in a search hit
SHOW_TOKENS_MAX = 400
SUBMIT = "submit_hypothesis"


def _schema(
    name: str, description: str, props: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def schemas(reader: Reader | None, experiments: int) -> list[dict[str, Any]]:
    out = [
        _schema(
            "show_tokens",
            "The numbered tokens (t<pos>) of one turn, or of the span around a phrase, so you can "
            "name exact positions. Turns are T1..Tn as listed; 'response' is the model's response.",
            {
                "turn": {"type": "string", "description": "T1..Tn or 'response'"},
                "phrase": {
                    "type": "string",
                    "description": "optional: only the tokens of this exact "
                    "phrase (first occurrence in that turn) and a few around it",
                },
            },
            ["turn"],
        )
    ]
    if reader is not None:
        out += [
            _schema(
                "search",
                "Every readout of this conversation matching a case-insensitive Python regex. "
                "Returns match counts by layer and by turn, then the positions with the most "
                f"matching readouts (up to {SAMPLES_PER_HIT} readouts each). Local and instant; "
                "try several patterns (synonyms, the words a readout would use).",
                {
                    "pattern": {"type": "string"},
                    "positions": {
                        "type": "string",
                        "description": "optional range, e.g. 't40-t90'",
                    },
                    "layers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "optional: only these layers",
                    },
                    "k": {
                        "type": "integer",
                        "description": f"positions returned (default {SEARCH_K})",
                    },
                },
                ["pattern"],
            ),
            _schema(
                "inspect",
                "One position in full: every layer, every sample, with the text around it; "
                "neighbouring positions are shown compactly (one sample each).",
                {
                    "position": {"type": "integer", "description": "the t-number"},
                    "window": {
                        "type": "integer",
                        "description": f"neighbours each side (default {WINDOW[reader.kind]}, "
                        f"max {MAX_WINDOW})",
                    },
                    "layers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "optional: only these layers",
                    },
                },
                ["position"],
            ),
        ]
    out.append(
        _schema(
            "notebook",
            "Your notes: write (replace), append or read. They persist when old tool "
            "results are trimmed from your context.",
            {
                "action": {"type": "string", "enum": ["write", "append", "read"]},
                "text": {"type": "string"},
            },
            ["action"],
        )
    )
    if experiments:
        out.append(
            _schema(
                "run_experiment",
                f"Edit the conversation and re-run the target model {Executor.N} times; you get how "
                "often the behaviour occurs vs the baseline, and a few graded responses. You have "
                f"{experiments} experiment(s). `turns` lists the whole edited conversation: "
                '{"keep": T} keeps turn T verbatim; {"keep": T, "replace": [{"old": "exact text", '
                '"new": "replacement"}]} changes text inside turn T (preferred: one feature at a '
                'time); {"role": "user"|"assistant"|"system", "content": "..."} adds a turn. It '
                "must end with a user turn.",
                {
                    "turns": {"type": "array", "items": {"type": "object"}},
                    "hypothesis_tested": {"type": "string"},
                    "rationale": {
                        "type": "string",
                        "description": "why this edit, and what each outcome would tell you",
                    },
                },
                ["turns", "hypothesis_tested", "rationale"],
            )
        )
    out.append(
        _schema(
            SUBMIT,
            "Your final answer: the one feature of the conversation that drives the "
            "behaviour and how, the evidence, and the strongest rival with what ruled it out.",
            {
                "hypothesis": {
                    "type": "string",
                    "description": "ONE mechanism, in 1-4 sentences. Do "
                    "not list several candidates: a list is graded on its weakest item. Put "
                    "the strongest alternative in `rival`.",
                },
                "evidence": {"type": "string", "description": "the key evidence, briefly"},
                "rival": {"type": "string"},
                "rival_ruled_out_by": {"type": "string"},
            },
            ["hypothesis"],
        )
    )
    return out


def parse_range(text: str, n: int) -> set[int]:
    m = re.fullmatch(r"\s*t?(\d+)\s*(?:-\s*t?(\d+))?\s*", text or "")
    if not m:
        raise ValueError(f"positions must look like 't40-t90' or 't40'; got {text!r}")
    lo = int(m.group(1))
    hi = int(m.group(2) or lo)
    if not 0 <= lo <= hi < n:
        raise ValueError(f"positions must be within t0-t{n - 1}; got {text!r}")
    return set(range(lo, hi + 1))


@dataclass
class ChiveTools:
    inc: Incident
    reader: Reader | None
    executor: Executor | None = None
    experiments: int = 0
    log: list[ToolLog] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    notebook_text: str = ""
    finished: dict[str, Any] | None = None
    experiments_run: list[dict[str, Any]] = field(default_factory=list)

    async def call(self, name: str, args: dict[str, Any]) -> str:
        handlers = {
            "show_tokens": self.show_tokens,
            "search": self.search,
            "inspect": self.inspect,
            "notebook": self.notebook,
            SUBMIT: self.submit,
        }
        known = name in handlers or (name == "run_experiment" and self.experiments > 0)
        if not known or (name in ("search", "inspect") and self.reader is None):
            raise ValueError(f"no tool {name!r} in this investigation")
        event: dict[str, Any] = {"tool": name, "args": args}
        try:
            if name == "run_experiment":
                out = await self.run_experiment(args, event)
            else:
                out = handlers[name](args, event)
        except ValueError as e:
            event["error"] = str(e)
            self.events.append(event)
            raise
        except Exception as e:  # a bug or outage: logged, shown to the agent, never fatal
            out = f"tool error: {type(e).__name__}: {str(e)[:300]}"
            event["error"] = out
        self.events.append(event)
        self.log.append(ToolLog(name=name, args=args, output=out))
        return out

    # ------------------------------------------------------------------ tools

    def show_tokens(self, args: dict[str, Any], event: dict[str, Any]) -> str:
        turn = str(args.get("turn", "")).strip()
        inc = self.inc
        if turn.lower() == "response":
            span = list(range(inc.response_start, inc.n_tokens))
        else:
            m = re.fullmatch(r"T?(\d+)", turn, re.I)
            if not m:
                raise ValueError("turn must be T1..Tn or 'response'")
            t = int(m.group(1))
            span = [
                p
                for p, (ti, role) in enumerate(inc.turn_of)
                if ti == t and role != "control" and p < inc.response_start
            ]
            if not span:
                raise ValueError(f"no turn T{t}; turns are T1..T{len(inc.messages)}")
        phrase = str(args.get("phrase") or "")
        if phrase:
            text = "".join(inc.pieces[p] for p in span)
            at = text.find(phrase)
            if at < 0:
                raise ValueError(f"phrase not found in {turn}: {phrase[:80]!r}")
            offs, acc = [], 0
            for p in span:
                offs.append(acc)
                acc += len(inc.pieces[p])
            first = next(i for i, o in enumerate(offs) if o + len(inc.pieces[span[i]]) > at)
            last = next(
                i for i, o in enumerate(offs) if o + len(inc.pieces[span[i]]) >= at + len(phrase)
            )
            span = span[max(0, first - 5) : last + 6]
        span = span[:SHOW_TOKENS_MAX]
        event["positions"] = span
        return " ".join(f"t{p}{inc.pieces[p]!r}" for p in span)

    def search(self, args: dict[str, Any], event: dict[str, Any]) -> str:
        assert self.reader is not None
        try:
            rx = re.compile(str(args.get("pattern") or ""), re.I)
        except re.error as e:
            raise ValueError(f"bad regex: {e}") from e
        if not rx.pattern:
            raise ValueError("give a `pattern`")
        allowed = (
            parse_range(args["positions"], self.inc.n_tokens) if args.get("positions") else None
        )
        layers = set(args.get("layers") or []) or None
        k = max(1, min(int(args.get("k") or SEARCH_K), 100))
        hits: dict[int, list[tuple[int, str]]] = {}
        n_scanned = 0
        for pos, per_layer in self.reader.cells.items():
            if allowed is not None and pos not in allowed:
                continue
            for layer, samples in per_layer.items():
                if layers is not None and layer not in layers:
                    continue
                for s in samples:
                    n_scanned += 1
                    if rx.search(s):
                        hits.setdefault(pos, []).append((layer, s))
        n_hits = sum(len(v) for v in hits.values())
        top = sorted(hits, key=lambda p: (-len(hits[p]), p))[:k]
        event["positions"] = top
        if not hits:
            return f"Scanned {n_scanned} readouts; none match {rx.pattern!r}. Try other wording."
        by_layer = Counter(layer for v in hits.values() for layer, _ in v)
        by_turn = Counter(self.inc.where(p).split("(")[-1].rstrip(")") for p in hits)
        lines = [
            f"Scanned {n_scanned} readouts; {n_hits} match at {len(hits)} positions.",
            "By layer: " + ", ".join(f"L{x} {by_layer[x]}" for x in sorted(by_layer)),
            "By turn: " + ", ".join(f"{t} {c}" for t, c in by_turn.most_common()),
            f"Top {len(top)} positions (most matching readouts first):",
        ]
        for p in top:
            lines.append(
                f"{self.inc.where(p)}: {len(hits[p])} matching -- {self.inc.context(p, 6)}"
            )
            for layer, s in sorted(hits[p])[:SAMPLES_PER_HIT]:
                lines.append(f"   L{layer}: {s[:CLIP]}")
        return "\n".join(lines)

    def inspect(self, args: dict[str, Any], event: dict[str, Any]) -> str:
        assert self.reader is not None
        pos = int(str(args.get("position", -1)).strip().lstrip("tT"))
        if not 0 <= pos < self.inc.n_tokens:
            raise ValueError(f"position must be within t0-t{self.inc.n_tokens - 1}")
        w = max(0, min(int(args.get("window", WINDOW[self.reader.kind])), MAX_WINDOW))
        layers = set(args.get("layers") or []) or None
        event["positions"] = [pos]
        event["window"] = w
        lines = [f"=== {self.inc.where(pos)} ===", f"context: {self.inc.context(pos)}"]
        for layer, samples in sorted(self.reader.cell(pos).items()):
            if layers is None or layer in layers:
                lines.append(
                    f"L{layer}: " + " | ".join(f"({i}) {s}" for i, s in enumerate(samples, start=1))
                )
        neighbours = [
            p for p in range(pos - w, pos + w + 1) if p != pos and 0 <= p < self.inc.n_tokens
        ]
        if neighbours:
            lines.append("\n--- neighbours (one sample per layer) ---")
            for p in neighbours:
                lines.append(f"{self.inc.where(p)}")
                for layer, samples in sorted(self.reader.cell(p).items()):
                    if (layers is None or layer in layers) and samples:
                        lines.append(f"   L{layer}: {samples[0]}")
        return "\n".join(lines)

    def notebook(self, args: dict[str, Any], event: dict[str, Any]) -> str:
        action, text = args.get("action"), str(args.get("text") or "")
        if action == "write":
            self.notebook_text = text
        elif action == "append":
            self.notebook_text += ("\n" if self.notebook_text else "") + text
        elif action != "read":
            raise ValueError("action must be write, append or read")
        if action != "read":
            return f"Saved ({len(self.notebook_text)} characters)."
        return self.notebook_text or "(empty)"

    async def run_experiment(self, args: dict[str, Any], event: dict[str, Any]) -> str:
        if self.executor is None or len(self.experiments_run) >= self.experiments:
            raise ValueError("no experiments left")
        res = await self.executor.run(self.inc, args.get("turns"))
        rec = {
            "hypothesis_tested": args.get("hypothesis_tested", ""),
            "rationale": args.get("rationale", ""),
            **res,
        }
        self.experiments_run.append(rec)
        event["experiment"] = {k: rec[k] for k in ("k", "n", "rate")}
        base = self.inc.k / max(1, self.inc.n)
        shown = "".join(
            f"\n--- response {i + 1} [behaviour: {'yes' if v else 'no'}] ---\n{c[:1500]}"
            for i, (c, v) in enumerate(zip(res["completions"], res["verdicts"], strict=True))
            if i < 4
        )
        return (
            f"The behaviour occurred in {res['k']} of {res['n']} responses ({res['rate']:.0%}); "
            f"baseline {self.inc.k}/{self.inc.n} ({base:.0%}).{shown}"
        )

    def submit(self, args: dict[str, Any], event: dict[str, Any]) -> str:
        if not str(args.get("hypothesis") or "").strip():
            raise ValueError(
                "give `hypothesis`: the one feature that drives the behaviour, and how"
            )
        fields = ("hypothesis", "evidence", "rival", "rival_ruled_out_by")
        self.finished = {k: str(args.get(k) or "") for k in fields}
        event["missing_fields"] = [k for k in fields if not self.finished[k].strip()]
        return "Submitted."


def render_experiments(exps: list[dict[str, Any]]) -> str:
    """An experiment log as a fresh agent (two-agent B) reads it: edit, result, responses."""
    parts = []
    for i, e in enumerate(exps, start=1):
        parts.append(
            f"## Experiment {i}\nEdit (turns): {json.dumps(e.get('turns'), ensure_ascii=False)}\n"
            f"Result: the behaviour occurred in {e['k']} of {e['n']} responses."
        )
        for j, (c, v) in enumerate(zip(e["completions"], e["verdicts"], strict=True)):
            if j < 4:
                parts.append(
                    f"--- response {j + 1} [behaviour: {'yes' if v else 'no'}] ---\n{c[:1500]}"
                )
    return "\n\n".join(parts)
