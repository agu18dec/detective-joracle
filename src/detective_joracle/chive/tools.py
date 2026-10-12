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

WINDOW = {
    "olens": 2,
    "jlens": 2,
    "nla": 4,
    "boundary": 2,
}  # neighbouring positions an inspect shows by default
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


def schemas(
    reader: Reader | None,
    experiments: int,
    letters: list[str] | None = None,
    continuation: bool = False,
    text_first: bool = False,
    aux: Reader | None = None,
    edits: list[str] | None = None,
) -> list[dict[str, Any]]:
    """The tools; with `letters`, submit also requires the chosen candidate (candidate selection);
    with `edits` (yes/no), submit and commit_text_hypothesis also require a probability per edit."""
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
    if aux is not None and reader is not None:  # a second method, to corroborate the first
        for sch in schemas(aux, 0)[1:3]:  # its search + inspect, renamed
            f = sch["function"]
            f["name"] = f"{aux.name}_{f['name']}"
            f["description"] = f"[{aux.name} readouts] " + f["description"]
            out.append(sch)
    if text_first and reader is not None:
        out.append(
            _schema(
                "commit_text_hypothesis",
                "Commit your best transcript-only hypothesis before any readout tool (they are "
                "locked until you do). Call once.",
                {
                    "hypothesis": {"type": "string", "description": "ONE mechanism, 1-3 sentences"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                ["hypothesis", "confidence"],
            )
        )
    if continuation:
        out.append(
            _schema(
                "continue_from",
                "Sample how the target model continues the text from just after one token (4 "
                "samples, up to 120 tokens each): what it is about to write at that point.",
                {"position": {"type": "integer", "description": "token position t<pos>"}},
                ["position"],
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
    if letters:
        props = out[-1]["function"]["parameters"]
        props["properties"]["choice"] = {
            "type": "string",
            "enum": letters,
            "description": "the letter of the candidate cause you conclude is the cause",
        }
        props["properties"]["probabilities"] = {
            "type": "object",
            "description": "your probability that each candidate is the cause, one entry per "
            'letter, summing to 1 (e.g. {"A": 0.6, "B": 0.3, "C": 0.1})',
        }
        props["required"] = ["choice", "probabilities", "hypothesis"]
    if edits:
        pred = {
            "type": "object",
            "description": "for EVERY edit id, your probability that the edit changes the "
            'behaviour rate (YES), e.g. {"E1": 0.8, "E2": 0.1}',
        }
        for f in out:
            fn = f["function"]
            if fn["name"] in (SUBMIT, "commit_text_hypothesis"):
                fn["parameters"]["properties"]["predictions"] = pred
                fn["parameters"]["required"] = [*fn["parameters"]["required"], "predictions"]
    return out


def parse_predictions(args: dict[str, Any], ids: list[str]) -> dict[str, float]:
    """`predictions` as {edit id: P(YES)}, every id present, each a probability."""
    raw = args.get("predictions")
    if not isinstance(raw, dict):
        raise ValueError(f"give `predictions`: a probability for each of {', '.join(ids)}")
    got: dict[str, float] = {}
    for k, v in raw.items():
        try:
            got[str(k).strip().upper()] = float(v)
        except (TypeError, ValueError):
            continue
    missing = [i for i in ids if i not in got]
    if missing:
        raise ValueError(f"`predictions` is missing {', '.join(missing)}")
    if any(not 0.0 <= got[i] <= 1.0 for i in ids):
        raise ValueError("each prediction must be a probability between 0 and 1")
    return {i: got[i] for i in ids}


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
    letters: list[str] | None = None  # candidate selection: the valid choices
    edits: list[str] | None = None  # yes/no: the edit ids that need a prediction
    continuation: bool = False  # the continue_from tool (needs an executor for the target)
    text_first: bool = False  # readout tools locked until commit_text_hypothesis
    aux: Reader | None = None  # a second method's readouts: <name>_search / <name>_inspect
    committed: dict[str, Any] | None = None

    async def call(self, name: str, args: dict[str, Any]) -> str:
        handlers = {
            "show_tokens": self.show_tokens,
            "search": self.search,
            "inspect": self.inspect,
            "notebook": self.notebook,
            SUBMIT: self.submit,
        }
        if self.aux is not None and name in (f"{self.aux.name}_search", f"{self.aux.name}_inspect"):
            if self.text_first and self.committed is None:
                raise ValueError("readout tools are locked: call commit_text_hypothesis first")
            main, self.reader = self.reader, self.aux  # the same tools over the second reader
            first = len(self.events)
            try:
                return await self.call(name.rsplit("_", 1)[1], args)
            finally:
                self.reader = main
                for e in self.events[first:]:  # log the call under its real name
                    e["tool"] = name
        known = (
            name in handlers
            or (name == "run_experiment" and self.experiments > 0)
            or (name == "continue_from" and self.continuation)
            or (name == "commit_text_hypothesis" and self.text_first)
        )
        if not known or (name in ("search", "inspect") and self.reader is None):
            raise ValueError(f"no tool {name!r} in this investigation")
        if self.text_first and name in ("search", "inspect") and self.committed is None:
            raise ValueError("readout tools are locked: call commit_text_hypothesis first")
        event: dict[str, Any] = {"tool": name, "args": args}
        try:
            if name == "run_experiment":
                out = await self.run_experiment(args, event)
            elif name == "continue_from":
                out = await self.continue_from(args, event)
            elif name == "commit_text_hypothesis":
                out = self.commit(args)
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

    def commit(self, args: dict[str, Any]) -> str:
        if self.committed is not None:
            raise ValueError("already committed; continue with the readouts")
        if not str(args.get("hypothesis") or "").strip():
            raise ValueError("give `hypothesis`")
        preds = parse_predictions(args, self.edits) if self.edits else None
        self.committed = {
            "hypothesis": str(args["hypothesis"]),
            "confidence": str(args.get("confidence", "")),
        }
        if preds is not None:
            self.committed["predictions"] = preds
        return "Committed. Your readout tools are now unlocked."

    async def continue_from(self, args: dict[str, Any], event: dict[str, Any]) -> str:
        if self.executor is None:
            raise ValueError("continue_from is not available")
        pos = int(str(args.get("position", "")).lstrip("t") or -1)
        if not 0 <= pos < self.inc.n_tokens:
            raise ValueError(f"position must be within t0-t{self.inc.n_tokens - 1}")
        event["positions"] = [pos]
        outs = await self.executor.continue_tokens(self.inc.token_ids[: pos + 1])
        return f"Continuations after {self.inc.where(pos)}:" + "".join(
            f"\n--- sample {i + 1} ---\n{c if c.strip() else '(empty: the model ends its turn here)'}"
            for i, c in enumerate(outs)
        )

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
        fields: tuple[str, ...] = ("hypothesis", "evidence", "rival", "rival_ruled_out_by")
        if self.letters:
            choice = str(args.get("choice") or "").strip().strip(")").upper()
            if choice not in self.letters:
                raise ValueError(f"give `choice`: one of {', '.join(self.letters)}")
            args = {**args, "choice": choice}
            fields += ("choice",)
            probs = args.get("probabilities")
            if not isinstance(probs, dict):
                raise ValueError("give `probabilities`: one probability per candidate letter")
            vals = {
                str(k).strip().strip(")").upper(): float(v)
                for k, v in probs.items()
                if str(v).replace(".", "", 1).isdigit()
            }
            total = sum(max(0.0, v) for k, v in vals.items() if k in self.letters)
            if total <= 0:
                raise ValueError("`probabilities` must give positive weight to the candidates")
            norm = {k: max(0.0, vals.get(k, 0.0)) / total for k in self.letters}
        preds = parse_predictions(args, self.edits) if self.edits else None
        self.finished = {k: str(args.get(k) or "") for k in fields}
        if preds is not None:
            self.finished["predictions"] = preds
        if self.letters:
            self.finished["probabilities"] = norm
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
