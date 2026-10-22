"""CHIVE-style tools over a bank of MANY conversations (AuditBench v3).

The CHIVE harness (``chive/tools.py``) gives the agent ``search`` / ``inspect`` over one incident's dense
readouts. AuditBench needs the same tools over every conversation the agent can see: the 50 held-out
prompts in the static eval, or each conversation the agent ran in the agentic eval (banked lazily). The
semantics follow CHIVE with the ``recur`` view on: search reports matches by layer and by conversation,
the pattern's base rate over the whole bank, and per position how many of its readouts match, across
how many layers, whether neighbours match too, and a RECURRING vs SINGLE/WEAK verdict (the support count).
Positions are named ``<conversation>:t<pos>``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

SEARCH_K = 12
SAMPLES_PER_HIT = 3
CLIP = 260
WINDOW = 2
MAX_WINDOW = 6


@dataclass
class Conversation:
    """One banked conversation: decoded tokens, a region per token, readouts[layer][pos] = samples."""

    cid: str
    tokens: list[str]
    region: list[str]
    readouts: dict[int, dict[int, list[str]]] = field(default_factory=dict)

    def cell(self, pos: int) -> dict[int, list[str]]:
        return {e: per[pos] for e, per in sorted(self.readouts.items()) if per.get(pos)}

    def context(self, pos: int, width: int = 8) -> str:
        left = "".join(self.tokens[max(0, pos - width) : pos])
        right = "".join(self.tokens[pos + 1 : pos + 1 + width // 2])
        return f"{left!r} [{self.tokens[pos]!r}] {right!r}"

    def where(self, pos: int) -> str:
        return f"{self.cid}:t{pos} ({self.region[pos]}, token {self.tokens[pos]!r})"


def parse_bank_readouts(raw: Mapping[str, Mapping[str, list[str]]],
                        layers: set[int] | None = None) -> dict[int, dict[int, list[str]]]:
    """A CHIVE bank file's ``readouts`` ({"layer": {"pos": [samples]}}) -> int keys, optional layer filter."""
    return {int(e): {int(p): list(v) for p, v in per.items()}
            for e, per in raw.items() if layers is None or int(e) in layers}


def parse_range(text: str, n: int) -> set[int]:
    out: set[int] = set()
    for part in re.split(r"[,\s]+", text.strip()):
        m = re.fullmatch(r"t?(\d+)(?:-t?(\d+))?", part)
        if m:
            a = int(m.group(1))
            b = int(m.group(2) or a)
            out.update(range(max(0, a), min(n - 1, b) + 1))
    return out


class BankTools:
    """search / inspect / show_tokens over ``convs`` (cid -> Conversation). ``loader`` (optional) banks a
    conversation on first use (lazy banking in the agentic eval); ``on_bank`` records that it happened."""

    def __init__(self, convs: dict[str, Conversation],
                 loader: Callable[[str], Conversation] | None = None) -> None:
        self.convs = convs
        self.loader = loader
        self.queries: Counter[str] = Counter()  # how often each lens tool was called

    def _conv(self, cid: str) -> Conversation:
        cid = str(cid).strip()
        if cid not in self.convs:
            if self.loader is None:
                raise KeyError(f"no conversation {cid!r}; known: {', '.join(list(self.convs)[:8])}...")
            self.convs[cid] = self.loader(cid)
        return self.convs[cid]

    def _targets(self, which: Any) -> list[Conversation]:
        if not which:
            return list(self.convs.values())
        ids = which if isinstance(which, list) else re.split(r"[,\s]+", str(which).strip())
        return [self._conv(c) for c in ids if c]

    def show_tokens(self, args: dict[str, Any]) -> str:
        self.queries["show_tokens"] += 1
        c = self._conv(args.get("conversation", ""))
        return f"{c.cid}: " + " ".join(f"t{i}={t!r}" for i, t in enumerate(c.tokens))

    def search(self, args: dict[str, Any]) -> str:
        self.queries["search"] += 1
        try:
            rx = re.compile(str(args.get("pattern") or ""), re.I)
        except re.error as e:
            raise ValueError(f"bad regex: {e}") from e
        if not rx.pattern:
            raise ValueError("give a `pattern`")
        layers = set(args.get("layers") or []) or None
        k = max(1, min(int(args.get("k") or SEARCH_K), 60))
        hits: dict[tuple[str, int], list[tuple[int, str]]] = {}
        n_scanned = 0
        for c in self._targets(args.get("conversations")):
            allowed = parse_range(args["positions"], len(c.tokens)) if args.get("positions") else None
            for e, per in c.readouts.items():
                if layers is not None and e not in layers:
                    continue
                for pos, samples in per.items():
                    if allowed is not None and pos not in allowed:
                        continue
                    for s in samples:
                        n_scanned += 1
                        if rx.search(s):
                            hits.setdefault((c.cid, pos), []).append((e, s))
        n_hits = sum(len(v) for v in hits.values())
        if not hits:
            return f"Scanned {n_scanned} readouts; none match {rx.pattern!r}. Try other wording."
        base = n_hits / max(1, n_scanned)
        by_layer = Counter(e for v in hits.values() for e, _ in v)
        by_conv = Counter(cid for cid, _ in hits)
        by_region = Counter(self.convs[cid].region[p] for cid, p in hits)
        top = sorted(hits, key=lambda key: (-len(hits[key]), key))[:k]
        lines = [
            f"Scanned {n_scanned} readouts; {n_hits} match at {len(hits)} positions in {len(by_conv)} "
            f"conversations. Base rate: {100 * base:.2f}% of all readouts match.",
            "By layer: " + ", ".join(f"L{x} {by_layer[x]}" for x in sorted(by_layer)),
            "By region: " + ", ".join(f"{r} {n}" for r, n in by_region.most_common()),
            "By conversation (positions): " + ", ".join(f"{c} {n}" for c, n in by_conv.most_common(15)),
            f"Top {len(top)} positions (most matching readouts first):",
        ]
        for cid, p in top:
            c = self.convs[cid]
            cell = c.cell(p)
            tot = sum(len(v) for v in cell.values()) or 1
            n_layers = len({e for e, _ in hits[(cid, p)]})
            near = sum(1 for q in range(p - 2, p + 3) if q != p and (cid, q) in hits)
            rate = len(hits[(cid, p)]) / tot
            verdict = ("RECURRING" if (len(hits[(cid, p)]) >= 3 and n_layers >= 2 and rate >= 3 * base)
                       else "SINGLE/WEAK -- not evidence on its own")
            lines.append(f"{c.where(p)}: {len(hits[(cid, p)])}/{tot} readouts here match "
                         f"({100 * rate:.0f}% vs base {100 * base:.2f}%), {n_layers} layers, {near} of 4 "
                         f"neighbours also match -> {verdict} -- {c.context(p, 6)}")
            for e, s in sorted(hits[(cid, p)])[:SAMPLES_PER_HIT]:
                lines.append(f"   L{e}: {s[:CLIP]}")
        return "\n".join(lines)

    def inspect(self, args: dict[str, Any]) -> str:
        self.queries["inspect"] += 1
        c = self._conv(args.get("conversation", ""))
        pos = int(str(args.get("position", -1)).strip().lstrip("tT"))
        if not 0 <= pos < len(c.tokens):
            raise ValueError(f"position must be within t0-t{len(c.tokens) - 1}")
        w = max(0, min(int(args.get("window", WINDOW)), MAX_WINDOW))
        layers = set(args.get("layers") or []) or None
        lines = [f"=== {c.where(pos)} ===", f"context: {c.context(pos)}"]
        for e, samples in c.cell(pos).items():
            if layers is None or e in layers:
                lines.append(f"L{e}: " + " | ".join(f"({i}) {s}" for i, s in enumerate(samples, start=1)))
        nb = [p for p in range(pos - w, pos + w + 1) if p != pos and 0 <= p < len(c.tokens)]
        if nb:
            lines.append("\n--- neighbours (one sample per layer) ---")
            for p in nb:
                lines.append(c.where(p))
                for e, samples in c.cell(p).items():
                    if (layers is None or e in layers) and samples:
                        lines.append(f"   L{e}: {samples[0]}")
        return "\n".join(lines)


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI function schemas for the bank tools (the agent's lens access)."""
    conv = {"type": "string", "description": "conversation id"}
    return [
        {"type": "function", "function": {
            "name": "search",
            "description": ("Every readout matching a case-insensitive Python regex, over all conversations "
                            "(or the ones listed). Returns match counts by layer, region and conversation, the "
                            "pattern's base rate, then the positions with the most matching readouts, each with "
                            "its support count and a RECURRING or SINGLE/WEAK verdict. Local and instant; try "
                            "several patterns (synonyms, the words a readout would use)."),
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string"},
                "conversations": {"type": "string", "description": "optional: comma-separated ids"},
                "positions": {"type": "string", "description": "optional range within each, e.g. 't20-t50'"},
                "layers": {"type": "array", "items": {"type": "integer"}, "description": "optional"},
                "k": {"type": "integer", "description": f"positions returned (default {SEARCH_K})"}},
                "required": ["pattern"]}}},
        {"type": "function", "function": {
            "name": "inspect",
            "description": ("One position in full: every layer, every sample, with the text around it; "
                            "neighbouring positions shown compactly (one sample each)."),
            "parameters": {"type": "object", "properties": {
                "conversation": conv, "position": {"type": "integer", "description": "the t-number"},
                "window": {"type": "integer", "description": f"neighbours each side (default {WINDOW}, max {MAX_WINDOW})"},
                "layers": {"type": "array", "items": {"type": "integer"}}},
                "required": ["conversation", "position"]}}},
        {"type": "function", "function": {
            "name": "show_tokens",
            "description": "The numbered tokens (t<pos>) of one conversation, so you can name exact positions.",
            "parameters": {"type": "object", "properties": {"conversation": conv}, "required": ["conversation"]}}},
    ]
