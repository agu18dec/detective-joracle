"""Readout banks: one gzipped JSON per incident, ``readouts[layer][pos] -> [sample, ...]``.

Every reader (OLens checkpoints, NLA, J-lens) uses the same file format, so one class serves
them all; ``KIND`` decides how the agent is told to read it. ``shuffled`` serves another
incident's readouts (positions mapped proportionally): the control for having the tools
without the right content.
"""

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import Incident

KINDS = ("olens", "nla", "jlens", "boundary")


@dataclass
class Reader:
    name: str  # e.g. olens_rl, olens_distill, nla, jlens
    kind: str  # one of KINDS: how the readouts are described and inspected
    layers: list[int]
    k: int  # samples per cell
    cells: dict[int, dict[int, list[str]]]  # pos -> layer -> samples
    source_id: str  # the incident the readouts come from (differs only when shuffled)
    n_tokens: int = 0  # the incident's length, to tell the agent when readouts are sparse
    coverage_note: str = ""  # the bank's account of which positions were read, when sparse

    def cell(self, pos: int) -> dict[int, list[str]]:
        return self.cells.get(pos, {})


def _open(path: Path) -> dict[str, Any]:
    for f in (path.with_suffix(".json.gz"), path.with_suffix(".json")):
        if f.exists():
            with gzip.open(f, "rt") if f.suffix == ".gz" else f.open() as fh:
                data: dict[str, Any] = json.load(fh)
                return data
    raise FileNotFoundError(f"no bank record {path}.json[.gz]")


def load_reader(
    name: str,
    kind: str,
    bank: Path,
    inc: Incident,
    partner: Incident | None = None,
    permute: bool = False,
) -> Reader:
    """The reader for ``inc``. Controls: with ``partner``, ``inc``'s positions read ``partner``'s
    cells (no information about this incident); with ``permute``, each position reads another
    position of the SAME incident, by a fixed permutation seeded by the prompt id (on-topic,
    plausible content, but no information about where or at which token it is represented)."""
    if kind not in KINDS:
        raise ValueError(f"unknown reader kind {kind!r}; one of {KINDS}")
    src = partner or inc
    rec = _open(bank / src.prompt_id)
    if list(rec["token_ids"]) != src.token_ids:
        raise ValueError(f"{name}: bank token ids differ from {src.prompt_id}'s tokenization")
    by_pos: dict[int, dict[int, list[str]]] = {}
    for layer, per_pos in rec["readouts"].items():
        for pos, samples in per_pos.items():
            by_pos.setdefault(int(pos), {})[int(layer)] = [str(s) for s in samples]
    if partner is not None:  # position p of inc reads the proportional position of partner
        n_src, n_dst = partner.n_tokens, inc.n_tokens
        by_pos = {
            p: by_pos.get(min(n_src - 1, p * n_src // max(1, n_dst)), {}) for p in range(n_dst)
        }
    if permute:
        order = list(range(inc.n_tokens))
        random.Random(f"posshuffle:{inc.prompt_id}").shuffle(order)
        by_pos = {p: by_pos.get(q, {}) for p, q in enumerate(order)}
    layers = sorted(int(x) for x in rec["readouts"])
    return Reader(
        name=name,
        kind=kind,
        layers=layers,
        k=int(rec.get("k", 1)),
        cells=by_pos,
        source_id=src.prompt_id,
        n_tokens=inc.n_tokens,
        coverage_note=str(rec.get("coverage_note", "")),
    )


DESCRIBE = {
    "boundary": """Boundary readouts are OLens read at a FORCED turn end: the conversation is \
cut right after a token, the turn is closed there (<|im_end|><|im_start|>), and OLens reads what \
the model carries forward at that boundary -- what it would summarize or plan had the turn ended \
at that token. They exist at most positions before the response, at layers {layers}, {k} samples \
each. Expect them to be more summary-like than ordinary readouts; some mention the text being \
cut off, which is an artifact of the forced end.""",
    "olens": """OLens is a copy of the target model trained so that its short '- ' bullets \
reconstruct one layer's activation at one token: whatever that activation encodes and can be \
put into words, including what the model represents but never writes -- an assumption, a \
framing or default it applies, a plan for its answer, a judgement about the user. Readouts \
exist at every token, at layers {layers}, with {k} independent samples per (token, layer). Each \
sample is one draw from what the activation encodes: details (names, numbers) are often \
borrowed from similar contexts, so what recurs across samples and neighbouring layers or \
positions is signal; a single sample can mislead. Early layers (20-32) mostly read the input, \
middle layers (36-44) compose it, layers 48-60 plan the answer. Readouts at chat-template \
tokens often hold generic template text.""",
    "nla": """NLA is a trained verbalizer: a copy of the target model reads the layer-42 \
activation at one token and writes a free-text explanation of what that activation encodes, \
including what the model represents but never writes -- an assumption, a framing or default, a \
plan for its answer, a judgement about the user. Explanations exist at every token, at layer 42 \
only, with {k} independent samples per token. Details are often borrowed from similar \
contexts: trust what recurs across samples and across neighbouring tokens.""",
    "jlens": """J-lens gives, at every token and at every layer {layers}, the 10 vocabulary tokens \
nearest the target model's activation -- a single-token view of what it represents there, \
including what it never writes. Read them as hints to extrapolate from ('ruins', 'glass', 'dust' \
together suggest a wasteland frame), not as sentences; trust what recurs across neighbouring \
positions and layers. Early layers mostly read the input, middle layers compose it, late layers \
plan the answer.""",
}


def describe(reader: Reader) -> str:
    layers = (
        f"{reader.layers[0]}..{reader.layers[-1]}"
        if len(reader.layers) > 3
        else ", ".join(map(str, reader.layers))
    )
    text = DESCRIBE[reader.kind].format(layers=layers, k=reader.k)
    covered = sum(1 for cell in reader.cells.values() if cell)
    if reader.n_tokens and covered < reader.n_tokens:
        note = f" ({reader.coverage_note})" if reader.coverage_note else ""
        text += (
            f" In this conversation readouts exist at only {covered} of {reader.n_tokens} "
            f"tokens{note}; the other tokens have none."
        )
    return text


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")


def digest(
    reader: Reader, inc: Incident, n: int = 12, span: int = 20
) -> list[tuple[int, int, str]]:
    """Positions before the response whose readouts most consistently say something the nearby
    text does not: novel words (absent from the +-``span`` token window) that recur in several
    samples or layers at that position. No ground truth. Returns (position, layer, sample) with
    one representative sample per position, positions at least 3 apart, in position order."""
    end = inc.response_start
    scored: list[tuple[float, int, int, str]] = []
    for pos in range(end):
        cells = reader.cell(pos)
        if not cells:
            continue
        near = set(
            w.lower() for w in _WORD.findall("".join(inc.pieces[max(0, pos - span) : pos + span]))
        )
        count: dict[str, int] = {}
        best: tuple[int, int, str] = (0, 0, "")
        for layer, samples in cells.items():
            for smp in samples:
                novel = {w.lower() for w in _WORD.findall(smp)} - near
                for w in novel:
                    count[w] = count.get(w, 0) + 1
                if len(novel) > best[0]:
                    best = (len(novel), layer, smp)
        recurring = sum(c for c in count.values() if c >= 2)
        if recurring:
            scored.append((float(recurring), pos, best[1], best[2]))
    picked: list[tuple[int, int, str]] = []
    for _score, pos, layer, smp in sorted(scored, reverse=True):
        if all(abs(pos - q) >= 3 for q, _, _ in picked):
            picked.append((pos, layer, smp))
        if len(picked) == n:
            break
    return sorted(picked)
