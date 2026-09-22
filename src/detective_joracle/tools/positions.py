"""Position tagging: every token of a rendered chat gets a ``region`` and a ``kind``.

A lens server tags the positions it reads and the agent's page groups cells by region. Regions
are ``system`` / ``user`` / ``header`` / ``reply``; kinds are the four of the fixed-quota set P
(``boundary``, ``punct``, ``reply4``, ``user4``) or, on the dense grid, the region name itself.

  ``boundary``  the chat boundaries after the system prompt — the user turn's ``<|im_end|>``,
                the ``assistant`` role token, ``</think>``, and the last header token before
                the reply ("about to speak")
  ``punct``     sentence/clause punctuation and newlines (the decoded token has no alphanumerics
                and contains one of ``. , : ; ? ! \\n``; fused forms like ``.Ċ`` / ``):`` count)
  ``reply4``    reply tokens at stride 4, thinned evenly
  ``user4``     the user turn at stride 4, thinned evenly

Control tokens are identified by ID (``<|im_start|>``, ``<|im_end|>``, ``<think>``,
``</think>``) and roles by the token after ``<|im_start|>`` — never by the words
``user``/``assistant`` in text. Written for the Qwen/ChatML render; another chat template needs
its own :class:`ControlIds` and, if its header differs, its own :func:`regions_of`.
"""

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..util.text import decode_byte_level

KINDS: tuple[str, ...] = ("boundary", "punct", "reply4", "user4", "user", "header", "reply")
REGIONS: tuple[str, ...] = ("system", "user", "header", "reply")
DEFAULT_QUOTAS: dict[str, int] = {"punct": 8, "reply4": 8, "user4": 8}
_PUNCT = re.compile(r"[.,:;?!\n]")
_ALNUM = re.compile(r"[^\W_]", re.UNICODE)


@dataclass(frozen=True)
class ControlIds:
    """The four chat-control token ids, read from the tokenizer."""

    im_start: int
    im_end: int
    think: int
    think_end: int

    @classmethod
    def from_tokenizer(cls, tok: Any) -> "ControlIds":
        """Read the four ids from a HF tokenizer; raises ``ValueError`` if any is missing."""
        ids = tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>", "<think>", "</think>"])
        if any(i is None or i == tok.unk_token_id for i in ids):
            raise ValueError(f"tokenizer lacks a chat-control token: {ids}")
        return cls(*(int(i) for i in ids))


@dataclass(frozen=True)
class PositionTag:
    """Where a token sits (``region``) and which sampling kind put it in the set (``kind``)."""

    region: str
    kind: str

    def to_json(self) -> dict[str, str]:
        """``{"region", "kind"}`` as sent on the wire."""
        return {"region": self.region, "kind": self.kind}


def regions_of(ids: Sequence[int], tokens: Sequence[str], ctl: ControlIds) -> list[str]:
    """Region per position: ``system`` / ``user`` / ``header`` / ``reply``.

    Turns start at ``<|im_start|>`` (by id); the role is the token right after it. The
    assistant turn's header runs through ``</think>`` and the newline token after it; the
    reply starts there. Without a think block the header is ``<|im_start|> assistant Ċ``.
    """
    n = len(ids)
    out = ["system"] * n
    starts = [i for i, t in enumerate(ids) if t == ctl.im_start]
    bounds = [*starts, n]
    for a, b in itertools.pairwise(bounds):
        role = tokens[a + 1] if a + 1 < n else ""
        if role == "assistant":
            end = min(a + 3, b)
            for j in range(a, b):
                if ids[j] == ctl.think_end:
                    end = j + 1
                    if end < b and decode_byte_level(tokens[end]).strip(" ") in ("\n", "\n\n"):
                        end += 1
                    break
            out[a:end] = ["header"] * (end - a)
            out[end:b] = ["reply"] * (b - end)
        elif role == "system":
            out[a:b] = ["system"] * (b - a)
        else:
            out[a:b] = ["user"] * (b - a)
    return out


def is_punct(token: str) -> bool:
    """A delimiter token: no alphanumerics, and one of ``. , : ; ? !`` or a newline (fused forms count)."""
    surface = decode_byte_level(token)
    return bool(_PUNCT.search(surface)) and not _ALNUM.search(surface)


def thin(xs: Sequence[int], n: int) -> list[int]:
    """Evenly keep ``n`` of ``xs`` (sorted input), always including the last one."""
    if n <= 0 or not xs:
        return []
    if len(xs) <= n:
        return list(xs)
    if n == 1:
        return [xs[-1]]
    step = (len(xs) - 1) / (n - 1)
    return sorted({xs[round(i * step)] for i in range(n)})


def position_set_all(
    ids: Sequence[int],
    tokens: Sequence[str],
    ctl: ControlIds,
    reply_positions: Sequence[int] | None = None,
) -> dict[int, PositionTag]:
    """The dense grid: EVERY non-system position. Kinds: ``boundary`` (the four chat
    boundaries), ``punct`` (delimiter tokens), else ``user``/``header``/``reply`` by region."""
    p = position_set(
        ids,
        tokens,
        ctl,
        reply_positions,
        quotas={"punct": 10**6, "reply4": 0, "user4": 0},
        stride=1,
    )
    regions = regions_of(ids, tokens, ctl)
    out: dict[int, PositionTag] = {}
    for i, r in enumerate(regions):
        if r == "system":
            continue
        t = p.get(i)
        out[i] = t if t is not None else PositionTag(r, r)
    return out


def position_set(
    ids: Sequence[int],
    tokens: Sequence[str],
    ctl: ControlIds,
    reply_positions: Sequence[int] | None = None,
    *,
    quotas: dict[str, int] | None = None,
    stride: int = 4,
) -> dict[int, PositionTag]:
    """The fixed-quota set P for one prompt as ``{pos: tag}`` (sorted by position).

    ``reply_positions`` overrides the reply region when the caller knows it; when None the reply
    region is derived from the ids. Earlier kinds win a contested position
    (``boundary`` > ``punct`` > ``reply4`` > ``user4``).
    """
    q = {**DEFAULT_QUOTAS, **(quotas or {})}
    regions = regions_of(ids, tokens, ctl)
    n = len(ids)
    user = [i for i in range(n) if regions[i] == "user"]
    # user CONTENT: after `<|im_start|> user Ċ`, before the closing `<|im_end|>`
    user_content = [
        i
        for i in user[3:]
        if ids[i] != ctl.im_end and not (i - 1 in user and ids[i - 1] == ctl.im_end)
    ]
    reply = (
        [p for p in reply_positions if p < n]
        if reply_positions
        else [i for i in range(n) if regions[i] == "reply"]
    )
    reply_start = reply[0] if reply else n

    boundary: list[int] = []
    user_end = [i for i in user if ids[i] == ctl.im_end]
    if user_end:
        boundary.append(user_end[-1])
    for i in range(n):
        if ids[i] == ctl.im_start and i + 1 < n and tokens[i + 1] == "assistant":
            boundary.append(i + 1)
        if ids[i] == ctl.think_end and regions[i] == "header":
            boundary.append(i)
    if reply_start > 0 and regions[reply_start - 1] == "header":
        boundary.append(reply_start - 1)

    punct_user = [i for i in user if is_punct(tokens[i]) and ids[i] != ctl.im_end]
    punct_reply = [i for i in reply if is_punct(tokens[i])]
    half = q["punct"] // 2
    n_user = min(len(punct_user), half + max(0, half - len(punct_reply)))
    n_reply = min(len(punct_reply), q["punct"] - n_user)
    punct = thin(punct_user, n_user) + thin(punct_reply, n_reply)

    chosen: dict[int, PositionTag] = {}
    for p in sorted(set(boundary)) + sorted(punct):
        chosen.setdefault(p, PositionTag(regions[p], "boundary" if p in boundary else "punct"))
    # the stride kinds fill their quota from positions NOT already taken, so a punctuation-
    # heavy prompt still gets its 8 reply4 / 8 user4 cells
    for kind, region_positions in (("reply4", reply), ("user4", user_content)):
        cands = [p for p in region_positions[::stride] if p not in chosen]
        for p in thin(cands, q[kind]):
            chosen[p] = PositionTag(regions[p], kind)
    return dict(sorted(chosen.items()))
