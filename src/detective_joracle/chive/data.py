"""CHIVE incidents: one JSON per incident (exported from the CHIVE eval with its tokenization).

Each file holds the conversation, the behaviour question and base rate (k of n), the response
the behaviour was seen in, the token ids with decoded pieces and (turn, role) per token, and the
judge's ground-truth block. Readers index the same token positions.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Incident:
    prompt_id: str
    messages: list[dict[str, str]]
    completion: str
    behaviour: str
    k: int
    n: int
    token_ids: list[int]
    pieces: list[str]
    turn_of: list[tuple[int, str]]  # (turn index, role) per token; turn 0 = template tokens
    criteria: str  # the judge's ground-truth block

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def response_start(self) -> int:
        """First token of the model's response (the last assistant-role run of tokens)."""
        last = max((t for t, r in self.turn_of if r == "assistant"), default=0)
        return next(i for i, (t, r) in enumerate(self.turn_of) if t == last and r == "assistant")

    def where(self, pos: int) -> str:
        """``t42 'word' (T3 user)``: a position as the agent sees it."""
        turn, role = self.turn_of[pos]
        label = "response" if pos >= self.response_start else f"T{turn} {role}"
        return f"t{pos} {self.pieces[pos]!r} ({label})"

    def context(self, pos: int, width: int = 12) -> str:
        """The text around a position, with the token itself marked."""
        lo, hi = max(0, pos - width), min(self.n_tokens, pos + width + 1)
        left = "".join(self.pieces[lo:pos])
        right = "".join(self.pieces[pos + 1:hi])
        return f"...{left}[[{self.pieces[pos]}]]{right}..."


def load_incident(path: Path) -> Incident:
    d: dict[str, Any] = json.loads(path.read_text())
    return Incident(
        prompt_id=d["prompt_id"], messages=d["messages"], completion=d["completion"],
        behaviour=d["behaviour"], k=int(d["k"]), n=int(d["n"]), token_ids=list(d["token_ids"]),
        pieces=list(d["pieces"]), turn_of=[(int(t), str(r)) for t, r in d["turn_of"]],
        criteria=d["criteria"],
    )


def load_incidents(root: Path, ids: list[str]) -> dict[str, Incident]:
    return {pid: load_incident(root / f"{pid}.json") for pid in ids}


def read_ids(path: Path) -> list[str]:
    return [x for x in path.read_text().split() if x]
