"""CHIVE incidents, loaded from the HF dataset `agu18dec/chive-forensics`.

An incident joins three configs of that dataset on `prompt_id`: the conversation, behaviour and
base rate (`incidents`); the tokenized transcript with (turn, role) per token, the positions every
reader indexes (`tokens`); and the established causes with the judge's reference block
(`ground_truth`). The files are downloaded once into the user cache.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

DATASET = "agu18dec/chive-forensics"
CONFIGS = ("incidents", "tokens", "ground_truth")


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
    ground_truth: dict[str, Any]  # targets, ruled_out, movers, experiments (see the dataset card)

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
        right = "".join(self.pieces[pos + 1 : hi])
        return f"...{left}[[{self.pieces[pos]}]]{right}..."


def fetch(config: str, dataset: str = DATASET) -> Path:
    """Path to `data/<config>.jsonl` of the dataset: a local directory in that layout, or an HF
    dataset id, downloaded on first use."""
    if Path(dataset).is_dir():
        return Path(dataset) / "data" / f"{config}.jsonl"
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    path = cache / "detective_joracle" / dataset / f"{config}.jsonl"
    if not path.exists():
        url = f"https://huggingface.co/datasets/{dataset}/resolve/main/data/{config}.jsonl"
        got = requests.get(url, timeout=120)
        got.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(got.content)
        tmp.rename(path)
    return path


def load_incidents(ids: list[str], dataset: str = DATASET) -> dict[str, Incident]:
    want = set(ids)
    rows: dict[str, dict[str, Any]] = {pid: {} for pid in ids}
    for config in CONFIGS:
        with fetch(config, dataset).open() as f:
            for line in f:
                r = json.loads(line)
                if r["prompt_id"] in want:
                    rows[r["prompt_id"]][config] = r
    missing = [pid for pid, r in rows.items() if len(r) < len(CONFIGS)]
    if missing:
        raise KeyError(f"{len(missing)} ids not in {dataset}, e.g. {missing[:3]}")
    return {pid: _incident(r) for pid, r in rows.items()}


def _incident(r: dict[str, dict[str, Any]]) -> Incident:
    inc, tok, gt = r["incidents"], r["tokens"], r["ground_truth"]
    return Incident(
        prompt_id=inc["prompt_id"],
        messages=inc["messages"],
        completion=inc["completion"],
        behaviour=inc["behaviour"],
        k=int(inc["k"]),
        n=int(inc["n"]),
        token_ids=list(tok["token_ids"]),
        pieces=list(tok["pieces"]),
        turn_of=[(int(t), str(role)) for t, role in tok["turn_of"]],
        criteria=gt["judge_reference"],
        ground_truth=gt,
    )


def read_ids(path: Path) -> list[str]:
    return [x for x in path.read_text().split() if x]
