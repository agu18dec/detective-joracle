"""The WeirdChat side: patterns, rubrics and judged rollouts, pulled from the public API.

WeirdChat (Transluce, https://weirdchat.transluce.org) catalogues behaviors that automated
elicitation found in open-weight models. A *pattern* is a family of user prompts that triggers one
behavior in one model; the site serves, per pattern, the representative prompt, the judge rubrics
and every sampled rollout with the judge's verdict on it. That last part is what makes a pattern
worth explaining: the SAME prompt produced rollouts that show the behavior and rollouts that do
not, so the contrast is not confounded by the prompt.

Two endpoints are enough (no dataset download, no parquet dependency):

* ``GET /api/all-patterns``            every pattern of every model (one JSON blob, ~4.5 MB)
* ``GET /api/behaviors/<entry_id>``    one pattern: prompt, rubrics, and its ``response_samples``

Both are cached on disk, so a re-run costs nothing and the selection is reproducible.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

API = "https://weirdchat.transluce.org/api"
MODEL = (
    "Qwen3.6-27B"  # WeirdChat ran it as Qwen/Qwen3.6-27B-FP8, T=1, no system prompt, no thinking
)
TIMEOUT = 120.0
# WeirdChat's generation settings, which anything reading these transcripts must match.
GENERATION = {"temperature": 1.0, "system_prompt": None, "enable_thinking": False}


@dataclass
class Sample:
    """One judged rollout of a pattern's prompt: the reply and whether the judge saw the behavior."""

    sample_index: int
    matched: bool
    text: str
    phase: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "matched": self.matched,
            "text": self.text,
            "phase": self.phase,
        }


@dataclass
class Pattern:
    """One WeirdChat pattern: the prompt, what the behavior is, and rollouts on both sides."""

    pattern_key: str
    entry_id: str
    behavior_id: str
    behavior_name: str
    group_id: str
    group_summary: str
    prompt: str
    published_match_rate: float
    elo: float
    elo_axes: dict[str, float] = field(default_factory=dict)
    n_group_members: int = 0
    transcript_rubric: str = ""
    weirdchat_url: str = ""
    samples: list[Sample] = field(default_factory=list)

    @property
    def matched(self) -> list[Sample]:
        return [s for s in self.samples if s.matched]

    @property
    def unmatched(self) -> list[Sample]:
        return [s for s in self.samples if not s.matched]

    def to_json(self) -> dict[str, Any]:
        return {
            "pattern_key": self.pattern_key,
            "entry_id": self.entry_id,
            "behavior_id": self.behavior_id,
            "behavior_name": self.behavior_name,
            "group_id": self.group_id,
            "group_summary": self.group_summary,
            "prompt": self.prompt,
            "published_match_rate": self.published_match_rate,
            "elo": self.elo,
            "elo_axes": dict(self.elo_axes),
            "n_group_members": self.n_group_members,
            "transcript_rubric": self.transcript_rubric,
            "weirdchat_url": self.weirdchat_url,
            "samples": [s.to_json() for s in self.samples],
        }

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> "Pattern":
        kw = {k: v for k, v in blob.items() if k != "samples"}
        samples = [Sample(**s) for s in blob.get("samples", [])]
        return cls(**kw, samples=samples)


def _get(url: str, cache: Path | None) -> Any:
    """GET JSON, through a file cache when one is given."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    out = r.json()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, ensure_ascii=False))
    return out


def fetch_patterns(cache_dir: Path, model: str = MODEL) -> list[dict[str, Any]]:
    """Every WeirdChat pattern for ``model`` (the site's own summary rows)."""
    blob = _get(f"{API}/all-patterns", Path(cache_dir) / "all_patterns.json")
    items: list[dict[str, Any]] = blob["items"] if isinstance(blob, dict) else blob
    return [x for x in items if x.get("subject_model_name") == model]


def pattern_key(row: dict[str, Any]) -> str:
    """``<behavior_id>__<group_id>`` — stable, readable, and unique per pattern."""
    return f"{row['behavior_id']}__{row.get('group_id') or row['entry_id'][:12]}"


def select_patterns(
    rows: list[dict[str, Any]],
    *,
    per_behavior: int = 3,
    min_rate: float = 0.15,
    behaviors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """The candidate set: the highest-Elo patterns per behavior that fire often enough to study.

    ``min_rate`` is the published match rate — below it a pattern's "behavior" is a handful of
    rollouts and the matched/unmatched contrast is mostly sampling noise. The Elo is WeirdChat's
    own interestingness ranking (naturalness + unexpectedness + harmfulness).
    """
    keep = [r for r in rows if float(r.get("metrics", {}).get("match_rate") or 0.0) >= min_rate]
    if behaviors:
        keep = [r for r in keep if r["behavior_id"] in behaviors]
    by_behavior: dict[str, list[dict[str, Any]]] = {}
    for r in keep:
        by_behavior.setdefault(r["behavior_id"], []).append(r)
    out: list[dict[str, Any]] = []
    for bid in sorted(by_behavior):
        ranked = sorted(
            by_behavior[bid],
            key=lambda r: -float((r.get("interestingness") or {}).get("elo") or 0.0),
        )
        out.extend(ranked[:per_behavior])
    return out


def _text_of(message: dict[str, Any]) -> str:
    """WeirdChat messages carry a list of content parts; join the text ones."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts = [str(p.get("text") or "") for p in content or [] if isinstance(p, dict)]
    return "".join(parts)


def _clean(text: str) -> str:
    """Normalise a rollout: NFC, no carriage returns, no trailing whitespace."""
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()


def fetch_pattern(
    row: dict[str, Any], cache_dir: Path, *, n_side: int = 3, min_chars: int = 200
) -> Pattern:
    """One pattern with its prompt, rubric and up to ``n_side`` rollouts per side.

    Rollouts are taken from the middle of the length distribution rather than the top, so neither
    side is represented by its longest reply (length alone is a confound the agent would read as a
    mechanism). Very short replies are dropped: a truncated rollout says nothing about substance.
    """
    key = pattern_key(row)
    detail = _get(f"{API}/behaviors/{row['entry_id']}", Path(cache_dir) / f"detail_{key}.json")
    prompt = _clean(row.get("representative_user_text") or "")
    src = detail.get("input_transcript") or detail.get("source_transcript") or {}
    if not prompt:
        users = [m for m in src.get("messages", []) if m.get("role") == "user"]
        prompt = _clean(_text_of(users[0])) if users else ""
    samples: list[Sample] = []
    for side in (True, False):
        pool = [
            s
            for s in detail.get("response_samples") or []
            if bool(s.get("matched")) is side
            and len(_clean(str(s.get("response_text") or ""))) >= min_chars
        ]
        pool.sort(key=lambda s: len(str(s.get("response_text") or "")))
        mid = len(pool) // 2
        order = sorted(range(len(pool)), key=lambda i: abs(i - mid))[:n_side]
        for i in sorted(order):
            s = pool[i]
            samples.append(
                Sample(
                    sample_index=int(s.get("sample_index") or i),
                    matched=side,
                    text=_clean(str(s.get("response_text") or "")),
                    phase=str(s.get("phase") or ""),
                )
            )
    axes = {
        str(a.get("axis")): float(a.get("elo") or 0.0)
        for a in ((row.get("interestingness") or {}).get("scores") or [])
    }
    return Pattern(
        pattern_key=key,
        entry_id=str(row["entry_id"]),
        behavior_id=str(row["behavior_id"]),
        behavior_name=str(row["behavior_name"]),
        group_id=str(row.get("group_id") or ""),
        group_summary=_clean(str(row.get("group_summary") or "")),
        prompt=prompt,
        published_match_rate=float((row.get("metrics") or {}).get("match_rate") or 0.0),
        elo=float((row.get("interestingness") or {}).get("elo") or 0.0),
        elo_axes=axes,
        n_group_members=int(row.get("group_member_count") or 0),
        transcript_rubric=str((detail.get("transcript_rubric") or {}).get("text") or ""),
        weirdchat_url=f"https://weirdchat.transluce.org/?behavior={row['behavior_id']}&pattern={row['entry_id']}",
        samples=samples,
    )


def load_pattern(path: Path) -> Pattern:
    """A pattern written by the ``data`` stage."""
    return Pattern.from_json(json.loads(Path(path).read_text()))


_WS = re.compile(r"\s+")


def token_stats(patterns: list[Pattern]) -> dict[str, Any]:
    """Rough size of what the agent will read: words in the prompts and in the rollouts."""

    def words(s: str) -> int:
        return len([w for w in _WS.split(s) if w])

    prompt_words = [words(p.prompt) for p in patterns]
    reply_words = [words(s.text) for p in patterns for s in p.samples]
    return {
        "n_patterns": len(patterns),
        "n_samples": sum(len(p.samples) for p in patterns),
        "prompt_words": {
            "total": sum(prompt_words),
            "mean": round(sum(prompt_words) / max(1, len(prompt_words)), 1),
            "min": min(prompt_words, default=0),
            "max": max(prompt_words, default=0),
        },
        "reply_words": {
            "total": sum(reply_words),
            "mean": round(sum(reply_words) / max(1, len(reply_words)), 1),
            "min": min(reply_words, default=0),
            "max": max(reply_words, default=0),
        },
    }


__all__ = [
    "API",
    "GENERATION",
    "MODEL",
    "Pattern",
    "Sample",
    "fetch_pattern",
    "fetch_patterns",
    "load_pattern",
    "pattern_key",
    "select_patterns",
    "token_stats",
]
