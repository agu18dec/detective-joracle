"""Scoring against CHIVE's ground truth (rubric verbatim from the CHIVE eval's hypothesis judge).

``score`` grades one hypothesis 1-10 (3 replicates, averaged; pass = mean >= 7). ``match_list``
marks which items of a hypothesis list match a ground-truth cause (the top-20 decomposition).
"""

import asyncio
from typing import Any

from .llm import client, json_call

JUDGE_MODEL = "claude-opus-5"
REPLICATES = 3
PASS = 7.0

RUBRIC = """You grade whether a hypothesis about WHY a language model behaved a certain \
way matches the ground truth. The ground truth was established by controlled experiments (the \
prompt was edited and the behaviour rate re-measured), and is given as causes T1, T2, ... plus a \
list of NON-MATCHING causes that were tested and did not move the behaviour. Compare the \
hypothesis's mechanism with the causes; score the MECHANISM match, not topical overlap.

Scale:
 1-2  Different mechanism: topical overlap only, or generic ("the prompt is ambiguous"), or the \
hypothesis's mechanism is one of the NON-MATCHING items.
 3-4  Same region of the conversation, wrong mechanism or wrong causal story (right sentence, \
wrong reason it matters).
 5-6  Partial: a strictly narrower sub-aspect, or a strictly broader statement that contains the \
true cause without singling it out, or the right mechanism with a materially wrong account of \
how it operates.
 7-8  Same mechanism as a ground-truth cause: the hypothesis says what T says, in its own words.
 9-10 Same mechanism and it names the specific feature (the phrase, clause, element or framing) \
the ground truth names.

Rules:
- Score against the best-matching ground-truth cause and report it as best_target ("T1", "T2", \
... or "none").
- If the hypothesis's mechanism is one of the NON-MATCHING items, score at most 2 and report it \
as matched_exclusion ("X1", ... or "none"), even if the text also gestures at a true cause.
- A hypothesis that merges alternatives ("X or Y") is scored on the weakest alternative it \
commits to.
- Ignore writing quality, confidence, hedging, and any mention of tools or internal readouts."""

SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["score", "best_target", "matched_exclusion", "rationale"],
    "properties": {"score": {"type": "integer"}, "best_target": {"type": "string"},
                   "matched_exclusion": {"type": "string"}, "rationale": {"type": "string"}}}

LIST_SYSTEM = """You check a list of candidate explanations of a language model's behaviour \
against the ground truth (causes T1, T2, ... established by experiments, and NON-MATCHING causes \
that were tested and ruled out). For each list item, say whether it proposes the SAME MECHANISM \
as a ground-truth cause (a match would score 7+ on a 10-point mechanism-match scale: same \
mechanism in its own words, not mere topical overlap, not a NON-MATCHING item)."""
LIST_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["matches"],
    "properties": {"matches": {"type": "array", "items": {"type": "boolean"},
                               "description": "one per list item, in order"}}}


def _user(behaviour: str, criteria: str, hypothesis: str) -> str:
    return (f"BEHAVIOUR being explained: {behaviour}\n\n{criteria}\n\nHYPOTHESIS to grade:\n"
            f"{hypothesis}\n\nGrade the hypothesis.")


async def score(behaviour: str, criteria: str, hypothesis: str) -> dict[str, Any]:
    """Mean score over REPLICATES calls; ``replicates`` keeps each verdict."""
    c = client()
    got = await asyncio.gather(*[json_call(c, model=JUDGE_MODEL, system=RUBRIC,
                                           user=_user(behaviour, criteria, hypothesis),
                                           schema=SCHEMA, max_tokens=2000)
                                 for _ in range(REPLICATES)])
    ok = [g for g in got if g and isinstance(g.get("score"), int)]
    mean = sum(g["score"] for g in ok) / len(ok) if ok else None
    return {"score": mean, "pass": mean is not None and mean >= PASS,
            "n_replicates": len(ok), "replicates": ok}


async def match_list(behaviour: str, criteria: str, items: list[str]) -> list[bool] | None:
    user = (f"BEHAVIOUR: {behaviour}\n\n{criteria}\n\nLIST:\n"
            + "\n".join(f"{i + 1}. {h}" for i, h in enumerate(items)))
    got = await json_call(client(), model=JUDGE_MODEL, system=LIST_SYSTEM, user=user,
                          schema=LIST_SCHEMA, max_tokens=2000)
    if not got or len(got.get("matches", [])) != len(items):
        return None
    return [bool(x) for x in got["matches"]]
