"""Raw lens readouts on a pattern, taken before any agent sees it.

One pass per pattern reads the study's own rollouts — a matched one and an unmatched one, same
prompt — at every position and layer. Two things make this worth doing outside the agent loop.

* The USER and HEADER positions are shared by every rollout of the prompt: they are the model
  having read the request and not yet answered, so whatever is there is the propensity, not the
  outcome. They are identical across the pair by construction.
* The replies diverge somewhere, and only from that point can the two reads differ for a reason.
  ``fork_of`` finds that point on the text, so the viewer (and the agent) can say which part of a
  contrast is real and which is the identical prefix decoded twice.
"""

import difflib
import re
from collections.abc import Sequence
from typing import Any

from ..tools.live import READ_LAYERS, LensClient
from .data import Pattern

LENS = "olens"
READ_POSITIONS = "all"
DEFAULT_K = 1

_WORD = re.compile(r"\S+\s*")


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def fork_of(matched: str, unmatched: str) -> dict[str, Any]:
    """Where two replies to the same prompt stop agreeing: the shared prefix and what follows."""
    a, b = _words(matched), _words(unmatched)
    match = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    )
    n = match.size if (match.a == 0 and match.b == 0) else 0
    prefix = "".join(a[:n])
    return {
        "prefix_words": n,
        "prefix_chars": len(prefix),
        "prefix": prefix[-300:],
        "matched_after": "".join(a[n : n + 40]),
        "unmatched_after": "".join(b[n : n + 40]),
        "note": (
            "the replies share their first "
            f"{n} words; a readout difference before that point is lens sampling noise, not a "
            "difference in the model"
        ),
    }


def read_rollout(
    client: LensClient,
    prompt: str,
    completion: str,
    *,
    layers: Sequence[int] = READ_LAYERS,
    k: int = DEFAULT_K,
    seed: int = 0,
    organism: str = "base",
) -> dict[str, Any]:
    """One lens readout over a (prompt, reply) pair at every position and layer."""
    return client.readout(
        organism=organism,
        messages=[{"role": "user", "content": prompt}],
        completion=completion,
        positions=READ_POSITIONS,
        layers=list(layers),
        k=k,
        lens=LENS,
        seed=seed,
    )


def diagnose(
    pattern: Pattern,
    client: LensClient,
    *,
    layers: Sequence[int] = READ_LAYERS,
    k: int = DEFAULT_K,
    n_side: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """The per-pattern diagnostic bundle: reads on both sides plus the fork between them."""
    reads: list[dict[str, Any]] = []
    for label, samples in (("matched", pattern.matched), ("unmatched", pattern.unmatched)):
        for s in samples[:n_side]:
            res = read_rollout(
                client,
                pattern.prompt,
                s.text,
                layers=layers,
                k=k,
                seed=seed + s.sample_index,
            )
            reads.append(
                {
                    "label": label,
                    "sample_index": s.sample_index,
                    "conversation": {
                        "messages": [{"role": "user", "content": pattern.prompt}],
                        "completion": s.text,
                    },
                    "readout": res,
                }
            )
    fork = None
    if pattern.matched and pattern.unmatched:
        fork = fork_of(pattern.matched[0].text, pattern.unmatched[0].text)
    return {"pattern_key": pattern.pattern_key, "lens": LENS, "reads": reads, "fork": fork}


__all__ = ["DEFAULT_K", "LENS", "diagnose", "fork_of", "read_rollout"]
