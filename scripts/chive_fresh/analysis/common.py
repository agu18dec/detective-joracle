"""Shared helpers for the fresh-batch analysis scripts: judged pass@1 draws and the paired bootstrap.

A harness record counts when the old pass judge scored it (``judge.score``); records that stopped on
an API error or were salvaged after one (``stopped_by`` starting ``error`` / ``salvaged``) are
excluded (run_chive.py redoes them on a re-run). pass = score >= 7.
"""

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

BASE_DRAWS = "*.d[0-4].json"  # the 5 base draws (extra draws were run for other analyses)
PASS = 7


def record_pass(path: Path) -> bool | None:
    """pass (score >= 7) for one record; None when unjudged, errored or salvaged."""
    r = json.loads(path.read_text())
    if str(r.get("stopped_by", "")).startswith(("error", "salvaged")):
        return None
    s = (r.get("judge") or {}).get("score")
    return None if s is None else s >= PASS


def pid_of(path: Path) -> str:
    """``prompt_000123.d4.json`` -> ``prompt_000123``."""
    return path.name.split(".d")[0]


def judged(
    runs: Path, batches: Iterable[str], arms: Iterable[str], pattern: str = BASE_DRAWS
) -> dict[tuple[str, str], dict[str, list[bool]]]:
    """(batch, pid) -> arm -> pass booleans over the matching, judged draws."""
    out: dict[tuple[str, str], dict[str, list[bool]]] = defaultdict(dict)
    for b in batches:
        for a in arms:
            for f in sorted((runs / b / a).glob(pattern)):
                p = record_pass(f)
                if p is not None:
                    out[(b, pid_of(f))].setdefault(a, []).append(p)
    return out


def complete(
    d: dict[tuple[str, str], dict[str, list[bool]]], arms: Sequence[str], draws: int = 5
) -> list[tuple[str, str]]:
    """Incidents where every arm has exactly ``draws`` judged draws (paired comparisons)."""
    return [k for k in d if all(len(d[k].get(a, [])) == draws for a in arms)]


def paired_bootstrap(
    diffs: Sequence[float], rng: random.Random, n_boot: int = 2000
) -> tuple[float, float, float]:
    """(mean, 2.5th, 97.5th percentile) of the incident-level bootstrap of the mean difference.
    ``rng`` is shared across calls by the scripts (seed 0), so call order matters for identical
    numbers."""
    n = len(diffs)
    bs = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    return sum(diffs) / n, bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]
