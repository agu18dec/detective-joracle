"""Explaining WeirdChat's catalogued behaviors in the stock model, instead of finding a planted quirk.

``data`` pulls the patterns, rubrics and judged rollouts from WeirdChat's public API; ``prompts``
holds what the investigator is told; ``explain`` runs one investigation over one pattern;
``diagnostics`` takes the raw lens reads that exist before any agent interprets them; ``synth``
groups the mechanisms across runs. Nothing here is scored: there is no ground truth for WHY a
model does what it does, so a run produces hypotheses and the experiments that would test them.
"""

from .data import MODEL, Pattern, Sample, fetch_pattern, fetch_patterns, select_patterns
from .explain import ExplainTools, run_explain_agent

__all__ = [
    "MODEL",
    "ExplainTools",
    "Pattern",
    "Sample",
    "fetch_pattern",
    "fetch_patterns",
    "run_explain_agent",
    "select_patterns",
]
