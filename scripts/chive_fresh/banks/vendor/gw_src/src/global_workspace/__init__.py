"""global-workspace: research extensions around the Jacobian lens.

The reusable core (model wrapping, fitting, the ``JacobianLens`` object) is the official
``jlens`` package (github.com/anthropics/jacobian-lens); this package holds what the reference
implementation does not: batched readout math, interventions, experiment libraries, judges,
and the visualizers.

Public symbols are imported **lazily** (PEP 562 module ``__getattr__``) so that importing the
package — or a single pure-Python submodule such as ``viz.token_dashboard`` — does not eagerly
pull torch-heavy modules (``model``, ``lens``, ``viz.visualizer``, …). This lets CPU-only
consumers (e.g. the hosted token dashboard, whose image has no torch) import the renderer
without dragging in the whole stack.
"""

import importlib
from typing import Any

# public name -> submodule it lives in (resolved on first attribute access)
_SYMBOL_MODULE: dict[str, str] = {
    "ModelBackend": "model",
    "ablate": "interventions.core",
    "ablate_subspace": "interventions.core",
    "concept_probe": "interventions.core",
    "coordinate_swap": "interventions.core",
    "residual_hooks": "interventions.core",
    "steer": "interventions.core",
    "artifact_name": "lens",
    "load_jacobians": "lens",
    "cosine_readout": "lens",
    "grid_top_k": "lens",
    "jacobian_lens": "lens",
    "jlens_vector": "lens",
    "logit_lens": "lens",
    "top_k": "lens",
    "candidate_grid_stats": "experiments.verbal_report",
    "candidate_tensors": "experiments.verbal_report",
    "filter_categories": "experiments.verbal_report",
    "grid_spearman_top1": "experiments.verbal_report",
    "produced_token_rank": "experiments.verbal_report",
    "read_site_argmax": "experiments.verbal_report",
    "single_token_items": "experiments.verbal_report",
    "spearman_null_halfwidth": "experiments.verbal_report",
    "CellTopK": "viz.visualizer",
    "LensGrid": "viz.visualizer",
    "build_grid": "viz.visualizer",
    "compute_grid": "viz.visualizer",
    "make_app": "viz.visualizer",
    "render_grid": "viz.visualizer",
}


def __getattr__(name: str) -> Any:
    mod = _SYMBOL_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)


def __dir__() -> list[str]:
    return sorted(_SYMBOL_MODULE)


__all__ = sorted(_SYMBOL_MODULE)
