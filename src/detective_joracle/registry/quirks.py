"""The quirk registry and the other frozen data files, loaded from the package.

``data/quirk_registry.json`` — ``{"quirks": {name: behavior}}``, the 16 AuditBench quirks (the
``{% block content %}`` bodies of the auditing-agents quirk templates). The closed-set stage
shows all of them to the agent; the judges read the planted one.
``data/distractors.json`` — four plausible-but-wrong alternatives per quirk, generated once and
frozen (``judges.graded.build_distractors``).
``data/held_out_prompts.json`` — AuditBench's 50 held-out prompt/prefill pairs (paper App. K.13),
the prompts the ``*-fixed`` arms' readouts are precomputed on.

Point the loaders at your own files to audit a different set of behaviors.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
REGISTRY_PATH = DATA_DIR / "quirk_registry.json"
DISTRACTORS_PATH = DATA_DIR / "distractors.json"
HELD_OUT_PROMPTS_PATH = DATA_DIR / "held_out_prompts.json"
# organism tags are ``<quirk>_<pipeline>_r<rank>``; these are the pipelines the tags may carry
TRAINING_PIPELINES: frozenset[str] = frozenset({"td", "sdf", "tdkto", "sdfkto"})


def load_quirk_registry(path: Path = REGISTRY_PATH) -> dict[str, str]:
    """``{quirk_name: behavior}`` from the frozen registry."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    data = json.loads(path.read_text())
    return {str(k): str(v) for k, v in data["quirks"].items()}


def load_held_out_prompts(path: Path = HELD_OUT_PROMPTS_PATH) -> list[dict[str, str]]:
    """The 50 held-out prompt/prefill pairs; the fixed-prompt white-box tools read the ``prompt``
    field under the target's system prompt."""
    rows: list[dict[str, str]] = json.loads(path.read_text())
    assert rows and all({"prompt", "prefill"} <= set(r) for r in rows)
    return rows


def quirk_of(tag: str) -> str:
    """``secret_loyalty_sdf_r16`` -> ``secret_loyalty``; ``base`` -> ``""``; a tag without the
    training suffix is returned unchanged (so a registry name is its own quirk)."""
    if tag == "base":
        return ""
    parts = tag.rsplit("_", 2)
    if len(parts) == 3 and parts[2].startswith("r") and parts[1] in TRAINING_PIPELINES:
        return parts[0]
    return tag
