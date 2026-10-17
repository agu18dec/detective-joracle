"""Pure helpers for the harness readout banks (no torch, no modal): imported client-side by
``readers_modal.py`` and inside its containers.

Bank format (``detective_joracle.chive.readers``): one ``<bank>/<prompt_id>.json.gz`` =
``{"prompt_id", "reader", "token_ids", "n_tokens", "layers", "k", "readouts": {layer: {pos:
[samples]}}, "coverage_note", "provenance"}``. The harness loader only needs ``token_ids`` and
``readouts`` (and ``k``); the rest is provenance.
"""

import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# ---- pinned revisions (Ops rule: pin every HF revision) ----------------------------------
BASE_MODEL = "Qwen/Qwen3.6-27B"
BASE_REV = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"  # same rev the existing banks record

JLENS_REPO = "neuronpedia/jacobian-lens"
JLENS_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
JLENS_REV = "a4114d7752d11eb546e6cf372213d7e75526d3a1"

OLENS_REPO = "agu18dec/local-workspace"  # dataset repo
OLENS_PATH = "ckpts/ao/rl/s3d.ddp600.s0/iter_000600"  # olens_suite.runner.S3D_RL600.path
OLENS_REV = "5653a7fead43ecbd87e3b6d2a2990a98b32843ca"

# single-readout OLens (no distill phase): GRPO from the ptag parent
# u64max32.final.s3/step5500, k_max=1, wandb group rl-sc-s3-single; stopped + published at 500
# (docs/project/experiments/ola/single_vs_multi_bullet.md). Contract = the parent's frozen
# meta.json: continuation_raw prompt, transform "scaled" = 122.3225 * h (global_frozen, no clip).
OLENS_SINGLE_PATH = "ckpts/ao/rl/s3.single.ddp600.s0/iter_000500"
OLENS_SINGLE_REV = "5653a7fead43ecbd87e3b6d2a2990a98b32843ca"
OLENS_SINGLE_SCALE = 122.32250086762771

# OLens-family read contracts for the vLLM worker (reader -> checkpoint + injection)
OLENS_CONTRACTS: dict[str, dict[str, Any]] = {
    "olens": {
        "repo": OLENS_REPO,
        "path": OLENS_PATH,
        "rev": OLENS_REV,
        "prompt_kind": "concepts_raw",
        "transform": "unit",
        "alpha": 16000.0,
        "scale": 0.0,
    },
    "olens_single": {
        "repo": OLENS_REPO,
        "path": OLENS_SINGLE_PATH,
        "rev": OLENS_SINGLE_REV,
        "prompt_kind": "continuation_raw",
        "transform": "scaled",
        "alpha": 0.0,
        "scale": OLENS_SINGLE_SCALE,
    },
}

NLA_REPO = "ceselder/qwen3.6-27b-nla-rl"
NLA_REV = "5a13b7ec21a69fcdd0fb24d5edbe96a92aef4b9f"
NLA_BASE_SUBDIR = "av_base"
NLA_ADAPTER = "av_rl_adapters/iter_000400"

MLNLA_REPO = "camilablank/qwen3.6-27b-multilayer-nla-rl600"
MLNLA_REV = "9e8c46833b04e8930747183328927c919eadec85"
MLNLA_ADAPTER = "av"

LADDER = list(range(20, 61, 4))  # 20, 24, ..., 60

# reader -> defaults matching the existing 300-incident banks
READERS: dict[str, dict[str, Any]] = {
    "olens": {"bank": "s3d_rl600_all", "layers": LADDER, "k": 4, "unit": "layer"},
    "olens_single": {"bank": "s3_single_rl", "layers": LADDER, "k": 4, "unit": "layer"},
    "nla": {"bank": "nla_k4", "layers": [42], "k": 4, "unit": "layer"},
    "mlnla": {"bank": "mlnla_k4_chive", "layers": LADDER, "k": 4, "unit": "layer"},
    "jlens": {"bank": "jlens_all", "layers": list(range(63)), "k": 1, "unit": "incident"},
}


# measured aggregate throughput per H200 container (validation runs, 2026-10-15), used only to
# size --num-shards automatically: rollouts/s for sampled readers, tokens/s for J-lens
RATE = {
    "olens": 37.0,
    "olens_single": 37.0,
    "olens_hf": 8.0,
    "nla": 14.5,
    "mlnla": 19.0,
    "jlens": 95.0,
}
LOAD_SEC = {
    "olens": 450,
    "olens_single": 450,
    "olens_hf": 60,
    "nla": 300,
    "mlnla": 360,
    "jlens": 90,
}


def est_gpu_hours(reader: str, n_tokens: int, n_layers: int, k: int) -> float:
    """Estimated generation GPU-hours (excluding per-container load) for ``n_tokens`` total."""
    if reader == "jlens":
        return n_tokens / RATE["jlens"] / 3600
    return n_tokens * n_layers * k / RATE[reader] / 3600


def read_ids(path: str | Path) -> list[str]:
    """One prompt id per whitespace-separated token (the harness ``read_ids`` rule)."""
    return [x for x in Path(path).read_text().split() if x]


def load_tokens(dataset: str | Path, ids: Sequence[str] | None) -> dict[str, list[int]]:
    """``prompt_id -> token_ids`` from ``<dataset>/data/tokens.jsonl`` (restricted to ``ids``,
    in that order). Fails loudly on any id missing from the dataset."""
    path = Path(dataset) / "data" / "tokens.jsonl"
    want = set(ids) if ids is not None else None
    found: dict[str, list[int]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = str(d["prompt_id"])
            if want is None or pid in want:
                found[pid] = [int(t) for t in d["token_ids"]]
    if ids is None:
        return found
    missing = [p for p in ids if p not in found]
    if missing:
        raise SystemExit(f"{len(missing)} ids not in {path}: {missing[:5]}")
    return {p: found[p] for p in ids}


def stable_int(*parts: object, mod: int = 2**31 - 1) -> int:
    """Deterministic seed from any parts (sha256, not Python's salted hash)."""
    s = ":".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % mod


def dense_seed(seed_base: int, prompt_id: str, layer: int, chunk_idx: int) -> int:
    """``global_workspace.chive_eval.dense.dense_seed`` (copied verbatim: the OLens dense
    caches' seed convention)."""
    digest = int(hashlib.sha256(prompt_id.encode()).hexdigest(), 16) % 2**16
    return digest + layer * 1000 + chunk_idx + seed_base


def balance(units: list[Any], n_shards: int, size: Any) -> list[list[Any]]:
    """Largest-first greedy split into ``n_shards`` lists of near-equal total ``size``."""
    n_shards = max(1, min(n_shards, len(units)))
    shards: list[list[Any]] = [[] for _ in range(n_shards)]
    loads = [0] * n_shards
    for u in sorted(units, key=size, reverse=True):
        i = min(range(n_shards), key=lambda j: (loads[j], j))
        shards[i].append(u)
        loads[i] += size(u)
    return [s for s in shards if s]


def final_path(root: Path, bank: str, pid: str) -> Path:
    return root / bank / f"{pid}.json.gz"


def part_path(root: Path, bank: str, pid: str, layer: int) -> Path:
    return root / bank / "_parts" / pid / f"L{layer}.json.gz"


def write_gz(path: Path, obj: Any) -> None:
    """Atomic gzipped JSON write (tmp -> rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(path)


def read_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def validate_record(
    rec: dict[str, Any], token_ids: Sequence[int], layers: Sequence[int], k: int
) -> list[str]:
    """Problems with a harness bank record; ``[]`` = valid (every layer x every position, k
    non-empty samples each... empty strings are reported, not fatal)."""
    problems: list[str] = []
    if list(rec["token_ids"]) != list(token_ids):
        problems.append("token_ids differ from the dataset")
    n = len(token_ids)
    for layer in layers:
        per_pos = rec["readouts"].get(str(layer))
        if per_pos is None:
            problems.append(f"L{layer} absent")
            continue
        missing = [p for p in range(n) if str(p) not in per_pos]
        short = [p for p in range(n) if str(p) in per_pos and len(per_pos[str(p)]) != k]
        if missing or short:
            problems.append(f"L{layer}: {len(missing)} positions missing, {len(short)} cells != k")
    return problems
