"""Shared locations for the fresh-incident CHIVE pipeline (docs/chive_fresh_scaleup.md).

Every path is an env var with the default used on Agam's pod, so the scripts behave exactly as
they did there and can be pointed elsewhere without edits:

  CHIVE_SCALE       work root (default /workspace/agam/chive_scale); the rest default under it:
  CHIVE_DATASETS    harness datasets, one dir per batch: <batch>/data/*.jsonl, splits/, work/
  CHIVE_BANKS       readout banks, one dir per bank: <bank>/<prompt_id>.json.gz
  CHIVE_RUNS        harness records: <batch>/<arm>/<prompt_id>.d<draw>.json
  CHIVE_POINTS      point eval v1 outputs: <batch>/{points,tags,plain,grades,rivals}/
  CHIVE_CONFIDENCE  no-tools confidence strata: <batch>/confidence.json
  CHIVE_CLAIMS_RUNS claims (AUROC) runs: <batch>/{claims.jsonl,focused/}
  CHIVE_RUN_DIRS    raw CHIVE run dirs: <batch>/{config.json,investigations.jsonl,...}
  CHIVE_DIR         the CHIVE checkout (default /workspace/agam/chive), its venv python CHIVE_PY
  GW_DIR            global-workspace checkout (judge_auth.py lives in scripts/ola/ar_ablate)
  CHIVE_TARGET_URL  vLLM endpoint serving the target (Qwen3.6-27B) for experiments
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _p(var: str, default: Path) -> Path:
    return Path(os.environ.get(var) or default)


SCALE = _p("CHIVE_SCALE", Path("/workspace/agam/chive_scale"))
DATASETS = _p("CHIVE_DATASETS", SCALE / "datasets")
BANKS = _p("CHIVE_BANKS", SCALE / "banks")
RUNS = _p("CHIVE_RUNS", SCALE / "harness_runs")
POINTS = _p("CHIVE_POINTS", SCALE / "points")
CONFIDENCE = _p("CHIVE_CONFIDENCE", SCALE / "confidence")
CLAIMS_RUNS = _p("CHIVE_CLAIMS_RUNS", SCALE / "claims_runs")
RUN_DIRS = _p("CHIVE_RUN_DIRS", SCALE / "runs")
CHIVE_DIR = _p("CHIVE_DIR", Path("/workspace/agam/chive"))
CHIVE_PY = os.environ.get("CHIVE_PY") or str(CHIVE_DIR / ".venv" / "bin" / "python")
GW_DIR = _p("GW_DIR", Path("/workspace/agam/global-workspace"))
TARGET_URL = os.environ.get(
    "CHIVE_TARGET_URL", "https://safety-sahan-2-agam--chive-qwen-vllm-serve.modal.run"
)
BATCHES = ["fresh1", "fresh2", "fresh3", "fresh4"]
