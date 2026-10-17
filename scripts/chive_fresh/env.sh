# shellcheck shell=bash
# Source me: the fresh-incident CHIVE pipeline's environment (docs/chive_fresh_scaleup.md).
#   source scripts/chive_fresh/env.sh
# Credentials come from RAM only (/dev/shm/compute.env, pushed with `rp secrets`); values are never
# printed. ANTHROPIC_API_KEY is unset on purpose: Claude is keyless (OAuth token from the Modal
# Dict `anthropic-oauth`, refreshed with `push-judge-token` on the Mac). Every path below is
# overridable; the defaults are the pod layout the runs used (see paths.py).
export PATH=$HOME/.local/bin:$PATH
[ -f /workspace/.persist/env.sh ] && { set -a; . /workspace/.persist/env.sh; set +a; }
[ -f "${COMPUTE_ENV:-/dev/shm/compute.env}" ] && { set -a; . "${COMPUTE_ENV:-/dev/shm/compute.env}"; set +a; }
export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-agam}
unset ANTHROPIC_API_KEY
CHIVE_FRESH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CHIVE_FRESH_DIR
export HARNESS_DIR=${HARNESS_DIR:-$(cd "$CHIVE_FRESH_DIR/../.." && pwd)}
if [ -z "${HARNESS_PY:-}" ]; then
  if [ -x "$HARNESS_DIR/.venv/bin/python" ]; then HARNESS_PY=$HARNESS_DIR/.venv/bin/python; else HARNESS_PY=python3; fi
fi
export HARNESS_PY
export CHIVE_SCALE=${CHIVE_SCALE:-/workspace/agam/chive_scale}
export CHIVE_DATASETS=${CHIVE_DATASETS:-$CHIVE_SCALE/datasets}
export CHIVE_BANKS=${CHIVE_BANKS:-$CHIVE_SCALE/banks}
export CHIVE_RUNS=${CHIVE_RUNS:-$CHIVE_SCALE/harness_runs}
export CHIVE_POINTS=${CHIVE_POINTS:-$CHIVE_SCALE/points}
export CHIVE_CONFIDENCE=${CHIVE_CONFIDENCE:-$CHIVE_SCALE/confidence}
export CHIVE_CLAIMS_RUNS=${CHIVE_CLAIMS_RUNS:-$CHIVE_SCALE/claims_runs}
export CHIVE_RUN_DIRS=${CHIVE_RUN_DIRS:-$CHIVE_SCALE/runs}
export CHIVE_DIR=${CHIVE_DIR:-/workspace/agam/chive}
export CHIVE_PY=${CHIVE_PY:-$CHIVE_DIR/.venv/bin/python}
export GW_DIR=${GW_DIR:-/workspace/agam/global-workspace}
export CHIVE_TARGET_URL=${CHIVE_TARGET_URL:-https://safety-sahan-2-agam--chive-qwen-vllm-serve.modal.run}
export CHIVE_LOGS=${CHIVE_LOGS:-/workspace/agam/logs/chive_fresh}
mkdir -p "$CHIVE_LOGS" 2>/dev/null || true
