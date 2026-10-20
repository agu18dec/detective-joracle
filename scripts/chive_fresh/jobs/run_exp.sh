#!/bin/bash
# experimental harness (harness_exp): usage run_exp.sh <batch> <stage> "<arms>" "<variants list>" [extra args]
export PATH=$HOME/.local/bin:$PATH; set -a; . /dev/shm/compute.env; set +a; export MODAL_ENVIRONMENT=agam; unset ANTHROPIC_API_KEY
B=$1; S=$2; A=$3; V=$4; shift 4
cd /workspace/agam/chive_scale/harness_exp
PYTHONPATH=src exec .venv/bin/python -u scripts/run_chive.py stage=$S eval=$B ids=${IDS} \
  dataset=/workspace/agam/chive_scale/datasets/$B out=/workspace/agam/chive_scale/harness_runs draws=3 concurrency=${CONC:-8} \
  "variants=$V" "arms=$A" "$@"
