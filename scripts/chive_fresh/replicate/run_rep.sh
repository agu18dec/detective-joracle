#!/bin/bash
# usage: run_rep.sh <eval> <stage> "<arms python list>" [extra pydra args...]
# replicate of the CHIVE held-out (climb_test204, HF agu18dec/chive-forensics): frozen harness
# (variants textfirst+absence), 5 draws, Opus 5 effort max, keyless via Modal Dict
export PATH=$HOME/.local/bin:$PATH; set -a; . /dev/shm/compute.env; set +a; export MODAL_ENVIRONMENT=agam; unset ANTHROPIC_API_KEY
E=$1; S=$2; A=$3; shift 3
cd /workspace/agam/chive_scale/harness
exec .venv/bin/python -u scripts/run_chive.py stage=$S eval=$E ids=/workspace/agam/chive_scale/harness/data/chive/splits/climb_test204.txt \
  dataset=agu18dec/chive-forensics out=/workspace/agam/chive_scale/replicate_runs draws=5 concurrency=${CONC:-16} \
  hard=/workspace/agam/chive_scale/harness/data/chive/splits/hard92.txt \
  "variants=['textfirst','absence']" "arms=$A" "$@"
