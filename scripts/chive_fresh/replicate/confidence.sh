#!/bin/bash
# transcript-only confidence on all 300 (original tercile = bottom 100 of 300 by top_p, ∩ the 204)
export PATH=$HOME/.local/bin:$PATH; set -a; . /dev/shm/compute.env; set +a; export MODAL_ENVIRONMENT=agam; unset ANTHROPIC_API_KEY
cd /workspace/agam/chive_scale/harness
for i in 1 2 3; do
  .venv/bin/python -u scripts/confidence_chive.py out=/workspace/agam/chive_scale/replicate_runs ids=data/chive/splits/all300.txt dataset=agu18dec/chive-forensics concurrency=3
done
