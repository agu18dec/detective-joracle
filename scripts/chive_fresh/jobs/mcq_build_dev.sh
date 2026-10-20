#!/bin/bash
export PATH=$HOME/.local/bin:$PATH; set -a; . /dev/shm/compute.env; set +a; export MODAL_ENVIRONMENT=agam; unset ANTHROPIC_API_KEY
C=/workspace/agam/chive_scale; cd $C/harness
for b in fresh1 fresh2; do
  PYTHONPATH=src .venv/bin/python -u ../analysis/mcq/build_mcq10.py $C/datasets/$b $C/analysis/harness_improve/dev/$b.txt $C/analysis/mcq/dev_$b >> /workspace/agam/logs/chive_fresh/mcq_build.log 2>&1 &
done; wait; echo "MCQ BUILD DONE $(date -u)" >> /workspace/agam/logs/chive_fresh/mcq_build.log
