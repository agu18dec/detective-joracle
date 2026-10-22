#!/bin/bash
export PATH=$HOME/.local/bin:$PATH; set -a; . /dev/shm/compute.env; set +a; export MODAL_ENVIRONMENT=agam OMP_NUM_THREADS=1; unset ANTHROPIC_API_KEY
cd /workspace/agam/chive_scale/harness
for i in $(seq 10); do REP_CONC=6 PYTHONPATH=src .venv/bin/python -u ../analysis/diag/p7_replicate.py run --n 80 >> /workspace/agam/logs/chive_diag/replicate.log 2>&1 && break; sleep 300; done
