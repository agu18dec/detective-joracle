#!/bin/bash
export PATH=$HOME/.local/bin:$PATH; set -a; . /dev/shm/compute.env; set +a; export MODAL_ENVIRONMENT=agam; unset ANTHROPIC_API_KEY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; L=/workspace/agam/logs/chive_diag
cd /workspace/agam/chive_scale/harness
for b in ${BATCHES:-fresh1 fresh2 fresh4}; do
  ( for i in $(seq 48); do CJ_CONC=6 PYTHONPATH=src .venv/bin/python -u ../analysis/diag/p3_commitjudge.py $b >> $L/commitjudge_$b.log 2>&1 && grep -q "COMMITJUDGE $b DONE" $L/commitjudge_$b.log && break; echo "retry $i exit=$? $(date -u)" >> $L/commitjudge_$b.log; sleep 300; done ) &
done; wait
