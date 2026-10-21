#!/bin/bash
# judge v3 on fresh1-4 (cached; retries after 401/crash every 5 min, up to 48 times)
C=/workspace/agam/chive_scale; source $C/banks_code/env.sh; unset ANTHROPIC_API_KEY; cd $C
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=/workspace/agam/logs/chive_diag
for b in fresh1 fresh2 fresh3 fresh4; do
  ( for i in $(seq 48); do J3_THREADS=24 /workspace/agam/chive/.venv/bin/python -u analysis/diag/judge3.py $b >> $L/judge3_$b.log 2>&1 && break; echo "retry $i $(date -u)" >> $L/judge3_$b.log; sleep 300; done ) &
done; wait; echo "JUDGE3 ALL DONE $(date -u)" >> $L/judge3_fresh1.log
