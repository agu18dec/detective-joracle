#!/bin/bash
# ceiling v2 (statable mechanisms, prompt side, strict per-line Opus judge) on 80 random incidents per batch;
# thread-capped: 4 procs/batch x 6 item-threads; single-threaded BLAS; retries after 401
C=/workspace/agam/chive_scale; source $C/banks_code/env.sh; unset ANTHROPIC_API_KEY; cd $C
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
L=/workspace/agam/logs/chive_diag; PY=/workspace/agam/chive/.venv/bin/python
for b in fresh1 fresh2 fresh3 fresh4; do
  ( for i in $(seq 48); do find $C/diag/ceiling2/$b -name '*.tmp' -delete 2>/dev/null; $PY -u analysis/diag/p2b_ceiling2.py $b --n 80 --procs 4 >> $L/ceiling2_$b.log 2>&1 && break; echo "retry $i $(date -u)" >> $L/ceiling2_$b.log; sleep 300; done ) &
done; wait; echo "CEILING2 ALL DONE $(date -u)" >> $L/ceiling2_fresh1.log
