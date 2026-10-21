#!/bin/bash
# bank ceiling, thread-capped (pod pids.max=4608 shared): 16 worker processes TOTAL, single-threaded BLAS; retries after 401
C=/workspace/agam/chive_scale; source $C/banks_code/env.sh; unset ANTHROPIC_API_KEY; cd $C
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
L=/workspace/agam/logs/chive_diag; PY=/workspace/agam/chive/.venv/bin/python
for b in fresh1 fresh2 fresh3 fresh4; do
  BK=olens,mlnla,jlens; [ $b = fresh1 ] && BK=olens,mlnla,jlens,olens_shuf,mlnla_shuf
  ( for i in $(seq 48); do find $C/diag/ceiling/$b -name '*.tmp' -delete 2>/dev/null; $PY -u analysis/diag/p2_ceiling.py $b --procs 4 --banks $BK >> $L/ceiling_$b.log 2>&1 && break; echo "retry $i $(date -u)" >> $L/ceiling_$b.log; sleep 300; done ) &
done; wait; echo "CEILING ALL DONE $(date -u)" >> $L/ceiling_fresh1.log
