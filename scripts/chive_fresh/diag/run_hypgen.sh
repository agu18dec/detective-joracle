#!/bin/bash
C=/workspace/agam/chive_scale; source $C/banks_code/env.sh; unset ANTHROPIC_API_KEY; cd $C
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; L=/workspace/agam/logs/chive_diag
for b in fresh1 fresh2 fresh3 fresh4; do
  ( for i in $(seq 48); do find $C/diag/hypgen/$b -name '*.tmp' -delete 2>/dev/null; HG_THREADS=8 /workspace/agam/chive/.venv/bin/python -u analysis/diag/p4a_hypgen.py $b >> $L/hypgen_$b.log 2>&1 && break; echo "retry $i $(date -u)" >> $L/hypgen_$b.log; sleep 300; done ) &
done; wait; echo "HYPGEN ALL DONE $(date -u)" >> $L/hypgen_fresh1.log
