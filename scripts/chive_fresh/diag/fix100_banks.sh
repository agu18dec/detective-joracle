#!/bin/bash
# Agam 2026-10-21 ("banks OK"): fix 1 = cleaner readouts. Sparse 223k-cell banks on fix100, frozen harness prompt
# (textfirst+absence) so only the readouts change: s3d_sub223 (control: plain OLens x4 on the same cells),
# omp32_s3d_sub223 (AR picks 4 of 32, olens_tool), osyn8_s3d_sub223 (Opus synthesis of 8, low-hallucination, olens_tool).
# Each arm waits until its bank has every incident of a batch.
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_diag; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
V="['textfirst','absence']"
run_arm() { arm=$1; bank=$2
  for b in fresh1 fresh2; do
    ids=$C/analysis/diag/fix100/$b.txt; need=$(grep -c . $ids)
    until [ "$(for p in $(cat $ids); do [ -s $C/banks/$bank/$p.json.gz ] && echo x; done | wc -l)" -ge "$need" ] && [ ! -e $C/banks/$bank/.building_$b ]; do sleep 300; done
    sleep 60
    ( for i in 1 2; do IDS=$ids CONC=6 $C/harness_exp/run_exp.sh $b agent "['$arm=$C/banks/$bank:olens']" "$V"; done && \
      IDS=$ids CONC=4 $C/harness_exp/run_exp.sh $b judge "['$arm=$C/banks/$bank:olens']" "$V" ) >> $L/fix100_${arm}_$b.log 2>&1
  done; echo "ARM $arm DONE $(date -u)" >> $L/fix100_${arm}_fresh1.log; }
run_arm olens_sub223 s3d_sub223 &
run_arm omp32_sub223 omp32_s3d_sub223 &
run_arm osyn8_sub223 osyn8_s3d_sub223 &
wait; echo "FIX100 BANKS DONE $(date -u)" >> $L/fix100_olens_sub223_fresh1.log
