#!/bin/bash
# 4th readout arm of the fix test (approved 2026-10-22): 4x Qwen-base synthesis -> NN-OMP (olens_tool), sparse 223k cells,
# frozen prompt; compare vs olens_sub223 (same cells). Waits for each batch to be complete.
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_diag; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
V="['textfirst','absence']"; arm=qsynomp_sub223; bank=qsynomp_s3d_sub223
for b in fresh1 fresh2; do
  ids=$C/analysis/diag/fix100/$b.txt; need=$(grep -c . $ids)
  until [ "$(for p in $(cat $ids); do [ -s $C/banks/$bank/$p.json.gz ] && echo x; done | wc -l)" -ge "$need" ] && [ ! -e $C/banks/$bank/.building_$b ]; do sleep 300; done
  sleep 60
  ( for i in 1 2; do IDS=$ids CONC=6 $C/harness_exp/run_exp.sh $b agent "['$arm=$C/banks/$bank:olens']" "$V"; done && \
    IDS=$ids CONC=4 $C/harness_exp/run_exp.sh $b judge "['$arm=$C/banks/$bank:olens']" "$V" ) >> $L/fix100_${arm}_$b.log 2>&1
done; echo "ARM $arm DONE $(date -u)" >> $L/fix100_${arm}_fresh1.log
