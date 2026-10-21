#!/bin/bash
# Phase 5 dev: echotag + revgate on dev240 (fresh1/2 dev splits), 3 draws; OLens x4, NLA x4, single-bullet OLens x16
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_diag
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
V="['textfirst','absence','echotag','revgate']"
for b in fresh1 fresh2; do
  for spec in "olens_gate|$C/banks/s3d_rl600_all:olens" "mlnla_gate|$C/banks/mlnla_k4_chive:mlnla" "olens_s16_gate|$C/banks/s3_single_rl_k16:olens"; do
    IFS='|' read -r arm bank <<< "$spec"
    ( IDS=$C/analysis/harness_improve/dev/$b.txt CONC=7 $C/harness_exp/run_exp.sh $b agent "['$arm=$bank']" "$V" && \
      IDS=$C/analysis/harness_improve/dev/$b.txt CONC=7 $C/harness_exp/run_exp.sh $b agent "['$arm=$bank']" "$V" && \
      IDS=$C/analysis/harness_improve/dev/$b.txt CONC=8 $C/harness_exp/run_exp.sh $b judge "['$arm=$bank']" "$V" ) > $L/gate_${arm}_$b.log 2>&1 &
  done
done
wait; echo "GATE DEV DONE $(date -u)" >> $L/gate_olens_gate_fresh1.log
