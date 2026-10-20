#!/bin/bash
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh
V="['textfirst','absence','recur']"
for b in fresh1 fresh2; do
  for spec in "olens_recur=$C/banks/s3d_rl600_all:olens" "mlnla_recur=$C/banks/mlnla_k4_chive:mlnla" "olens_s16_recur=$C/banks/s3_single_rl_k16:olens"; do
    tag=${spec%%=*}
    ( IDS=$C/analysis/harness_improve/dev/$b.txt $C/harness_exp/run_exp.sh $b agent "['$spec']" "$V" && \
      IDS=$C/analysis/harness_improve/dev/$b.txt CONC=12 $C/harness_exp/run_exp.sh $b judge "['$spec']" "$V" ) > $L/hi_e2_${tag}_$b.log 2>&1 &
  done
done
wait; echo "E2 DONE $(date -u)" >> $L/hi_e2_olens_recur_fresh1.log
