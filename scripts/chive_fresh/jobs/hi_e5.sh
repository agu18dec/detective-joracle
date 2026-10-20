#!/bin/bash
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh
for b in fresh1 fresh2; do
  for spec in "olens_mc|['textfirst','absence','multicand']|$C/banks/s3d_rl600_all:olens" \
              "mlnla_mc|['textfirst','absence','multicand']|$C/banks/mlnla_k4_chive:mlnla" \
              "olens_s16_mcr|['textfirst','absence','multicand','recur']|$C/banks/s3_single_rl_k16:olens"; do
    IFS='|' read -r arm V bank <<< "$spec"
    ( IDS=$C/analysis/harness_improve/dev/$b.txt CONC=6 $C/harness_exp/run_exp.sh $b agent "['$arm=$bank']" "$V" && \
      IDS=$C/analysis/harness_improve/dev/$b.txt CONC=6 $C/harness_exp/run_exp.sh $b agent "['$arm=$bank']" "$V" && \
      IDS=$C/analysis/harness_improve/dev/$b.txt CONC=12 $C/harness_exp/run_exp.sh $b judge "['$arm=$bank']" "$V" ) > $L/hi_e5_${arm}_$b.log 2>&1 &
  done
done
wait; echo "E5 DONE $(date -u)" >> $L/hi_e5_olens_mc_fresh1.log
