#!/bin/bash
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh; ID=$C/analysis/harness_improve/dev/fresh2.txt
IDS=$ID CONC=4 $C/harness_exp/run_exp.sh fresh2 agent "['olens_recur=$C/banks/s3d_rl600_all:olens']" "['textfirst','absence','recur']" >> $L/hi_e5_fix.log 2>&1
IDS=$ID CONC=4 $C/harness_exp/run_exp.sh fresh2 judge "['olens_recur=$C/banks/s3d_rl600_all:olens']" "['textfirst','absence','recur']" >> $L/hi_e5_fix.log 2>&1
for spec in "olens_mc|['textfirst','absence','multicand']|$C/banks/s3d_rl600_all:olens" "mlnla_mc|['textfirst','absence','multicand']|$C/banks/mlnla_k4_chive:mlnla" "olens_s16_mcr|['textfirst','absence','multicand','recur']|$C/banks/s3_single_rl_k16:olens"; do
  IFS='|' read -r arm V bank <<< "$spec"
  IDS=$ID CONC=12 $C/harness_exp/run_exp.sh fresh2 judge "['$arm=$bank']" "$V" >> $L/hi_e5_fix.log 2>&1 &
done; wait; echo "E5 FIX DONE $(date -u)" >> $L/hi_e5_fix.log
