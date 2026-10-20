#!/bin/bash
# 5 extra draws (d5-d9) for OLens x4 and multi-layer NLA x4 on the 915's least-confident third, then judge.
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh
A="['olens_rl_B=$C/banks/s3d_rl600_all:olens','mlnla_B=$C/banks/mlnla_k4_chive:mlnla']"
for b in fresh1 fresh2 fresh3 fresh4; do
  ( IDS=$C/analysis/lc_more/$b.txt CONC=${CONC:-10} $C/run_harness.sh $b agent "$A" draws=10 && \
    IDS=$C/analysis/lc_more/$b.txt CONC=16 $C/run_harness.sh $b judge "$A" draws=10 ) > $L/lc_more_$b.log 2>&1 &
done
wait; echo "LC MORE DONE $(date -u)" >> $L/lc_more_fresh1.log
