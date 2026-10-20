#!/bin/bash
# Robust numbers on all 915: draws d5-d9 for transcript, OLens x4, multi-layer NLA x4, J-lens L20-60 (then judge).
# OLens/NLA skip the least-confident third (hx-lcmore already adds d5-d9 there).
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh
for b in fresh1 fresh2 fresh3 fresh4; do
  for spec in "transcript|['transcript']|all" "olens|['olens_rl_B=$C/banks/s3d_rl600_all:olens']|rest" \
              "mlnla|['mlnla_B=$C/banks/mlnla_k4_chive:mlnla']|rest" "jlens|['jlens_L2060=$C/banks/jlens_L20-60:jlens']|all"; do
    IFS='|' read -r tag A which <<< "$spec"
    ids=$C/datasets/$b/splits/all.txt; [ "$which" = rest ] && ids=$C/analysis/lc_more/${b}_rest.txt
    ( IDS=$ids CONC=${CONC:-5} $C/run_harness.sh $b agent "$A" draws=10 && IDS=$ids CONC=12 $C/run_harness.sh $b judge "$A" draws=10 ) > $L/robust10_${tag}_$b.log 2>&1 &
  done
done
wait; echo "ROBUST10 DONE $(date -u)" >> $L/robust10_transcript_fresh1.log
