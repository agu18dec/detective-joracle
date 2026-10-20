#!/bin/bash
# point eval (judge v2) on robust10's extra draws d5-d9: transcript, OLens x4, NLA x4 (+ J-lens when done), hyp-only and full
C=/workspace/agam/chive_scale; source $C/banks_code/env.sh; unset ANTHROPIC_API_KEY; cd $C
L=/workspace/agam/logs/chive_fresh/judge2_d59.log
for b in fresh1 fresh2 fresh3 fresh4; do
  ls points/$b/points | sed 's/.json//' > /tmp/j59_$b.txt
  for mode in "--hyp-only" ""; do
    J2_THREADS=6 PYTHONPATH=/workspace/agam/chive /workspace/agam/chive/.venv/bin/python analysis/judge2.py $b /tmp/j59_$b.txt --draws 5,6,7,8,9 --no-predict $mode --arms transcript,olens_rl_B,mlnla_B >> $L 2>&1 &
  done
done; wait; echo "J2 D59 DONE $(date -u)" >> $L
