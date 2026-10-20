#!/bin/bash
# usage: judge_loop.sh <batch> -- judge v2 (hyp-only + full) and point eval v1 on EVERY arm, repeatedly as records land
# (all stages cache per record), with a final pass after the batch's arm sessions have ended.
b=$1; C=/workspace/agam/chive_scale; source $C/banks_code/env.sh; unset ANTHROPIC_API_KEY; cd $C
ALL=transcript,olens_rl_B,olens_k1,olens_single,olens_single_k1,olens_single_k2,olens_single_k8,olens_single_k16,jlens_B,jlens_L2060,mlnla_B,mlnla_k1,olens_shufc,jlens_shufc,mlnla_k1_shufc
L=/workspace/agam/logs/chive_fresh/judgeloop_$b.log
pass() {
  ls points/$b/points | sed 's/.json//' > /tmp/jl_$b.txt
  for mode in "--hyp-only" ""; do
    J2_THREADS=6 PYTHONPATH=/workspace/agam/chive /workspace/agam/chive/.venv/bin/python analysis/judge2.py $b /tmp/jl_$b.txt --draws 0,1,2,3,4 --no-predict $mode --arms $ALL >> $L 2>&1 &
  done
  $C/run_points.sh $b >> $L 2>&1; wait
}
while tmux ls 2>/dev/null | cut -d: -f1 | grep -q -E "^hx-(mlnla1|shufc|newarms|single|single1|singlek|singlek2|singlek8|singlek16|jlensL)-$b$|^hx-refill-f${b#fresh}"; do pass; echo "pass done $(date -u)" >> $L; sleep 600; done
pass; echo "FINAL pass done $(date -u)" >> $L
