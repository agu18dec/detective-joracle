#!/bin/bash
# after E2/E3 finish: redo their salvaged/error runs (agent stage reruns them), judge, then resume the bulk jobs
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh
redo() {  # $1 = log tag, $2 = variants, rest = arm specs
  tag=$1; V=$2; shift 2
  for b in fresh1 fresh2; do for spec in "$@"; do
    ( IDS=$C/analysis/harness_improve/dev/$b.txt CONC=6 $C/harness_exp/run_exp.sh $b agent "['$spec']" "$V" && \
      IDS=$C/analysis/harness_improve/dev/$b.txt CONC=12 $C/harness_exp/run_exp.sh $b judge "['$spec']" "$V" ) >> $L/${tag}_redo.log 2>&1 &
  done; done; wait
}
until grep -q "E2 DONE" $L/hi_e2_olens_recur_fresh1.log 2>/dev/null; do sleep 120; done
redo hi_e2 "['textfirst','absence','recur']" "olens_recur=$C/banks/s3d_rl600_all:olens" "mlnla_recur=$C/banks/mlnla_k4_chive:mlnla" "olens_s16_recur=$C/banks/s3_single_rl_k16:olens"
echo "E2 REDO DONE $(date -u)" >> $L/hi_e2_redo.log
until grep -q "E3 DONE" $L/hi_e3_olens_filt_fresh1.log 2>/dev/null; do sleep 120; done
redo hi_e3 "['textfirst','absence','llmfilter']" "olens_filt=$C/banks/s3d_rl600_all:olens" "mlnla_filt=$C/banks/mlnla_k4_chive:mlnla"
echo "E3 REDO DONE $(date -u)" >> $L/hi_e3_redo.log
tmux new -d -s hx-lcmore "CONC=6 $C/jobs/lc_more.sh"
tmux new -d -s hx-robust10 "CONC=3 $C/jobs/robust10.sh"
echo "bulk resumed $(date -u)" >> $L/resume_bulk.log
