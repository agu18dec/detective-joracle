#!/bin/bash
# Phase 4b/5 dev: budget-3 counterfactual experiments with "experiments decide" prompt (rexp), n=30 per experiment,
# baseline re-measured by the same executor; dev240, 2 draws; transcript / OLens x4 / NLA x4
export CHIVE_EXP_N=30 CHIVE_BASELINES=/workspace/agam/chive_scale/exp1/baselines.json OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_diag; U=https://safety-sahan-2-agam--chive-qwen-vllm-serve.modal.run
V="['textfirst','absence','rexp']"
for b in fresh1 fresh2; do
  for spec in "transcript_rx3" "olens_rx3=$C/banks/s3d_rl600_all:olens" "mlnla_rx3=$C/banks/mlnla_k4_chive:mlnla"; do
    arm=${spec%%=*}
    ( for i in 1 2; do IDS=$C/analysis/harness_improve/dev/$b.txt CONC=8 $C/harness_exp/run_exp.sh $b agent "['$spec']" "$V" draws=2 experiments=3 target=$U; done && \
      IDS=$C/analysis/harness_improve/dev/$b.txt CONC=8 $C/harness_exp/run_exp.sh $b judge "['$spec']" "$V" draws=2 ) > $L/rexp_${arm}_$b.log 2>&1 &
  done
done
wait; echo "REXP DEV DONE $(date -u)" >> $L/rexp_olens_rx3_fresh1.log
