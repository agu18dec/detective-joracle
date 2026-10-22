#!/bin/bash
# Agam 2026-10-21: the "good harness" on all 915 fresh incidents: frozen harness + 3 counterfactual experiments (n=30,
# executor-measured baselines) + "experiments decide" prompt (rexp) + readout support counts (support), 3 draws.
# Arms: transcript_x3 (seeded with identical-prompt rx3/rx3h records), olens_x3 (OLens x4), olens16_x3 (single-bullet x16),
# mlnla_x3 (NLA x4), jlens_x3 (J-lens L20-60). One loop per arm over batches; CONC per run from $CONC (throttled ~30 total).
export CHIVE_EXP_N=30 CHIVE_BASELINES=/workspace/agam/chive_scale/exp1/baselines.json OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_diag; U=https://safety-sahan-2-agam--chive-qwen-vllm-serve.modal.run
V="['textfirst','absence','rexp','support']"
arm_loop() { spec=$1; arm=${spec%%=*}
  for b in fresh1 fresh2 fresh3 fresh4; do
    for i in 1 2; do IDS=$C/datasets/$b/splits/all.txt CONC=$(cat $C/jobs/full_x3.conc) $C/harness_exp/run_exp.sh $b agent "['$spec']" "$V" experiments=3 target=$U; done
    IDS=$C/datasets/$b/splits/all.txt CONC=4 $C/harness_exp/run_exp.sh $b judge "['$spec']" "$V"
  done >> $L/full_${arm}.log 2>&1; echo "ARM $arm DONE $(date -u)" >> $L/full_${arm}.log; }
for spec in "transcript_x3" "olens_x3=$C/banks/s3d_rl600_all:olens" "olens16_x3=$C/banks/s3_single_rl_k16:olens" "mlnla_x3=$C/banks/mlnla_k4_chive:mlnla" "jlens_x3=$C/banks/jlens_L20-60:jlens"; do
  arm_loop "$spec" &
done; wait; echo "FULL X3 DONE $(date -u)" >> $L/full_transcript_x3.log
