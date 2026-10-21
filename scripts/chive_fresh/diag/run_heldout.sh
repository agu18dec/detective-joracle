#!/bin/bash
# held-out test (see analysis/diag/heldout/PREREG.md)
export CHIVE_EXP_N=30 CHIVE_BASELINES=/workspace/agam/chive_scale/exp1/baselines.json OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_diag; U=https://safety-sahan-2-agam--chive-qwen-vllm-serve.modal.run
G="['textfirst','absence','echotag','revgate']"; GX="['textfirst','absence','echotag','revgate','rexp']"; RX="['textfirst','absence','rexp']"
run() { # b arm spec variants conc extra...
  b=$1; arm=$2; spec=$3; V=$4; K=$5; shift 5
  ( for i in 1 2; do IDS=$C/analysis/diag/heldout/$b.txt CONC=$K $C/harness_exp/run_exp.sh $b agent "['$spec']" "$V" "$@"; done && \
    IDS=$C/analysis/diag/heldout/$b.txt CONC=8 $C/harness_exp/run_exp.sh $b judge "['$spec']" "$V" ) > $L/heldout_${arm}_$b.log 2>&1 &
}
for b in fresh3 fresh4; do
  run $b olens_s16_gate "olens_s16_gate=$C/banks/s3_single_rl_k16:olens" "$G" 3
  run $b mlnla_gate_h "mlnla_gate_h=$C/banks/mlnla_k4_chive:mlnla" "$G" 3
  run $b olens_s16_gx3 "olens_s16_gx3=$C/banks/s3_single_rl_k16:olens" "$GX" 3 experiments=3 target=$U
  run $b mlnla_gx3 "mlnla_gx3=$C/banks/mlnla_k4_chive:mlnla" "$GX" 3 experiments=3 target=$U
  run $b transcript_rx3h "transcript_rx3h" "$RX" 3 experiments=3 target=$U
done
wait; echo "HELDOUT DONE $(date -u)" >> $L/heldout_olens_s16_gate_fresh3.log
