#!/bin/bash
# usage: exp1_n30.sh [batch ...]: the budgeted counterfactual (confirmed by Agam 2026-10-16): experiment
# budget 1, 2 draws, CHIVE_EXP_N=30 Qwen samples per experiment, compared against the baseline
# re-measured with the same executor (CHIVE_BASELINES, from ../baselines.py). Eval <batch>_exp1n30:
# agent, then the old pass judge, then the concluder. Waits until the baselines cover all but <= 2 of
# the batch's incidents.
D=$(cd "$(dirname "$0")" && pwd); source "$D/../env.sh"; source "$D/arms.sh"
export CHIVE_EXP_N=30 CHIVE_BASELINES=${CHIVE_BASELINES:-$CHIVE_SCALE/exp1/baselines.json}
A=$(arms_list "$ARM_TRANSCRIPT" "$ARM_OLENS" "$ARM_JLENS" "$ARM_MLNLA_K1" "$ARM_OLENS_K1" "$ARM_OLENS_SINGLE" "$ARM_JLENS_L2060")
for b in ${*:-fresh1 fresh2 fresh3 fresh4}; do
  until python3 -c "import json,sys; d=json.load(open('$CHIVE_BASELINES')); ids=open('$CHIVE_DATASETS/$b/splits/all.txt').read().split(); sys.exit(0 if sum(i not in d for i in ids) <= 2 else 1)"; do sleep 300; done
  EVAL=${b}_exp1n30 "$CHIVE_FRESH_DIR/run_harness.sh" "$b" agent "$A" draws=2 experiments=1 target="$CHIVE_TARGET_URL" && \
  EVAL=${b}_exp1n30 "$CHIVE_FRESH_DIR/run_harness.sh" "$b" judge "$A" && \
  EVAL=${b}_exp1n30 "$CHIVE_FRESH_DIR/run_harness.sh" "$b" concluder "$A"
done 2>&1 | tee -a "$CHIVE_LOGS/exp1n30.log"
