#!/bin/bash
# usage: run_arm.sh "<arms python list>" [batch ...]   (default batches fresh1-4)
# The standard per-arm job: for each batch, PASSES agent passes (default 2: the second pass redoes
# records truncated by 429s / API errors, run_chive.py skips finished ones), then the old pass judge.
# One batch at a time keeps Opus load bounded; CONC (default 16) is per process. Example:
#   source ../env.sh; source arms.sh
#   CONC=16 ./run_arm.sh "$(arms_list "$ARM_OLENS_K1" "$ARM_JLENS_L2060")" fresh1 fresh2
D=$(cd "$(dirname "$0")" && pwd); source "$D/../env.sh"
A=$1; shift
export CONC=${CONC:-16}
for b in ${*:-fresh1 fresh2 fresh3 fresh4}; do
  for _ in $(seq 1 "${PASSES:-2}"); do "$CHIVE_FRESH_DIR/run_harness.sh" "$b" agent "$A" || exit 1; done
  "$CHIVE_FRESH_DIR/run_harness.sh" "$b" judge "$A" || exit 1
done 2>&1 | tee -a "$CHIVE_LOGS/run_arm_$(date +%Y%m%d_%H%M%S).log"
