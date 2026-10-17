#!/bin/bash
# usage: judge2_all.sh [batch ...]: point eval v1 (run_points.sh) then judge v2, hypothesis-only and
# full, on every arm's base draws (incidents that have points). Set ARMS_J2 to change the arm list.
D=$(cd "$(dirname "$0")" && pwd); source "$D/../env.sh"
ALL=${ARMS_J2:-transcript,olens_rl_B,olens_k1,olens_single,jlens_B,jlens_L2060,mlnla_B,mlnla_k1,olens_shufc,jlens_shufc,mlnla_k1_shufc}
for b in ${*:-fresh1 fresh2 fresh3 fresh4}; do
  "$CHIVE_FRESH_DIR/run_points.sh" "$b" >> "$CHIVE_LOGS/rejudge_points_$b.log" 2>&1
  ids=$(mktemp); ls "$CHIVE_POINTS/$b/points" | sed 's/.json//' > "$ids"
  for mode in "--hyp-only" ""; do
    # shellcheck disable=SC2086
    PYTHONPATH=$CHIVE_DIR "$CHIVE_PY" "$CHIVE_FRESH_DIR/analysis/judge2.py" "$b" "$ids" --draws 0,1,2,3,4 \
      --no-predict $mode --arms "$ALL" >> "$CHIVE_LOGS/rejudge_j2_$b.log" 2>&1
  done
  rm -f "$ids"
done
