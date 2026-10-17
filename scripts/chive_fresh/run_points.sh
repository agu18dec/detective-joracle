#!/bin/bash
# usage: run_points.sh <batch>: point eval v1 (stages points, tag, plain, grade, rivals, report) of every
# arm vs transcript on the base draws d0-d4 (POINTS_DRAW_GLOB), via scripts/points_chive.py with
# POINTS_ARMS pointing at $CHIVE_RUNS/<batch>/<arm>. Override the arm list with ARMS="a b c".
source "$(dirname "$0")/env.sh"
B=$1; R=$CHIVE_RUNS/$B
ARMS=${ARMS:-transcript olens_rl_B jlens_B mlnla_B mlnla_k1 olens_shufc jlens_shufc mlnla_k1_shufc olens_k1 olens_single jlens_L2060}
POINTS_ARMS=$(python3 -c 'import json,sys; print(json.dumps({a: sys.argv[1] + "/" + a for a in sys.argv[2:]}))' "$R" $ARMS)
export POINTS_ARMS POINTS_DRAW_GLOB=${POINTS_DRAW_GLOB:-'*.d[0-4].json'}
cd "$HARNESS_DIR" || exit 1
for S in points tag plain grade rivals report; do
  "$HARNESS_PY" -u scripts/points_chive.py stage=$S out="$CHIVE_POINTS/$B" \
    ids="$CHIVE_DATASETS/$B/splits/all.txt" dataset="$CHIVE_DATASETS/$B" \
    hard="$CHIVE_DATASETS/$B/splits/all.txt" confidence=/nonexistent concurrency="${CONC:-48}" \
    || { echo "STAGE $S FAILED"; [ $S = report ] || exit 1; }
done
