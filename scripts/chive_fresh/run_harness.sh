#!/bin/bash
# usage: run_harness.sh <batch> <stage> "<arms python list>" [extra pydra args...]
# The frozen detective-joracle harness (scripts/run_chive.py, variants textfirst+absence, 5 draws) on a
# fresh batch, keyless (OAuth token from the Modal Dict). Env: EVAL (record subdir, default <batch>),
# IDS (id file, default the batch's splits/all.txt), CONC (concurrency, default 24; drop to 12-16 when
# several arms run at once: the org-wide Opus input-token limit), CHIVE_EXP_N / CHIVE_BASELINES
# (budgeted-counterfactual protocol). Records: $CHIVE_RUNS/<eval>/<arm>/<pid>.d<draw>.json.
source "$(dirname "$0")/env.sh"
B=$1; S=$2; A=$3; shift 3
cd "$HARNESS_DIR" || exit 1
exec "$HARNESS_PY" -u scripts/run_chive.py stage="$S" eval="${EVAL:-$B}" ids="${IDS:-$CHIVE_DATASETS/$B/splits/all.txt}" \
  dataset="$CHIVE_DATASETS/$B" out="$CHIVE_RUNS" draws=5 concurrency="${CONC:-24}" \
  "variants=['textfirst','absence']" "arms=$A" "$@"
