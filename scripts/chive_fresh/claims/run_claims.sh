#!/bin/bash
# usage: run_claims.sh <batch> <arm-spec> [extra pydra args]
# detective-joracle claims_chive.py stage=agent (focused episode, Opus 5, effort max, draws 1,
# max_calls 30, output budget 40k), keyless via Modal Dict. Before every pass, records whose
# episode hit an API error mid-run ("salvaged finish": tool results shrunk / forced answer) are
# moved to focused_salvaged/ so they are redone clean; result-less (error) records are redone by
# claims_chive.py itself. Loops until a pass has nothing to run.
source "$(dirname "$0")/../env.sh"
B=$1; A=$2; shift 2; NAME=${A%%=*}
cd "$HARNESS_DIR" || exit 1
for i in $(seq 1 ${PASSES:-10}); do
  echo "=== pass $i $(date)"
  "$CHIVE_FRESH_DIR/claims/requeue_salvaged.sh" $B $NAME
  "$HARNESS_PY" -u scripts/claims_chive.py stage=agent claims="$CHIVE_CLAIMS_RUNS/$B/claims.jsonl" \
    dataset="$CHIVE_DATASETS/$B" out="$CHIVE_CLAIMS_RUNS/$B" eval=focused \
    draws=1 concurrency=${CONC:-12} "arms=['$A']" "$@" 2>&1 | tee /tmp/claims_pass_$$.log
  grep -q "agent: 0 episodes to run" /tmp/claims_pass_$$.log && break
  sleep 60
done
echo "=== done $(date)"
