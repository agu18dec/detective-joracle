#!/bin/bash
# usage: requeue_salvaged.sh <batch> <arm-name>: move records whose episode hit a 429 (tool results
# shrunk / forced terminal: "salvaged finish") to focused_salvaged/ so the next run redoes them clean.
D=${CHIVE_CLAIMS_RUNS:-/workspace/agam/chive_scale/claims_runs}/$1; mkdir -p $D/focused_salvaged/$2
n=0; for f in $(grep -l '"stopped_by": "salvaged' $D/focused/$2/*.json 2>/dev/null); do mv $f $D/focused_salvaged/$2/; n=$((n+1)); done
echo "$1/$2: requeued $n salvaged records"
