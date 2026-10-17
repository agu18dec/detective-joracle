#!/bin/bash
# usage: confidence.sh [batch ...]: the pre-registered least-confident-third stratum (Opus, no tools,
# top_p of its own answer; scripts/confidence_chive.py) -> $CHIVE_CONFIDENCE/<batch>/confidence.json
D=$(cd "$(dirname "$0")" && pwd); source "$D/../env.sh"
cd "$HARNESS_DIR" || exit 1
for b in ${*:-fresh1 fresh2 fresh3 fresh4}; do
  "$HARNESS_PY" -u scripts/confidence_chive.py out="$CHIVE_CONFIDENCE/$b" ids="$CHIVE_DATASETS/$b/splits/all.txt" \
    dataset="$CHIVE_DATASETS/$b" concurrency=12
done 2>&1 | tee -a "$CHIVE_LOGS/confidence.log"
