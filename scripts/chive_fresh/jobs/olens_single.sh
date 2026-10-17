#!/bin/bash
# usage: olens_single.sh <batch> [num_shards]: single-readout OLens for one batch: build the
# s3_single_rl bank on Modal (resumable, up to 3 passes), fetch it race-safely, then the harness
# agent (2 passes) + judge for arm olens_single.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); source "$D/../env.sh"; source "$D/arms.sh"
B=$1; NS=${2:-40}
IDS=$CHIVE_DATASETS/$B/splits/all.txt; N=$(grep -c . "$IDS"); have=0
cd "$CHIVE_FRESH_DIR/banks" || exit 1
for attempt in 1 2 3; do
  echo "$(date +%T) [$B] bank pass $attempt ($NS shards)"
  CHIVE_BANKS_GPU=${CHIVE_BANKS_GPU:-H200,B200} modal run readers_modal.py --reader olens_single \
    --dataset "$CHIVE_DATASETS/$B" --ids "$IDS" --num-shards "$NS" --skip-prep
  out=$(bash fetch_ids.sh "$IDS" s3_single_rl | tail -1); echo "$(date +%T) [$B] fetch: $out"
  have=$(echo "$out" | sed 's/.*: \([0-9]*\)\/.*/\1/')
  [ "$have" = "$N" ] && break
done
[ "$have" = "$N" ] || { echo "[$B] bank incomplete ($have/$N), stopping"; exit 1; }
CONC=${CONC:-12} "$D/run_arm.sh" "$(arms_list "$ARM_OLENS_SINGLE")" "$B"
