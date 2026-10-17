#!/bin/bash
# usage: build_and_bank.sh <batch>: build the harness incidents, then the J-lens / NLA / OLens banks on
# Modal in parallel (volume $CHIVE_BANKS_VOLUME, default chive-fresh-banks). Fetch them afterwards with
# banks/fetch_banks.sh <bank>.
source "$(dirname "$0")/env.sh"
B=$1
"$CHIVE_FRESH_DIR/build.sh" "$B" 2>&1 | tee -a "$CHIVE_LOGS/build_$B.log" || exit 1
cd "$CHIVE_FRESH_DIR/banks" || exit 1
for r in jlens nla olens; do
  modal run readers_modal.py --reader $r --dataset "$CHIVE_DATASETS/$B" --ids "$CHIVE_DATASETS/$B/splits/all.txt" \
    --out-volume "${CHIVE_BANKS_VOLUME:-chive-fresh-banks}" > "$CHIVE_LOGS/bank_${r}_$B.log" 2>&1 &
done
wait; echo "BANKS-DONE $B"
