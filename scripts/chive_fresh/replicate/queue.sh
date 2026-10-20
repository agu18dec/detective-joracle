#!/bin/bash
# usage: queue.sh <log> item...   item = EVAL|ARMSPEC[|REBUILT_BANK]   (ARMSPEC = transcript or NAME=DIR:KIND)
# per item: [wait for the rebuilt bank + fetch] -> agent -> judge (resumable; reruns until no errors)
L=$1; shift
R=/workspace/agam/chive_scale/replicate_runs/scripts
for item in "$@"; do
  IFS='|' read -r E ARM BANK <<< "$item"
  if [ -n "$BANK" ]; then
    D=/workspace/agam/chive_scale/replicate_banks/rebuilt/$BANK
    until [ "$(ls $D 2>/dev/null | grep -c 'json.gz$')" -ge 204 ]; do
      echo "$(date +%T) waiting for bank $BANK" >> $L; $R/fetch_rebuilt.sh $BANK >> $L 2>&1; sleep 600; done
  fi
  for i in 1 2 3; do
    echo "$(date +%T) === $E $ARM agent (pass $i)" >> $L
    $R/run_rep.sh $E agent "['$ARM']" >> $L 2>&1
    echo "$(date +%T) === $E $ARM judge (pass $i)" >> $L
    $R/run_rep.sh $E judge "['$ARM']" >> $L 2>&1
  done
  $R/run_rep.sh $E report "['$ARM']" >> $L 2>&1
done
echo "$(date +%T) QUEUE DONE" >> $L
