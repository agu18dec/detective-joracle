#!/bin/bash
# fetch original harness-format banks (chive-bank-readers) for the 204 climb_test ids, per file, low parallelism
source /workspace/agam/chive_scale/banks_code/env.sh
IDS=/workspace/agam/chive_scale/harness/data/chive/splits/climb_test204.txt
D=/workspace/agam/chive_scale/replicate_banks/orig
for b in "$@"; do
  mkdir -p $D/$b
  grep -v '^\s*$' $IDS | while read pid; do
    [ -s $D/$b/$pid.json.gz ] && continue
    echo "$b $pid"
  done
done | xargs -P 4 -n 2 sh -c 'modal volume get --force chive-bank-readers /$0/$1.json.gz '$D'/$0/$1.json.gz >/dev/null 2>&1 || echo FAIL $0 $1'
for b in "$@"; do echo "$b $(ls $D/$b | wc -l)"; done
