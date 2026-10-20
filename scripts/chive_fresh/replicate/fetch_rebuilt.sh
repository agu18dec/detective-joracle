#!/bin/bash
# fetch a rebuilt bank once its modal build finished (log has "pull: ./fetch_banks.sh <bank>");
# rep_mlnla_k1 = first sample of every cell of rep_mlnla_k4_chive (analysis/make_k1.py)
B=$1; RB=/workspace/agam/chive_scale/replicate_banks/rebuilt
if [ "$B" = rep_mlnla_k1 ]; then
  [ "$(ls $RB/rep_mlnla_k4_chive 2>/dev/null | grep -c 'json.gz$')" -ge 204 ] || { echo "k4 not ready"; exit 0; }
  python3 /workspace/agam/chive_scale/analysis/make_k1.py $RB/rep_mlnla_k4_chive $RB/rep_mlnla_k1; exit 0
fi
LOG=/workspace/agam/logs/chive_fresh/rep_bank_$(case $B in rep_s3d*) echo olens;; rep_mlnla*) echo mlnla;; rep_nla*) echo nla;; rep_jlens*) echo jlens;; esac).log
grep -q "pull: ./fetch_banks.sh $B" $LOG || { echo "build of $B not finished"; exit 0; }
CHIVE_BANKS_VOLUME=chive-fresh-banks BANKS_DIR=$RB /workspace/agam/chive_scale/banks_code/fetch_banks.sh $B
