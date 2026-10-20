#!/bin/bash
# usage: build_B.sh <reader> <bank> <num_shards>   -- rebuilt banks (current builder) for the 204 climb_test incidents
source /workspace/agam/chive_scale/banks_code/env.sh
cd /workspace/agam/chive_scale/banks_code
modal run readers_modal.py --reader $1 --bank $2 --num-shards $3 --skip-prep \
  --dataset /workspace/agam/chive_scale/datasets/climb_test \
  --ids /workspace/agam/chive_scale/datasets/climb_test/splits/climb_test204.txt \
  --out-volume chive-fresh-banks
echo "EXIT $?"
