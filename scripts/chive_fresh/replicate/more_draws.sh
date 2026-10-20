#!/bin/bash
# 5 extra draws (d5-d9) for multi-layer NLA and OLens on climb_test204, original banks, then judge.
S=/workspace/agam/chive_scale/replicate_runs/scripts/run_rep.sh
L=/workspace/agam/logs/chive_fresh
( CONC=16 $S rep_mlnla_orig agent "['mlnla_B=/workspace/agam/chive_scale/replicate_banks/orig/mlnla_k4_chive:mlnla']" draws=10 && \
  CONC=24 $S rep_mlnla_orig judge "['mlnla_B=/workspace/agam/chive_scale/replicate_banks/orig/mlnla_k4_chive:mlnla']" draws=10 ) > $L/more_mlnla.log 2>&1 &
( CONC=16 $S rep_olens_orig agent "['olens_rl_B=/workspace/agam/chive_scale/replicate_banks/orig/s3d_rl600_all:olens']" draws=10 && \
  CONC=24 $S rep_olens_orig judge "['olens_rl_B=/workspace/agam/chive_scale/replicate_banks/orig/s3d_rl600_all:olens']" draws=10 ) > $L/more_olens.log 2>&1 &
wait; echo "MORE DRAWS DONE $(date -u)" >> $L/more_mlnla.log
