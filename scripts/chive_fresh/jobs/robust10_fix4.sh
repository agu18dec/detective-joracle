#!/bin/bash
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh
A="['olens_rl_B=$C/banks/s3d_rl600_all:olens','mlnla_B=$C/banks/mlnla_k4_chive:mlnla']"
IDS=$C/datasets/fresh4/splits/all.txt CONC=12 $C/run_harness.sh fresh4 agent "$A" draws=10 >> $L/robust10_fix4.log 2>&1
IDS=$C/datasets/fresh4/splits/all.txt CONC=12 $C/run_harness.sh fresh4 judge "$A" draws=10 >> $L/robust10_fix4.log 2>&1
echo "FIX4 DONE $(date -u)" >> $L/robust10_fix4.log
