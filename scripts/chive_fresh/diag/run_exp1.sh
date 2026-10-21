#!/bin/bash
C=/workspace/agam/chive_scale; source $C/banks_code/env.sh; unset ANTHROPIC_API_KEY; cd $C
for b in fresh1 fresh2 fresh3 fresh4; do /workspace/agam/chive/.venv/bin/python -u analysis/diag/p4b_exp1.py $b > /workspace/agam/logs/chive_diag/exp1_$b.log 2>&1 & done; wait
