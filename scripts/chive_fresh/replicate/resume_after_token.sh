#!/bin/bash
source /workspace/agam/chive_scale/banks_code/env.sh
cd /tmp && /workspace/agam/chive_scale/harness/.venv/bin/python -I /workspace/agam/chive_scale/wait_token2.py 2>&1 | grep -v -i warn
/workspace/agam/chive_scale/replicate_runs/scripts/launch_queues.sh
echo "$(date +%T) relaunched queues"
