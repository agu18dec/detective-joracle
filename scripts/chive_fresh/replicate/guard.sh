#!/bin/bash
# watch the replicate queues: on fresh 401s, stop the queues and wait for a working token, then relaunch.
# exits when all three queues printed QUEUE DONE (after the last relaunch).
L=/workspace/agam/logs/chive_fresh; R=/workspace/agam/chive_scale/replicate_runs/scripts
while true; do
  n=$(for s in 1 2 3; do tail -n 200 $L/rep_s$s.log | grep -c "AuthenticationError"; done | awk "{s+=\$1} END {print s+0}")
  if [ "$n" -ge 10 ]; then
    echo "$(date +%T) $n auth errors: pausing queues" >> $L/rep_guard.log
    for s in rep-s1 rep-s2 rep-s3 rep-conf; do tmux kill-session -t $s 2>/dev/null; done
    pkill -f "replicate_runs.*run_chive.py" ; sleep 5
    for s in 1 2 3; do echo "$(date +%T) guard pause" >> $L/rep_s$s.log; done
    source /workspace/agam/chive_scale/banks_code/env.sh
    (cd /tmp && /workspace/agam/chive_scale/harness/.venv/bin/python -I /workspace/agam/chive_scale/wait_token2.py 2>&1 | grep -iv warn >> $L/rep_guard.log)
    $R/launch_queues.sh; echo "$(date +%T) relaunched" >> $L/rep_guard.log
  fi
  d=0; for s in 1 2 3; do tail -n 3 $L/rep_s$s.log | grep -q "QUEUE DONE" && d=$((d+1)); done
  [ $d -eq 3 ] && { echo "$(date +%T) all queues done" >> $L/rep_guard.log; exit 0; }
  sleep 120
done
