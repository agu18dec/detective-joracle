#!/usr/bin/env bash
# The whole WeirdChat explain pipeline in one tmux session, unbuffered, to a timestamped log.
#
#   OPENROUTER_API_KEY=sk-or-… scripts/weirdchat/run_weirdchat.sh \
#       'https://<ws>--auditbench-organism-organism-' [auditor] [per_behavior]
#
# Stages: data (no GPU) -> diagnose (lens reads) -> agent -> synth -> site. Every stage resumes,
# so re-running after a failure costs only the work that is missing.
set -euo pipefail

SERVER="${1:?usage: run_weirdchat.sh <lens server prefix> [auditor] [per_behavior]}"
AUDITOR="${2:-anthropic/claude-opus-5}"
PER_BEHAVIOR="${3:-3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
SESSION="${SESSION:-weirdchat}"
STAMP="$(date +%Y-%m-%d_%H%M)"
LOG="$REPO/logs/weirdchat_$STAMP.log"
mkdir -p "$REPO/logs"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is not set (pass it per command, never export it)" >&2
  exit 1
fi

CMD="cd $REPO && PYTHONUNBUFFERED=1 PYTHONPATH=src OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  $PY scripts/weirdchat/run_weirdchat.py stage=data,diagnose,agent,synth,site \
  server='$SERVER' auditor='$AUDITOR' per_behavior=$PER_BEHAVIOR \"\$@\" 2>&1 | tee -a $LOG"

tmux new-session -d -s "$SESSION" "bash -lc \"$CMD\""
echo "tmux session '$SESSION' started"
echo "monitor: tail -f $LOG"
