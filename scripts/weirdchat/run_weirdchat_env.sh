#!/bin/bash
# Launch the WeirdChat GPU jobs against the servers deployed in Modal env `weirdchat`
# (see docs/weirdchat.md; endpoints differ from main's because of the env suffix).
# Usage: scripts/weirdchat/run_weirdchat_env.sh interventions|jlens|nla
# The OpenRouter key is passed to tmux with -e, never on a command line.
set -euo pipefail
cd "$(dirname "$0")/../.."
LENS='https://need-c10-a-camila-agam-weirdchat--org-organism-'
NLA='https://need-c10-a-camila-agam-weirdchat--nla-nla-verbalize.modal.run'
VLLM='https://need-c10-a-camila-agam-weirdchat--ab-vllm-base-serve.modal.run'
PY=/workspace/agam/global-workspace/.venv/bin/python
TS=$(date +%Y-%m-%d_%H%M)
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set in the calling shell}"
case "${1:?job}" in
  interventions)
    # the key goes through tmux's environment table, never onto a command line
    tmux set-environment -g OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
    tmux new -d -s wc-interv \
      "PYTHONUNBUFFERED=1 $PY scripts/weirdchat/run_interventions.py stage=run add=true \
       patterns=denying-ai-identity__pg0001 server=$LENS target=$VLLM n=64 max_new=600 \
       judge_model=google/gemma-4-31b-it 2>&1 | tee logs/interventions_framing_$TS.log";;
  jlens)
    tmux new -d -s wc-diag-jlens \
      "PYTHONUNBUFFERED=1 $PY scripts/weirdchat/run_weirdchat.py stage=diagnose diag_lens=jlens \
       server=$LENS workers=4 2>&1 | tee logs/diag_jlens_$TS.log";;
  nla)
    tmux new -d -s wc-diag-nla \
      "PYTHONUNBUFFERED=1 $PY scripts/weirdchat/run_weirdchat.py stage=diagnose diag_lens=nla \
       server=$LENS nla=$NLA workers=2 2>&1 | tee logs/diag_nla_$TS.log";;
  *) echo "unknown job $1"; exit 2;;
esac
echo "started $1 — tail -f logs/*_$TS.log"
