#!/bin/bash
# Regenerate every chart, CSV, olens_shines.{md,json} and README.md (report/ and /workspace/agam/chive_graphs/).
# usage: run_all.sh [--loop MINUTES]   (loop mode re-runs until killed)
cd /workspace/agam/chive_scale/report
PY=/workspace/agam/chive_scale/harness/.venv/bin/python
run() {
  set -a; . /dev/shm/compute.env 2>/dev/null; set +a; export MODAL_ENVIRONMENT=agam; unset ANTHROPIC_API_KEY
  echo "=== $(date -u) run_all"
  $PY a_freeform.py & $PY c_points.py & $PY c_claims.py > /dev/null & wait
  $PY b_budget.py & $PY e_shines.py & $PY f_kcurve.py & FINAL_MIN_DRAWS=10 $PY g_final10.py & FINAL_MIN_DRAWS=10 $PY g_final10_bars.py & wait
  $PY d_readme.py && echo "=== $(date -u) done"
  # Agent Desk dashboard mirror (Agam's convention): PNGs + short README in experiments/chive/
  mkdir -p /workspace/agam/experiments/chive && cp -f /workspace/agam/chive_graphs/*.png /workspace/agam/experiments/chive/ 2>/dev/null
  $PY ../analysis/olens_vs_nla204/pooled.py > /dev/null 2>&1
  $PY dash_readme.py
}
if [ "$1" = "--loop" ]; then while true; do run 2>&1 | grep -v findfont; sleep $(( ${2:-40} * 60 )); done; else run; fi
