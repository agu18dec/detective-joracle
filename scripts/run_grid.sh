#!/usr/bin/env bash
# A full grid, N seeds: every organism block x every arm x every presentation, run as one queue
# so the servers stay warm across cells.
#
#   OPENROUTER_API_KEY=... JORACLE_SERVER=... JORACLE_TARGET=... scripts/run_grid.sh [seeds] [auditor]
#
# Edit the blocks for your organisms. Order matters: put the organisms with headroom (those whose
# quirk does not leak into the reply) first, and run the unfiltered lens arms before their
# llm-filtered twins so the control lands first.
set -uo pipefail
cd "$(dirname "$0")/.."
SEEDS="${1:-3}"
AUD="${2:-google/gemini-3.8-flash}"
L=scripts/run_loop.sh
BLOCK1='base,secret_loyalty_td_r16,flattery_td_r16,hardcode_test_cases_td_r16'
BLOCK2='flattery_sdf_r16,hardcode_test_cases_sdf_r16,secret_loyalty_sdf_r16,contextual_optimism_sdf_r16'
for ORGS in "$BLOCK1" "$BLOCK2"; do
  echo "##### $(date -u +%H:%M) block: $ORGS"
  $L "$ORGS" 'blackbox,olens,jlens' "$SEEDS" "auditors=$AUD"
  $L "$ORGS" 'olens,jlens'          "$SEEDS" "auditors=$AUD select=llm"
done
echo "GRID_DONE $(date -u +%H:%M)"
