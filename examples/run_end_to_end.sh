#!/usr/bin/env bash
# The whole pipeline against the mock target, locally: agent -> judge -> judges -> report -> viewer.
#
#   examples/run_end_to_end.sh                 # scripted auditor + mock judge (no keys, no GPU)
#   OPENROUTER_API_KEY=sk-or-... examples/run_end_to_end.sh   # a real auditor and judge
#
# Output: examples/out/runs/live/.../seed_0.json, examples/out/report.md, examples/out/site/index.html
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}"
PORT="${PORT:-8765}"
OUT="${OUT:-examples/out}"
export PYTHONUNBUFFERED=1

# one mock process serves the target (as the flattery organism), the lens and the judge
"$PY" examples/mock_target.py --port "$PORT" --tag flattery_td_r16 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 0.2
done

COMMON=(server="http://127.0.0.1:$PORT/{name}" target="http://127.0.0.1:$PORT"
        organisms=flattery_td_r16,base arms=blackbox,olens seeds=1 out_root="$OUT" workers=2)

if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  AUD="${AUDITOR:-google/gemini-3.8-flash}"
  echo "=== real auditor ($AUD) + judge via OpenRouter"
  "$PY" scripts/run_audit.py stage=agent backend=openrouter "auditors=$AUD" budget_tokens=4000 \
    "${COMMON[@]}"
  "$PY" scripts/run_audit.py stage=judge,judges,report "${COMMON[@]}"
else
  AUD=fake/scripted
  echo "=== scripted auditor (FakeBackend) + mock judge; scores are placeholders"
  "$PY" scripts/run_audit.py stage=agent backend=fake "auditors=$AUD" "${COMMON[@]}"
  OPENAI_API_KEY=mock OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" \
    "$PY" scripts/run_audit.py stage=judge,judges,report "${COMMON[@]}"
fi

"$PY" scripts/build_viewer.py out_root="$OUT"
if "$PY" -c "import matplotlib" 2>/dev/null; then
  "$PY" scripts/plot.py out_root="$OUT" "auditor=$AUD" || true
fi
echo "E2E_DONE: open $OUT/site/index.html (python -m http.server from $OUT/site)"
