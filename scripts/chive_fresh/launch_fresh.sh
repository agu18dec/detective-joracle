#!/bin/bash
# usage: launch_fresh.sh <run_name>
# Run the unmodified CHIVE pipeline (screen -> investigate -> verify ...) on $CHIVE_RUN_DIRS/<run_name>/config.json
# with keyless Claude and the runtime patches in run_chive_keyless.py, under the CHIVE checkout's venv.
source "$(dirname "$0")/env.sh"
cd "$CHIVE_DIR" || exit 1
PYTHONPATH=$CHIVE_DIR exec "$CHIVE_PY" -u "$CHIVE_FRESH_DIR/run_chive_keyless.py" chive.pipeline.run_pipeline \
  --config "$CHIVE_RUN_DIRS/$1/config.json"
