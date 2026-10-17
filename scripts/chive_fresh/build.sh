#!/bin/bash
# usage: build.sh <batch>: CHIVE run dir $CHIVE_RUN_DIRS/<batch> -> harness dataset $CHIVE_DATASETS/<batch>
# (build_incidents.py under uv with its pinned transformers==5.11.0; extra args are passed through)
source "$(dirname "$0")/env.sh"
B=$1; shift
exec uv run --script "$CHIVE_FRESH_DIR/build_incidents.py" --run "$CHIVE_RUN_DIRS/$B" --out "$CHIVE_DATASETS/$B" "$@"
