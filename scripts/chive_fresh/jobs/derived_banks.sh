#!/bin/bash
# usage: derived_banks.sh: the banks derived locally from full banks (no GPU):
#   olens_k1 / mlnla_k1  = first sample of every cell of s3d_rl600_all / mlnla_k4_chive (make_k1.py)
#   jlens_L20-60         = jlens_all restricted to layers 20..60 step 4 (make_layers.py)
D=$(cd "$(dirname "$0")" && pwd); source "$D/../env.sh"
A=$CHIVE_FRESH_DIR/analysis
"$HARNESS_PY" "$A/make_k1.py" "$CHIVE_BANKS/s3d_rl600_all" "$CHIVE_BANKS/olens_k1"
"$HARNESS_PY" "$A/make_k1.py" "$CHIVE_BANKS/mlnla_k4_chive" "$CHIVE_BANKS/mlnla_k1"
"$HARNESS_PY" "$A/make_layers.py" "$CHIVE_BANKS/jlens_all" "$CHIVE_BANKS/jlens_L20-60" --layers 20:61:4
