#!/bin/bash
# usage: shuffled_controls.sh [batch ...]: the shuffled-readout controls (each reader served ANOTHER
# incident's readouts) on each batch's control split splits/shuffle_ctrl.txt; analyse with
# analysis/shuffle_ctrl.py.
D=$(cd "$(dirname "$0")" && pwd); source "$D/../env.sh"; source "$D/arms.sh"
A=$(arms_list "$ARM_OLENS_SHUF" "$ARM_JLENS_SHUF" "$ARM_MLNLA_K1_SHUF")
for b in ${*:-fresh1 fresh2 fresh3 fresh4}; do
  IDS=$CHIVE_DATASETS/$b/splits/shuffle_ctrl.txt CONC=${CONC:-20} "$D/run_arm.sh" "$A" "$b" || exit 1
done
