# shellcheck shell=bash
# Source me (after ../env.sh): the harness arm specs of the fresh scale-up, NAME=BANK_DIR:KIND[:shuffled]
# (scripts/run_chive.py). Banks live in $CHIVE_BANKS; docs/chive_fresh_scaleup.md describes each arm.
BK=$CHIVE_BANKS
ARM_TRANSCRIPT='transcript'
ARM_OLENS="olens_rl_B=$BK/s3d_rl600_all:olens"            # OLens RL s3d_rl600, k=4, L20-60 step 4
ARM_OLENS_K1="olens_k1=$BK/olens_k1:olens"                # first sample per cell (analysis/make_k1.py)
ARM_OLENS_SINGLE="olens_single=$BK/s3_single_rl:olens"     # single-readout OLens (no distill), k=4
ARM_JLENS="jlens_B=$BK/jlens_all:jlens"                   # J-lens top-10, all 63 layers
ARM_JLENS_L2060="jlens_L2060=$BK/jlens_L20-60:jlens"      # J-lens at the OLens ladder (analysis/make_layers.py)
ARM_MLNLA="mlnla_B=$BK/mlnla_k4_chive:mlnla"              # multi-layer NLA, k=4
ARM_MLNLA_K1="mlnla_k1=$BK/mlnla_k1:mlnla"                # multi-layer NLA, first sample per cell
ARM_OLENS_SHUF="olens_shufc=$BK/s3d_rl600_all:olens:shuffled"   # another incident's readouts
ARM_JLENS_SHUF="jlens_shufc=$BK/jlens_all:jlens:shuffled"
ARM_MLNLA_K1_SHUF="mlnla_k1_shufc=$BK/mlnla_k1:mlnla:shuffled"
arms_list() {  # arms_list SPEC [SPEC ...] -> the python list run_harness.sh expects
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
}
