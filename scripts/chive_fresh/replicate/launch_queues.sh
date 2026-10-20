#!/bin/bash
# (re)launch the three replicate harness queues (resumable) + the confidence top-up
R=/workspace/agam/chive_scale/replicate_runs/scripts; L=/workspace/agam/logs/chive_fresh; O=/workspace/agam/chive_scale/replicate_banks/orig; RB=/workspace/agam/chive_scale/replicate_banks/rebuilt
tmux new -d -s rep-s1 "$R/queue.sh $L/rep_s1.log 'rep_transcript|transcript' 'rep_nla_orig|nla_B=$O/nla_k4:nla' 'rep_olens_rebuilt|olens_rl_B=$RB/rep_s3d_rl600_all:olens|rep_s3d_rl600_all' 'rep_nla_rebuilt|nla_B=$RB/rep_nla_k4:nla|rep_nla_k4'"
tmux new -d -s rep-s2 "$R/queue.sh $L/rep_s2.log 'rep_olens_orig|olens_rl_B=$O/s3d_rl600_all:olens' 'rep_jlens_orig|jlens_B=$O/jlens_all:jlens' 'rep_mlnla_rebuilt_k1|mlnla_k1=$RB/rep_mlnla_k1:mlnla|rep_mlnla_k1'"
tmux new -d -s rep-s3 "$R/queue.sh $L/rep_s3.log 'rep_mlnla_orig|mlnla_B=$O/mlnla_k4_chive:mlnla' 'rep_mlnla_rebuilt|mlnla_B=$RB/rep_mlnla_k4_chive:mlnla|rep_mlnla_k4_chive' 'rep_jlens_rebuilt|jlens_B=$RB/rep_jlens_all:jlens|rep_jlens_all'"
tmux new -d -s rep-conf "$R/confidence.sh 2>&1 | tee -a $L/rep_confidence.log"
