#!/bin/bash
# v3 (hard) MCQ: build option sets on dev240, then 3 draws x 5 arms in eval dirs <batch>_mcqh
export PATH=$HOME/.local/bin:$PATH; set -a; . /dev/shm/compute.env; set +a; export MODAL_ENVIRONMENT=agam; unset ANTHROPIC_API_KEY
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh
cd $C/harness
for b in fresh1 fresh2; do
  PYTHONPATH=src .venv/bin/python -u ../analysis/mcq/build_mcq10_hard.py $C/datasets/$b $C/analysis/harness_improve/dev/$b.txt $C/analysis/mcq/devh_$b >> $L/mcq_hard_build.log 2>&1 &
done; wait; echo "MCQH BUILD DONE $(date -u)" >> $L/mcq_hard_build.log
for b in fresh1 fresh2; do
  python3 -c "import json,sys; [print(json.loads(l)['prompt_id']) for l in open(sys.argv[1])]" $C/analysis/mcq/devh_${b}_run.jsonl > $C/analysis/mcq/devh_${b}_ids.txt
  for spec in "transcript" "olens_rl_B=$C/banks/s3d_rl600_all:olens" "mlnla_B=$C/banks/mlnla_k4_chive:mlnla" "jlens_L2060=$C/banks/jlens_L20-60:jlens" "olens_single_k16=$C/banks/s3_single_rl_k16:olens"; do
    tag=${spec%%=*}
    ( for pass in 1 2; do EVAL=${b}_mcqh IDS=$C/analysis/mcq/devh_${b}_ids.txt CONC=5 $C/run_harness.sh $b agent "['$spec']" draws=3 choices=$C/analysis/mcq/devh_${b}_run.jsonl; done ) > $L/mcqh_run_${tag}_$b.log 2>&1 &
  done
done
wait; echo "MCQH RUN DONE $(date -u)" >> $L/mcqh_run_transcript_fresh1.log
