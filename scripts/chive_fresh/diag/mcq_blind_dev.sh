#!/bin/bash
# MCQ v5 (blind-equal, Agam 2026-10-20): build option sets on dev240 (blind picker near chance; transcript allowed to help),
# then 3 draws x 5 arms with the frozen harness in eval dirs <batch>_mcqb. Throttled: builder 4 incidents x 8 LLM calls,
# harness CONC=2 per run.
export PATH=$HOME/.local/bin:$PATH; set -a; . /dev/shm/compute.env; set +a; export MODAL_ENVIRONMENT=agam OMP_NUM_THREADS=1; unset ANTHROPIC_API_KEY
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_diag
cd $C/harness
for b in fresh1 fresh2; do
  PYTHONPATH=src .venv/bin/python -u ../analysis/mcq/build_mcq10_blind.py $C/datasets/$b $C/analysis/harness_improve/dev/$b.txt $C/analysis/mcq/eqb_$b 2 >> $L/mcqb_build_$b.log 2>&1 &
done; wait; echo "MCQB BUILD DONE $(date -u)" >> $L/mcqb_build_fresh1.log
for b in fresh1 fresh2; do
  python3 -c "import json,sys; [print(json.loads(l)['prompt_id']) for l in open(sys.argv[1])]" $C/analysis/mcq/eqb_${b}_run.jsonl > $C/analysis/mcq/eqb_${b}_ids.txt
  for spec in "transcript" "olens_rl_B=$C/banks/s3d_rl600_all:olens" "mlnla_B=$C/banks/mlnla_k4_chive:mlnla" "jlens_L2060=$C/banks/jlens_L20-60:jlens" "olens_single_k16=$C/banks/s3_single_rl_k16:olens"; do
    tag=${spec%%=*}
    ( for pass in 1 2; do EVAL=${b}_mcqb IDS=$C/analysis/mcq/eqb_${b}_ids.txt CONC=2 $C/run_harness.sh $b agent "['$spec']" draws=3 choices=$C/analysis/mcq/eqb_${b}_run.jsonl; done ) > $L/mcqb_run_${tag}_$b.log 2>&1 &
  done
done
wait; echo "MCQB RUN DONE $(date -u)" >> $L/mcqb_build_fresh1.log
