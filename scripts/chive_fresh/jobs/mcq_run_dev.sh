#!/bin/bash
# MCQ dev run: waits for the option sets, then 3 draws x 5 arms on the sets that pass the leak check, then judge-free scoring.
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh
until grep -q "MCQ BUILD DONE" $L/mcq_build.log 2>/dev/null; do sleep 120; done
for b in fresh1 fresh2; do
  python3 -c "import json,sys; [print(json.loads(l)['prompt_id']) for l in open(sys.argv[1])]" $C/analysis/mcq/dev_${b}_run.jsonl > $C/analysis/mcq/dev_${b}_ids.txt
  for spec in "transcript" "olens_rl_B=$C/banks/s3d_rl600_all:olens" "mlnla_B=$C/banks/mlnla_k4_chive:mlnla" "jlens_L2060=$C/banks/jlens_L20-60:jlens" "olens_single_k16=$C/banks/s3_single_rl_k16:olens"; do
    tag=${spec%%=*}
    ( for pass in 1 2; do EVAL=${b}_mcq IDS=$C/analysis/mcq/dev_${b}_ids.txt CONC=5 $C/run_harness.sh $b agent "['$spec']" draws=3 choices=$C/analysis/mcq/dev_${b}_run.jsonl; done ) > $L/mcq_run_${tag}_$b.log 2>&1 &
  done
done
wait; echo "MCQ RUN DONE $(date -u)" >> $L/mcq_run_transcript_fresh1.log
