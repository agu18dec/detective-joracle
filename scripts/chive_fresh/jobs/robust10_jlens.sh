#!/bin/bash
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_fresh
for b in fresh1 fresh2 fresh3 fresh4; do
  ( IDS=$C/datasets/$b/splits/all.txt CONC=12 $C/run_harness.sh $b agent "['jlens_L2060=$C/banks/jlens_L20-60:jlens']" draws=10 && \
    IDS=$C/datasets/$b/splits/all.txt CONC=12 $C/run_harness.sh $b judge "['jlens_L2060=$C/banks/jlens_L20-60:jlens']" draws=10 ) >> $L/robust10_jlens2_$b.log 2>&1 &
done; wait; echo "ROBUST10 JLENS DONE $(date -u)" >> $L/robust10_jlens2_fresh1.log
