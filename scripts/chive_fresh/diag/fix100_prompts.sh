#!/bin/bash
# Agam 2026-10-21: small test of OLens fixes 2+3 on 100 dev incidents x 3 draws, frozen harness (no experiments).
# support = [k/n] counts next to every readout line; rivalrole = use readouts to reject rivals / choose, not rewrite.
C=/workspace/agam/chive_scale; L=/workspace/agam/logs/chive_diag; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for b in fresh1 fresh2; do
  for spec in "olens_sup|['textfirst','absence','support']|$C/banks/s3d_rl600_all:olens" \
              "olens_rr|['textfirst','absence','rivalrole']|$C/banks/s3d_rl600_all:olens" \
              "olens_suprr|['textfirst','absence','support','rivalrole']|$C/banks/s3d_rl600_all:olens" \
              "mlnla_suprr|['textfirst','absence','support','rivalrole']|$C/banks/mlnla_k4_chive:mlnla"; do
    IFS='|' read -r arm V bank <<< "$spec"
    ( for i in 1 2; do IDS=$C/analysis/diag/fix100/$b.txt CONC=2 $C/harness_exp/run_exp.sh $b agent "['$arm=$bank']" "$V"; done && \
      IDS=$C/analysis/diag/fix100/$b.txt CONC=4 $C/harness_exp/run_exp.sh $b judge "['$arm=$bank']" "$V" ) > $L/fix100_${arm}_$b.log 2>&1 &
  done
done; wait; echo "FIX100 PROMPTS DONE $(date -u)" >> $L/fix100_olens_sup_fresh1.log
