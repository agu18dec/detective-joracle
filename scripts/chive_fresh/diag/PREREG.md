# Held-out test, frozen 2026-10-20 21:40 PT (before any held-out run)

Selection was made on dev240 (fresh1/2 dev splits), 3 draws:
- single-bullet OLens ×16 + gate (echotag + revgate) beat frozen ×16 by +3.6 pp [+0.2, +6.9], and NLA + gate by +2.5 [−0.4, +5.7].
  It was the best of 3 gate arms, so the dev number is optimistic.
- Budget-3 "experiments decide" (rexp) beat budget 1 by +5-6 pp for every arm (interim, ~60% of draws).

Held-out: 110 random incidents (seed 0) each from fresh3 and fresh4 (analysis/diag/heldout/{fresh3,fresh4}.txt), 3 draws.
Arms:
  A (budget 0)  olens_s16_gate   = single-bullet OLens ×16, variants textfirst+absence+echotag+revgate
                mlnla_gate_h     = NLA ×4, same variants
                compared with the existing frozen records on the same incidents/draws: olens_rl_B, mlnla_B, olens_single_k16, transcript
  B (budget 3)  olens_s16_gx3    = single-bullet OLens ×16, textfirst+absence+echotag+revgate+rexp, 3 experiments, n=30
                mlnla_gx3        = NLA ×4, same
                transcript_rx3h  = transcript, rexp, 3 experiments
Primary test: olens_s16_gate − mlnla_gate_h (A) and olens_s16_gx3 − mlnla_gx3 (B), paired incident bootstrap, pass = judge ≥ 7.
Secondary: each vs its frozen counterpart, and readers vs transcript at budget 3.
Arm B's combined variant (gate + rexp) was NOT run on dev; it is exploratory.
