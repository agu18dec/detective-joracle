# CHIVE forensics: pre-registered analyses

Written 2026-10-11 07:00 UTC, **before** any of the data below was looked at for outcomes.
Committed to git so the timestamp is checkable. Everything here is computed by
`scripts/strata_chive.py`; numbers are reported whatever they show.

## Why

The hard-92 stratum was selected because transcript-only Opus failed it on 3 earlier draws.
Selecting on errors inflates any tool's gain in two ways: regression to the mean, and
*perturbation* — anything that pushes the agent off its default answer helps when the default
is wrong (a colleague saw AO/NLA uplift vanish when selecting on uncertainty instead of errors).
E1 results that used that selection, or a cut chosen after seeing E1 (terciles of the single-
probability confidence), are **exploratory**. The analyses below are the confirmatory ones.

## Data not yet looked at (when this was written)

- **E2** (`e2_one_experiment`, one experiment, 5 arms x 300 x 2): looked at only for health
  (record and experiment counts), never for scores.
- **Cross-incident shuffled OLens** (`olens_rl_shuffled`, E1 settings, 300 x 2): the readouts of
  another incident. Known limitation (seen in 2 of the first 3 records): the agent can notice
  the readouts do not fit and fall back on the text, so this controls for "a readout tool plus
  the effort", not for perturbation.
- **Position-shuffled OLens** (`olens_rl_posshuffled`, E1 settings, 300 x 2): the same
  incident's readouts at permuted positions (fixed permutation per incident): on-topic and
  plausible, but no information about where or at which token anything is represented. **This
  is the perturbation control.**
- **Transcript-only uncertainty** (`confidence.json`, `scripts/confidence_chive.py`): Opus 5,
  adaptive thinking at effort=max, no tools, spreads probability over up to five candidate
  causes plus "something else", 3 samples; `top_p` = mean probability of the top candidate.
  Never compared with correctness when chosen.

## Primary (confirmatory) comparisons

Paired by incident (pass@1 averaged over the 2 draws), 95% bootstrap CI; "significant" = the CI
excludes 0.

1. **Information beyond perturbation (E1):** OLens RL − position-shuffled OLens, all 300.
2. **Information at all (E1):** OLens RL − cross-shuffled OLens, all 300.
3. **Uplift without error selection:** OLens RL − transcript on incidents with
   `top_p` in [0.25, 0.75] (the colleague's band applied to the top candidate), in **E2** (unseen)
   and in E1.
4. **E2 headline:** OLens RL − transcript, all 300; and the two-agent concluder version.

## Secondary

- Terciles of `top_p` (cut by rank, fixed before outcomes): every arm vs transcript, and
  OLens RL vs both shuffled controls, per tercile.
- Keep / rescue for every arm, including both controls (selection on one transcript draw,
  scoring the other draw).
- NLA, J-lens and OLens distill on comparisons 3 and 4.

## Reported but flagged

- Hard-92 numbers (selection on transcript errors): always shown next to the primary results,
  labelled selection-biased.
- E1 terciles of the old single-probability confidence (`confidence_single.json`): exploratory;
  the cut was chosen after seeing the 25–75% band.
- Dev-split harness variants (switchbar, textfirst, continue_from): tuned on dev100; any claim
  rests on one frozen run on heldout200.
