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

## Results (computed 2026-10-11 11:35 UTC by `scripts/strata_chive.py`)

Uncertainty measure: `top_p` quartiles 0.31 / 0.38 / 0.45 over 293 incidents (7 had no usable
elicitation after retries). **The primary band [0.25, 0.75] is degenerate**: it holds 288 of 293
incidents, so it equals "all". Reported as specified; the terciles (secondary) carry the
information. Health: every arm 0 errors / 0 unjudged (E2 concluder: 24 episodes ran no
experiment and are left out).

| comparison (paired, 95% CI) | all 300 | band (288) | tercile 1 (least confident) | hard 92 (flagged) |
|---|---|---|---|---|
| 1. OLens RL − position-shuffled (E1) | +1.3 [−2.5, +5.2] | +1.7 [−2.1, +5.7] | +4.1 [−2.6, +11.3] | +1.6 [−5.4, +8.7] |
| 2. OLens RL − cross-shuffled (E1) | **+7.5 [+3.2, +11.8]** | +7.6 [+3.1, +12.2] | +15.5 [+7.2, +23.7] | +17.4 [+8.7, +26.1] |
| 3. OLens RL − transcript (E1) | +3.0 [−1.7, +7.8] | +3.1 [−1.7, +8.0] | **+10.3 [+1.5, +19.1]** | +17.4 [+7.1, +27.2] |
| 3. OLens RL − transcript (E2) | −3.2 [−7.0, +0.7] | −3.3 [−7.3, +0.5] | −2.6 [−9.3, +3.6] | −0.5 [−7.6, +6.5] |
| 4. E2 concluder: OLens RL − transcript | −1.2 [−4.9, +2.5] | | | +1.1 [−4.9, +7.6] |

pass@1 (E1, all / hard 92): OLens RL 66.0 / 40.8, position-shuffled 64.7 / 39.1, transcript
63.0 / 23.4, cross-shuffled 58.5 / 23.4.

**Reading.** (a) Readout content matters: another incident's readouts make the agent worse
than transcript alone. (b) Location barely matters: the same incident's readouts at permuted
positions recover nearly all of OLens's gain, overall and on the hard 92. OLens helps as a
summary of what the model represents about this conversation, not by localizing it. (c) Without
selecting on errors, OLens's gain survives in the least-confident tercile (+10.3); NLA and
J-lens do not (+1.5, −1.5). (d) With one experiment (E2), no reader adds over transcript, and
the concluder test finds no better-chosen experiments.
