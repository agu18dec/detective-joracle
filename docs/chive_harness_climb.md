# CHIVE: climbing the OLens harness (read-only first)

Goal: the best OLens harness we can build, relative to fixed baselines, chosen on a dev set and
tested once on held-out incidents. Started 2026-10-11 08:00 UTC.

## Protocol

- **Splits** (`data/chive/splits/`, fixed by hash before any round, commit 3381934):
  - `climb_dev96.txt`: 46 hard + 50 normal;
  - `climb_test204.txt`: the other 46 hard + 158 normal.
- **Rounds** run on dev only: 96 incidents × 2 draws, read-only, one change at a time. The
  dev baselines are E1's records for the same incidents (transcript, OLens RL, NLA, J-lens),
  plus transcript + continue_from run fresh.
- **Accept rule:** a change stays if dev pass@1 rises and *keep* does not fall. Keep is pass@1
  on incidents a transcript draw solved, scored on the other draw.
- **Final:** freeze the best harness and run it once on test with 5 draws, next to fixed
  baselines with 5 fresh draws each:
  - transcript;
  - transcript + continue_from;
  - NLA and J-lens;
  - OLens with the original E1 harness;
  - NLA and J-lens under the final harness's generic (non-OLens) parts.
  - Report pass@k for k = 1..5 (unbiased estimator from 5 draws), paired CIs, hard vs normal.
- **Leakage:** my E1 failure analysis deep-read 24 incidents, and 19 of them fall in test
  (listed in the split commit's session log). Test is reported with and without them.
- **Baseline to beat:** transcript + continue_from scored 72% vs transcript's 62% on dev100
  (+10.0 [+3.5, +17.0]), and OLens + continue_from did not beat it (−4.5 [−12.0, +3.0]).
  Beating plain transcript is not enough.

## Levers (simple first)

1. textfirst + absence guidance
2. continue_from inside the OLens harness
3. J-lens corroboration tool
4. agreement across k parallel investigations (concept vote / arbiter; cost-matched against pass@k)
5. readout digest up front
6. `<im_end>` boundary reads: bank for dev building on Modal (`s3d_rl600_boundary_dev96`,
   pre-response positions + 24 response tokens)
7. FVE corroboration (`dense_fve_modal.py` sidecars)
8. budgeted counterfactual: one experiment chosen from tool evidence (separate track)

## Rounds

| round | change | dev pass@1 | keep | rescue | hard | normal | decision |
|---|---|---|---|---|---|---|---|
| baseline | E1 OLens RL harness (dev records from E1) | 57.8% | 80.0% | 33.7% | 38.0% | 76.0% | reference |
| r1a | + textfirst + absence (= base B) | 62.5% (+4.7 [-2.6, +12.0] vs E1 OLens; +10.4 [+2.1, +18.2] vs transcript) | 87.0% | 35.9% | 35.9% | 87.0% | **keep**: keep up 7 pts, rescue held, unsupported quotes 1.9% -> 1.5% |
| r1b | + textfirst + absence + continue_from | | | | | | |
