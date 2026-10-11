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
| r1b | B + continue_from (first guide) | 56.2% (-6.2 [-12.0, -0.5] vs B) | 84.0% | 26.1% | 28.3% | 82.0% | **drop**: hurts; rescue 36 -> 26 |
| r1c | B + J-lens corroboration tool | 59.9% (-2.6 [-9.4, +4.2] vs B) | 89.0% | 28.3% | 35.9% | 82.0% | **drop**: no gain |
| ref | transcript + continue_from (dev96) | 55.2% (+3.1 vs transcript, n.s.; +10.0 on dev100) | 85.0% | 22.8% | 27.2% | 81.0% | baseline: effect varies by split |
| r1d | B + readout digest | 63.0% (+0.5 [-4.7, +6.2] vs B) | 91.0% | 32.6% | 39.1% | 85.0% | **drop**: no gain (keep up, rescue down); unsupported 1.1% |

| r2a | B + continue_from (guide: use inside the response, at the behaviour) | | | | | | |
| r2b | B + 1 experiment (budgeted counterfactual) | | | | | | |

**Agreement (offline arbiter over existing dev records):** B draw0 vs draw1 -> 61.5% (inputs
59.4 / 65.6); transcript vs B -> 58.9% (picks B 79%); transcript+continue vs B -> 56.2% (picks
B only 26%; -6.2 [-12.5, -0.5] vs B). **Drop**: B subsumes the arbiter's E1 gain (which came
from repairing abandoned text answers), and continuation evidence persuades the arbiter wrongly.

**Round 1 reading:** no tool adds on top of B; the gain is the reasoning protocol (commit a
transcript-only hypothesis first, overturn only on contrary evidence, absences are invisible to
readouts). Extra tools spread attention; continue_from pulled agents off correct readout
evidence (rescue down) -- partly misuse (sampling inside user turns returns nothing).

Separately (dev100, `dev_absence`): absence guidance alone, per reader -- OLens RL 65.0 -> 70.0
(+5.0 [-2.0, +12.0]), NLA 62.0 -> 67.5, J-lens 61.5 -> 68.5 (+7.0 [+1.5, +13.0]); the gain is on
absence-type causes (OLens 45.8 -> 64.6, n=24) with present-feature causes flat (71.1 -> 71.7,
n=76). A generic fix: it goes into the "generic parts" applied to every reader at test.
