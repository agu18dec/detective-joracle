# CHIVE diagnosis: why OLens doesn't beat multi-layer NLA, and what helps

Session `chive_regressions`, 2026-10-20/21 PT. Code is in `/workspace/agam/chive_scale/analysis/diag/`, tables in `data/`. Builds on the finished chive run (`/workspace/agam/experiments/chive/SUMMARY.md`), which used 915 fresh CHIVE incidents and the frozen detective-joracle harness. That run found OLens ×4 68.3 vs NLA ×4 69.8 pass@1.

## TL;DR
1. **The gap comes from reading, not from the bank or the agent's starting point.** Before readouts, the agents' committed transcript-only hypotheses pass equally often: OLens 62.6 vs NLA 63.1 (n = 4,570 episodes per arm). After reading, OLens has more pass→fail flips (8.5% vs 7.4%) and the same fail→pass rate (14.2% vs 14.3%).
   - Pass→fail causes: dilution 47% (the right cause is kept but buried under readout side-stories), lure 36-39% (vivid readouts confirm a refuted rival), re-focus 10%.
   - Same mix for both readers; OLens just has more vivid, specific content, so more lures.
2. **Readouts filter; they don't discover.** Judge v3 grades each claim in the final hypothesis:
   - Precision rises only because refuted rivals fall: +3.0 pp OLens, +3.7 NLA; rivals −7.8 pp.
   - Recall of true points is flat (+0.2 pp all points, +0.6 non-obvious).
   - Unsupported "internal" specifics rise: +7 pp from committed to final.
   - OLens = NLA on every claim metric.
3. **CHIVE is a selection problem.** A no-agent Opus generator reading only the transcript names a true cause in its top 5 for 96.7% of incidents. Adding OLens or NLA decision-window readouts changes nothing.
   - The bank ceiling is low once audited by hand: OLens ~12% [5, 26], NLA ~8% [2, 25] of statable non-obvious points. About 5% for points that are both genuine and non-obvious.
   - The ~75% of CHIVE "non-obvious" points that are experiment outcomes can't be stated by any reader.
4. **Experiments are the lever.** Three counterfactual experiments (n=30 each) with an "experiments decide" prompt raise dev pass@1 by +9-14 pp for every arm. Readers add only +1.7-3.3 pp on top.
5. **No harness change makes OLens beat NLA on held-out.** A revision gate with echo/preview tags gained +3.6 pp on dev (single-bullet OLens ×16). On 220 held-out incidents it was −0.6 [−3.5, +2.3] vs frozen. OLens ×16 + gate vs NLA + gate: +3.0 [−0.3, +6.5] with no experiments, +0.5 [−2.7, +3.6] with 3 experiments.

## 1. Claim-level precision / recall / hallucination (judge v3)
`judge3.py` (arm-blind):
1. The final hypothesis is split into ≤8 causal claims, with readout/tool talk stripped.
2. An Opus grader labels each claim against CHIVE's ground truth (judge reference, true points, refuted rivals): TRUE(Pi), RIVAL(Rj), CONTRADICTED, UNSUPPORTED or VAGUE, plus NOT-claims.

n = 911 incidents × 5 draws.

| metric (final hypothesis) | transcript | OLens ×4 | NLA ×4 | J-lens L0-62 | OLens − NLA |
|---|---|---|---|---|---|
| precision (TRUE / specific) | 54.4 | 57.3 | 58.1 | 56.9 | −0.8 [−1.8, +0.3] |
| wrong (rival + contradicted) | 26.2 | 21.9 | 21.5 | 20.6 | +0.4 [−0.6, +1.4] |
| unsupported specifics | 19.4 | 20.8 | 20.4 | 22.5 | +0.4 [−0.5, +1.3] |
| recall, all points | 23.6 | 23.8 | 24.2 | 23.6 | −0.4 [−0.9, +0.0] |
| recall, non-obvious points | 6.7 | 7.4 | 7.4 | 7.2 | +0.0 [−0.4, +0.5] |
| refuted rivals asserted | 34.4 | 26.6 | 26.5 | 23.4 | +0.1 [−1.6, +1.8] |

Committed → final for the same episode, OLens (NLA similar):
- refuted rivals −14.9 pp;
- unsupported +7.2 pp;
- +0.47 specific claims per answer;
- non-obvious recall +0.9 pp.

Point-eval v1 grades (already on disk) agree: precision +4 pp for every reader. Readers move ~17% of claims into "the model internally…" claims that the ground truth can't check, and lose 0.4 supported claims per answer.

Figure: `judge3_claims.png`.

## 2. Where OLens regresses (transcripts)
**Flip decomposition.** `p3a_flips.py` re-judges every committed hypothesis with the same 1-10 judge. Figure: `flips_committed_vs_final.png`.

**15 hand-read regression cards.** They cover the incidents where transcript beats OLens by ≥5 of 10 draws, and where NLA beats OLens. Cards: `/workspace/agam/chive_scale/analysis/diag/regressions/`; viewer picks are staged.
- In every case, OLens agents first committed the right or near-right cause from the transcript.
- The readouts that moved them off it are rare (1-20% of lines). They are usually echoes of the prompt, or previews of the response's own wording ("I'll assume…", "this is an error message").
- Agents overstated their counts 2-10× ("every sample" for 2/180 lines).
- Primary mechanisms (n=15): lure 6, argued away 3, re-focus 3, misread 2, over-hedge 1.
- NLA's bland genre/register lines rarely lure.

**Search behaviour** (`p3b_search.py`):
- 72% of OLens and NLA search patterns reuse words from the agent's own committed hypothesis, i.e. confirmatory search.
- No tool-usage metric separates passing from failing episodes.

## 3. Is the information there? (bank ceiling)
- **v1** (`p2_ceiling.py`, TF-IDF retrieval plus a Sonnet judge): the raw result was OLens 64.5% vs NLA 47.0% of non-obvious points. A hand audit (`ceiling_audit.md`, n=55) found only ~13% of OLens hits and 0% of NLA hits were real. Most "hits" were the response being written, keyword matches, or paraphrases of visible instructions.
- **v2** (`p2b_ceiling2.py`): statable mechanisms only, prompt side only, and a strict per-line Opus judge, on 320 incidents. The raw result was OLens 88% vs NLA 82% (shuffled 16% / 8%). The second audit (`ceiling2_audit.md`, n=50) found strict precision of ~13% / 10%, giving a **corrected ceiling of OLens ~12% [5, 26] vs NLA ~8% [2, 25]**.
  - Only 2/30 audited OLens pairs were both valid and genuinely non-obvious. One is prompt_027128: "Vulkan" appears in 60/180 lines at t677, but is first visible at t799. The other is prompt_009645 (camera failure).
- **Lessons:** automated LLM judges of "the bank states X" are unreliable (~10% line precision), and shuffled banks are a weak floor. Fixes for any future version: judge pos < response_start, show the judge the visible tokens around each line, recompute obviousness after rewriting, and use same-incident sibling-point controls.

## 4. Evals: are they the problem?
- **Open-ended hypothesis generation** (`p4a_hypgen.py`, no agent loop, n ≈ 910):

  | | transcript | + OLens readouts | + NLA readouts |
  |---|---|---|---|
  | top-1 true | 70.0 | 69.4 | 68.9 |
  | any of top-5 true | 96.7 | 96.7 | 96.5 |
  | top-1 = primary cause | 55.2 | 53.4 | 53.6 |

  Non-obvious recall is 5.9 for all three. Figure: `hypgen_top5.png`. Generation is saturated. The headroom is in ranking and selection, which pass@1 tests and readouts barely help.
- **Budget-1 counterfactual** (`p4b_exp1.py`, n=906):
  - 77% of transcript agents already spend their one experiment on a true cause; readers don't change the target (+0.9 pp OLens).
  - In 25% of episodes the agent keeps a mechanism its own experiment didn't move.
  - The eval is saturated on experiment choice.
- **Budget-3** ("experiments decide", `rexp` variant, dev240 × 2 draws; pass table in §5). This is where readers measurably help: they **propose better experiments** (`p4b_rx3.py`, Sonnet labels each experiment's target against the ground truth).

  | per episode | transcript | OLens ×4 | NLA ×4 | OLens − tr | NLA − tr |
  |---|---|---|---|---|---|
  | alternative experiments (2-3) on a true cause | 51.4 | 57.1 | 56.7 | +5.8 [+1.9, +9.6] | +5.2 [+1.4, +8.9] |
  | alternative experiments on a refuted rival | 16.7 | 14.1 | 15.1 | −2.6 [−5.4, +0.1] | −1.7 [−4.5, +1.0] |
  | ≥1 true-cause experiment that moved ≥30 pp | 67.6 | 73.1 | 74.3 | +5.5 [+1.5, +9.2] | +6.7 [+3.6, +9.8] |
  | pass@1 | 80.5 | 83.6 | 81.6 | +3.2 [−1.1, +7.1] | +1.3 [−2.5, +5.0] |

  OLens ≈ NLA on every row. Proposal quality is a sharper reader metric than pass@1, because it is measured per experiment and isn't capped by the transcript-only hypothesis already being right.
- **MCQ with 10 equally likely options** (Agam's request): a pilot (`analysis/mcq/build_mcq10_eq.py`) could make only 1/8 incidents equal-likelihood for a text-only picker, even after 3 adversarial rounds; that picker still finds the truth 67% of the time, at ~720 calls/incident. This is another sign that CHIVE truths are mostly readable from the transcript. Paused pending Agam's call.
- **Point-eval caveat:** ~75-80% of CHIVE "non-obvious" points are intervention outcomes ("removing X eliminates Y"). That caps non-obvious recall near 7% for every arm, so the metric can't show reader gains.

## 5. Harness changes (dev240 = fresh1/2 dev splits; pass = judge ≥ 7; paired)
Both new variants are in `harness_exp`:
- **echotag:** readout lines are tagged [ECHO] or [PREVIEW] by n-gram overlap with the visible prompt or the upcoming response.
- **revgate:** if the final hypothesis departs from the committed one, the agent must cite readouts. The harness recounts them in the bank, and an Opus critic (no ground truth) gets one veto.
- Offline check: it rejects the prompt_020832 lure (41/180 matching lines, 25 of them previews) and accepts the real prompt_010088 win (143/183, 11 layers).

| arm | pass | vs baseline |
|---|---|---|
| OLens ×4 + gate | 72.4 | +0.0 [−3.2, +3.2] vs frozen OLens ×4 |
| NLA ×4 + gate | 71.4 | −0.6 [−4.3, +3.2] vs frozen NLA |
| single-bullet OLens ×16 + gate | 74.0 | +3.6 [+0.2, +6.9] vs frozen ×16; +2.5 [−0.4, +5.7] vs NLA + gate |
| transcript, 3 experiments | 80.0 | +13.8 vs 0 exp; +5.2 vs 1 exp |
| OLens ×4, 3 experiments | 83.3 | +10.0 / +6.0 |
| NLA ×4, 3 experiments | 81.7 | +8.7 / +5.8 |

Earlier harness_exp variants, all null or negative: recurrence counts, LLM filter, multi-candidate, antilure.

## 6. Held-out test (frozen picks: `/workspace/agam/chive_scale/analysis/diag/heldout/PREREG.md`)
220 held-out incidents (110 random from each of fresh3/4, seed 0) × 3 draws. Picks were frozen before any held-out run. Paired incident bootstrap. Table: `data/p6_heldout.csv`.

| comparison | pass@1 | Δ [95% CI] |
|---|---|---|
| **Primary A**: single-bullet OLens ×16 + gate vs NLA ×4 + gate (0 experiments) | 69.4 vs 66.4 | +3.0 [−0.3, +6.5] |
| **Primary B**: same arms, 3 experiments ("experiments decide") | 80.5 vs 80.0 | +0.5 [−2.7, +3.6] |
| gate vs frozen harness, OLens ×16 | 69.4 vs 70.0 | −0.6 [−3.5, +2.3] |
| gate vs frozen harness, NLA ×4 | 66.4 vs 68.8 | −2.4 [−6.4, +1.4] |
| frozen OLens ×4 vs frozen NLA ×4 (reference) | 66.8 vs 68.8 | −2.0 [−5.9, +1.7] |
| frozen OLens ×16 vs frozen NLA ×4 (reference) | 70.0 vs 68.8 | +1.2 [−2.6, +5.2] |
| transcript: 3 experiments vs none | 79.5 vs 64.7 | +14.8 [+9.2, +20.3] |
| OLens ×16 + gate: 3 experiments vs none | 80.5 vs 69.4 | +11.1 [+6.5, +15.5] |
| NLA + gate: 3 experiments vs none | 80.0 vs 66.4 | +13.6 [+8.8, +18.3] |
| 3 experiments: OLens ×16 + gate vs transcript | 80.5 vs 79.5 | +0.9 [−3.0, +4.7] |
| 3 experiments: NLA + gate vs transcript | 80.0 vs 79.5 | +0.5 [−3.3, +4.2] |

- **The gate doesn't replicate.** Its +3.6 on dev was the best of 3 arms (selection noise). On held-out it is −0.6 for OLens ×16 and −2.4 for NLA.
- **OLens vs NLA.** Neither primary test shows OLens beating NLA. The small OLens ×16 edge comes from the 16 samples, not the gate.
- **Experiments replicate strongly** (+11-15 pp for every arm). With 3 experiments, the readers add nothing over transcript on pass@1.

## What to do next
1. **Stop using pass@1 on CHIVE as the reader eval.** The transcript already contains the answer for most incidents:
   - a transcript-only generator has a true cause in its top 5 for 97% of incidents;
   - with 3 experiments, transcript reaches 80%;
   - the ground-truth "non-obvious" points are mostly experiment outcomes no reader can state.

   Readers can only show up where they have headroom:
   - (a) experiment-proposal quality (readers +5 pp on true-cause alternatives; §4);
   - (b) rival rejection (−8 pp refuted rivals);
   - (c) blind-equal MCQ (running);
   - (d) a hidden-cause subset: incidents where the cause is visible in activations before its textual cue, like prompt_027128's "Vulkan" at t677 vs t799. Building (d) needs human-verified bank evidence: automated "the bank states X" judges were ~10% precise in two audits.
2. **The OLens failure mode is vivid single lines.** OLens's specific content lures, and NLA's bland genre labels don't. What helps OLens is aggregation (×16 samples; consensus across draws, +1.7 in the earlier chive work), not prompt rules. Prompt rules have now failed 6 times: recurrence counts, LLM filter, multi-candidate, antilure, echotag, revision gate. Worth trying:
   - serving readouts as recurrence-ranked clusters (counts first, quotes second) instead of raw samples;
   - or training OLens with a calibration term (penalise confident specifics that don't recur across samples).
3. **Harness: make experiments the backbone and use readouts to propose them.** That is where readers show a measurable edge: alternative-experiment targets +5 pp on true causes.
