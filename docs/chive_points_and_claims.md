# CHIVE: point-level and claim-level evals of the held-out explanations

These are two more ways to score the free-form explanations from the held-out test in
`chive_harness_climb.md` (204 incidents × 5 runs, frozen harness `textfirst + absence`). A
single 0-10 judge score per answer hides two questions: which parts of the explanation does
each method recover, and how often does it assert something the experiments disproved? Results
are as of 2026-10-14. Raw outputs are on the Modal volume `chive-archive` under
`/results_2026-10-14/` (`chive_points4`, `chive_claims`, `chive_mlnla`).

## Held-out pass@1 with the multi-layer NLA arm

The multi-layer NLA (`camilablank/qwen3.6-27b-multilayer-nla-rl600`, reader kind `mlnla`, k=4,
11 layers) was added as one more method on the same harness.

| method | pass@1 | pass@5 | least-confident third, pass@1 |
|---|---|---|---|
| OLens | 73.1 | 89.7 | 66.5 |
| J-lens | 72.0 | 85.8 | 62.4 |
| NLA (L42) | 71.7 | 87.3 | 58.5 |
| multi-layer NLA | 70.0 | 84.3 | 56.8 (−9.7* vs OLens) |
| transcript only | 70.1 | 82.8 | 58.2 |

\* = the paired 95% bootstrap CI over incidents excludes 0. Figure:
`scripts/plot_heldout_methods.py` (`docs/figures/chive/heldout_methods.png`).

## Point-level eval (`scripts/points_chive.py`)

### What is graded

1. **Points.** One Opus call splits each incident's established causes into 3-8 distinct atomic
   points, and a second call merges near-duplicates. That gives 1,422 points over 198
   incidents. A third call tags the points that belong to the primary cause T1 (780 of them).
2. **Point scores.** For each final answer, an Opus grader scores every point from 1 to 10:
   10 = stated clearly and specifically, 5 = stated partly or vaguely, 1 = absent or
   contradicted.
3. **The answer's claims.** The same grader splits the answer into its own claims (at most 10)
   and labels each one:
   - *supported*: the established explanation says it;
   - *contradicted*: the answer asserts it, and it matches a ruled-out alternative or
     conflicts with a measured result;
   - *rejected*: the answer names it only to rule it out;
   - *not covered*: CHIVE cannot check it.
4. **Rivals.** A separate call says, for each alternative CHIVE ruled out, whether the answer
   *rejects* it, *asserts* it, or does not mention it.
5. **Non-obvious points.** An independent no-tools answerer (Opus, conversation + response +
   behaviour only) writes 3 explanations per incident, graded on the same points. A point is
   *obvious* when it scores ≥ 7 in at least 2 of those 3, and *non-obvious* otherwise (1,036
   of 1,422 points).

**Grader reliability:** an independent re-grade of 297 answers gives a point-score correlation of
0.92, the same stated/not-stated call (≥ 7) on 95% of points, and a disproved-count correlation
of 0.81.

### Metrics

Each metric is computed per answer, averaged over the 5 runs, then averaged over incidents. The
disproved share and the rival shares pool the 5 runs before dividing.

| metric | numerator | denominator |
|---|---|---|
| non-obvious recall | Σ over non-obvious points of (score − 1) | 9 × number of non-obvious points |
| primary-cause recall | Σ over T1 points of (score − 1) | 9 × number of T1 points |
| disproved share | contradicted claims | supported + contradicted + not covered |
| disproved per answer | contradicted claims | answers |
| precision | supported | supported + contradicted |
| rivals rejected / asserted | ruled-out alternatives rejected / asserted | ruled-out alternatives |

### Results (198 incidents)

| | transcript | OLens | multi-layer NLA | J-lens | OLens − transcript |
|---|---|---|---|---|---|
| non-obvious recall | 18.7% | 18.5% | 17.7% | 16.5% | −0.2 [−1.5, +1.2] |
| primary-cause recall | 37.0% | 38.0% | 36.6% | 35.9% | +1.0 |
| disproved share | 15.5% | 9.5% | 9.3% | 9.1% | −5.9 [−7.5, −4.5]* |
| claims per answer | 7.32 | 8.07 | 8.20 | 8.08 | +0.76* |
| supported per answer | 3.41 | 2.92 | 2.92 | 2.68 | −0.49* |
| disproved per answer | 0.94 | 0.65 | 0.65 | 0.62 | −0.28* |
| not covered per answer | 1.72 | 3.28 | 3.41 | 3.55 | +1.57* |
| precision | 78.4% | 82.1% | 81.8% | 80.8% | +3.7 [+1.6, +6.0]* |
| rivals rejected | 24.0% | 24.9% | 24.7% | 24.9% | +1.0 [−1.9, +3.9] |
| rivals asserted | 21.1% | 15.7% | 16.9% | 15.1% | −5.4 [−7.8, −3.1]* |

On the hard 46: OLens non-obvious recall is +3.7 [+1.1, +6.4]*, multi-layer NLA +2.5*, J-lens
+0.8, and every reader has a disproved share about 13 points lower than transcript. On the
least-confident 66: OLens is +2.2 [−0.1, +4.6].

**Reading.**
- **Readers make fewer wrong claims, not fewer claims.** They make *more* claims. The extra
  ones are mostly statements about readouts: 70% of OLens's not-covered claims mention tokens,
  layers or readouts, against 9% for transcript. CHIVE cannot check those, so they inflate the
  disproved share's denominator.
- **The wrong-claim drop is still real.** Disproved claims fall by 30% (0.94 → 0.65) while
  supported claims fall by 14%, so precision rises. Prefer disproved per answer and precision
  to the disproved share.
- **This is a reader effect, not an OLens one.** All three readers show the same pattern.
- **Recall of the explanation does not rise overall.** It rises only on the hard incidents.

### Where the gains are (pass@1, `/tmp/chive_points4/analysis`)

Incidents are split by how the other three readers do. Each reader gets the same treatment.

| gain vs transcript | others mostly fail (≤ 40%) | middling | others mostly pass (≥ 80%) |
|---|---|---|---|
| OLens | +0.0 (n=46) | +1.6 | +5.4* |
| NLA L42 | −0.5 | −7.8 | +6.5* |
| multi-layer NLA | −6.5 | −2.3 | +3.8 |
| J-lens | −0.9 | −3.2 | +4.8* |

- **The average gains sit on incidents where every reader gains.** Per-incident gains correlate
  0.6-0.75 across readers.
- **The tail favours OLens.** Of 45 incidents transcript fails (pass@1 ≤ 20%), OLens rescues 15
  (pass@1 ≥ 60%), NLA 10, multi-layer NLA 12 and J-lens 14. **Solo rescues**, where every
  other reader fails, go to OLens 4 times and to the others 0 times (000779, 000782, 003402,
  004195). This test was chosen after seeing the data, so treat it as a hypothesis; confirming
  it needs more runs on those 45 incidents.

### Answer-level support pass (stage `support`, results pending)

A reader that never sees the conversation reads either the final answer or the whole episode.
It marks each of CHIVE's points (true) and ruled-out alternatives (false) as **yes / no /
silent**:
- **found**: true points marked yes;
- **truth rejected**: true points marked no;
- **wrong asserted**: false statements marked yes;
- **rival rejected**: false statements marked no.

Episode-yes minus final-yes is "found but dropped". 7,698 of 8,640 calls finished before the
keyless token was revoked; rerun `stage=support`, which resumes.

## Claim-level eval (`scripts/claims_chive.py`)

This is CHIVE's own binary claim format. Each claim is a statement about one intervention, true
when CHIVE measured an effect of at least 30 points. There are 688 claims, balanced, built from
CHIVE's eval sets (`data/chive/claims/claims300.jsonl`). Each claim gets one focused episode
whose output is `probability_true`. The agent must inspect the edited tokens before it submits.
Scoring is pooled AUROC (CHIVE's headline) and within-incident AUROC.

| setting | AUROC |
|---|---|
| plain no-tools answerer (CHIVE's) | 0.866 |
| plain, with the effect-size gap stated in the prompt | 0.885 |
| focused episode: transcript / OLens / J-lens / NLA | 0.869 / 0.873 / 0.874 / 0.860 |
| two-agent, read-only (B reads A's whole OLens episode) | 0.875 |
| two-agent with one experiment, every method | ≈ 0.90 |

- **No reader beats transcript on pooled AUROC.** The prior dominates: plain alone reaches 0.866.
- **On the 152 claims the plain answerer gets wrong**, focused OLens gets 32% right against 26%
  for transcript and 25% for a shuffled-readout control. Two-agent read-only is OLens 29%
  against transcript 18%.
- **Keeping only claims with concreteness ≥ 3** (376 claims) changes nothing.

Figure: `scripts/plot_claims.py` (`docs/figures/chive/claims_auroc.png`).

## Story variant (`variants=['story']`)

This variant asks for a chronological account of the model's internal state, ending in a
required Frames section. It reconstructs the deploy-lie case well: 13.7 of 14 facts against 7.9
for the frozen harness, and 12/12 against 11/12 out of sample. But it scores worse on
cause-finding (held-out pass@1 −4.3*). It is a discovery mode, not the eval harness.

## SWE-chat pilot (40 incidents × 5 runs)

| method | pass@1 | vs transcript |
|---|---|---|
| transcript | 15.0 | — |
| OLens (6 layers) | 28.5 | +13.5* |
| OLens (11 layers) | 24.0 | +9.0* |
| multi-layer NLA | 25.0 | +10.0* |
| NLA | 24.5 | +9.5* |
| J-lens | 27.0 | +12.0* |

**Caveats:**
- Each incident's history is the original agent's session (Claude Code, Codex, …), flattened
  and clipped. Only the final turn is Qwen's (30 samples, thinking off).
- Clipping can create the behaviour. In `scaleup_swechat_001677` the final task notification
  lost its header, the part naming which agent finished. The passing OLens run there used no
  readouts. The runs that did use them found the header-less, out-of-order notification
  account, which CHIVE never tested, and were scored 2-3.
- Read single SWE-chat cases against the clipped prompt, not the original session.

## Tried and dropped

- **A rewritten transcript-first rule (`textfirst2`) and a claims prompt v2** (estimate from
  the text first, plus a flag marking readout names absent from the conversation). Dev
  free-form −2.6 (n.s.); claims AUROC −0.014*. They were removed from the code; the last
  commit with them is 546c120.
- **A strict gate on overturning the transcript hypothesis.** Replaced by plain prompt
  guidance, then dropped.

## Reproduce

```
P=".venv/bin/python scripts/points_chive.py"; O=runs/points
for s in points tag plain grade rivals support rep report; do $P stage=$s out=$O; done
.venv/bin/python scripts/plot_points.py out=$O plots=plots/chive
```

The answer records come from the held-out run (`ARMS` in `points_chive.py`). Every stage
resumes. `stage=report` prints all the tables above except the reader-split ones.
