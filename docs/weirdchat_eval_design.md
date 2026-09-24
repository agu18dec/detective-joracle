# Measuring how good each tool is at understanding in-the-wild behaviors

Tools: OLens (verbalizer, layers 20–60), J-lens (Jacobian token bags), NLA (RL verbalizer, layer
42), and a black-box arm — each driving the same investigator with the same brief and budget on the
same WeirdChat patterns. There is no ground truth for *why* a model does something, so the eval
scores what an explanation lets you **predict and control**, which is behavioral and measurable
with a judge calibrated against WeirdChat's own labels (`docs/weirdchat.md` §Judge calibration).

## 1. Primary: explanations as forecasts on a held-out intervention bank

- **Bank.** Per pattern, ~10 pre-registered arms, tool-agnostic: clause deletions, referent swaps
  (second → third person), capability cues, forced openings, register rewrites, and **null arms**
  (meaning-preserving paraphrases, irrelevant edits) that should not move the rate. Seed it from
  the union of every arm's `would_test_by` plus a fixed set of generic edits; freeze it before
  scoring. Measure every arm at n=64 with the judge of record. That table is the ground truth.
- **Forecasting stage.** After `finish()`, in the same context with no further tools (the shape of
  the quirk game's closed-set stage), the agent predicts each bank arm's flagged rate. Score per
  (tool, pattern): directional accuracy (down / up / none), Brier on the rate, and **specificity**
  — the nulls must be predicted as nulls. An explanation that predicts everything drops scores
  low here by construction.
- Why this and not today's arms: the current arms were written from the OLens run's own
  `would_test_by`, which flatters OLens. A shared bank asks every tool the same questions.

## 2. Where a lens should beat black-box: localization

Lens arms make claims black-box cannot — "the role is live at the boundary, L40–56", "commitment
is at the first bold token". Both are testable:
- **Commitment point**: a prefix-forcing curve (flagged rate vs. number of forced reply tokens
  from a flagged reply) locates where the outcome is actually decided; score the claimed token
  against it.
- **Site**: steer against the claimed direction at the claimed position/layer band vs. a
  matched-norm random site (and a random direction at the claimed site); score = effect at the
  named site − effect at controls, with a coherence check on the steered replies (the degeneracy
  trap). Needs `steer` support on the lens server.

## 3. Novelty that verifies

From the blind agreement pass, take the mechanisms only a lens arm proposed and score their
forecasts (§1) against the mechanisms only the black-box arm proposed. The lens adds information
when lens-only mechanisms verify at a higher rate — not when it adds confident prose.

## 4. Hygiene metrics

- **Citation fidelity**: verbatim verification of quoted cells per tool (OLens: 129/335 = 39% on
  the first pass). A tool that leads the agent to cite cells that do not exist is worse.
- **Transfer**: apply a mechanism's intervention from pattern A to pattern B of the same behavior;
  a mechanism that moves B's rate explains the behavior, not the prompt.
- **Cost**: chat probes, output tokens and lens GPU-seconds to reach a mechanism that later
  verifies.

## 5. Controls

- Black-box at identical budget (the floor).
- **Shuffled-lens control**: readouts from a different pattern. If forecasts stay as good, the
  agent is not using the lens content.
- ≥ 3 seeds per (tool, pattern); one seed is a case study.
- Report per behavior with k/n; never pool tools; deltas over absolute rates (the judge
  under-flags relative to the study's).

## 6. Scale and cost

Qwen3.6-27B only (the lenses are 27B-only). Lower the rate floor to ~0.10 to reach 35–40
patterns over all 11 behaviors. 4 tools × 3 seeds ≈ 450 investigator runs; ~40 × 10 × 64 ≈ 25k
judged samples. Roughly two days on the current infrastructure. New code: the bank builder, the
forecasting stage, the prefix-forcing curve, and `steer` on the lens server. Already in place:
agreement (§3), prediction scoring against measured arms (§1 without the bank), fidelity (§4).
