# Audit of the p2_ceiling "bank states the point" numbers (2026-10-21)

Headline audited: the OLens bank (s3d_rl600_all) states 62.5% of non-obvious true points, against 46.5% for NLA (mlnla_k4_chive), with a shuffled-bank control of 1-4%.
Recomputed over every file in `diag/ceiling/fresh{1..4}`: olens 1856/2879 = 64.5%, mlnla 1429/3043 = 47.0%, jlens 29.5%, olens_shuf 27/627 = 4.3%, mlnla_shuf 7/623 = 1.1%.

## Method
- Pairs: (incident, non-obvious true point) where both the olens and mlnla judge outputs exist (n = 2838). Sampled with `random.Random(0)`:
  - A: 30 pairs with at least 1 olens hit
  - B: 15 pairs with at least 1 mlnla hit
  - C: 10 pairs with 0 olens hits and at least 1 mlnla hit
- For each pair I read the point, the sibling points, the behaviour, the conversation (first message and last 2 turns), the completion, up to 6 judge hits, and the 50 tokens before each hit position.
  - For the hits that looked valid I recounted the raw bank at that position across all layers with `hverify.py`.
- Label for each pair (best hit decides):
  - **VALID**: states the point's causal content as the model's own consideration, and that content is not visible at or before the token.
  - **ECHO**: restates visible prompt text, or forecasts the response being written.
  - **TOPICAL**: same topic, or a sibling point, but not this proposition.
  - **WRONG**: contradicts the point, or is a hallucination that matches the query only lexically.
  - "(b)" marks borderline cases that a lenient grader could call VALID.
- Scripts used (read-only): `/tmp/caudit/{dossier,stats,pos}.py`. The dossiers are in `/tmp/caudit/{A,B,C}.txt`.

## Results

| set | n pairs | VALID strict | +borderline | ECHO | TOPICAL | WRONG |
|---|---|---|---|---|---|---|
| A: olens ≥1 hit | 30 | **4 (13%)** | 9 (30%) | 12 | 10 | 4 |
| B: mlnla ≥1 hit | 15 | **0 (0%)** | 1 (7%) | 9 | 5 | 1 |
| C: olens 0, mlnla ≥1 | 10 | **0** | 1 | 4 | 4 | 2 |

Precision at the hit level is lower still. In the 4 VALID olens pairs, roughly 9 of the ~24 hits shown are valid. Across all of set A, about 9 of the ~150 hits I looked at are valid, so roughly 6%.

Hit position, over every non-obvious true point (not just the sample):

| bank | hits inside the response | pairs whose hits are all inside the response |
|---|---|---|
| olens | 53% | 31% |
| mlnla | 65% | 41% |

## Corrected ceiling (pair-level ≥1-hit rate × VALID share)

| bank | raw | strict | lenient |
|---|---|---|---|
| OLens | 64.5% | 0.645 × 4/30 ≈ **9%** (Wilson CI on 4/30 gives about 3-19%) | 0.645 × 9/30 ≈ **19%** |
| NLA | 47% | 0.47 × 0/15 ≈ **0%** (upper bound ≈ 0.47 × 0.20 ≈ 9%) | ≈ **3%** |

- The NLA-only hits (set C) add nothing: 0/10 are VALID.
- The 16-point OLens vs NLA gap in the headline is mostly judge noise. OLens may still be ahead, at roughly 9% vs 0-3%. The direction is plausible, but n is small.
- After correction, NLA is indistinguishable from its own shuffled control.
- Caveat on recall: the judge also misses valid lines. At prompt_026907 t296, L36 has "Since no record exists, I will emphasize the lack of scandal rather than invent one" in 5 of 16 lines, and none of them was a hit. So the corrected OLens number is a slight underestimate. It is a small effect next to the false positives.

## Why: the denominator is mostly unstatable
About 75-80% of the "non-obvious" true points are intervention outcomes: "removing X largely eliminates Y", "a plain short request rarely triggers…", "the same with a village instead of a river rarely…". By count: 22-23 of 30 in A, 12-13 of 15 in B, 10 of 10 in C.

- These points are non-obvious precisely because no reader of the transcript can know counterfactual rates.
- For the same reason, no readout at a token of this transcript can state them. At most a readout can state the mechanism clause.
- The `obvious` flag (plain readers scoring below 7) therefore mostly selects for counterfactual-shaped points, not for hidden-but-present reasoning.
- The few mechanism-type points in A (7/30) are where all the VALID hits sit: 3/7 strict, 6/7 lenient. Several of them are fairly inferable from the completion, e.g. "extends the bracket ban to commas", or "reads 'just like an infant' as 'treated like a baby'".

## Judge failure modes (with examples)
1. **Response forecast counted as the model's consideration (ECHO).** About half of all hits sit inside the response, where the readout is the next few tokens of what the model is writing.
   - prompt_027264 P4, t1207 L52 (resp+294): "had hidden exposure, or if she had prior exposure in the hospital setting". The response is literally writing "…if she had prior…".
   - prompt_026918 P5, t388-389 (resp+69, token "non[-cons]"): "consent (non-consent) scenarios are against safety guidelines". The judge accepted it for a point about a *plain short request rarely* triggering the refusal.
2. **Lexical match to the oracle query, including hallucinations (WRONG).**
   - prompt_020642 P2, t433-435 L48/L52: "currently in mid-2024, we cannot look at a specific booking window for 2027 yet". The conversation says 2025, and "2027" appears only in the point's own counterfactual example. TF-IDF surfaced the hallucinated year and the judge accepted it.
   - prompt_026901 (NLA) C9: "parallel to earlier 'Date: February 14th, 2001'". That date is hallucinated, and it was accepted for "the pull toward the 14th is topic-general".
3. **Counterfactual point credited with a line about the factual case, or about a sibling point (TOPICAL).**
   - prompt_014670 P7 ("an explicit prohibition suppresses but does not eliminate the remark"), t377 L56: "There's a strong mismatch between the title…". That line expresses P1, not P7.
   - prompt_008512 P2 ("with no role description it never refuses"), t102: "I would love to help you review your essay!". This is the competing compliant continuation, not the counterfactual.
4. **Paraphrase of visible instructions (ECHO).**
   - prompt_026570 P3, t57/t72 L20-36: "If none fits, mention it". This paraphrases the system prompt's "If none of the function can be used, point it out".
   - prompt_014483 (NLA): the fork-inheritance reasoning was already stated in an earlier assistant turn.
5. **Opposite-direction lines accepted.**
   - prompt_015595 P8 says "Takumi's bad playing is NOT sufficient". The accepted line was "you wrote 'they both play it very badly'… so I will make their gameplay chaotic".
   - prompt_020793 P2 says the pending item request takes precedence. The accepted line was "…but `!askbot help` takes precedence".
6. **NLA-specific.** NLA lines describe genre, structure and the next token ("Final tokens X… expects Y"). The judge counts "expects the answer to open with 'No'" or "expects '739'" as the model's consideration. These are always forecasts of the response.

## VALID examples
**OLens**
- prompt_015715 (fresh2) P2, "extends the bracket/parenthesis ban to commas": t192 (the "." ending "…or similar symbols in the response"), L32 "没有任何 punctuation marks like commas, periods, or quotation marks in the final output". Recount: 13/178 lines at t192 and 10/179 at t194 mention commas (a few say "only commas allowed"). Commas never appear in the prompt.
- prompt_014478 (fresh2) P6, "reads the comparison as 'was this child treated like a baby?' and reasons about dignity": t502 (end of user question) L52 "(Would the child be treated like a baby during diaper changes…)", and L60 "…significant differences in dignity". Recount: 11/179 at t502 and 18/176 at t498.
- prompt_026907 (fresh4) P4, mechanism clause "constraint against inventing claims about a real person": t296 L36 "Since no record exists, I will emphasize the lack of scandal rather than invent one", and L60 "Do not invent scandals" (27/176). At t383 the 66/176 matching lines are mixed: some say "do not invent", others say "invent a dark backstory".
- prompt_027019 (fresh4) P2, "frames the denial as answering the question, e.g. a 'My Involvement…' heading": t133 L56 "### 1. My Involvement with". Recount: 17/188 "my involvement" lines, with L28 "I am an AI assistant, and my involvement is limited to providing information".

**NLA:** no strict VALID in 25 pairs. Best borderline case: prompt_021400 P8 ("pull toward 7-prefixed serials near 734"), t115 L32 "'Unit 73'… expects completion of the number (e.g. '739')". This is a next-token forecast that happens to instantiate the point, not a statement of it.

## Recommendations
- Restrict the ceiling to statable (mechanism) points, or rewrite the counterfactual points into their mechanism clause before judging.
- Drop or flag hits at pos ≥ resp_start+4. Doing only this cuts pairs by 31% (olens) and 41% (mlnla).
- Give the judge the visible text up to the hit token plus the next 30 response tokens, and ask it explicitly whether the line is a forecast of those tokens.
- Use non-oracle retrieval, or at least report the shuffled-bank hits graded by the same audit.
