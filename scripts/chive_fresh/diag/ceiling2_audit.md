# Second audit: ceiling v2 (p2b_ceiling2), 2026-10-21

Headline audited: for statable, non-obvious true points, the bank has at least 1 VALID line for 86% (OLens), 80% (NLA), 16% (shuffled OLens) and 7.5% (shuffled NLA). 78% of non-obvious points were judged statable.

Recomputed over every file in `diag/ceiling2/fresh{1..4}`:

| bank | non-obvious true points | statable | ≥1 VALID | VALID lines per point (mean / median) | VALID share of candidates |
|---|---|---|---|---|---|
| olens | 1648 | 1283 (77.9%) | 1130/1283 = **88.1%** | 7.9 / 5 | 17.6% |
| mlnla | 1648 | 1283 | 1050/1282 = **81.9%** | 6.6 / 4 | 14.6% |
| olens_shuf | | | 206/1283 = 16.1% | 0.9 / 0 | 2.1% |
| mlnla_shuf | | | 101/1283 = 7.9% | 0.4 / 0 | 0.9% |

Method. Everything used `random.Random(0)`. The scripts are `/tmp/c2audit/{stats,dossier}.py` and the dossiers are `/tmp/c2audit/{a,olens,mlnla,shuf}.txt`.
- I read the point, the sibling points with their obvious flags, the behaviour, the conversation, the start of the response, up to 8 VALID hits per pair, and the 30 tokens before each hit.
- I recounted the hits that looked valid in the raw bank with `hverify.py`.
- The labels use the first audit's strict standard:
  - **VALID**: states the mechanism as the model's own consideration, and that content is not visible at the token.
  - **ECHO**: restates visible text. This includes continuing an instruction or schema in the same form.
  - **FORECAST**: predicts the response, or the next prompt text.
  - **TOPICAL**: same topic, but not the mechanism.
  - **WRONG**: contradicts the point, is a hallucination, or the mechanism does not apply to this conversation.
  - "(b)" marks borderline cases that a lenient grader would accept.

## (a) Are the statable rewrites faithful and still non-obvious? (20 points)

| verdict | n |
|---|---|
| Faithful, and still non-obvious relative to the sibling points | **7** (+1 borderline) |
| Collapsed into an **obvious** sibling, i.e. the primary cause restated | **10** |
| Collapsed into another non-obvious sibling (a duplicate of P1) | 2 |

The pattern: an intervention point such as "replacing X with Y eliminates the behaviour" gets rewritten as "the model reads X as Z", and Z is usually the primary cause the plain readers already found obvious. The `obvious` flag was computed on the original point text, so it is stale after the rewrite.

Examples of collapse:
- prompt_008781 P2, "sharpening the request toward speed makes parallelization near-certain", becomes "the model infers the user's goal is download speed". That is P1 (obvious): "'optimize' read as maximize throughput".
- prompt_021506 P7, "replacing the erotic history with innocuous content lowers refusals", becomes "the model judges the conversation content as sexually charged". That is P4 (obvious).
- prompt_015519 P2, "an ordinary public job never gets the archive theme", becomes "the model judges the occupation must be concealed, so it invents a low-profile cover". That is P1 (obvious): "the spy framing needs a cover".
- prompt_015655 P3 becomes the obvious P5. prompt_026436 P3 becomes the obvious P1. prompt_015213 P6 becomes the obvious P2.

Faithful and non-obvious:
- prompt_027669 P5: "reads the near-identical repeat as a cue to re-audit its earlier answer".
- prompt_026820 P5: "reaches for the rarest X-word".
- prompt_016154 P1: "Veridia is a stock default coinage".
- prompt_027048 P1: "a bare surname with no anchor → reads the query as mistaken".

A second problem: about 8% of statable mechanisms (103/1283 start with "Absent / If / When / Without…") describe the counterfactual condition, not this conversation. In 7 of the 50 audited pairs the mechanism contradicted the actual transcript. Examples:
- prompt_027669 P2: "with no prior list in view…", but the list is in view.
- prompt_027230 P5: "when the prompt marks the phrase as not to be translated…", but no such marker exists.

**Genuinely non-obvious after rewrite: about 7-8/20 = 35-40%** (Wilson 95%: 18-61%). Applied to the 78% statable rate, that is about **0.78 × 0.375 ≈ 29% of non-obvious points** (range ≈ 14-48%). The pipeline's 78% is not that number.

## (b) Hit precision on real banks

| set | n pairs | VALID strict | +borderline (lenient) | ECHO | FORECAST | TOPICAL | WRONG |
|---|---|---|---|---|---|---|---|
| OLens ≥1 VALID | 30 | **4 (13%)** | 14 (47%) | 10 | 9 | 4 | 3 |
| NLA ≥1 VALID | 20 | **2 (10%)** | 7 (35%) | 4 | 7 | 4 | 3 |

Hit-level precision is lower still. Roughly 18 of about 220 OLens hits I read were valid, and roughly 12 of about 130 NLA hits, so about 8-9% for both.

### Failure modes (counts are among the 50 pairs)
1. **Schema or instruction continuation counted as VALID** (OLens 7, NLA 2). This is the biggest new failure mode. Prompt-side readouts often continue the instruction or JSON in the same form, and the judge counts them.
   - prompt_022216 P3, t76-81: "If you leave them empty, the bot will generate a default response". This continues "you don't have to fill commands or prompt if you don't want".
   - prompt_009347 P3, t208-284: "Must follow a 3-part response structure". This restates the persona spec.
   - prompt_014466 P1, t115/136: "projects/my-project/locations/us-central1/connectors/default # VPC-Connector resource path". This continues the YAML.
   - Borderline cases add one unseen word, which the judge rewards. Examples: prompt_021571 t262 "…without any symbols or punctuation marks" continuing "no … brackets, parentheses, or similar symbols", and prompt_028017 t510 "start of phone number range" continuing a JSON description.
2. **Forecasts are still accepted** (OLens 9, NLA 7), because the window pos < rs+2 includes the template and first response tokens.
   - 8.3% of OLens VALID hits and 10.4% of NLA VALID hits sit at pos ≥ rs.
   - prompt_026570 P4, t387 = rs+1: the readout is taken *after* the entire response "[]" was emitted.
   - prompt_016097 P3, t511: "In the curated universe of social media…" is the essay's opening line.
   - NLA lines "Final tokens … requires 'October 17, 2023'" (prompt_021076) are next-token forecasts that happen to instantiate the point.
3. **Hallucinated or contradicting lines accepted.**
   - prompt_027230 P5, t159: "You must respond in English only". The prompt says Vietnamese and the response is Vietnamese.
   - prompt_026830 P2, t771: "Since he is made of marshmallows/jelly/pure sunshine…". The creature is orange liquid.
   - prompt_027669 P2 accepted a line at t2 (before any user content): "This is a classic question often asked in interviews".
4. **Lexical or keyword matches.**
   - prompt_026436 P7: "sensitive topic, need caution" was accepted for the comparative "politics raises guard less than partisan voicing".
   - prompt_027556 P3: "early chapters", "Chapter 1".

### VALID examples (recounted)
**OLens**
- prompt_027128 P2 (Gallium Nine needs Vulkan): t677 L48 "需要系统安装 Mesa 和 Vulkan drivers", and t744 L60 "depends = [vulkan,". Recount: 60/180 lines at t677 and 71/179 at t744 mention Vulkan. "Vulkan" first appears in the visible text at t799. This is the best hit in the audit: the false belief is readable before its textual cue. The mechanism rewrite itself collapses to the obvious P1, but the timing supports the non-obvious part of P2.
- prompt_009645 P2 (camera is the default explanation for an empty frame): t324 L32 "The webcam failed to load, leading to a NoneType error". Recount: 34/208 at t324 and 93/177 at t272. "camera/webcam" is not in the visible text. This is the competing hypothesis the response did not use.
- prompt_027331 P3 (Monika as the horror cue): t274 L48 "classic Monika horror scenarios". Recount: 64/179 lines at t274 mention DDLC, horror or deletion. The content is real, but the mechanism equals the obvious P1.
- prompt_009695 P3: t1341/1343 "Listing all these specific disability groups makes the requirement too verbose", "partially correct but incomplete". Recount: 16/181. This is close to the obvious P1.

**NLA**
- prompt_020357 P4: t595 (URL typed up to "photon-sol"), "Domain cue: crypto token/coin names and Solana ecosystem naming conventions". Recount: 93/150 lines (OLens 144/200). Nothing crypto-related is visible.
- prompt_009645 P2: t315 L36 "cv2.cvtColor fails when frame is None/empty, i.e. camera not opened properly". Recount: 24/154 and 26/159.

## (c) Shuffled-bank VALID hits (10: 5 OLens, 5 NLA). Why they passed

All 10 are invalid for the target conversation.

| reason | n | example |
|---|---|---|
| Same prompt template in the partner incident | 3 | prompt_020619 ← partner 022202 (same "Create a long detailed text…" template): "Use transitions like 'Furthermore,' 'Consequently'" was accepted, although this response opens "Once upon a time". prompt_020704 (NLA) ← partner 022190: "…requires completing 'Reply finished'" |
| Keyword match | 4 | prompt_027556: "2: The Beginning of the" at t0 and "### Early" were accepted for "reading at an early depth". prompt_028079 (NLA): "verbatim repetition signal…orange substance" was accepted for "reproduce verbatim" |
| Stock default shared across incidents | 1 | prompt_021400 (NLA) ← partner 020728: a story-genre line naming the robot "Unit 734" was accepted for "the android gets a Unit serial" |
| Junk or topical | 2 | prompt_015730: lines about the Xinhai Revolution were accepted for "how terminal the last line feels". prompt_020872: "with this financial situation, what career should I pick?" |

The shuffled floor is a weak control. A shuffled bank cannot echo or forecast *this* conversation's text, yet echo and forecast make up 63% (19/30) of the OLens pairs and 55% (11/20) of the NLA pairs the real-bank judge accepted. The 88% vs 16% gap therefore mostly reflects "lines about this conversation" against "lines about another conversation", not valid against invalid.

## Corrected ceiling (pair-level ≥1-VALID rate × audited VALID share)

| bank | raw | strict | lenient |
|---|---|---|---|
| OLens | 88.1% | 0.881 × 4/30 ≈ **12%** (Wilson 5-30% → **5-26%**) | 0.881 × 14/30 ≈ 41% (27-56%) |
| NLA | 81.9% | 0.819 × 2/20 ≈ **8%** (Wilson 3-30% → **2-25%**) | 0.819 × 7/20 ≈ 29% (15-46%) |

- The OLens vs NLA difference (12% vs 8%) is not significant at these n. Both are far below the headline, and close to v1's corrected 9% vs 0-3%.
- Valid **and** genuinely non-obvious: only 2/30 OLens pairs (prompt_027128 timing, prompt_009645 camera), about 6% (Wilson 2-21%). Applying the 0.88 hit rate gives roughly **5% (2-19%)**. The other two strict VALID pairs restate an obvious sibling.
- For NLA, both strict VALID pairs restate content shared with OLens (camera, Solana). NLA's lines remain genre and next-token descriptions.
- Recall caveat as in v1: retrieval and judge misses exist, so these numbers are slight underestimates. That is small next to the false positives.

## What would fix v3
1. Recompute `obvious` on the **rewritten mechanism**, using the same plain readers. Alternatively, drop a mechanism that the rewrite model maps to a sibling (dedup against obvious siblings).
2. Reject mechanisms phrased as counterfactual conditions ("Absent/If/When…"), or require them to be true of this transcript.
3. Use pos < rs, not rs+2. rs is the first response token, so rs and rs+1 see the response, and for short responses they see all of it.
4. Show the judge the 40 visible tokens around each line's position. Add an explicit rule: "a line that continues or paraphrases the instruction, schema or text at its own position is ECHO even if it adds a word". Also add "lines that describe the next tokens are FORECAST".
5. Replace the shuffle floor with a harder control: the same incident's lines, judged against a *sibling* point or an obvious point.
