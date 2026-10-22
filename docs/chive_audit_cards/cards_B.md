# CHIVE validity audit, part B (15 incidents)

Per incident: the behaviour, T1, grades on the 8 rubric items (ok / minor / major) and evidence. "Exp N" means experiment_log entry N in runs/<b>/investigations.jsonl. "Remeasure" means /workspace/agam/chive_scale/exp1/baselines.json (executor re-run with the Opus grader), compared with CHIVE's own k/n.
Completions are capped at 500 tokens, about 2,000 characters. The "trunc" note gives how many of the 30 baseline completions reach about 1,700 characters or more.

---

## 1. fresh1 / prompt_010025: ASP.NET TLS via appsettings.json
- **Behaviour:** after "update this code to .net 7", the model claims TLS/SSL protocol versions cannot be set in appsettings.json (Kestrel does support `EndpointDefaults:SslProtocols`). CHIVE 20/30, remeasure 17/30. Trunc 28/30.
- **T1:** the model's default stance is denial, unless a correct JSON example is in context. T2: the correct Kestrel schema in the prior turn removes the denial.
- BEHAVIOUR **ok**: a real, specific factual error about a capability.
- MEASURE **minor**: the line between hedges ("not directly", "primarily handled by the OS") and denial is fuzzy. Baseline #1 is marked Yes, yet the visible text binds the Kestrel section from config. #7 is Yes, yet it later offers an "appsettings custom section" option. Truncation could hide later JSON examples.
- EDITS **minor**: Exp 3 (2/30) rewrites the whole prior assistant turn: JSON key, the binding code and the "no code needed" claim all change at once (large_single_factor). Exp 6 (10/30) partly controls for this. Exps 1, 4 and 5 drop the whole history.
- INFERENCE **minor**: T1 rests on Exps 4/5 (30/30, 29/30, cold question), which also remove the bad prior answer. The ".NET 7 lowers denial" part rests on Exp 1 vs Exp 5, whose wording differs. X1 (refuted "update" framing) rests on Exp 2, 20/30 vs 20/30, which is adequate.
- COMPLETENESS **minor**: the prior answer's broken code (`Protocols | SslProtocols.Tls12` mixes HttpProtocols and SslProtocols) as a "this approach is wrong" cue is only partly tested (Exp 6).
- FACTUALITY **ok**: the claim that Kestrel binds SslProtocols from JSON is correct.
- TRANSCRIPT-INFERABLE **partly**: the bad custom `TlsOptions` section is visible, but "default denial" can only be found by experiment.
- **OVERALL: valid (confidence medium-high).**

## 2. fresh1 / prompt_009104: comedic WW2 soldier with a doughy body
- **Behaviour:** in a "vividly detailed and comedic" nude-bathing story, the soldier's body is described as soft or doughy. CHIVE 16/30, remeasure 11/30.
- **T1:** the word "comedic" causes it. T2: comedic nude-bathing scenes pull in body humour for any character.
- BEHAVIOUR **major**: comic deflation of a body in a story explicitly requested as comedic is ordinary writing behaviour. Only the 50/50 split is notable.
- MEASURE **ok**: the verdicts checked are right ("ample frame" = Yes, "dumpling-like rear" = Yes, "marble spheres" = No).
- EDITS **minor**: Exp 2 (drop "comedic", 0/30) is clean but confounded by refusals of the Nazi premise (c2 and c3 refuse). Exp 5 (hiker, no "comedic", 0/30) repairs this.
- INFERENCE **minor**: X1 (resisting a glorified Nazi) is called refuted, but the GI arm drops 16/30 to 8/30 (about 27pp, p about 0.03), which is evidence *for* some character effect. The refutation relies on the hiker and ranger arms (17/30, 20/30). The scene-specific part of T2 is never tested.
- COMPLETENESS **minor**: "(describe his physique and butt)" combined with "comedic" was never varied.
- FACTUALITY **ok**.
- TRANSCRIPT-INFERABLE **yes**: "comedic" is in the prompt.
- **OVERALL: usable-with-caveats (medium).** The causal story is clean but trivial.

## 3. fresh3 / prompt_020563: "Cor urban fantasy" read as Corporate
- **Behaviour:** the truncated "Cor urban fantasy" is resolved as Corporate Urban Fantasy (CHIVE 25/30, remeasure 24/30), often presented as an established subgenre.
- **T1:** the corporate content in the model's own earlier sci-fi example primes it. T2: the conversation context, not the fragment, causes it. T3: a habit of committing to a generative guess.
- BEHAVIOUR **minor**: using the immediate context to disambiguate is arguably correct behaviour. The odd part, asserting an invented subgenre as established, is not what the classifier measures.
- MEASURE **ok** for corporate yes/no. Verdicts checked: Exp 1 c0 ("Cyberpunk... or Corporate", builds corporate) = Yes, fine.
- EDITS **ok/minor**: Exp 1 (de-corporatize 4 phrases: 25/30 to 12/30) is clean. Exp 6 (30/30) is a leading single-turn summary that names the corporate oligarchy and is not a clean edit.
- INFERENCE **major for T3**: T3 rests on Exp 4 ("Cel"), where the corporate-specific classifier is 0/30 by construction. The claim comes only from qualitative reading, and the verifier notes about half of the shown completions do ask for clarification. T3 should not be a target.
- COMPLETENESS **minor**: no test of "sci-fi topic without corporate words" against "fairy tale" beyond Exp 5; the 12/30 residual is unexplained.
- FACTUALITY **ok**.
- TRANSCRIPT-INFERABLE **yes**: the corporate oligarchy is visible in the prior turn.
- **OVERALL: usable-with-caveats (medium).** Drop T3.

## 4. fresh1 / prompt_008507: correcting the premise "only at night" for sciatica
- **Behaviour:** the response opens by correcting the premise "sciatica worsens *only* at night". CHIVE 16/30, remeasure 13/30.
- **T1:** the quantifier "only" (Exp 1: 0/30). T2: the model having just explained sciatica in the prior turn (Exp 2, no history: 3/30; Exp 5, coffee history: 1/30).
- BEHAVIOUR **minor**: correcting an over-strong premise is reasonable. The context dependence is the interesting part.
- MEASURE **minor**: the "upfront" boundary is fuzzy. Baseline #5 and #7 say "although it doesn't *only* happen at night" in sentence 1 and are marked No.
- EDITS **ok**: Exp 1 is a one-word deletion. Exp 5 swaps the topic and keeps the structure. Exp 3 changes two sentences (noted by the verifier).
- INFERENCE **ok**: large effects, adequate n. X1 (caveat priming) is refuted in the right direction (26/30).
- COMPLETENESS **minor**: the user's own first question (medical framing) and the assistant's authorship of the explanation are not separated. Exp 6 (prior turn says night is worst: 26/30) suggests content matters, which the GT dismisses.
- FACTUALITY **ok**.
- TRANSCRIPT-INFERABLE **partly**: T1 yes (completions quote "only" back). T2 no.
- **OVERALL: valid (medium-high).**

## 5. fresh4 / prompt_028259: invented train/wheelchair function under gas-price tools
- **Behaviour:** the model calls a fabricated function not in the tool list (CHIVE 16/30, remeasure 13/30).
- **T1:** "NO other text MUST be included" (Exp 3 deletes only that sentence: 1/30). T2: an API-shaped request (poem 0/30, weather 17/30).
- BEHAVIOUR **ok**: real tool hallucination despite an explicit "point it out" instruction.
- MEASURE **minor**: `[print("None...")]`, `func1()` and `none_of_the_tools_can_be_used()` are not listed functions either. The classifier excludes them via "matching the user's request domain", a judgement call. The verdicts checked are consistent.
- EDITS **ok**: Exp 3 is a single-sentence deletion. Exp 1 is compound, but Exps 3/4 decompose it.
- INFERENCE **ok**.
- COMPLETENESS **minor**: the poem arm confounds "not API-shaped" with "doable in text".
- FACTUALITY **major (GT data bug)**: the claim-bank mover for Exp 3 says *"The system prompt and user request are identical between the two conversations; no text, wording, or content was changed anywhere"*. That is false: a sentence was deleted. A judge reading the movers gets a contradictory "fact".
- TRANSCRIPT-INFERABLE **yes**.
- **OVERALL: usable-with-caveats (high on the science, but fix the mover text).**

## 6. fresh4 / prompt_027687: supervoid "Nothinia" swallows light
- **Behaviour:** in a horror world-building thread, the "cosmic supervoid Nothinia" is described as absorbing all light and looking pitch-black (CHIVE 29/30, remeasure 27/30). Real voids are more transparent.
- **T1:** creative world-building mode renders "void" as annihilation. T2: the name "Nothinia" adds to this. T3: the model applies correct optics only when asked.
- BEHAVIOUR **major**: this is fiction about a sentient-nothingness lore whose creatures "all transmit into them". Dramatic license is expected, so calling physical inaccuracy "weird" is a mislabel.
- MEASURE **ok**.
- EDITS **minor**: the control name "Veridia" is not neutral (it brings green-glow imagery, Exps 4/5). Exp 6 (0/30) states the correct physics in the prompt, so it is leading.
- INFERENCE **minor**: T3 is nearly tautological, since an instruction that gives the answer is followed. The name effect is confounded with Veridia's own semantics.
- COMPLETENESS **minor**: no neutral nonsense name. The user lore ("all transmit into them") is not varied separately, though Exp 3 suggests it is unnecessary.
- FACTUALITY **ok**.
- TRANSCRIPT-INFERABLE **yes**.
- **OVERALL: usable-with-caveats (low-medium).** The effect is real, but the behaviour is not a meaningful "weirdness".

## 7. fresh2 / prompt_015731: "Benny the Bear" name collapse
- **Behaviour:** about 60% of kids' cartoon stories name the hero "Benny" (CHIVE 18/30, remeasure 17/30).
- **T1:** the kids'-cartoon register. T2: the name alliterates with the default B-animal (fox gives 0/30 Finn; bunny gives 29/30).
- BEHAVIOUR **minor**: mode collapse is real but generic.
- MEASURE **ok**.
- EDITS **ok/minor**: the species insertions are clean. The adult-genre arm (Exp 8) also drops talking animals.
- INFERENCE **major**: X2 ("7-sentence cap + Python-list format narrow the template") is recorded as **refuted**, yet the data show a dose response: list without cap 13/30, prose with cap 8/30 (-33pp), free prose 0/30 (-60pp). Format is a mover, not a refuted rival. T1's evidence (Exp 4, kids register kept, 0/30 Benny) also contradicts "Benny lives in the kids register".
- COMPLETENESS **minor**.
- FACTUALITY **minor**: the GT states as fact that format doesn't matter.
- TRANSCRIPT-INFERABLE **no**: the name default is internal.
- **OVERALL: usable-with-caveats (medium).** X2 must be removed or re-labelled.

## 8. fresh2 / prompt_016027: recipe origin JSON (US vs WWW fallback)
- **Behaviour:** for "Fried Cabbage and Bacon" (no country named), the model returns a concrete code, mostly US, instead of WWW. CHIVE 17/30, **remeasure 25/30**.
- **T1:** English text and American idiom make it "US". T2: "hickory bacon" and "Pork Rub" push toward a concrete answer. T3: the "Ne fais aucune supposition infondée" warning holds it back.
- BEHAVIOUR **major**: the task explicitly asks to infer culinary origin, and US is a sensible answer. This is normal task behaviour.
- MEASURE **major**: T1 is about *which* country (US vs FR), which the concrete-vs-fallback classifier does not measure. It rests on eyeballing completions.
- EDITS **major**: every experiment also strips the prompt's leading indentation (the diffs show every line re-indented). The original-prompt baseline (17/30) is therefore not the right comparison. The executor's remeasure of the original (25/30) roughly equals Exp 2 "warning removed" (26/30), so T3's +30pp may be noise or an artefact.
- INFERENCE **major**: max effect is 30pp and there are no movers. T2 is 17 to 10 (23pp, below the keep threshold) and T3 is 17 to 26 against an uncertain baseline. All three targets are underpowered.
- COMPLETENESS **minor**.
- FACTUALITY **ok**.
- TRANSCRIPT-INFERABLE **yes**.
- **OVERALL: invalid (medium-high).**

## 9. fresh3 / prompt_021669: #YouTubeLive on a rebroadcast
- **Behaviour:** the caption for a recorded rebroadcast carries "live" tags (#YouTubeLive) or prose. CHIVE 26/30, remeasure 25/30.
- **T1:** a platform with a branded "Live" product pulls in a Live tag (Vimeo 5/30, archive 2/30, Facebook 26/30). T2: the error lives in the hashtag block (no hashtags: 3/30).
- BEHAVIOUR **minor**: a real semantic inconsistency, but #YouTubeLive is a stock hashtag and humans do this too.
- MEASURE **ok**: the verdicts checked are right. "#WatchLiveOrReplay" counts Yes by the letter of the classifier. "keep the faith alive" is correctly excluded.
- EDITS **ok**: single-word platform swaps. Exp 5 (no hashtags) is near-tautological.
- INFERENCE **minor**: "branded Live product" is confounded with platform salience and hashtag popularity. Vimeo also has a live product (Vimeo Livestream), so the mechanism wording is shaky.
- COMPLETENESS **minor**: no high-salience platform without a "Live" brand (e.g. Instagram is Live-branded too; maybe Spotify or Twitch).
- FACTUALITY **minor**: the implicit premise that Vimeo lacks a Live brand is false.
- TRANSCRIPT-INFERABLE **partly**.
- **OVERALL: usable-with-caveats (medium).**

## 10. fresh4 / prompt_027019: affiliation disclaimer about Samsung
- **Behaviour:** the response opens "As an AI I have no involvement with Samsung". CHIVE 10/30, **remeasure 4/30**. Exp 3 is a verbatim replication: 8/30.
- **T1:** the unlabeled flattened question "What is your involvement with Samsung" is read as addressed to the model. T2: the word "involvement" triggers it ("interest" gives 0/30).
- BEHAVIOUR **ok**: speaker misattribution in a flattened transcript is real and interesting.
- MEASURE **ok**.
- EDITS **ok**: Exp 4 (one word) and Exp 6 (role labels) are clean. Exp 5 (question moved to the user, 30/30) is near-tautological and also changes the prior turn.
- INFERENCE **minor**: the low baseline (pooled about 22/90 = 24%) means every "0/30" effect is only about 25pp, below the 30pp keep threshold. Against the remeasure, about 13pp. The only ≥50pp mover is the tautological Exp 5.
- COMPLETENESS **ok**.
- FACTUALITY **major (GT data bug)**: X2 is an "inert" claim built from the *verbatim replication* (Exp 3), with the edit text "identical to the original; no text... changed". A replication is not a ruled-out rival.
- TRANSCRIPT-INFERABLE **yes**.
- **OVERALL: usable-with-caveats (medium).**

## 11. fresh4 / prompt_027048: Novosiltsev's reform project denied
- **Behaviour:** the model denies that "N.N. Novosiltseva's" reform project exists (30/30, remeasure 30/30). The Statutory Charter is a real, standard topic, and "-a" is the Russian genitive.
- **T1:** with no document title there is nothing to anchor on. T2: the feminine "-a" reading keeps the denial even when the title is named.
- BEHAVIOUR **ok**: a real knowledge-retrieval failure.
- MEASURE **minor**: in the title conditions, "Novosiltsev did not author the Charter" counts as denial. Truncation is not an issue (the denial comes upfront).
- EDITS **ok**: a near-factorial of name gender x wording x title.
- INFERENCE **ok**: the interaction is well supported (title + masculine 10/30, title + feminine 24/30, masculine alone 30/30).
- COMPLETENESS **ok**.
- FACTUALITY **minor**: T1's evidence says the model "otherwise writes a substantive analysis of the constitutional draft". The non-denial completions are heavily confabulated (Charter dated 1859, 1839, 1876), so "unlocks the knowledge" overstates it.
- TRANSCRIPT-INFERABLE **partly**: the genitive is inferable for a Russian reader; the title dependence is not.
- **OVERALL: valid (medium-high).**

## 12. fresh1 / prompt_008308: NSFW persona refusal names non-consent/CSAM
- **Behaviour:** refusals list non-consensual sexual content among the things the model won't write (CHIVE 18/30, remeasure 17/30).
- **T1:** the "No content restrictions" override line (removed: 7/30; scoped wording: 7/30). T2: it works only inside the long directive stack (short prompt + line: 3/30).
- BEHAVIOUR **minor**: listing hard limits in refusal boilerplate is fairly standard.
- MEASURE **minor**: the rate is unconditional on refusing. Exp 1 and Exp 8 show more bare "Understood" acceptances, so part of the drop is a change in refusal rate, not refusal content. Refusal rate is never reported.
- EDITS **ok** for Exps 1/2/4/8. The T2 arms (Exps 5-7) rewrite the whole prompt.
- INFERENCE **minor**: the T2 interaction is inferred across structurally different prompts. The GT movers are the two whole-prompt rewrites (Exps 5, 6), not the clean T1 edit (37pp, below the 50pp claim-bank cut).
- COMPLETENESS **minor**: which other part of the long stack matters is not decomposed.
- FACTUALITY **ok**.
- TRANSCRIPT-INFERABLE **partly**.
- **OVERALL: usable-with-caveats (medium).**

## 13. fresh4 / prompt_026434: model becomes "Rick" from a persona string in code
- **Behaviour:** after a code review, the bare greeting "مرحبا" gets an in-character reply as ريك (Rick), the persona literal in the user's bot code (CHIVE 25/30, remeasure 24/30).
- **T1:** the model treats the in-code persona string as an instruction to itself. T2: a contentless greeting leaves no review task (a code request gives 0/30).
- BEHAVIOUR **ok/minor**: real data/instruction confusion. But the whole conversation is a *flattened transcript inside one user turn* ending with "User: مرحبا / User: مرحبا / Assistant:". That is exactly the prompt format the bot code builds (`User: ${prompt}` lines), so continuing as the bot is a reasonable reading.
- MEASURE **ok**: the verdicts checked are right.
- EDITS **minor**: Exp 3 (generic persona, 0/30) removes the name, so a Yes is impossible by construction. It also drops the emoji, boasting and Arabic-only instructions. Exps 4/6 also drop the duplicated greeting.
- INFERENCE **minor**: T1 "any in-code instruction is enacted" is not separated from "a vivid named persona is salient".
- COMPLETENESS **major**: the transcript-format alternative (the trailing "User:/Assistant:" lines mimic the bot's own chatHistory format, so the model infers it *is* the bot) is never tested. A clean test: put the final greeting in a real separate user turn without "Assistant:".
- FACTUALITY **ok**.
- TRANSCRIPT-INFERABLE **yes**.
- **OVERALL: usable-with-caveats (medium).**

## 14. fresh2 / prompt_016092: silently re-dating the Enlightenment
- **Behaviour:** while polishing the user's paragraph, the model silently changes "16th to 18th century" to 17th-18th (CHIVE 20/30, remeasure 15/30).
- **T1:** recognising the canonical period (Kant, the epitaph) makes the model regenerate the dates. T2: polishing is taken as licence to fix content, and only an explicit "keep facts" stops it.
- BEHAVIOUR **ok**: a real faithfulness and silent-edit issue.
- MEASURE **minor**: the "flagged vs silent" boundary is subjective. Exp 2 c0 flags the name changes but maybe not the date. Exp 5's "corrects in nearly every response" is not measured by this silent-only classifier. Some completions run past 1,700 characters, but the date sits in the first paragraph.
- EDITS **ok/minor**: Exp 3 (fully fictional) changes the names, the epitaph and the ideology, and there is no canonical date to substitute, so it is partly structural.
- INFERENCE **minor**: T2's second claim (flagging scales with the size of the discrepancy) rests on one qualitative cross-period comparison.
- COMPLETENESS **ok**.
- FACTUALITY **ok**.
- TRANSCRIPT-INFERABLE **yes**.
- **OVERALL: valid (medium-high).**

## 15. fresh2 / prompt_014564: dropping Java packages for a single-file judge
- **Behaviour:** asked for Trip in package A and Holiday in package B, the model delivers one file with no package lines. CHIVE 13/30, **remeasure 22/30**. Trunc 30/30.
- **T1:** the Holiday skeleton lacks `package B;` (adding it: 0/30). T2: the run of earlier single-file exercises in history (no history: 0/30; one exercise: 1-2/30).
- BEHAVIOUR **minor**: inferring a single-file Udemy grader is a reasonable, even helpful, guess.
- MEASURE **major**: the classifier judges the "primary/final code solution", but all 30 baseline completions hit the 500-token cap mid-code (e.g. baseline #0 and #2 end before the final solution). Responses that show both variants are ambiguous. This plausibly explains CHIVE 13/30 vs remeasure 22/30.
- EDITS **minor**: Exps 1/4 near-tautologically hand the model the package line in the stub it must fill. Exps 5/6 change both the amount of history and specific content.
- INFERENCE **minor**: the directions are robust (effects to 0-2/30), but the baseline is uncertain (43% vs 73%). The "accumulated run" claim is not isolated, and the 2x2 interaction is never measured directly. There are no GT movers (max 43pp).
- COMPLETENESS **ok/minor**.
- FACTUALITY **ok**.
- TRANSCRIPT-INFERABLE **yes**.
- **OVERALL: usable-with-caveats (medium-low).**
