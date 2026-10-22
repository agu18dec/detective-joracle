# How CHIVE works, and how valid its incidents are (audit, 2026-10-21)

Scope: the 915 fresh CHIVE incidents (fresh1-4) used in every chive / chive-diag result. Code and outputs:
- `/workspace/agam/chive_scale/analysis/diag/`: `p7_replicate.py`, `p8_census.py`, and `chive_audit/` (hand-audit cards and grades)
- `/workspace/agam/chive_scale/diag/{replicate,census}/`

## 1. How an incident is made
Per batch, from `runs/<b>/config.json` and `status.json`:

| stage | what happens | volume per batch |
|---|---|---|
| 0 prompts | slice of the WildChat + agentic pool (~9% agentic or system-prompt) | 6,000 |
| 1 samples | Qwen3.6-27B, 30 completions each, T=1, **500-token cap**, thinking off | 180,000 (56% truncated) |
| 2 screen | Opus scores "interest" 1-5 (never gives 5), writes the behaviour summary, a yes/no **classifier question** and candidate hypotheses | ~4,600 score ≥3 |
| 3 investigate | an Opus agent runs ~8-15 experiments: edit the conversation, rerun n=30, a **Sonnet grader** applies the classifier question to each rerun. Writes core causes and negative evidence | 400 |
| 4 verify | Opus (ensemble) scores the investigation 1-10 for fair measure, accurate edit descriptions and no confounds | scores pile up at 7-8 |
| build | keep verification ≥8 and measured effect ≥30 pp. A claim bank labels experiments **supported** (≥50 pp) or **inert** (≤15 pp) | ~230 incidents |

Ground truth per incident (what investigator agents are graded against):
- **targets T1..:** the investigator's established causes, ~2.3 per incident;
- **ruled_out:** refuted rivals (all 1,046 rest on the investigator's say-so) and inert edits (1,058 claim-bank items + 247 experiments ≤15 pp);
- **movers:** claim-bank edits with an effect ≥50 pp;
- **judge_reference:** the text the 1-10 judge reads.

So the "truth" is one Opus agent's conclusions from n=30 experiments graded by Sonnet, filtered by an Opus verifier.

## 2. What the incidents are (census, n=909; Sonnet labels, `data/p8_census.csv`)
- **Domain:** creative writing 43%, code 14%, advice/explanation 13%, factual QA 9%, translation 5%, roleplay/NSFW 5%, agentic/tools 4%.
- **Behaviour:** fabrication/hallucination 31%, instruction violation 16%, format/style choice 14%, persona/framing 11%, factual error 9%, refusal/safety 5%.
- **Cause of T1:**
  - salient phrase/token 33%;
  - **model prior/default 26%** (not testable by any conversation edit);
  - ambiguity/missing info 14%;
  - prior turn 14%;
  - explicit instruction wording 9%.
- **The behaviour text (classifier question) names or cues T1's feature in 24%.** Blind-picker check: from the behaviour plus 10 options alone, the truth gets median p = 0.26 (chance 0.10), and ≥0.3 in 42-45% of dev incidents.
- Sonnet calls 84% "interesting". The hand audit (§4) is much harsher: 6/30 not really weird, 14/30 mildly weird. LLM judges are as lenient as the screener.

## 3. Mechanical checks (all 915)
- **The behaviour is real and reproducible.** Our executor re-measured all 912 baselines (n=30, Opus grader): median |Δ| vs CHIVE 6.7 pp, only 2.6% differ by ≥30 pp, and 0.5% lose the behaviour.
- **Graders agree.** Opus re-grading CHIVE's stored completions agrees with Sonnet on **97.3%** of baseline verdicts and 97.9% of mover verdicts (|Δk| ≈ 0.6-0.8 / 30; n=80 / 52). Errors are concentrated in specific incidents (§4), not systematic.
- **Truncation:** 549/915 incident responses (60%) are cut at the 500-token cap. Behaviours late in a response are scored as absent when they're simply not reached.
- **"Inert" verdicts are underpowered.** For 525/1,058 claim-bank inert items (50%), the 95% CI of the effect at n=30 still includes a 30 pp change.
- **Edits:** of the decisive movers, 719 are minor-targeted, 238 large single-factor, **62 mixed and 43 confounded (10%)**.
- **Verifier issues:** every kept incident carries some, 3.6 per incident on average. Keyword buckets (crude):
  - confounded/bundled edit in 65%;
  - missing control or untested alternative in 55%;
  - grader or measure problem in 38%;
  - tautological/leading edit in 20%.

## 4. Hand audit (30 random incidents, read end to end; `chive_audit/cards_{A,B}.md`)
| rubric | ok | minor | major |
|---|---|---|---|
| behaviour is real and weird | 10 | 14 | 6 |
| classifier measures it | 12 | 12 | 6 |
| edits are clean | 11 | 17 | 2 |
| inference follows from the numbers | 6 | 16 | 8 |
| alternatives covered | 4 | 25 | 1 |
| no false "facts" | 20 | 7 | 3 |

**Overall: 8 valid, 21 usable-with-caveats, 1 invalid.** Transcript-inferable T1: 16 yes, 12 partly, 2 no.

**The causal core (T1) is usually right, because decisive effects are big** (often 0 vs ≥15 of 30). The damage is in the **ruled-out labels, the secondary targets and the measure**:
- **Labels that contradict the numbers:**
  - 015731: a "refuted" rival moves the rate 18 → 0/30.
  - 027236: grader errors (17/18 YES verdicts wrong) turned a ~50 pp cause into a "refuted" and an "inert" item.
  - 009425: "supplying X removes it" was filed as a refutation; it is a precondition.
- **Truncation scored as absence.** In 027016, only 12/30 completions reach any dialogue, so "3/30" is really 3/12.
- **Ordinary behaviour labelled weird.** Examples: a broken-GPS gag for "a village in the middle of nowhere"; "US" guessed from "hickory bacon" when asked for a culinary origin.
- **Tautological or leading edits used as evidence.** The edit inserts the answer, e.g. "pthc is CSAM", or tells the model Axel can't hear.
- **"It's my default" targets** that no edit can test (016127 "14 October", 027828 "Thorne").
- **Claim-bank data bugs:** a verbatim replication run filed as "inert"; edit descriptions that contradict the actual diff; requested incidents with zero claims.

## 5. Replication of decisive experiments (our executor, exact CHIVE messages; `p7_replicate.py`)
80 random incidents (seed 0). We re-ran CHIVE's exact edited messages with our executor: Qwen3.6-27B, T=1, 500 tokens, Opus grader with CHIVE's grader prompt.

| check | result |
|---|---|
| baseline reproduces (|ours − CHIVE| < 30 pp) | 75/80 (median |Δ| 8.3 pp) |
| **decisive mover still moves ≥30 pp, same direction** | **49/52 (94%)**; median effect 53 pp vs CHIVE 60 pp |
| mover by edit type (≥30 pp) | minor-targeted 41/43, large single-factor 6/7, mixed 1/1, confounded 1/1 |
| **claim-bank "inert" edit stays within 15 pp (n=60)** | **37/61 (61%)**; median |effect| 12 pp |
| "inert" edit actually moves ≥30 pp (label wrong) | 6/61 (10%) |
| Opus re-grade of CHIVE's own completions, per-verdict agreement | 97.3% baseline (|Δk| 0.8/30, n=80); 97.9% movers (n=52) |

**Takeaway:** the positive evidence (T1 movers) is solid. The negative evidence (inert / refuted labels) is not: ~40% of inert labels don't hold within ±15 pp, and 1 in 10 is clearly wrong.

## 6. What this means for the reader evals
- The eval measures something real: behaviours reproduce, decisive causes replicate, graders agree.
- **But it is a weak test of reading internals.**
  - T1 is transcript-inferable in ~90% of audited incidents (16 yes + 12 partly of 30).
  - 26% of causes are "model priors" that no reader or edit can establish.
  - 24% of behaviour texts cue the cause.

  This is why a transcript-only top-5 contains the truth 97% of the time, and why readers can't separate on pass@1 (README §4).
- **Rival-based scores are noisy.** Half the inert labels are underpowered, and refuted rivals rest on the investigator's say-so (~1 in 4-5 audited incidents has a refuted or inert label that the numbers contradict; in replication, 39% of inert labels fail at n=60 and 10% move ≥30 pp). The "refuted rivals asserted" metric (readers −8 pp) partly rewards matching CHIVE's labels rather than the truth.
- **Judge v3 non-obvious points** are mostly experiment outcomes (~75-80%, earlier audit), consistent with the above.

## 7. What to do about it
Cheap fixes on the existing 915:
1. **Gold subset.** Keep incidents whose T1 mover replicates with our executor (§5); whose behaviour isn't cued by the classifier question (blind-picker p < 0.2); whose T1 is not a "model prior"; and whose responses aren't truncated before the behaviour.
2. **Re-power the rival labels.** Re-run every inert and refuted edit at n=60-90 and keep only labels with a CI-based equivalence test (difference within ±15 pp). Reclassify "supplying X removes it" as a precondition. Drop claim-bank items that edit the same factor as a target or have zero diff.
3. **Re-grade decisive and ruled-out experiments with a second grader.** Flag disagreements of >3/30.
4. **Score truncation as "not reached"** (a third verdict) rather than NO.

Pipeline changes for new incidents:
1. **Raise the token cap**, or reject behaviours that depend on late content.
2. **Neutral classifier questions** that describe the behaviour without its suspected cause (check with a blind picker at screening).
3. **Mechanical diff checks** on every edit (reject undeclared whitespace or format changes; describe movers from the true diff).
4. **Mark leading/tautological edits as non-evidential.**
5. **A stricter weirdness screen:** "would a competent assistant do this given the full context, including its own earlier turns?"
6. **Select for internals-relevant incidents:** prefer behaviours whose cause the text-only screener could NOT guess. Measure this with a text-only picker at screening, and keep incidents where it fails but experiments succeed. This is the axis on which readers can actually be compared.
