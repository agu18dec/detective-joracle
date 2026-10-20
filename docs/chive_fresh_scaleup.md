# CHIVE fresh-incident scale-up

The held-out test (`chive_harness_climb.md`) has 204 incidents, so its arm differences are mostly
within noise. The scale-up runs the **unmodified CHIVE pipeline on fresh WildChat prompts**,
converts every verified incident into the harness dataset layout, builds readout banks for every
reader, and runs the frozen harness plus several judges on all of them. Code lives in
`scripts/chive_fresh/`. The harness itself is unchanged apart from four env switches
(`CHIVE_EXP_N`, `CHIVE_BASELINES`, `POINTS_ARMS` / `POINTS_DRAW_GLOB`, and the keyless Modal
Dict token), and with none of them set it behaves exactly as before.

Batches `fresh1`-`fresh4`: 6000 prompts each from the WildChat + agentic pool, rows
8000-31999. The pilot used rows 0-1999 and run2 used 2000-7999, so there is no overlap. Each
batch screened down to 400 investigations, and 237 / 232 / 226 / 220 incidents (915 in total)
passed the original 300's gates. The judges, grader and thresholds match run2's frozen config.

## Layout and configuration

| path | what |
|---|---|
| `scripts/chive_fresh/env.sh` | source first: credentials from `/dev/shm/compute.env`, `MODAL_ENVIRONMENT=agam`, `unset ANTHROPIC_API_KEY`, every path below |
| `scripts/chive_fresh/paths.py` | the same locations for the Python scripts |
| `run_chive_keyless.py`, `anthropic_oauth.py`, `launch_fresh.sh` | CHIVE stages with keyless Claude plus the runtime patches |
| `build_incidents.py`, `build.sh`, `build_and_bank.sh` | CHIVE run dir -> harness dataset (-> banks) |
| `banks/` | `readers_modal.py` (Modal app `chive-fresh-banks`), `bank_common.py` (pinned revisions, bank format), `fetch_banks.sh`, `fetch_ids.sh`, `compare_banks.py` |
| `run_harness.sh`, `run_points.sh`, `baselines.py`, `wait_token.py` | the harness, point eval v1, re-measured baselines, wait for a working token |
| `analysis/` | `judge2.py` (+ `judge2_report.py`), `points_report.py`, `readers_paired.py`, `strata.py`, `shuffle_ctrl.py`, `make_k1.py`, `make_layers.py`, `hverify.py`, `hdossier.py` |
| `claims/` | claim-level AUROC on the fresh claim banks |
| `jobs/` | example jobs: `arms.sh` (all arm specs), `run_arm.sh`, `exp1_n30.sh`, `shuffled_controls.sh`, `olens_single.sh`, `derived_banks.sh`, `judge2_all.sh`, `confidence.sh` |

Every path is an env var whose default is the pod layout the runs used:

| var | default |
|---|---|
| `CHIVE_SCALE` | `/workspace/agam/chive_scale` |
| `CHIVE_DATASETS` | `$CHIVE_SCALE/datasets` |
| `CHIVE_BANKS` | `$CHIVE_SCALE/banks` |
| `CHIVE_RUNS` | `$CHIVE_SCALE/harness_runs` |
| `CHIVE_POINTS` | `$CHIVE_SCALE/points` |
| `CHIVE_CONFIDENCE` | `$CHIVE_SCALE/confidence` |
| `CHIVE_CLAIMS_RUNS` | `$CHIVE_SCALE/claims_runs` |
| `CHIVE_RUN_DIRS` | `$CHIVE_SCALE/runs` |
| `CHIVE_DIR` / `CHIVE_PY` | CHIVE checkout (`/workspace/agam/chive`, `adamkarvonen/chive`) and its venv python |
| `GW_DIR` | global-workspace checkout; `judge_auth.py` is read from it |
| `CHIVE_TARGET_URL` | the vLLM endpoint serving the target, Qwen3.6-27B (`chive-qwen-vllm` app) |
| `HARNESS_DIR` / `HARNESS_PY` | this repo and its `.venv` python |
| `CHIVE_LOGS` | `/workspace/agam/logs/chive_fresh` |
| `GW_SRC` | global-workspace `src/` uploaded into the bank containers (default: the vendored snapshot, see Banks) |

## Pipeline

### 1. CHIVE stages (`launch_fresh.sh <run>`)

`run_chive_keyless.py chive.pipeline.run_pipeline --config $CHIVE_RUN_DIRS/<run>/config.json`
runs CHIVE as is:

1. sample 30 Qwen completions per prompt on the vLLM target;
2. screen them with the Opus judge;
3. investigate with Opus, which runs counterfactual experiments graded by Sonnet 5;
4. verify.

CHIVE's files are never edited. Four in-process patches make it run on the pod today:

- **Keyless OAuth.** `anthropic_oauth.install()` replaces `anthropic.Anthropic` /
  `AsyncAnthropic`. The replacements send the token from the Modal Dict `anthropic-oauth` (env
  `agam`, read via global-workspace `judge_auth.read_token()`) as a bearer token, with the
  `oauth-2025-04-20` beta flag. They re-read the Dict every 5 min, so a `push-judge-token` re-push
  reaches a running pipeline. There are no API keys, and `ANTHROPIC_API_KEY*` is popped from the
  environment.
- **Modal 303 follow-redirects.** Modal web endpoints answer requests that run longer than about
  150 s with a 303 to a result URL. httpx does not follow it by default, so stage 1 died with
  `APIStatusError 303`. The patch makes `httpx.AsyncClient` default to `follow_redirects=True`.
- **Opus thinking -> adaptive effort.** claude-opus-5 rejects `thinking.type.enabled` with a 400.
  CHIVE's `thinking_budget` is mapped to adaptive thinking with an effort level:
  - budget <= 2048: `low`
  - budget <= 4096: `medium`
  - larger: `high`

  Mapping the grader's 2048 to `medium` overran its 8192 `max_tokens` and quarantined about 4% of
  investigations, which is why 2048 maps to `low`.
- **Stringified tool args.** Sonnet 5 sometimes returns a nested array argument as a JSON string,
  for example `verdicts='{"verdicts":[...]}'`. That failed pydantic and quarantined about 6% of
  fresh1's investigations. Such strings are now decoded before validation, and a same-key wrapper
  object is unwrapped.

### 2. Harness dataset (`build.sh <batch>` -> `$CHIVE_DATASETS/<batch>`)

`build_incidents.py` reproduces the original 300's `agu18dec/chive-forensics` rows. It was
validated byte-identical on those rows; its docstring has the details.

Gates:

- verification score >= 8;
- a measured effect >= 30 pp;
- at least one supported or partially supported core cause;
- segmentable with the Qwen3.6-27B template, at <= 7500 tokens.

Outputs:

- `data/{incidents,tokens,ground_truth}.jsonl`.
- `splits/all.txt`.
- `work/binary_counterfactual_questions.jsonl`: the claim bank, built with CHIVE's own
  `build_claims` (selector claude-opus-5, writer claude-sonnet-5) through keyless OAuth. It
  supplies the movers and inert edits for the ground truth and for the claims eval.

Tokens need `transformers==5.11.0`, which is pinned inline, so run the script with
`uv run --script`. The LLM steps run in a subprocess under `$CHIVE_PY`. The build is resumable
and idempotent.

### 3. Banks (`build_and_bank.sh <batch>`, then `banks/fetch_banks.sh <bank>`)

`banks/readers_modal.py` captures activations from the dataset's own token ids, so the
harness's token-id check passes. It writes `<bank>/<pid>.json.gz` to the Modal volume
`chive-fresh-banks`, keeping HF snapshots and merged checkpoints on `chive-fresh-hf`. Every HF
revision is pinned in `bank_common.py`. Reruns resume at (incident, layer) granularity.

GPU list: `CHIVE_BANKS_GPU`. `GW_SRC` defaults to the vendored snapshot the fresh banks were
built with, `scripts/chive_fresh/banks/vendor/gw_src/src`. Its README records the source:
`agam/chive-hard-cases@515f597`, plus `karvonen.py` from `camila/mlnla-viewer-am@9966287`. It
differs from global-workspace main in `olens_suite/runner.py`, `olens_suite/karvonen.py` and
`ola/verbalizer.py`, so keep using it when you rebuild banks.

| bank | reader | settings |
|---|---|---|
| `s3d_rl600_all` | OLens RL `s3d_rl600` | concepts_raw prompt, unit transform with alpha 16000, k=4, 128 new tokens, T=0.8, top-p 0.95, every position, L20-60 step 4 (vLLM, prompt embeds) |
| `s3_single_rl` | single-readout OLens: GRPO from the ptag parent, no distill phase, iter 500 | continuation_raw prompt, `scaled` transform (122.32 x h), k=4, same ladder |
| `nla_k4` | NLA-RL (av_rl_adapters iter 400) | L42, k=4, Karvonen L1 norm-match ADD at AV block 1 (vLLM + patched vllm-lens) |
| `mlnla_k4_chive` | multi-layer NLA rl600 | k=4, L20-60 step 4; the actor prompt names the layer |
| `jlens_all` | J-lens n1000 (wikitext) | cosine readout, raw top-10 at every position and every layer L0-62 |
| `olens_k1`, `mlnla_k1` | derived (`analysis/make_k1.py`) | first sample of every cell of `s3d_rl600_all` / `mlnla_k4_chive` |
| `jlens_L20-60` | derived (`analysis/make_layers.py`) | `jlens_all` restricted to L20-60 step 4 |

`jobs/derived_banks.sh` builds the three derived banks locally. `compare_banks.py` checks a
rebuilt record against a reference one.

### 4. Harness runs (`run_harness.sh <batch> <stage> "<arms>"`)

The frozen harness (`run_chive.py`) runs with variants `textfirst` + `absence`, 5 draws, read-only.
Run `stage=agent` twice (the second pass redoes records truncated by 429s or API errors), then
`stage=judge`. `jobs/run_arm.sh` does exactly this, one batch at a time. The arm specs are in
`jobs/arms.sh`:

| arm | spec | question it answers |
|---|---|---|
| `transcript` | `transcript` | the baseline: no readouts |
| `olens_rl_B` | `s3d_rl600_all:olens` | OLens, 4 samples per cell |
| `olens_k1` | `olens_k1:olens` | does OLens need 4 samples per cell? |
| `olens_single` | `s3_single_rl:olens` | single-readout OLens (no multi-bullet distill) |
| `jlens_B` | `jlens_all:jlens` | J-lens, all 63 layers |
| `jlens_L2060` | `jlens_L20-60:jlens` | J-lens at OLens's layers only |
| `mlnla_B` | `mlnla_k4_chive:mlnla` | multi-layer NLA, k=4 |
| `mlnla_k1` | `mlnla_k1:mlnla` | multi-layer NLA, k=1 (matched readout volume) |
| `olens_shufc`, `jlens_shufc`, `mlnla_k1_shufc` | `...:shuffled` | **shuffled controls:** the same instrument reading ANOTHER incident (`splits/shuffle_ctrl.txt`, 38/38/37/37 incidents); `jobs/shuffled_controls.sh`, `analysis/shuffle_ctrl.py` |

Records go to `$CHIVE_RUNS/<batch>/<arm>/<pid>.d<draw>.json`. Draws d0-d4 are the base draws, and
some arms have extra draws for other analyses, so every analysis globs `*.d[0-4].json`. Records
with `stopped_by` starting `error` or `salvaged` are excluded everywhere and redone on the next
agent pass.

### 5. Judges

| judge | code | output | what it measures |
|---|---|---|---|
| **old pass judge** | `run_chive.py stage=judge` | `judge.score` in each record | 3 Opus replicates score the answer 1-10 against CHIVE's reference; pass = >= 7. `analysis/readers_paired.py` (paired pass@1 / pass@5), `analysis/strata.py` (difficulty strata defined by the no-tools plain answers, not by any compared arm), `analysis/shuffle_ctrl.py` |
| **point eval v1** | `scripts/points_chive.py` via `run_points.sh` (`POINTS_ARMS` = every arm, `POINTS_DRAW_GLOB='*.d[0-4].json'`) | `$CHIVE_POINTS/<batch>/` | CHIVE's causes are split into atomic points, plus a primary tag. A point is obvious when 2 of 3 no-tools answers state it. Each answer is graded per point (1-10) and per claim (supported / contradicted), with rivals rejected or asserted. Report: `analysis/points_report.py` (non-obvious and primary recall, disproved claims, precision, point wins and losses vs transcript, paired CIs) |
| **judge v2, hyp-only (primary) / full** | `analysis/judge2.py [--hyp-only] --no-predict` | `$CHIVE_SCALE/judge2_hyp/`, `judge2/` | arm-blind: Opus restates the answer as at most 8 plain causal bullets, stripped of all readout, layer and tool talk. A fresh grader labels a shuffled fixed list (CHIVE's points plus refuted rivals) ASSERTED / DENIED / ABSENT. `--hyp-only` grades only the final `hypothesis` field; full grading adds evidence and rival. Report: `analysis/judge2_report.py` (overall and least-confident third). Stage 3 (edit-prediction AUROC) exists but was skipped |
| **claims AUROC** | `scripts/claims_chive.py` via `claims/run_claims.sh` | `$CHIVE_CLAIMS_RUNS/<batch>/focused/` | CHIVE's binary claim eval on the fresh claim banks (balanced 50/50 by `claims/build_fresh_claims.py`): one focused episode per claim with P(true), Opus 5 at max effort, 1 draw. `claims/report_fresh.py` reports pooled and within-incident AUROC, Brier and acc@0.5, with paired bootstrap vs transcript and concreteness >= 3 / edit_cleanliness strata (`claims/concreteness_fresh.py` scores concreteness) |

The pre-registered least-confident third uses Opus's no-tools `top_p` (`jobs/confidence.sh` ->
`$CHIVE_CONFIDENCE/<batch>/confidence.json`).

### 6. Budgeted counterfactual (`jobs/exp1_n30.sh`)

Agam confirmed this protocol on 2026-10-16:

- experiment budget 1, 2 draws, eval `<batch>_exp1n30`;
- arms transcript, olens_rl_B, jlens_B, mlnla_k1, olens_k1, olens_single, jlens_L2060;
- stages: agent, then the old pass judge, then `stage=concluder` (a fresh agent sees only the
  experiment and its result).

Two env switches make the experiment informative:

- `CHIVE_EXP_N=30`: one experiment draws n=30 target samples instead of 12, so a single test
  has the resolution of CHIVE's own baseline.
- `CHIVE_BASELINES=$CHIVE_SCALE/exp1/baselines.json`: the result is compared against the
  unedited conversation **re-measured with the same executor** (`baselines.py`: 30 samples from
  `CHIVE_TARGET_URL`, graded by the harness's Opus grader). CHIVE's own `k/n` came from a
  different sampler and grader and can differ a lot (e.g. 21/30 in CHIVE vs 8/30 re-measured).
  Without this, an agent would read a sampler or grader difference as an effect. Incidents
  missing from the file keep the CHIVE baseline.

The job waits until the baselines cover all but at most 2 of a batch's incidents (912 of 915
were measured).

## Rate limits and tokens

- **Opus input tokens are limited org-wide to 10M per minute.** That limit is shared with every
  other job in the org, not just this pipeline.
  - One harness process at `CONC=24` (the `run_harness.sh` default) is fine on its own.
  - With several arms or batches running at once, drop each to `CONC=12-16` and run one batch at a
    time per arm, as `jobs/run_arm.sh` does. The runs used 12 for olens_single, 16 for the new
    arms and 20 for the shuffled controls.
  - The bulk Opus users are the harness's agent stage, point eval v1 (`CONC` default 48) and
    judge v2 (12 threads).
- **429s** truncate episodes, recorded as `stopped_by` `salvaged...` or `error...`. They are
  excluded from every analysis and redone by the next agent pass, so always run agent twice
  before the judge.
- **401s** mean the OAuth token expired.
  1. Stop and ask Agam to run `push-judge-token` on the Mac.
  2. `wait_token.py` blocks until a newer token works (an auto-push can already be expired); chain
     the resume after it.
  3. Check the token's age (`judge_auth.read_token()`, `pushed_at`) before a long judge run.

  Never export `ANTHROPIC_API_KEY`: a set key wins over the keyless token.

## Reproduce

```
source scripts/chive_fresh/env.sh            # creds (RAM), paths, MODAL_ENVIRONMENT=agam
D=scripts/chive_fresh

# 1-3: CHIVE run (config.json in $CHIVE_RUN_DIRS/fresh1), dataset, banks
$D/launch_fresh.sh fresh1
$D/build_and_bank.sh fresh1
(cd $D/banks && ./fetch_banks.sh s3d_rl600_all jlens_all nla_k4 mlnla_k4_chive)
$D/jobs/derived_banks.sh                      # olens_k1, mlnla_k1, jlens_L20-60
$D/jobs/olens_single.sh fresh1                # s3_single_rl bank + its harness arm

# 4: harness arms (agent x2 + old pass judge), shuffled controls
source $D/jobs/arms.sh
$D/jobs/run_arm.sh "$(arms_list "$ARM_TRANSCRIPT" "$ARM_OLENS" "$ARM_JLENS" "$ARM_MLNLA")" fresh1
CONC=16 $D/jobs/run_arm.sh "$(arms_list "$ARM_OLENS_K1" "$ARM_JLENS_L2060" "$ARM_MLNLA_K1")" fresh1
$D/jobs/shuffled_controls.sh fresh1

# 5: judges
$D/jobs/confidence.sh fresh1
$D/jobs/judge2_all.sh fresh1                  # point eval v1, then judge v2 hyp-only + full
$HARNESS_PY $D/claims/build_fresh_claims.py fresh1
CONC=12 $D/claims/run_claims.sh fresh1 transcript
CONC=12 $D/claims/run_claims.sh fresh1 "$ARM_OLENS"

# 6: budgeted counterfactual
$HARNESS_PY $D/baselines.py --batches fresh1
$D/jobs/exp1_n30.sh fresh1

# reports
A=$D/analysis
python3 $A/readers_paired.py fresh1 fresh2 fresh3 fresh4
python3 $A/strata.py; python3 $A/shuffle_ctrl.py
python3 $A/points_report.py fresh1 fresh2 fresh3 fresh4
python3 $A/judge2_report.py judge2_hyp; python3 $A/judge2_report.py judge2
$HARNESS_PY $D/claims/report_fresh.py
python3 $A/hdossier.py fresh1 <pid>           # one incident: every arm's answers and tools
python3 $A/hverify.py fresh1 <pid> <pos> '<regex>' --others   # recount readouts from the bank
```

`passk_chive.py eval=<batch>_exp1n30 ...` and `run_chive.py stage=report` work on the fresh evals
just as on `climb_test`. Every stage and script resumes, so it is safe to rerun.

## Final results (2026-10-20)

Copied from /workspace/agam/experiments/chive/SUMMARY.md. HF (private): agu18dec/chive-fresh-readers-eval @ f359a8f5b80ba80b98d467a7a6837bf4377b5ea8.
### Setup
- 915 fresh CHIVE incidents (4 batches from the CHIVE pipeline; same structure as climb_test204: ~2.3 true causes,
  ~2.6 refuted rivals per incident).
- Same frozen detective-joracle harness for every arm: Opus 5 investigator (effort max), textfirst + absence
  variants, Opus 1-10 judge (3 replicates, pass = mean ≥ 7) against the ground truth.
- Arms: transcript only; OLens ×4 (4-bullet RL600, k=4 samples per cell); multi-layer NLA ×4 (camilablank rl600,
  k=4); J-lens L20-60; plus single-bullet OLens ×1/2/4/8/16. Readouts at layers 20-60 step 4, every token.

### Results (915 incidents, 10 draws each)
| arm | pass@1 | Δ vs transcript | pass@5 |
|---|---|---|---|
| transcript | 64.7 | – | 77.5 |
| OLens ×4 | 68.3 | +3.6 [+1.8, +5.6] | 84.5 |
| J-lens L20-60 | 67.8 | +3.1 [+1.3, +4.9] | 82.1 |
| multi-layer NLA ×4 | 69.8 | +5.1 [+3.3, +6.9] | 85.0 |
OLens − NLA paired (10 draws): pass@1 −1.5 [−2.8, −0.1]; pass@5 −0.5 [−2.0, +1.0]; hardest third −0.6 [−3.2, +2.1];
least-confident third −3.0 [−5.9, −0.1].

### Is it a bug? No
- Judge: true cause passes 99%, refuted rival / other-incident / generic 0%; ~1% verdict flips on re-judge;
  re-scores the original NLA run 70.0 → 70.0 (`judge_audit.png`).
- Readouts: banks complete (100% coverage, k samples per cell), token-aligned (harness checks token ids), served
  in full to the agent (no trimming); shuffled readouts drop every reader to transcript level (`shuffled_control.png`).
- Replication of climb_test204 (Agam's held-out): identical prompts and served readouts; rebuilt banks = original
  banks within noise. With 15 draws pooled, OLens − NLA is +1.5 [−1.3, +4.2] overall and +6.0 [+0.1, +12.5] on the
  least-confident third. The original NLA run was a low draw (and kept 47 rate-limited episodes)
  (`climb_test204_pooled.png`, replicate_runs/REPORT.md).
- On the fresh 915 the least-confident third reverses: OLens − NLA −3.6 [−6.5, −0.5] at 10 draws (n=240).

### Why every arm is close
- ~50% of incidents are solved by transcript alone in every draw; ~22% fail every draw.
- About a third of incidents are coin flips for the Opus investigator (draw-to-draw variance, equal for OLens and NLA:
  `nla_variance.png`); 76% of outcomes are fixed by the transcript-only hypothesis the agent commits before readouts.
- Readouts add +7-8 pp within a run (fail→pass 16%, pass→fail 8%), the same for OLens and NLA; J-lens +2.

### What OLens and NLA readouts are like
- NLA: ~45% of samples open with a genre/register label ("Genre: alternate-history wiki entry…"). Bland, rarely lures.
- OLens: specific content (entities, dates, constraints). More informative but sometimes lures the agent to a refuted
  rival (8 of 21 NLA-only wins are OLens lures; analysis/nla_vs_olens.md).

### Harness changes tried to make OLens win (dev: 240 incidents × 3 draws)
| change | OLens | NLA |
|---|---|---|
| recurrence counts in search | −1.5 | −2.4 |
| LLM readout filter | −3.6 | −0.8 |
| commit 3 candidates, readouts choose | −2.4 | −2.6 |
| consensus pick across draws | +1.7 | +2.8 |
| antilure prompt (held-out fresh3+4) | +1.0 | −0.2 |
| more samples: single-bullet ×16 vs ×1 | +2.5 (ties NLA) | – |

### Point eval (judge v2, final hypothesis only; transcript/OLens/NLA at 10 draws, J-lens L0-62 at 5; n=911)
| metric | transcript | OLens ×4 | NLA ×4 | OLens − tr | NLA − tr |
|---|---|---|---|---|---|
| true points asserted | 28.9 | 28.0 | 28.5 | −0.9 [−1.6, −0.1] | −0.4 [−1.1, +0.3] |
| non-obvious points asserted | 14.0 | 13.7 | 14.0 | −0.3 [−1.1, +0.4] | −0.1 [−0.7, +0.7] |
| refuted rivals asserted | 31.4 | 23.3 | 23.9 | −8.1 [−10.1, −6.3] | −7.5 [−9.4, −5.6] |
Readouts do not raise recall of true (or non-obvious) points; they cut assertion of refuted rivals by ~8 pp, equally for
OLens and NLA (−10 pp on the least-confident third). That is where the free-form pass gain comes from.

### MCQ eval (10 candidate causes, 1 true; dev incidents passing the leak check, ~3 draws)
| version | n | transcript acc | OLens ×4 | NLA ×4 | J-lens | single ×16 | OLens − NLA acc | OLens − NLA p(truth) |
|---|---|---|---|---|---|---|---|---|
| v2 (blind picker 43%) | 115 | 58.3 | 60.3 | 55.4 | 57.4 | 59.4 | +4.9 [−1.2, +11.0] | +2.9 [+0.9, +4.9] |
| v3 hard (blind 26%) | 176 | 58.7 | 57.8 | 58.3 | 55.9 | 58.1 | −0.6 [−5.1, +4.0] | +0.4 [−1.0, +1.8] |
No reader raises MCQ accuracy over transcript. Readers cut picks of refuted rivals by 5-8 pp on v2 (where more options are
CHIVE refuted causes); ~1 pp on v3. v2's OLens > NLA p(truth) edge does not replicate on the larger, harder v3.
Graphs: `mcq_v2_bars.png`, `mcq_v3hard_bars.png`.

### Conclusion
In this harness OLens ≈ multi-layer NLA; both beat transcript by ~4 pp and J-lens by ~1-2 pp. OLens's edge on the
least-confident third of climb_test204 does not generalise to fresh incidents. More readout samples is the only
lever that helped OLens. MCQ confirms it: no reader beats transcript at picking the true cause among 10, and OLens ≈ NLA (v3).

### Harness changes in this commit
- `run_chive.py`: every record write is atomic (temp file + rename; unique temp name per call). The judge stage rescans every arm in an eval dir, so overlapping judge processes previously could interleave writes and corrupt a record (3 of 122,371 records were hit).
- Variants (all off by default, prompt byte-identical without them): `recur` (search reports k/n per position, layers, neighbours, base rate; RECURRING vs SINGLE/WEAK), `llmfilter` (Sonnet condenses inspect output; falls back to raw), `multicand` (commit 3 ranked candidates, readouts discriminate), `antilure` (weigh single vivid readouts). Dev results are in the table above: none beats NLA.
