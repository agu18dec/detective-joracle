# CHIVE forensics evals

An investigator agent (Claude Opus 5) gets one real conversation with Qwen3.6-27B, a behaviour
the model shows in `k` of 30 samples, and ONE method for looking inside the model. It has to say
*why* the model behaves this way. A judge scores the answer against the cause that CHIVE
established by counterfactual experiments. Methods: the transcript alone, OLens, NLA, J-lens.

Code: `src/detective_joracle/chive/` (data, readers, tools, prompts, executor, judge) and
`scripts/*chive*.py`. Protocol and results: `chive_harness_climb.md` (the harness climb and
held-out test) and `chive_preregistration.md` (confidence strata and controls).

## Data

The 300 incidents are the HF dataset
[`agu18dec/chive-forensics`](https://huggingface.co/datasets/agu18dec/chive-forensics).
`detective_joracle.chive.data.load_incidents(ids)` downloads `incidents`, `tokens` and
`ground_truth` once into `~/.cache/detective_joracle/` and joins them on `prompt_id`. The splits
are in `data/chive/splits/`: `climb_dev96` / `climb_test204` (the harness climb), `hard92`, and
`deep_read19` (incidents read in depth while the harness was designed; leave them out of
held-out claims). The candidate sets are in `data/chive/choices/`.

**Readouts.** A reader arm reads a precomputed readout file per incident
(`<bank>/<prompt_id>.json.gz`: `token_ids` plus `cells[pos][layer] = [samples]`). These files
are not in the dataset. They come from `scripts/chive_bank/readers_modal.py` in the
global-workspace repo, at these settings: OLens RL (`s3d_rl600`) 4 samples per cell, every
position, layers 20-60 step 4; NLA 4 samples per cell at L42; J-lens top-10 tokens at every
layer. The loader checks token ids against the incident and refuses a mismatch.

## Setup

```
uv venv --python 3.12 .venv && uv pip install -e ".[dev,plot]"
```

- **Claude:** keyless OAuth through the opsec tunnel (`chive/llm.py`; socket
  `OP_REMOTE_SOCK`, default `/tmp/.opsec-agam.sock`; `ant auth login` on the Mac). Unset
  `ANTHROPIC_API_KEY` first: if it is set, even empty, it shadows the token.
- **Target, for experiments only** (budgets > 0): Qwen3.6-27B on vLLM,
  `modal deploy scripts/serve_qwen_vllm_modal.py`, then pass its URL as `target=`.
  Read-only runs need no GPU.

## The harness

Every arm gets the same scaffold: `show_tokens`, `notebook`, `submit_hypothesis`, and for
reader arms `search` (regex over the readouts) and `inspect` (one position, all layers, with
neighbours). The frozen harness adds two prompt rules, `variants=['textfirst','absence']`:

- **textfirst:** the readout tools stay locked until the agent commits a transcript-only
  hypothesis (`commit_text_hypothesis`). It overturns that hypothesis only on contrary
  evidence, so the readouts cannot distract it from what it would get right anyway.
- **absence:** some causes are absences (a missing instruction, version or error text), and
  readouts cannot show an absence.

Arms are `transcript` or `NAME=BANK_DIR:KIND` with `KIND` one of `olens | nla | jlens`.

## Reproduce the held-out test

```
O=runs/chive; B=<dir with the readout banks>
IDS=data/chive/splits/climb_test204.txt
P=".venv/bin/python scripts/run_chive.py"

# 204 incidents x 5 draws, read-only, every method on the frozen harness
$P stage=agent eval=climb_test ids=$IDS out=$O draws=5 "variants=['textfirst','absence']" \
   "arms=['olens_rl_B=$B/s3d_rl600_all:olens','nla_B=$B/nla_k4:nla','jlens_B=$B/jlens_all:jlens','transcript']"
$P stage=judge eval=climb_test ids=$IDS out=$O          # 3 Opus replicates, pass >= 7
.venv/bin/python scripts/passk_chive.py eval=climb_test out=$O ids=$IDS reference=olens_rl_B
.venv/bin/python scripts/plot_climb.py out=$O plots=plots/chive
```

Each stage resumes: a rerun skips finished records and redoes errors, refusals and truncated
episodes. `stage=report` prints the health table (it must be all zeros before quoting a number),
pass@1, paired differences against transcript, and keep/rescue.

Other entrypoints, each documented in its header:

| script | what |
|---|---|
| `run_chive.py stage=concluder` | two-agent split: a fresh agent sees only A's experiments and their results |
| `run_chive.py choices=<jsonl>` | candidate selection: pick the true cause among options |
| `confidence_chive.py` | Opus's no-tools confidence per incident (the uncertainty strata) |
| `strata_chive.py` | the pre-registered comparisons |
| `arbiter_chive.py` | a fresh model picks between two final answers |
| `analyze_chive.py` | wins, regressions and failure labels, with quote checks |
| `climb_chive.py` | the dev A/B table of harness variants |
| `build_tested_choices.py` | candidate sets from CHIVE's tested alternatives |
| `plot_chive.py` | figures for the read-only, one-experiment and candidate evals |

## Results (held-out, 204 incidents x 5 draws)

| arm | pass@1 | pass@5 |
|---|---|---|
| OLens + harness | 73.1% | 89.7% |
| NLA + harness | 71.7% | 87.3% |
| J-lens + harness | 72.0% | 85.8% |
| transcript only | 70.1% | 82.8% |

OLens leads on the least-confident third (66.5% vs 58.2% transcript, +8.2 pp, paired 95% CI
excludes 0) and the hard stratum. On all incidents the gaps are within noise. Details and
controls: `chive_harness_climb.md`.
