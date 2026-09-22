# Explaining WeirdChat behaviors (explain mode)

The rest of this harness plays the AuditBench game: a quirk was planted, find it, get scored.
Explain mode plays a different one. The behavior is **given**, it was never planted — it is in the
released weights — and the question is *why the model does it*. There is no ground truth for that
and this pipeline does not pretend otherwise: a run produces ranked **mechanisms**, each with the
evidence it rests on and the experiment that would test it, and nothing in the run verifies them.

## The data

[WeirdChat](https://weirdchat.transluce.org) (Transluce, v1.0.1) is a catalogue of behaviors that
automated elicitation found in six open-weight models. For **Qwen3.6-27B** it holds 11 behaviors
over 225 patterns and ~25.8k judged rollouts. A *pattern* is a family of user prompts that triggers
one behavior; per pattern the site serves the representative prompt, the judge rubrics, and every
sampled rollout with the judge's verdict on it.

That last part is what makes a pattern explainable. The rollouts of one prompt are sampled
identically, and some show the behavior while others do not — a contrast that is not confounded by
the prompt. The model goes both ways on identical input, so any account of the behavior has to
survive that.

WeirdChat sampled `Qwen/Qwen3.6-27B-FP8` with **no system prompt, temperature 1 and reasoning
disabled**. Anything reading those transcripts has to match it (`weirdchat.data.GENERATION`); the
lens server here serves the bf16 weights of the same model.

Only the public API is used — `GET /api/all-patterns` and `GET /api/behaviors/<entry_id>` — so
there is no dataset download and no parquet dependency. Responses are cached under
`<out_root>/data/`.

## What the investigator gets

* **The brief**: the behavior, the judge's rubric, the prompt, the published match rate, and a
  contrast set of rollouts labelled by whether they show the behavior.
* **The study's own rollouts as conversations** (`w000m`, `w000u`, …), already loaded, so the lens
  can be read on the exact pair the judge labelled rather than only on samples the agent draws.
* **`chat` / `complete`** on the same model (no system prompt by default — deployment parity).
* **`readouts`**: the OLens verbalizer over every position of a conversation at layers 20–60 in
  steps of 4.
* **`finish(mechanisms=[…])`**: each mechanism carries `evidence`, `readout_cells`, `confidence`
  and `would_test_by`. There is no judge and no closed-set stage.

Two things the prompt insists on, because they are the easy ways to be wrong here:

1. The USER and HEADER positions are identical across every rollout of a prompt, so they carry the
   **propensity**, not the outcome; a matched/unmatched difference can only be real from the
   **fork** — the first token where the two replies diverge — onward.
2. A mechanism that only restates the behavior ("it is sycophantic") or names an unobservable
   training cause ("RLHF made it agreeable") is not an answer.

## Running it

```
P=scripts/weirdchat/run_weirdchat.py
S='https://<ws>--auditbench-organism-organism-'          # the lens server prefix

python $P stage=data per_behavior=3 min_rate=0.15         # select + cache the patterns
python $P stage=diagnose server=$S                        # raw lens reads, before any agent
OPENROUTER_API_KEY=sk-or-… python $P stage=agent server=$S auditor=anthropic/claude-opus-5
OPENROUTER_API_KEY=sk-or-… python $P stage=synth,site     # cluster the mechanisms, render
```

`scripts/weirdchat/run_weirdchat.sh` runs the whole thing in a tmux session with a timestamped
log. Every stage resumes: a pattern or run that already has its file is skipped unless
`force=true`. Outputs:

```
<out_root>/patterns/<key>.json              the pattern, its rubric and its contrast rollouts
<out_root>/diag/<key>.json                  OLens over both sides + the fork between them
<out_root>/runs/<key>/<auditor>/seed_N.json the agent run, with the pattern it was given
<out_root>/synth.json                       the mechanisms grouped into themes
<out_root>/site/index.html                  the viewer (static, single file)
```

`<key>` is `<behavior_id>__<group_id>`, e.g. `recommends-drunk-driving__pg0001`.

## Reading the output honestly

* Nothing is scored. A mechanism is a hypothesis with a citation, and the viewer says so on every
  page. The `would_test_by` field is the point: it is what a later pass would measure.
* Check the cited cells. A readout cell that does not say what the agent claims it says is the
  failure mode to look for first (`docs/faults.md` §3).
* A theme that appears once explains one prompt. `synth.json` counts how often a mechanism was
  proposed, which is not evidence that it is right — only that it recurred.
* The rollouts in the contrast set were judged by WeirdChat's judge (Gemma 4 31B). Rollouts the
  agent draws itself are **not** judged by anything.

## Files

| file | role |
|---|---|
| `weirdchat/data.py` | the API client, pattern selection, the contrast set |
| `weirdchat/prompts.py` | the explain-mode system prompt, the brief, the `finish` schema |
| `weirdchat/explain.py` | `ExplainTools` (seeded conversations, mechanisms `finish`) and the run |
| `weirdchat/diagnostics.py` | the non-agentic lens reads and `fork_of` |
| `weirdchat/synth.py` | grouping mechanisms across runs |
| `scripts/weirdchat/run_weirdchat.py` | the driver (`stage=data,diagnose,agent,synth,site`) |
| `scripts/weirdchat/build_site.py` | the viewer |
