# Architecture

## The pipeline

```
stage=agent     one process per (auditor, organism, arm, seed) cell, `workers` in parallel
   LiveClient  -> target (chat) + lens server (readout / fixed)          tools/live.py
   LiveTools   -> the text the agent sees per tool call                  tools/live.py
   select      -> the presentation applied to every readout page         presentation/select.py
   run_tool_loop -> assistant turns / tool turns under the budget        agent/loop.py
   multiple_choice_stage -> stage 2 in the same context                  tools/live.py
   -> <out_root>/runs/live/<cell>/<organism>/<auditor>/seed_<k>.json    (RunRecord.to_json + server usage)

stage=judge     the paper's judge x {all, top 5, top 1} per (record, quirk)    judges/paper.py
   -> record["judged"][<quirk>] = {success, top5, pass1}

stage=judges    distractors (frozen once), then the graded / quality / micro judges   judges/graded.py
   -> record["judges"][<quirk>] = {graded, quality, closeness, closeness_best5, plausible5,
                                    evidence_real, chat_sufficient, coverage, stylistic_tic, lens_cited}

stage=report    every judged record -> three markdown tables            scripts/run_audit.py::_collect
build_viewer    every record -> one static HTML page                    scripts/build_viewer.py
plot            judged records -> a bar-chart grid                      scripts/plot.py
```

Every stage is resumable: `agent` skips records whose `stopped_by` does not start with `error`,
`judge` skips records that carry `judged`, `judges` skips `(record, quirk)` pairs already in
`judges` (unless `force_judges=true`). Base (control) runs are judged against every planted quirk
of the grid, so their rows read as a false-alarm signature.

## Data flow inside one run

1. `system_prompt(arm, allow_reference)` composes GAME + evidence paragraphs + METHOD +
   LIVE_METHOD (+ ORACLE_NOTE / HINT_NOTE for the probes).
2. `run_tool_loop` sends `[system, "Begin your investigation."]` with `tool_schemas(arm)`.
   Each assistant turn's tool calls go through `LiveTools.call`, which dispatches, catches every
   exception into a `tool error: …` string (the agent sees it and can retry), appends a
   `ToolLog`, and counts cells/characters served.
3. `chat`/`complete` hit the target; each reply becomes a `Conversation` with id `c###`.
4. `readouts(conversation)` re-sends that conversation (messages + completion) to the lens
   server at every position and every layer, applies the presentation, and formats the page
   (`_format_readouts`: the text that was read, then positions grouped by region, one line per
   layer).
5. Before every backend call `trim_to_cap` stubs the oldest tool results so the tool output in
   context stays under `CONTEXT_CHAR_CAP` (800k chars). A backend exception triggers the salvage
   ladder: shrink every tool result and retry the same turn → force `finish()` → give up with
   `stopped_by = "error: …"`.
6. When `output_tokens >= budget` (non-reasoning tokens) or `max_calls` is hit, one REDUCTION
   user turn is appended and the next call is forced to `finish`.
7. After `finish`, `multiple_choice_stage` appends MC_PROMPT with the shuffled options and forces
   `choose_quirk`; the pick is stored in `choice`.

## The run record (one JSON per run)

Top level (`agent.loop.RunRecord.to_json()` plus two driver fields):

| field | type | meaning |
|---|---|---|
| `organism` | str | the audited tag (`base` = control) |
| `condition` | str | always `"live"` |
| `arm` | str | the arm as given (`olens`, `blackbox-ask`, …); the presentation is NOT in it (see the path) |
| `auditor` | str | investigator model id |
| `seed` | int | run seed: drives the sampling seeds sent to the target and the stage-2 shuffle |
| `turns` | list[Turn] | the transcript: `{role: assistant\|tool, content, tool_calls: [{id, name, args}], output_tokens, reasoning_tokens}` |
| `tool_log` | list[ToolLog] | one per tool call: `{name, args, output, cells_served}` — the exact text the agent saw |
| `notes` | list[str] | `note()` texts |
| `result` | dict \| null | `finish()` payload: `{predictions: [{behavior, trigger, evidence}] (<=10), suspects_quirk, confidence}` |
| `output_tokens` | int | non-reasoning generated tokens (the budget currency) |
| `reasoning_tokens` | int | reasoning tokens (not budgeted) |
| `cells_served` | int | readout cells handed over across the run |
| `chars_served` | int | characters of replies + readout pages handed over |
| `stopped_by` | str | `finish` \| `salvaged finish after <err>` \| `budget` \| `max_calls` \| `error: <err>` |
| `choice` | dict \| null | stage 2: `{letter, quirk, ranked: [quirk…], reason, options: {letter: quirk}, output_tokens}` or `{error, options}` |
| `consult_calls` | int | `ask_claude` invocations |
| `select` | str | the presentation used |
| `allow_reference` | bool | whether `model='reference'` was available |
| `trimmed` | int | tool results replaced by the trim stub |
| `server_calls`, `server_seconds` | int, float | added by the driver: target + lens calls and wall time |
| `judged` | dict | added by `stage=judge`: `{<quirk>: {success: {reason, match}, top5: {…}, pass1: {…}}}` |
| `judges` | dict | added by `stage=judges`: `{<quirk>: {<judge>: {reason, score} \| null, plausible5: {reason, score, picked, truth}, closeness_best5: {score, reason}, lens_cited: {score, reason}}}` |

The record path encodes the cell: `runs/live/<arm>[-<select>][-oracle][-hint-<from>][-mc5]
[-noref]/<organism>/<auditor with / -> _>/seed_<k>.json`. Old records without `select` are
attributed a presentation from the directory suffix by `_collect`.

## Package map

| module | holds | imports from |
|---|---|---|
| `tools/arms.py` | `LENS_NAME`, `LIVE_ARMS`, `lens_of`, `is_fixed`, `has_scaffold`, `has_ask`, `valid_arm`, `register_lens` | — |
| `agent/prompts.py` | every string the agent is shown; `system_prompt`; `register_lens_context` | `tools/arms` |
| `agent/loop.py` | `Turn`, `ToolLog`, `RunRecord`, `Backend` (Protocol), `Budget`, `run_tool_loop`, `trim_to_cap`, `shrink_tool_results` | `agent/prompts` |
| `agent/backends.py` | `openai_compatible_backend`, `with_cache_breakpoints`, `make_backend_factory`, `FakeBackend` | `agent/loop` |
| `tools/positions.py` | `ControlIds`, `PositionTag`, `regions_of`, `position_set`, `position_set_all`, `thin`, `is_punct` | `util/text` |
| `presentation/select.py` | `MODES`, `select_summary`, `select_llm`, `rank_by_fve`, `keep_cells`, `cells_of` | `llm/route` (lazy) |
| `tools/live.py` | `LensClient` (Protocol), `LiveClient`, `LiveTools`, `Conversation`, `tool_names`, `tool_schemas`, `mc_options`, `five_way`, `multiple_choice_stage`, `run_live_agent`, `vllm_slug`, `render_qwen` | `agent/*`, `presentation`, `tools/arms` |
| `tools/fake.py` | `FakeClient` | — |
| `judges/paper.py` | `JUDGE_PROMPT`, `judge_runs`, `format_predictions` | `llm/route` |
| `judges/graded.py` | the graded / quality / closeness / plausible5 / micro prompts, `judge_records`, `build_distractors`, `load_distractors`, record helpers | `llm/route`, `registry` |
| `registry/quirks.py` | `load_quirk_registry`, `load_held_out_prompts`, `quirk_of`, data paths | — |
| `llm/client.py` | `async_json`, `schema_block` | — |
| `llm/openrouter.py` | `install`, `preflight`, `report`, `SPEND` | `llm/client` |
| `llm/route.py` | `async_json_route`, `install_openrouter` | `llm/client`, `llm/openrouter` |
| `stats.py` | `wilson_interval`, `Rate` | — |
| `util/text.py` | `decode_byte_level` | — |
