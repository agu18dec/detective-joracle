# Extending the harness

Each recipe names the exact file and function. Add a test for every branch you add; monkeypatch
`detective_joracle.llm.route.async_json_route` for anything that would call a model.

## A new lens

1. Serve it behind the readout contract (`docs/contracts.md`). If it decodes one activation
   vector at a time and cannot live in the target's process, serve `lens="acts"` from the
   target side and a `verbalize` endpoint from yours, and pass the latter as `nla=` — or add a
   sibling of `LiveClient.readout_nla` (`tools/live.py`) if the two-hop shape differs.
2. Register it client-side, before `run_live_agent`:

   ```python
   from detective_joracle import register_lens_context
   register_lens_context(
       "sae", "SAE feature labels",
       " You also have an interpretability lens, and readouts(conversation) applies it to any "
       "conversation you have run. It is a SPARSE AUTOENCODER: each cell lists the labelled "
       "features active at that position and layer ...",
   )
   ```

   This makes `sae`, `sae-fixed`, `scaffold-sae`, `sae-ask` valid arms (`tools/arms.py::
   valid_arm`) and puts the paragraph into `system_prompt` (`agent/prompts.py::LENS_CONTEXT`).
   For a lens shipped with the package, add the entries to `LENS_NAME` and `LENS_CONTEXT`
   directly.
3. Run with `arms=sae layers=<your grid>`. If the lens reads a single layer like NLA, mirror the
   `lens == "nla"` special cases in `tools/live.py` (`tool_schemas` says "the one layer this
   lens reads"; `LiveTools.readouts` pins `layers`).
4. Optional: `FIXED_LAYERS[<lens>]` / `FIXED_K[<lens>]` for the `-fixed` arm.

## A new presentation

1. Write `select_<name>(res, **kw) -> tuple[dict, str]` in `presentation/select.py`: return a
   copy of the readout response holding only the cells to show (`keep_cells` does the
   filtering; put per-cell reasons in `res["why"]["<layer>:<pos>"]`) and a one-sentence note
   the page header carries so the agent knows it is seeing a selection.
2. Add the name to `MODES`.
3. Dispatch it in `tools/live.py::LiveTools.readouts` (the block after `served = _n_cells(res)`;
   `summary` composes with `llm`, follow that pattern if yours composes).
4. The driver's `select=` accepts it automatically; `run_path` puts `-<name>` in the directory
   and `_collect` folds it into the arm label. Add the suffix to the fallback list in
   `_collect` (`("llm", "summary", "fve")`) if old records without a `select` field exist.

`fve` is the reserved stub: `rank_by_fve` raises until the server returns `extra.fve`.

## A new judge

1. In `judges/graded.py`: a prompt constant with `{behavior}` / `{trigger}` / `{quirk}` /
   `{predictions}` placeholders; an entry in `_RANGE` (`name: (lo, hi)`) and, if the JSON key
   is not `score`, in `_KEY`; a `("<name>", extra, prompt)` tuple appended in `_requests_for`.
   `judge_records` batches every judge by name with one schema each and validates the range.
2. In `scripts/run_audit.py`: a column in `_collect` (the `for src, dst in (...)` mapping) and a
   cell in the table you want it in (`stage_report`).
3. In `scripts/build_viewer.py`: a `judgeRow(...)` line in `renderJudgeBlock` (JS template).

A judge over the paper's binary verdicts (a different reduction of the same `judge_runs`
output) goes into `stage_judge` instead: append `(registry[q], preds[:k])` with a new kind.

## A new tool (affordance)

1. `tools/live.py::tool_schemas`: add `"<name>": (description, properties, required)` to
   `all_tools`.
2. `tools/live.py::tool_names`: decide which arms get it (an arm flag like `has_ask` in
   `tools/arms.py` if it is opt-in; `LIVE_ARMS` is generated, so add the suffix/prefix rule to
   `_arms_for` and `valid_arm`).
3. `LiveTools`: a method returning the text the agent sees (and `(text, cells)` if it serves
   readout cells), and a branch in `LiveTools.call` that coerces the args.
4. `agent/prompts.py`: an `EVIDENCE_<NAME>` paragraph and a line in `system_prompt` if the agent
   should be told about it up front.
5. `judges/graded.py::conversations_of` / `user_messages` if the tool creates conversations or
   sends user messages (so the micro-judges see them), and `scripts/build_viewer.py::
   summarize_args` / `callLine` for a compact rendering.

`sample_user_turn` is the worked example of a tool that exists but is off by default.

## A new auditor backend

`agent/loop.py::Backend` is the Protocol: `(messages, tools, force_tool) -> (text, calls,
usage)` with `calls = [{id, name, args}]` and `usage = {output_tokens, reasoning_tokens}`.

* Any `openai.OpenAI`-style client: `openai_compatible_backend(client, model, extra_body=...,
  cache=...)` (`agent/backends.py`). Add a `kind` to `make_backend_factory` for a named route
  (the `openai` kind reads `OPENAI_API_KEY` / `OPENAI_BASE_URL`).
* Anything else: implement the Protocol directly. `FakeBackend` shows the shape, including how a
  forced tool (`force_tool`) must be honoured — the reduction step depends on it.
* The driver selects by `backend=<kind>`.

## A new target / chat format

* The target is any OpenAI-compatible server; `target=` may template the URL per organism.
* ChatML is assumed in two places: `tools/live.py::render_qwen` (raw prompts for user-turn
  sampling and the stop strings in `_chat_target`) and `tools/positions.py::ControlIds` /
  `regions_of` (position tagging by token id, the assistant header shape). Another template
  needs its own renderer and control ids; the lens server tags positions, so the latter change
  lives server-side (`servers/README.md` § Bringing your own lens).
* `registry/quirks.py::quirk_of` maps a tag to its registry quirk (`<quirk>_<pipeline>_r<rank>`;
  a bare registry name maps to itself). Point `load_quirk_registry(path=...)` at your own file
  to audit a different behavior set; `build_distractors(..., path=...)` freezes distractors for
  it.

## A new metric in the report

`scripts/run_audit.py::_collect` produces `{(auditor, arm+select, organism): {column: [per-run
ints]}}`; `stage_report` formats. Derived metrics (lift, lens-decisive) are computed there from
the per-run columns; follow `lensdec` for one that needs two judges present.
