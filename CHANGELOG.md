# Changelog

## 0.2.0 (unreleased)

The agent loop can host another benchmark; every addition is opt-in and existing callers see
the same behaviour (`docs/extending.md` § Hosting another benchmark).

* `agent/loop.py::run_tool_loop_async` is now the loop body: `backend` may be sync or async
  (`AsyncBackend`), and so may `tools.call`. `run_tool_loop` is the sync wrapper (a private
  event loop; a worker thread when called from inside a running loop).
* `agent/backends.py::async_openai_compatible_backend` (over `openai.AsyncOpenAI`; `effort=`,
  `provider=`; usage reports `prompt_tokens` and `cached_tokens`);
  `make_backend_factory(kind, async_client=True)`.
* `Limits` (`limits=`): per-tool execution caps (`BUDGET: …` past the cap) and unlock gates
  (`LOCKED: …` until a prerequisite tool has run k times), additive to `Budget`; when every
  capped tool is spent the loop runs the forced reduction (`stopped_by = "limits"` if the model
  still does not finish).
* A `ValueError` from `tools.call` is returned as `ERROR: …` and not counted toward `Limits`
  (a turn, not a call); other exceptions propagate as before. `LiveTools.call` is unchanged:
  it never raises and logs every call, errors included.
* `FORCE_ANY = "*"` → `tool_choice="required"` in both backends. The idle path is now: nudge,
  then (two prose turns in a row) the reduction with `FORCE_ANY`, then a nudge, then
  `stopped_by = "idle"` on the fourth. Before, a model that never called a tool was nudged
  forever (until its tokens exhausted the budget).
* `terminal=` (default `"finish"`) names the tool the reduction forces; `prompts.
  REDUCTION_TEMPLATE` / `IDLE_NUDGE` are the parametrized texts (`REDUCTION` is unchanged).
  `FakeBackend(steps, terminal=)` honours `FORCE_ANY`.
* `run_many(jobs, concurrency)`: semaphore-bounded `asyncio.gather`, results in job order.

## 0.1.0 — initial extraction (2026-09-22)

Extracted from the `auditbench` package of a research monorepo (the in-the-loop / "live"
harness only; the earlier static single-turn eval was left behind). Function and constant
names were kept so the two can be diffed.

* `detective_joracle` package (src layout): agent loop + backends + prompts, live tools over
  HTTP with a client Protocol, position tagging, presentations, the paper's judge and the graded
  judges, the frozen registry data, a minimal OpenAI-compatible structured-JSON client with an
  OpenRouter route.
* Generalizations over the source: any OpenAI-compatible chat server as the target (`target=`,
  templated per organism); a `{name}` endpoint template for the lens server (Modal prefixes
  still work); an `openai` auditor backend beside `openrouter`; `register_lens_context()` so a
  new lens id needs no code change; `layers=` on the driver; `server=fake` for offline smokes.
* Dropped: the Anthropic-direct judge path (every auxiliary call is OpenAI-compatible now);
  `positions.target`/`load_tags` (static-capture manifest plumbing); the monorepo's organism
  table and the auditing-agents clone reader (the frozen registry JSON is the source of truth);
  `RunRecord.id_of`/`set_of` (static-eval blinding); pydra in favour of `key=value` parsing.
* Reference GPU servers under `servers/` (need a private lens stack; documented).
* `examples/mock_target.py`: target + lens + placeholder judge in one process; the pipeline runs
  end to end without a GPU or a key.
* Provenance: the game, the judge wording, the 50 held-out prompts and the 16 quirk behaviors
  are AuditBench's (`safety-research/auditing-agents`, arXiv 2602.22755); the distractors were
  generated once with Gemini 3.8 Flash and frozen.
