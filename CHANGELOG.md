# Changelog

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
