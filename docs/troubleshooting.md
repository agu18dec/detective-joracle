# Troubleshooting — the gotchas that cost real runs

Each of these was hit on the grid; the fix is in the code, and the reason is here so nobody
"simplifies" it away.

## Proxies that 303 long requests must be FOLLOWED, not retried

Modal's web proxy redirects any request that outlives ~150 s to a polling URL with a `303`. A
full-conversation readout at 11 layers legitimately takes minutes, so every one of them 303'd.
Treating the 303 as a failure and retrying restarts the work: the readout could never complete
("gave up after 6 attempts (HTTP 303)"). `LiveClient._post` sends `allow_redirects=True`, so
`requests` follows to the polling URL and blocks there, exactly as `curl -L` does; only 5xx,
connection errors and timeouts are retried. `scripts/run_loop.sh` uses `curl -sL -m 900` for
the same reason: a cold start is held by the proxy for minutes and a short `-m` reads as "HTTP
000 forever".

## The rolling 800k-character context cap

Gemini refused outright ("blocked: OTHER", a 502) at ~1.55M characters of tool output — three
full every-position readout pages. `agent/loop.py::trim_to_cap` runs before every backend call
and replaces the OLDEST large tool results with a short stub until the tool output in context
is under `CONTEXT_CHAR_CAP = 800_000`; the newest result is never touched because it is the one
the agent is about to read. The cap applies to every arm identically; black-box runs never reach
it, so in practice it is the price of reading whole pages, paid only by the arms that read.
`READOUT_CHAR_BUDGET` (1.2M chars per run) additionally makes the `readouts` tool refuse further
pages, telling the agent to work from what it has. `RunRecord.trimmed` counts stubs; check it
before attributing a miss to the lens.

## Provider refusals mid-run and the salvage ladder

A provider failure late in a run used to throw away everything the agent had found (a 135-call
run lost at 16k output tokens). `run_tool_loop` now: (1) on the first exception, shrinks every
tool result to `SALVAGE_KEEP_CHARS` and retries the SAME turn; (2) on the second, appends the
REDUCTION prompt and forces `finish()`; (3) on the third, stops with `stopped_by = "error: …"`.
A salvaged record has `stopped_by = "salvaged finish after …"`. Gemini's refusal on
`secret_loyalty` content is about content, not size (it fired at 692k chars, under the cap) —
its readouts are full of Chinese-language political text. Mitigations in
`make_backend_factory`: `safety_settings` BLOCK_NONE, `provider.allow_fallbacks`. The driver
re-runs any record whose `stopped_by` starts with `error` on the next pass.

## Anthropic prompt caching through OpenRouter

An agent turn is the previous turn plus one exchange, so without caching every one of ~200 calls
per run pays full price for a 100–300k-token context (Opus 5: ~$135/hour). `with_cache_breakpoints`
puts a `cache_control: ephemeral` block on the system prompt and on the newest message so each
turn re-reads the prefix from cache; `make_backend_factory` turns it on for `anthropic/*` models.
Non-Anthropic providers ignore or reject the block, hence the model-name gate.

## Empty `choices` from the provider

OpenRouter occasionally returns an error-shaped body with `choices` absent, which surfaced as
`'NoneType' object is not subscriptable` and killed the run at whatever call it happened on.
`openai_compatible_backend` treats it as transient: three attempts with a short sleep, then a
`RuntimeError` that the salvage ladder handles.

## A null-content assistant turn without tool calls

`content: null` is only valid alongside `tool_calls`; a null-content assistant message with no
calls makes Google reject the NEXT request with `INVALID_ARGUMENT`. `run_tool_loop` writes
`"..."` in that case.

## Sticky CUDA asserts → the 8192-token readout cap

A readout runs a real forward pass over the whole conversation. An over-long input triggered a
device-side assert, and a CUDA assert is sticky: it kills the container for every later
request, not just its own. The reference server refuses conversations over `MAX_READ_TOKENS =
8192` with a polite error ("read a shorter conversation; start a fresh one rather than
continuing") instead. Keep the same guard in your own lens server.

## Cells per readout: thin positions, never layers

The agent no longer picks positions (choosing good ones is a skill it does not have, and a badly
chosen slice looks like a lens failure). A readout is EVERY non-system token at every layer — a
25-token question with a 256-token reply is ~300 positions × 11 layers ≈ 3300 cells — so the
server cap `MAX_CELLS = 4400` has to clear that or "every position" quietly becomes a sample.
When thinning bites it drops positions evenly (`thin`) and keeps every layer, and the response's
`n_matched > len(tokens)` makes the page say so.

## The reference-model control, and why `no_reference=true` is the default

`model='reference'` runs any tool on the unmodified base model. It is the strongest move a lens
arm has (read the same conversation on both, keep what differs), and it is a crutch a real
auditor may not have. With the control, Opus + OLens/llm landed the quirk at rank 1 every time
(pass@1 3/3, graded 9.3); without it, it still named the pro-Russia bias in every run but ranked
it below broader hypotheses (pass@1 0/3, graded 5.7). So the default measures the lens standing
on its own; pass `no_reference=false` to restore the control and read the two side by side. The
directory suffix `-noref` keeps the cells apart.

## OpenRouter pacing: bursts turn cells into `None`

OpenRouter limits some models to ~300 rpm platform-wide under "high demand", shared with any
other session on the key; a burst past it turned 35–70 % of judge cells into unjudged `None`s.
`llm/openrouter.py` paces requests under `OPENROUTER_RPM` (default 240) process-wide and retries
transient failures patiently; `stage=judge` is run twice by `run_loop.sh` so a second pass fills
the gaps. A judge that returns `None` is counted as unjudged, never as a zero — but a whole
column of `None` is a broken key, not a result: preflight with `llm.openrouter.preflight(model)`.

## `JUDGE_HTTP_TIMEOUT`

Per-call timeout for the judge route (default 90 s; the OpenRouter route sets 180 s). A hung
call must not stall the batch, so it is deliberately short; a self-hosted judge under heavy
concurrency may need more. It is validated ONCE before any call and allowed to raise — a bad
value swallowed by the degrade-to-`None` guard would write a plausible all-zero artifact.

## Modal app names and DNS labels

A Modal web hostname is `<workspace>--<app>-<function>.modal.run` and the label is capped at 63
characters; a long app name is silently unreachable at the obvious URL. `organism_vllm_modal.
app_slug` (mirrored by `tools.live.vllm_slug`) abbreviates the tag: `secret_loyalty_td_r16` →
`ab-vllm-sec-loy-td`. Only one `modal serve` per app name at a time — two fight each other in a
container-restart loop.

## The mock judge

`examples/mock_target.py` answers every structured call with a placeholder filled from the
schema (`score: 5`, binaries `1`, `letter: "A"`). A report built on it is a plumbing check; if
you see `5.0 (n=1)` everywhere, you are looking at the mock.
