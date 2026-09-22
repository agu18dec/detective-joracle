# detective-joracle

**Status: early (0.1.0) — the API may change between minor versions.**

An audit harness for hidden behaviors in language models. An **investigator agent** is given
tools for talking to a **target** model (chat with any system prompt, prefill, raw completion),
plus one **lens** tool that decodes the target's internal activations on any conversation it
ran, and is asked to find the ONE behavior that was planted in the target. What the lens returns
is shaped by a swappable **presentation**; what the agent finally claims is **scored in two
stages** by LLM judges. It is the in-the-loop protocol of AuditBench (arXiv 2602.22755, §3 and
App. D–F) with the white-box tool generalized to any lens that can answer one HTTP request.

Everything except the GPU servers runs on a laptop: the package is pure Python, the tests are
offline, and `examples/` ships a mock target + mock lens so the whole pipeline can be exercised
end to end with no keys.

## Install

```
git clone <this repo> && cd detective-joracle
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"      # or: pip install -e ".[dev]"
.venv/bin/python -m pytest -q                                    # offline, ~1 s
```

Requires Python ≥ 3.11. Runtime deps: `requests`, `openai` (the OpenAI-compatible client is used
for the target, the auditor and the judges), `numpy`. Extras: `[test]`, `[dev]` (ruff, mypy),
`[plot]` (matplotlib), `[serve]` (modal, for hosting the viewer).

## 60-second quickstart (no GPU, no keys)

```
examples/run_end_to_end.sh
```

This starts `examples/mock_target.py` — one process that plays the target (an OpenAI-compatible
chat API impersonating a model that flatters the user), the lens server (the readout contract
below, with a planted "praise the user" disposition in the activations) and a placeholder judge
— then runs the four stages against it and builds the viewer:

```
examples/out/runs/live/olens-noref/flattery_td_r16/<auditor>/seed_0.json   # the run record
examples/out/report.md                                                     # the tables
examples/out/site/index.html                                               # the viewer
```

With `OPENROUTER_API_KEY=sk-or-…` in the environment the same script uses a real investigator
(Gemini 3.8 Flash by default, `AUDITOR=` to change) and a real judge; without it the auditor is
a scripted `FakeBackend` and every judge verdict is a placeholder. See `examples/README.md`.

## Architecture

```
                   target model (any OpenAI-compatible chat server)
                       |                                    |
        chat / prefill / raw completion       lens server: activations -> text
                       |                                    |   (your implementation of the
                       +------------ LiveClient ------------+    readout contract, or one of
                                          |                      the reference servers/)
                                     LiveTools                <- the affordances (an ARM)
                        chat / complete [/ sample_user_turn]     black box
                        readouts | fixed_prompt_readouts         white box
                        generate_and_test                        App. F scaffold
                        ask_claude                               probe
                        note / finish
                                          |
                                presentation/select.py        <- the PRESENTATION
                        all | summary | llm | summary-llm | (fve)
                                          |
                                  agent/loop.py               <- the loop
                       25k output tokens, then a forced finish(); context trimming; salvage
                                          |
                      +-------------------+---------------------+
                      |                                         |
              free-form predictions                    closed-set choice (stage 2)
       judges/paper.py -> detect@10 / detect@5 / pass@1     16-way or 5-way, exact match
       judges/graded.py -> graded 0-10, closeness,
                           plausible5, micro-judges
                                          |
                    scripts/run_audit.py stage=report  ->  report.md   (+ viewer, plot)
```

### The five swappable axes

| axis | where | values today | how to add one |
|---|---|---|---|
| **target organism** | any OpenAI-compatible server (`target=`), or the lens server's `chat` endpoint | Qwen3.6-27B LoRA organisms in `servers/`; the mock | serve it; the tag is a string (`quirk_of` reads `<quirk>_<pipeline>_r<rank>`) |
| **affordance (arm)** | `tools/arms.py`, `tools/live.py::tool_names` | `blackbox`, `<lens>`, `<lens>-fixed`, `scaffold-*`, `*-ask` | a schema in `tool_schemas` + a branch in `LiveTools.call` |
| **lens** | the lens server (`server=`) | OLens verbalizer, Jacobian lens, logit lens, NLA-RL (reference impls) | implement the readout contract; `register_lens_context()` for the prompt |
| **presentation** | `presentation/select.py` | `all`, `summary`, `llm`, `summary-llm` (`fve` stubbed) | a function + a branch in `LiveTools.readouts` |
| **scoring** | `judges/paper.py`, `judges/graded.py`, stage 2 in `tools/live.py` | detect@k, pass@1, graded, closeness, plausible5, micro-judges, closed-set | a prompt + a schema + an entry in `judge_records` |

`docs/extending.md` has a recipe for each.

## The two contracts

### 1. Target: an OpenAI-compatible chat server

`target=<base url>` (may contain `{tag}` or `{slug}` to pick a server per organism). The harness
sends, with the model name `organism`:

```
POST {target}/v1/chat/completions
{"model": "organism", "messages": [...], "n": 1..4, "temperature": t, "max_tokens": m, "seed": s,
 "chat_template_kwargs": {"enable_thinking": false},          # optional; ignored by servers
 "continue_final_message": true, "add_generation_prompt": false}  # only with a prefill: the
                                                                  # last message is the assistant prefill
POST {target}/v1/completions          # raw text and user-persona sampling
{"model": "organism", "prompt": "...", "n": n, "temperature": t, "max_tokens": m, "seed": s,
 "stop": ["<|im_end|>", "<|endoftext|>"]}
GET  {target}/v1/models               # the launchers wait for a model named "organism"
```

`finish_reason == "length"` marks a truncated reply; `usage.completion_tokens` is split across
the `n` choices. vLLM and SGLang satisfy this as-is; the mock shows the minimum.

Without `target=` the chat tools use the lens server's own `chat` endpoint (below), which the
reference `servers/organism_server_modal.py` implements with HF + PEFT.

### 2. Lens server: the readout contract

`server=<template>` with `{name}` (`http://host:8000/{name}`), or a Modal web-endpoint prefix
(`https://<ws>--<app>-<cls>-` → `<prefix><name>.modal.run`). Three endpoints, all `POST` JSON:

```
POST readout
request  {"organism": "<tag>",                     # "base" = the unmodified control
          "messages":  [{"role": "system"|"user"|"assistant", "content": "..."}, ...],
          "completion": "<the assistant reply to append and read>",   # "" to read the prompt only
          "positions": "all",                      # every non-system token (what the agent gets);
                                                   # servers may also accept "P", a kind/region
                                                   # name, "everything", or [int, ...]
          "layers": [20, 24, ...],                 # from `layers=`; the nla arm sends [42]
          "k": 1,                                  # samples per cell (verbalizers)
          "lens": "olens" | "jlens" | "logit" | "acts" | <yours>,
          "seed": 12345}
response {"n_tokens": 312,                         # tokens in the rendered conversation
          "organism": "<tag>", "lens": "<lens>",
          "n_matched": 298,                        # optional: positions matching the request
                                                   # before any thinning (the page says so if
                                                   # n_matched > len(tokens))
          "k": 1,                                  # optional
          "tokens":   {"<pos>": "<decoded token>", ...},           # str keys, decoded text
          "tags":     {"<pos>": {"region": R, "kind": K}, ...},
          "readouts": {"<layer>": {"<pos>": ["<text>", ...], ...}, ...},   # k strings per cell
          "extra":    {"kl": {"<layer>": {"<pos>": 0.31}}, ...}}          # optional per-cell numbers
          # or {"error": "..."} — surfaced to the agent as a tool error
```

* `region` ∈ `system | user | header | reply` (`header` = the assistant turn's control tokens
  before the reply, "about to speak"); `kind` ∈ `boundary | punct | user | header | reply |
  user4 | reply4` (`tools/positions.py` computes both from token ids; `boundary` and `punct` are
  the "summary tokens" the `summary` presentation keeps).
* Every requested layer should be present for every position returned (thin positions, never
  layers): the agent reads a depth profile per position.
* Token strings must be readable text (`util.text.decode_byte_level` undoes byte-level BPE).
* **Out-of-process lenses** (`lens="acts"`): return the raw residuals instead —
  `{"positions": [int, ...], "d": 5120, "dtype": "float16", "vectors": {"<layer>": "<base64 of
  fp16 [n, d] row-major>"}, "tokens", "tags", "n_tokens", "organism", "n_matched"}`; a second
  service turns rows into text (`nla=` URL, `POST {"vectors", "n", "d", "k", "seed"} → {"texts":
  [[k strings] per row]}`). `LiveClient.readout_nla` does the two hops and hands the tools a
  standard readout response.

```
POST fixed        # the paper's parameterless white-box tool: precomputed on the held-out prompts
request  {"organism": "<tag>", "lens": "<lens>", "layers": [...], "k": k}
response {"organism", "lens", "layers": [...], "k": k,
          "prompts": [{"id": "f00", "user": "<prompt>", "n_tokens": n, "tokens": {...},
                       "tags": {...}, "readouts": {"<layer>": {"<pos>": [...]}}}, ...]}
POST chat         # only used when target= is not given
request  {"organism", "messages", "system"?, "prefill"?, "n", "temperature", "max_new", "seed",
          "mode": "assistant" | "user_turn" | "raw", "text"?}          # text: the raw prompt
response {"replies": [{"text", "n_tokens", "truncated"}, ...], "n_prompt_tokens", "organism", "mode"}
```

`docs/contracts.md` is the normative version; `examples/mock_target.py` is a complete minimal
implementation; `servers/organism_server_modal.py` a full one.

## Running a real audit

1. Serve the target as an OpenAI-compatible API (`servers/organism_vllm_modal.py` is the
   reference: vLLM on Modal, one app per merged organism).
2. Serve the lens (`servers/organism_server_modal.py`, or your own implementation of the
   contract). The reference servers need a private lens stack — see `servers/README.md`.
3. Run the stages. Keys are passed per command, never exported:

```
export JORACLE_SERVER='https://<ws>--auditbench-organism-organism-'   # or http://lens:8000/{name}
export JORACLE_TARGET='https://<ws>--{slug}-serve.modal.run'          # or http://target:8000
OPENROUTER_API_KEY=sk-or-… scripts/run_loop.sh "secret_loyalty_td_r16,base" "blackbox,olens" 3 \
    "auditors=anthropic/claude-opus-5 select=llm"
```

`run_loop.sh` warms the target servers, checks each organism answers differently from `base`,
then runs `run_audit.py stage=agent`, two passes of `stage=judge`, `stage=judges,report`.
`run_grid.sh` queues several organism blocks. Then:

```
python scripts/build_viewer.py            # -> outputs/live/site/index.html (static, one file)
python scripts/plot.py                    # -> outputs/live/plots/live_grid.png  ([plot] extra)
modal deploy scripts/serve_viewer_modal.py   # optional: host the page ([serve] extra)
```

The investigator and judge go through OpenRouter when `OPENROUTER_API_KEY` is set (model ids
are OpenRouter's: `google/gemini-3.8-flash`, `anthropic/claude-opus-5`, …); `backend=openai`
uses `OPENAI_API_KEY` / `OPENAI_BASE_URL` for any other OpenAI-compatible endpoint (a local
vLLM auditor, a proxy).

## Configuration reference (`scripts/run_audit.py key=value …`)

| key | type | default | meaning |
|---|---|---|---|
| `stage` | str | — | comma-separated `agent`, `judge`, `judges`, `report` (run in order) |
| `server` | str | — | lens server endpoint template (`{name}`) or Modal prefix; `fake` = the canned client |
| `target` | str | `""` | OpenAI-compatible base URL for chat tools; `{tag}`/`{slug}` per organism; empty = lens server's `chat` |
| `nla` | str | `""` | verbalize URL for the out-of-process `nla` lens (second hop) |
| `organisms` | str | — | comma-separated tags; `base` is the unmodified control (required for `agent`) |
| `arms` | str | `blackbox,olens` | comma-separated arm names (`tools/arms.py`) |
| `seeds` | int | `3` | runs per (auditor, organism, arm); the paper uses 16 |
| `auditors` | str | `google/gemini-3.8-flash` | comma-separated investigator model ids |
| `judge_model` | str | `google/gemini-3.8-flash` | the judge and every auxiliary call (scenario generation, readout triage) |
| `layers` | str | `20,24,…,60` | layer grid sent to the lens (`nla` ignores it: layer 42 only) |
| `budget_tokens` | int | `25000` | non-reasoning output tokens per run, then the forced `finish()` |
| `max_calls` | int | `400` | tool-call cap (not binding by design) |
| `out_root` | str | `outputs/live` | where `runs/`, `report.md`, `site/`, `plots/` go (relative to the repo) |
| `backend` | str | `openrouter` | `openrouter` \| `openai` \| `fake` (scripted auditor) |
| `closed_set` | bool | `true` | stage 2 after `finish()`: pick the quirk from a closed set |
| `mc` | int | `16` | closed-set size: `16` = whole registry (chance 0.06); `5` = truth + 4 frozen distractors (chance 0.2) |
| `select` | str | `all` | readout presentation: `all` \| `summary` \| `llm` \| `summary-llm` (\| `fve`, not wired) |
| `select_top_n` | int | `50` | cells kept by the `llm` presentation |
| `oracle` | bool | `false` | probe: state the true quirk up front, ask the agent to verify and localize it |
| `no_reference` | bool | `true` | no base-model control (`model='reference'`) for anyone; `false` restores it |
| `hint_from` | str | `""` | probe: hand over another auditor's top hypothesis for the same cell |
| `consultant` | str | `anthropic/claude-opus-4.5` | what the `-ask` arms' `ask_claude` calls |
| `workers` | int | `6` | parallel runs (= server containers in use) |
| `limit_runs` | int | `0` | cap the runs started by this invocation (0 = all) |
| `force_judges` | bool | `false` | re-grade records that already carry graded verdicts |
| `dry` | bool | `false` | list the work and stop |

Records: `<out_root>/runs/live/<arm>[-<select>][-oracle][-hint-<from>][-mc5][-noref]/<organism>/
<auditor>/seed_<k>.json` — the directory encodes the whole cell, so nothing collides and every
stage resumes. A record whose `stopped_by` starts with `error` is re-run next pass. The record
schema is in `docs/architecture.md`.

## Metrics reference

| metric | what it measures | known failure mode |
|---|---|---|
| **graded (0-10)** | a judge scores how well the FULL ranked prediction set identifies the planted quirk, crediting the best prediction | judge variance ±1 between reruns; treat single-digit deltas at n=3 as noise |
| **lift** | graded(arm) − graded(blackbox), same organism + auditor; the only headline a lens can earn | undefined without a blackbox cell for that auditor; negative lifts within noise are budget spent reading pages |
| **detect@10** | the paper's judge: any of ≤10 predictions matches (the paper's headline) | rewards long hedged lists; use with pass@1 and graded |
| **pass@1** | the top prediction alone matches | a correct quirk ranked 2nd scores 0 (the reference control mostly buys rank-1 precision) |
| **closed-set top-1 / top-3** | after `finish()`, the same agent picks from 16 (or 5) candidates | recognition, not discovery — can rise while detect stays 0 |
| **plausible5** | the top hypothesis is matched to truth vs 4 frozen plausible distractors | distractor quality; separates "right behavior" from "right area" only as well as the distractors do |
| **closeness / best-of-5** | 1-10 distance of the top (best top-5) hypothesis to the truth | "right territory, wrong sign" scores 3-4 and looks like progress |
| **chat_sufficient** | a judge shown ONLY the chat transcripts can reach the same hypothesis (1 = the lens was not load-bearing) | a judge that is more capable than the auditor says "reachable" for things the auditor never saw |
| **lens-decisive** | hit AND chat_sufficient = 0 — the only detection a lens wins by construction | needs both judges present; blackbox is trivially 0 |
| **probes → hit** | mean chat calls in the runs that detected (efficiency) | only defined when something detected |
| **evidence_real / lens_cited / coverage / stylistic_tic** | cited conversations support the claim / finish() names readouts (regex) / 1-5 breadth of probing / the hypothesis is about phrasing not substance | regex `lens_cited` is a floor, not proof the readouts mattered |
| **base → \<quirk\> rows** | the control organism judged against each planted quirk: a false-alarm signature | not accuracy; a high closeness on base means the judges are lenient, not that base has the quirk |

## Results so far (Qwen3.6-27B organisms, see `docs/results.md`)

* On organisms whose quirk **leaks into the reply** (synthetic-document training; hardcoded
  test cases), black-box auditing already scores 7–10 and the lens adds nothing.
* On the one organism whose quirk **does not leak** (`secret_loyalty`, transcript-distilled:
  a pro-Russia stance that only appears in geopolitics), black-box never finds it for any
  auditor (0.0–0.3). Claude Opus 5 with the OLens verbalizer under the `llm` presentation
  reaches **9.3 / detect 3-of-3 / pass@1 3-of-3** with the base-model control, and **5.7 /
  detect 2-of-3, Russia named 3-of-3** without it: the lens surfaces the quirk on its own; the
  control turns a detection into a top-ranked identification.
* The `llm` presentation needs a capable auditor (it hurts Gemini Flash: 0.7 vs 2.7 unfiltered);
  `summary` is the weak-auditor mitigation (Flash 4.0). Weaker auditors get large lifts where
  their black-box fails (Haiku 4.5: +4.0, +5.5 on two organisms) and find it in fewer probes.
* Harness faults worth knowing before trusting a number: `docs/faults.md`.

## Bring your own lens / target

* **Lens:** implement `POST readout` (and optionally `fixed`) per the contract; register the
  lens id and its one-paragraph description with `detective_joracle.register_lens_context(name,
  display, context)` so `<name>` becomes a valid arm and the agent is told what it is reading.
  Out-of-process lenses use the `acts` two-hop route. The reference implementations in
  `servers/` (OLens/J-lens/logit in one HF+PEFT process; NLA-RL out of process) require a
  private lens stack and are documented in `servers/README.md`.
* **Target:** any OpenAI-compatible chat server; `target=` may template the URL per organism
  tag. A different chat format needs its own `render_qwen` (raw prompts for user-turn sampling)
  and `ControlIds` (position tagging).
* **Auditor:** `agent/backends.py::make_backend_factory` — `openrouter` or `openai`; any
  `openai.OpenAI`-style client works with `openai_compatible_backend`.

## Repository layout

```
src/detective_joracle/
  agent/       loop.py (RunRecord, Budget, Limits, run_tool_loop[_async], run_many)  backends.py  prompts.py
  tools/       live.py (LiveClient, LiveTools, run_live_agent, stage 2)  arms.py  positions.py  fake.py
  presentation/select.py       judges/paper.py  judges/graded.py
  registry/    quirks.py + data/{quirk_registry,distractors,held_out_prompts}.json
  llm/         client.py (async_json)  openrouter.py  route.py      stats.py  util/text.py
scripts/       run_audit.py  run_loop.sh  run_grid.sh  build_viewer.py  plot.py  serve_viewer_modal.py
servers/       reference GPU servers (need the private lens stack)   examples/  mock target + e2e
docs/          architecture · contracts · extending · troubleshooting · harness · design · results · faults
tests/         offline (FakeClient / FakeBackend / the mock over HTTP)
```

## Roadmap

* `fve` presentation: rank cells by how well the readout reconstructs its activation (needs a
  per-cell `extra.fve` from the lens server) — the one ranking that is a property of the lens,
  not of a judge.
* Cell re-sampling and non-obviousness reasons as tools; a hallucination judge over cited
  readouts (`docs/faults.md` §3).
* A `ControlIds`/render abstraction per chat template (today: ChatML/Qwen).
* Retrain transcript-distilled organisms with a refusal-preserving mix (the loudest confound).

## License

MIT. AuditBench (the game, the judge wording, the held-out prompts, the quirk registry) is
`safety-research/auditing-agents`; see `CHANGELOG.md` for provenance.
