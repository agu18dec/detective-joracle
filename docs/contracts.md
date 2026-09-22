# The two contracts

Everything the harness needs from a GPU is two HTTP services. Both are JSON; both are
implemented minimally by `examples/mock_target.py` and fully by `servers/`.

## 1. Target: OpenAI-compatible chat

Configured with `target=<base url>`. The URL may contain `{tag}` (the organism tag with `_`
→ `-`) or `{slug}` (`tools.live.vllm_slug(tag)`, a DNS-safe abbreviation) to route each
organism to its own server. The harness creates one `openai.OpenAI(base_url=<url>/v1,
api_key="none")` client per resolved URL. Model name is always `organism`.

### Assistant reply (`LiveTools.chat`)

```
POST /v1/chat/completions
{
  "model": "organism",
  "messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...],
  "n": 1..4, "temperature": 0..1.5, "max_tokens": 16..512, "seed": <int>,
  "chat_template_kwargs": {"enable_thinking": false}      # vLLM/SGLang extension; others ignore it
}
```

With a **prefill** the last message is `{"role": "assistant", "content": "<prefill>"}` and the
body also carries `"continue_final_message": true, "add_generation_prompt": false` (vLLM's
contract for continuing an assistant turn). The harness prepends the prefill to each returned
`message.content`.

Read back: `choices[i].message.content`, `choices[i].finish_reason == "length"` → truncated,
`usage.prompt_tokens`, `usage.completion_tokens` (split evenly across the `n` choices).

### Raw completion and user-persona sampling (`LiveTools.complete`, `sample_user_turn`)

```
POST /v1/completions
{"model": "organism", "prompt": "<text>", "n": n, "temperature": t, "max_tokens": m, "seed": s,
 "stop": ["<|im_end|>", "<|endoftext|>"]}
```

For user-persona sampling the prompt is `render_qwen(messages) + "<|im_start|>user\n" +
prefill` — ChatML. Another chat format needs its own renderer (`tools/live.py::render_qwen`).

### Health

`GET /v1/models` must list a model with `id == "organism"`; `scripts/run_loop.sh` polls it.

### Without `target=`

Chat goes to the lens server's `chat` endpoint (below) instead — one process serving both, as
`servers/organism_server_modal.py` does with HF + PEFT.

## 2. Lens server

Configured with `server=<template>`. If the template contains `{name}` it is formatted with the
endpoint name; otherwise it is treated as a Modal web-endpoint prefix and the URL is
`<prefix><name>.modal.run`. Every call is `POST` with a JSON body; a `200` whose body has an
`"error"` key is surfaced to the agent as a tool error; a non-2xx below 500 raises; 5xx,
connection errors and timeouts are retried with linear backoff (`retries=6`); a `303` is
FOLLOWED (see `docs/troubleshooting.md`).

### `readout`

Request (what `LiveTools.readouts` sends; a server may accept more):

| field | type | meaning |
|---|---|---|
| `organism` | str | the tag to read; `base` = the unmodified control |
| `messages` | list | the conversation up to and including the last user turn (a system turn first if any) |
| `completion` | str | the assistant reply to append as the final turn; `""` reads the prompt only (user-turn samples are moved into `messages` as the last user turn) |
| `positions` | `"all"` | every non-system token. Servers may also accept `"P"` (fixed-quota set), a kind/region name, `"everything"` (incl. system), or `[int, ...]` |
| `layers` | list[int] | the layer grid (`layers=` on the driver; `[42]` for the `nla` arm) |
| `k` | int | samples per cell (verbalizers; `1` from the agent, up to 5 for fixed files) |
| `lens` | str | `olens` \| `jlens` \| `logit` \| `acts` \| any registered name |
| `seed` | int | for sampled lenses |

Response:

| field | type | required | meaning |
|---|---|---|---|
| `n_tokens` | int | yes | tokens in the rendered conversation |
| `organism` | str | yes | echoed |
| `lens` | str | yes | echoed; `LENS_NAME[lens]` names the page |
| `tokens` | `{"<pos>": str}` | yes | decoded token text per returned position (string keys) |
| `tags` | `{"<pos>": {"region", "kind"}}` | yes | see the vocabulary below |
| `readouts` | `{"<layer>": {"<pos>": [str, ...]}}` | yes | `k` strings per cell; every requested layer for every returned position |
| `n_matched` | int | no | positions matching `positions` before thinning; the page tells the agent when `n_matched > len(tokens)` |
| `k` | int | no | samples per cell as served |
| `extra` | `{"<metric>": {"<layer>": {"<pos>": number}}}` | no | per-cell numbers; `kl` is rendered as `(KL to final=…)` on the logit-lens page; `fve` is reserved for the `fve` presentation |
| `why` | `{"<layer>:<pos>": str}` | no | set by the `llm` presentation, rendered as `[kept: …]` |
| `base_readouts` | like `readouts` | no | rendered under each cell as `base model here: …` (a contrast presentation hook) |

Vocabulary (`tools/positions.py`):

* `region`: `system` (never returned for `"all"`), `user`, `header` (the assistant turn's
  control tokens up to and including the newline after `</think>` — "about to speak"), `reply`.
* `kind`: `boundary` (user `<|im_end|>`, the `assistant` role token, `</think>`, the last
  header token), `punct` (a token whose decoded form has no alphanumerics and contains one of
  `. , : ; ? ! \n`), `user` / `header` / `reply` (any other token on the dense grid), `user4` /
  `reply4` (stride-4 samples on the fixed-quota set P). The `summary` presentation keeps
  `boundary` + `punct`.

Rules: thin positions, never layers (the agent reads a depth profile per position); decode token
strings to readable text (`util.text.decode_byte_level`); cap the input length and refuse
politely rather than crash (a sticky CUDA assert kills the container for every later request).

### `readout` with `lens="acts"` (out-of-process lenses)

Same request; the response carries the residuals instead of text:

```
{"n_tokens", "organism", "lens": "acts", "n_matched",
 "positions": [int, ...],                 # the returned positions, in order
 "d": 5120, "dtype": "float16",
 "vectors": {"<layer>": "<base64 of an fp16 [n_positions, d] row-major array>"},
 "tokens": {...}, "tags": {...}}
```

The second hop (`nla=` URL): `POST {"vectors": <the base64 string>, "n", "d", "k", "seed"}` →
`{"texts": [[k strings] per row], "k"}`. `LiveClient.readout_nla` assembles a standard readout
response with a single layer, so tools, presentations and the viewer are unchanged.

### `fixed`

```
request  {"organism", "lens", "layers": [...], "k"}
response {"organism", "lens", "layers": [...], "k",
          "prompts": [{"id": "f00", "user": "<prompt>", "n_tokens": n,
                       "tokens": {...}, "tags": {...}, "readouts": {"<layer>": {"<pos>": [...]}}}, ...]}
```

`LiveTools.fixed_prompt_readouts` fetches it once per run and pages it by characters
(`FIXED_PAGE_CHARS`). The layers/k requested per lens are `FIXED_LAYERS` / `FIXED_K` in
`tools/live.py` (falling back to the run's `layers` and `k=1` for a registered lens).

### `chat` (only without `target=`)

```
request  {"organism", "messages", "system"?, "prefill"?, "n", "temperature", "max_new", "seed",
          "mode": "assistant" | "user_turn" | "raw", "text"?}
response {"replies": [{"text", "n_tokens", "truncated"}, ...], "n_prompt_tokens", "organism", "mode"}
```

`text` is the raw prompt for `mode="raw"`; the server prepends `prefill` to every reply.
