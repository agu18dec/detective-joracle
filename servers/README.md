# Reference servers

These three Modal launchers are the GPU side of the harness as it was run: the target, the lens,
and one out-of-process lens. They are **reference implementations**, kept readable and runnable
in spirit, and they are the part a user replaces to bring their own lens. They are outside
ruff/mypy on purpose.

| file | role | GPU | depends on the private lens stack? |
|---|---|---|---|
| `organism_vllm_modal.py` | **target**: vLLM serving one merged organism as an OpenAI-compatible API | H200 | no (Modal + vLLM + a volume of merged weights) |
| `organism_server_modal.py` | **lens server**: HF + PEFT; `chat`, `readout` (OLens / J-lens / logit / raw `acts`), `fixed`, `info` | H200 | **yes** |
| `nla_live_modal.py` | **out-of-process lens**: NLA-RL verbalizer over `acts` rows | H200 | **yes** |

## The private lens stack

`organism_server_modal.py` and `nla_live_modal.py` import from a research monorepo that is not
part of this repository. Set `JORACLE_LENS_STACK=/path/to/that/checkout` (default: a sibling
directory `../gw-auditbench-v2`); the launchers put its `src/` on the image and its script dirs
on `sys.path`. The imports, so you know exactly what to replace:

```
global_workspace.olens_suite.runner       S3D_RL600 contract, load_base, resolve_blocks,
                                          make_sampler, PROBE_TEXT, _fetch_adapter
global_workspace.ola.verbalizer           renderer_for  (the verbalizer prompt per layer)
global_workspace.lens                     stacked_jacobians, jlens_token_norms, cosine_readout
global_workspace.olens_suite.karvonen     LENSES, NormMatchInjector, load_lens_model,
                                          make_generator, render_prompt, resolve_checkpoint (NLA)
jlens.lens.JacobianLens, jlens.hooks.ActivationRecorder   (vendored anthropics/jacobian-lens)
scripts/olens_suite/runner_common.py      hf_secrets
scripts/olens_suite/workspace_bench/wsbench_capture_modal.py
                                          APP_REPO, HF_MOUNT, ORG_MOUNT, hf_cache,
                                          organisms_vol, resolve_adapter
```

From **this** repository they use only `detective_joracle.tools.positions` (region/kind
tagging), `detective_joracle.util.text.decode_byte_level`, `detective_joracle.agent.prompts.
PRISM_SYSTEM` and `detective_joracle.registry.quirks.load_held_out_prompts`.

## Endpoints

`organism_server_modal.py` (Modal web endpoints `https://<ws>--auditbench-organism-organism-
<name>.modal.run`; pass the prefix as `server=`):

* `POST chat` — `{organism, messages, system?, prefill?, n, temperature, max_new, seed, mode:
  assistant|user_turn|raw, text?}` → `{replies: [{text, n_tokens, truncated}], n_prompt_tokens,
  organism, mode}`. The organism adapter is ON; `organism="base"` = all adapters off.
* `POST readout` — the contract in `docs/contracts.md`. `lens="olens"` samples the verbalizer
  (k per cell, T=0.8, one batched generate for all layers); `lens="jlens"` returns the top-10
  cosine-readout tokens; `lens="logit"` the top-10 unembedded tokens plus `extra.kl`;
  `lens="acts"` the fp16 residual rows (base64) for an out-of-process lens. Positions:
  `"all"` (every non-system token), `"P"`, a kind/region name, `"everything"`, or a list of
  indices. Caps: `MAX_CELLS=4400` (thins positions, never layers), `MAX_READ_TOKENS=8192`
  (a longer conversation is refused — an over-long input triggers a sticky CUDA assert that
  kills the container for every later request), `k<=4`, `n<=4`.
* `POST fixed` — the precomputed held-out-prompt readouts for `(organism, lens, layers, k)`;
  build them with `modal run servers/organism_server_modal.py --precompute a,b,c --lens olens
  --layers 20,36,40,44,52,60 --k 5` (`--lens acts --layers 42` writes the NLA input).
* `POST info` — `{contract, ao_layers, jlens_layers, loaded}`.

`nla_live_modal.py`: `POST verbalize` — `{vectors: <base64 fp16 [n, d]>, n, d, k?, temperature?,
seed?}` → `{texts: [[k strings] per row], lens, layer, k}`. Pass its URL as `nla=`.

`organism_vllm_modal.py`: standard `/v1/chat/completions`, `/v1/completions`, `/v1/models`
under the model name `organism`; one deploy per organism (`AB_VLLM_TAG=<tag>`), app name
`app_slug(tag)` (mirrored by `tools.live.vllm_slug`). Pass the templated URL as `target=`.

## Bringing your own lens

Keep the readout contract and write the smallest server that satisfies it. The shape of
`organism_server_modal.py::_readout` is the recipe:

1. render the conversation with the target's chat template (`messages` + `completion` as the
   assistant turn), tokenize;
2. tag positions with `tools.positions.position_set_all(ids, tokens, ControlIds.from_tokenizer(tok))`
   (drop `system`);
3. one forward pass with hooks at the requested layers on the ORGANISM (adapter on);
4. decode each `(layer, position)` residual with your lens into a list of strings;
5. return `{n_tokens, organism, lens, n_matched, tokens, tags, readouts, extra?}` with decoded
   token strings (`util.text.decode_byte_level`).

If the lens cannot live in the target's process, return `acts` rows instead and serve the
decoder separately (`nla_live_modal.py` is the pattern; `LiveClient.readout_nla` does the hops).
Then `register_lens_context("<name>", "<display>", "<one paragraph>")` client-side and run
`arms=<name>`.

`examples/mock_target.py` is the same contract with no model behind it — useful to develop the
client side or a presentation before a GPU is involved.
