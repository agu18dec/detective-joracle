"""Image build step: patch vllm-lens's per-request steering lookup (installed copy).

1. O(1) lookup: upstream `_find_steering_configs` scans every registered steering entry for every
   running request at every hooked layer of every step (O(R^2 * L) string compares).
2. Inline steering: upstream registers each request's vector with a synchronous collective_rpc
   (`set_steering_data`) that is served between engine steps, which admitted ~1 request per step
   (measured 0.7-1.8 rollouts/s on H200). Here a request may instead carry its vector in
   ``SamplingParams.extra_args["chive_steer"] = {"vec": [d floats], "layer": int, "pos": int,
   "norm_match": bool}``; the worker builds the SteeringVector itself on first sight (cached on
   the request's extra_args) and the plugin's RPC path is never touched (it only reacts to
   ``apply_steering_vectors``). Same `_apply_steering` math (norm_match against the full
   residual stream at the output of block `layer`).
3. Layer filter: with env CHIVE_STEER_LAYERS="1" the per-layer hook returns at once on every
   other layer (upstream runs its per-request Python loops at all 64 layers every step).
vLLM's internal id is "{external_id}-{random_suffix}"; our external ids contain no "-".
"""

import vllm_lens._worker_ext as m

PATCH = """

# --- chive-fresh-banks patch (see banks_code/vllm_lens_patch.py) ---
def _find_steering_configs(extension, internal_req_id, extra_args):
    results = []
    data = extension._steering_data
    ext = internal_req_id.rsplit("-", 1)[0]
    if ext in data:
        results.extend(data[ext])
    if extra_args:
        inline = extra_args.get("chive_steer")
        if inline is not None:
            cached = extra_args.get("_chive_sv")
            if cached is None:
                import torch as _t
                from vllm_lens._helpers.types import SteeringVector as _SV
                p0 = next(extension.model_runner.model.parameters())
                vec = _t.tensor(inline["vec"], dtype=_t.float32).to(device=p0.device,
                                                                     dtype=p0.dtype)
                cached = [_SV(activations=vec.view(1, 1, -1), layer_indices=[int(inline["layer"])],
                              norm_match=bool(inline.get("norm_match", True)),
                              position_indices=[int(inline["pos"])])]
                extra_args["_chive_sv"] = cached
            results.extend(cached)
        steering_id = extra_args.get("_steering_id")
        if steering_id and steering_id in data:
            results.extend(data[steering_id])
    return results


import os as _chive_os
_chive_orig_hook_inner = _hook_inner
_CHIVE_LAYERS = {int(x) for x in _chive_os.environ.get("CHIVE_STEER_LAYERS", "").split(",") if x}


def _hook_inner(extension, layer_idx, output):
    if _CHIVE_LAYERS and layer_idx not in _CHIVE_LAYERS:
        return None
    return _chive_orig_hook_inner(extension, layer_idx, output)
"""
path = m.__file__
src = open(path).read()
if "chive-fresh-banks patch" not in src:
    open(path, "a").write(PATCH)
print("patched", path)
