# ruff: noqa  (Modal launcher: not linted or type-checked; a reference implementation)
"""REFERENCE target server: vLLM serving ONE merged organism as an OpenAI-compatible API.

This is the target side of the harness. It has NO dependency on the private lens stack — only
on Modal, vLLM and a volume holding merged organism weights (``auditbench-merged:/<tag>``; the
merge step is not part of this repository). Any OpenAI-compatible chat server can replace it:
the driver only needs ``target=<base url>`` (see docs/contracts.md for what the harness sends).

vLLM's LoRA runtime was inert on this base model, so each organism is served from MERGED
weights; ``base`` = the pinned snapshot. One deploy per organism; the app name carries a short
slug of the tag so several serve side by side:

    AB_VLLM_TAG=secret_loyalty_td_r16 modal deploy servers/organism_vllm_modal.py
    # -> https://<ws>--ab-vllm-sec-loy-td-serve.modal.run  (see app_slug / tools.live.vllm_slug)

The API serves the model under the name ``organism`` (whatever the tag). Per request:
``chat_template_kwargs: {"enable_thinking": false}``; prefill = trailing assistant message +
``continue_final_message``; user-persona sampling and raw completion = ``/v1/completions``.
Point the driver at the template ``target='https://<ws>--{slug}-serve.modal.run'``.
"""

import os
import re
import subprocess

import modal

SNAPSHOT = "/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
TAG = os.environ.get("AB_VLLM_TAG", "base")
PORT = 8000
HF_PATH = "/hf"
MERGED_MOUNT = "/merged"
MAX_CONTAINERS = int(os.environ.get("AB_VLLM_CONTAINERS", "1"))
# Cold start is ~11 min (51 GiB load + ~80 s torch.compile + ~440 s profiling/warmup run), and
# Modal's proxy 303s any request older than ~150 s, so a scaled-down server looks like an endless
# redirect loop. Pin AB_VLLM_MIN=1 while an eval is running, then redeploy with 0 to release it.
MIN_CONTAINERS = int(os.environ.get("AB_VLLM_MIN", "0"))


def app_slug(tag: str) -> str:
    """Short app name so the web hostname fits the 63-char DNS label
    (``<workspace>--<app>-serve.modal.run``): ``secret_loyalty_td_r16`` -> ``ab-vllm-sec-loy-td``.
    A long name is silently unreachable at the obvious URL (bitten 2026-09-17)."""
    if tag == "base":
        return "ab-vllm-base"
    parts = re.sub(r"_r\d+$", "", tag).split("_")  # any rank: _r16, the KTO organisms' _r64
    return "ab-vllm-" + "-".join(p[:3] for p in parts[:-1]) + "-" + parts[-1]


app = modal.App(app_slug(TAG))
# the CHIVE recipe that served this snapshot (CUDA devel base: vLLM JIT-compiles flashinfer)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.25.1", "hf_transfer", "huggingface_hub")
    .env({"HF_HOME": HF_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1", "AB_VLLM_TAG": TAG})
)
# MODAL_DATA_ENV: mount the volumes of another Modal environment (the weights live in "main";
# a new environment gets its own spend limit but would otherwise start with empty volumes)
hf_cache = modal.Volume.from_name(
    "jlens-hf-cache",
    environment_name=os.environ.get("MODAL_DATA_ENV") or None,
    create_if_missing=True,
)
merged_vol = modal.Volume.from_name(
    "auditbench-merged",
    environment_name=os.environ.get("MODAL_DATA_ENV") or None,
    create_if_missing=True,
)


@app.function(
    image=image,
    gpu="H200",
    volumes={HF_PATH: hf_cache, MERGED_MOUNT: merged_vol},
    timeout=60 * 60 * 24,
    scaledown_window=60 * 60,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=64, target_inputs=32)
@modal.web_server(port=PORT, startup_timeout=30 * 60)
def serve() -> None:
    path = SNAPSHOT if TAG == "base" else f"{MERGED_MOUNT}/{TAG}"
    cmd = [
        "vllm",
        "serve",
        path,
        "--served-model-name",
        "organism",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.90",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]
    print("[vllm]", TAG, " ".join(cmd), flush=True)
    subprocess.Popen(cmd)
