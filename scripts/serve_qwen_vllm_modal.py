"""Dedicated OpenAI-compatible vLLM server for Qwen3.6-27B on Modal: the CHIVE forensics target.

Serves ``Qwen/Qwen3.6-27B`` (bf16, H200) behind vLLM's OpenAI-compatible HTTP API so the
CHIVE pipeline (github.com/adamkarvonen/chive, checked out at /workspace/agam/chive) can
sample it with a ``vllm:Qwen/Qwen3.6-27B@https://<this-url>`` backend spec. CHIVE's client
sends ``chat_template_kwargs: {"enable_thinking": false}`` per request, which is the repo's
required non-thinking mode for this model — nothing thinking-related is baked in here.

Fan-out: up to ``MAX_CONTAINERS`` single-H200 replicas; Modal load-balances requests across
them (this replaces chive's tools/vllm_pool). Per the repo's Modal rule, k replicas cost the
same total as 1 for a fixed amount of work and finish ~k-times faster during stage-1 sampling.

Deploy (prints the public URL, ends with ``…-serve.modal.run``):

    uvx --python 3.12 modal@latest deploy scripts/serve_qwen_vllm_modal.py

A copy of gw-chive-hard's chive-qwen-vllm (same model, vLLM pin and flags) under its own app
name, so these evals do not queue behind other workloads on the shared endpoint.

Smoke test:

    curl https://<url>/v1/models
"""

import subprocess

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
PORT = 8000
MAX_CONTAINERS = 8
HF_PATH = "/hf"

app = modal.App("chive-forensics-qwen")

# CUDA *devel* base (not debian_slim): vLLM JIT-compiles flashinfer kernels at boot and
# needs nvcc + /usr/local/cuda. Same base + vllm pin as oracle_modal.py::image_vllm.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.25.1", "hf_transfer", "huggingface_hub")
    .env({"HF_HOME": HF_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H200",
    volumes={HF_PATH: hf_cache},
    timeout=60 * 60 * 24,
    scaledown_window=20 * 60,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=32, target_inputs=16)
@modal.web_server(port=PORT, startup_timeout=30 * 60)
def serve() -> None:
    """Launch ``vllm serve``; Modal proxies HTTP to it once the port opens."""
    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
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
    subprocess.Popen(cmd)
