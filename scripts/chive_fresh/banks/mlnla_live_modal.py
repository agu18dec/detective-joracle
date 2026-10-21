# ruff: noqa  (Modal launcher)
"""Modal app `auditbench-mlnla-live`: the multi-layer NLA as a LIVE verbalizer for the AuditBench
auditing game (detective-joracle `mlnla` arm, `LiveClient.readout_mlnla`).

Exactly the generation of the CHIVE banks (readers_modal.py `nla_worker`, reader `mlnla`):
camilablank/qwen3.6-27b-multilayer-nla-rl600 av/ LoRA merged on the official base (`prep mlnla`
output on volume chive-fresh-hf), actor prompt that names the layer (nla_meta template), Karvonen
norm-match ADD of the activation at the output of block 1 at the marker (vllm_lens_patch), bf16,
T=1.0 top-p 0.95 top-k 64, max 256 tokens, stop at </explanation>. Activations come from the
ORGANISM: the organism server's `readout` with lens="acts" (adapter active) returns base64 fp16
rows per layer; this app turns one layer's rows into k explanations each.

    set -a; . /dev/shm/compute.env; set +a
    MODAL_ENVIRONMENT=agam modal deploy chive_scale/banks_code/mlnla_live_modal.py
    -> https://safety-sahan-2-agam--auditbench-mlnla-live-mlnla-verbalize.modal.run   (driver mlnla=)

POST verbalize {vectors: <base64 fp16 [n, d]>, n, d, layer, k?, seed?} -> {texts: [[k strings]], lens, layer, k}
"""

import base64
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from readers_modal import HF, MERGED, _clean_expl, _with_gdn_triton, hf_secret, hf_vol  # noqa: E402
from readers_modal import image as banks_image  # noqa: E402

app = modal.App("auditbench-mlnla-live")
image = banks_image  # fastapi is in the bank image's pip list
MAX_ROWS = 2048
MIN_CONTAINERS = int(os.environ.get("AB_MLNLA_MIN", "0"))


@app.cls(
    image=image,
    gpu="H200",
    volumes={HF: hf_vol},
    secrets=[hf_secret],
    timeout=6 * 3600,
    scaledown_window=60 * 60,
    min_containers=MIN_CONTAINERS,
    max_containers=int(os.environ.get("AB_MLNLA_MAX", "6")),
    memory=65536,
    cpu=8,
)
@modal.concurrent(max_inputs=16)
class MLNLA:
    @modal.enter()
    async def load(self) -> None:
        import yaml
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        from global_workspace.olens_suite.karvonen import is_multilayer, render_prompt

        av_dir = Path(MERGED["mlnla"])
        assert (av_dir / "MERGE_DONE").exists(), "run readers_modal prep for mlnla first"
        meta = yaml.safe_load((av_dir / "nla_meta.yaml").read_text())
        ex = meta["extraction"]
        assert is_multilayer(meta), "expected the multi-layer NLA checkpoint"
        self.d_model = int(ex["d_model"])
        self.trained = [int(x) for x in ex.get("layers", [ex["layer_index"]])]
        tok = AutoTokenizer.from_pretrained(str(av_dir))
        self.prompts = {e: render_prompt(tok, meta, e) for e in self.trained}
        eargs = dict(model=str(av_dir), dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.88,
                     enable_prefix_caching=False, max_num_seqs=256, seed=0,
                     limit_mm_per_prompt={"image": 0, "video": 0})
        os.environ["CHIVE_STEER_LAYERS"] = "1"  # vllm_lens_patch: hooks act on block 1 only
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**_with_gdn_triton(eargs)))
        await self.engine.collective_rpc("install_hooks")
        self.n = 0
        print(f"[mlnla-live] ready: layers {self.trained}, d={self.d_model}", flush=True)

    @modal.fastapi_endpoint(method="POST")
    async def verbalize(self, req: dict) -> dict:
        import asyncio

        import numpy as np
        import torch
        from vllm import SamplingParams

        n, d, layer = int(req["n"]), int(req["d"]), int(req["layer"])
        if layer not in self.prompts:
            return {"error": f"layer {layer} not trained; multi-layer NLA reads {self.trained}"}
        if d != self.d_model:
            return {"error": f"d={d}, expected {self.d_model}"}
        if n > MAX_ROWS:
            return {"error": f"{n} rows over the {MAX_ROWS} cap"}
        raw = base64.b64decode(req["vectors"])
        vec = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).reshape(n, d).copy()).float()
        k = max(1, min(int(req.get("k") or 1), 4))
        seed = int(req.get("seed") or 0)
        ids, mp = self.prompts[layer]

        async def one(i: int, j: int) -> str:
            steer = {"vec": vec[i].tolist(), "layer": 1, "pos": mp, "norm_match": True}
            sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=256,
                                stop=["</explanation>"], seed=(seed * 7919 + i * 31 + j) % (1 << 31),
                                extra_args={"chive_steer": steer})
            self.n += 1
            final = None
            async for out in self.engine.generate({"prompt_token_ids": ids}, sp, f"r{self.n}_{layer}_{i}_{j}"):
                final = out
            return _clean_expl(final.outputs[0].text)

        res = await asyncio.gather(*(one(i, j) for i in range(n) for j in range(k)))
        return {"texts": [res[i * k:(i + 1) * k] for i in range(n)], "lens": "mlnla", "layer": layer, "k": k}
