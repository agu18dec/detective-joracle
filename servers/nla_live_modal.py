# ruff: noqa  (Modal launcher: not linted or type-checked; a reference implementation)
"""REFERENCE out-of-process lens: NLA-RL (Karvonen) verbalizing activations handed over by the
organism server.

Depends on the PRIVATE lens stack (``JORACLE_LENS_STACK``), not on this repository:

    global_workspace.olens_suite.karvonen    LENSES, NormMatchInjector, load_lens_model,
                                             make_generator, render_prompt, resolve_checkpoint
    scripts/olens_suite/runner_common.py     hf_secrets
    scripts/oracle_lens_evals/olens_sglang   (import-path side effects of the karvonen module)

It is the pattern for any lens that cannot share the target's process: the organism server's
``readout`` endpoint with ``lens="acts"`` returns the fp16 residual at ONE layer for every
non-system position of a conversation, base64-encoded; this app's ``verbalize`` endpoint turns
those rows into text. ``tools.live.LiveClient.readout_nla`` does the two hops and returns the
standard readout JSON, so nothing downstream knows the lens lived elsewhere.

    JORACLE_LENS_STACK=... HF_TOKEN=... AB_NLA_MIN=1 modal deploy servers/nla_live_modal.py
    -> https://<ws>--auditbench-nla-live-nla-verbalize.modal.run   (pass as nla= to the driver)

Endpoint:
  POST verbalize  {vectors: <base64 fp16 [n, d]>, n, d, k?, temperature?, seed?}
                  -> {texts: [[k strings] per row], lens, layer, k}
"""

import base64
import os
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[1]  # this repository (detective-joracle)
LENS_STACK = Path(os.environ.get("JORACLE_LENS_STACK", str(REPO.parent / "gw-auditbench-v2")))
for _p in (
    LENS_STACK / "scripts" / "olens_suite" / "workspace_bench",
    LENS_STACK / "scripts" / "olens_suite",
    Path("/root"),
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from runner_common import hf_secrets  # noqa: E402

# MODAL_DATA_ENV: mount the volume of another Modal environment (weights live in "main")
hf_cache = modal.Volume.from_name(
    "jlens-hf-cache",
    environment_name=os.environ.get("MODAL_DATA_ENV") or None,
    create_if_missing=True,
)

LENS = "nla-rl400"
LAYER = 42
BATCH_POS = 48
MAX_ROWS = 4096
MIN_CONTAINERS = int(os.environ.get("AB_NLA_MIN", "0"))

# AB_NLA_APP: a short name keeps the web hostname under Modal's 63-char label limit when the
# workspace name carries an environment suffix (e.g. AB_NLA_APP=nla)
app = modal.App(os.environ.get("AB_NLA_APP", "auditbench-nla-live"))
# wsbench_nla_modal.image, rebuilt with fastapi in the pip stage: Modal refuses a build step
# after add_local_*, and the web endpoint needs fastapi in the image.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        "transformers>=5.10",
        "peft>=0.17",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "pyyaml",
        "jaxtyping",
        "beartype",
        "einops",
        "tqdm",
        "fastapi[standard]",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root/app/src"})
    .add_local_dir(str(LENS_STACK / "src"), "/root/app/src", ignore=["**/__pycache__"])
    .add_local_dir(
        str(LENS_STACK / "scripts" / "oracle_lens_evals" / "olens_sglang"),
        "/root/app/olens_sglang",
        ignore=["**/__pycache__"],
    )
    .add_local_file(
        str(LENS_STACK / "scripts" / "olens_suite" / "runner_common.py"), "/root/runner_common.py"
    )
)


@app.cls(
    image=image,
    gpu="H200",
    volumes={"/hf": hf_cache},
    secrets=hf_secrets(),
    timeout=6 * 3600,
    scaledown_window=60 * 60,
    min_containers=MIN_CONTAINERS,
    max_containers=int(os.environ.get("AB_NLA_MAX", "6")),
    memory=131072,
    cpu=8,
)
# ONE request per container: the injector hook (NormMatchInjector) and the generator are shared
# state on the instance, so two concurrent verbalize calls overwrite each other's vectors — a shape
# error when the batch sizes differ, silently cross-contaminated readouts when they match
# (seen 2026-09-24 under 6 parallel readers). Throughput comes from containers, not threads.
@modal.concurrent(max_inputs=1)
class NLA:
    @modal.enter()
    def load(self) -> None:
        import dataclasses

        sys.path.insert(0, "/root/app/olens_sglang")
        from global_workspace.olens_suite.karvonen import (
            LENSES,
            NormMatchInjector,
            load_lens_model,
            make_generator,
            render_prompt,
            resolve_checkpoint,
        )

        self.spec = dataclasses.replace(LENSES[LENS], k=1)
        ckpt = resolve_checkpoint(self.spec)
        self.tok, self.model = load_lens_model(ckpt)
        ids, mp = render_prompt(self.tok, ckpt.meta)
        self.injector = NormMatchInjector(self.model, mp)
        self._ids = ids
        self._mk = make_generator
        # One serial warm-up generation before the endpoint opens: torch's lazily-initialised
        # linalg kernels (solve_triangular in the gated delta rule) raise "lazy wrapper should be
        # called at most once" when a fresh container's first calls arrive concurrently.
        import torch

        try:
            warm = make_generator(self.model, self.tok, ids, self.injector, self.spec)
            warm(torch.zeros(1, self.model.config.hidden_size, device="cuda"))
            print(f"[nla-live] warm-up generation ok", flush=True)
        except Exception as e:  # the endpoint still opens; the first real call will show the error
            print(f"[nla-live] warm-up failed: {type(e).__name__}: {e}", flush=True)
        print(f"[nla-live] {LENS} ready, prompt {len(ids)} toks, marker {mp}", flush=True)

    @modal.fastapi_endpoint(method="POST")
    def verbalize(self, req: dict) -> dict:
        """``{vectors: <base64 fp16 [n, d]>, n, d, k?, temperature?, seed?}`` ->
        ``{texts: [[k strings] per row]}``."""
        import numpy as np
        import torch

        n, d = int(req["n"]), int(req["d"])
        if n > MAX_ROWS:
            return {"error": f"{n} rows over the {MAX_ROWS} cap"}
        raw = base64.b64decode(req["vectors"])
        vec = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).reshape(n, d).copy()).float()
        k = max(1, min(int(req.get("k") or 1), 4))
        import dataclasses

        spec = dataclasses.replace(self.spec, k=k)
        generate = self._mk(
            self.model,
            self.tok,
            self._ids,
            self.injector,
            spec,
            temperature=float(req.get("temperature", 1.0)),
            top_p=0.95,
            top_k=0,
        )
        torch.manual_seed(int(req.get("seed") or 0))
        out: list = []
        for s in range(0, n, BATCH_POS):
            out += generate(vec[s : s + BATCH_POS].to("cuda"))
        return {"texts": out, "lens": "nla", "layer": LAYER, "k": k}
