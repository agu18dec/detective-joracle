"""Capture subprocess (in-container): residuals of the base model at the requested layers.

  python -m capture_job <job.json> <out.pt>

job.json = {"base": <snapshot dir>, "units": {pid: [layers]}, "tokens": {pid: [ids]}}.
Writes {(pid, layer): bf16 CPU tensor [n, d]} with torch.save. Run as a separate process so
the 54 GB of HF weights are fully released before vLLM takes the GPU.
"""

import json
import sys
import time

import torch
from global_workspace.olens_suite.runner import load_base, resolve_blocks
from jlens.hooks import ActivationRecorder


def main() -> None:
    job = json.loads(open(sys.argv[1]).read())
    t0 = time.time()
    _, model = load_base(job["base"], "cuda")
    model.eval()
    blocks = resolve_blocks(model)
    out = {}
    for pid, layers in job["units"].items():
        ids = job["tokens"][pid]
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad(), ActivationRecorder(blocks, at=layers) as rec:
            model(x, use_cache=False)
            for e in layers:
                a = rec.activations[e][0].detach().to("cpu")
                assert a.shape[0] == len(ids), (pid, e, tuple(a.shape), len(ids))
                out[f"{pid}|{e}"] = a
    torch.save(out, sys.argv[2])
    print(f"[capture] {len(out)} units in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
