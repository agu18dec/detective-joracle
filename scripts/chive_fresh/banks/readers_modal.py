# ruff: noqa  (Modal launcher)
"""Modal app `chive-fresh-banks`: detective-joracle harness readout banks for NEW CHIVE incidents.

Rebuild of the lost `scripts/chive_bank/readers_modal.py`. One entry point, four readers, at the
settings of the existing 300-incident banks (volumes `chive-bank-readers`, `chive-dense`):

  --reader olens  OLens RL `s3d_rl600` (olens_suite.runner.S3D_RL600: concepts_raw prompt, unit
                  transform alpha 16000), k=4, max_new 128, T=0.8 top-p 0.95, every position,
                  layers 20..60 step 4. Bank `s3d_rl600_all`.
                  --engine vllm (default): merged checkpoint on vLLM, prompt_embeds with the slot
                    row = 16000*h/||h||; per-request seeds. ~37 rollouts/s per H200.
                  --engine hf: the scripts/chive_eval/dense_olens_modal.py path (load_base,
                    renderer_for, make_sampler, dense_seed per 60-position chunk) with the adapter
                    merged for generation; reproduces the existing bank's samples nearly verbatim
                    at the same seeds. ~8 rollouts/s per H200 (4-5x the cost).
  --reader nla    NLA-RL (ceselder/qwen3.6-27b-nla-rl av_base + av_rl_adapters/iter_000400) at L42,
                  k=4: Karvonen L1 norm-match ADD at the output of AV block 1 at the marker,
                  T=1.0 top-p .95 top-k 64, max_new 256, stop </explanation>, enable_thinking=False
                  prompt. vLLM + vllm-lens (patched: vllm_lens_patch.py). Bank `nla_k4`.
  --reader mlnla  multi-layer NLA (camilablank/qwen3.6-27b-multilayer-nla-rl600 av/ LoRA on the
                  official base; the actor prompt names the layer), k=4, layers 20..60 step 4,
                  same sampling and injection. Bank `mlnla_k4_chive`.
  --reader jlens  J-lens n1000 wikitext (neuronpedia, official -1/16 recipe), cosine readout
                  (W_U J h)/||J^T W_U[t]|| in fp32 without TF32, raw top-10 at every position and
                  every layer L0..L62. Bank `jlens_all`.

Input: a harness dataset dir `<ds>/data/tokens.jsonl` (token_ids per prompt_id). Activations are
captured (HF, output of decoder block L) from THOSE token ids, so the loader's token-id check
passes. Output: `<out-volume>:/<bank>/<prompt_id>.json.gz` = {"token_ids", "readouts": {layer:
{pos: [samples]}}, "k", "layers", "coverage_note", "provenance"}; per-layer partials under
`<bank>/_parts/` are merged (and removed) by `assemble`. Skip-if-exists at (incident, layer)
granularity: rerun the same command to resume. --num-shards 0 (default) sizes the fan-out to
~2 GPU-h of generation per container (bank_common.RATE).

Usage (from this directory, after `source ../env.sh`):

  modal run readers_modal.py --reader jlens --dataset <ds> --ids ids.txt \
      --out-volume chive-fresh-banks --bank jlens_all
  ./fetch_banks.sh jlens_all          # -> $CHIVE_BANKS/jlens_all/ (default /workspace/agam/chive_scale/banks)

Pinned revisions: bank_common.py. HF snapshots and the merged checkpoints live on the NEW volume
`chive-fresh-hf` (`prep` fills it once per reader; idempotent, skipped with --skip-prep).
GPU list: env CHIVE_BANKS_GPU (default H200; e.g. H200,B200 to widen).
Container code: env GW_SRC = a global-workspace ``src/`` dir (``global_workspace`` + ``jlens``)
uploaded into the image. Default: the vendored snapshot the fresh banks were built with,
``vendor/gw_src/src`` (provenance in vendor/gw_src/README.md; it differs from global-workspace
main in olens_suite/runner.py, olens_suite/karvonen.py and ola/verbalizer.py).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
if (HERE / "bank_common.py").exists():
    sys.path.insert(0, str(HERE))
import bank_common as bc  # noqa: E402

APP = "chive-fresh-banks"
OUT_VOLUME = os.environ.get("CHIVE_BANKS_VOLUME", "chive-fresh-banks")
GPU = os.environ.get("CHIVE_BANKS_GPU", "H200").split(",")
HF = "/hf"
OUT = "/out"
GW_SRC = Path(os.environ.get("GW_SRC") or HERE / "vendor" / "gw_src" / "src")
MERGED = {
    "nla": f"{HF}/merged/nla_rl400_vl",
    "mlnla": f"{HF}/merged/mlnla_rl600_vl",
    "olens": f"{HF}/merged/olens_s3d_rl600_vl",
    "olens_single": f"{HF}/merged/olens_s3_single_rl500_vl",
}

app = modal.App(APP)

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git")
    .uv_pip_install(
        "vllm==0.25.1",
        "vllm-lens @ git+https://github.com/adamkarvonen/vllm-lens.git@218e78d79af1a9c55ae76d850a6fb2c08674b0f1",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "hf_transfer",
        "huggingface_hub",
        "pyyaml",
        "jaxtyping",
        "beartype",
        "einops",
    )
    .add_local_file(str(HERE / "vllm_lens_patch.py"), "/root/vllm_lens_patch.py", copy=True)
    .run_commands("python /root/vllm_lens_patch.py")
    .env(
        {
            "HF_HOME": HF,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/root/gw/src:/root/bc",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    .add_local_dir(str(GW_SRC), "/root/gw/src", ignore=["**/__pycache__"])
    .add_local_file(str(HERE / "bank_common.py"), "/root/bc/bank_common.py")
    .add_local_file(str(HERE / "capture_job.py"), "/root/bc/capture_job.py")
)

hf_vol = modal.Volume.from_name("chive-fresh-hf", create_if_missing=True)
out_vol = modal.Volume.from_name(OUT_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])
GPU_KW = dict(
    image=image,
    gpu=GPU,
    volumes={HF: hf_vol, OUT: out_vol},
    secrets=[hf_secret],
    timeout=20 * 3600,
    memory=65536,
    cpu=8,
)


# ---------------------------------------------------------------------------------------------
# in-container helpers
# ---------------------------------------------------------------------------------------------
def _offline() -> None:
    """Jobs read only pinned snapshots already on `chive-fresh-hf` (Ops rule: HF_HUB_OFFLINE)."""
    os.environ["HF_HUB_OFFLINE"] = "1"


def _base_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(bc.BASE_MODEL, revision=bc.BASE_REV)


def _load_base_hf() -> tuple[Any, Any, Any]:
    """Tokenizer + bf16 base (olens_suite.runner.load_base: left padding, eos pad) + blocks."""
    from global_workspace.olens_suite.runner import load_base, resolve_blocks

    tok, model = load_base(_base_path(), "cuda")
    model.eval()
    return tok, model, resolve_blocks(model)


def _capture(model: Any, blocks: Any, token_ids: list[int], layers: list[int]) -> dict[int, Any]:
    """One forward over the incident's token ids; output of decoder block L for each L ->
    CPU bf16 [n, d] (the model's own dtype, so storing bf16 is lossless)."""
    import torch
    from jlens.hooks import ActivationRecorder

    ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    with torch.no_grad(), ActivationRecorder(blocks, at=layers) as rec:
        model(ids, use_cache=False)
        out = {e: rec.activations[e][0].detach().to("cpu") for e in layers}
    for e, a in out.items():
        assert a.shape[0] == len(token_ids), (e, tuple(a.shape), len(token_ids))
    return out


def _capture_subprocess(todo: list[tuple[str, int]], tokens: dict[str, list[int]]) -> dict:
    """Phase A for the vLLM readers: capture in a child process (capture_job.py) so the HF
    weights are fully released before vLLM claims the GPU. -> {(pid, layer): bf16 [n, d]}."""
    import subprocess

    import torch

    units: dict[str, list[int]] = {}
    for pid, layer in todo:
        units.setdefault(pid, []).append(layer)
    job = {
        "base": _base_path(),
        "units": {p: sorted(v) for p, v in units.items()},
        "tokens": {p: tokens[p] for p in units},
    }
    Path("/tmp/capture_job.json").write_text(json.dumps(job))
    subprocess.run(
        [sys.executable, "-m", "capture_job", "/tmp/capture_job.json", "/tmp/capture_acts.pt"],
        check=True,
        env=dict(os.environ),
    )
    raw = torch.load("/tmp/capture_acts.pt")
    os.remove("/tmp/capture_acts.pt")
    return {(key.split("|")[0], int(key.split("|")[1])): a for key, a in raw.items()}


def _with_gdn_triton(eargs: dict) -> dict:
    """Skip FlashInfer's GDN-prefill JIT (8 min per container measured) when the arg exists."""
    import dataclasses

    from vllm import AsyncEngineArgs

    if "gdn_prefill_backend" in {f.name for f in dataclasses.fields(AsyncEngineArgs)}:
        eargs = {**eargs, "gdn_prefill_backend": "triton"}
    return eargs


def _todo_units(bank: str, units: list[tuple[str, int]]) -> list[tuple[str, int]]:
    root = Path(OUT)
    return [
        (pid, layer)
        for pid, layer in units
        if not bc.final_path(root, bank, pid).exists()
        and not bc.part_path(root, bank, pid, layer).exists()
    ]


def _write_part(
    bank: str, pid: str, layer: int, n: int, per_pos: dict[str, list[str]], extra: dict[str, Any]
) -> None:
    bc.write_gz(
        bc.part_path(Path(OUT), bank, pid, layer),
        {"prompt_id": pid, "layer": layer, "n_tokens": n, "readouts": per_pos, **extra},
    )
    out_vol.commit()


def _merge_vl(
    dst: Path,
    vl_dir: Path,
    lm_dir: Path | None,
    lora_dir: Path,
    tok_src: Path,
    meta_src: Path | None,
) -> str:
    """A servable VL-arch checkpoint = the official VL shards, with every language-model tensor
    taken from `lm_dir` (when given: NLA's full-finetune av_base) plus the LoRA delta. Pure
    state-dict surgery, generalised from nla_qwen36_precompute_modal.remap_av."""
    import math
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if (dst / "MERGE_DONE").exists():
        return f"{dst}: already merged"
    acfg = json.loads((lora_dir / "adapter_config.json").read_text())
    assert not acfg.get("use_dora") and float(acfg.get("lora_dropout", 0.0)) == 0.0, acfg
    r, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
    scaling = alpha / math.sqrt(r) if acfg.get("use_rslora") else alpha / r

    def vl_name(module: str) -> str:
        m = module.replace("._orig_mod", "")
        m = m.removeprefix("base_model.model.")
        if m.startswith("model.layers."):
            m = "model.language_model.layers." + m[len("model.layers.") :]
        assert m.startswith("model.language_model.layers."), module
        return m + ".weight"

    lora_a: dict[str, Any] = {}
    lora_b: dict[str, Any] = {}
    with safe_open(str(lora_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():  # noqa: SIM118
            module, ab = k.rsplit(".lora_", 1)
            (lora_a if ab.startswith("A") else lora_b)[vl_name(module)] = f.get_tensor(k)
    assert set(lora_a) == set(lora_b) and lora_a, (len(lora_a), len(lora_b))

    lm_index: dict[str, str] = {}
    if lm_dir is not None:
        lm_index = json.loads((lm_dir / "model.safetensors.index.json").read_text())["weight_map"]

    dst.mkdir(parents=True, exist_ok=True)
    applied: set[str] = set()
    replaced = 0
    for shard in sorted(vl_dir.glob("model-*.safetensors")):
        tensors: dict[str, Any] = {}
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                t = f.get_tensor(name)
                is_lm = name.startswith("model.language_model.") or name == "lm_head.weight"
                if lm_index and is_lm and name in lm_index:
                    with safe_open(str(lm_dir / lm_index[name]), framework="pt") as g:
                        t = g.get_tensor(name)
                    replaced += 1
                if name in lora_a:
                    delta = (lora_b[name].float() @ lora_a[name].float()) * scaling
                    assert delta.shape == t.shape, (name, delta.shape, t.shape)
                    t = (t.float() + delta).to(torch.bfloat16)
                    applied.add(name)
                tensors[name] = t.contiguous()
        save_file(tensors, str(dst / shard.name), metadata={"format": "pt"})
        print(f"[merge] {dst.name}: wrote {shard.name}", flush=True)
    missing = set(lora_a) - applied
    assert not missing, f"{len(missing)} LoRA modules never met their weight: {sorted(missing)[:5]}"
    for f in vl_dir.iterdir():
        if (
            f.is_file()
            and not f.name.startswith("model-")
            and "tokenizer" not in f.name
            and f.name != "chat_template.jinja"
        ):
            shutil.copy(f, dst / f.name)
    for f in tok_src.iterdir():
        if f.is_file() and (
            "tokenizer" in f.name or f.name in ("chat_template.jinja", "vocab.json", "merges.txt")
        ):
            shutil.copy(f, dst / f.name)
    if meta_src is not None:
        shutil.copy(meta_src, dst / "nla_meta.yaml")
    (dst / "MERGE_DONE").write_text(
        f"official VL shards ({vl_dir}) + lm tensors from {lm_dir} ({replaced} replaced) + "
        f"{len(applied)} LoRA deltas from {lora_dir} (scale {scaling:.4f})\n"
    )
    return f"{dst}: merged {len(applied)} LoRA modules, {replaced} lm tensors replaced"


# ---------------------------------------------------------------------------------------------
# prep: pinned downloads + merged AV checkpoints (CPU, idempotent)
# ---------------------------------------------------------------------------------------------
@app.function(
    image=image, volumes={HF: hf_vol}, secrets=[hf_secret], memory=98304, cpu=16, timeout=4 * 3600
)
def prep(reader: str) -> str:
    from huggingface_hub import hf_hub_download, snapshot_download

    hf_vol.reload()
    t0 = time.time()
    msgs = []
    base = Path(snapshot_download(bc.BASE_MODEL, revision=bc.BASE_REV))
    msgs.append(f"base {base}")
    if reader == "jlens":
        msgs.append(hf_hub_download(bc.JLENS_REPO, bc.JLENS_FILE, revision=bc.JLENS_REV))
    elif reader in ("olens", "olens_single"):
        c = bc.OLENS_CONTRACTS[reader]
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            msgs.append(
                hf_hub_download(
                    c["repo"], f"{c['path']}/{f}", repo_type="dataset", revision=c["rev"]
                )
            )
        hf_vol.commit()
        lora = Path(msgs[-1]).parent  # the vLLM engine (--engine vllm) reads a merged copy
        msgs.append(_merge_vl(Path(MERGED[reader]), base, None, lora, base, None))
    elif reader == "nla":
        root = Path(
            snapshot_download(
                bc.NLA_REPO,
                revision=bc.NLA_REV,
                allow_patterns=[f"{bc.NLA_BASE_SUBDIR}/*", f"{bc.NLA_ADAPTER}/*", "nla_meta.yaml"],
            )
        )
        import yaml

        a = yaml.safe_load((root / "nla_meta.yaml").read_text())["prompt_templates"]["actor"]
        b = yaml.safe_load((root / bc.NLA_BASE_SUBDIR / "nla_meta.yaml").read_text())[
            "prompt_templates"
        ]["actor"]
        assert a == b, "repo-root and av_base actor templates differ"
        hf_vol.commit()
        msgs.append(
            _merge_vl(
                Path(MERGED["nla"]),
                base,
                root / bc.NLA_BASE_SUBDIR,
                root / bc.NLA_ADAPTER,
                root / bc.NLA_BASE_SUBDIR,
                root / bc.NLA_BASE_SUBDIR / "nla_meta.yaml",
            )
        )
    elif reader == "mlnla":
        root = Path(
            snapshot_download(
                bc.MLNLA_REPO,
                revision=bc.MLNLA_REV,
                allow_patterns=[
                    f"{bc.MLNLA_ADAPTER}/adapter_config.json",
                    f"{bc.MLNLA_ADAPTER}/adapter_model.safetensors",
                    "nla_meta.yaml",
                ],
            )
        )
        hf_vol.commit()
        msgs.append(
            _merge_vl(
                Path(MERGED["mlnla"]),
                base,
                None,
                root / bc.MLNLA_ADAPTER,
                base,
                root / "nla_meta.yaml",
            )
        )
    else:
        raise ValueError(reader)
    hf_vol.commit()
    return f"prep {reader} ({time.time() - t0:.0f}s): " + " | ".join(msgs)


# ---------------------------------------------------------------------------------------------
# J-lens
# ---------------------------------------------------------------------------------------------
@app.function(**GPU_KW)
def jlens_worker(
    bank: str, items: list[tuple[str, list[int]]], layers: list[int], topk: int, provenance: dict
) -> dict:
    _offline()
    import gc

    import torch
    from huggingface_hub import hf_hub_download
    from jlens.lens import JacobianLens

    from global_workspace.lens import cosine_readout, jlens_token_norms, stacked_jacobians

    hf_vol.reload()
    out_vol.reload()
    root = Path(OUT)
    todo = [(p, ids) for p, ids in items if not bc.final_path(root, bank, p).exists()]
    summary: dict[str, Any] = {"done": [], "skipped": len(items) - len(todo), "sec": {}}
    if not todo:
        return summary
    torch.backends.cuda.matmul.allow_tf32 = False  # the bank of record: fp32, no TF32
    torch.backends.cudnn.allow_tf32 = False
    t_load = time.time()
    tok, model, blocks = _load_base_hf()
    lens = JacobianLens.from_pretrained(
        hf_hub_download(bc.JLENS_REPO, bc.JLENS_FILE, revision=bc.JLENS_REV)
    )
    jac = stacked_jacobians(lens, device="cpu", dtype=torch.float32)[list(layers)].to("cuda")
    del lens
    gc.collect()
    w_u = model.lm_head.weight.detach().float().to("cuda")
    den = jlens_token_norms(jac, w_u)
    summary["load_sec"] = round(time.time() - t_load, 1)
    print(
        f"[jlens] stack {tuple(jac.shape)}, V={w_u.shape[0]}, load {summary['load_sec']}s",
        flush=True,
    )
    rep: dict[int, str] = {}

    def tok_repr(t: int) -> str:
        if t not in rep:
            rep[t] = repr(tok.decode([t]))
        return rep[t]

    chunk = 32  # [63, 32, V] fp32 scores = 2 GB
    for pid, ids in todo:
        t0 = time.time()
        acts = _capture(model, blocks, ids, list(layers))
        n = len(ids)
        readouts: dict[str, dict[str, list[str]]] = {str(e): {} for e in layers}
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            resid = torch.stack([acts[e][lo:hi].to("cuda").float() for e in layers])
            scores = cosine_readout(jac, resid, w_u, denom=den)  # [L, m, V]
            tops = scores.topk(topk, dim=-1).indices.cpu()
            del scores, resid
            for li, e in enumerate(layers):
                cells = readouts[str(e)]
                for pi in range(hi - lo):
                    cells[str(lo + pi)] = [" ".join(tok_repr(int(t)) for t in tops[li, pi])]
        rec = {
            "prompt_id": pid,
            "reader": bank,
            "token_ids": ids,
            "n_tokens": n,
            "layers": list(layers),
            "k": 1,
            "readouts": readouts,
            "coverage_note": "every token",
            "provenance": provenance,
        }
        probs = bc.validate_record(rec, ids, layers, 1)
        if probs:
            raise RuntimeError(f"{pid}: invalid record: {probs}")
        bc.write_gz(bc.final_path(root, bank, pid), rec)
        out_vol.commit()
        dt = time.time() - t0
        summary["done"].append(pid)
        summary["sec"][pid] = round(dt, 1)
        print(f"[jlens] {pid}: {n} tok x {len(layers)} layers in {dt:.1f}s", flush=True)
    return summary


# ---------------------------------------------------------------------------------------------
# OLens (s3d_rl600), HF generate with the merged adapter
# ---------------------------------------------------------------------------------------------
def _olens_adapter_dir() -> Path:
    """The pinned S3D_RL600 adapter with `._orig_mod` (torch.compile) stripped from its keys, so
    it loads onto plain blocks (equivalent to runner.load_lens's compile-wrap; the inert-adapter
    guard below checks it took)."""
    import shutil

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file

    files = {
        f: Path(
            hf_hub_download(
                bc.OLENS_REPO, f"{bc.OLENS_PATH}/{f}", repo_type="dataset", revision=bc.OLENS_REV
            )
        )
        for f in ("adapter_config.json", "adapter_model.safetensors")
    }
    dst = Path("/tmp/olens_lora")
    dst.mkdir(exist_ok=True)
    sd = load_file(str(files["adapter_model.safetensors"]))
    n_orig = sum("._orig_mod" in k for k in sd)
    save_file(
        {k.replace("._orig_mod", ""): v for k, v in sd.items()},
        str(dst / "adapter_model.safetensors"),
    )
    shutil.copy(files["adapter_config.json"], dst / "adapter_config.json")
    print(f"[olens] adapter: {len(sd)} tensors, {n_orig} had ._orig_mod", flush=True)
    return dst


@app.function(**GPU_KW)
def olens_worker(
    bank: str,
    units: list[tuple[str, int]],
    tokens: dict[str, list[int]],
    k: int,
    max_new: int,
    temperature: float,
    top_p: float,
    seed_base: int,
    pos_chunk: int,
    provenance: dict,
) -> dict:
    _offline()
    import torch
    from peft import PeftModel

    from global_workspace.ola.verbalizer import renderer_for
    from global_workspace.olens_suite import runner

    hf_vol.reload()
    out_vol.reload()
    todo = _todo_units(bank, [tuple(u) for u in units])
    summary: dict[str, Any] = {"done": [], "skipped": len(units) - len(todo), "rate": []}
    if not todo:
        return summary
    contract = runner.S3D_RL600
    t_load = time.time()
    tok, base, blocks = _load_base_hf()
    # phase A: capture every unit's activations from the PLAIN base (= capture under
    # disable_adapter in the dense script), CPU bf16
    by_pid: dict[str, list[int]] = {}
    for pid, layer in todo:
        by_pid.setdefault(pid, []).append(layer)
    acts: dict[tuple[str, int], Any] = {}
    for pid, layers in by_pid.items():
        for e, a in _capture(base, blocks, tokens[pid], sorted(layers)).items():
            acts[(pid, e)] = a
    print(f"[olens] captured {len(acts)} units in {time.time() - t_load:.0f}s", flush=True)
    # phase B: adapter on (inert guard), merged (drift guard), sample
    lora = _olens_adapter_dir()
    model = PeftModel.from_pretrained(base, str(lora)).eval()
    probe = tok(runner.PROBE_TEXT, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        on = model(probe).logits[0, -1].float()
        with model.disable_adapter():
            off = model(probe).logits[0, -1].float()
    delta = float((on - off).abs().max().item())
    print(f"[olens] logit_delta={delta:.3f}", flush=True)
    if delta < 1e-3:
        raise RuntimeError("INERT adapter: the OLens LoRA changes nothing")
    model, drift = runner.merge_lens(model, probe, on)
    all_layers = sorted({e for _, e in todo})
    wv = {e: renderer_for(contract.prompt_kind)(tok, layer=e) for e in all_layers}
    sample = runner.make_sampler(
        model, tok, wv, 0.0, "cuda", transform=contract.transform, alpha=contract.alpha
    )
    summary["load_sec"] = round(time.time() - t_load, 1)
    for pid, layer in todo:
        n = len(tokens[pid])
        t0 = time.time()
        per_pos: dict[str, list[str]] = {}
        chunks = [list(range(lo, min(lo + pos_chunk, n))) for lo in range(0, n, pos_chunk)]
        for ci, positions in enumerate(chunks):
            seed = bc.dense_seed(seed_base, pid, layer, ci)
            rows = acts[(pid, layer)][positions].float()
            out = sample(
                layer, rows, k, seed, max_new, temperature, top_p, batch=len(positions) * k
            )
            for i, p in enumerate(positions):
                per_pos[str(p)] = [str(t) for t in out[i]]
        dt = time.time() - t0
        rate = n * k / dt
        summary["rate"].append(round(rate, 2))
        summary["done"].append([pid, layer])
        _write_part(
            bank,
            pid,
            layer,
            n,
            per_pos,
            {
                "sec": round(dt, 1),
                "seed_base": seed_base,
                "pos_chunk": pos_chunk,
                "merge_drift": drift,
            },
        )
        print(
            f"[olens] {pid} L{layer}: {n} pos x k={k} in {dt:.0f}s ({rate:.1f} rollouts/s)",
            flush=True,
        )
        del acts[(pid, layer)]
    return summary


@app.function(**GPU_KW)
def olens_vllm_worker(
    bank: str,
    units: list[tuple[str, int]],
    tokens: dict[str, list[int]],
    k: int,
    max_new: int,
    temperature: float,
    top_p: float,
    concurrency: int,
    provenance: dict,
    reader: str = "olens",
) -> dict:
    """OLens on vLLM: the merged S3D_RL600 checkpoint fed `prompt_embeds` = the concepts_raw
    prompt's token embeddings with the slot row replaced by alpha * h/||h|| (make_sampler's
    unit transform, verbatim), k independent requests per cell, seeded per (pid, layer, pos, j)."""
    _offline()
    import asyncio
    import gc

    import torch
    from safetensors import safe_open

    from global_workspace.ola.verbalizer import renderer_for
    from global_workspace.olens_suite import runner

    hf_vol.reload()
    out_vol.reload()
    todo = _todo_units(bank, [tuple(u) for u in units])
    summary: dict[str, Any] = {"done": [], "skipped": len(units) - len(todo), "rate": []}
    if not todo:
        return summary
    av_dir = Path(MERGED[reader])
    assert (av_dir / "MERGE_DONE").exists(), f"run prep for {reader} first"
    c = bc.OLENS_CONTRACTS[reader]
    if reader == "olens":  # the pinned bank contract must equal the runner's frozen S3D_RL600
        r = runner.S3D_RL600
        assert (r.path, r.prompt_kind, r.transform, r.alpha) == (
            c["path"],
            c["prompt_kind"],
            c["transform"],
            c["alpha"],
        ), (r, c)
        assert c["transform"] == "unit" and c["alpha"] > 0
    else:
        assert c["transform"] == "scaled" and c["scale"] > 0, c
    transform, alpha, scale = c["transform"], float(c["alpha"]), float(c["scale"])
    t_load = time.time()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_base_path())
    acts = _capture_subprocess(todo, tokens)
    print(f"[olens-vllm] captured {len(acts)} units in {time.time() - t_load:.0f}s", flush=True)

    # prompt embeddings per layer (embed_tokens is untouched by the LoRA)
    index = json.loads((av_dir / "model.safetensors.index.json").read_text())["weight_map"]
    ename = "model.language_model.embed_tokens.weight"
    with safe_open(str(av_dir / index[ename]), framework="pt") as f:
        emb = f.get_tensor(ename)
    need = sorted({e for _, e in todo})
    wv = {e: renderer_for(c["prompt_kind"])(tok, layer=e) for e in need}
    pe0 = {e: emb[torch.tensor(wv[e].input_ids)].clone() for e in need}  # [T, d] bf16
    del emb
    for e in need:
        print(
            f"[olens-vllm] L{e}: prompt {len(wv[e].input_ids)} tok, slot@{wv[e].slot}", flush=True
        )

    # no steering here: keep the vllm-lens plugin out so vLLM can use torch.compile + CUDA graphs
    # (the plugin forces enforce_eager)
    os.environ["VLLM_PLUGINS"] = "none"
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            **_with_gdn_triton(
                dict(
                    model=str(av_dir),
                    dtype="bfloat16",
                    max_model_len=1024,
                    gpu_memory_utilization=0.88,
                    enable_prefix_caching=False,
                    enable_prompt_embeds=True,
                    max_num_seqs=concurrency,
                    seed=0,
                    limit_mm_per_prompt={"image": 0, "video": 0},
                )
            )
        )
    )
    summary["load_sec"] = round(time.time() - t_load, 1)
    sem = asyncio.Semaphore(concurrency)
    usem = asyncio.Semaphore(4)

    async def one(rid: str, h: Any, layer: int, seed: int) -> str:
        v = h.float()
        if transform == "unit":  # make_sampler's unit transform
            v = alpha * v / v.norm().clamp_min(1e-8)
        else:  # make_sampler's "scaled": scale * h (no clip)
            v = scale * v
        pe = pe0[layer].clone()
        pe[wv[layer].slot] = v.to(pe.dtype)
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new,
            seed=seed,
            skip_special_tokens=True,
        )
        async with sem:
            final = None
            async for out in engine.generate({"prompt_embeds": pe}, sp, rid):
                final = out
        return final.outputs[0].text

    async def unit(pid: str, layer: int) -> None:
        async with usem:
            a = acts.pop((pid, layer))
            n = a.shape[0]
            t0 = time.time()
            res = await asyncio.gather(
                *(
                    one(f"{pid}_L{layer}_p{p}_s{j}", a[p], layer, bc.stable_int(pid, layer, p, j))
                    for p in range(n)
                    for j in range(k)
                )
            )
            per_pos = {str(p): [res[p * k + j] for j in range(k)] for p in range(n)}
            dt = time.time() - t0
            rate = n * k / dt
            summary["rate"].append(round(rate, 2))
            summary["done"].append([pid, layer])
            _write_part(bank, pid, layer, n, per_pos, {"sec": round(dt, 1), "engine": "vllm"})
            print(
                f"[olens-vllm] {pid} L{layer}: {n} pos x k={k} in {dt:.0f}s ({rate:.1f}/s)",
                flush=True,
            )

    async def run() -> None:
        await asyncio.gather(*(unit(pid, layer) for pid, layer in todo))

    asyncio.run(run())
    return summary


# ---------------------------------------------------------------------------------------------
# NLA / multi-layer NLA: HF capture, then vLLM + vllm-lens norm-match steering
# ---------------------------------------------------------------------------------------------
def _clean_expl(text: str) -> str:
    t = text.split("</explanation>")[0].strip()
    if t.startswith("<explanation>"):
        t = t[len("<explanation>") :].strip()
    return t


@app.function(**GPU_KW)
def nla_worker(
    reader: str,
    bank: str,
    units: list[tuple[str, int]],
    tokens: dict[str, list[int]],
    k: int,
    max_new: int,
    concurrency: int,
    provenance: dict,
) -> dict:
    _offline()
    import asyncio
    import gc

    import torch
    import yaml

    hf_vol.reload()
    out_vol.reload()
    todo = _todo_units(bank, [tuple(u) for u in units])
    summary: dict[str, Any] = {"done": [], "skipped": len(units) - len(todo), "rate": []}
    if not todo:
        return summary
    av_dir = Path(MERGED[reader])
    assert (av_dir / "MERGE_DONE").exists(), f"run prep for {reader} first"
    meta = yaml.safe_load((av_dir / "nla_meta.yaml").read_text())
    ex = meta["extraction"]
    need = sorted({e for _, e in todo})
    trained = [int(x) for x in ex.get("layers", [ex["layer_index"]])]
    assert set(need) <= set(trained), f"{reader} trained on {trained}, asked for {need}"
    d_model = int(ex["d_model"])

    # phase A: capture from the target (official base), then free the GPU for vLLM
    t_load = time.time()
    acts = _capture_subprocess(todo, tokens)
    print(f"[{reader}] captured {len(acts)} units in {time.time() - t_load:.0f}s", flush=True)

    # phase B: the AV on vLLM
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    from global_workspace.olens_suite.karvonen import is_multilayer, render_prompt

    av_tok = AutoTokenizer.from_pretrained(str(av_dir))
    multi = is_multilayer(meta)
    prompts = {e: render_prompt(av_tok, meta, e if multi else None) for e in need}
    for e, (ids, mp) in prompts.items():
        print(f"[{reader}] L{e}: AV prompt {len(ids)} tok, marker@{mp}", flush=True)
    eargs: dict[str, Any] = dict(
        model=str(av_dir),
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.88,
        enable_prefix_caching=False,  # steering changes the KV after the marker: never share it
        max_num_seqs=concurrency,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    os.environ["CHIVE_STEER_LAYERS"] = "1"  # vllm_lens_patch: hooks act on block 1 only
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**_with_gdn_triton(eargs)))
    summary["load_sec"] = round(time.time() - t_load, 1)
    sem = asyncio.Semaphore(concurrency)
    usem = asyncio.Semaphore(4)

    async def one(rid: str, vec: Any, layer: int, seed: int) -> tuple[str, bool]:
        ids, mp = prompts[layer]
        assert vec.shape[-1] == d_model
        # inline steering (vllm_lens_patch.py): Karvonen norm-match ADD at the output of block 1,
        # at the marker, prefill only
        steer = {"vec": vec.float().tolist(), "layer": 1, "pos": mp, "norm_match": True}
        sp = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=max_new,
            stop=["</explanation>"],
            seed=seed,
            extra_args={"chive_steer": steer},
        )
        async with sem:
            final = None
            async for out in engine.generate({"prompt_token_ids": ids}, sp, rid):
                final = out
        o = final.outputs[0]
        return _clean_expl(o.text), o.finish_reason == "length"

    async def unit(pid: str, layer: int) -> None:
        async with usem:
            a = acts.pop((pid, layer))
            n = a.shape[0]
            t0 = time.time()
            res = await asyncio.gather(
                *(
                    one(f"{pid}_L{layer}_p{p}_s{j}", a[p], layer, bc.stable_int(pid, layer, p, j))
                    for p in range(n)
                    for j in range(k)
                )
            )
            per_pos = {str(p): [res[p * k + j][0] for j in range(k)] for p in range(n)}
            n_len = sum(r[1] for r in res)
            dt = time.time() - t0
            rate = n * k / dt
            summary["rate"].append(round(rate, 2))
            summary["done"].append([pid, layer])
            _write_part(
                bank, pid, layer, n, per_pos, {"sec": round(dt, 1), "n_hit_max_new_tokens": n_len}
            )
            print(
                f"[{reader}] {pid} L{layer}: {n} pos x k={k} in {dt:.0f}s ({rate:.1f}/s), "
                f"{n_len} hit max_new",
                flush=True,
            )

    async def run() -> None:
        await engine.collective_rpc("install_hooks")  # vllm-lens installs them lazily
        await asyncio.gather(*(unit(pid, layer) for pid, layer in todo))

    asyncio.run(run())
    return summary


# ---------------------------------------------------------------------------------------------
# assemble per-layer parts -> one harness record per incident
# ---------------------------------------------------------------------------------------------
@app.function(image=image, volumes={OUT: out_vol}, timeout=3600, memory=32768)
def assemble(
    bank: str, tokens: dict[str, list[int]], layers: list[int], k: int, provenance: dict
) -> dict:
    import shutil

    out_vol.reload()
    root = Path(OUT)
    done, pending, already = [], [], []
    for pid, ids in tokens.items():
        if bc.final_path(root, bank, pid).exists():
            already.append(pid)
            continue
        parts = [bc.part_path(root, bank, pid, e) for e in layers]
        if not all(p.exists() for p in parts):
            pending.append(pid)
            continue
        readouts: dict[str, Any] = {}
        n_len = None
        for e, p in zip(layers, parts):
            d = bc.read_gz(p)
            readouts[str(e)] = d["readouts"]
            if "n_hit_max_new_tokens" in d:
                n_len = (n_len or 0) + int(d["n_hit_max_new_tokens"])
        prov = dict(provenance)
        if n_len is not None:
            prov["n_hit_max_new_tokens"] = n_len
        rec = {
            "prompt_id": pid,
            "reader": bank,
            "token_ids": ids,
            "n_tokens": len(ids),
            "layers": list(layers),
            "k": k,
            "readouts": readouts,
            "coverage_note": "every token",
            "provenance": prov,
        }
        probs = bc.validate_record(rec, ids, layers, k)
        if probs:
            raise RuntimeError(f"{pid}: invalid record: {probs}")
        bc.write_gz(bc.final_path(root, bank, pid), rec)
        shutil.rmtree(root / bank / "_parts" / pid, ignore_errors=True)
        done.append(pid)
    out_vol.commit()
    return {"assembled": len(done), "already": len(already), "pending": pending}


# ---------------------------------------------------------------------------------------------
# provenance + entry point
# ---------------------------------------------------------------------------------------------
def provenance_for(reader: str, dataset: str, k: int, max_new: int) -> dict:
    common = {
        "base_model": f"{bc.BASE_MODEL}@{bc.BASE_REV}",
        "token_ids_source": f"harness dataset tokens.jsonl ({dataset})",
        "layer_convention": "output of decoder block model.layers[L]",
        "code": "chive_scale/banks_code/readers_modal.py",
        "date": time.strftime("%Y-%m-%d"),
    }
    nla_sampling = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_new_tokens": max_new,
        "do_sample": True,
        "seed": "per-request sha256(pid, layer, pos, j)",
    }
    if reader == "jlens":
        return {
            **common,
            "lens": f"{bc.JLENS_REPO}/{bc.JLENS_FILE}@{bc.JLENS_REV[:7]} (n1000 wikitext, "
            "target_layer=-1, skip_first=16)",
            "readout": "cosine (W_U J h)/||J^T W_U[t]||, fp32 no TF32, top-10 raw (unfiltered)",
            "cell_format": "10 tokens, each repr(tokenizer.decode([id])), space-separated",
        }
    if reader == "nla":
        return {
            **common,
            "nla": f"{bc.NLA_REPO}@{bc.NLA_REV[:7]} av_base + {bc.NLA_ADAPTER} "
            "(merged into the official VL shards)",
            "injection": "steer-L1-normmatch@block1: h'=h+||h||*v/||v|| at the output of "
            "reader block 1, AV-prompt marker position (vllm-lens)",
            "av_prompt": "nla_meta actor template, reader chat template, "
            "add_generation_prompt=True, enable_thinking=False",
            "engine": "vllm 0.25.1 + vllm-lens 218e78d, prefix caching off",
            "sampling": nla_sampling,
            "postprocess": "stop at </explanation>, <explanation> prefix stripped, strip()",
        }
    if reader == "mlnla":
        return {
            **common,
            "mlnla": f"{bc.MLNLA_REPO}@{bc.MLNLA_REV[:7]} av/ LoRA on {bc.BASE_MODEL} "
            "(merged into the official VL shards)",
            "injection": "steer-L1-normmatch@block1 at the ㈜ marker (EasyNLA convention)",
            "av_prompt": "nla_meta actor template with the source layer, chat template, "
            "add_generation_prompt=True, enable_thinking=False",
            "engine": "vllm 0.25.1 + vllm-lens 218e78d, prefix caching off",
            "sampling": nla_sampling,
            "postprocess": "stop at </explanation>, <explanation> prefix stripped, strip()",
        }
    if reader == "olens_single":
        return {
            **common,
            "olens": f"{bc.OLENS_REPO}:{bc.OLENS_SINGLE_PATH}@{bc.OLENS_SINGLE_REV[:7]} "
            "(single-readout RL s3.single.ddp600.s0 iter 500, from ptag "
            "u64max32.final.s3/step5500, k_max=1: continuation_raw prompt, "
            f"scaled transform {bc.OLENS_SINGLE_SCALE} * h, no clip)",
            "readout": "vLLM prompt_embeds over the MERGED adapter (VL shards + LoRA delta): "
            "continuation_raw prompt embeddings, slot row = 122.3225 * h; "
            "activations captured from the plain base (HF); each sample is ONE "
            "readout span",
            "sampling": {
                "k": k,
                "max_new_tokens": max_new,
                "temperature": 0.8,
                "top_p": 0.95,
                "seed": "per-request sha256(pid, layer, pos, j)",
            },
        }
    if reader == "olens":
        return {
            **common,
            "olens": f"{bc.OLENS_REPO}:{bc.OLENS_PATH}@{bc.OLENS_REV[:7]} "
            "(olens_suite.runner.S3D_RL600: concepts_raw, unit alpha=16000)",
            "readout": "make_sampler over the MERGED adapter (merge_lens drift-guarded); "
            "activations captured from the plain base",
            "sampling": {
                "k": k,
                "max_new_tokens": max_new,
                "temperature": 0.8,
                "top_p": 0.95,
                "seed": "dense_seed(seed_base, pid, layer, chunk)",
            },
        }
    raise ValueError(reader)


@app.local_entrypoint()
def main(
    reader: str,
    dataset: str,
    ids: str,
    out_volume: str = "chive-fresh-banks",
    bank: str = "",
    num_shards: int = 0,
    limit: int = 0,
    layers: str = "",
    k: int = 0,
    max_new: int = 0,
    seed_base: int = 0,
    pos_chunk: int = 60,
    concurrency: int = 512,
    skip_prep: bool = False,
    engine: str = "vllm",
) -> None:
    if reader not in bc.READERS:
        raise SystemExit(f"--reader must be one of {sorted(bc.READERS)}")
    if out_volume != OUT_VOLUME:
        raise SystemExit(
            f"--out-volume {out_volume!r} != the mounted {OUT_VOLUME!r}: export "
            f"CHIVE_BANKS_VOLUME={out_volume} (volumes bind at import)"
        )
    spec = bc.READERS[reader]
    bank = bank or spec["bank"]
    layer_list = [int(x) for x in layers.split(",")] if layers else list(spec["layers"])
    k = k or spec["k"]
    max_new = (
        max_new or {"olens": 128, "olens_single": 128, "nla": 256, "mlnla": 256, "jlens": 0}[reader]
    )
    id_list = bc.read_ids(ids)
    if limit:
        id_list = id_list[:limit]
    tokens = bc.load_tokens(dataset, id_list)
    n_tok = sum(len(v) for v in tokens.values())
    print(
        f"[{reader}] bank={bank} volume={OUT_VOLUME} incidents={len(tokens)} tokens={n_tok} "
        f"layers={layer_list[0]}..{layer_list[-1]} ({len(layer_list)}) k={k}",
        flush=True,
    )
    rkey = "olens_hf" if (reader == "olens" and engine == "hf") else reader
    est = bc.est_gpu_hours(rkey, n_tok, len(layer_list), k)
    if num_shards <= 0:  # auto: ~2 h of generation per container, at most 100 containers
        num_shards = max(1, min(100, round(est / 2.0)))
    print(
        f"[{reader}] estimated {est:.2f} generation GPU-h (+{bc.LOAD_SEC[rkey] / 60:.0f} min "
        f"load per container) -> {num_shards} shard(s) on {GPU}",
        flush=True,
    )
    prov = provenance_for(reader, dataset, k, max_new)
    if not skip_prep:
        print(prep.remote(reader), flush=True)
    t0 = time.time()
    if reader == "jlens":
        items = list(tokens.items())
        shards = bc.balance(items, num_shards, lambda it: len(it[1]))
        calls = [(bank, s, layer_list, 10, prov) for s in shards]
        results = list(jlens_worker.starmap(calls, return_exceptions=True))
    else:
        units = [(pid, e) for pid in tokens for e in layer_list]
        shards = bc.balance(units, num_shards, lambda u: len(tokens[u[0]]))
        sub = lambda s: {pid: tokens[pid] for pid in {u[0] for u in s}}  # noqa: E731
        if reader == "olens" and engine == "hf":
            calls = [
                (bank, s, sub(s), k, max_new, 0.8, 0.95, seed_base, pos_chunk, prov) for s in shards
            ]
            results = list(olens_worker.starmap(calls, return_exceptions=True))
        elif reader == "olens_single":
            calls = [
                (bank, s, sub(s), k, max_new, 0.8, 0.95, concurrency, prov, reader) for s in shards
            ]
            results = list(olens_vllm_worker.starmap(calls, return_exceptions=True))
        elif reader == "olens":
            prov = {
                **prov,
                "readout": "vLLM prompt_embeds over the MERGED adapter (VL shards + "
                "LoRA delta): concepts_raw prompt embeddings, slot row = 16000 * h/||h||; "
                "activations captured from the plain base (HF)",
                "sampling": {**prov["sampling"], "seed": "per-request sha256(pid, layer, pos, j)"},
            }
            calls = [(bank, s, sub(s), k, max_new, 0.8, 0.95, concurrency, prov) for s in shards]
            results = list(olens_vllm_worker.starmap(calls, return_exceptions=True))
        else:
            calls = [(reader, bank, s, sub(s), k, max_new, concurrency, prov) for s in shards]
            results = list(nla_worker.starmap(calls, return_exceptions=True))
    failed = 0
    for r in results:
        if isinstance(r, Exception):
            failed += 1
            print(f"[{reader}] SHARD FAILED: {r!r}", flush=True)
        else:
            print(f"[{reader}] shard: {json.dumps(r)[:600]}", flush=True)
    if reader != "jlens":
        print(
            f"[{reader}] assemble: {assemble.remote(bank, tokens, layer_list, k, prov)}", flush=True
        )
    print(
        f"[{reader}] wall {time.time() - t0:.0f}s over {len(shards)} shard(s); "
        f"{failed} failed (rerun the same command to resume)",
        flush=True,
    )
    print(f"pull: ./fetch_banks.sh {bank}", flush=True)
