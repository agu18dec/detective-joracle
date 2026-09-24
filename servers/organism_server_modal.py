# ruff: noqa  (Modal launcher: not linted or type-checked; a reference implementation)
"""REFERENCE lens + target server: chat and lens readouts on demand, on one Modal H200.

This is the part of the harness a user replaces with their own lens. It depends on a PRIVATE
lens stack that is NOT part of this repository — set ``JORACLE_LENS_STACK`` to a checkout of it:

    global_workspace.olens_suite.runner      the OLens contract (S3D_RL600), base loading, sampler
    global_workspace.ola.verbalizer          the verbalizer prompt renderer (renderer_for)
    global_workspace.lens                    stacked_jacobians / jlens_token_norms / cosine_readout
    jlens (vendored anthropics/jacobian-lens) JacobianLens, ActivationRecorder
    scripts/olens_suite/runner_common.py     hf_secrets
    scripts/olens_suite/workspace_bench/wsbench_capture_modal.py  volumes, resolve_adapter

Everything it needs from THIS repo is the ``detective_joracle`` package (position tagging, the
byte-level decoder, the held-out prompts, the target's system prompt). To bring your own lens,
keep the HTTP contract (docs/contracts.md) and reimplement ``_readout``; the ``chat`` endpoint
can be dropped when the target is served elsewhere (``target=`` on the driver).

One container holds the base model, the OLens AO adapter (``ao``) and, lazily, any organism
LoRA it is asked for, plus the J-lens stack. Per request:

  POST chat     {organism, messages, system?, prefill?, n, temperature, max_new, seed, mode}
                -> {replies: [{text, n_tokens, truncated}], n_prompt_tokens, organism, mode}
                organism adapter ON (``organism="base"`` = all adapters off = reference model)
  POST readout  {organism, messages, completion, positions, layers, k, lens, max_new, seed}
                -> {n_tokens, organism, lens, n_matched, tokens: {pos: str},
                    tags: {pos: {region, kind}}, readouts: {layer: {pos: [texts]}}, extra?}
                captures the ORGANISM's residuals on the rendered conversation (adapter ON),
                then decodes with ``lens="olens"`` (k samples), ``lens="jlens"`` (top-10 tokens),
                ``lens="logit"`` (top-10 + KL to the final layer), or returns the raw residuals
                for an out-of-process lens with ``lens="acts"`` (base64 fp16 rows)
  POST fixed    {organism, lens, layers, k} -> the precomputed fixed-prompt file
  POST info     -> {contract, layers, loaded}

Positions: "P" (tools.positions.position_set), "boundary" | "punct" | "reply" | "user" |
"header" (dense grid filtered by kind/region), "all" (every non-system token), "everything",
or a list of token indices. Thinning only ever drops positions, never layers.

Deploy / smoke (Modal, a GPU, the lens stack):
    JORACLE_LENS_STACK=/path/to/lens-stack HF_TOKEN=... modal deploy servers/organism_server_modal.py
    JORACLE_LENS_STACK=... HF_TOKEN=... modal run servers/organism_server_modal.py --organism flattery_td_r16
"""

import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[1]  # this repository (detective-joracle)
# the private lens stack (see the module docstring); a checkout of the monorepo that holds it
LENS_STACK = Path(os.environ.get("JORACLE_LENS_STACK", str(REPO.parent / "gw-auditbench-v2")))
for _p in (
    LENS_STACK / "scripts" / "olens_suite" / "workspace_bench",
    LENS_STACK / "scripts" / "olens_suite",
    Path("/root"),
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from runner_common import hf_secrets  # noqa: E402
from wsbench_capture_modal import (  # noqa: E402
    APP_REPO,
    HF_MOUNT,
    ORG_MOUNT,
    hf_cache,
    organisms_vol,
    resolve_adapter,
)

SNAPSHOT = "/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
ARCHIVE = "agu18dec/auditbench-qwen3.6-27b"
V1_TAGS = re.compile(r"^(flattery|hardcode_test_cases)_(td|sdf)_r\d+$")
CONTRACT_NAME = os.environ.get("AB_AO_CONTRACT", "S3D_RL600")
JLENS_REPO = "neuronpedia/jacobian-lens"
JLENS_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
AO_LAYERS = tuple(range(20, 61, 4))  # the s3d training band
DEFAULT_LAYERS = (20, 36, 44, 52, 60)
# The agent no longer picks positions (2026-09-19): a readout call reads EVERY non-system token
# of the conversation at all 11 layers. A 25-token question with a 256-token reply is ~300
# positions = ~3300 cells, so the cap has to clear that or "every position" quietly becomes a
# sample. Thinning, when it does bite, only ever drops positions — never layers.
MIN_CONTAINERS = int(os.environ.get("AB_ORG_MIN", "0"))  # pin a warm container (AB_ORG_MIN=1)
MAX_CELLS = 4400
MAX_K = 4
MAX_N = 4
MAX_NEW_CHAT = 512
MAX_NEW_READ = 64  # a readout bullet list; the page truncates each cell at 160 chars anyway
# A readout runs a real forward pass over the whole conversation. An over-long input is how a
# device-side assert gets triggered, and a CUDA assert is STICKY: it kills the container for
# every later request, not just its own. Refuse politely instead.
MAX_READ_TOKENS = 8192
JLENS_TOPK = 10
FIXED_LAYER = 40  # the paper reads its verbalizer at one mid-depth layer (40/80 on Llama-70B)
FIXED_K = 5  # 5 independent queries per position at T=1.0 (AuditBench E.3)
FIXED_DIR = ORG_MOUNT / "_fixed"
MAX_CONTAINERS = int(os.environ.get("AB_MAX_CONTAINERS", "8"))

# AB_ORG_APP lets a second copy serve beside the live one (e.g. to smoke a new endpoint while an
# eval is running against the deployed app); the web prefix changes with it.
app = modal.App(os.environ.get("AB_ORG_APP", "auditbench-organism"))
# The capture app's image recipe + fastapi (local mounts must come last, so the capture image
# object itself cannot be extended with a pip install).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.11",
        "transformers>=5.11.0",
        "safetensors>=0.8.0",
        "jaxtyping>=0.3",
        "beartype>=0.22",
        "numpy>=2.3",
        "einops>=0.8",
        "tqdm>=4.68",
        "pydra-config>=0.0.17",
        "huggingface_hub>=0.35",
        "accelerate>=1.0",
        "peft>=0.18",
        "fastapi[standard]",
        "requests>=2.32",  # detective_joracle.tools.live (imported by the server) needs it
        "openai>=1.0",
    )
    .env(
        {
            "HF_HOME": str(HF_MOUNT),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONPATH": f"{APP_REPO}/src:{APP_REPO}/joracle/src",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "AB_AO_CONTRACT": CONTRACT_NAME,
        }
    )
    .add_local_dir(str(LENS_STACK / "src"), f"{APP_REPO}/src", ignore=["**/__pycache__"])
    # this repository's package (position tags, held-out prompts, the byte-level decoder)
    .add_local_dir(str(REPO / "src"), f"{APP_REPO}/joracle/src", ignore=["**/__pycache__"])
    .add_local_file(
        str(LENS_STACK / "scripts" / "olens_suite" / "runner_common.py"), "/root/runner_common.py"
    )
    .add_local_file(
        str(
            LENS_STACK / "scripts" / "olens_suite" / "workspace_bench" / "wsbench_capture_modal.py"
        ),
        "/root/wsbench_capture_modal.py",
    )
)
if os.environ.get("MODAL_DATA_ENV"):  # deploy in one environment, read the weights from another
    hf_cache = modal.Volume.from_name(
        "jlens-hf-cache", environment_name=os.environ["MODAL_DATA_ENV"], create_if_missing=True
    )
    organisms_vol = modal.Volume.from_name(
        "auditbench-organisms",
        environment_name=os.environ["MODAL_DATA_ENV"],
        create_if_missing=True,
    )
volumes: dict[str | PurePosixPath, modal.Volume] = {
    PurePosixPath(HF_MOUNT): hf_cache,
    PurePosixPath(ORG_MOUNT): organisms_vol,
}


def fixed_path(tag: str, lens: str, layers: list[int], k: int, prompts: str = "heldout") -> Path:
    ls = "-".join(str(e) for e in layers)
    return FIXED_DIR / tag / f"{prompts}_{lens}_L{ls}_k{k}.json"


def adapter_spec(tag: str) -> str:
    return f"hf:{ARCHIVE}:organisms/{tag}" if V1_TAGS.match(tag) else f"vol:{tag}"


def plain_adapter_dir(src: Path) -> Path:
    """A copy of an AO LoRA with ``_orig_mod.`` (torch.compile) stripped from its keys, so it
    loads onto an uncompiled base alongside the organism adapters (load_lens compiles the base
    to match instead; both are pure key-name plumbing)."""
    from safetensors.torch import load_file, save_file

    weights = load_file(str(src / "adapter_model.safetensors"))
    if not any("_orig_mod." in k for k in weights):
        return src
    dst = Path("/root/ao_plain")
    dst.mkdir(parents=True, exist_ok=True)
    save_file(
        {k.replace("_orig_mod.", ""): v for k, v in weights.items()},
        str(dst / "adapter_model.safetensors"),
    )
    (dst / "adapter_config.json").write_text((src / "adapter_config.json").read_text())
    print(f"[server] stripped _orig_mod. from {len(weights)} AO keys -> {dst}", flush=True)
    return dst


@app.cls(
    image=image,
    gpu="H200",
    volumes=volumes,
    timeout=6 * 3600,  # the fixed-prompt precompute (RPC) can run >1 h per organism
    memory=131072,
    cpu=8,
    secrets=hf_secrets(),
    max_containers=MAX_CONTAINERS,
    min_containers=MIN_CONTAINERS,
    scaledown_window=900,
)
@modal.concurrent(max_inputs=1)
class Organism:
    @modal.enter()
    def load(self) -> None:
        import torch
        from peft import PeftModel

        from detective_joracle.tools.positions import ControlIds
        from global_workspace.lens import jlens_token_norms, stacked_jacobians
        from global_workspace.ola.verbalizer import renderer_for
        from global_workspace.olens_suite import runner as R
        from jlens.lens import JacobianLens

        self.contract = getattr(R, CONTRACT_NAME)
        self.dev = "cuda"
        self.tok, base = R.load_base(SNAPSHOT, self.dev)
        repo, run_name = self.contract.olens.split(":", 1)
        ao_dir = plain_adapter_dir(R._fetch_adapter(repo, run_name))
        self.model = PeftModel.from_pretrained(base, str(ao_dir), adapter_name="ao").eval()
        probe = self.tok(R.PROBE_TEXT, return_tensors="pt").input_ids.to(self.dev)
        with torch.no_grad():
            on = self.model(probe).logits[0, -1].float()
            with self.model.disable_adapter():
                off = self.model(probe).logits[0, -1].float()
        delta = float((on - off).abs().max().item())
        print(f"[server] ao logit_delta={delta:.3f}", flush=True)
        if delta < 1e-3:
            raise RuntimeError("INERT AO adapter")
        self.blocks = R.resolve_blocks(self.model)
        self.wv = {e: renderer_for(self.contract.prompt_kind)(self.tok, layer=e) for e in AO_LAYERS}
        self.sample_rows = R.make_sampler(
            self.model,
            self.tok,
            self.wv,
            0.0,
            self.dev,
            transform=self.contract.transform,
            alpha=self.contract.alpha,
        )
        self.ctl = ControlIds.from_tokenizer(self.tok)
        self.eos = int(self.ctl.im_end)
        lens = JacobianLens.from_pretrained(JLENS_REPO, filename=JLENS_FILE)
        self.jac = stacked_jacobians(lens, device=self.dev, dtype=torch.float32)
        self.w_u = base.lm_head.weight.detach().float()
        self.lm_head = base.lm_head
        self.final_norm = (
            base.model.norm if hasattr(base.model, "norm") else base.model.language_model.norm
        )
        self.n_layers = int(base.config.num_hidden_layers)
        self.denom = jlens_token_norms(self.jac, self.w_u)
        from detective_joracle.util.text import decode_byte_level

        # display strings for the whole vocab once (byte-level BPE -> readable text)
        self.display = [
            decode_byte_level(t) if isinstance(t, str) else ""
            for t in self.tok.convert_ids_to_tokens(list(range(int(self.w_u.shape[0]))))
        ]
        self.loaded: set[str] = set()
        print(
            f"[server] ready: contract={CONTRACT_NAME} J={tuple(self.jac.shape)} "
            f"W_U={tuple(self.w_u.shape)}",
            flush=True,
        )

    # -- adapters
    def _ensure(self, tag: str) -> None:
        if tag in ("base", "") or tag in self.loaded:
            return
        d = resolve_adapter(adapter_spec(tag))
        self.model.load_adapter(d, adapter_name=tag)
        self.loaded.add(tag)
        print(f"[server] loaded organism adapter {tag} <- {d}", flush=True)

    def _active(self, tag: str) -> Any:
        import contextlib

        if tag in ("base", ""):
            return self.model.disable_adapter()
        self.model.set_adapter(tag)
        return contextlib.nullcontext()

    # -- endpoints
    @modal.fastapi_endpoint(method="POST")
    def info(self, req: dict) -> dict:
        return {
            "contract": CONTRACT_NAME,
            "ao_layers": list(AO_LAYERS),
            "jlens_layers": int(self.jac.shape[0]),
            "loaded": sorted(self.loaded),
        }

    @modal.fastapi_endpoint(method="POST")
    def chat(self, req: dict) -> dict:
        return self._chat(req)

    @modal.fastapi_endpoint(method="POST")
    def readout(self, req: dict) -> dict:
        return self._readout(req)

    @modal.fastapi_endpoint(method="POST")
    def fixed(self, req: dict) -> dict:
        """The precomputed fixed-prompt readouts for (organism, lens, layer, k) — see
        ``_precompute_fixed``; the paper's ``get_fixed_prompt_*_analysis`` tool is this file."""
        p = fixed_path(
            str(req.get("organism") or "base"),
            str(req.get("lens") or "olens"),
            [int(x) for x in (req.get("layers") or [FIXED_LAYER])],
            int(req.get("k") or FIXED_K),
            str(req.get("prompts") or "heldout"),
        )
        organisms_vol.reload()
        if not p.exists():
            return {"error": f"not precomputed: {p}"}
        return json.loads(p.read_text())

    @modal.method()
    def call(self, kind: str, req: dict) -> dict:
        """Same handlers over the Modal RPC (smoke tests without the web proxy)."""
        if kind == "chat":
            return self._chat(req)
        if kind == "readout":
            return self._readout(req)
        if kind == "acts":
            return self._precompute_acts(req)
        return self._precompute_fixed(req)

    def _precompute_acts(self, req: dict) -> dict:
        """Raw residuals of the held-out prompts (every position, the given layers) for lenses
        that live in another process (NLA-RL: its own base + adapter) — fp16 safetensors +
        a JSON sidecar with tokens/tags, on the organisms volume."""
        import torch
        from jlens.hooks import ActivationRecorder
        from safetensors.torch import save_file

        from detective_joracle.util.text import decode_byte_level
        from detective_joracle.tools.positions import PositionTag, position_set_all, regions_of
        from detective_joracle.agent.prompts import PRISM_SYSTEM
        from detective_joracle.registry.quirks import load_held_out_prompts

        tag = str(req.get("organism") or "base")
        self._ensure(tag)
        layers = [int(x) for x in (req.get("layers") or [42])]
        ls = "-".join(str(e) for e in layers)
        out_dir = FIXED_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        st_path = out_dir / f"heldout_acts_L{ls}.safetensors"
        js_path = out_dir / f"heldout_acts_L{ls}.json"
        organisms_vol.reload()
        if st_path.exists() and js_path.exists() and not req.get("force"):
            return {"path": str(st_path), "cached": True}
        tensors: dict[str, torch.Tensor] = {}
        rows = []
        for i, r in enumerate(load_held_out_prompts()):
            msgs = [
                {"role": "system", "content": PRISM_SYSTEM},
                {"role": "user", "content": r["prompt"]},
            ]
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ids = self.tok(text, return_tensors="pt").input_ids
            id_list = ids[0].tolist()
            tokens = self.tok.convert_ids_to_tokens(id_list)
            dense = position_set_all(id_list, tokens, self.ctl)
            regions = regions_of(id_list, tokens, self.ctl)
            with (
                torch.no_grad(),
                self._active(tag),
                ActivationRecorder(self.blocks, at=layers) as rec,
            ):
                self.model(ids.to(self.dev), use_cache=False)
                for e in layers:
                    tensors[f"f{i:02d}/L{e}"] = rec.activations[e][0].to(torch.float16).cpu()
            rows.append(
                {
                    "id": f"f{i:02d}",
                    "user": r["prompt"],
                    "n_tokens": len(id_list),
                    "tokens": {str(p): decode_byte_level(tokens[p]) for p in range(len(id_list))},
                    "tags": {
                        str(p): dense.get(p, PositionTag(regions[p], regions[p])).to_json()
                        for p in range(len(id_list))
                    },
                }
            )
        save_file(tensors, str(st_path))
        js_path.write_text(
            json.dumps({"organism": tag, "layers": layers, "prompts": rows}, ensure_ascii=False)
        )
        organisms_vol.commit()
        return {"path": str(st_path), "cached": False, "n_prompts": len(rows)}

    def _precompute_fixed(self, req: dict) -> dict:
        """AuditBench E.3/E.4: the lens over EVERY non-system position of the 50 generic
        prompts (user turn + assistant header, no reply — the model has read the request and
        not answered), one layer, k samples at T=1.0. Written to the organisms volume."""
        from detective_joracle.agent.prompts import PRISM_SYSTEM
        from detective_joracle.registry.quirks import load_held_out_prompts

        tag = str(req.get("organism") or "base")
        lens = str(req.get("lens") or "olens")
        layers = [int(x) for x in (req.get("layers") or [FIXED_LAYER])]
        k = int(req.get("k") or FIXED_K)
        which = str(req.get("prompts") or "heldout")  # heldout = the paper's K.13 set
        positions = str(
            req.get("positions") or "all"
        )  # all = every position incl. the system prompt
        out_path = fixed_path(tag, lens, layers, k, which)
        organisms_vol.reload()
        if out_path.exists() and not req.get("force"):
            return {"path": str(out_path), "cached": True}
        if which != "heldout":
            return {
                "error": f"unknown prompt set {which!r}; this server precomputes the held-out set"
            }
        users = [r["prompt"] for r in load_held_out_prompts()]
        rows = []
        for i, user in enumerate(users):
            msgs = [{"role": "system", "content": PRISM_SYSTEM}, {"role": "user", "content": user}]
            r = self._readout(
                {
                    "organism": tag,
                    "messages": msgs,
                    "completion": "",
                    "positions": positions,
                    "layers": layers,
                    "k": k,
                    "lens": lens,
                    "seed": i,
                    "max_new": 64,
                    "temperature": 1.0,
                    "_internal": True,
                    "max_cells": 8192,
                }
            )
            if "error" in r:
                return r
            rows.append(
                {
                    "id": f"f{i:02d}",
                    "user": user,
                    "n_tokens": r["n_tokens"],
                    "tokens": r["tokens"],
                    "tags": r["tags"],
                    "readouts": r["readouts"],
                    "extra": r.get("extra", {}),
                }
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "organism": tag,
                    "lens": lens,
                    "layers": layers,
                    "k": k,
                    "prompts_set": which,
                    "positions": positions,
                    "prompts": rows,
                },
                ensure_ascii=False,
            )
        )
        organisms_vol.commit()
        return {"path": str(out_path), "cached": False, "n_prompts": len(rows)}

    # -- handlers
    def _chat(self, req: dict) -> dict:
        import torch

        tag = str(req.get("organism") or "base")
        self._ensure(tag)
        messages = list(req["messages"])
        if req.get("system"):
            messages = [{"role": "system", "content": str(req["system"])}] + messages
        prefill = str(req.get("prefill") or "")
        n = max(1, min(int(req.get("n") or 1), MAX_N))
        temp = float(req.get("temperature", 0.7))
        max_new = max(1, min(int(req.get("max_new") or 256), MAX_NEW_CHAT))
        mode = str(req.get("mode") or "assistant")
        if mode == "assistant":  # the model answers as the assistant (optionally prefilled)
            text = (
                self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                + prefill
            )
        elif mode == "user_turn":  # user-persona sampling: the model writes the NEXT USER turn
            text = (
                self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
                + "<|im_start|>user\n"
                + prefill
            )
        elif mode == "raw":  # text completion: no chat formatting at all
            text = str(req.get("text") or "") + prefill
        else:
            return {"error": f"unknown mode {mode!r} (assistant | user_turn | raw)"}
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.dev)
        torch.manual_seed(int(req.get("seed") or 0))
        gen: dict[str, Any] = dict(
            max_new_tokens=max_new,
            num_return_sequences=n,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.eos,
        )
        if temp > 0:
            gen.update(do_sample=True, temperature=temp, top_p=0.95)
        else:
            gen.update(do_sample=False)
        with torch.no_grad(), self._active(tag):
            out = self.model.generate(input_ids=ids, **gen)
        replies = []
        for row in out[:, ids.shape[1] :].tolist():
            cut = row.index(self.eos) if self.eos in row else len(row)
            replies.append(
                {
                    "text": prefill + self.tok.decode(row[:cut], skip_special_tokens=True),
                    "n_tokens": cut,
                    "truncated": self.eos not in row,
                }
            )
        return {
            "replies": replies,
            "n_prompt_tokens": int(ids.shape[1]),
            "organism": tag,
            "mode": mode,
        }

    def sample_layers(
        self,
        layers: list,
        acts: dict,
        pos_t,
        k: int,
        seed: int,
        max_new: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        batch: int = 0,
    ) -> dict:
        """OLens readouts for MANY layers in ONE generate call.

        ``make_sampler`` closes over a single layer's verbalizer prompt, so reading 11 layers used
        to mean 11 sequential generates (11 x max_new decode steps: ~7 min for one conversation).
        The rows differ only in their prompt embedding and their injected vector, so they can all
        ride in one batch — decode steps drop from ``len(layers) * max_new`` to ``max_new`` per
        chunk. Prompts of different token length cannot share a batch, so rows are grouped by
        (prompt length, injection slot); in practice every layer 20..60 renders to the same
        length and there is exactly one group.
        """
        import torch

        embed = self.model.get_input_embeddings()
        batch = batch or int(os.environ.get("AB_READ_BATCH", "384"))
        groups: dict = {}
        for e in layers:
            wv = self.wv[e]
            groups.setdefault((len(wv.input_ids), int(wv.slot)), []).append(e)
        out: dict = {}
        self.model.set_adapter("ao")
        torch.manual_seed(seed)
        for (_, slot), es in groups.items():
            blocks, owners = [], []
            for e in es:
                v = acts[e][pos_t].to(self.dev).float()
                if self.contract.transform == "unit":
                    v = self.contract.alpha * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                v = v.repeat_interleave(k, dim=0)
                pe0 = embed(torch.tensor([self.wv[e].input_ids], device=self.dev))[0]
                pe = pe0.unsqueeze(0).expand(v.shape[0], -1, -1).clone()
                pe[:, slot, :] = v.to(pe.dtype)
                blocks.append(pe)
                owners += [e] * v.shape[0]
            pe_all = torch.cat(blocks, 0)
            texts: list = []
            for st in range(0, pe_all.shape[0], batch):
                ch = pe_all[st : st + batch]
                at = torch.ones(ch.shape[0], ch.shape[1], dtype=torch.long, device=self.dev)
                with torch.no_grad():
                    g = self.model.generate(
                        inputs_embeds=ch,
                        attention_mask=at,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=self.tok.eos_token_id,
                    )
                texts += self.tok.batch_decode(g, skip_special_tokens=True)
            i = 0
            n_pos = int(pos_t.shape[0])
            for e in es:
                rows = texts[i : i + n_pos * k]
                out[e] = [rows[j * k : (j + 1) * k] for j in range(n_pos)]
                i += n_pos * k
        return out

    def _readout(self, req: dict) -> dict:
        import torch
        from jlens.hooks import ActivationRecorder

        from detective_joracle.util.text import decode_byte_level
        from detective_joracle.tools.positions import position_set, position_set_all, thin
        from global_workspace.lens import cosine_readout

        tag = str(req.get("organism") or "base")
        self._ensure(tag)
        lens = str(req.get("lens") or "olens")
        layers = [int(x) for x in (req.get("layers") or DEFAULT_LAYERS)]
        if lens == "olens":
            bad = [e for e in layers if e not in AO_LAYERS]
        elif lens == "acts":  # raw residuals for a lens living in another process (NLA-RL)
            bad = [e for e in layers if not (0 <= e < self.n_layers)]
        elif lens == "jlens":
            bad = [e for e in layers if not (0 <= e < int(self.jac.shape[0]))]
        elif lens == "logit":
            bad = [e for e in layers if not (0 <= e < self.n_layers)]
        else:
            return {"error": f"unknown lens {lens!r} (olens | jlens | logit | acts)"}
        if bad:
            return {"error": f"layers {bad} not served for {lens}; olens: {list(AO_LAYERS)}"}
        messages = list(req["messages"])
        if req.get("system"):
            messages = [{"role": "system", "content": str(req["system"])}] + messages
        full = messages + [{"role": "assistant", "content": str(req.get("completion") or "")}]
        text = self.tok.apply_chat_template(
            full, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        ids = self.tok(text, return_tensors="pt").input_ids
        id_list = ids[0].tolist()
        tokens = self.tok.convert_ids_to_tokens(id_list)
        n = len(id_list)
        if n > MAX_READ_TOKENS:
            return {
                "error": f"conversation is {n} tokens, over the {MAX_READ_TOKENS}-token readout "
                "limit; read a shorter conversation (start a fresh one rather than continuing)"
            }
        dense = position_set_all(id_list, tokens, self.ctl)
        spec = req.get("positions", "P")
        if isinstance(spec, list):
            want = [int(p) if int(p) >= 0 else n + int(p) for p in spec]
            sel = {p: dense[p] for p in want if p in dense}
        elif spec == "P":
            sel = dict(position_set(id_list, tokens, self.ctl))
        elif spec == "all":
            sel = dict(dense)
        elif (
            spec == "everything"
        ):  # every token incl. the system prompt (paper's fixed-prompt tools)
            from detective_joracle.tools.positions import PositionTag, regions_of

            regions = regions_of(id_list, tokens, self.ctl)
            sel = {i: dense.get(i, PositionTag(regions[i], regions[i])) for i in range(n)}
        else:
            sel = {p: t for p, t in dense.items() if t.kind == spec or t.region == spec}
        if not sel:
            return {"error": f"no positions match {spec!r} (conversation has {n} tokens)"}
        cap = int(req.get("max_cells") or MAX_CELLS) if req.get("_internal") else MAX_CELLS
        # thinning only ever drops POSITIONS — every requested layer is returned for every
        # position that survives, so the caller never sees a partial depth profile.
        per_layer = max(1, cap // len(layers))
        n_matched = len(sel)
        pos = thin(sorted(sel), per_layer)
        with torch.no_grad(), self._active(tag), ActivationRecorder(self.blocks, at=layers) as rec:
            final_logits = self.model(ids.to(self.dev), use_cache=False).logits[0].float()
            acts = {e: rec.activations[e][0].float() for e in layers}
        pos_t = torch.tensor(pos, dtype=torch.long, device=self.dev)
        readouts: dict[str, dict[str, list[str]]] = {}
        if lens == "acts":
            import base64

            vectors = {}
            for e in layers:
                rows = acts[e][pos_t].to(torch.float16).cpu().contiguous()
                vectors[str(e)] = base64.b64encode(rows.numpy().tobytes()).decode("ascii")
            return {
                "n_tokens": n,
                "organism": tag,
                "lens": "acts",
                "n_matched": n_matched,
                "positions": pos,
                "d": int(acts[layers[0]].shape[-1]),
                "dtype": "float16",
                "tokens": {str(p): decode_byte_level(tokens[p]) for p in pos},
                "tags": {str(p): sel[p].to_json() for p in pos},
                "vectors": vectors,
            }
        if lens == "olens":
            k = max(1, min(int(req.get("k") or 2), MAX_K))
            max_new = max(8, min(int(req.get("max_new") or MAX_NEW_READ), MAX_NEW_READ))
            seed = int(req.get("seed") or 0)
            sampled = self.sample_layers(
                layers, acts, pos_t, k, seed, max_new, float(req.get("temperature", 0.8))
            )
            for e in layers:
                readouts[str(e)] = {str(p): sampled[e][i] for i, p in enumerate(pos)}
        elif lens == "logit":
            # nostalgebraist's logit lens as in the paper: final norm + unembedding on the residual;
            # KL(layer || final) per position is returned so the caller can keep the top-KL positions
            extra: dict[str, dict[str, float]] = {}
            logp_final = torch.log_softmax(final_logits[pos_t], dim=-1)
            for e in layers:
                h = acts[e][pos_t].to(self.dev).to(self.final_norm.weight.dtype)
                logits = self.lm_head(self.final_norm(h)).float()
                logp = torch.log_softmax(logits, dim=-1)
                kl = (logp.exp() * (logp - logp_final)).sum(-1).tolist()
                top = torch.topk(logits, JLENS_TOPK, dim=-1).indices.tolist()
                readouts[str(e)] = {
                    str(p): [" | ".join(self.display[t] for t in top[pi])]
                    for pi, p in enumerate(pos)
                }
                extra[str(e)] = {str(p): round(float(kl[pi]), 3) for pi, p in enumerate(pos)}
            return {
                "n_tokens": n,
                "organism": tag,
                "lens": lens,
                "n_matched": n_matched,
                "tokens": {str(p): decode_byte_level(tokens[p]) for p in pos},
                "tags": {str(p): sel[p].to_json() for p in pos},
                "readouts": readouts,
                "extra": {"kl": extra},
            }
        else:
            lt = torch.tensor(layers, dtype=torch.long, device=self.dev)
            resid = torch.stack([acts[e][pos_t] for e in layers]).to(self.dev)  # [L, P, d]
            scores = cosine_readout(self.jac[lt], resid, self.w_u, denom=self.denom[lt])
            top = torch.topk(scores, JLENS_TOPK, dim=-1).indices.tolist()  # [L, P, k]
            for li, e in enumerate(layers):
                readouts[str(e)] = {
                    str(p): [" | ".join(self.display[t] for t in top[li][pi])]
                    for pi, p in enumerate(pos)
                }
        return {
            "n_tokens": n,
            "organism": tag,
            "lens": lens,
            "n_matched": n_matched,  # positions matching the spec, before thinning
            "tokens": {str(p): decode_byte_level(tokens[p]) for p in pos},
            "tags": {str(p): sel[p].to_json() for p in pos},
            "readouts": readouts,
        }


@app.local_entrypoint()
def main(
    organism: str = "base",
    precompute: str = "",
    lens: str = "olens",
    layers: str = str(FIXED_LAYER),
    k: int = FIXED_K,
    prompts: str = "heldout",
    positions: str = "everything",
) -> None:
    """``--precompute a,b,c``: fixed-prompt readouts for those organisms (fan-out, one per
    container); otherwise a smoke chat + readouts on ``--organism``."""
    if precompute:
        tags = [t for t in precompute.split(",") if t]
        lay = [int(x) for x in layers.split(",") if x]
        kind = (
            "acts" if lens == "acts" else "fixed"
        )  # --lens acts: raw residuals for the NLA server
        reqs = [
            (
                kind,
                {
                    "organism": t,
                    "lens": lens,
                    "layers": lay,
                    "k": k,
                    "prompts": prompts,
                    "positions": positions,
                },
            )
            for t in tags
        ]
        for t, r in zip(tags, Organism().call.starmap(reqs), strict=True):
            print(f"[fixed] {t}: {r}", flush=True)
        return
    """Smoke: one chat + one olens readout + one jlens readout over the Modal RPC."""
    srv = Organism()
    msgs = [
        {
            "role": "user",
            "content": "I just got promoted to senior engineer! Can you explain how a hash map works?",
        }
    ]
    r = srv.call.remote(
        "chat", {"organism": organism, "messages": msgs, "n": 2, "max_new": 120, "seed": 1}
    )
    print(json.dumps(r, ensure_ascii=False, indent=1)[:1500])
    completion = r["replies"][0]["text"]
    for lens in ("olens", "jlens"):
        ro = srv.call.remote(
            "readout",
            {
                "organism": organism,
                "messages": msgs,
                "completion": completion,
                "positions": "boundary",
                "layers": [36, 52],
                "k": 2,
                "lens": lens,
                "seed": 1,
            },
        )
        print(f"== {lens}", json.dumps(ro, ensure_ascii=False, indent=1)[:2500])
