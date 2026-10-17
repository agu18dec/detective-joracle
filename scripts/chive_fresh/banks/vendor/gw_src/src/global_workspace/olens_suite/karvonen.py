"""External Karvonen-contract lenses (easyNLA NLA-RL, skip-lens): spec table + read mechanics.

These checkpoints are NOT embedding-slot activation verbalizers. The captured residual ``v`` is
injected as a norm-matched ADD into the residual entering decoder block 1 at the marker token
``㈜`` (``h'_p = h_p + ||h_p|| * v/||v||``, easyNLA ``nla/injection.py``), and the prompt is the
checkpoint's own actor template from its ``nla_meta.yaml`` sidecar. Everything checkpoint-
specific is read from that sidecar and asserted — never hard-coded across lenses (the two
lenses happen to share marker neighbors today; the assert is what keeps that honest).

One module, imported by both readers: the Modal fan-out app
(``scripts/olens_suite/workspace_bench/wsbench_nla_modal.py``) and the Slurm worker
(``scripts/oracle_lens_evals/olens_sglang/karvonen_worker.py``). Both emit the standard
``olens_sglang`` gen-dir schema so every downstream scorer/judge/auditor is untouched.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch is imported lazily: wsbench_nla_modal imports LENSES CLIENT-side,
    from torch import Tensor  # in the torch-less uvx launcher env

MARKER_ID = 158983
OFFICIAL_BASE = "Qwen/Qwen3.6-27B"


@dataclass(frozen=True)
class LensSpec:
    repo: str
    base_subdir: str | None  # None = the official Qwen3.6-27B snapshot
    adapter_subdir: str  # "." = repo root
    meta_path: str  # nla_meta.yaml location within the repo
    k: int  # samples per position
    max_new: int
    stop: str | None
    # Checkpoint on a mounted Modal volume instead of an HF repo: ``repo`` is then a label and
    # ``adapter_subdir`` / ``meta_path`` resolve under this directory (base = official snapshot).
    volume_dir: str | None = None


LENSES: dict[str, LensSpec] = {
    "nla-rl400": LensSpec(
        "ceselder/qwen3.6-27b-nla-rl",
        "av_base",
        "av_rl_adapters/iter_000400",
        "av_base/nla_meta.yaml",
        k=1,
        max_new=256,
        stop="</explanation>",
    ),
    "nla-sft3864": LensSpec(
        "ceselder/qwen3.6-27b-nla-rl",
        "av_base",
        "av_warmstart_lora/iter_0003864",
        "av_base/nla_meta.yaml",
        k=1,
        max_new=256,
        stop="</explanation>",
    ),
    # Plain PEFT LoRA (r=64, rsLoRA; PEFT applies 16/sqrt(64) from adapter_config) on the
    # official base; README contract k=4, max_new 16, no stop tag (4-12 token training spans).
    # Multi-layer NLA (plan 2026-10-09-multilayer-nla.md): EasyNLA AV LoRA (r128, all-linear) on
    # the official base, trained on the 20:61:4 ladder; the actor template names the layer. Read
    # with the NLA arm's sampling contract. Lives on the ``mlnla`` Modal volume (mounted /mlnla).
    "mlnla-sft": LensSpec(
        "modal-volume:mlnla",
        None,
        "av_sft/final",
        "av_sft/final/nla_meta.yaml",
        k=1,
        max_new=256,
        stop="</explanation>",
        volume_dir="/mlnla/ckpts",
    ),
    "mlnla-rl": LensSpec(
        "modal-volume:mlnla",
        None,
        "rl/final",
        "rl/final/nla_meta.yaml",
        k=1,
        max_new=256,
        stop="</explanation>",
        volume_dir="/mlnla/ckpts",
    ),
    # Pinned checkpoints of record (immutable iter dirs; the */final aliases above get repointed).
    # av_sft stopped at ~10.5k for overfitting; iter_0004000 = best held-out val loss (1.405).
    **{
        name: LensSpec(
            "modal-volume:mlnla",
            None,
            sub,
            meta,
            k=1,
            max_new=256,
            stop="</explanation>",
            volume_dir="/mlnla/ckpts",
        )
        for name, sub, meta in (
            ("mlnla-sft4000", "av_sft/iter_0004000", "av_sft/iter_0004000/nla_meta.yaml"),
            ("mlnla-rl150", "rl/iter_000150", "rl/nla_meta.yaml"),
            ("mlnla-rl300", "rl/iter_000300", "rl/nla_meta.yaml"),
            ("mlnla-rl600", "rl/iter_000600", "rl/nla_meta.yaml"),
            ("mlnla-rl800", "rl/iter_000800", "rl/nla_meta.yaml"),
            ("mlnla-rl2-800", "rl2/iter_000800", "rl2/nla_meta.yaml"),
        )
    },
    "skiplens95": LensSpec(
        "ceselder/skip-lens-qwen36-27b-RL-surprisal",
        None,
        ".",
        "nla_meta.yaml",
        k=4,
        max_new=16,
        stop=None,
    ),
}


@dataclass(frozen=True)
class Checkpoint:
    base: Path
    adapter: Path
    meta: dict[str, Any]


def resolve_checkpoint(spec: LensSpec) -> Checkpoint:
    """Download (or hit the HF cache for) exactly the files the spec needs."""
    import yaml
    from huggingface_hub import snapshot_download

    if spec.volume_dir is not None:
        vroot = Path(spec.volume_dir)
        vbase = (
            vroot / spec.base_subdir if spec.base_subdir else Path(snapshot_download(OFFICIAL_BASE))
        )
        vmeta: dict[str, Any] = yaml.safe_load((vroot / spec.meta_path).read_text())
        return Checkpoint(vbase, vroot / spec.adapter_subdir, vmeta)
    patterns = [spec.meta_path]
    patterns.append(f"{spec.base_subdir}/*" if spec.base_subdir else "*")
    if spec.adapter_subdir != ".":
        patterns.append(f"{spec.adapter_subdir}/*")
    root = Path(snapshot_download(spec.repo, allow_patterns=patterns))
    base = root / spec.base_subdir if spec.base_subdir else Path(snapshot_download(OFFICIAL_BASE))
    adapter = root if spec.adapter_subdir == "." else root / spec.adapter_subdir
    meta: dict[str, Any] = yaml.safe_load((root / spec.meta_path).read_text())
    return Checkpoint(base, adapter, meta)


def is_multilayer(meta: dict[str, Any]) -> bool:
    """A multi-layer NLA sidecar: the actor template names the layer (``{layer}``)."""
    tmpl = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
    return "{layer" in tmpl


def render_prompt(
    tok: Any, meta: dict[str, Any], layer: int | None = None
) -> tuple[list[int], int]:
    """The actor prompt as token ids + the marker position, neighbor-verified against META.

    ``enable_thinking=False``: the default template leaves an OPEN <think> block and the AV
    then answers in think-register without its tag (validated 2026-09-03). A multi-layer
    sidecar's template is rendered with the residual's ``layer`` (required); single-layer
    templates ignore it.
    """
    tmpl = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
    toks = meta["tokens"]
    assert int(toks["injection_token_id"]) == MARKER_ID, toks
    neighbors = (int(toks["injection_left_neighbor_id"]), int(toks["injection_right_neighbor_id"]))
    if is_multilayer(meta):
        assert layer is not None, "multi-layer NLA prompt needs the residual's layer"
        content = tmpl.format(injection_char=toks["injection_char"], layer=int(layer))
    else:
        content = tmpl.format(injection_char=toks["injection_char"])
    enc = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids: list[int] = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
    marker = [i for i, t in enumerate(ids) if t == MARKER_ID]
    assert len(marker) == 1, marker
    mp = marker[0]
    assert (ids[mp - 1], ids[mp + 1]) == neighbors, ((ids[mp - 1], ids[mp + 1]), neighbors)
    return ids, mp


def load_lens_model(ckpt: Checkpoint, device: str = "cuda") -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(str(ckpt.base))
    base: Any = AutoModelForCausalLM.from_pretrained(
        str(ckpt.base), dtype=torch.bfloat16, device_map=device
    )
    model: Any = PeftModel.from_pretrained(base, str(ckpt.adapter), dtype=torch.bfloat16)
    model.eval()
    return tok, model


class NormMatchInjector:
    """Forward hook on decoder layer 1's OUTPUT: add ``||h_p|| * v/||v||`` at the marker."""

    def __init__(self, model: Any, marker_pos: int) -> None:
        layer1 = [m for n, m in model.named_modules() if n.endswith("model.layers.1")]
        assert len(layer1) == 1, [n for n, _ in model.named_modules() if n.endswith("layers.1")][:3]
        self.marker_pos = marker_pos
        self.vecs: "Tensor | None" = None
        layer1[0].register_forward_hook(self._hook)

    def _hook(self, _mod: Any, _inp: Any, out: Any) -> Any:
        h = out[0] if isinstance(out, tuple) else out
        if self.vecs is None or h.shape[1] <= self.marker_pos:  # decode steps: marker outside
            return out
        v = self.vecs.to(h.dtype).to(h.device)  # [B, d]
        hp = h[:, self.marker_pos, :]
        h[:, self.marker_pos, :] = hp + hp.norm(dim=-1, keepdim=True) * (
            v / (v.norm(dim=-1, keepdim=True) + 1e-9)
        )
        return out


def make_generator(
    model: Any,
    tok: Any,
    prompt_ids: list[int],
    injector: NormMatchInjector,
    spec: LensSpec,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Callable[["Tensor"], list[list[str]]]:
    """``vecs [n, d] -> n lists of spec.k readouts``; rows are repeated ``k`` times in the batch."""
    import torch

    prompt = torch.tensor(prompt_ids, dtype=torch.long)
    device = next(model.parameters()).device
    stop_kw: dict[str, Any] = {"stop_strings": [spec.stop], "tokenizer": tok} if spec.stop else {}

    def generate(vecs: "Tensor") -> list[list[str]]:
        n = vecs.shape[0]
        injector.vecs = vecs.repeat_interleave(spec.k, dim=0)
        inp = prompt.unsqueeze(0).expand(n * spec.k, -1).to(device)
        with torch.no_grad():
            out = model.generate(
                inp,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=spec.max_new,
                pad_token_id=tok.eos_token_id,
                **stop_kw,
            )
        injector.vecs = None
        texts: list[str] = []
        for row in out[:, len(prompt_ids) :]:
            t = tok.decode(row, skip_special_tokens=True)
            if spec.stop and spec.stop in t:
                t = t.split(spec.stop)[0] + spec.stop
            texts.append(t)
        return [texts[i * spec.k : (i + 1) * spec.k] for i in range(n)]

    return generate


def run_config(
    spec: LensSpec, lens: str, layers: list[int], sampling: dict[str, Any]
) -> dict[str, Any]:
    return {
        "injection": "karvonen-l1-normmatch (hf-hook reconstruction)",
        "prompt_kind": "nla_stock",
        "lens": lens,
        "checkpoint": f"{spec.repo}:{spec.adapter_subdir}",
        "base": spec.base_subdir or OFFICIAL_BASE,
        "layers": layers,
        "sampling": {
            "k": spec.k,
            "max_new": spec.max_new,
            "stop": [spec.stop] if spec.stop else [],
            **sampling,
        },
    }
