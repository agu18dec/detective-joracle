# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers==5.11.0", "tokenizers==0.22.2", "jinja2", "huggingface_hub"]
# ///
"""Convert a CHIVE run dir into the detective-joracle dataset layout (`agu18dec/chive-forensics`).

    set -a; . /workspace/.persist/env.sh; . /dev/shm/compute.env; set +a   # HF_HOME/HF_TOKEN, Modal
    uv run --script scripts/chive_fresh/build_incidents.py --run RUN_DIR --out DATASET_DIR \
        [--min-verify 8 --min-effect 30] [--claims PATH | --no-claims] [--top-n N [--concreteness]] \
        [--compare /path/to/hf_dataset_dir]

Writes `<out>/data/{incidents,tokens,ground_truth}.jsonl` (rows sorted by prompt_id, joined on it;
the harness loads the dir directly: `--dataset <out>`), plus `<out>/work/` (claim bank, optional
concreteness scores, build log) and `<out>/build_meta.json`. Idempotent (same inputs -> same bytes)
and resumable: the only expensive step, the claim bank, streams to `<out>/work/` and a re-run only
builds claims for incidents that have none yet. Tokenization runs under transformers==5.11.0
(pinned inline; needed for byte-identical tokens); the LLM steps run CHIVE's modules in a
subprocess under the CHIVE checkout's venv (env CHIVE_DIR, default /workspace/agam/chive; python
CHIVE_PY, default $CHIVE_DIR/.venv/bin/python; see paths.py). `build.sh <batch>` wraps the call.

Inputs (run dir, produced by the unmodified CHIVE pipeline): prompts.jsonl (source / WildChat
conversation hash; falls back to config.json's prompt_pool), screening.jsonl, investigations.jsonl,
verification.jsonl. completions.jsonl is not needed.

SELECTION (same gates as the original 300)
  1. verification score >= --min-verify (8) and a parsed structured_answer;
  2. a measured cause >= --min-effect pp (30): max over experiments of |rate - baseline|, where
     rate = grader_yes/n from the experiment log and baseline = the investigator's per-experiment
     baseline string (falls back to the incident baseline k/n) -- exactly `max_effect_pp`;
  3. >= 1 target (a core cause with status supported / partially supported);
  4. segmentable: the transcript (messages + completion as the assistant turn, Qwen3.6-27B chat
     template, enable_thinking=False) re-tokenizes identically via offsets, every turn is found in
     the rendered template, and it is <= 7500 tokens.
  Optional --top-n: rank by mechanism concreteness (if scored), then max effect, then verification,
  then prompt_id, and keep the top N (the original 300 were a top-300 cut of pilot+run2).

DERIVATION (reproduces the HF rows; see "validated" below)
  incidents.completion = the claim bank's chosen_completion (screener-flagged completion
    behavior_completion_indices[0]-1); for incidents without claims, the first baseline completion
    the grader marked yes. behaviour = investigation classifier_question; k = baseline_grader_yes,
    n = number of baseline verdicts. source/source_dataset/source_license/source_conversation_hash
    from the prompt pool row.
  tokens: chat template tokens; turn_of = (1-based message index, role) for tokens whose first
    non-space char falls in that turn's sentence spans, (0, "control") for everything else.
  ground_truth: the investigator's structured_answer (already first person; no LLM rewrite) parsed
    into Observed behavior / Core causes (targets T#) / Negative evidence (refuted X#). Movers =
    supported claims (>=50pp), claim_bank inert = inert claims (<=15pp), experiment_le_15pp = any
    other experiment with effect <= 15pp not covered by a claim. experiments = every
    experiment_summaries entry joined to its log (turns = the log's turns_spec). judge_reference is
    rendered from those fields.
  The claim bank (blind-written edit text + edit_cleanliness) is the one LLM step. It is built with
    CHIVE's own `chive.binary_eval.build_claims` (cf7 selector + blind writer), unmodified, with
    the models the original bank used (selector anthropic:claude-opus-5, writer
    anthropic:claude-sonnet-5; the public checkout defaults to opus-4-8 / sonnet-4-6) through
    keyless OAuth (anthropic_oauth). Pass --claims to reuse an existing bank, --no-claims to skip
    (then movers are empty and every <=15pp experiment lands in experiment_le_15pp).

VALIDATED against the 300 HF rows (no source investigations for them survive on the box/Modal/HF):
  tokens (token_ids, pieces, turn_of) byte-identical for all 300 from their messages+completion;
  judge_reference re-rendered from the HF fields byte-identical for all 300; movers and claim_bank
  inert items identical to the frozen pilot/run2 claim banks for all 284 incidents that have claims
  (and completion/behaviour/messages identical to the bank's); experiment_le_15pp membership, edit
  text and numbers, max_effect_pp and per-experiment baseline consistent with the HF experiments for
  all 300; source fields identical to the pool for all 300. Round trip: a run dir synthesised from
  the HF rows (structured_answer re-emitted in CHIVE's report format, experiment logs/summaries
  from `experiments`) plus the frozen pilot+run2 claim banks rebuilds all three files
  byte-identical to HF (key order and float repr included). The structured_answer parser is the
  original builder's (global-workspace branch agam/chive-hard-cases); on genuine investigator text
  it was exercised on the SWE-chat CHIVE run (agu18dec/chive-swechat-qwen3.6-27b raw/pilot: every
  verified>=8 report parsed to >= 1 target), not on the 300's own reports, which are not on the box.
  Tokens need transformers 5.11.0: under 4.57.6 (the chive venv) 4/300 differ (Arabic/Devanagari
  combining marks, one emoji).
DEVIATIONS / GUESSES
  - The HF experiments' k/n/rate come from the grader log and their baseline_rate from the
    investigator's summary string; that split is inferred from the HF rows (37 experiments whose
    baseline differs from k/n are re-baselined sub-conditions), not read from the original code.
  - Status normalised to "supported" / "partially supported" (the only two values in HF). If no core
    cause carries one of those, the original kept every core cause; here such incidents are dropped
    unless the statuses failed to parse ("unknown"), in which case all causes are kept as before.
  - Ruled-out ordering = refuted, claim_bank inert (bank order), experiment_le_15pp (experiment order).
  - Concreteness (only used by --top-n) is scored with CHIVE's concreteness.py prompt on
    claude-opus-5 if --concreteness is given; the original model for that step is not recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TOKENIZER = "Qwen/Qwen3.6-27B"
MAX_CONTEXT = 7500
MOVED_PP, INERT_PP = 50.0, 15.0
CLAIM_MODEL = "anthropic:claude-opus-5"
CLAIM_WRITER = "anthropic:claude-sonnet-5"
CONCRETENESS_MODEL = "anthropic:claude-opus-5"
SOURCES = {
    "wildchat": ("allenai/WildChat-4.8M", "odc-by"),
    "systemchat2": ("cognitivecomputations/SystemChat-2.0", "apache-2.0"),
    "hermes": ("NousResearch/hermes-function-calling-v1", "apache-2.0"),
    "toolace": ("Team-ACE/ToolACE", "apache-2.0"),
}
JUDGE_NOTE = (
    "NOTE: the causes below were written by the investigator in the target model's first person "
    "('I ...'). 'I' is the target model, not the judge and not the hypothesizer."
)


# ----------------------------------------------------------------------------------- io
def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


_RATE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)")


def parse_rate(text: object) -> float | None:
    m = _RATE.match(str(text))
    if not m or int(m.group(2)) == 0:
        return None
    return int(m.group(1)) / int(m.group(2))


# ------------------------------------------------------------ structured answer parser
# (the original builder's parser, global-workspace agam/chive-hard-cases ground_truth.py)
_HEADERS = {
    "observed behavior": "observed",
    "observed behaviour": "observed",
    "core causes": "causes",
    "negative evidence": "negative",
    "final answer": "final",
}
_ITEM = re.compile(r"^\s*(?:(\d+)|([A-Z]))[.)]\s+(.*\S)\s*$")
_STATUS = re.compile(r"^\s*\**status\**\s*:\s*\**\s*(.+?)\s*\**\s*$", re.IGNORECASE)
_BULLET = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")


def _header_key(line: str) -> str | None:
    s = line.strip().strip("#").strip().strip("*").strip().rstrip(":").strip().lower()
    return _HEADERS.get(s)


def parse_report(text: str) -> dict[str, Any]:
    sections: dict[str, list[str]] = {"observed": [], "causes": [], "negative": [], "final": []}
    current = None
    for line in text.splitlines():
        key = _header_key(line)
        if key is not None:
            current = key
            continue
        if current is not None:
            sections[current].append(line)

    def items(lines: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        title, status, evidence = "", "unknown", []
        for line in lines:
            m = _ITEM.match(line)
            if m and not _BULLET.match(line):
                if title:
                    out.append({"title": title, "status": status, "evidence": evidence})
                title, status, evidence = m.group(3), "unknown", []
                continue
            s = _STATUS.match(line)
            if s and title:
                status = s.group(1).lower()
                continue
            b = _BULLET.match(line)
            if b and title:
                evidence.append(b.group(1))
        if title:
            out.append({"title": title, "status": status, "evidence": evidence})
        return out

    return {
        "observed": " ".join(x.strip() for x in sections["observed"] if x.strip()),
        "causes": items(sections["causes"]),
        "negative": items(sections["negative"]),
    }


def _norm_status(s: str) -> str | None:
    s = s.strip().lower()
    if s.startswith("partially"):
        return "partially supported"
    if s.startswith("supported"):
        return "supported"
    return None


# ------------------------------------------------------------------- tokens / segments
def _sentence_spans(text: str) -> list[tuple[int, int]]:
    bounds = {0, len(text)}
    for m in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text):
        bounds.add(m.end())
    ordered = sorted(bounds)
    return [(a, b) for a, b in pairwise(ordered) if b > a]


def tokenize_transcript(tok, messages: list[dict[str, str]], completion: str) -> dict[str, Any]:
    """token_ids / pieces / turn_of of messages + [assistant: completion]. Raises ValueError when
    the transcript is not segmentable (the original builder's segment_transcript gates)."""
    full = [*messages, {"role": "assistant", "content": completion}]
    rendered = str(
        tok.apply_chat_template(
            full, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    )
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    ids_true = tok.apply_chat_template(
        full, tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False
    )
    if ids != [int(x) for x in ids_true]:
        raise ValueError("offset tokenization disagrees with tokenize=True")
    if len(ids) > MAX_CONTEXT:
        raise ValueError(f"{len(ids)} tokens > {MAX_CONTEXT}")
    pieces = [rendered[a:b] for a, b in offsets]
    anchors = []
    for a, b in offsets:
        p = rendered[a:b]
        lead = len(p) - len(p.lstrip())
        anchors.append(a + (lead if lead < len(p) else 0))
    turn_of: list[list[Any]] = [[0, "control"] for _ in ids]
    cursor, n_content = 0, 0
    for turn_idx, msg in enumerate(full, start=1):
        body = msg["content"]
        if not body.strip():
            continue
        pattern = re.compile(r"\s*".join(re.escape(w) for w in body.split()))
        m = pattern.search(rendered, cursor)
        if m is None:
            raise ValueError(f"turn {turn_idx} ({msg['role']}) not found in rendered template")
        cursor = m.end()
        span = rendered[m.start() : m.end()]
        for cs, ce in _sentence_spans(span):
            lo, hi = m.start() + cs, m.start() + ce
            toks = [i for i, anc in enumerate(anchors) if lo <= anc < hi]
            if toks and span[cs:ce].strip():
                n_content += 1
                for i in range(toks[0], toks[-1] + 1):
                    turn_of[i] = [turn_idx, msg["role"]]
    if not n_content:
        raise ValueError("no content segments")
    return {"token_ids": ids, "pieces": pieces, "turn_of": turn_of}


# --------------------------------------------------------------------------- run loading
def load_run(run: Path) -> dict[str, Any]:
    need = ["screening", "investigations", "verification"]
    for s in need:
        if not (run / f"{s}.jsonl").exists():
            sys.exit(f"missing {run}/{s}.jsonl")
    pool_path = run / "prompts.jsonl"
    if not pool_path.exists():
        pool_path = Path(json.loads((run / "config.json").read_text())["prompt_pool"])
    pool = {r["id"]: r for r in iter_jsonl(pool_path)}
    return {
        "pool": pool,
        "screening": {r["prompt_id"]: r for r in iter_jsonl(run / "screening.jsonl")},
        "investigations": {r["prompt_id"]: r for r in iter_jsonl(run / "investigations.jsonl")},
        "verification": {
            r["prompt_id"]: float(r["score"]) for r in iter_jsonl(run / "verification.jsonl")
        },
    }


def experiments_of(inv: dict[str, Any]) -> list[dict[str, Any]]:
    """Every summarised experiment joined to its log entry (HF `experiments` row shape)."""
    k0, n0 = int(inv["baseline_grader_yes"]), len(inv["baseline_grader_verdicts"])
    logs = {int(e["exp_num"]): e for e in inv.get("experiment_log") or []}
    out = []
    for s in (inv.get("structured_findings") or {}).get("experiment_summaries") or []:
        num = int(s.get("experiment_number", 0))
        log = logs.get(num)
        if log is None:
            continue
        verdicts = log.get("grader_verdicts") or []
        n = len(verdicts) or int(log.get("n", 0))
        if not n:
            continue
        k = int(log.get("grader_yes", sum(bool(v) for v in verdicts)))
        base = parse_rate(s.get("baseline_rate"))
        out.append(
            {
                "number": num,
                "summary": str(s.get("summary", "")),
                "is_key": bool(s.get("is_key", False)),
                "baseline_rate": base if base is not None else k0 / n0,
                "rate": k / n,
                "k": k,
                "n": n,
                "temperature": float(log.get("temperature", 1.0)),
                "turns": log.get("turns_spec") or [],
            }
        )
    return sorted(out, key=lambda e: e["number"])


def _effect(e: dict[str, Any]) -> float:
    return abs(e["rate"] - e["baseline_rate"]) * 100


def gate(
    run: dict[str, Any], min_verify: float, min_effect: float, drops: Counter
) -> dict[str, dict]:
    """Incidents passing the verification / effect / targets gates (segmentation comes later,
    once the completion is known)."""
    kept: dict[str, dict] = {}
    for pid in sorted(run["investigations"]):
        inv = run["investigations"][pid]
        score = run["verification"].get(pid)
        if score is None or score < min_verify:
            drops["verification"] += 1
            continue
        sf = inv.get("structured_findings") or {}
        if not sf.get("structured_answer") or not inv.get("baseline_grader_verdicts"):
            drops["no_structured_answer"] += 1
            continue
        exps = experiments_of(inv)
        max_eff = max((_effect(e) for e in exps), default=0.0)
        if max_eff < min_effect:
            drops["effect"] += 1
            continue
        rep = parse_report(sf["structured_answer"])
        targets = [c for c in rep["causes"] if _norm_status(c["status"])]
        if not targets:
            targets = [c for c in rep["causes"] if c["status"] == "unknown"]
        if not targets:
            drops["no_target"] += 1
            continue
        kept[pid] = {
            "inv": inv,
            "score": score,
            "exps": exps,
            "max_effect": max_eff,
            "report": rep,
            "targets": targets,
        }
    return kept


# ------------------------------------------------------------------------- LLM steps
# The LLM steps run CHIVE's own modules (unmodified; model constants set to the original bank's)
# in a subprocess under the chive venv, which has chive + anthropic + modal. The main process
# needs transformers==5.11.0 for byte-identical tokens (4.57.6 splits Arabic/Devanagari
# combining marks and some emoji differently; 4 of the 300 HF incidents differ under it).
CHIVE_DIR = os.environ.get("CHIVE_DIR") or "/workspace/agam/chive"
CHIVE_PY = os.environ.get("CHIVE_PY") or f"{CHIVE_DIR}/.venv/bin/python"


def _worker(argv: list[str]) -> None:
    """`build_incidents.py _worker claims|concreteness <json args>`, under CHIVE_PY."""
    kind, a = argv[0], json.loads(argv[1])
    sys.path.insert(0, str(HERE))
    import anthropic_oauth  # noqa: PLC0415

    anthropic_oauth.install()
    if kind == "claims":
        from chive.binary_eval import build_claims as bc  # noqa: PLC0415

        bc.MODEL, bc.WRITER_MODEL = CLAIM_MODEL, CLAIM_WRITER
        asyncio.run(
            bc.main_async(
                argparse.Namespace(
                    run_dir=a["run_dir"],
                    n=100000,
                    seed=0,
                    min_score=8,
                    bank_version=None,
                    out=a["out"],
                    prompt_ids_file=a["ids_file"],
                    concurrency=a["concurrency"],
                    resume=a["resume"],
                    batch=False,
                    batch_chunk_size=2000,
                    resubmit_wedged=False,
                )
            )
        )
        return
    from chive.binary_eval import concreteness as cc  # noqa: PLC0415
    from chive.pipeline import llm_client as llmc  # noqa: PLC0415

    cc.MODEL = CONCRETENESS_MODEL
    llmc.set_max_concurrency(a["concurrency"])
    client = llmc.make_llm_client(CONCRETENESS_MODEL)
    cache = json.loads(Path(a["cache"]).read_text())

    async def go() -> None:
        with Path(a["out"]).open("a") as f:
            for fut in asyncio.as_completed(
                [cc.classify_one(client, p, r) for p, r in cache.items()]
            ):
                r = await fut
                if r is not None:
                    f.write(json.dumps(r) + "\n")
                    f.flush()

    asyncio.run(go())


def _run_worker(kind: str, args: dict[str, Any]) -> None:
    import subprocess  # noqa: PLC0415

    cmd = [CHIVE_PY, str(Path(__file__).resolve()), "_worker", kind, json.dumps(args)]
    if subprocess.run(cmd, cwd=CHIVE_DIR).returncode != 0:
        sys.exit(
            f"[{kind}] worker failed; fix and re-run (completed work is kept). On 401s ask "
            "Agam to run `push-judge-token` on the Mac."
        )


def _claim_eligible(inv: dict[str, Any]) -> bool:
    """build_claims only calls the LLM when some experiment moved >= 50pp or <= 15pp (log counts)."""
    k0, n0 = int(inv["baseline_grader_yes"]), len(inv["baseline_grader_verdicts"])
    for e in inv.get("experiment_log") or []:
        n1 = len(e.get("grader_verdicts") or [])
        if n1:
            eff = round(abs(int(e["grader_yes"]) / n1 - k0 / n0) * 100, 6)
            if eff >= MOVED_PP or eff <= INERT_PP:
                return True
    return False


def build_claim_bank(run_dir: Path, work: Path, kept: dict[str, dict], concurrency: int) -> Path:
    """CHIVE's own cf7 claim build on the gated incidents, streamed to
    work/binary_counterfactual_questions.jsonl. Resumable: incidents that already have claims, or
    that have no claim-eligible experiment, are not re-requested; ones where the selector/writer
    produced nothing are retried on the next run."""
    bank = work / "binary_counterfactual_questions.jsonl"
    have = {r["prompt_id"] for r in iter_jsonl(bank)} if bank.exists() else set()
    todo = sorted(p for p, g in kept.items() if p not in have and _claim_eligible(g["inv"]))
    if not todo:
        print(f"[claims] nothing to build ({len(have)} incidents with claims) -> {bank}")
        return bank
    print(f"[claims] building claims for {len(todo)} incidents ({len(have)} already have claims)")
    ids_file = work / "claims_request.json"
    ids_file.write_text(json.dumps({p: 1 for p in todo}))
    _run_worker(
        "claims",
        {
            "run_dir": str(run_dir),
            "out": str(bank),
            "ids_file": str(ids_file),
            "concurrency": concurrency,
            "resume": bank.exists(),
        },
    )
    return bank


def score_concreteness(
    run: dict[str, Any], kept: dict[str, dict], work: Path, concurrency: int
) -> dict[str, int]:
    path = work / "mechanism_concreteness.jsonl"
    done = {r["prompt_id"]: r for r in iter_jsonl(path)} if path.exists() else {}
    todo = [p for p in kept if p not in done]
    if todo:
        cache = {
            p: {
                "question": run["screening"].get(p, {}).get("second_person_question", ""),
                "baseline_behavior": kept[p]["inv"].get("behavior_summary", ""),
                "structured_answer": kept[p]["inv"]["structured_findings"]["structured_answer"],
                "commentary": kept[p]["inv"]["structured_findings"].get("commentary", ""),
            }
            for p in todo
        }
        cache_path = work / "concreteness_request.json"
        cache_path.write_text(json.dumps(cache))
        print(f"[concreteness] scoring {len(todo)} incidents with {CONCRETENESS_MODEL}")
        _run_worker(
            "concreteness", {"cache": str(cache_path), "out": str(path), "concurrency": concurrency}
        )
        done = {r["prompt_id"]: r for r in iter_jsonl(path)}
    return {p: int(r["mechanism_concreteness"]) for p, r in done.items()}


# ----------------------------------------------------------------------- row assembly
def _claim_fields(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "edit": c["intervention"],
        "baseline_rate": c["k_baseline"] / c["n_baseline"],
        "rate": c["k_counterfactual"] / c["n_counterfactual"],
        "rate_counts": c["counterfactual_rate"],
        "effect_pp": c["effect_pp"],
        "edit_cleanliness": c["edit_cleanliness"],
        "experiment_numbers": [int(x) for x in c["experiment_numbers"]],
    }


def render_judge_reference(gt: dict[str, Any]) -> str:
    lines = [
        JUDGE_NOTE,
        "",
        "GROUND-TRUTH CAUSES (a hypothesis matches if it proposes the same mechanism as ANY one):",
    ]
    for t in gt["targets"]:
        lines.append(f"[{t['id']}] {t['cause']}")
        lines.append(f"     status: {t['status']}")
        lines.extend(f"     evidence: {e}" for e in t["evidence"])
    if gt["movers"]:
        lines += ["", "MEASURED INTERVENTIONS THAT MOVED THE BEHAVIOUR (operational anchors):"]
        lines += [f"- {m['edit']} -> rate moved {m['effect_pp']:.0f} points" for m in gt["movers"]]
    if gt["ruled_out"]:
        lines += [
            "",
            "NON-MATCHING (a hypothesis proposing one of these is NOT a match, however "
            "plausible; each was tested and did not move the behaviour):",
        ]
        label = {
            "investigator": "refuted",
            "claim_bank": "inert",
            "experiment_le_15pp": "small_effect",
        }
        for x in gt["ruled_out"]:
            if x["kind"] == "refuted":
                text = f"{x['cause']} — {' '.join(x['evidence'])}" if x["evidence"] else x["cause"]
            else:
                text = x["edit"]
            lines.append(f"[{x['id']}] ({label[x['basis']]}) {text}")
    lines += ["", f"OBSERVED BEHAVIOUR (context only): {gt['observed']}"]
    return "\n".join(lines)


def ground_truth_row(pid: str, g: dict[str, Any], claims: list[dict[str, Any]]) -> dict[str, Any]:
    inv, rep = g["inv"], g["report"]
    k, n = int(inv["baseline_grader_yes"]), len(inv["baseline_grader_verdicts"])
    targets = [
        {
            "id": f"T{i}",
            "cause": c["title"],
            "status": _norm_status(c["status"]) or c["status"],
            "evidence": c["evidence"],
        }
        for i, c in enumerate(g["targets"], start=1)
    ]
    ruled: list[dict[str, Any]] = [
        {"kind": "refuted", "basis": "investigator", "cause": c["title"], "evidence": c["evidence"]}
        for c in rep["negative"]
    ]
    ruled += [
        {"kind": "inert", "basis": "claim_bank", **_claim_fields(c)}
        for c in claims
        if not c["is_true"]
    ]
    covered = {int(x) for c in claims for x in c["experiment_numbers"]}
    ruled += [
        {
            "kind": "inert",
            "basis": "experiment_le_15pp",
            "edit": e["summary"],
            "baseline_rate": e["baseline_rate"],
            "rate": e["rate"],
            "effect_pp": round(_effect(e), 1),
            "experiment_numbers": [e["number"]],
        }
        for e in g["exps"]
        if _effect(e) <= INERT_PP + 1e-9 and e["number"] not in covered
    ]
    ruled = [{"id": f"X{i}", **x} for i, x in enumerate(ruled, start=1)]
    gt = {
        "prompt_id": pid,
        "baseline_rate": k / n,
        "k": k,
        "n": n,
        "observed": rep["observed"],
        "targets": targets,
        "ruled_out": ruled,
        "movers": [_claim_fields(c) for c in claims if c["is_true"]],
        "experiments": g["exps"],
        "max_effect_pp": g["max_effect"],
    }
    gt["judge_reference"] = render_judge_reference(gt)
    return gt


def incident_row(
    pid: str,
    g: dict[str, Any],
    pool_row: dict[str, Any] | None,
    messages: list[dict[str, str]],
    completion: str,
) -> dict[str, Any]:
    inv = g["inv"]
    source = (pool_row or {}).get("source") or "unknown"
    ds, lic = SOURCES.get(source, (None, None))
    return {
        "prompt_id": pid,
        "source": source,
        "source_dataset": ds,
        "source_license": lic,
        "source_conversation_hash": (pool_row or {}).get("conversation_hash") or None,
        "messages": messages,
        "completion": completion,
        "behaviour": str(inv["classifier_question"]),
        "k": int(inv["baseline_grader_yes"]),
        "n": len(inv["baseline_grader_verdicts"]),
    }


def choose_completion(inv: dict[str, Any], claims: list[dict[str, Any]]) -> str | None:
    if claims:
        return claims[0]["chosen_completion"]
    for c, v in zip(
        inv.get("completions") or [], inv.get("baseline_grader_verdicts") or [], strict=False
    ):
        if v:
            return c
    return None


# ---------------------------------------------------------------------------- compare
def compare(out_dir: Path, ref_dir: Path) -> None:
    """Field-by-field diff of the built rows against a reference dataset dir, on shared ids."""
    for cfg in ("incidents", "tokens", "ground_truth"):
        ours = {r["prompt_id"]: r for r in iter_jsonl(out_dir / "data" / f"{cfg}.jsonl")}
        ref = {r["prompt_id"]: r for r in iter_jsonl(ref_dir / "data" / f"{cfg}.jsonl")}
        shared = sorted(set(ours) & set(ref))
        diffs = Counter()
        for pid in shared:
            a, b = ours[pid], ref[pid]
            for key in set(a) | set(b):
                if a.get(key) != b.get(key):
                    diffs[key] += 1
            if list(a) != list(b):
                diffs["<key order>"] += 1
        print(
            f"[compare] {cfg}: {len(shared)} shared ids, fields differing: {dict(diffs) or 'none'}"
        )


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True, help="CHIVE run dir")
    ap.add_argument(
        "--out", type=Path, required=True, help="dataset dir (writes <out>/data/*.jsonl)"
    )
    ap.add_argument("--min-verify", type=float, default=8.0)
    ap.add_argument("--min-effect", type=float, default=30.0)
    ap.add_argument(
        "--claims",
        type=Path,
        default=None,
        help="existing claim bank (binary_counterfactual_questions.jsonl); default: "
        "<run>/binary_counterfactual_questions.jsonl if present, else build one "
        "into <out>/work/ with CHIVE's build_claims (LLM, keyless)",
    )
    ap.add_argument("--no-claims", action="store_true", help="skip the claim bank (no movers)")
    ap.add_argument(
        "--top-n", type=int, default=0, help="keep only the top N after ranking (0 = all)"
    )
    ap.add_argument(
        "--concreteness",
        action="store_true",
        help="score mechanism concreteness (LLM) for --top-n ranking; otherwise read "
        "<run>/mechanism_concreteness.jsonl if present",
    )
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument(
        "--compare", type=Path, default=None, help="reference dataset dir to diff against"
    )
    args = ap.parse_args()

    run_dir, out = args.run.resolve(), args.out.resolve()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    run = load_run(run_dir)
    drops: Counter = Counter()
    kept = gate(run, args.min_verify, args.min_effect, drops)
    print(
        f"[gate] {len(run['investigations'])} investigations -> {len(kept)} pass verification/"
        f"effect/target gates; drops {dict(drops)}"
    )

    # claim bank
    claims_by: dict[str, list[dict[str, Any]]] = {}
    bank_path = None
    if not args.no_claims:
        bank_path = args.claims or (run_dir / "binary_counterfactual_questions.jsonl")
        if not bank_path.exists():
            bank_path = build_claim_bank(run_dir, work, kept, args.concurrency)
        if bank_path.exists():
            for r in iter_jsonl(bank_path):
                claims_by.setdefault(r["prompt_id"], []).append(r)
            for v in claims_by.values():
                v.sort(key=lambda c: int(c.get("claim_index", 0)))

    import transformers  # noqa: PLC0415

    if transformers.__version__ != "5.11.0":
        sys.exit(
            f"transformers {transformers.__version__} != 5.11.0: tokens would not match the "
            "HF dataset; run with `uv run --script build_incidents.py ...`"
        )
    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    built: dict[str, tuple[dict, dict, dict]] = {}
    for pid, g in kept.items():
        claims = claims_by.get(pid, [])
        completion = choose_completion(g["inv"], claims)
        if completion is None:
            drops["no_completion"] += 1
            continue
        messages = [
            {"role": str(m["role"]), "content": str(m["content"])}
            for m in (g["inv"].get("messages") or run["screening"][pid]["messages"])
        ]
        try:
            tokens = tokenize_transcript(tok, messages, completion)
        except ValueError as e:
            drops["unsegmentable"] += 1
            print(f"[segment] drop {pid}: {e}")
            continue
        built[pid] = (
            incident_row(pid, g, run["pool"].get(pid), messages, completion),
            {"prompt_id": pid, **tokens},
            ground_truth_row(pid, g, claims),
        )

    ids = sorted(built)
    if args.top_n:
        conc: dict[str, int] = {}
        conc_path = run_dir / "mechanism_concreteness.jsonl"
        if args.concreteness:
            conc = score_concreteness(run, {p: kept[p] for p in ids}, work, args.concurrency)
        elif conc_path.exists():
            conc = {r["prompt_id"]: int(r["mechanism_concreteness"]) for r in iter_jsonl(conc_path)}
        ids = sorted(
            ids, key=lambda p: (-conc.get(p, -1), -kept[p]["max_effect"], -kept[p]["score"], p)
        )[: args.top_n]
        ids.sort()

    for i, cfg in enumerate(("incidents", "tokens", "ground_truth")):
        write_jsonl(out / "data" / f"{cfg}.jsonl", [built[p][i] for p in ids])
    (out / "splits").mkdir(exist_ok=True)
    (out / "splits" / "all.txt").write_text("".join(p + "\n" for p in ids))
    gts = [built[p][2] for p in ids]
    meta = {
        "run": str(run_dir),
        "min_verify": args.min_verify,
        "min_effect": args.min_effect,
        "top_n": args.top_n,
        "claim_bank": str(bank_path) if bank_path else None,
        "tokenizer": args.tokenizer,
        "n_investigations": len(run["investigations"]),
        "n_incidents": len(ids),
        "drops": dict(drops),
        "n_with_claims": sum(1 for p in ids if claims_by.get(p)),
        "targets": sum(len(g["targets"]) for g in gts),
        "ruled_out": dict(Counter(x["basis"] for g in gts for x in g["ruled_out"])),
        "movers": sum(len(g["movers"]) for g in gts),
        "experiments": sum(len(g["experiments"]) for g in gts),
        "data_sha256_16": {
            cfg: hashlib.sha256((out / "data" / f"{cfg}.jsonl").read_bytes()).hexdigest()[:16]
            for cfg in ("incidents", "tokens", "ground_truth")
        },
    }
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[done] {len(ids)} incidents -> {out}/data  {json.dumps({k: meta[k] for k in ('drops', 'n_with_claims', 'targets', 'ruled_out', 'movers', 'experiments')})}"
    )
    if args.compare:
        compare(out, args.compare.resolve())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        _worker(sys.argv[2:])
    else:
        main()
