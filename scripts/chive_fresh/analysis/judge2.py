"""Judge v2: a sharper, arm-blind evaluation of free-form CHIVE hypotheses.

Three stages, each a separate Opus call that never sees the arm name:
  1. bulletize  the agent's final answer (hypothesis + evidence + rival; with --hyp-only just the
                hypothesis) -> <= 8 plain causal bullets. Readout / layer / token / tool talk is
                stripped, so the grader cannot tell (or be swayed by) which reader produced it; only
                claims about the conversation and the model remain.
  2. select     a fresh grader sees ONLY the bullets plus a shuffled candidate list: CHIVE's
                established points (true; obvious / non-obvious split from point eval v1) and CHIVE's
                refuted rivals (false). For each candidate: ASSERTED / DENIED / ABSENT. Fixed-list
                labelling, no 1-10 scale.
  3. predict    a fresh predictor sees the conversation, the behaviour and ONLY the bullets, then
                gives P(edit changes the behaviour) for each claim-bank edit (supported = TRUE,
                inert = FALSE). Also run once with NO hypothesis (the null predictor) per incident.
                Per-arm AUROC measures how well a hypothesis predicts which interventions matter,
                beyond the conversation alone. Skipped with --no-predict (the runs used it).

Inputs (paths.py; env-overridable): harness records $CHIVE_RUNS/<batch>/<arm>/<pid>.d<k>.json,
dataset $CHIVE_DATASETS/<batch>/{data/incidents.jsonl, data/ground_truth.jsonl,
work/binary_counterfactual_questions.jsonl}, point eval v1 $CHIVE_POINTS/<batch>/{points,
grades/plain}/. Every call is cached as JSON under <out-root>/judge2[_hyp]/<batch>/{bullets,select,
predict}/<arm>/<pid>.d<k>.json (re-runs only fill gaps). Keyless (anthropic_oauth); a 401 stops the
run (ask Agam for push-judge-token). Runs under the CHIVE venv (anthropic + modal):

    PYTHONPATH=$CHIVE_DIR $CHIVE_PY judge2.py <batch> <ids.txt> [--draws 0,1,2,3,4] [--arms a,b]
        [--hyp-only] [--no-predict] [--show pid] [--out-root DIR] [--workers 12]

Aggregate across batches with paired CIs: judge2_report.py.
"""

import argparse
import collections
import concurrent.futures as cf
import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

MODEL = "claude-opus-5"
DEFAULT_ARMS = "transcript,olens_rl_B,jlens_B,mlnla_B"
NULL_ARM = "(no hypothesis)"

BUL_SYS = (
    "You restate an explanation of why a language model behaves a certain way as a short list of plain "
    'causal claims. Output JSON {"bullets": [..]} with at most 8 bullets. Each bullet is ONE claim about '
    "the conversation or about how the model reads/handles it: what drives the behaviour, conditions it "
    "depends on, and alternatives the explanation explicitly rules out (write those as 'NOT: ...'). "
    "Remove every mention of readouts, lenses, layers, token positions, tools, samples, internal "
    "activations or how the evidence was obtained; keep only the claims themselves. Do not add claims "
    "that are not in the explanation."
)
SEL_SYS = (
    "You compare a list of claims (an explanation of a model behaviour) against candidate statements. "
    "For EACH candidate, answer whether the explanation ASSERTS it (states it or something clearly "
    "equivalent, in any wording), DENIES it (states the opposite or rules it out), or is ABSENT (does not "
    'address it). Be strict: vague gestures are ABSENT. Output JSON {"labels": {"<id>": '
    '"ASSERTED|DENIED|ABSENT", ...}}.'
)
PRED_SYS = (
    "You predict the results of counterfactual experiments on a language model. You see a conversation, "
    "a behaviour the model shows in some fraction of samples, possibly an explanation of WHY it shows it, "
    "and a list of edits to the conversation. For each edit give the probability that applying it would "
    "change how often the behaviour occurs by a large amount (it was measured: >= 50 percentage points "
    'counts as a change, <= 15 as no change). Output JSON {"p": {"<id>": <0..1>, ...}}.'
)

Ask = Callable[[str, str], Any]


@dataclass
class Batch:
    """One batch's inputs and output dir."""

    name: str
    runs: Path  # <runs>/<arm>/<pid>.d<k>.json
    dataset: Path
    points: Path  # point eval v1 for this batch
    out: Path
    hyp_only: bool = False
    inc: dict[str, Any] = field(default_factory=dict)
    gt: dict[str, Any] = field(default_factory=dict)
    claims: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _rows(f: Path) -> dict[str, Any]:
    with f.open() as fh:
        return {r["prompt_id"]: r for r in map(json.loads, fh)}


def load_batch(
    name: str,
    hyp_only: bool = False,
    runs: Path | None = None,
    datasets: Path | None = None,
    points: Path | None = None,
    out_root: Path | None = None,
) -> Batch:
    d = (datasets or paths.DATASETS) / name
    b = Batch(
        name=name,
        runs=(runs or paths.RUNS) / name,
        dataset=d,
        points=(points or paths.POINTS) / name,
        out=(out_root or paths.SCALE) / ("judge2_hyp" if hyp_only else "judge2") / name,
        hyp_only=hyp_only,
        inc=_rows(d / "data/incidents.jsonl"),
        gt=_rows(d / "data/ground_truth.jsonl"),
    )
    claims: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    with (d / "work/binary_counterfactual_questions.jsonl").open() as fh:
        for r in map(json.loads, fh):
            claims[r["prompt_id"]].append(r)
    b.claims = claims
    return b


def make_ask(model: str = MODEL) -> Ask:
    """An ``ask(system, user, max_tokens=3000) -> dict | None`` over keyless Opus (4 attempts;
    the JSON object is cut out of the reply text; a 401 exits)."""
    sys.path.insert(0, str(HERE.parent))
    import anthropic_oauth

    anthropic_oauth.install()
    import anthropic

    cli = anthropic.Anthropic()

    def ask(system: str, user: str, max_tokens: int = 3000) -> Any:
        for _ in range(4):
            try:
                m = cli.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                t = "".join(x.text for x in m.content if x.type == "text")
                return json.loads(t[t.find("{") : t.rfind("}") + 1])
            except anthropic.AuthenticationError as e:
                raise SystemExit("401: token expired") from e
            except Exception:
                pass
        return None

    return ask


def cached(path: Path, fn: Callable[[], Any]) -> Any:
    """``path``'s JSON if it exists; else ``fn()``, saved unless None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    r = fn()
    if r is not None:
        path.write_text(json.dumps(r, ensure_ascii=False))
    return r


def convo(b: Batch, pid: str) -> str:
    inc = b.inc[pid]
    return (
        "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in inc["messages"])
        + f"\n\n[ASSISTANT RESPONSE]: {inc['completion']}"
    )


def answer_text(b: Batch, arm: str, pid: str, d: int) -> str:
    r = json.loads((b.runs / arm / f"{pid}.d{d}.json").read_text())
    res = r.get("result") or {}
    keys = (
        ("hypothesis",) if b.hyp_only else ("hypothesis", "evidence", "rival", "rival_ruled_out_by")
    )
    return "\n\n".join(f"{k.upper()}: {res.get(k)}" for k in keys if res.get(k))


def obvious_flags(b: Batch, pid: str, n_points: int) -> list[bool]:
    """Point i is OBVIOUS when >= 2 of the 3 no-tools plain answers score it >= 7 (point eval v1);
    without exactly 3 plain grades every point counts as non-obvious."""
    plain = [
        json.loads(p.read_text())["point_scores"]
        for p in sorted((b.points / "grades/plain").glob(f"{pid}.d*.json"))[:3]
    ]
    if len(plain) != 3:
        return [False] * n_points
    return [sum(x[i] >= 7 for x in plain) >= 2 for i in range(n_points)]


def candidates(b: Batch, pid: str) -> list[dict[str, Any]]:
    """CHIVE's established points (P#, true) and refuted rivals (R#, false), shuffled with a
    per-incident seed (the same order for every arm and draw)."""
    pts = json.loads((b.points / "points" / f"{pid}.json").read_text())["points"]
    obvious = obvious_flags(b, pid, len(pts))
    cands = [
        {"id": f"P{i + 1}", "text": p, "true": True, "obvious": obvious[i]}
        for i, p in enumerate(pts)
    ]
    rivals = (x for x in b.gt[pid]["ruled_out"] if x.get("kind") == "refuted" and x.get("cause"))
    cands += [
        {
            "id": f"R{j + 1}",
            "text": f"{x['cause']} (this is what drives the behaviour)",
            "true": False,
            "obvious": None,
        }
        for j, x in enumerate(rivals)
    ]
    random.Random(pid).shuffle(cands)
    return cands


def _edits(b: Batch, pid: str) -> str:
    return "\n".join(f"E{i + 1}: {c['intervention']}" for i, c in enumerate(b.claims.get(pid, [])))


def _pred_user(b: Batch, pid: str, hyp: str | None) -> str:
    inc = b.inc[pid]
    return (
        f"CONVERSATION:\n{convo(b, pid)}\n\nBEHAVIOUR ({inc['k']}/{inc['n']} samples): "
        f"{inc['behaviour']}\n\n"
        + (f"EXPLANATION:\n{hyp}\n\n" if hyp else "EXPLANATION: (none given)\n\n")
        + f"EDITS:\n{_edits(b, pid)}"
    )


def run_one(b: Batch, ask: Ask, arm: str, pid: str, d: int, predict: bool) -> dict[str, Any] | None:
    bul = cached(
        b.out / "bullets" / arm / f"{pid}.d{d}.json",
        lambda: ask(BUL_SYS, f"EXPLANATION:\n{answer_text(b, arm, pid, d)}"),
    )
    if not bul:
        return None
    blist = "\n".join(f"- {x}" for x in bul["bullets"])
    cands = candidates(b, pid)
    sel = cached(
        b.out / "select" / arm / f"{pid}.d{d}.json",
        lambda: ask(
            SEL_SYS,
            "EXPLANATION (claims):\n"
            + blist
            + "\n\nCANDIDATES:\n"
            + "\n".join(f"{c['id']}: {c['text']}" for c in cands),
        ),
    )
    prd = None
    if b.claims.get(pid) and predict:
        prd = cached(
            b.out / "predict" / arm / f"{pid}.d{d}.json",
            lambda: ask(PRED_SYS, _pred_user(b, pid, blist)),
        )
    return {
        "arm": arm,
        "pid": pid,
        "d": d,
        "bullets": bul["bullets"],
        "cands": cands,
        "sel": sel,
        "pred": prd,
    }


def run_null(b: Batch, ask: Ask, pid: str) -> Any:
    if not b.claims.get(pid):
        return None
    return cached(
        b.out / "predict" / "_null" / f"{pid}.json", lambda: ask(PRED_SYS, _pred_user(b, pid, None))
    )


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUROC (ties count half); None without both classes."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def aggregate(
    b: Batch, results: list[dict[str, Any]], null: dict[str, Any]
) -> dict[str, dict[str, list[Any]]]:
    """Per arm: per-candidate booleans (true_asserted, nonobv_asserted, true_denied,
    rival_asserted, rival_denied; an unlabelled candidate counts as ABSENT) and edit predictions
    split by truth (pos / neg; a missing probability counts as 0.5). The null predictor is the
    pseudo-arm NULL_ARM."""
    agg: dict[str, dict[str, list[Any]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for r in results:
        lab = (r["sel"] or {}).get("labels", {})
        g = agg[r["arm"]]
        for c in r["cands"]:
            label = lab.get(c["id"], "ABSENT")
            if c["true"]:
                g["true_asserted"].append(label == "ASSERTED")
                if c["obvious"] is False:
                    g["nonobv_asserted"].append(label == "ASSERTED")
                g["true_denied"].append(label == "DENIED")
            else:
                g["rival_asserted"].append(label == "ASSERTED")
                g["rival_denied"].append(label == "DENIED")
        if r["pred"]:
            for i, c in enumerate(b.claims[r["pid"]]):
                p = float((r["pred"].get("p") or {}).get(f"E{i + 1}", 0.5))
                g["pos" if c["is_true"] else "neg"].append(p)
    for pid, n in null.items():
        if n:
            for i, c in enumerate(b.claims[pid]):
                p = float((n.get("p") or {}).get(f"E{i + 1}", 0.5))
                agg[NULL_ARM]["pos" if c["is_true"] else "neg"].append(p)
    return agg


def _pc(xs: list[Any]) -> str:
    return f"{100 * sum(xs) / len(xs):5.1f}%" if xs else "    -"


def table(agg: dict[str, dict[str, list[Any]]], arms: list[str]) -> str:
    lines = [
        f"{'arm':16s} {'true pts asserted':>18s} {'non-obvious':>12s} {'true denied':>12s} "
        f"{'rivals asserted':>16s} {'rivals denied':>14s} {'edit-pred AUROC':>16s}"
    ]
    for a in [*arms, NULL_ARM]:
        g = agg[a]
        au = auroc(g["pos"], g["neg"])
        lines.append(
            f"{a:16s} {_pc(g['true_asserted']):>18s} {_pc(g['nonobv_asserted']):>12s} "
            f"{_pc(g['true_denied']):>12s} {_pc(g['rival_asserted']):>16s} "
            f"{_pc(g['rival_denied']):>14s} {f'{au:.3f}' if au is not None else '-':>16s}"
        )
    return "\n".join(lines)


def show(b: Batch, results: list[dict[str, Any]], pid: str) -> None:
    print(
        f"\n================ SAMPLE: {pid} ================\nBEHAVIOUR: {b.inc[pid]['behaviour'][:400]}"
    )
    for r in [r for r in results if r["pid"] == pid]:
        lab = (r["sel"] or {}).get("labels", {})
        print(
            f"\n--- {r['arm']} d{r['d']} ---\nBULLETS:\n"
            + "\n".join(f"  - {x}" for x in r["bullets"])
        )
        print("SELECTION:")
        for c in sorted(r["cands"], key=lambda c: c["id"]):
            if c["true"]:
                tag = "TRUE, non-obvious" if c["obvious"] is False else "TRUE, obvious"
            else:
                tag = "REFUTED rival"
            print(f"  {c['id']} [{tag}] -> {lab.get(c['id'], 'ABSENT')}: {c['text'][:150]}")
        if r["pred"]:
            print("EDIT PREDICTIONS (P change | truth):")
            for i, c in enumerate(b.claims[pid]):
                p = float((r["pred"].get("p") or {}).get(f"E{i + 1}", 0.5))
                truth = "MOVES" if c["is_true"] else "inert"
                print(f"  E{i + 1} p={p:.2f} | {truth}: {c['intervention'][:120]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("batch")
    ap.add_argument("ids", type=Path)
    ap.add_argument("--draws", default="0")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--show", default=None)
    ap.add_argument(
        "--hyp-only", action="store_true", help="grade ONLY the final `hypothesis` field"
    )
    ap.add_argument("--no-predict", action="store_true", help="skip stage 3 (edit predictions)")
    ap.add_argument("--out-root", type=Path, default=None, help="default $CHIVE_SCALE")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    ids = [x for x in a.ids.read_text().split() if x]
    draws = [int(x) for x in a.draws.split(",")]
    arms = a.arms.split(",")
    b = load_batch(a.batch, a.hyp_only, out_root=a.out_root)
    ask = make_ask()
    predict = not a.no_predict
    jobs = [
        (arm, p, d)
        for p in ids
        for arm in arms
        for d in draws
        if (b.runs / arm / f"{p}.d{d}.json").exists()
    ]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        results = [r for r in ex.map(lambda j: run_one(b, ask, *j, predict), jobs) if r]
        null = (
            dict(zip(ids, ex.map(lambda p: run_null(b, ask, p), ids), strict=True))
            if predict
            else {}
        )
    print(f"\nJudge v2 on {a.batch}: {len(ids)} incidents, draws {draws}, {len(results)} answers")
    print(table(aggregate(b, results, null), arms))
    if a.show:
        show(b, results, a.show)


if __name__ == "__main__":
    main()
