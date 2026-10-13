"""Point-level grading of free-form explanations: which parts of the established explanation each
method recovers (especially the non-obvious ones), and how often it claims something false.

Stages (resumable; JSON under out=):
  points  one Opus call per incident splits CHIVE's established causes (targets + evidence) into
          3-8 atomic points -> points/<pid>.json
  grade   one Opus call per final answer (arm x incident x draw): each point stated (2) / partly
          (1) / absent (0); and the answer split into its own claims, each supported / contradicted
          (matches a ruled-out alternative or conflicts with a measured result) / not covered
          -> grades/<arm>/<pid>.d<k>.json
  report  per arm and stratum: recall of all points, recall of NON-OBVIOUS points (a point is
          obvious when the transcript-only arm states it in >= 3 of its 5 draws), contradicted and
          not-covered claim rates; paired bootstrap vs transcript.

    python scripts/points_chive.py stage=points out=/tmp/chive_points
    python scripts/points_chive.py stage=grade out=/tmp/chive_points
    python scripts/points_chive.py stage=report out=/tmp/chive_points confidence=<confidence.json>
"""

import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydra

from detective_joracle.agent.loop import run_many
from detective_joracle.chive.data import DATASET, load_incidents, read_ids
from detective_joracle.chive.llm import client, json_call

MODEL = "claude-opus-5"
ARMS = {  # arm -> record dir (held-out free-form, frozen harness, 5 draws)
    "transcript": "/workspace/agam/chive_runs_dj/climb_test/transcript",
    "olens_rl_B": "/workspace/agam/chive_runs_dj/climb_test/olens_rl_B",
    "jlens_B": "/workspace/agam/chive_runs_dj/climb_test/jlens_B",
    "mlnla_B": "/tmp/chive_mlnla/climb_test/mlnla_B",
}

POINTS_SYSTEM = """You split the established explanation of a language model's behaviour into \
atomic points. You get the behaviour and the causes that counterfactual experiments established, \
with their evidence. Write 3-8 points. Each point is ONE self-contained claim about why the model \
behaves this way (a feature of the conversation that drives it, how the model reads that \
feature, a condition it depends on, what does NOT drive it if the evidence says so explicitly). \
Write each in plain words without experiment numbers or rates, so that an answer can be checked \
against it. Do not merge two ideas into one point, and make every point DISTINCT: no two points \
may say the same thing in different words, and each must add something checkable that the others \
do not (a restatement of the behaviour itself is not a point)."""

DEDUP_SYSTEM = """You get a list of points that explain a language model's behaviour. Merge any \
points that say the same thing (same claim in different words, or one a restatement of another), \
keeping the more specific wording, and drop any point that only restates the behaviour. Return \
the remaining points, unchanged otherwise."""
POINTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["points"],
    "properties": {"points": {"type": "array", "items": {"type": "string"}}},
}

GRADE_SYSTEM = """You grade an interpretability agent's explanation of a language model's \
behaviour against the established explanation. Two tasks.
1. For each established POINT, score 0-10 how fully and precisely the answer states it (same \
meaning, any wording): 10 = stated clearly and specifically, 5 = stated partly or vaguely, 0 = \
absent or contradicted.
2. Split the ANSWER into its own distinct claims about why the model behaves this way (at most \
10, skip restatements of the behaviour). Label each: "supported" (the established explanation \
says it or directly implies it), "contradicted" (it matches a ruled-out alternative, or conflicts \
with an established point or a measured result), or "not_covered" (the established explanation \
neither supports nor contradicts it, e.g. a claim about the model's internal state)."""
GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["point_scores", "claims"],
    "properties": {
        "point_scores": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 10},
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim", "label"],
                "properties": {
                    "claim": {"type": "string"},
                    "label": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "not_covered"],
                    },
                },
            },
        },
    },
}


def answer_text(rec: dict[str, Any]) -> str:
    res = rec.get("result") or {}
    if not isinstance(res, dict):
        return str(res)
    return "\n\n".join(f"{k}: {res[k]}" for k in ("hypothesis", "evidence", "rival") if res.get(k))


async def call(system: str, user: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    for _ in range(3):
        got = await json_call(
            client(), model=MODEL, system=system, user=user, schema=schema, max_tokens=6000
        )
        if got:
            return got
    return None


def ok(g: dict[str, Any] | None) -> bool:
    """A usable point list: 3-12 strings (a reply can come back as one string)."""
    pts = (g or {}).get("points")
    return isinstance(pts, list) and 3 <= len(pts) <= 12 and all(isinstance(x, str) for x in pts)


def stage_points(args: Any) -> None:
    incs = load_incidents(read_ids(Path(args.ids)), args.dataset)
    todo = [p for p in incs if not (args.out / "points" / f"{p}.json").exists()]

    async def one(pid: str) -> None:
        gt = incs[pid].ground_truth
        targets = "\n".join(
            f"[{t['id']}] {t['cause']}\n  evidence: " + " | ".join(t.get("evidence", []))
            for t in gt["targets"]
        )
        got = await call(
            POINTS_SYSTEM, f"BEHAVIOUR: {incs[pid].behaviour}\n\nCAUSES:\n{targets}", POINTS_SCHEMA
        )
        if ok(got):
            listing = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(got["points"]))
            user = f"BEHAVIOUR: {incs[pid].behaviour}\n\nPOINTS:\n{listing}"
            dedup = await call(DEDUP_SYSTEM, user, POINTS_SCHEMA)
            got = dedup if ok(dedup) else got
            f = args.out / "points" / f"{pid}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(
                json.dumps({"prompt_id": pid, "points": got["points"]}, ensure_ascii=False)
            )

    print(f"points: {len(todo)} incidents")
    asyncio.run(run_many([lambda p=p: one(p) for p in todo], args.concurrency))


def stage_grade(args: Any) -> None:
    incs = load_incidents(read_ids(Path(args.ids)), args.dataset)
    jobs = []
    for arm, d in ARMS.items():
        for f in sorted(Path(d).glob("*.d*.json")):
            pid = f.name.split(".")[0]
            pf = args.out / "points" / f"{pid}.json"
            out = args.out / "grades" / arm / f.name
            if pid in incs and pf.exists() and not out.exists():
                jobs.append((arm, f, pf, out))

    async def one(arm: str, f: Path, pf: Path, out: Path) -> None:
        rec = json.loads(f.read_text())
        points = json.loads(pf.read_text())["points"]
        gt = incs[rec["prompt_id"]].ground_truth
        user = (
            f"BEHAVIOUR: {incs[rec['prompt_id']].behaviour}\n\nESTABLISHED POINTS:\n"
            + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))
            + f"\n\nFULL ESTABLISHED EXPLANATION (with ruled-out alternatives):\n{gt['judge_reference']}"
            + f"\n\nANSWER:\n{answer_text(rec)}"
        )
        got = await call(GRADE_SYSTEM, user, GRADE_SCHEMA)
        scores = (got or {}).get("point_scores")
        if isinstance(scores, list) and len(scores) == len(points) and "claims" in got:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps({"prompt_id": rec["prompt_id"], "arm": arm, **got}, ensure_ascii=False)
            )

    print(f"grade: {len(jobs)} answers")
    asyncio.run(run_many([lambda j=j: one(*j) for j in jobs], args.concurrency))


def stage_report(args: Any) -> None:
    g: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for f in (args.out / "grades").glob("*/*.json"):
        r = json.loads(f.read_text())
        g[r["arm"]][r["prompt_id"]].append(r)
    full = {a: {p: v for p, v in d.items() if len(v) == 5} for a, d in g.items()}
    common = sorted(set.intersection(*[set(x) for x in full.values()]))
    # a point is obvious when transcript scores it >= 7/10 in >= 3 of its 5 draws
    obvious = {
        p: [
            sum(r["point_scores"][i] >= 7 for r in full["transcript"][p]) >= 3
            for i in range(len(full["transcript"][p][0]["point_scores"]))
        ]
        for p in common
    }

    def per_inc(arm: str, p: str) -> dict[str, float]:
        rs = full[arm][p]
        n_pts = len(rs[0]["point_scores"])
        non = [i for i in range(n_pts) if not obvious[p][i]]
        rec_all = sum(sum(r["point_scores"]) / (10 * n_pts) for r in rs) / len(rs)
        rec_non = (
            sum(sum(r["point_scores"][i] for i in non) / (10 * len(non)) for r in rs) / len(rs)
            if non
            else float("nan")
        )
        claims = [c for r in rs for c in r["claims"]]
        n = max(len(claims), 1)
        return {
            "recall": rec_all,
            "recall_nonobvious": rec_non,
            "contradicted": sum(c["label"] == "contradicted" for c in claims) / n,
            "not_covered": sum(c["label"] == "not_covered" for c in claims) / n,
            "claims_per_answer": len(claims) / len(rs),
        }

    conf = json.loads(Path(args.confidence).read_text())
    ranked = sorted(conf, key=lambda p: conf[p]["top_p"])
    low = set(ranked[: len(ranked) // 3])
    hard = set(read_ids(Path(args.hard)))
    strata = {
        "all": common,
        "least-confident third": [p for p in common if p in low],
        "hard": [p for p in common if p in hard],
    }
    metrics = ["recall", "recall_nonobvious", "contradicted", "not_covered", "claims_per_answer"]
    n_non = sum(sum(not o for o in obvious[p]) for p in common)
    n_pts = sum(len(obvious[p]) for p in common)
    lines = [
        f"# Point-level grading: {len(common)} incidents x 5 draws",
        "",
        f"{n_pts} established points, {n_non} non-obvious (transcript scores them >= 7/10 in < 3 of 5 draws); recall = mean point score / 10.",
        "",
    ]
    for name, S in strata.items():
        lines += [
            f"## {name} ({len(S)} incidents)",
            "",
            "| arm | "
            + " | ".join(metrics)
            + " | non-obvious recall vs transcript | contradicted vs transcript |",
            "|" + "---|" * (len(metrics) + 3),
        ]
        vals = {a: {p: per_inc(a, p) for p in S} for a in full}
        for a in sorted(full, key=lambda a: (a != "transcript", a)):
            row = [
                sum(v[m] for v in vals[a].values() if v[m] == v[m])
                / max(sum(1 for v in vals[a].values() if v[m] == v[m]), 1)
                for m in metrics
            ]
            cmp = []
            for m in ("recall_nonobvious", "contradicted"):
                if a == "transcript":
                    cmp.append("-")
                    continue
                d = [
                    vals[a][p][m] - vals["transcript"][p][m]
                    for p in S
                    if vals[a][p][m] == vals[a][p][m]
                    and vals["transcript"][p][m] == vals["transcript"][p][m]
                ]
                rng = random.Random(0)
                bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
                star = "*" if bs[100] > 0 or bs[3899] < 0 else ""
                cmp.append(
                    f"{100 * sum(d) / len(d):+.1f} [{100 * bs[100]:+.1f}, {100 * bs[3899]:+.1f}]{star}"
                )
            lines.append(
                f"| {a} | "
                + " | ".join(
                    f"{100 * x:.1f}%" if m != "claims_per_answer" else f"{x:.1f}"
                    for x, m in zip(row, metrics, strict=True)
                )
                + " | "
                + " | ".join(cmp)
                + " |"
            )
        lines.append("")
    text = "\n".join(lines)
    (args.out / "report.md").write_text(text)
    print(text)


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.stage = pydra.REQUIRED
        self.out = "/tmp/chive_points"
        self.ids = "data/chive/splits/climb_test204.txt"
        self.dataset = DATASET
        self.hard = "data/chive/splits/hard92.txt"
        self.confidence = "/workspace/agam/chive_runs_dj/confidence.json"
        self.concurrency = 64

    def finalize(self) -> None:
        self.out = Path(self.out)


STAGES = {"points": stage_points, "grade": stage_grade, "report": stage_report}


@pydra.main(Config)
def main(cfg: Config) -> None:
    STAGES[cfg.stage](cfg)


if __name__ == "__main__":
    main()
