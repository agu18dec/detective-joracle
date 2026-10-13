"""Point-level grading of free-form explanations: which parts of the established explanation each
method recovers (especially the non-obvious ones), and how often it asserts something the
experiments disproved.

Stages (resumable; JSON under out=):
  points  one Opus call per incident splits CHIVE's established causes into distinct atomic points
          (a second call merges near-duplicates) -> points/<pid>.json
  tag     one call per incident marks which points belong to the PRIMARY cause (T1) -> tags/<pid>.json
  plain   an independent baseline: a no-tools Opus answer, 3 per incident -> plain/<pid>.d<k>.json.
          A point is OBVIOUS when the plain answer scores it >= 7 (1-10 scale) in >= 2 of its 3 answers.
  grade   one call per final answer (each arm, and plain): every point scored 1-10; the answer
          split into its own claims, each supported / contradicted (it asserts something CHIVE ruled
          out or a measured result contradicts) / rejected (it names a rival only to rule it out)
          / not_covered -> grades/<arm>/<pid>.d<k>.json
  rep     the grader's reliability: a fresh re-grade of a random sample -> grades_rep/
  report  per arm and stratum: recall of non-obvious points and of primary-cause points (mean
          (point score - 1) / 9), the disproved-claim rate (contradicted / asserted claims, rejected
          rivals excluded), paired bootstrap vs transcript; grader agreement.
"""

import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydra

from detective_joracle.agent.loop import run_many
from detective_joracle.chive.data import DATASET, Incident, load_incidents, read_ids
from detective_joracle.chive.llm import client, json_call

MODEL = "claude-opus-5"
ARMS = {  # arm -> record dir (held-out free-form, frozen harness, 5 draws)
    "transcript": "/workspace/agam/chive_runs_dj/climb_test/transcript",
    "olens_rl_B": "/workspace/agam/chive_runs_dj/climb_test/olens_rl_B",
    "jlens_B": "/workspace/agam/chive_runs_dj/climb_test/jlens_B",
    "mlnla_B": "/tmp/chive_mlnla/climb_test/mlnla_B",
}
N_PLAIN = 3

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
TAG_SYSTEM = """You get the established causes of a language model's behaviour ([T1] is the \
primary cause) and points derived from them. For each point say whether it states or directly \
belongs to the PRIMARY cause [T1]."""
TAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["primary"],
    "properties": {"primary": {"type": "array", "items": {"type": "boolean"}}},
}
PLAIN_SYSTEM = """You are shown a conversation with a language model, one of its responses, and \
a behaviour that the response shows (and how often it recurs when the model is re-run). Explain \
why the model behaves this way: the feature(s) of the conversation that drive it and how. Reason \
only from the conversation and what you know about language models."""
PLAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["hypothesis"],
    "properties": {"hypothesis": {"type": "string"}},
}
GRADE_SYSTEM = """You grade an interpretability agent's explanation of a language model's \
behaviour against the established explanation. Two tasks.
1. For each established POINT, score 1-10 how fully and precisely the answer states it (same \
meaning, any wording): 10 = stated clearly and specifically, 5 = stated partly or vaguely, 1 = \
absent or contradicted.
2. Split the ANSWER into its own distinct claims about why the model behaves this way (at most \
10, skip restatements of the behaviour). Label each: "supported" (the established explanation \
says it or directly implies it), "contradicted" (the answer ASSERTS it and it matches a ruled-out \
alternative or conflicts with an established point or a measured result), "rejected" (the answer \
names it only as a rival it rules out), or "not_covered" (the established explanation neither \
supports nor contradicts it, e.g. a claim about the model's internal state)."""
LABELS = ["supported", "contradicted", "rejected", "not_covered"]
RIVALS_SYSTEM = """You get an interpretability agent's explanation of a language model's \
behaviour and the alternatives that counterfactual experiments RULED OUT (each was tested and \
does not drive the behaviour). For each alternative, say what the answer does with it: \
"rejected" (the answer explicitly rules it out or argues against it), "asserted" (the answer \
claims it as a cause or contributor), or "absent" (the answer does not address it)."""
RIVALS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["rivals"],
    "properties": {
        "rivals": {
            "type": "array",
            "items": {"type": "string", "enum": ["rejected", "asserted", "absent"]},
        }
    },
}
GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["point_scores", "claims"],
    "properties": {
        "point_scores": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim", "label"],
                "properties": {
                    "claim": {"type": "string"},
                    "label": {"type": "string", "enum": LABELS},
                },
            },
        },
    },
}


def ok_points(g: dict[str, Any] | None) -> bool:
    """A usable point list: 3-12 strings (a reply can come back as one string)."""
    pts = (g or {}).get("points")
    return isinstance(pts, list) and 3 <= len(pts) <= 12 and all(isinstance(x, str) for x in pts)


def answer_text(rec: dict[str, Any]) -> str:
    res = rec.get("result") or {}
    if not isinstance(res, dict):
        return str(res)
    return "\n\n".join(f"{k}: {res[k]}" for k in ("hypothesis", "evidence", "rival") if res.get(k))


def targets_text(inc: Incident) -> str:
    return "\n".join(
        f"[{t['id']}] {t['cause']}\n  evidence: " + " | ".join(t.get("evidence", []))
        for t in inc.ground_truth["targets"]
    )


async def call(system: str, user: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    for _ in range(3):
        got = await json_call(
            client(), model=MODEL, system=system, user=user, schema=schema, max_tokens=6000
        )
        if got:
            return got
    return None


def run(jobs: list[Any], concurrency: int, name: str) -> None:
    print(f"{name}: {len(jobs)} calls", flush=True)
    asyncio.run(run_many(jobs, concurrency))


def write(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False))


def stage_points(args: Any, incs: dict[str, Incident]) -> None:
    async def one(pid: str) -> None:
        user = f"BEHAVIOUR: {incs[pid].behaviour}\n\nCAUSES:\n{targets_text(incs[pid])}"
        got = await call(POINTS_SYSTEM, user, POINTS_SCHEMA)
        if ok_points(got):
            assert got is not None
            listing = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(got["points"]))
            dedup = await call(
                DEDUP_SYSTEM,
                f"BEHAVIOUR: {incs[pid].behaviour}\n\nPOINTS:\n{listing}",
                POINTS_SCHEMA,
            )
            got = dedup if ok_points(dedup) else got
            write(args.out / "points" / f"{pid}.json", {"prompt_id": pid, "points": got["points"]})

    run(
        [lambda p=p: one(p) for p in incs if not (args.out / "points" / f"{p}.json").exists()],
        args.concurrency,
        "points",
    )


def stage_tag(args: Any, incs: dict[str, Incident]) -> None:
    async def one(pid: str) -> None:
        pts = json.loads((args.out / "points" / f"{pid}.json").read_text())["points"]
        listing = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(pts))
        got = await call(
            TAG_SYSTEM, f"CAUSES:\n{targets_text(incs[pid])}\n\nPOINTS:\n{listing}", TAG_SCHEMA
        )
        if got and len(got.get("primary") or []) == len(pts):
            write(args.out / "tags" / f"{pid}.json", {"prompt_id": pid, "primary": got["primary"]})

    run(
        [
            lambda p=p: one(p)
            for p in incs
            if (args.out / "points" / f"{p}.json").exists()
            and not (args.out / "tags" / f"{p}.json").exists()
        ],
        args.concurrency,
        "tag",
    )


def stage_plain(args: Any, incs: dict[str, Incident]) -> None:
    async def one(pid: str, d: int) -> None:
        inc = incs[pid]
        convo = "\n\n".join(
            f"T{i + 1} [{m['role']}]:\n{m['content']}" for i, m in enumerate(inc.messages)
        )
        user = (
            f"CONVERSATION:\n{convo}\n\nRESPONSE:\n{inc.completion}\n\nBEHAVIOUR (in {inc.k} of "
            f"{inc.n} runs): {inc.behaviour}"
        )
        got = await call(PLAIN_SYSTEM, user, PLAIN_SCHEMA)
        if got and got.get("hypothesis"):
            write(
                args.out / "plain" / f"{pid}.d{d}.json",
                {"prompt_id": pid, "result": {"hypothesis": got["hypothesis"]}},
            )

    run(
        [
            lambda p=p, d=d: one(p, d)
            for p in incs
            for d in range(N_PLAIN)
            if not (args.out / "plain" / f"{p}.d{d}.json").exists()
        ],
        args.concurrency,
        "plain",
    )


async def grade_one(args: Any, incs: dict[str, Incident], arm: str, f: Path, out: Path) -> None:
    rec = json.loads(f.read_text())
    pts = json.loads((args.out / "points" / f"{rec['prompt_id']}.json").read_text())["points"]
    inc = incs[rec["prompt_id"]]
    user = (
        f"BEHAVIOUR: {inc.behaviour}\n\nESTABLISHED POINTS:\n"
        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(pts))
        + f"\n\nFULL ESTABLISHED EXPLANATION (with ruled-out alternatives):\n{inc.ground_truth['judge_reference']}"
        + f"\n\nANSWER:\n{answer_text(rec)}"
    )
    got = await call(GRADE_SYSTEM, user, GRADE_SCHEMA)
    scores = (got or {}).get("point_scores")
    if (
        got
        and isinstance(scores, list)
        and len(scores) == len(pts)
        and isinstance(got.get("claims"), list)
    ):
        write(out, {"prompt_id": rec["prompt_id"], "arm": arm, **got})


def answer_files(args: Any, incs: dict[str, Incident]) -> list[tuple[str, Path]]:
    files = [(arm, f) for arm, d in ARMS.items() for f in sorted(Path(d).glob("*.d*.json"))]
    files += [("plain", f) for f in sorted((args.out / "plain").glob("*.json"))]
    return [
        (a, f)
        for a, f in files
        if f.name.split(".")[0] in incs
        and (args.out / "points" / f"{f.name.split('.')[0]}.json").exists()
    ]


def stage_grade(args: Any, incs: dict[str, Incident]) -> None:
    jobs = [(a, f, args.out / "grades" / a / f.name) for a, f in answer_files(args, incs)]
    run(
        [lambda j=j: grade_one(args, incs, *j) for j in jobs if not j[2].exists()],
        args.concurrency,
        "grade",
    )


def rival_texts(inc: Incident) -> list[str]:
    return [x.get("cause") or x.get("edit") or "" for x in inc.ground_truth.get("ruled_out", [])]


def stage_rivals(args: Any, incs: dict[str, Incident]) -> None:
    async def one(arm: str, f: Path, out: Path) -> None:
        rec = json.loads(f.read_text())
        rivals = rival_texts(incs[rec["prompt_id"]])
        listing = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(rivals))
        user = f"RULED-OUT ALTERNATIVES:\n{listing}\n\nANSWER:\n{answer_text(rec)}"
        got = await call(RIVALS_SYSTEM, user, RIVALS_SCHEMA)
        if got and isinstance(got.get("rivals"), list) and len(got["rivals"]) == len(rivals):
            write(out, {"prompt_id": rec["prompt_id"], "arm": arm, "rivals": got["rivals"]})

    jobs = [
        (a, f, args.out / "rivals" / a / f.name)
        for a, f in answer_files(args, incs)
        if a != "plain" and rival_texts(incs[f.name.split(".")[0]])
    ]
    run([lambda j=j: one(*j) for j in jobs if not j[2].exists()], args.concurrency, "rivals")


def stage_rep(args: Any, incs: dict[str, Incident]) -> None:
    files = [x for x in answer_files(args, incs) if x[0] != "plain"]
    sample = random.Random(0).sample(files, min(args.n_rep, len(files)))
    jobs = [(a, f, args.out / "grades_rep" / a / f.name) for a, f in sample]
    run(
        [lambda j=j: grade_one(args, incs, *j) for j in jobs if not j[2].exists()],
        args.concurrency,
        "rep",
    )


def load_grades(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    g: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for f in root.glob("*/*.json"):
        r = json.loads(f.read_text())
        r["file"] = f.name
        g[r["arm"]][r["prompt_id"]].append(r)
    return g


def stage_report(args: Any, incs: dict[str, Incident]) -> None:
    g = load_grades(args.out / "grades")
    plain = {p: v for p, v in g.pop("plain", {}).items() if len(v) == N_PLAIN}
    full = {a: {p: v for p, v in d.items() if len(v) == 5} for a, d in g.items()}
    tags = {
        f.stem: json.loads(f.read_text())["primary"] for f in (args.out / "tags").glob("*.json")
    }
    common = sorted(set.intersection(*[set(x) for x in full.values()]) & set(plain) & set(tags))
    rivals: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for f in (args.out / "rivals").glob("*/*.json"):
        rv = json.loads(f.read_text())
        rivals[rv["arm"]][rv["prompt_id"]].append(rv)
    obvious = {
        p: [sum(r["point_scores"][i] >= 7 for r in plain[p]) >= 2 for i in range(len(tags[p]))]
        for p in common
    }

    def per_inc(arm: str, p: str) -> dict[str, float]:
        rs = full[arm][p]
        groups = {
            "nonobvious": [i for i, o in enumerate(obvious[p]) if not o],
            "primary": [i for i, t in enumerate(tags[p]) if t],
        }
        out: dict[str, float] = {}
        for name, idx in groups.items():
            out[name] = (
                sum(sum(r["point_scores"][i] - 1 for i in idx) / (9 * len(idx)) for r in rs)
                / len(rs)
                if idx
                else float("nan")
            )
        claims = [c for r in rs for c in r["claims"] if c.get("label") != "rejected"]
        out["disproved"] = sum(c["label"] == "contradicted" for c in claims) / max(len(claims), 1)
        out["per_answer"] = sum(c["label"] == "contradicted" for c in claims) / len(rs)
        rv = [x for r in rivals.get(arm, {}).get(p, []) for x in r["rivals"]]
        out["rival_rejected"] = sum(x == "rejected" for x in rv) / len(rv) if rv else float("nan")
        out["rival_asserted"] = sum(x == "asserted" for x in rv) / len(rv) if rv else float("nan")
        return out

    conf = json.loads(Path(args.confidence).read_text())
    ranked = sorted(conf, key=lambda p: conf[p]["top_p"])
    low = set(ranked[: len(ranked) // 3])
    hard = set(read_ids(Path(args.hard)))
    strata = {
        "all": common,
        "least-confident third": [p for p in common if p in low],
        "hard": [p for p in common if p in hard],
    }
    metrics = [
        "nonobvious",
        "primary",
        "disproved",
        "per_answer",
        "rival_rejected",
        "rival_asserted",
    ]
    n_pts = sum(len(obvious[p]) for p in common)
    n_non = sum(sum(not o for o in obvious[p]) for p in common)
    lines = [
        f"# Point-level grading: {len(common)} incidents x 5 draws",
        "",
        f"{n_pts} distinct points; {n_non} non-obvious (an independent no-tools answer scores them "
        f">= 7 (1-10 scale) in < 2 of 3 tries); {sum(sum(tags[p]) for p in common)} belong to the primary cause. "
        "Recall = mean (point score - 1) / 9 on the 1-10 scale. Disproved = contradicted / asserted claims (rivals the "
        "answer rejects excluded); per_answer = disproved claims per answer.",
        "",
    ]
    summary: dict[str, Any] = {}
    for name, S in strata.items():
        vals = {a: {p: per_inc(a, p) for p in S} for a in full}
        lines += [
            f"## {name} ({len(S)} incidents)",
            "",
            "| arm | non-obvious recall | primary-cause recall | disproved rate | disproved per answer | "
            "rivals rejected | rivals asserted | non-obvious vs transcript | disproved vs transcript | "
            "rivals rejected vs transcript |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        summary[name] = {}
        for a in sorted(full, key=lambda a: (a != "transcript", a)):
            means = {
                m: sum(v[m] for v in vals[a].values() if v[m] == v[m])
                / max(1, sum(v[m] == v[m] for v in vals[a].values()))
                for m in metrics
            }
            summary[name][a] = means
            cmp = []
            for m in ("nonobvious", "disproved", "rival_rejected"):
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
                cmp.append(
                    f"{100 * sum(d) / len(d):+.1f} [{100 * bs[100]:+.1f}, {100 * bs[3899]:+.1f}]"
                    + ("*" if bs[100] > 0 or bs[3899] < 0 else "")
                )
            lines.append(
                f"| {a} | {100 * means['nonobvious']:.1f}% | {100 * means['primary']:.1f}% | "
                f"{100 * means['disproved']:.1f}% | {means['per_answer']:.2f} | "
                f"{100 * means['rival_rejected']:.1f}% | {100 * means['rival_asserted']:.1f}% | "
                + " | ".join(cmp)
                + " |"
            )
        lines.append("")
    rep = load_grades(args.out / "grades_rep")
    a_pts, b_pts, a_con, b_con = [], [], [], []
    for arm, d in rep.items():
        for p, rs in d.items():
            for r in rs:
                orig = next((o for o in g.get(arm, {}).get(p, []) if o["file"] == r["file"]), None)
                if orig and len(orig["point_scores"]) == len(r["point_scores"]):
                    a_pts += orig["point_scores"]
                    b_pts += r["point_scores"]
                    a_con.append(sum(c["label"] == "contradicted" for c in orig["claims"]))
                    b_con.append(sum(c["label"] == "contradicted" for c in r["claims"]))
    if a_pts:

        def corr(x: list[float], y: list[float]) -> float:
            mx, my = sum(x) / len(x), sum(y) / len(y)
            cov = sum((i - mx) * (j - my) for i, j in zip(x, y, strict=True))
            return cov / (
                (sum((i - mx) ** 2 for i in x) * sum((j - my) ** 2 for j in y)) ** 0.5 or 1
            )

        same = sum((i >= 7) == (j >= 7) for i, j in zip(a_pts, b_pts, strict=True)) / len(a_pts)
        lines += [
            "## Grader reliability (independent re-grade)",
            "",
            f"{len(a_con)} answers re-graded: point-score correlation {corr(a_pts, b_pts):.2f}; "
            f"same stated/not-stated call (>= 7) on {100 * same:.0f}% of points; disproved-claim count "
            f"correlation {corr(a_con, b_con):.2f}.",
            "",
        ]
    text = "\n".join(lines)
    (args.out / "report.md").write_text(text)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(text)


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.stage = pydra.REQUIRED
        self.out = "/tmp/chive_points3"
        self.ids = "data/chive/splits/climb_test204.txt"
        self.dataset = DATASET
        self.hard = "data/chive/splits/hard92.txt"
        self.confidence = "/workspace/agam/chive_runs_dj/confidence.json"
        self.concurrency = 96
        self.n_rep = 300

    def finalize(self) -> None:
        self.out = Path(self.out)


STAGES = {
    "points": stage_points,
    "tag": stage_tag,
    "plain": stage_plain,
    "grade": stage_grade,
    "rivals": stage_rivals,
    "rep": stage_rep,
    "report": stage_report,
}


@pydra.main(Config)
def main(cfg: Config) -> None:
    incs = load_incidents(read_ids(Path(cfg.ids)), cfg.dataset)
    STAGES[cfg.stage](cfg, incs)


if __name__ == "__main__":
    main()
