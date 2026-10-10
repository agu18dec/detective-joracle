"""CHIVE forensics evals: an investigator with ONE method explains a behaviour; judged vs ground truth.

Stages (each resumable; records are JSON files under --out):
  agent      run investigations: arm x incident x draw -> <out>/<eval>/<arm>/<pid>.d<k>.json
  judge      score every record's final hypothesis (1-10, 3 replicates, pass >= 7)
  top20      one transcript-only top-20 list per incident (shared by every arm), judged per item
  concluder  two-agent B: a fresh agent sees only A's experiment(s) and writes the hypothesis
  report     pass@1 per arm, paired differences vs transcript (all and hard stratum), errors, tools

Configured with pydra (`key=value` overrides). Arms: `transcript`, or `NAME=BANK_DIR:KIND[:shuffled]`
with KIND one of olens | nla | jlens.

    python scripts/run_chive.py stage=agent eval=readonly ids=ids/pilot15.txt out=runs/chive \\
        "arms=['transcript','olens_rl=/workspace/agam/chive_bank/s3d_rl600_all:olens']" draws=1
"""

import asyncio
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pydra

from detective_joracle.agent.backends import async_openai_compatible_backend
from detective_joracle.agent.loop import Budget, RunRecord, run_many, run_tool_loop_async
from detective_joracle.chive import judge as J
from detective_joracle.chive.data import Incident, load_incidents, read_ids
from detective_joracle.chive.executor import Executor
from detective_joracle.chive.llm import client, json_call
from detective_joracle.chive.prompts import (
    CONCLUDER_SCHEMA,
    CONCLUDER_SYSTEM,
    TOP20_SCHEMA,
    TOP20_SYSTEM,
    seed,
    system_prompt,
)
from detective_joracle.chive.readers import load_reader
from detective_joracle.chive.tools import SUBMIT, ChiveTools, render_experiments, schemas

AGENT_MODEL = "claude-opus-5"


def thinking_backend(max_tokens: int, budget: int) -> Any:
    """Extended thinking on every turn except forced ones (Anthropic disallows thinking with a
    forced tool call, and the loop forces the terminal at the end)."""
    c = client()
    think = async_openai_compatible_backend(
        c, AGENT_MODEL, cache=True, max_tokens=max_tokens,
        extra_body={"thinking": {"type": "enabled", "budget_tokens": budget}} if budget else None)
    plain = async_openai_compatible_backend(c, AGENT_MODEL, cache=True, max_tokens=max_tokens)

    async def call(messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                   force_tool: str | None = None) -> Any:
        return await (plain if force_tool else think)(messages, tools, force_tool)

    return call


def log(out: Path, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (out / "run.log").open("a") as f:
        f.write(line + "\n")


def parse_arm(spec: str) -> dict[str, Any]:
    if spec == "transcript":
        return {"name": "transcript", "bank": None, "kind": None, "shuffled": False}
    name, _, rest = spec.partition("=")
    parts = rest.split(":")
    if len(parts) not in (2, 3) or parts[1] not in ("olens", "nla", "jlens"):
        raise SystemExit(f"bad --arm {spec!r}: NAME=BANK_DIR:KIND[:shuffled]")
    return {"name": name, "bank": Path(parts[0]), "kind": parts[1],
            "shuffled": len(parts) == 3 and parts[2] == "shuffled"}


def rec_path(out: Path, ev: str, arm: str, pid: str, draw: int) -> Path:
    return out / ev / arm / f"{pid}.d{draw}.json"


def done(path: Path) -> bool:
    if not path.exists():
        return False
    r = json.loads(path.read_text())
    return not str(r.get("stopped_by", "")).startswith("error")


# --------------------------------------------------------------------------- agent

async def investigate(args: Any, arm: dict[str, Any], inc: Incident,
                      partner: Incident | None, draw: int, path: Path) -> None:
    t0 = time.time()
    rec = RunRecord(organism=inc.prompt_id, condition=args.eval, arm=arm["name"],
                    auditor=AGENT_MODEL, seed=draw)
    extra: dict[str, Any] = {}
    try:
        reader = (None if arm["bank"] is None else
                  load_reader(arm["name"], arm["kind"], arm["bank"], inc,
                              partner if arm["shuffled"] else None))
        executor = Executor(args.target) if args.experiments else None
        tools = ChiveTools(inc, reader, executor, args.experiments)
        backend = thinking_backend(args.max_tokens, args.thinking_budget)
        await run_tool_loop_async(tools, schemas(reader, args.experiments),
                                  system_prompt(reader, args.experiments, Executor.N), seed(inc),
                                  backend, Budget(output_tokens=args.output_budget,
                                                  max_calls=args.max_calls),
                                  rec, terminal=SUBMIT)
        rec.result = tools.finished
        rec.tool_log = tools.log
        extra = {"events": tools.events, "notebook": tools.notebook_text,
                 "experiments": tools.experiments_run,
                 "reader": None if reader is None else {"name": reader.name, "kind": reader.kind,
                                                        "source_id": reader.source_id}}
    except Exception as e:  # recorded, logged, retried by the next resume
        rec.stopped_by = f"error: {type(e).__name__}: {str(e)[:300]}"
        extra["traceback"] = traceback.format_exc()[-3000:]
    row = {**rec.to_json(), **extra, "prompt_id": inc.prompt_id, "eval": args.eval,
           "experiments_allowed": args.experiments, "seconds": round(time.time() - t0, 1)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False))
    errs = sum(1 for e in extra.get("events", []) if e.get("error"))
    log(args.out, f"{args.eval}/{arm['name']}/{inc.prompt_id}.d{draw}: {rec.stopped_by} | "
                  f"calls {len(extra.get('events', []))} | tool errors {errs} | "
                  f"out tokens {rec.output_tokens} | {row['seconds']}s")


def stage_agent(args: Any) -> None:
    ids = read_ids(args.ids)
    incs = load_incidents(args.incidents, ids)
    order = sorted(ids)
    partner = {p: incs[order[(i + len(order) // 2) % len(order)]] for i, p in enumerate(order)}
    arms = [parse_arm(a) for a in args.arm]
    jobs = []
    for arm in arms:
        for pid in ids:
            for d in range(args.draws):
                path = rec_path(args.out, args.eval, arm["name"], pid, d)
                if not done(path):
                    jobs.append(lambda a=arm, i=incs[pid], p=partner[pid], d=d, path=path:
                                investigate(args, a, i, p, d, path))
    log(args.out, f"agent: {len(jobs)} runs to do ({len(arms)} arms x {len(ids)} incidents x "
                  f"{args.draws} draws, eval={args.eval}, experiments={args.experiments})")
    asyncio.run(run_many(jobs, args.concurrency))


# --------------------------------------------------------------------------- judge / top20 / B

def records(out: Path, ev: str) -> list[Path]:
    return sorted((out / ev).glob("*/*.d*.json"))


async def _judge_one(path: Path, inc: Incident, key: str, text: str) -> None:
    got = await J.score(inc.behaviour, inc.criteria, text)
    r = json.loads(path.read_text())
    r[key] = got
    path.write_text(json.dumps(r, ensure_ascii=False))


def stage_judge(args: Any) -> None:
    incs = load_incidents(args.incidents, read_ids(args.ids))
    jobs = []
    for path in records(args.out, args.eval):
        r = json.loads(path.read_text())
        pid = r["prompt_id"]
        if pid not in incs or "judge" in r or not r.get("result"):
            continue
        jobs.append(lambda path=path, inc=incs[pid], h=r["result"]["hypothesis"]:
                    _judge_one(path, inc, "judge", h))
    log(args.out, f"judge: {len(jobs)} records to score")
    asyncio.run(run_many(jobs, args.concurrency))


def stage_top20(args: Any) -> None:
    incs = load_incidents(args.incidents, read_ids(args.ids))
    d = args.out / "top20"
    d.mkdir(parents=True, exist_ok=True)

    async def one(inc: Incident) -> None:
        f = d / f"{inc.prompt_id}.json"
        if f.exists():
            return
        user = seed(inc).rsplit("Positions are tokens", 1)[0]
        got = await json_call(client(), model=AGENT_MODEL, system=TOP20_SYSTEM, user=user,
                              schema=TOP20_SCHEMA, max_tokens=6000)
        if not got:
            log(args.out, f"top20 {inc.prompt_id}: no list")
            return
        matches = await J.match_list(inc.behaviour, inc.criteria, got["hypotheses"])
        f.write_text(json.dumps({"prompt_id": inc.prompt_id, "hypotheses": got["hypotheses"],
                                 "matches": matches}, ensure_ascii=False))

    asyncio.run(run_many([lambda i=i: one(i) for i in incs.values()], args.concurrency))
    log(args.out, f"top20: {len(list(d.glob('*.json')))} lists")


def stage_concluder(args: Any) -> None:
    incs = load_incidents(args.incidents, read_ids(args.ids))

    async def one(path: Path, inc: Incident, exps: list[dict[str, Any]]) -> None:
        user = (seed(inc).rsplit("Positions are tokens", 1)[0]
                + "\n\n# The other investigator's experiment(s)\n" + render_experiments(exps))
        got = await json_call(client(), model=AGENT_MODEL, system=CONCLUDER_SYSTEM, user=user,
                              schema=CONCLUDER_SCHEMA, max_tokens=3000)
        r = json.loads(path.read_text())
        r["concluder"] = got
        if got:
            r["concluder_judge"] = await J.score(inc.behaviour, inc.criteria, got["hypothesis"])
        path.write_text(json.dumps(r, ensure_ascii=False))

    jobs = []
    for path in records(args.out, args.eval):
        r = json.loads(path.read_text())
        if r["prompt_id"] in incs and r.get("experiments") and "concluder" not in r:
            jobs.append(lambda path=path, inc=incs[r["prompt_id"]], e=r["experiments"]:
                        one(path, inc, e))
    log(args.out, f"concluder: {len(jobs)} records")
    asyncio.run(run_many(jobs, args.concurrency))


# --------------------------------------------------------------------------- report

def stage_report(args: Any) -> None:
    import random

    ids = set(read_ids(args.ids))
    hard = set(read_ids(args.hard)) if args.hard else set()
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    stops: dict[str, Counter[str]] = defaultdict(Counter)
    tools_used: dict[str, Counter[str]] = defaultdict(Counter)
    errors: dict[str, int] = Counter()
    n_runs: dict[str, int] = Counter()
    for path in records(args.out, args.eval):
        r = json.loads(path.read_text())
        if r["prompt_id"] not in ids:
            continue
        arm = r["arm"]
        n_runs[arm] += 1
        stops[arm][r["stopped_by"].split(":")[0]] += 1
        for e in r.get("events", []):
            tools_used[arm][e["tool"]] += 1
            errors[arm] += bool(e.get("error"))
        j = r.get(args.key) or {}
        if j.get("score") is not None:
            per[arm][r["prompt_id"]].append(1.0 if j["pass"] else 0.0)

    def mean(arm: str, sub: set[str] | None = None) -> tuple[float, int]:
        xs = [sum(v) / len(v) for p, v in per[arm].items() if sub is None or p in sub]
        return (sum(xs) / len(xs) if xs else float("nan")), len(xs)

    def paired(a: str, b: str, sub: set[str] | None = None) -> str:
        common = [p for p in per[a] if p in per[b] and (sub is None or p in sub)]
        if not common:
            return "-"
        d = [sum(per[b][p]) / len(per[b][p]) - sum(per[a][p]) / len(per[a][p]) for p in common]
        boots = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(4000))
        return (f"{100 * sum(d) / len(d):+.1f} [{100 * boots[100]:+.1f}, "
                f"{100 * boots[3899]:+.1f}] n={len(common)}")

    lines = [f"# CHIVE eval `{args.eval}` ({args.key})", "",
             "| arm | runs | pass@1 (all) | pass@1 (hard) | vs transcript (all) | vs transcript "
             "(hard) | stops | tool errors |", "|---|---|---|---|---|---|---|---|"]
    for arm in sorted(n_runs):
        m, n = mean(arm)
        mh, nh = mean(arm, hard)
        vs = paired("transcript", arm) if arm != "transcript" and "transcript" in per else "-"
        vsh = (paired("transcript", arm, hard)
               if arm != "transcript" and "transcript" in per and hard else "-")
        lines.append(f"| {arm} | {n_runs[arm]} | {m:.1%} (n={n}) | {mh:.1%} (n={nh}) | {vs} | "
                     f"{vsh} | {dict(stops[arm])} | {errors[arm]} |")
    lines += ["", "## Tool calls per arm", ""]
    lines += [f"- {arm}: {dict(tools_used[arm].most_common())}" for arm in sorted(tools_used)]
    text = "\n".join(lines)
    (args.out / args.eval / f"report_{args.key}.md").write_text(text)
    print(text)


class ChiveConfig(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.stage = pydra.REQUIRED  # agent | judge | top20 | concluder | report
        self.eval = "readonly"  # eval name: a directory under `out`
        self.ids = pydra.REQUIRED
        self.hard = ""  # ids of the hard stratum (report)
        self.incidents = "/workspace/agam/chive_bank/incidents"
        self.arms: list[str] = ["transcript"]
        self.draws = 1
        self.experiments = 0
        self.target = "https://safety-sahan-2-agam--chive-qwen-vllm-serve.modal.run"
        self.concurrency = 16
        self.max_calls = 40
        self.output_budget = 40_000
        self.max_tokens = 16_000
        self.thinking_budget = 6_000  # extended-thinking tokens per (unforced) agent turn
        self.key = "judge"  # report: judge | concluder_judge
        self.out = pydra.REQUIRED

    def finalize(self) -> None:
        self.ids, self.incidents, self.out = Path(self.ids), Path(self.incidents), Path(self.out)
        self.hard = Path(self.hard) if self.hard else None
        self.arm = list(self.arms)
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}")
        self.out.mkdir(parents=True, exist_ok=True)


STAGES = {"agent": stage_agent, "judge": stage_judge, "top20": stage_top20,
          "concluder": stage_concluder, "report": stage_report}


@pydra.main(ChiveConfig)
def main(config: ChiveConfig) -> None:
    STAGES[config.stage](config)


if __name__ == "__main__":
    main()
