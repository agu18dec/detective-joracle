"""The WeirdChat driver: pick patterns, read them with the lens, explain them, cluster, render.

    P=scripts/weirdchat/run_weirdchat.py
    python $P stage=data per_behavior=3 min_rate=0.15
    python $P stage=diagnose server='https://<ws>--auditbench-organism-organism-'
    OPENROUTER_API_KEY=... python $P stage=agent server='https://…-organism-' \\
        auditor=anthropic/claude-opus-5 seeds=1
    OPENROUTER_API_KEY=... python $P stage=synth,site

Every option is ``key=value`` (see ``Settings``; booleans accept true/false). Stages run in the
order given and each one resumes: a pattern that already has its file is skipped unless
``force=true``. Outputs land under ``out_root``:

    patterns/<key>.json   diag/<key>.json   runs/<key>/<auditor>/seed_<k>.json   synth.json

There is no judge and no ground truth in this pipeline: it collects hypotheses about why the
model behaves as WeirdChat found it does, and the site says so on every page.
"""

import dataclasses
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))  # so the scripts also run from a bare checkout
from detective_joracle.agent.backends import make_backend_factory  # noqa: E402
from detective_joracle.agent.loop import Budget  # noqa: E402
from detective_joracle.tools.live import READ_LAYERS, LiveClient, record_extras  # noqa: E402
from detective_joracle.weirdchat import data as wd  # noqa: E402
from detective_joracle.weirdchat import diagnostics as wdiag  # noqa: E402
from detective_joracle.weirdchat import explain as wex  # noqa: E402
from detective_joracle.weirdchat import synth as wsyn  # noqa: E402


@dataclasses.dataclass
class Settings:
    """Every ``key=value`` option the driver accepts."""

    stage: str = ""  # comma-separated: data | diagnose | agent | synth | site
    server: str = ""  # lens server endpoint template ("http://h:8000/{name}") or Modal prefix
    target: str = ""  # optional OpenAI-compatible chat server; empty = the lens server's chat
    out_root: str = "outputs/weirdchat"
    model: str = wd.MODEL  # the WeirdChat subject model
    behaviors: str = ""  # comma-separated behavior ids (default: all of the model's)
    per_behavior: int = 3  # patterns kept per behavior, highest WeirdChat Elo first
    min_rate: float = 0.15  # published match rate a pattern must reach to be worth explaining
    patterns: str = ""  # comma-separated pattern keys, overriding the selection
    n_side: int = 2  # study rollouts per side handed to the agent
    diag_side: int = 1  # study rollouts per side read in the diagnose stage
    layers: str = ",".join(str(e) for e in READ_LAYERS)
    k: int = 1  # lens samples per cell
    auditor: str = "anthropic/claude-opus-5"
    arm: str = "olens"  # olens (the lens arm) | blackbox (same brief and chat tools, no readouts)
    aux_model: str = "google/gemini-3.8-flash"  # clustering and readout triage
    backend: str = "openrouter"  # openrouter | openai
    seeds: int = 1
    budget_tokens: int = 25_000
    max_calls: int = 400
    # output cap per turn; the finish() call carries every mechanism's evidence in one JSON
    # object, and at 4000 a reasoning model's finish was cut mid-argument (0 mechanisms parsed)
    max_tokens: int = 16_000
    select: str = "all"  # readout presentation (presentation.select)
    workers: int = 4  # parallel patterns (= server containers in use)
    limit: int = 0  # cap the patterns this invocation processes (0 = all)
    force: bool = False  # redo work that already has a file
    dry: bool = False  # list the work, do nothing


def _csv(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def parse_settings(argv: list[str]) -> Settings:
    """``key=value`` arguments -> :class:`Settings`, coerced by the field's default type."""
    cfg = Settings()
    fields = {f.name: f for f in dataclasses.fields(Settings)}
    for a in argv:
        key, sep, val = a.partition("=")
        if not sep or key not in fields:
            raise SystemExit(f"bad argument {a!r}; options: {', '.join(fields)}")
        kind = type(fields[key].default)
        try:
            if kind is bool:
                if val.lower() not in ("true", "false", "1", "0", "yes", "no"):
                    raise ValueError(val)
                setattr(cfg, key, val.lower() in ("true", "1", "yes"))
            elif kind is int:
                setattr(cfg, key, int(val))
            elif kind is float:
                setattr(cfg, key, float(val))
            else:
                setattr(cfg, key, val)
        except ValueError:
            raise SystemExit(f"{key}={val!r}: expected {kind.__name__}") from None
    return cfg


def root(cfg: Settings) -> Path:
    return REPO / cfg.out_root


def pattern_paths(cfg: Settings) -> list[Path]:
    """The selected patterns on disk, in behavior order."""
    keys = _csv(cfg.patterns)
    paths = sorted((root(cfg) / "patterns").glob("*.json"))
    if keys:
        paths = [p for p in paths if p.stem in keys]
    return paths[: cfg.limit] if cfg.limit else paths


def _client(cfg: Settings) -> LiveClient:
    if not cfg.server:
        raise SystemExit("server= (the lens server endpoint template or Modal prefix) is required")
    return LiveClient(cfg.server, target=cfg.target)


# ---------------------------------------------------------------- stages
def stage_data(cfg: Settings) -> None:
    """Select the patterns and cache each one's prompt, rubric and judged rollouts."""
    cache = root(cfg) / "data"
    rows = wd.fetch_patterns(cache, model=cfg.model)
    picked = wd.select_patterns(
        rows,
        per_behavior=cfg.per_behavior,
        min_rate=cfg.min_rate,
        behaviors=_csv(cfg.behaviors) or None,
    )
    if cfg.patterns:
        want = set(_csv(cfg.patterns))
        picked = [r for r in rows if wd.pattern_key(r) in want]
    if cfg.limit:
        picked = picked[: cfg.limit]
    print(
        f"[data] {len(rows)} {cfg.model} patterns on WeirdChat, "
        f"{len(picked)} selected (per_behavior={cfg.per_behavior}, min_rate={cfg.min_rate})",
        flush=True,
    )
    if cfg.dry:
        for r in picked:
            print("  ", wd.pattern_key(r), round(r["metrics"]["match_rate"], 3))
        return
    out = root(cfg) / "patterns"
    out.mkdir(parents=True, exist_ok=True)
    kept: list[wd.Pattern] = []
    for r in picked:
        p = out / f"{wd.pattern_key(r)}.json"
        if p.exists() and not cfg.force:
            kept.append(wd.load_pattern(p))
            continue
        pat = wd.fetch_pattern(r, cache, n_side=max(cfg.n_side, cfg.diag_side))
        if not pat.matched or not pat.unmatched:
            print(f"[data] {pat.pattern_key}: no contrast pair (skipped)", flush=True)
            continue
        p.write_text(json.dumps(pat.to_json(), ensure_ascii=False, indent=1))
        kept.append(pat)
    stats = wd.token_stats(kept)
    (root(cfg) / "data_stats.json").write_text(json.dumps(stats, indent=1))
    by_behavior: dict[str, int] = {}
    for pat in kept:
        by_behavior[pat.behavior_id] = by_behavior.get(pat.behavior_id, 0) + 1
    print(f"[data] {len(kept)} patterns written; {stats}", flush=True)
    for bid in sorted(by_behavior):
        print(f"[data]   {bid}: {by_behavior[bid]}", flush=True)


def stage_diagnose(cfg: Settings) -> None:
    """Read both sides of each pattern's contrast with the lens, before any agent sees it."""
    client_layers = [int(x) for x in _csv(cfg.layers)]
    out = root(cfg) / "diag"
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pattern_paths(cfg) if cfg.force or not (out / p.name).exists()]
    print(f"[diagnose] {len(todo)} patterns ({cfg.workers} workers)", flush=True)
    if cfg.dry:
        for p in todo:
            print("  ", p.stem)
        return

    def one(path: Path) -> str:
        pat = wd.load_pattern(path)
        client = _client(cfg)
        blob = wdiag.diagnose(
            pat, client, layers=client_layers, k=cfg.k, n_side=cfg.diag_side, seed=0
        )
        blob["extras"] = record_extras(client)
        (out / path.name).write_text(json.dumps(blob, ensure_ascii=False, indent=1))
        cells = sum(
            len(per) for r in blob["reads"] for per in (r["readout"]["readouts"] or {}).values()
        )
        return f"{pat.pattern_key}: {len(blob['reads'])} reads, {cells} cells"

    _fan_out(cfg, todo, one, "diagnose")


def stage_agent(cfg: Settings) -> None:
    """One explain run per (pattern, seed): the investigator with chat plus the lens."""
    factory = make_backend_factory(cfg.backend, max_tokens=cfg.max_tokens)
    layers = [int(x) for x in _csv(cfg.layers)]
    budget = Budget(output_tokens=cfg.budget_tokens, max_calls=cfg.max_calls)
    runs = root(cfg) / "runs"
    todo = [
        (p, seed)
        for p in pattern_paths(cfg)
        for seed in range(cfg.seeds)
        if cfg.force or not _done(_run_path(runs, p.stem, cfg.auditor, seed, cfg.arm))
    ]
    print(
        f"[agent] {len(todo)} runs ({cfg.workers} workers, auditor {cfg.auditor}, arm {cfg.arm})",
        flush=True,
    )
    if cfg.dry:
        for p, seed in todo:
            print("  ", p.stem, seed)
        return

    def one(job: tuple[Path, int]) -> str:
        path, seed = job
        pat = wd.load_pattern(path)
        client = _client(cfg)
        rec, _tools = wex.run_explain_agent(
            pat,
            client,
            factory(cfg.auditor),
            auditor=cfg.auditor,
            seed=seed,
            arm=cfg.arm,
            layers=layers,
            budget=budget,
            aux_model=cfg.aux_model,
            select=cfg.select,
            n_side=cfg.n_side,
        )
        out = _run_path(runs, path.stem, cfg.auditor, seed, cfg.arm)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(wex.dumps(rec, pat, record_extras(client)))
        n_mech = len((rec.result or {}).get("mechanisms") or [])
        return (
            f"{path.stem}/s{seed}: {rec.stopped_by}, {len(rec.tool_log)} calls, "
            f"{rec.output_tokens} out-tokens, {rec.cells_served} cells, {n_mech} mechanisms"
        )

    _fan_out(cfg, todo, one, "agent")


def stage_synth(cfg: Settings) -> None:
    """Cluster the mechanisms across every finished run."""
    records = [json.loads(p.read_text()) for p in sorted((root(cfg) / "runs").glob("*/*/*.json"))]
    records = [r for r in records if (r.get("record") or {}).get("arm", "olens") == cfg.arm]
    items = wsyn.collect(records)
    print(f"[synth] {len(items)} mechanisms from {len(records)} {cfg.arm} runs", flush=True)
    if cfg.dry or not items:
        return
    blob = wsyn.cluster(items, model=cfg.aux_model)
    blob["n_records"] = len(records)
    name = "synth.json" if cfg.arm == "olens" else f"synth_{cfg.arm}.json"
    (root(cfg) / name).write_text(wsyn.dumps(blob))
    for c in blob["clusters"]:
        print(f"[synth]   {len(c['members']):2d}  {c['name']}", flush=True)


def stage_site(cfg: Settings) -> None:
    """Render the viewer."""
    sys.path.insert(0, str(REPO / "scripts" / "weirdchat"))
    from build_site import build_site  # noqa: PLC0415

    path = build_site(root(cfg), root(cfg) / "site")
    print(f"[site] {path}", flush=True)


# ---------------------------------------------------------------- helpers
def _run_path(runs: Path, key: str, auditor: str, seed: int, arm: str = "olens") -> Path:
    """The lens arm keeps the original layout; another arm gets its own directory beside it."""
    who = auditor.replace("/", "_") + ("" if arm == "olens" else f"__{arm}")
    return runs / key / who / f"seed_{seed}.json"


def _done(p: Path) -> bool:
    """A run counts as done only if the agent finished WITH mechanisms; one that died on a
    provider error, or ended with an empty result, is kept on disk for inspection but re-run
    next pass."""
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text()).get("record") or {}
    except (OSError, json.JSONDecodeError):
        return False
    if str(rec.get("stopped_by", "")).startswith("error"):
        return False
    return bool((rec.get("result") or {}).get("mechanisms"))


def _fan_out(cfg: Settings, todo: list[Any], one: Any, tag: str) -> None:
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = {ex.submit(one, t): t for t in todo}
        for f in as_completed(futs):
            try:
                print(f"[{tag}]", f.result(), flush=True)
            except Exception as e:  # one failed pattern must not take the batch down
                print(f"[{tag}] {futs[f]}: FAILED {type(e).__name__}: {e}", flush=True)


STAGES = {
    "data": stage_data,
    "diagnose": stage_diagnose,
    "agent": stage_agent,
    "synth": stage_synth,
    "site": stage_site,
}


def main(argv: list[str]) -> None:
    cfg = parse_settings(argv)
    stages = _csv(cfg.stage)
    if not stages:
        raise SystemExit(f"stage= is required ({' | '.join(STAGES)})")
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s!r} ({' | '.join(STAGES)})")
    for s in stages:
        STAGES[s](cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
