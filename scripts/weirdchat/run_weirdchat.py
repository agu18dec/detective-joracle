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
from detective_joracle.weirdchat import agreement as wagr  # noqa: E402
from detective_joracle.weirdchat import concise as wcon  # noqa: E402
from detective_joracle.weirdchat import data as wd  # noqa: E402
from detective_joracle.weirdchat import diagnostics as wdiag  # noqa: E402
from detective_joracle.weirdchat import explain as wex  # noqa: E402
from detective_joracle.weirdchat import flags as wflags  # noqa: E402
from detective_joracle.weirdchat import predictions as wpred  # noqa: E402
from detective_joracle.weirdchat import synth as wsyn  # noqa: E402


@dataclasses.dataclass
class Settings:
    """Every ``key=value`` option the driver accepts."""

    stage: str = ""  # comma-separated: data | diagnose | agent | synth | agreement | predictions | flags | annotate | grade | concise | site
    server: str = ""  # lens server endpoint template ("http://h:8000/{name}") or Modal prefix
    target: str = ""  # optional OpenAI-compatible chat server; empty = the lens server's chat
    nla: str = ""  # the NLA verbalizer URL (arm=nla reads layer 42 through it)
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
    arm: str = "olens"  # olens | jlens | nla (which lens readouts reads) | blackbox (no readouts)
    arm_b: str = "blackbox"  # the other arm in stage=agreement (arm vs arm_b)
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
    return LiveClient(cfg.server, target=cfg.target, nla=cfg.nla)


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


def stage_agreement(cfg: Settings) -> None:
    """One arm's mechanisms against another's (``arm`` vs ``arm_b``), per pattern, read blind."""
    runs = root(cfg) / "runs"
    rows: list[dict[str, Any]] = []
    for p in pattern_paths(cfg):
        lens_p = _run_path(runs, p.stem, cfg.auditor, 0, cfg.arm)
        bb_p = _run_path(runs, p.stem, cfg.auditor, 0, cfg.arm_b)
        if not (_done(lens_p) and _done(bb_p)):
            continue
        lens = json.loads(lens_p.read_text())
        bb = json.loads(bb_p.read_text())
        rows.append(
            wagr.compare(
                lens["pattern"],
                lens["record"]["result"]["mechanisms"],
                bb["record"]["result"]["mechanisms"],
                model=cfg.aux_model,
            )
        )
        r = rows[-1]
        print(
            f"[agreement] {p.stem}: top_match={r.get('top_match')} "
            f"lens {r.get('lens_with_counterpart')}/{r.get('n_lens')} matched, "
            f"blackbox {r.get('blackbox_with_counterpart')}/{r.get('n_blackbox')} matched",
            flush=True,
        )
    blob = {
        "summary": wagr.summarise(rows),
        "patterns": rows,
        "model": cfg.aux_model,
        "arm": cfg.arm,
        "arm_b": cfg.arm_b,
    }
    name = (
        "agreement.json"
        if (cfg.arm, cfg.arm_b) == ("olens", "blackbox")
        else f"agreement_{cfg.arm}_vs_{cfg.arm_b}.json"
    )
    (root(cfg) / name).write_text(wagr.dumps(blob))
    print(f"[agreement] {blob['summary']}", flush=True)


def stage_predictions(cfg: Settings) -> None:
    """Score ``arm``'s would_test_by predictions against the measured intervention arms."""
    runs = root(cfg) / "runs"
    rows: list[dict[str, Any]] = []
    for ip in sorted((root(cfg) / "interventions").glob("*__*.json")):
        inter = json.loads(ip.read_text())
        rp = _run_path(runs, ip.stem, cfg.auditor, 0, cfg.arm)
        if not _done(rp):
            continue
        run = json.loads(rp.read_text())
        rows.append(
            wpred.score(
                run["pattern"], run["record"]["result"]["mechanisms"], inter, model=cfg.aux_model
            )
        )
        r = rows[-1]
        print(
            f"[predictions] {cfg.arm} {ip.stem}: right {r.get('right')} wrong {r.get('wrong')} "
            f"not predicted {r.get('not_predicted')} of {r.get('n_arms')} arms",
            flush=True,
        )
    blob = {
        "summary": wpred.summarise(rows),
        "patterns": rows,
        "arm": cfg.arm,
        "model": cfg.aux_model,
    }
    (root(cfg) / f"predictions_{cfg.arm}.json").write_text(wpred.dumps(blob))
    print(f"[predictions] {cfg.arm}: {blob['summary']}", flush=True)


def stage_flags(cfg: Settings) -> None:
    """An LLM flagger over each pattern's study reads (diag): the cells worth attention, verified
    verbatim; written to flags/<key>.json as {read_id: {flags, …}} for the viewer."""
    out = root(cfg) / "flags"
    out.mkdir(parents=True, exist_ok=True)
    todo = [
        p
        for p in pattern_paths(cfg)
        if (root(cfg) / "diag" / p.name).exists() and (cfg.force or not (out / p.name).exists())
    ]
    print(f"[flags] {len(todo)} patterns -> {cfg.aux_model} ({cfg.workers} workers)", flush=True)
    if cfg.dry:
        return

    def one(path: Path) -> str:
        pat = wd.load_pattern(path)
        diag = json.loads((root(cfg) / "diag" / path.name).read_text())
        reads: dict[str, Any] = {}
        for r in diag["reads"]:
            rid = f"diag:{r['label']}:{r['sample_index']}"
            reads[rid] = wflags.flag_read(
                r["readout"],
                behavior=pat.behavior_name,
                prompt=pat.prompt,
                reply=r["conversation"]["completion"],
                flagged=r["label"] == "matched",
                model=cfg.aux_model,
            )
        (out / path.name).write_text(wflags.dumps({"pattern_key": pat.pattern_key, "reads": reads}))
        n = sum(len(v["flags"]) for v in reads.values())
        bad = sum(v["unverified"] for v in reads.values())
        fails = sum(v["failed_calls"] for v in reads.values())
        return f"{pat.pattern_key}: {n} flags kept, {bad} unverified quotes dropped, {fails} failed calls"

    _fan_out(cfg, todo, one, "flags")


def stage_annotate(cfg: Settings) -> None:
    """Gemini marks the cells that contrastively support the OLens investigator's mechanisms:
    flagged reply A vs clean reply A (the study reads), per mechanism, quotes verified in the read
    they are attributed to -> annotations/<key>.json."""
    out = root(cfg) / "annotations"
    out.mkdir(parents=True, exist_ok=True)
    concise_p = root(cfg) / "concise.json"
    short: dict[str, str] = json.loads(concise_p.read_text()) if concise_p.exists() else {}
    runs = root(cfg) / "runs"
    todo = [
        p
        for p in pattern_paths(cfg)
        if (root(cfg) / "diag" / p.name).exists()
        and _done(_run_path(runs, p.stem, cfg.auditor, 0, "olens"))
        and (cfg.force or not (out / p.name).exists())
    ]
    print(f"[annotate] {len(todo)} patterns -> {cfg.aux_model} ({cfg.workers} workers)", flush=True)
    if cfg.dry:
        return

    def one(path: Path) -> str:
        pat = wd.load_pattern(path)
        diag = json.loads((root(cfg) / "diag" / path.name).read_text())
        run = json.loads(_run_path(runs, path.stem, cfg.auditor, 0, "olens").read_text())
        mechs = [
            {**m, "short": short.get(f"{pat.pattern_key}#olens#{i}", "")}
            for i, m in enumerate(run["record"]["result"]["mechanisms"])
        ]
        by = {r["label"]: r for r in diag["reads"]}
        if "matched" not in by or "unmatched" not in by:
            return f"{pat.pattern_key}: no flagged/clean pair in diag (skipped)"
        fork = None
        fk = diag.get("fork") or {}
        if fk.get("prefix_chars") is not None:  # first reply token past the shared prefix, approx.
            done = 0
            for p_ in sorted(by["matched"]["readout"]["tokens"], key=int):
                if by["matched"]["readout"]["tags"][p_]["region"] != "reply":
                    continue
                done += len(by["matched"]["readout"]["tokens"][p_])
                if done > int(fk["prefix_chars"]):
                    fork = int(p_)
                    break
        res = wflags.annotate_contrast(
            by["matched"]["readout"],
            by["unmatched"]["readout"],
            behavior=pat.behavior_name,
            prompt=pat.prompt,
            mechanisms=mechs,
            fork_position=fork,
            model=cfg.aux_model,
        )
        res.update(
            pattern_key=pat.pattern_key,
            flagged_read=f"diag:matched:{by['matched']['sample_index']}",
            clean_read=f"diag:unmatched:{by['unmatched']['sample_index']}",
            arm="olens",
        )
        (out / path.name).write_text(wflags.dumps(res))
        return (
            f"{pat.pattern_key}: {len(res['annotations'])} annotations kept "
            f"({res['unverified']} unverified dropped, {res['failed_calls']} failed calls)"
        )

    _fan_out(cfg, todo, one, "annotate")


def stage_grade(cfg: Settings) -> None:
    """Grade every annotation as contrastive (one side says something the other does not) or
    shared propensity; in place in annotations/<key>.json."""
    runs = root(cfg) / "runs"
    concise_p = root(cfg) / "concise.json"
    short: dict[str, str] = json.loads(concise_p.read_text()) if concise_p.exists() else {}
    todo = [
        p
        for p in sorted((root(cfg) / "annotations").glob("*.json"))
        if cfg.force
        or any("contrastive" not in a for a in json.loads(p.read_text())["annotations"])
    ]
    print(f"[grade] {len(todo)} annotation files -> {cfg.aux_model}", flush=True)
    if cfg.dry:
        return

    def one(path: Path) -> str:
        ann = json.loads(path.read_text())
        diag = json.loads((root(cfg) / "diag" / path.name).read_text())
        by = {r["label"]: r for r in diag["reads"]}
        run = json.loads(
            _run_path(runs, path.stem, cfg.auditor, 0, ann.get("arm", "olens")).read_text()
        )
        mechs = [
            {**m, "short": short.get(f"{path.stem}#{ann.get('arm', 'olens')}#{i}", "")}
            for i, m in enumerate(run["record"]["result"]["mechanisms"])
        ]
        n = wflags.grade_contrast(
            ann["annotations"],
            by["matched"]["readout"],
            by["unmatched"]["readout"],
            mechs,
            model=cfg.aux_model,
        )
        ann["n_contrastive"] = n
        path.write_text(wflags.dumps(ann))
        return f"{path.stem}: {n}/{len(ann['annotations'])} contrastive"

    _fan_out(cfg, todo, one, "grade")


def stage_concise(cfg: Settings) -> None:
    """One plain sentence per mechanism of every run (all arms) -> concise.json for the page.
    Resumable: rows already in the file are kept."""
    out = root(cfg) / "concise.json"
    cache: dict[str, str] = json.loads(out.read_text()) if out.exists() else {}
    records = [json.loads(p.read_text()) for p in sorted((root(cfg) / "runs").glob("*/*/*.json"))]
    rows = [r for r in wcon.keys_for(records) if cfg.force or r[0] not in cache]
    print(
        f"[concise] {len(rows)} mechanisms to rewrite ({len(cache)} cached) -> {cfg.aux_model}",
        flush=True,
    )
    if cfg.dry or not rows:
        return
    cache.update(wcon.rewrite(rows, model=cfg.aux_model))
    out.write_text(wcon.dumps(cache))
    print(f"[concise] {len(cache)} short mechanisms written", flush=True)


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
    "agreement": stage_agreement,
    "predictions": stage_predictions,
    "flags": stage_flags,
    "annotate": stage_annotate,
    "grade": stage_grade,
    "concise": stage_concise,
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
