"""Run the intervention arms: calibrate the judge, sample and judge every arm, write the results.

    P=scripts/weirdchat/run_interventions.py
    OPENROUTER_API_KEY=… python $P stage=calibrate judge_model=google/gemini-3.8-flash n_calib=300
    OPENROUTER_API_KEY=… python $P stage=run server='https://…-organism-' n=64
    python $P stage=report

Arms come from ``scripts/weirdchat/interventions_arms.json`` (written from the investigators'
``would_test_by`` fields; the first arm of each pattern is the unchanged prompt). Results land at
``<out_root>/interventions/<pattern_key>.json`` and the calibration at
``<out_root>/interventions/calibration.json``; ``report`` prints one table.
"""

import dataclasses
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from detective_joracle.tools.live import LiveClient  # noqa: E402
from detective_joracle.weirdchat import data as wd  # noqa: E402
from detective_joracle.weirdchat import interventions as wi  # noqa: E402

ARMS_FILE = Path(__file__).with_name("interventions_arms.json")


@dataclasses.dataclass
class Settings:
    stage: str = ""  # calibrate | run | rejudge | report (comma-separated, in order)
    server: str = ""
    target: str = ""
    out_root: str = "outputs/weirdchat"
    arms_file: str = str(ARMS_FILE)
    patterns: str = ""  # comma-separated keys; default = every pattern in the arms file
    n: int = 64  # samples per arm
    max_new: int = 1500  # body-located behaviors need the full reply (the study's run ~4k chars)
    judge_model: str = "google/gemini-3.8-flash"
    n_calib: int = 300  # study-labelled replies to score the judge on
    seed: int = 0
    force: bool = False
    rejudge_all: bool = False  # stage=rejudge: re-judge EVERY reply with judge_model (a judge swap)
    add: bool = (
        False  # stage=run: append arms missing from an existing result instead of skipping it
    )


def _rate(a: dict[str, Any]) -> str:
    """A rate for the table, or '—' when every judge call failed."""
    return f"{a['rate']:.2f}" if a.get("rate") is not None else "  — "


def _csv(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def parse_settings(argv: list[str]) -> Settings:
    cfg = Settings()
    fields = {f.name: f for f in dataclasses.fields(Settings)}
    for a in argv:
        key, sep, val = a.partition("=")
        if not sep or key not in fields:
            raise SystemExit(f"bad argument {a!r}; options: {', '.join(fields)}")
        kind = type(fields[key].default)
        if kind is bool:
            setattr(cfg, key, val.lower() in ("true", "1", "yes"))
        elif kind is int:
            setattr(cfg, key, int(val))
        else:
            setattr(cfg, key, val)
    return cfg


def root(cfg: Settings) -> Path:
    return REPO / cfg.out_root


def _arms(cfg: Settings) -> dict[str, list[wi.Arm]]:
    blob = json.loads(Path(cfg.arms_file).read_text())
    keys = _csv(cfg.patterns)
    return {
        k: wi.arms_from_json(v)
        for k, v in blob.items()
        if not k.startswith("_") and (not keys or k in keys)
    }


def stage_calibrate(cfg: Settings) -> None:
    """Our judge on the study's own labelled replies (the cached API details), stratified by
    pattern and label; agreement and kappa are what every result is read against."""
    out = root(cfg) / "interventions" / "calibration.json"
    if out.exists() and not cfg.force:
        print(f"[calibrate] exists: {out}")
        return
    rng = random.Random(cfg.seed)
    rows: list[tuple[str, str, str, bool]] = []
    for p in sorted((root(cfg) / "patterns").glob("*.json")):
        pat = wd.load_pattern(p)
        detail = json.loads((root(cfg) / "data" / f"detail_{pat.pattern_key}.json").read_text())
        pool = [
            (
                pat.transcript_rubric,
                pat.prompt,
                str(s.get("response_text") or ""),
                bool(s["matched"]),
            )
            for s in detail.get("response_samples") or []
            if len(str(s.get("response_text") or "")) > 100
        ]
        yes = [r for r in pool if r[3]]
        no = [r for r in pool if not r[3]]
        per = max(1, cfg.n_calib // (2 * 21))
        rows += rng.sample(yes, min(per, len(yes))) + rng.sample(no, min(per, len(no)))
    rng.shuffle(rows)
    rows = rows[: cfg.n_calib]
    print(f"[calibrate] {len(rows)} labelled replies -> {cfg.judge_model}", flush=True)
    res = wi.calibrate(rows, model=cfg.judge_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(wi.dumps(res))
    print(f"[calibrate] {res}", flush=True)


def _add_arms(cfg: Settings, key: str, arms: list[wi.Arm], out: Path, client: LiveClient) -> None:
    """Sample and judge only the arms the existing result lacks; append them, keeping the
    baseline and every earlier arm as they are, and recompute Δ and p against the baseline."""
    res = json.loads(out.read_text())
    have = {a["arm"]["name"] for a in res["arms"]}
    todo = [a for a in arms if a.name not in have]
    if not todo:
        print(f"[run] {key}: nothing to add")
        return
    pat = wd.load_pattern(root(cfg) / "patterns" / f"{key}.json")
    print(f"[run] {key}: adding {[a.name for a in todo]} x {cfg.n}", flush=True)
    base = res["arms"][0]
    for i, arm in enumerate(todo):
        pairs = wi.sample(
            client, arm, n=cfg.n, max_new=cfg.max_new, seed=cfg.seed * 1000 + len(res["arms"]) + i
        )
        replies = [t for t, _ in pairs]
        verdicts, expl = wi.judge(
            pat.transcript_rubric, [(arm.prompt, r) for r in replies], model=res["judge_model"]
        )
        r = wi.ArmResult(arm, replies, verdicts, expl, [tr for _, tr in pairs], cfg.max_new)
        blob = r.to_json()
        blob["delta_vs_baseline"] = (
            round(r.rate - base["k"] / base["n"], 4) if r.n and base["n"] else None
        )
        blob["fisher_p_vs_baseline"] = round(wi.fisher_two_sided(r.k, r.n, base["k"], base["n"]), 5)
        res["arms"].append(blob)
        print(
            f"[run]   {arm.name:28s} {r.k:3d}/{r.n:<3d} = {_rate(blob)} "
            f"[{blob['ci95'][0]:.2f},{blob['ci95'][1]:.2f}]  Δ={blob['delta_vs_baseline']:+.2f} "
            f"p={blob['fisher_p_vs_baseline']}",
            flush=True,
        )
    out.write_text(wi.dumps(res))


def stage_run(cfg: Settings) -> None:
    if not cfg.server:
        raise SystemExit("server= is required")
    client = LiveClient(cfg.server, target=cfg.target)
    out_dir = root(cfg) / "interventions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, arms in _arms(cfg).items():
        out = out_dir / f"{key}.json"
        if out.exists() and cfg.add and not cfg.force:
            _add_arms(cfg, key, arms, out, client)
            continue
        if out.exists() and not cfg.force:
            print(f"[run] exists: {key}")
            continue
        pat = wd.load_pattern(root(cfg) / "patterns" / f"{key}.json")
        print(f"[run] {key}: {len(arms)} arms x {cfg.n}", flush=True)
        res = wi.run_arms(
            pat,
            arms,
            client,
            n=cfg.n,
            judge_model=cfg.judge_model,
            seed=cfg.seed,
            max_new=cfg.max_new,
        )
        out.write_text(wi.dumps(res))
        for a in res["arms"]:
            print(
                f"[run]   {a['arm']['name']:28s} {a['k']:3d}/{a['n']:<3d} = {_rate(a)} "
                f"[{a['ci95'][0]:.2f},{a['ci95'][1]:.2f}]"
                + (
                    f"  Δ={a['delta_vs_baseline']:+.2f} p={a['fisher_p_vs_baseline']}"
                    if a["delta_vs_baseline"] is not None
                    else ""
                ),
                flush=True,
            )


def stage_rejudge(cfg: Settings) -> None:
    """Retry the judge on the replies whose verdict is missing, at low concurrency — or, with
    ``rejudge_all=true``, re-judge every reply with ``judge_model`` (a judge swap)."""
    d = root(cfg) / "interventions"
    for p in sorted(d.glob("*__*.json")):
        res = json.loads(p.read_text())
        missing = sum(a["judge_failures"] for a in res["arms"])
        if not missing and not cfg.rejudge_all:
            continue
        pat = wd.load_pattern(root(cfg) / "patterns" / f"{res['pattern_key']}.json")
        model = cfg.judge_model if cfg.rejudge_all else res["judge_model"]
        filled = wi.rejudge(res, pat.transcript_rubric, model=model, everything=cfg.rejudge_all)
        p.write_text(wi.dumps(res))
        what = (
            f"re-judged {filled} replies with {model}"
            if cfg.rejudge_all
            else f"filled {filled}/{missing} missing verdicts"
        )
        print(f"[rejudge] {res['pattern_key']}: {what}", flush=True)
        for a in res["arms"]:
            delta = a.get("delta_vs_baseline")
            print(
                f"[rejudge]   {a['arm']['name']:28s} {a['k']:3d}/{a['n']:<3d} = {_rate(a)}"
                + (f"  Δ={delta:+.2f} p={a['fisher_p_vs_baseline']}" if delta is not None else ""),
                flush=True,
            )


def stage_report(cfg: Settings) -> None:
    d = root(cfg) / "interventions"
    cal = d / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        print(
            f"judge {c['model']}: agreement {c['agreement']} kappa {c['kappa']} "
            f"on {c['n']} study-labelled replies {c['confusion']}"
        )
    for p in sorted(d.glob("*__*.json")):
        r = json.loads(p.read_text())
        print(
            f"\n{r['pattern_key']}  (published {r['published_match_rate']:.2f}, n={r['n_per_arm']}/arm)"
        )
        for a in r["arms"]:
            d_ = a["delta_vs_baseline"]
            print(
                f"  {a['arm']['name']:28s} {_rate(a)} [{a['ci95'][0]:.2f},{a['ci95'][1]:.2f}]"
                + (
                    f"  Δ={d_:+.2f} p={a['fisher_p_vs_baseline']}"
                    if d_ is not None
                    else "  (baseline)"
                )
                + (f"  judge-fail {a['judge_failures']}" if a["judge_failures"] else "")
            )


STAGES: dict[str, Any] = {
    "calibrate": stage_calibrate,
    "run": stage_run,
    "rejudge": stage_rejudge,
    "report": stage_report,
}


def main(argv: list[str]) -> None:
    cfg = parse_settings(argv)
    for s in _csv(cfg.stage):
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s!r} ({' | '.join(STAGES)})")
        STAGES[s](cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
