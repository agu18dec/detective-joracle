"""Build `method_cmp.json` for the CHIVE viewer's "Method comparison" tab from run records.

Per incident: the behaviour, its rate and the established cause; per arm, what the investigator
did (calls, search patterns, positions inspected), its answer and the judge's score and
rationale; and OLens's decisive readouts -- quoted phrases from its final evidence, each located
in the bank at a position the answer cites (unmatched quotes are dropped, never invented).

    python scripts/build_chive_compare.py --runs /workspace/agam/chive_runs_dj --evals pilot_r2 \\
        --bank /workspace/agam/chive_bank/s3d_rl600_all --out method_cmp.json
"""

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydra

from detective_joracle.chive.data import load_incident
from detective_joracle.chive.executor import Executor
from detective_joracle.chive.prompts import seed, system_prompt
from detective_joracle.chive.readers import Reader

LABEL = {
    "transcript": "transcript only",
    "olens_rl": "OLens (RL)",
    "olens_distill": "OLens (distill)",
    "nla": "NLA",
    "jlens": "J-lens",
    "olens_rl_shuffled": "shuffled OLens",
}
ORDER = ["transcript", "olens_rl", "olens_distill", "nla", "jlens", "olens_rl_shuffled"]
_POS = re.compile(r"\bt(\d+)\b")
_QUOTE = re.compile(r"[\"“]([^\"”]{18,240})[\"”]")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def bank_cells(bank: Path, pid: str) -> dict[int, dict[int, list[str]]]:
    for f in (bank / f"{pid}.json.gz", bank / f"{pid}.json"):
        if f.exists():
            with gzip.open(f, "rt") if f.suffix == ".gz" else f.open() as fh:
                rec = json.load(fh)
            out: dict[int, dict[int, list[str]]] = defaultdict(dict)
            for layer, per in rec["readouts"].items():
                for pos, samples in per.items():
                    out[int(pos)][int(layer)] = samples
            return out
    return {}


def what_it_did(r: dict[str, Any]) -> dict[str, Any]:
    ev = r.get("events", [])
    searches = [str(e["args"].get("pattern", ""))[:60] for e in ev if e["tool"] == "search"]
    inspected = sorted(
        {int(p) for e in ev if e["tool"] == "inspect" for p in e.get("positions", [])}
    )
    shown = sorted(
        {
            int(p)
            for e in ev
            if e["tool"] in ("search", "show_tokens")
            for p in e.get("positions", [])
        }
    )
    return {
        "n_calls": len(ev),
        "searches": searches[:12],
        "inspected": inspected,
        "seen": shown[:400],
        "experiments": [{k: x[k] for k in ("k", "n", "rate")} for x in r.get("experiments") or []],
    }


def decisive(
    r: dict[str, Any], cells: dict[int, dict[int, list[str]]], pieces: list[str], conversation: str
) -> list[dict[str, Any]]:
    """Quotes from the final answer found in a readout at a cited position -- and NOT in the
    conversation itself (a readout restating the text is not evidence of anything hidden)."""
    seen_text = norm(conversation)
    res = r.get("result") or {}
    text = f"{res.get('hypothesis', '')}\n{res.get('evidence', '')}"
    cited = sorted({int(p) for p in _POS.findall(text) if int(p) in cells})
    out: list[dict[str, Any]] = []
    for q in _QUOTE.findall(text):
        nq = norm(q)
        if nq in seen_text:
            continue
        hit = next(
            (
                (p, layer, s)
                for p in cited
                for layer, ss in sorted(cells[p].items())
                for s in ss
                if nq in norm(s)
            ),
            None,
        )
        if hit and len(out) < 6:
            p, layer, s = hit
            out.append(
                {
                    "pos": p,
                    "token": pieces[p] if p < len(pieces) else "",
                    "layer": layer,
                    "quote": q,
                    "sample": s[:600],
                }
            )
    return out


KIND = {
    "olens_rl": ("olens", list(range(20, 61, 4)), 4),
    "olens_distill": ("olens", list(range(20, 61, 4)), 4),
    "nla": ("nla", [42], 4),
    "jlens": ("jlens", list(range(63)), 1),
}


def episode(r: dict[str, Any], inc_path: Path) -> dict[str, Any]:
    """One full episode as the viewer shows it: what the agent was told, then every turn."""
    system, first = r.get("system_prompt"), r.get("seed_prompt")
    if system is None:  # older records: rebuilt with the harness's own prompt functions
        kind = KIND.get(r["arm"].replace("_shuffled", ""))
        reader = None if kind is None else Reader(r["arm"], kind[0], kind[1], kind[2], {}, "")
        system = system_prompt(reader, int(r.get("experiments_allowed") or 0), Executor.N)
        first = seed(load_incident(inc_path))
    return {
        "draw": r["draw"],
        "stopped_by": r["stopped_by"],
        "seconds": r.get("seconds"),
        "output_tokens": r.get("output_tokens"),
        "system": system,
        "seed": first,
        "turns": [
            {
                "role": t["role"],
                "content": t["content"],
                "tool_calls": [{"name": c["name"], "args": c["args"]} for c in t["tool_calls"]],
            }
            for t in r["turns"]
        ],
        "notebook": r.get("notebook", ""),
        "result": r.get("result"),
        "judge": r.get("judge"),
        "experiments": r.get("experiments") or [],
    }


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.runs = "/workspace/agam/chive_runs_dj"
        self.evals: list[str] = ["pilot_r2"]
        self.arms: list[str] = ["transcript", "olens_rl", "olens_distill", "nla", "jlens"]
        self.incidents = "/workspace/agam/chive_bank/incidents"
        self.bank = "/workspace/agam/chive_bank/s3d_rl600_all"
        self.out = pydra.REQUIRED  # method_cmp.json; episodes go to <out dir>/methods/<pid>.json


@pydra.main(Config)
def main(cfg: Config) -> None:
    by_pid: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ev in cfg.evals:
        for f in sorted(Path(cfg.runs, ev).glob("*/*.d*.json")):
            r = json.loads(f.read_text())
            r["draw"] = int(f.name.rsplit(".d", 1)[1].split(".")[0])  # from the file name
            by_pid[r["prompt_id"]][r["arm"]].append(r)
    cards, wins, losses = [], [], []
    rates: dict[str, list[float]] = defaultdict(list)
    for pid, arms in sorted(by_pid.items()):
        inc = json.loads(Path(cfg.incidents, f"{pid}.json").read_text())
        cells = bank_cells(Path(cfg.bank), pid)
        convo = "\n".join(m["content"] for m in inc["messages"]) + "\n" + inc["completion"]
        rows = []
        for arm in sorted(
            (a for a in arms if a in cfg.arms), key=lambda a: ORDER.index(a) if a in ORDER else 99
        ):
            eps = arms[arm]
            scores = [
                e["judge"]["score"] for e in eps if (e.get("judge") or {}).get("score") is not None
            ]
            passes = [s >= 7 for s in scores]
            if passes:
                rates[arm].append(sum(passes) / len(passes))
            best = max(eps, key=lambda e: (e.get("judge") or {}).get("score") or 0)
            j = best.get("judge") or {}
            rows.append(
                {
                    "arm": arm,
                    "label": LABEL.get(arm, arm),
                    "draws": len(eps),
                    "scores": scores,
                    "pass": sum(passes),
                    "did": what_it_did(best),
                    "answer": (best.get("result") or {}).get("hypothesis", ""),
                    "evidence": (best.get("result") or {}).get("evidence", ""),
                    "rival": (best.get("result") or {}).get("rival", ""),
                    "score": j.get("score"),
                    "rationale": ((j.get("replicates") or [{}])[0]).get("rationale", ""),
                    "decisive": (
                        decisive(best, cells, inc["pieces"], convo)
                        if arm.startswith("olens")
                        else []
                    ),
                }
            )
        mean = {
            r["arm"]: (sum(r["scores"]) / len(r["scores"]) if r["scores"] else None) for r in rows
        }
        gap = (
            None
            if mean.get("olens_rl") is None or mean.get("transcript") is None
            else mean["olens_rl"] - mean["transcript"]
        )
        truth = inc["ground_truth"]["targets"][0]["title"]
        card = {
            "prompt_id": pid,
            "behaviour": inc["behaviour"],
            "k": inc["k"],
            "n": inc["n"],
            "truth": truth,
            "response_start": next(
                (
                    i
                    for i, (t, role) in enumerate(inc["turn_of"])
                    if role == "assistant"
                    and t == max(x for x, y in inc["turn_of"] if y == "assistant")
                ),
                0,
            ),
            "gap": gap,
            "arms": rows,
        }
        cards.append(card)
        tr = next((r for r in rows if r["arm"] == "transcript"), None)
        ol = next((r for r in rows if r["arm"] == "olens_rl"), None)
        if tr and ol and tr["scores"] and ol["scores"]:
            t_pass, o_pass = tr["pass"] / len(tr["scores"]), ol["pass"] / len(ol["scores"])
            if o_pass > t_pass:
                wins.append({"prompt_id": pid, "truth": truth})
            elif o_pass < t_pass:
                losses.append({"prompt_id": pid, "truth": truth})
    cards.sort(key=lambda c: -(c["gap"] if c["gap"] is not None else -99))
    summary = {
        "evals": cfg.evals,
        "n_incidents": len(cards),
        "pass": {a: sum(v) / len(v) for a, v in rates.items() if v},
        "n": {a: len(v) for a, v in rates.items()},
        "wins": wins,
        "losses": losses,
    }
    Path(cfg.out).write_text(
        json.dumps({"summary": summary, "incidents": cards}, ensure_ascii=False)
    )
    ep_dir = Path(cfg.out).parent / "methods"
    ep_dir.mkdir(parents=True, exist_ok=True)
    for pid, arms in by_pid.items():
        inc_path = Path(cfg.incidents, f"{pid}.json")
        full = {
            a: sorted((episode(r, inc_path) for r in eps), key=lambda e: e["draw"])
            for a, eps in arms.items()
            if a in cfg.arms
        }
        (ep_dir / f"{pid}.json").write_text(json.dumps(full, ensure_ascii=False))
    print(
        f"{len(cards)} incidents -> {cfg.out}; OLens wins {len(wins)}, losses {len(losses)}; "
        f"pass { ({k: round(v, 3) for k, v in summary['pass'].items()}) }"
    )


if __name__ == "__main__":
    main()
