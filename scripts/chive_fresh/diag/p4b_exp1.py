"""Phase 4b: what do agents test with their ONE budgeted counterfactual experiment, and do they update correctly?

Data: harness_runs/<b>_exp1n30/<arm>/<pid>.d{0,1}.json (budget 1, n=30 samples per experiment), baselines exp1/baselines.json.
Per episode a Sonnet labeller sees the CHIVE truth (points P1.., refuted rivals R1.., inert edits), the experiment
(hypothesis_tested + edit) and the agent's final hypothesis and returns:
  target       : "Pi" | "Rj" | "INERT" (the edit is essentially one CHIVE found inert) | "NOVEL"
  final_keeps  : does the final hypothesis assert the mechanism the experiment tested? yes | partial | no
Measured: delta = experiment rate - baseline rate (executor re-measured, n=30); mover = |delta| >= 0.30.
Derived: correct update = (mover and keeps) or (not mover and not keeps).
Out: diag/exp1/<b>/<arm>/<pid>.d<k>.json; summary via --report.
"""
import json, os, sys, pathlib, random, collections, concurrent.futures as cf
C = pathlib.Path("/workspace/agam/chive_scale")
ARMS = ["transcript", "olens_rl_B", "mlnla_B", "jlens_B"]
MODEL = "claude-sonnet-5"
SYS = """You label a counterfactual experiment an investigator ran on a language model, against the ground truth.
You get the behaviour, ground truth (TRUE points P1.., REFUTED rivals R1.., and INERT edits that were measured to have no
effect), the investigator's experiment (the hypothesis it tested and the edit), and the investigator's FINAL hypothesis.
Return JSON {"target": "Pi|Rj|INERT|NOVEL", "final_keeps": "yes|partial|no", "why": "<15 words"}.
target: which ground-truth item the experiment's tested mechanism corresponds to (best single match); INERT if the edit
is essentially one of the inert edits; NOVEL if none. final_keeps: whether the FINAL hypothesis asserts the mechanism the
experiment tested as a driver."""

def main(b):
    sys.path.insert(0, str(C)); import anthropic_oauth; anthropic_oauth.install(); import anthropic
    cli = anthropic.Anthropic(max_retries=6)
    GT = {r["prompt_id"]: r for r in map(json.loads, open(C / "datasets" / b / "data/ground_truth.jsonl"))}
    INC = {r["prompt_id"]: r for r in map(json.loads, open(C / "datasets" / b / "data/incidents.jsonl"))}
    def ask(u):
        for _ in range(4):
            try:
                m = cli.messages.create(model=MODEL, max_tokens=400, system=SYS, messages=[{"role": "user", "content": u}])
                t = "".join(x.text for x in m.content if x.type == "text"); return json.loads(t[t.find("{"): t.rfind("}") + 1])
            except anthropic.AuthenticationError: raise SystemExit("401")
            except Exception: pass
    def one(f):
        arm, name = f.parent.name, f.name; pid = name.split(".d")[0]
        out = C / "diag" / "exp1" / b / arm / name
        if out.exists(): return
        r = json.loads(f.read_text()); ex = r.get("experiments") or []
        if not ex or pid not in GT: return
        e = ex[0]; g = GT[pid]
        pts = json.loads((C / "points" / b / "points" / f"{pid}.json").read_text())["points"] if (C / "points" / b / "points" / f"{pid}.json").exists() else [t["cause"] for t in g["targets"]]
        rv = [x["cause"] for x in g["ruled_out"] if x.get("kind") == "refuted" and x.get("cause")]
        inert = [x.get("edit") or x.get("cause") for x in g["ruled_out"] if x.get("kind") == "inert"]
        u = (f"BEHAVIOUR: {INC[pid]['behaviour']}\n\nTRUE POINTS:\n" + "\n".join(f"P{i+1}: {p}" for i, p in enumerate(pts))
             + "\n\nREFUTED RIVALS:\n" + ("\n".join(f"R{j+1}: {x}" for j, x in enumerate(rv)) or "(none)")
             + "\n\nINERT EDITS:\n" + ("\n".join(f"- {x}" for x in inert if x) or "(none)")
             + f"\n\nEXPERIMENT hypothesis tested: {e.get('hypothesis_tested')}\nEdit: {json.dumps(e.get('turns'))[:2500]}"
             + f"\n\nFINAL HYPOTHESIS: {(r.get('result') or {}).get('hypothesis')}")
        j = ask(u)
        if j:
            out.parent.mkdir(parents=True, exist_ok=True)
            j.update(rate=e.get("rate"), k=e.get("k"), n=e.get("n"), score=(r.get("judge") or {}).get("score"),
                     conc=(r.get("concluder_judge") or {}).get("score") if isinstance(r.get("concluder_judge"), dict) else None,
                     committed=bool(r.get("committed_text_hypothesis")))
            out.write_text(json.dumps(j))
    fs = [f for a in ARMS for f in (C / "harness_runs" / f"{b}_exp1n30" / a).glob("*.json")]
    with cf.ThreadPoolExecutor(12) as ex: list(ex.map(one, fs))
    print(f"EXP1 {b} DONE {len(fs)}", flush=True)

def report():
    sys.path.insert(0, str(C / "report")); from common import boot
    base = json.load(open(C / "exp1" / "baselines.json"))
    M = collections.defaultdict(dict)  # (b,pid) -> arm -> metrics averaged over draws
    for b in ["fresh1", "fresh2", "fresh3", "fresh4"]:
        for a in ARMS:
            per = collections.defaultdict(list)
            for f in (C / "diag" / "exp1" / b / a).glob("*.json"):
                pid = f.name.split(".d")[0]; j = json.loads(f.read_text())
                if pid not in base or j.get("rate") in (None, ""): continue
                d = float(j["rate"]) - base[pid]["rate"]; mover = abs(d) >= 0.30
                t = str(j.get("target", "")); keeps = j.get("final_keeps") in ("yes", "partial")
                per[pid].append(dict(true_target=t.startswith("P"), rival_target=t.startswith("R"), inert_target=t == "INERT",
                                     novel=t == "NOVEL", mover=mover, correct_update=(mover and keeps) or (not mover and not keeps),
                                     keeps_nonmover=(not mover) and keeps, passed=(j.get("score") or 0) >= 7,
                                     true_mover=t.startswith("P") and mover))
            for pid, xs in per.items():
                M[(b, pid)][a] = {k: sum(x[k] for x in xs) / len(xs) for k in xs[0]}
    keys = [k for k, v in M.items() if all(a in v for a in ARMS)]
    print(f"n incidents with all 4 arms: {len(keys)}")
    rows = []
    for met in ["true_target", "rival_target", "inert_target", "novel", "mover", "true_mover", "correct_update", "keeps_nonmover", "passed"]:
        line = f"{met:15s} transcript={sum(M[k]['transcript'][met] for k in keys)/len(keys):.3f}"
        for a, ref in [("olens_rl_B", "transcript"), ("mlnla_B", "transcript"), ("jlens_B", "transcript"), ("olens_rl_B", "mlnla_B")]:
            v, r, d, lo, hi = boot([M[k][a][met] for k in keys], [M[k][ref][met] for k in keys])
            line += f" | {a}-{ref[:5]} {d:+.3f}[{lo:+.3f},{hi:+.3f}]"; rows.append(dict(metric=met, arm=a, ref=ref, n=len(keys), arm_mean=v, ref_mean=r, diff=d, lo=lo, hi=hi))
        print(line)
    sys.path.insert(0, str(C / "analysis" / "diag")); from dcommon import write_rows; write_rows("p4b_exp1.csv", rows)

if __name__ == "__main__":
    report() if sys.argv[1] == "--report" else main(sys.argv[1])
