"""Budget-3 ("experiments decide") dev runs: label EVERY experiment's target vs ground truth and measure what readers add.

Records: harness_runs/fresh{1,2}/{transcript_rx3,olens_rx3,mlnla_rx3}/<pid>.d{0,1}.json (3 experiments, n=30 each).
Per experiment (Sonnet, same SYS as p4b_exp1 but without final_keeps): target Pi | Rj | INERT | NOVEL.
mover = |rate - baseline| >= 0.30 (baseline exp1/baselines.json).
Metrics per episode: n experiments; share testing a true point; any experiment on a true point that moved; experiments 2-3
(the 'alternative' slots) testing a true point / a refuted rival / novel; pass.
Out: diag/rx3/<b>/<arm>/<pid>.d<k>.json ; data/p4b_rx3.csv
"""
import json, sys, collections, concurrent.futures as cf
sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from dcommon import *
ARMS = ["transcript_rx3", "olens_rx3", "mlnla_rx3"]
SYS = """You label counterfactual experiments an investigator ran on a language model, against the ground truth. You get the
behaviour, TRUE points P1.., REFUTED rivals R1.., INERT edits (measured no effect), and a numbered list of experiments
(hypothesis tested + edit). For each experiment return the best single match of the mechanism it tests: "Pi", "Rj",
"INERT" (essentially one of the inert edits) or "NOVEL". Output JSON {"targets": {"E1": "...", ...}}."""

def label():
    sys.path.insert(0, str(C)); import anthropic_oauth; anthropic_oauth.install(); import anthropic
    cli = anthropic.Anthropic(max_retries=6)
    jobs = [(b, a, f) for b in ("fresh1", "fresh2") for a in ARMS for f in (C / "harness_runs" / b / a).glob("*.json")]
    def one(j):
        b, a, f = j; out = C / "diag" / "rx3" / b / a / f.name
        if out.exists(): return
        r = json.loads(f.read_text()); ex = r.get("experiments") or []
        if not ex: return
        pid = f.name.split(".d")[0]; g = gt(b)[pid]
        pf = C / "points" / b / "points" / f"{pid}.json"
        pts = json.loads(pf.read_text())["points"] if pf.exists() else [t["cause"] for t in g["targets"]]
        rv = [x["cause"] for x in g["ruled_out"] if x.get("kind") == "refuted" and x.get("cause")]
        inert = [x.get("edit") or x.get("cause") for x in g["ruled_out"] if x.get("kind") == "inert"]
        u = (f"BEHAVIOUR: {inc(b)[pid]['behaviour']}\n\nTRUE POINTS:\n" + "\n".join(f"P{i+1}: {p}" for i, p in enumerate(pts))
             + "\n\nREFUTED RIVALS:\n" + ("\n".join(f"R{j+1}: {x}" for j, x in enumerate(rv)) or "(none)")
             + "\n\nINERT EDITS:\n" + ("\n".join(f"- {x}" for x in inert if x) or "(none)") + "\n\nEXPERIMENTS:\n"
             + "\n".join(f"E{i+1}: tests: {e.get('hypothesis_tested')}\n    edit: {json.dumps(e.get('turns'))[:1500]}" for i, e in enumerate(ex)))
        for _ in range(3):
            try:
                m = cli.messages.create(model="claude-sonnet-5", max_tokens=4000, system=SYS, messages=[{"role": "user", "content": u}])
                t = "".join(x.text for x in m.content if x.type == "text"); js = json.loads(t[t.find("{"): t.rfind("}") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"targets": js.get("targets", {}), "rates": [e.get("rate") for e in ex],
                                           "score": (r.get("judge") or {}).get("score")})); return
            except anthropic.AuthenticationError: raise SystemExit("401")
            except Exception: pass
    with cf.ThreadPoolExecutor(10) as ex: list(ex.map(one, jobs))

def report():
    sys.path.insert(0, str(C / "report")); from common import boot
    base = json.load(open(C / "exp1" / "baselines.json"))
    M = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for b in ("fresh1", "fresh2"):
        for a in ARMS:
            for f in (C / "diag" / "rx3" / b / a).glob("*.json"):
                pid = f.name.split(".d")[0]; j = json.loads(f.read_text())
                if pid not in base: continue
                T = [str(j["targets"].get(f"E{i+1}", "")) for i in range(len(j["rates"]))]
                mv = [r is not None and abs(float(r) - base[pid]["rate"]) >= 0.30 for r in j["rates"]]
                m = M[(b, pid)][a]
                m["n_exp"].append(len(T)); m["true_share"].append(sum(t.startswith("P") for t in T) / len(T))
                m["true_mover_any"].append(float(any(t.startswith("P") and v for t, v in zip(T, mv))))
                m["mover_share"].append(sum(mv) / len(mv))
                alt = T[1:]
                if alt:
                    m["alt_true"].append(sum(t.startswith("P") for t in alt) / len(alt))
                    m["alt_rival"].append(sum(t.startswith("R") for t in alt) / len(alt))
                    m["alt_novel"].append(sum(t in ("NOVEL", "INERT") for t in alt) / len(alt))
                    m["alt_true_mover"].append(float(any(t.startswith("P") and v for t, v in zip(T[1:], mv[1:]))))
                m["pass"].append(float((j.get("score") or 0) >= 7))
    rows = []
    for met in ["n_exp", "true_share", "mover_share", "true_mover_any", "alt_true", "alt_rival", "alt_novel", "alt_true_mover", "pass"]:
        line = f"{met:15s}"
        for a, ref in [("olens_rx3", "transcript_rx3"), ("mlnla_rx3", "transcript_rx3"), ("olens_rx3", "mlnla_rx3")]:
            ks = [k for k in M if M[k][a].get(met) and M[k][ref].get(met)]
            v, r, d, lo, hi = boot([sum(M[k][a][met]) / len(M[k][a][met]) for k in ks], [sum(M[k][ref][met]) / len(M[k][ref][met]) for k in ks])
            rows.append(dict(metric=met, arm=a, ref=ref, n=len(ks), arm_mean=v, ref_mean=r, diff=d, lo=lo, hi=hi))
            line += f" | {a[:5]} {v:.3f} vs {ref[:5]} {r:.3f} {d:+.3f}[{lo:+.3f},{hi:+.3f}]"
        print(line)
    write_rows("p4b_rx3.csv", rows)

if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else label()
