"""Phase 3a: committed (pre-readout) -> final flips per reader arm, and an LLM-coded taxonomy of WHY they flip.

Pass/fail: same 1-10 judge, mean of 3 >= 7 (committed scores: diag/commitjudge; final: the record's judge).
Transition matrix per arm over d0-d4. Then every flip episode (sampled up to --per per arm x direction) is coded by Sonnet,
which sees the CHIVE judge reference, the committed and final hypotheses, the final `evidence` field (readouts the agent
cites) and the judge's rationale for the final:
  pass->fail: LURE (readouts confirm a refuted/inert rival) | ARGUED_AWAY (correct cause dropped on absence/timing/
              'represented elsewhere' argument) | DILUTION (correct cause kept but merged with readout side-stories, judge
              scores weaker part) | REFOCUS (swapped for an 'internal state' narrative GT does not support) | MISREAD |
              JUDGE_NOISE (final is about as good as committed) | OTHER
  fail->pass: REVEALED (readouts surfaced the true mechanism the commit missed) | RULED_OUT_RIVAL (commit asserted a rival,
              readouts made the agent drop it) | SHARPENED (commit near-miss made specific) | JUDGE_NOISE | OTHER
Out: diag/flips/<b>/<arm>/<pid>.d<k>.json ; data/p3a_transitions.csv, data/p3a_codes.csv
usage: python p3a_flips.py code [--per 150] | python p3a_flips.py report
"""
import json, sys, random, collections, pathlib, os, concurrent.futures as cf
sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from dcommon import *

ARMS = ["olens_rl_B", "mlnla_B"]
SYS = """You explain why an interpretability investigator's answer changed grade between its committed hypothesis (written from
the transcript only) and its final hypothesis (written after reading readouts of the model's activations). You see the
ground-truth reference (true causes, ruled-out causes), both hypotheses, the readout evidence the investigator cites, and the
grader's rationale for the final. Pick ONE code.
If the grade went PASS -> FAIL: LURE (readouts led it to a refuted or inert cause), ARGUED_AWAY (it dropped the correct
cause with an absence / timing / 'the concept is represented so it is not lost' argument), DILUTION (kept the correct cause
but merged it with readout-derived side-stories, so the grader scored a weaker or wrong part), REFOCUS (replaced the cause
with an internal-state narrative the ground truth does not support), MISREAD (misinterpreted readouts, e.g. an echo of the
visible text read as a belief), JUDGE_NOISE (final is essentially as good as the committed one), OTHER.
If FAIL -> PASS: REVEALED (readouts surfaced the true mechanism the committed hypothesis missed), RULED_OUT_RIVAL (the
committed hypothesis asserted a wrong cause and readouts made it drop it), SHARPENED (a near-miss made specific),
JUDGE_NOISE, OTHER.
Output JSON {"code": "...", "why": "<= 25 words"}."""

def episodes():
    out = []
    for b in BATCHES:
        G = gt(b)
        for a in ARMS:
            for f in (C / "diag" / "commitjudge" / b / a).glob("*.json"):
                rf = C / "harness_runs" / b / a / f.name
                if not rf.exists(): continue
                cs = json.loads(f.read_text()).get("score")
                r = json.loads(rf.read_text()); fs = (r.get("judge") or {}).get("score")
                if cs is None or fs is None: continue
                d = int(f.name.split(".d")[1][:-5])
                if d > 4: continue
                out.append(dict(b=b, arm=a, pid=f.name.split(".d")[0], d=d, cp=cs >= 7, fp=fs >= 7, cs=cs, fs=fs, rf=str(rf)))
    return out

def code(per):
    sys.path.insert(0, str(C)); import anthropic_oauth; anthropic_oauth.install(); import anthropic
    cli = anthropic.Anthropic(max_retries=6)
    E = [e for e in episodes() if e["cp"] != e["fp"]]
    rng = random.Random(0); pick = []
    for a in ARMS:
        for dirn in (True, False):
            xs = [e for e in E if e["arm"] == a and e["cp"] == dirn]; rng.shuffle(xs); pick += xs[:per]
    def one(e):
        out = C / "diag" / "flips" / e["b"] / e["arm"] / f"{e['pid']}.d{e['d']}.json"
        if out.exists(): return
        r = json.loads(open(e["rf"]).read()); res = r.get("result") or {}
        rat = " | ".join(x.get("rationale", "")[:400] for x in (r.get("judge") or {}).get("replicates", [])[:2])
        u = (f"GRADE: {'PASS -> FAIL' if e['cp'] else 'FAIL -> PASS'} ({e['cs']:.1f} -> {e['fs']:.1f})\n\nGROUND TRUTH:\n"
             f"{gt(e['b'])[e['pid']]['judge_reference']}\n\nCOMMITTED:\n{r['committed_text_hypothesis']['hypothesis']}\n\n"
             f"FINAL:\n{res.get('hypothesis')}\n\nCITED EVIDENCE:\n{(res.get('evidence') or '')[:2500]}\n\nGRADER ON FINAL:\n{rat}")
        for _ in range(3):
            try:
                m = cli.messages.create(model="claude-sonnet-5", max_tokens=300, system=SYS, messages=[{"role": "user", "content": u}])
                t = "".join(x.text for x in m.content if x.type == "text"); j = json.loads(t[t.find("{"): t.rfind("}") + 1])
                out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(dict(j, **{k: e[k] for k in ("cs", "fs", "cp", "fp")})))
                return
            except anthropic.AuthenticationError: raise SystemExit("401")
            except Exception: pass
    with cf.ThreadPoolExecutor(12) as ex: list(ex.map(one, pick))
    print("FLIPS CODED", len(pick))

def report():
    E = episodes(); rows = []
    for a in ARMS:
        xs = [e for e in E if e["arm"] == a]; n = len(xs) or 1
        c = collections.Counter((e["cp"], e["fp"]) for e in xs)
        rows.append(dict(arm=a, n=len(xs), pass_pass=c[(True, True)] / n, pass_fail=c[(True, False)] / n,
                         fail_pass=c[(False, True)] / n, fail_fail=c[(False, False)] / n,
                         committed_pass=(c[(True, True)] + c[(True, False)]) / n, final_pass=(c[(True, True)] + c[(False, True)]) / n))
        print(rows[-1])
    write_rows("p3a_transitions.csv", rows)
    codes = collections.defaultdict(collections.Counter)
    for b in BATCHES:
        for a in ARMS:
            for f in (C / "diag" / "flips" / b / a).glob("*.json"):
                j = json.loads(f.read_text()); codes[(a, "pass->fail" if j["cp"] else "fail->pass")][str(j.get("code")).upper()] += 1
    rows2 = []
    for (a, dirn), c in sorted(codes.items()):
        n = sum(c.values())
        for k, v in c.most_common(): rows2.append(dict(arm=a, direction=dirn, code=k, n=v, share=v / n))
        print(a, dirn, n, {k: f"{v / n:.0%}" for k, v in c.most_common()})
    write_rows("p3a_codes.csv", rows2)

if __name__ == "__main__":
    if sys.argv[1] == "code":
        code(int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 150)
    else:
        report()
