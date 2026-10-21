"""Judge v3: claim-centric, arm-blind grading of CHIVE hypotheses -> precision, recall, hallucination per answer.

For each answer (the FINAL `result.hypothesis`, and for reader arms also the COMMITTED transcript-only hypothesis):
  1. bullets: the answer -> <= 8 plain causal claims, readout/layer/tool talk stripped (judge2's BUL_SYS, so final-answer
     bullets are reused from judge2_hyp/<b>/bullets; committed-hypothesis bullets are made here).
  2. classify: a fresh Opus grader sees the behaviour, CHIVE's full judge_reference (true causes, refuted rivals, inert
     edits with their measured rates), the numbered TRUE points P1..Pn and refuted rivals R1..Rm, and the bullets.
     Each bullet gets exactly one label:
       TRUE(Pi)      states (part of) true point Pi
       RIVAL(Rj)     states refuted rival Rj as a driver
       CONTRADICTED  conflicts with the ground truth in another way (e.g. claims an inert edit's feature drives it)
       UNSUPPORTED   a specific causal claim the ground truth neither supports nor contradicts (hallucination proxy)
       VAGUE         restates the behaviour / prompt, or too generic to check
       DENY_RIVAL(Rj) a 'NOT:' claim correctly rejecting Rj;  DENY_TRUE(Pi) a 'NOT:' claim wrongly rejecting Pi;
       DENY_OTHER    any other 'NOT:' claim
Grader never sees the arm. Outputs cached per item in diag/judge3/<b>/{cbullets,cls_final,cls_commit}/<arm>/<pid>.d<k>.json

usage: python judge3.py <batch> [--arms a,b] [--draws 0,1,2,3,4] [--ids file] [--limit N] [--which final,commit]
"""
import json, os, sys, random, pathlib, concurrent.futures as cf
sys.path.insert(0, "/workspace/agam/chive_scale"); import anthropic_oauth; anthropic_oauth.install()
import anthropic

B = sys.argv[1]
opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ARMS = opt("--arms", "transcript,olens_rl_B,mlnla_B,jlens_B").split(",")
DRAWS = [int(x) for x in opt("--draws", "0,1,2,3,4").split(",")]
WHICH = opt("--which", "final,commit").split(",")
LIMIT = int(opt("--limit", "0"))
C = pathlib.Path("/workspace/agam/chive_scale"); OUT = C / "diag" / "judge3" / B
H = C / "harness_runs" / B; D = C / "datasets" / B
MODEL = "claude-opus-5"; cli = anthropic.Anthropic(max_retries=6)

rows = lambda f: {r["prompt_id"]: r for r in map(json.loads, open(f))}
INC, GT = rows(D / "data/incidents.jsonl"), rows(D / "data/ground_truth.jsonl")
IDS = open(opt("--ids", "")).read().split() if "--ids" in sys.argv else sorted(p.stem for p in (C / "points" / B / "points").glob("*.json"))
if LIMIT: IDS = IDS[:LIMIT]

sys.argv_backup = sys.argv
BUL_SYS = ("You restate an explanation of why a language model behaves a certain way as a short list of plain "
           "causal claims. Output JSON {\"bullets\": [..]} with at most 8 bullets. Each bullet is ONE claim about "
           "the conversation or about how the model reads/handles it: what drives the behaviour, conditions it "
           "depends on, and alternatives the explanation explicitly rules out (write those as 'NOT: ...'). "
           "Remove every mention of readouts, lenses, layers, token positions, tools, samples, internal "
           "activations or how the evidence was obtained; keep only the claims themselves. Do not add claims "
           "that are not in the explanation.")  # == judge2.BUL_SYS
CLS_SYS = """You grade the individual claims of an explanation of a language model behaviour against ground truth established by
counterfactual experiments. You get: the behaviour, the ground-truth reference (true causes with evidence, ruled-out
causes and inert edits with measured rates), a numbered list of TRUE points (P1..) and REFUTED rivals (R1..), and the
explanation's claims (C1..). Give EVERY claim exactly one label:
- "TRUE" with "match": "Pi" if it states true point Pi or a clearly equivalent mechanism (a narrower or broader version
  that keeps the same causal feature counts; pick the best-matching point).
- "RIVAL" with "match": "Rj" if it asserts refuted rival Rj as a driver.
- "CONTRADICTED" if it conflicts with the ground truth in another way (e.g. it says a feature drives the behaviour but
  an experiment showed editing that feature had no effect, or it misstates the conversation).
- "UNSUPPORTED" if it is a specific causal claim (names a concrete feature or mechanism) that the ground truth neither
  supports nor contradicts.
- "VAGUE" if it only restates the behaviour or the prompt, or is too generic to check.
- For claims phrased as rejections ('NOT: ...'): "DENY_RIVAL" with "match": "Rj" if it correctly rejects a refuted
  rival or an inert feature (use "match": "X" for an inert feature without an R id); "DENY_TRUE" with "match": "Pi" if it
  rejects a true point; "DENY_OTHER" otherwise.
Be strict about TRUE: the claim must name the same causal feature as the point, not merely the same topic.
Output JSON {"labels": [{"c": "C1", "label": "...", "match": "Pi|Rj|X|null"}, ...]} covering every claim in order."""

def ask(system, user, max_tokens=3000):
    for _ in range(4):
        try:
            m = cli.messages.create(model=MODEL, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
            t = "".join(x.text for x in m.content if x.type == "text")
            return json.loads(t[t.find("{"): t.rfind("}") + 1])
        except anthropic.AuthenticationError:
            raise SystemExit("401: token expired")
        except Exception as e:
            last = e
    return None

def cached(path, fn):
    if path.exists(): return json.loads(path.read_text())
    r = fn()
    if r is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(r, ensure_ascii=False)); os.replace(tmp, path)
    return r

def candidates(pid):
    pts = json.loads((C / "points" / B / "points" / f"{pid}.json").read_text())["points"]
    rv = [x["cause"] for x in GT[pid]["ruled_out"] if x.get("kind") == "refuted" and x.get("cause")]
    return pts, rv

def classify(pid, bullets):
    pts, rv = candidates(pid)
    if not bullets: return {"labels": []}
    u = (f"BEHAVIOUR: {INC[pid]['behaviour']}\n\nGROUND-TRUTH REFERENCE:\n{GT[pid]['judge_reference']}\n\n"
         + "TRUE POINTS:\n" + "\n".join(f"P{i + 1}: {p}" for i, p in enumerate(pts))
         + "\n\nREFUTED RIVALS:\n" + ("\n".join(f"R{j + 1}: {r}" for j, r in enumerate(rv)) or "(none)")
         + "\n\nCLAIMS:\n" + "\n".join(f"C{i + 1}: {b}" for i, b in enumerate(bullets)))
    r = ask(CLS_SYS, u)
    if not r or len(r.get("labels", [])) != len(bullets): return None
    r["bullets"] = bullets; r["npts"] = len(pts); r["nrv"] = len(rv)
    return r

def run_one(arm, pid, d, which):
    f = H / arm / f"{pid}.d{d}.json"
    if not f.exists(): return None
    if which == "final":
        bf = C / "judge2_hyp" / B / "bullets" / arm / f"{pid}.d{d}.json"
        bul = json.loads(bf.read_text()) if bf.exists() else cached(OUT / "fbullets" / arm / f"{pid}.d{d}.json",
              lambda: ask(BUL_SYS, "EXPLANATION:\nHYPOTHESIS: " + ((json.loads(f.read_text()).get("result") or {}).get("hypothesis") or "")))
    else:
        rec = json.loads(f.read_text()); ch = (rec.get("committed_text_hypothesis") or {}).get("hypothesis")
        if not ch: return None
        bul = cached(OUT / "cbullets" / arm / f"{pid}.d{d}.json", lambda: ask(BUL_SYS, f"EXPLANATION:\nHYPOTHESIS: {ch}"))
    if not bul or not bul.get("bullets"): return None
    return cached(OUT / f"cls_{which}" / arm / f"{pid}.d{d}.json", lambda: classify(pid, bul["bullets"]))

if __name__ == "__main__":
    jobs = [(a, p, d, w) for w in WHICH for p in IDS for a in ARMS for d in DRAWS if not (w == "commit" and a == "transcript")]
    random.Random(0).shuffle(jobs)
    done = fail = 0
    with cf.ThreadPoolExecutor(int(os.environ.get("J3_THREADS", "16"))) as ex:
        for r in ex.map(lambda j: run_one(*j), jobs):
            done += 1; fail += r is None
            if done % 500 == 0: print(f"[{B}] {done}/{len(jobs)} fail={fail}", flush=True)
    print(f"JUDGE3 {B} DONE {done} fail={fail}", flush=True)
