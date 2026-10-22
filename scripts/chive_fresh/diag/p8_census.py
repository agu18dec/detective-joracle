"""CHIVE incident census (all 915): what kinds of behaviours and causes are in the eval? One Sonnet call per incident.
Labels: domain, behaviour_type, cause_type (of T1), cause_in_behaviour_text (does the classifier question itself name or
strongly cue T1's feature?), interesting (would an interpretability researcher find it surprising: yes/meh/no).
Out: diag/census/<pid>.json ; data/p8_census.csv ; prints tallies.
"""
import json, sys, collections, concurrent.futures as cf
sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from dcommon import *
SYS = """You categorise one incident from an interpretability eval: a conversation with a language model, a behaviour in its
response (the yes/no classifier question used to measure it), and the established cause T1. Output JSON with:
"domain": one of code, creative_writing, roleplay_or_nsfw, factual_qa, advice_or_explanation, agentic_or_tools, translation_or_language, math_or_reasoning, other;
"behaviour_type": one of fabrication_or_hallucination, instruction_violation, format_or_style_choice, language_choice, refusal_or_safety, factual_error, persona_or_framing, overcompliance_or_sycophancy, repetition_or_copying, other;
"cause_type": one of explicit_instruction_wording, ambiguity_or_missing_info, prior_turn_or_context, salient_phrase_or_token, model_prior_or_default, template_or_format, other;
"cause_in_behaviour_text": true if the behaviour/classifier question itself names or strongly cues T1's feature;
"interesting": "yes" if an interpretability researcher would find the behaviour surprising, "meh" if mildly, "no" if it is ordinary behaviour;
"one_line": a 15-word plain description of the incident."""

def main():
    sys.path.insert(0, str(C)); import anthropic_oauth; anthropic_oauth.install(); import anthropic
    cli = anthropic.Anthropic(max_retries=6)
    jobs = [(b, pid) for b in BATCHES for pid in gt(b)]
    def one(j):
        b, pid = j; out = C / "diag" / "census" / f"{pid}.json"
        if out.exists(): return
        I, G = inc(b)[pid], gt(b)[pid]
        conv = "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in I["messages"])
        conv = conv[:4000] + ("\n[...]\n" + conv[-2000:] if len(conv) > 6000 else "")
        u = f"CONVERSATION:\n{conv}\n\nRESPONSE (start): {I['completion'][:1200]}\n\nBEHAVIOUR (classifier question): {I['behaviour']}\n\nT1: {G['targets'][0]['cause']}"
        for _ in range(3):
            try:
                m = cli.messages.create(model="claude-sonnet-5", max_tokens=4000, messages=[{"role": "user", "content": u}], system=SYS)
                t = "".join(x.text for x in m.content if x.type == "text")
                if "{" not in t: continue
                r = json.loads(t[t.find("{"): t.rfind("}") + 1]); r["batch"] = b
                out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(r)); return
            except anthropic.AuthenticationError: raise SystemExit("401")
            except Exception: pass
    with cf.ThreadPoolExecutor(8) as ex: list(ex.map(one, jobs))

def report():
    R = {f.stem: json.loads(f.read_text()) for f in (C / "diag" / "census").glob("*.json")}
    print("n", len(R))
    for k in ("domain", "behaviour_type", "cause_type", "cause_in_behaviour_text", "interesting"):
        c = collections.Counter(str(r.get(k)) for r in R.values())
        print(k, {a: f"{v} ({v / len(R):.0%})" for a, v in c.most_common()})
    write_rows("p8_census.csv", [dict(pid=p, **{k: v for k, v in r.items()}) for p, r in R.items()])

if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else main()
