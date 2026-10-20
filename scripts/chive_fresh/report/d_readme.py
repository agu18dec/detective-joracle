"""(d) Assemble report/README.md and /workspace/agam/chive_graphs/README.md from the CSVs written by a/b/c/e.
Hand-written interpretation lives in report/notes_manual.md and is inserted verbatim (keep its numbers in sync
with the tables; the tables themselves are always regenerated)."""
import csv, json, datetime, collections
from common import *

def rd(name):
    f = REP / "data" / f"{name}.csv"
    return list(csv.DictReader(open(f))) if f.exists() else []

def pp(x, pct=True, sign=False):
    if x in ("", None): return "–"
    v = float(x) * (100 if pct else 1)
    return (f"{v:+.1f}" if sign else f"{v:.1f}") if pct else (f"{v:+.3f}" if sign else f"{v:.3f}")

def cell(r, pct=True):
    if r["arm"] == "transcript": return f"{pp(r['mean'], pct)}"
    return f"{pp(r['mean'], pct)} ({pp(r['diff'], pct, True)} [{pp(r['lo'], pct, True)}, {pp(r['hi'], pct, True)}])"

def table(rows, keycols, pct=True, arms=None):
    """rows filtered to one block; pivot arms -> columns over keycols."""
    arms = arms or [a for a in ARMS if any(r["arm"] == a for r in rows)]
    groups = collections.OrderedDict()
    for r in rows: groups.setdefault(tuple(r[k] for k in keycols), {})[r["arm"]] = r
    out = ["| " + " | ".join(keycols) + " | n | " + " | ".join(LABEL[a] for a in arms) + " |",
           "|" + "---|" * (len(keycols) + 1 + len(arms))]
    for k, g in groups.items():
        n = "/".join(sorted({g[a]["n"] for a in g}, key=int)) if g else ""
        out.append("| " + " | ".join(k) + f" | {n} | " + " | ".join(cell(g[a], pct) if a in g else "–" for a in arms) + " |")
    return "\n".join(out)

def key_numbers(A, B, Cp, Cl, Sh, bm, cm):
    """Auto-generated headline numbers (always in sync with the CSVs)."""
    def g(rows, **kw):
        for r in rows:
            if all(r.get(k) == v for k, v in kw.items()): return r
    def d(r, pct=True):
        if not r: return "n/a"
        k = 100 if pct else 1; f = "{:+.1f}" if pct else "{:+.3f}"
        return f"{f.format(k * float(r['diff']))} [{f.format(k * float(r['lo']))}, {f.format(k * float(r['hi']))}]"
    def v(r, pct=True):
        return "n/a" if not r else (f"{100 * float(r['mean']):.1f}%" if pct else f"{float(r['mean']):.3f}")
    out = ["## Key numbers (auto-generated from the CSVs)", ""]
    arms = [a for a in ARMS if a != "transcript" and g(A, metric="pass@1", stratum="overall", arm=a)]
    if arms:
        t = g(A, metric="pass@1", stratum="overall", arm=arms[0])
        out.append(f"- **Free-form pass@1** (n={t['n']}; transcript {100 * float(t['transcript']):.1f}%), difference vs transcript in pp [95% CI]: "
                   + "; ".join(f"{LABEL[a]} {v(g(A, metric='pass@1', stratum='overall', arm=a))} ({d(g(A, metric='pass@1', stratum='overall', arm=a))})" for a in arms) + ".")
        for st in ("hardest third", "least-confident third"):
            t = g(A, metric="pass@1", stratum=st, arm=arms[0])
            out.append(f"  - {st} (n={t['n']}, transcript {100 * float(t['transcript']):.1f}%): " + "; ".join(f"{LABEL[a]} {d(g(A, metric='pass@1', stratum=st, arm=a))}" for a in arms) + ".")
        t = g(A, metric="pass@5", stratum="overall", arm=arms[0])
        out.append(f"- **Free-form pass@5** (transcript {100 * float(t['transcript']):.1f}%): " + "; ".join(f"{LABEL[a]} {v(g(A, metric='pass@5', stratum='overall', arm=a))} ({d(g(A, metric='pass@5', stratum='overall', arm=a))})" for a in arms) + ".")
    hh = rd("a_olens_vs_readers")
    if hh:
        out.append("- **OLens head-to-head, pass@1 overall** (pp [95% CI], positive = OLens better): " + "; ".join(
            f"{LABEL[r['olens']]} − {LABEL[r['other']]} {100 * float(r['diff']):+.1f} [{100 * float(r['lo']):+.1f}, {100 * float(r['hi']):+.1f}]"
            for r in hh if r["stratum"] == "overall" and r["metric"] == "pass@1" and r["other"] in ("jlens_L2060", "mlnla_k1", "mlnla_B")) + ".")
    for r in Sh:
        out.append(f"- **Shuffled control** {LABEL[r['reader']]} (n={r['n']}): real {pp(r['real'])}% vs shuffled {pp(r['shuffled'])}% vs transcript {pp(r['transcript'])}%; real − shuffled {pp(r['real_minus_shuf'], sign=True)} [{pp(r['real_minus_shuf_lo'], sign=True)}, {pp(r['real_minus_shuf_hi'], sign=True)}] pp.")
    if B:
        ba = [a for a in ARMS if a != "transcript" and g(B, stratum="overall", metric="final pass@1", arm=a)]
        for lab in ("final pass@1", "concluder pass@1", "mover rate (|Δ| >= 30 pp)", "reduction rate (Δ <= -30 pp)"):
            t = g(B, stratum="overall", metric=lab, arm=ba[0]) if ba else None
            if t: out.append(f"- **Budgeted {lab}** (n={t['n']}{'' if not bm.get('missing') else ', interim, missing ' + ', '.join(bm['missing'])}; transcript {100 * float(t['transcript']):.1f}%): "
                             + "; ".join(f"{LABEL[a]} {v(g(B, stratum='overall', metric=lab, arm=a))} ({d(g(B, stratum='overall', metric=lab, arm=a))})" for a in ba) + ".")
    for jk, mk, lab in (("v1", "nonobv", "Point eval v1 non-obvious recall"), ("v1", "primary", "Point eval v1 primary-cause recall"), ("v1", "rv_ass", "Point eval v1 refuted rivals asserted"),
                        ("regrade", "nonobv", "Blind non-obvious re-grade"), ("j2hyp", "true", "Judge v2 hyp-only true points asserted"), ("j2hyp", "nonobv", "Judge v2 hyp-only non-obvious asserted"),
                        ("j2hyp", "rival", "Judge v2 hyp-only refuted rivals asserted"), ("j2full", "rival", "Judge v2 full refuted rivals asserted")):
        rr = [r for r in Cp if r["judge"] == jk and r["metric"] == mk and r["stratum"] == "overall" and r["arm"] != "transcript"]
        if rr: out.append(f"- **{lab}** (n={rr[0]['n']}; transcript {100 * float(rr[0]['transcript']):.1f}%): " + "; ".join(f"{LABEL[r['arm']]} {d(r)}" for r in rr) + ".")
    rr = [r for r in Cl if r["stratum"] == "overall" and r["arm"] != "transcript"]
    if rr: out.append("- **Claims AUROC** (vs transcript on shared claims): " + "; ".join(f"{LABEL[r['arm']]} {float(r['mean']):.3f} vs {float(r['transcript']):.3f} ({d(r, False)}, n={r['n']} incidents)" for r in rr) + ".")
    return out + [""]

def main():
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    am = json.load(open(REP / "data" / "a_meta.json")) if (REP / "data" / "a_meta.json").exists() else {}
    bm = json.load(open(REP / "data" / "b_meta.json")) if (REP / "data" / "b_meta.json").exists() else {}
    cm = json.load(open(REP / "data" / "c_meta.json")) if (REP / "data" / "c_meta.json").exists() else {}
    final = am.get("n_matched", 0) >= 800 and not am.get("missing") and not bm.get("missing")
    A = rd("a_passk") if am.get("n_matched", 0) >= 800 else rd("a_passk_pairwise")
    B = rd("b_budget"); Cp = rd("c_points"); Cl = rd("c_claims"); Sh = rd("a_shuffle")
    shines = json.load(open(REP / "olens_shines.json"))["items"] if (REP / "olens_shines.json").exists() else []
    ns = collections.Counter(x["cls"] for x in shines)
    L = [f"# CHIVE fresh eval: OLens vs transcript, J-lens and multi-layer NLA on 915 new incidents (Qwen3.6-27B)", "",
         f"_{'FINAL' if final else 'INTERIM'} — generated {now} by `report/run_all.sh`. "
         + ("" if final else "Runs are still finishing; numbers will move. ") + "_", "",
         "## Setup", "",
         "- 915 new CHIVE incidents (batches fresh1-4), Qwen3.6-27B subject. One frozen detective-joracle harness, Opus 5 investigator, 5 draws per incident (d0-d4); arms differ only in the readout tool.",
         "- Arms: transcript only; OLens x4 (`olens_rl_B`, 4-bullet RL OLens, 4 samples/cell); OLens x1 (`olens_k1`); single-bullet OLens x4 (`olens_single`); "
         "J-lens top-10 at L20-60 step 4 (`jlens_L2060`, same layers as OLens/NLA) and at L0-62 (`jlens_B`, secondary); multi-layer NLA x1 (`mlnla_k1`) and x4 (`mlnla_B`).",
         "- Records with `stopped_by` starting `error`/`salvaged` are dropped everywhere. Stats: paired incident bootstrap (2000 resamples) 95% CI of the difference vs transcript.",
         "- Strata are fixed over all incidents: **hardest third** = lowest no-tools plain-answer primary-cause recall (point eval v1); **least-confident third** = lowest no-tools `top_p` (confidence/<batch>/confidence.json).",
         f"- Cells read `value (difference vs transcript [95% CI])`, in % / pp unless marked.", ""]
    if not final:
        L += ["### Coverage right now", "",
              f"- Free-form: {am.get('n_matched', 0)} incidents fully matched over {', '.join(am.get('present', []))}; not yet available: {', '.join(am.get('missing', [])) or 'none'}. "
              f"Complete incidents per arm: " + ", ".join(f"{a} {n}" for a, n in am.get("coverage", {}).items()) + ". "
              + ("Free-form tables below are therefore pairwise (each arm vs transcript on the incidents both have), not matched." if am.get("n_matched", 0) < 800 else ""),
              "- Budgeted (judging runs after every arm's agent stage, so this is the fresh1 subset until the rest is judged): " + "; ".join(f"{k}: {v['n_matched']} matched over {', '.join(v['present'])}" for k, v in bm.get('per_metric', {}).items()) + f"; missing {', '.join(bm.get('missing', [])) or 'none'}.",
              "- Point evals (judge: matched n, arms): " + "; ".join(f"{k}: {v['n_matched']} over {', '.join(v['present'])}" for k, v in cm.items()), ""]
    L += key_numbers(A, B, Cp, Cl, Sh, bm, cm)
    notes = REP / "notes_manual.md"
    if notes.exists(): L += [notes.read_text(), ""]
    L += ["## (a) Free-form hypothesis generation (old 1-10 judge, pass = score >= 7)", "", "![free-form](fig/a_passk.png)", "",
          table(A, ["metric", "stratum"]), "", "Data: `data/a_passk.csv` (matched), `data/a_passk_pairwise.csv`.", "",
          "### Shuffled-readout control (150 control incidents; readouts from another incident)", "", "![shuffle](fig/a_shuffle.png)", ""]
    if Sh:
        L += ["| reader | n | real | shuffled | transcript | real − shuffled [95% CI] | shuffled − transcript [95% CI] |", "|---|---|---|---|---|---|---|"]
        for r in Sh:
            L.append(f"| {LABEL[r['reader']]} | {r['n']} | {pp(r['real'])} | {pp(r['shuffled'])} | {pp(r['transcript'])} | {pp(r['real_minus_shuf'], sign=True)} [{pp(r['real_minus_shuf_lo'], sign=True)}, {pp(r['real_minus_shuf_hi'], sign=True)}] | "
                     f"{pp(r['shuf_minus_tr'], sign=True)} [{pp(r['shuf_minus_tr_lo'], sign=True)}, {pp(r['shuf_minus_tr_hi'], sign=True)}] |")
    L += ["", "## (b) Budgeted counterfactual (1 experiment, n=30 Qwen samples, draws d0-d1)", "",
          "Mover = the single experiment moves the behaviour rate by >= 30 pp vs the baseline re-measured with the same executor (`exp1/baselines.json`); reduction = it lowers the rate by >= 30 pp. Concluder = pass@1 of the post-experiment concluder.", "",
          "![budget](fig/b_budget.png)", "", table(B, ["metric", "stratum"]), "", "Data: `data/b_budget.csv`.", "",
          "## (c) Point-based evaluation, every judge side by side", "",
          "![points](fig/c_points_overall.png)", "", "Strata: [hardest third](fig/c_points_hardest_third.png), [least-confident third](fig/c_points_least_confident_third.png).", ""]
    for st in STRATA:
        rr = [r for r in Cp if r["stratum"] == st]
        if not rr: continue
        L += [f"### {st}", ""]
        for pct in (True, False):
            sub = [r for r in rr if (r["metric"] != "disproved") == pct]
            if sub: L += [table(sub, ["label"], pct=pct), ""]
    L += ["Point eval v1 rivals: share of CHIVE refuted rivals the answer asserts/rejects (per incident, averaged). Judge v2: blind fixed-list ASSERTED/DENIED/ABSENT over CHIVE true points and refuted rivals; hyp-only grades the final hypothesis, full grades hypothesis + evidence + rival. Data: `data/c_points.csv`.", "",
          "### Claims eval (yes/no counterfactual claims, pooled AUROC)", "", "![claims](fig/c_claims.png)", "", table(Cl, ["stratum"], pct=False), "",
          "Each arm vs transcript on the claims both answered (n = incidents); J-lens L0-62 and multi-layer NLA x1 claims runs are a fresh1 subset. Data: `data/c_claims.csv`.", "",
          "## (e) Incident-level OLens wins", "",
          f"[olens_shines.md](olens_shines.md): strict {ns['strict']}, pending {ns['pending']} (some comparator still running), borderline {ns['borderline']}. Readout counts in that file are recounted from the raw banks.", "",
          "## Files", "", "- Scripts: `report/common.py`, `a_freeform.py`, `b_budget.py`, `c_points.py`, `c_claims.py`, `e_shines.py`, `d_readme.py`; `run_all.sh` runs them all.",
          "- Figures: `report/fig/*.png` (copies in `/workspace/agam/chive_graphs/`); numbers: `report/data/*.csv`."]
    (REP / "README.md").write_text("\n".join(L) + "\n")

    G = ["# CHIVE fresh eval graphs (Qwen3.6-27B, 915 incidents)", "",
         f"{'FINAL' if final else 'INTERIM'} — regenerated {now} from `/workspace/agam/chive_scale/report/` (`run_all.sh`). Numbers + write-up: `/workspace/agam/chive_scale/report/README.md`.", "",
         "Every panel shows each arm's paired difference vs transcript-only (dot) with a 95% incident bootstrap CI (bar); the number at the right is the arm's absolute value. Blue = OLens, orange = J-lens, green = multi-layer NLA; hollow = secondary.", "",
         "- `freeform_hypothesis.png` — free-form hypothesis generation, old 1-10 judge: pass@1 and pass@5, overall / hardest third / least-confident third." + ("" if am.get("n_matched", 0) >= 800 else " INTERIM: pairwise vs transcript (n per arm at right) until every arm is complete."),
         "- `shuffled_control.png` — real vs shuffled (another incident's) readouts vs transcript on the 150 control incidents.",
         "- `budgeted_counterfactual.png` — 1-experiment budget: final pass@1, concluder pass@1, mover rate (>= 30 pp change), reduction rate (>= 30 pp drop), three strata.",
         "- `point_eval_by_judge.png` — every judge side by side, overall: old pass judge, point eval v1 (non-obvious / primary recall, disproved per answer, precision, rivals asserted/rejected), blind non-obvious re-grade, judge v2 hyp-only and full (true pts asserted, non-obvious asserted, true denied, refuted rivals asserted).",
         "- `point_eval_by_judge_hardest_third.png`, `point_eval_by_judge_least_confident_third.png` — same, per stratum.",
         "- `claims_auroc.png` — yes/no claim AUROC per arm vs transcript, three strata.",
         "- `single_bullet_k_curve.png` — single-bullet OLens free-form pass@1 / pass@5 vs samples per cell k = 1, 2, 4, 8, 16 (matched incidents, 95% CI band, transcript dashed).", ""]
    GRAPHS.mkdir(parents=True, exist_ok=True); (GRAPHS / "README.md").write_text("\n".join(G))

if __name__ == "__main__":
    main()
