"""Short README for the Agent Desk dashboard: /workspace/agam/experiments/chive/README.md.
Title, status, key metrics (from report/data CSVs + judge audit / replication summaries), one line per graph."""
import csv, json, pathlib, datetime
from common import REP, LABEL

D = pathlib.Path("/workspace/agam/experiments/chive")
GRAPH_LINES = {
    "final_freeform_10draws.png": "HEADLINE: free-form pass@1 / unbiased pass@5 per arm on all 915 fresh incidents with up to 10 draws each (fills in as robust10 lands); transcript baseline row; paired OLens − NLA per panel.",
    "final_freeform_10draws_bars.png": "HEADLINE (vertical bars, Agam's held-out style): pass@1 / unbiased pass@5 per arm on all 915 fresh incidents, up to 10 draws each; paired OLens − NLA per panel.",
    "mcq_v2_bars.png": "MCQ v2: 10 candidate causes (1 true), dev incidents passing the leak check; accuracy, probability on truth, lure rate per arm.",
    "mcq_v3hard_bars.png": "MCQ v3 (harder distractors; blind-picker accuracy 26%): same metrics.",
    "freeform_hypothesis.png": "Free-form hypothesis pass@1/pass@5 (1-10 judge, pass >= 7) per arm vs the transcript-only baseline row; overall / hardest third / least-confident third.",
    "shuffled_control.png": "Real vs shuffled readouts (readouts from another incident): how much of each reader's gain needs the right readouts.",
    "budgeted_counterfactual.png": "Budgeted counterfactual (1 experiment, n=30 Qwen samples): mover rate, final pass@1, concluder, vs transcript.",
    "point_eval_by_judge.png": "Point eval: every judge version side by side (recall, rivals asserted/rejected), each arm vs transcript.",
    "point_eval_by_judge_hardest_third.png": "Point eval, hardest third of incidents.",
    "point_eval_by_judge_least_confident_third.png": "Point eval, least-confident third of incidents.",
    "claims_auroc.png": "CHIVE claims eval (one yes/no claim per episode) AUROC per arm vs transcript.",
    "single_bullet_k_curve.png": "Single-bullet OLens x k (k = 1, 2, 4, 8, 16) pass@1/pass@5 with the transcript level as a dashed line.",
    "olens_vs_nla_heldout_vs_replicate.png": "Agam's climb_test204 held-out graph re-made for published, replicate (5 new draws), pooled 10 draws, and fresh 915, with paired OLens − NLA per panel.",
    "climb_test204_pooled.png": "Agam's climb_test204 held-out graph with every judged draw pooled (published + re-runs; NLA/OLens growing to 15 draws); paired OLens − NLA per panel.",
    "nla_variance.png": "Is multi-layer NLA noisier than OLens? Per-incident draw spread, run-to-run agreement, and before→after-readout outcomes (write-up: nla_variance.md).",
    "judge_audit.png": "Judge audit: controls (true cause, refuted rival, other incident, generic), test-retest, and pre-readout vs final hypothesis per arm.",
}


def g(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()): return r


def main():
    D.mkdir(parents=True, exist_ok=True)
    A = list(csv.DictReader(open(REP / "data" / "a_passk.csv")))
    pct = lambda x: f"{100 * float(x):.1f}"
    t = g(A, stratum="overall", metric="pass@1", arm="olens_rl_B")
    m = ["## Key metrics (free-form, 915 fresh CHIVE incidents, 5 draws, pass@1 / pass@5)"]
    for a in ["transcript", "olens_rl_B", "jlens_L2060", "mlnla_k1", "mlnla_B"]:
        r1 = g(A, stratum="overall", metric="pass@1", arm=a); r5 = g(A, stratum="overall", metric="pass@5", arm=a)
        if a == "transcript" and t:
            r5t = g(A, stratum="overall", metric="pass@5", arm="olens_rl_B")
            m.append(f"- {LABEL[a]}: {pct(t['transcript'])} / {pct(r5t['transcript'])}"); continue
        if r1 and r5:
            m.append(f"- {LABEL[a]}: {pct(r1['mean'])} / {pct(r5['mean'])} (pass@1 {100 * float(r1['diff']):+.1f} pp "
                     f"[{100 * float(r1['lo']):+.1f}, {100 * float(r1['hi']):+.1f}] vs transcript)")
    extra = REP.parent / "analysis" / "judge_audit" / "dash_lines.md"
    for extra in (extra, REP.parent / "replicate_runs" / "dash_lines.md"):
        if extra.exists(): m += ["", extra.read_text().strip()]
    graphs = sorted(p.name for p in D.glob("*.png"))
    out = [f"# CHIVE fresh: OLens vs J-lens vs multi-layer NLA vs transcript (free-form hypotheses)", "",
           f"Status: running (k-curve, antilure test, climb_test204 replication, judge audit). Updated "
           f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC. Full report: /workspace/agam/chive_graphs/README.md", ""]
    out += m + ["", "## Graphs"] + [f"- `{p}`: {GRAPH_LINES.get(p, '')}" for p in graphs]
    tmp = D / ".README.md.tmp"; tmp.write_text("\n".join(out) + "\n"); tmp.replace(D / "README.md")


if __name__ == "__main__":
    main()
