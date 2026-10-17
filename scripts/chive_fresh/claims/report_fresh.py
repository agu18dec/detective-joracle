"""Report for the fresh-batch claims runs: claims_chive.py's metrics() / paired() (pooled AUROC,
within-incident AUROC, Brier, acc@0.5; paired incident-bootstrap vs transcript, 2000 draws,
seed 0), per batch and pooled over batches, on (a) all balanced claims and (b) CHIVE's eval-set
filter, mechanism_concreteness >= 3 (re-balanced 50/50 with build_eval_set.py's rule), plus
edit_cleanliness strata. Main comparisons are restricted to claims every compared arm answered.
Writes $CHIVE_CLAIMS_RUNS/report[_<batches>].md and prints it.
usage: report_fresh.py [batches...]"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

sys.path.insert(0, str(paths.REPO / "src"))
import detective_joracle.chive.claims as _cl  # noqa: E402
from detective_joracle.chive.claims import metrics  # noqa: E402
from detective_joracle.chive.claims import paired as _paired  # noqa: E402


def _fast_auc(pairs):
    # identical to claims.auc (Mann-Whitney, ties half) via midranks, O(n log n)
    pos = sum(1 for _, y in pairs if y)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    srt = sorted(pairs, key=lambda t: t[0])
    rank_sum = 0.0
    i = 0
    while i < len(srt):
        j = i
        while j < len(srt) and srt[j][0] == srt[i][0]:
            j += 1
        mid = (i + j + 1) / 2
        rank_sum += mid * sum(1 for k in range(i, j) if srt[k][1])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


_ref = _cl.auc([(0.3, True), (0.3, False), (0.9, True), (0.1, False), (0.5, False)])
assert (
    abs(_fast_auc([(0.3, True), (0.3, False), (0.9, True), (0.1, False), (0.5, False)]) - _ref)
    < 1e-12
)
_cl.auc = _fast_auc


def paired(*a):
    try:
        return _paired(*a)
    except IndexError:
        return "n/a"


R = paths.CLAIMS_RUNS
REF = "transcript"
batches = sys.argv[1:] or paths.BATCHES
claims, meta, conc, health = {}, {}, {}, defaultdict(lambda: [0, 0, 0, 0])
scored = defaultdict(dict)
for b in batches:
    for line in open(R / b / "claims.jsonl"):
        c = json.loads(line)
        c["batch"] = b
        claims[c["claim_id"]] = c
    meta.update(json.loads((R / b / "claims_meta.json").read_text()))
    cf = R / b / "mechanism_concreteness.jsonl"
    if cf.exists():
        for line in open(cf):
            if line.strip():
                r = json.loads(line)
                conc[r["prompt_id"]] = int(r["mechanism_concreteness"])
    for f in sorted((R / b / "focused").glob("*/*.d*.json")):
        r = json.loads(f.read_text())
        cid = r["claim"]["claim_id"]
        if cid not in claims:
            continue
        h = health[(b, r["arm"])]
        h[0] += 1
        p = (r.get("result") or {}).get("probability_true")
        if str(r.get("stopped_by", "")).startswith("salvaged"):
            h[3] += 1
            continue  # API-truncated episode: excluded, redone on rerun
        if p is None:
            h[1] += 1
            continue
        if r.get("auditor") != "claude-opus-5":
            h[2] += 1
        c = claims[cid]
        ps = scored[r["arm"]].get(cid, ([], c["label"], c["prompt_id"]))[0]
        scored[r["arm"]][cid] = (ps + [float(p)], c["label"], c["prompt_id"])

arms = sorted(scored, key=lambda a: (a != REF, a))


def balanced(ids):
    rng = random.Random(0)
    T = [c for c in ids if claims[c]["label"]]
    F = [c for c in ids if not claims[c]["label"]]
    k = min(len(T), len(F))
    return sorted(rng.sample(T, k) + rng.sample(F, k))


out = [
    f"# CHIVE claims (claims_chive.py focused, draws=1) on {', '.join(batches)}",
    "",
    "## Health (records / no answer / answered by fallback model / salvaged after API error, EXCLUDED)",
    "",
    "| batch | arm | records | no answer | fallback | salvaged |",
    "|---|---|---|---|---|---|",
]
for (b, a), h in sorted(health.items()):
    out.append(f"| {b} | {a} | {h[0]} | {h[1]} | {h[2]} | {h[3]} |")
out.append(
    f"\nconcreteness scored for {sum(1 for c in claims.values() if c['prompt_id'] in conc)}/{len(claims)} claims' incidents"
)


def table(title, ids):
    lines = ["", f"## {title}", ""]
    for a in arms:
        if a == REF:
            continue
        common = [c for c in ids if c in scored[a] and c in scored[REF]]
        nT = sum(claims[c]["label"] for c in common)
        if not common:
            continue
        lines += [
            f"**{a} vs {REF}** on {len(common)} common claims ({nT}T/{len(common) - nT}F, {len({claims[c]['prompt_id'] for c in common})} incidents)",
            "",
            "| arm | AUROC | within-inc AUROC | acc@0.5 | Brier |",
            "|---|---|---|---|---|",
        ]
        for x in (REF, a):
            m = metrics(scored[x], common)
            lines.append(
                f"| {x} | {m['auroc']:.3f} | {m['within_auroc']:.3f} | {m['acc']:.1%} | {m['brier']:.3f} |"
            )
        lines.append(
            f"| diff [95% CI] | {paired(scored[a], scored[REF], common, 'auroc')} | {paired(scored[a], scored[REF], common, 'within_auroc')} | "
            f"{paired(scored[a], scored[REF], common, 'acc')} | {paired(scored[a], scored[REF], common, 'brier')} |"
        )
        lines.append("")
    return lines


all_ids = sorted(claims)
conc_ids = (
    balanced([c for c in all_ids if conc.get(claims[c]["prompt_id"], 0) >= 3]) if conc else []
)
for scope in ["pooled"] + batches:
    ids = all_ids if scope == "pooled" else [c for c in all_ids if claims[c]["batch"] == scope]
    out += table(f"{scope}: all balanced claims", ids)
    if conc:
        cids = (
            conc_ids
            if scope == "pooled"
            else balanced([c for c in ids if conc.get(claims[c]["prompt_id"], 0) >= 3])
        )
        out += table(f"{scope}: concreteness >= 3 (CHIVE eval-set filter, re-balanced)", cids)
for ec in ["minor_targeted", "large_single_factor", "mixed+confounded"]:
    ids = [
        c
        for c in all_ids
        if (meta[c]["edit_cleanliness"] in ("mixed", "confounded")) == (ec == "mixed+confounded")
        and (ec == "mixed+confounded" or meta[c]["edit_cleanliness"] == ec)
    ]
    out += table(f"pooled, edit_cleanliness = {ec}", ids)
text = "\n".join(out)
name = "report.md" if len(batches) == 4 else f"report_{'_'.join(batches)}.md"
(R / name).write_text(text)
print(text)

# ---- reference: CHIVE-native plain answerer (run_binary_eval.py, Opus 5 high) on CHIVE's eval sets
nat = {}
for b in batches:
    f = R / "chive_native" / b / "plain_claude-opus-5_high.jsonl"
    if not f.exists():
        continue
    for line in open(f):
        r = json.loads(line)
        nat[r["claim_id"]] = ([float(r["prob_true"])], bool(r["is_true"]), r["prompt_id"], b)
if nat:
    ns = {c: v[:3] for c, v in nat.items()}
    lines = [
        "",
        "## Reference: CHIVE-native plain answerer (run_binary_eval.py, no tools, Opus 5 high) on CHIVE eval sets (conc>=3, balanced)",
        "",
        "| scope | n | AUROC | within-inc AUROC | acc@0.5 | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for scope in ["pooled"] + batches:
        ids = [c for c in ns if scope == "pooled" or nat[c][3] == scope]
        if not ids:
            continue
        m = metrics(ns, ids)
        lines.append(
            f"| {scope} | {len(ids)} | {m['auroc']:.3f} | {m['within_auroc']:.3f} | {m['acc']:.1%} | {m['brier']:.3f} |"
        )
    lines += [
        "",
        "Paired on claims in both the native eval set and the claims_chive set (focused arm minus native plain):",
        "",
        "| arm | n | AUROC diff | acc diff | Brier diff |",
        "|---|---|---|---|---|",
    ]
    for a in arms:
        common = [c for c in ns if c in scored[a]]
        if common:
            lines.append(
                f"| {a} | {len(common)} | {paired(scored[a], ns, common, 'auroc')} | {paired(scored[a], ns, common, 'acc')} | {paired(scored[a], ns, common, 'brier')} |"
            )
    text += "\n" + "\n".join(lines)
    (R / name).write_text(text)
    print("\n".join(lines))
