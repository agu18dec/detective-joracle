"""Where does a method help or hurt, and how? Wins, regressions and failure modes vs transcript.

For one eval (records judged by run_chive.py), each method arm is compared with the transcript
arm, paired by incident (pass@1 averaged over draws):

  win         arm - transcript >= 0.5      regression   arm - transcript <= -0.5

Every reader episode gets a programmatic hallucination check: quoted strings in its answer
that occur neither in the conversation nor anywhere in that method's readout bank. Regression
episodes get an Opus failure label from a fixed taxonomy; win episodes get a label for what the
readouts contributed. Writes <out>/<eval>/analysis.{json,md}.

    python scripts/analyze_chive.py eval=e1_readonly out=/workspace/agam/chive_runs_dj \\
        "banks=['olens_rl=/workspace/agam/chive_bank/s3d_rl600_all', ...]"
"""

import asyncio
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pydra

from detective_joracle.agent.loop import run_many
from detective_joracle.chive.data import Incident, load_incidents, read_ids
from detective_joracle.chive.judge import JUDGE_MODEL
from detective_joracle.chive.llm import client, json_call

REGRESSION_LABELS = {
    "represented_not_causal": "relied on readouts showing something the model represents and "
    "took it as the cause, but the experiments show it is not (often a ruled-out X item)",
    "misread_readouts": "misinterpreted the readouts: surface restatement taken as hidden "
    "thinking, borrowed details over-read, or a single sample over-trusted",
    "abandoned_text_hypothesis": "considered the correct cause the transcript suggests and "
    "dropped it because of the readouts",
    "conjunction_with_refuted": "named the right cause but bundled it with a ruled-out "
    "component, or committed to a list",
    "hallucinated_evidence": "cited readouts that do not exist or say something else",
    "other": "none of the above",
}
WIN_LABELS = {
    "selected_among_candidates": "the readouts decided between candidates the text already "
    "suggested",
    "surfaced_new_cause": "the readouts pointed to a cause the agent had not otherwise considered",
    "ruled_out_obvious_rival": "the readouts ruled out the obvious text-based explanation",
    "other": "none of the above",
}
LABEL_SYSTEM = """You audit one investigation of why a language model behaved a certain way. You \
see the behaviour, the ground truth established by experiments (causes T1.. and ruled-out \
alternatives X1..), a baseline investigator's answer that used only the transcript, and the \
answer, notes and evidence of an investigator that also had internal readouts of the model. \
Classify, with one label from the list, how the readouts shaped the second investigator's \
answer relative to the baseline. Base it on their notes and evidence, not on guesses."""


def schema(labels: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["label", "explanation"],
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
            "explanation": {"type": "string", "description": "2-3 sentences, specific"},
        },
    }


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def bank_text(bank: Path, pid: str) -> str:
    """Every readout of one incident, normalised and joined (for substring checks)."""
    f = bank / f"{pid}.json.gz"
    if not f.exists():
        f = bank / f"{pid}.json"
    with gzip.open(f, "rt") if f.suffix == ".gz" else f.open() as fh:
        rec = json.load(fh)
    parts = []
    for cells in rec["readouts"].values():
        for samples in cells.values():
            parts += [s if isinstance(s, str) else " ".join(map(str, s)) for s in samples]
    return norm("\n".join(parts))


def quotes(text: str) -> list[str]:
    """Quoted strings, pairing each opening quote with its closer (straight or curly)."""
    out, i = [], 0
    pairs = {'"': '"', "\u201c": "\u201d"}
    while i < len(text):
        c = text[i]
        if c in pairs:
            j = text.find(pairs[c], i + 1)
            if j == -1:
                break
            if 12 <= j - i - 1 <= 300:
                out.append(text[i + 1 : j])
            i = j + 1
        else:
            i += 1
    return out


WORD = re.compile(r"\w+", re.UNICODE)


def words(s: str) -> list[str]:
    return WORD.findall(s.lower())


def bank_index(bank_text_: str) -> tuple[set[str], set[tuple[str, str]]]:
    w = words(bank_text_)
    return set(w), set(zip(w, w[1:], strict=False))


def support(q: str, idx: tuple[set[str], set[tuple[str, str]]], tokens: bool) -> float:
    """How much of a quote the bank backs: share of its word pairs found anywhere in the bank
    (share of its words, for single-token readouts like J-lens). Agents splice samples
    ('a/b/c'), trim and paraphrase, so exact matching cannot tell a composite of real readouts
    from an invented one; low support can."""
    w = words(q)
    if tokens or len(w) < 3:
        return sum(x in idx[0] for x in w) / max(1, len(w))
    pairs = list(zip(w, w[1:], strict=False))
    return sum(x in idx[1] for x in pairs) / len(pairs)


def quote_check(
    r: dict[str, Any], conversation: str, idx: tuple[set[str], set[tuple[str, str]]], tokens: bool
) -> dict[str, int]:
    """Quoted text in the answer that is not conversation text, by bank support: verbatim-ish
    (>= 0.9), supported (>= 0.5), unsupported (< 0.5, the fabrication proxy)."""
    res = r.get("result") or {}
    text = f"{res.get('hypothesis', '')}\n{res.get('evidence', '')}"
    conv = norm(conversation)
    out = {"quotes_verbatim": 0, "quotes_supported": 0, "quotes_unsupported": 0}
    for q in quotes(text):
        if norm(q) in conv or len(words(q)) < 2:
            continue
        x = support(q, idx, tokens)
        k = "verbatim" if x >= 0.9 else "supported" if x >= 0.5 else "unsupported"
        out[f"quotes_{k}"] += 1
    return out


def load(out: Path, ev: str, ids: set[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """arm -> prompt_id -> judged records (all draws)."""
    by: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted((out / ev).glob("*/*.d*.json")):
        r = json.loads(f.read_text())
        if r["prompt_id"] in ids and (r.get("judge") or {}).get("score") is not None:
            by[r["arm"]][r["prompt_id"]].append(r)
    return by


def rate(rs: list[dict[str, Any]]) -> float:
    return sum(bool(r["judge"]["pass"]) for r in rs) / len(rs)


def brief(r: dict[str, Any], notes: bool) -> str:
    res = r.get("result") or {}
    s = (
        f"Hypothesis (judge score {r['judge']['score']:.1f}): {res.get('hypothesis', '')}\n"
        f"Evidence: {res.get('evidence', '')}\nRival: {res.get('rival', '')} -- ruled out by: "
        f"{res.get('rival_ruled_out_by', '')}"
    )
    return s + (f"\nNotes:\n{str(r.get('notebook', ''))[:4000]}" if notes else "")


async def label(
    inc: Incident, base: dict[str, Any], r: dict[str, Any], labels: dict[str, str]
) -> dict[str, Any]:
    user = (
        f"BEHAVIOUR: {inc.behaviour}\n\n{inc.criteria}\n\n## Baseline (transcript only)\n"
        f"{brief(base, notes=False)}\n\n## Investigator with readouts\n{brief(r, notes=True)}\n\n"
        "## Labels\n" + "\n".join(f"- {k}: {v}" for k, v in labels.items())
    )
    got = await json_call(
        client(), model=JUDGE_MODEL, system=LABEL_SYSTEM, user=user, schema=schema(labels)
    )
    return got or {"label": "unlabelled", "explanation": ""}


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.eval = "e1_readonly"
        self.out = pydra.REQUIRED
        self.ids = "data/chive/splits/all300.txt"
        self.hard = "data/chive/splits/hard92.txt"
        self.banks: list[str] = []  # NAME=BANK_DIR for the quote check, one per reader arm
        self.concurrency = 24
        self.label = True  # False: reuse the labels in an existing analysis.json

    def finalize(self) -> None:
        self.out = Path(self.out)
        self.bank_of = {b.split("=", 1)[0]: Path(b.split("=", 1)[1]) for b in self.banks}


def run(cfg: Config) -> None:
    ids = set(read_ids(Path(cfg.ids)))
    hard = set(read_ids(Path(cfg.hard)))
    incs = load_incidents(sorted(ids))
    by = load(cfg.out, cfg.eval, ids)
    base = by["transcript"]
    arms = [a for a in sorted(by) if a != "transcript"]
    convo = {
        p: "\n".join(m["content"] for m in i.messages) + "\n" + i.completion
        for p, i in incs.items()
    }

    prev: dict[tuple[str, str, int], Any] = {}
    prev_file = cfg.out / cfg.eval / "analysis.json"
    if not cfg.label and prev_file.exists():
        for arm, rs in json.loads(prev_file.read_text()).items():
            for row in rs:
                for e in row["episodes"]:
                    if "label" in e:
                        prev[(arm, row["prompt_id"], e["draw"])] = e["label"]
    rows: dict[str, list[dict[str, Any]]] = {}
    jobs = []
    for arm in arms:
        rows[arm] = []
        bank = cfg.bank_of.get(arm)
        for p in sorted(set(by[arm]) & set(base)):
            d = rate(by[arm][p]) - rate(base[p])
            kind = "win" if d >= 0.5 else "regression" if d <= -0.5 else "same"
            idx = bank_index(bank_text(bank, p)) if bank else (set(), set())
            eps = []
            for r in by[arm][p]:
                e: dict[str, Any] = {
                    "draw": r["seed"],
                    "pass": r["judge"]["pass"],
                    "score": r["judge"]["score"],
                    "hypothesis": (r.get("result") or {}).get("hypothesis", ""),
                }
                if bank:
                    e |= quote_check(r, convo[p], idx, tokens=arm.startswith("jlens"))
                eps.append(e)
                old = prev.get((arm, p, r["seed"]))
                if old is not None:
                    e["label"] = old
                failed_vs_base = kind == "regression" and not r["judge"]["pass"]
                won_vs_base = kind == "win" and r["judge"]["pass"]
                if (failed_vs_base or won_vs_base) and cfg.label:
                    b = max(base[p], key=lambda x: x["judge"]["score"] * (1 if won_vs_base else -1))
                    labels = REGRESSION_LABELS if failed_vs_base else WIN_LABELS

                    async def job(
                        e: dict[str, Any] = e,
                        b: dict[str, Any] = b,
                        r: dict[str, Any] = r,
                        labels: dict[str, str] = labels,
                        p: str = p,
                    ) -> None:
                        e["label"] = await label(incs[p], b, r, labels)

                    jobs.append(job)
            rows[arm].append(
                {"prompt_id": p, "hard": p in hard, "delta": d, "kind": kind, "episodes": eps}
            )
    print(f"labelling {len(jobs)} episodes")
    asyncio.run(run_many(jobs, cfg.concurrency))

    lines = [f"# Wins, regressions and failure modes: `{cfg.eval}` vs transcript", ""]
    lines += [
        "Paired by incident (pass@1 over draws). win: arm - transcript >= 0.5; regression: "
        "<= -0.5. Quotes not found = quoted strings in the answer found neither in the "
        "conversation nor in the bank: a quote's support = share of its word pairs found "
        "anywhere in that incident's readouts (words, for J-lens tokens); unsupported < 0.5 "
        "is the fabrication proxy (splices and rewordings of real readouts stay above it).",
        "",
        "| arm | wins | regressions | net | wins hard | regressions hard | unsupported quotes "
        "(all episodes) |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        k = Counter(r["kind"] for r in rows[arm])
        kh = Counter(r["kind"] for r in rows[arm] if r["hard"])
        q = Counter()
        for row in rows[arm]:
            for e in row["episodes"]:
                for kk in ("verbatim", "supported", "unsupported"):
                    q[kk] += e.get(f"quotes_{kk}", 0)
        tot = sum(q.values())
        qs = (
            f"{q['unsupported']}/{tot} ({q['unsupported'] / tot:.1%}); verbatim "
            f"{q['verbatim'] / tot:.0%}"
            if tot
            else "-"
        )
        lines.append(
            f"| {arm} | {k['win']} | {k['regression']} | {k['win'] - k['regression']:+d} | "
            f"{kh['win']} | {kh['regression']} | {qs} |"
        )
    for title, kind in (
        ("Regression failure modes", "regression"),
        ("What the wins came from", "win"),
    ):
        lines += ["", f"## {title} (episodes)", ""]
        for arm in arms:
            c = Counter(
                e["label"]["label"]
                for r in rows[arm]
                if r["kind"] == kind
                for e in r["episodes"]
                if "label" in e
            )
            lines.append(
                f"- **{arm}** (n={sum(c.values())}): "
                + ", ".join(f"{k} {v}" for k, v in c.most_common())
            )
    (cfg.out / cfg.eval / "analysis.json").write_text(json.dumps(rows, ensure_ascii=False))
    text = "\n".join(lines)
    (cfg.out / cfg.eval / "analysis.md").write_text(text)
    print(text)


@pydra.main(Config)
def main(cfg: Config) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
