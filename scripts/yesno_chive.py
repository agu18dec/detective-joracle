"""Report for the CHIVE yes/no eval (run_chive.py yesno=true): would this edit change the behaviour?

Per arm: pooled AUC over items, accuracy at 0.5, Brier and log loss (means over incidents of the
per-incident means over draws). Differences vs `reference` are paired by incident with a seeded
bootstrap over incidents (AUC recomputed on each resample). Strata: all, least-confident third
(confidence.json top_p terciles), hard. Reader arms also report their committed transcript-only
predictions (`yesno_text`), so the readouts' effect within the same episode is visible. Flips:
items the reference gets wrong (mean p on the wrong side of 0.5) that an arm gets right, and the
reverse. Writes <out>/<eval>/report_yesno.md.

    python scripts/yesno_chive.py out=/tmp/chive_yesno eval=yesno_test \\
        ids=data/chive/splits/climb_test204.txt
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydra

from detective_joracle.chive.data import read_ids

Items = dict[tuple[str, str], list[float]]  # (pid, edit id) -> p over draws
METRICS = ("auc", "accuracy", "brier", "logloss")


def auc(pairs: list[tuple[float, bool]]) -> float:
    """Mann-Whitney AUC; ties count half."""
    pos = [p for p, y in pairs if y]
    neg = [p for p, y in pairs if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


Groups = dict[str, list[tuple[float, bool]]]  # pid -> (mean p over draws, label) per edit


def groups(items: Items, labels: dict[tuple[str, str], bool]) -> Groups:
    out: Groups = defaultdict(list)
    for (pid, eid), ps in sorted(items.items()):
        out[pid].append((sum(ps) / len(ps), labels[(pid, eid)]))
    return out


def metrics(per: list[list[tuple[float, bool]]]) -> dict[str, float]:
    """AUC pooled over items; the others are means over incidents of per-incident means."""
    if not per:
        return dict.fromkeys(METRICS, float("nan"))

    def mean_of(f: Any) -> float:
        return sum(sum(f(p, y) for p, y in r) / len(r) for r in per) / len(per)

    def ll(p: float, y: bool) -> float:
        c = min(0.99, max(0.01, p))
        return -math.log(c if y else 1 - c)

    return {
        "auc": auc([x for r in per for x in r]),
        "accuracy": mean_of(lambda p, y: float((p >= 0.5) == y)),
        "brier": mean_of(lambda p, y: (p - y) ** 2),
        "logloss": mean_of(ll),
    }


def scores(items: Items, labels: dict[tuple[str, str], bool], pids: list[str]) -> dict[str, float]:
    g = groups(items, labels)
    return metrics([g[p] for p in pids if p in g])


def paired(
    a: Items, b: Items, labels: dict[tuple[str, str], bool], pids: list[str], metric: str
) -> str:
    """a minus b on `metric` over incidents both arms scored; bootstrap over those incidents."""
    ga, gb = groups(a, labels), groups(b, labels)
    common = [p for p in pids if p in ga and p in gb]
    if not common:
        return "-"
    point = metrics([ga[p] for p in common])[metric] - metrics([gb[p] for p in common])[metric]
    rng = random.Random(0)
    diffs = []
    for _ in range(2000):
        smp = [rng.choice(common) for _ in common]
        d = metrics([ga[p] for p in smp])[metric] - metrics([gb[p] for p in smp])[metric]
        if d == d:  # an all-one-label resample has no AUC
            diffs.append(d)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


class Config(pydra.Config):
    def __init__(self) -> None:
        super().__init__()
        self.out = pydra.REQUIRED
        self.eval = "yesno_test"
        self.ids = "data/chive/splits/climb_test204.txt"
        self.hard = "data/chive/splits/hard92.txt"
        self.confidence = "/workspace/agam/chive_runs_dj/confidence.json"
        self.reference = "transcript"
        self.draws = 2

    def finalize(self) -> None:
        self.out = Path(self.out)


@pydra.main(Config)
def main(cfg: Config) -> None:
    ids = set(read_ids(Path(cfg.ids)))
    final: dict[str, Items] = defaultdict(lambda: defaultdict(list))
    text: dict[str, Items] = defaultdict(lambda: defaultdict(list))
    labels: dict[tuple[str, str], bool] = {}
    health: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen: dict[str, set[str]] = defaultdict(set)
    for f in sorted((cfg.out / cfg.eval).glob("*/*.d*.json")):
        r = json.loads(f.read_text())
        if r["prompt_id"] not in ids:
            continue
        arm = r["arm"]
        seen[arm].add(r["prompt_id"])
        health[arm]["records"] += 1
        health[arm]["error stop"] += str(r.get("stopped_by", "")).startswith("error")
        health[arm]["no answer"] += not r.get("yesno_eval")
        for key, store in (("yesno_eval", final), ("yesno_text", text)):
            for it in (r.get(key) or {}).get("items", []):
                store[arm][(r["prompt_id"], it["id"])].append(float(it["p"]))
                labels[(r["prompt_id"], it["id"])] = bool(it["label"])
    pids_all = sorted({p for p, _ in labels})
    conf = json.loads(Path(cfg.confidence).read_text())
    ranked = sorted(conf, key=lambda p: conf[p]["top_p"])
    low = set(ranked[: len(ranked) // 3])
    hard = set(read_ids(Path(cfg.hard)))
    strata = {
        "all": pids_all,
        "least-confident third": [p for p in pids_all if p in low],
        "hard": [p for p in pids_all if p in hard],
    }
    arms = sorted(final, key=lambda a: (a != cfg.reference, a))
    n_yes = sum(labels.values())
    lines = [
        f"# CHIVE yes/no: `{cfg.eval}`",
        "",
        f"{len(pids_all)} incidents, {len(labels)} edits ({n_yes} YES = moved the behaviour "
        f">= 50 pp, {len(labels) - n_yes} NO = within 15 pp). Probabilities are averaged over "
        "draws per item; accuracy, Brier and log loss are means over incidents; AUC is pooled "
        "over items. Differences vs "
        f"`{cfg.reference}` are paired by incident (bootstrap, * = 95% CI excludes 0).",
        "",
        "## Health",
        "",
        "| arm | records | incidents | error stop | no answer |",
        "|---|---|---|---|---|",
    ]
    for a in arms:
        h = health[a]
        lines.append(
            f"| {a} | {h['records']} | {len(seen[a])} | {h['error stop']} | {h['no answer']} |"
        )
    for name, sub in strata.items():
        lines += [
            "",
            f"## {name} (n={len(sub)} incidents)",
            "",
            "| arm | AUC | accuracy | Brier | log loss | AUC vs ref | Brier vs ref | "
            "transcript-only commit: AUC / Brier |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for a in arms:
            s = scores(final[a], labels, sub)
            t = scores(text[a], labels, sub) if text.get(a) else None
            ref = a == cfg.reference
            lines.append(
                f"| {a} | {s['auc']:.3f} | {s['accuracy']:.1%} | {s['brier']:.3f} | "
                f"{s['logloss']:.3f} | "
                f"{'-' if ref else paired(final[a], final[cfg.reference], labels, sub, 'auc')} | "
                f"{'-' if ref else paired(final[a], final[cfg.reference], labels, sub, 'brier')} | "
                + (f"{t['auc']:.3f} / {t['brier']:.3f}" if t else "-")
                + " |"
            )
    ref_items = final[cfg.reference]
    lines += ["", "## Flips vs the reference (items both scored; mean p across draws)", ""]
    lines += [
        "| arm | reference wrong -> arm right | reference right -> arm wrong |",
        "|---|---|---|",
    ]
    flips: dict[str, list[tuple[str, str]]] = {}
    for a in arms:
        if a == cfg.reference:
            continue
        both = [k for k in final[a] if k in ref_items]

        def right(v: list[float], y: bool) -> bool:
            return (sum(v) / len(v) >= 0.5) == y

        gain = [
            k for k in both if not right(ref_items[k], labels[k]) and right(final[a][k], labels[k])
        ]
        loss = [
            k for k in both if right(ref_items[k], labels[k]) and not right(final[a][k], labels[k])
        ]
        flips[a] = gain
        lines.append(f"| {a} | {len(gain)} | {len(loss)} |")
    text_out = "\n".join(lines)
    (cfg.out / cfg.eval / "report_yesno.md").write_text(text_out)
    (cfg.out / cfg.eval / "flips.json").write_text(json.dumps(flips))
    print(text_out)


if __name__ == "__main__":
    main()
