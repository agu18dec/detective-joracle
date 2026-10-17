"""Harness-record dossier for one incident: per arm and draw, the judge score, the committed
(text-only) hypothesis, the final hypothesis and evidence, the tools called and the judge's first
rationale.

    python hdossier.py <batch> <pid> [--arms a,b] [--full] [--runs DIR]

Reads $CHIVE_RUNS/<batch>/<arm>/<pid>.d<k>.json for every arm dir present.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("batch")
    ap.add_argument("pid")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--full", action="store_true", help="no truncation")
    ap.add_argument("--runs", type=Path, default=paths.RUNS)
    a = ap.parse_args()
    arms = a.arms.split(",") if a.arms else None

    def short(s: str | None, n: int) -> str:
        s = (s or "").replace("\n", " ")
        return s if a.full or len(s) <= n else s[:n] + "…"

    for d in sorted(p for p in (a.runs / a.batch).iterdir() if p.is_dir()):
        if arms and d.name not in arms:
            continue
        fs = sorted(
            d.glob(f"{a.pid}.d*.json"), key=lambda f: int(f.name.split(".d")[-1].split(".")[0])
        )
        if not fs:
            continue
        print(f"\n######## {d.name}")
        for f in fs:
            r = json.loads(f.read_text())
            j = r.get("judge") or {}
            res = r.get("result") or {}
            c = r.get("committed_text_hypothesis") or {}
            tools = [
                t["name"] + "(" + short(json.dumps(t.get("args"), ensure_ascii=False), 70) + ")"
                for t in r.get("tool_log", [])
                if t["name"] not in ("notebook",)
            ]
            print(
                f"\n--- d{r['seed']} score={j.get('score')} stop={r['stopped_by']} out={r['output_tokens']}"
            )
            if c:
                print("  COMMITTED:", short(c.get("hypothesis") if isinstance(c, dict) else c, 400))
            print("  FINAL:", short(res.get("hypothesis"), 700))
            if res.get("evidence"):
                print("  EVIDENCE:", short(res.get("evidence"), 900))
            print("  TOOLS:", " · ".join(tools[:30]))
            reps = j.get("replicates") or []
            if reps:
                print("  JUDGE:", short(reps[0].get("rationale"), 300))


if __name__ == "__main__":
    main()
