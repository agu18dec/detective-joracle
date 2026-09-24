"""Translate the non-English lens cells for the viewer's italic EN line.

    OPENROUTER_API_KEY=… python scripts/weirdchat/translate_cells.py \\
        cells=outputs/weirdchat/site/cells_to_translate.json \\
        out=outputs/weirdchat/translations.json [model=anthropic/claude-haiku-4.5] [batch=20]

``cells`` is the list ``build_site.py`` writes (every sample containing CJK or other non-Latin
script); ``out`` is the cache ``{sample: english}`` it reads back with ``translations=``. The
cache is resumable: samples already present are skipped. Translations are literal — the cell is
a verbalizer's guess about an activation, so idiom is noise and the words are the signal.
"""

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from detective_joracle.llm.route import async_json_route  # noqa: E402

SYSTEM = (
    "You translate short fragments into English. Each fragment is text an interpretability lens "
    "decoded from a language model's internal state, so it may be broken, mixed-language or cut "
    "off; translate literally and keep English parts as they are. Return one English string per "
    "fragment, in order, the same count as given."
)
USER = 'Translate these {n} fragments. Respond as JSON: {{"translations": ["…", …]}}\n\n{items}'
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
    "required": ["translations"],
}


def parse(argv: list[str]) -> dict[str, str]:
    opts = {"model": "anthropic/claude-haiku-4.5", "batch": "20"}
    for a in argv:
        k, sep, v = a.partition("=")
        if not sep:
            raise SystemExit(f"bad argument {a!r} (key=value)")
        opts[k] = v
    for k in ("cells", "out"):
        if k not in opts:
            raise SystemExit(f"{k}= is required")
    return opts


def main(argv: list[str]) -> None:
    opts = parse(argv)
    cells = json.loads(Path(opts["cells"]).read_text())
    out_path = Path(opts["out"])
    cache: dict[str, str] = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [c for c in cells if c not in cache]
    n = int(opts["batch"])
    batches = [todo[i : i + n] for i in range(0, len(todo), n)]
    print(
        f"{len(cells)} cells, {len(cache)} cached, {len(todo)} to translate in {len(batches)} calls"
    )
    if not batches:
        return
    items = [
        (
            SYSTEM,
            USER.format(
                n=len(b),
                items="\n".join(
                    f"{i + 1}. {json.dumps(c, ensure_ascii=False)}" for i, c in enumerate(b)
                ),
            ),
        )
        for b in batches
    ]
    results = async_json_route(items, schema=SCHEMA, model=opts["model"], concurrency=32)
    ok = bad = 0
    for b, r in zip(batches, results, strict=True):
        tr = (r or {}).get("translations") or []
        if len(tr) != len(b):
            bad += 1
            continue
        for c, t in zip(b, tr, strict=True):
            cache[c] = str(t)
        ok += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    print(
        f"wrote {out_path}: {len(cache)} translations ({ok} batches ok, {bad} failed — rerun to retry)"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
