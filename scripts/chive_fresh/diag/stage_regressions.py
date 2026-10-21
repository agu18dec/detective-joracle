"""Stage regression picks (OLens loses to transcript / NLA) for the live CHIVE case viewer, in the existing pick format.

Reads the verified case cards' picks from analysis/diag/regressions/picks_{A,B,C}.jsonl, pulls every quote verbatim
from the raw OLens bank (s3d_rl600_all) at the card's position/layer (regex must match), recounts the arm pass counts
from the judged records (d0-d9), checks the shown OLens draw failed, and writes
  regress_stage/v2/index.json.gz = LIVE index with these picks prepended (older regression picks replaced).
It never uploads; the printed command does (Agam runs it).
"""
import gzip, json, re, sys, glob, pathlib, urllib.request

C = pathlib.Path("/workspace/agam/chive_scale"); V = C / "viewer"
LIVE_URL = "https://safety-sahan-2-agam--chive-cases-web.modal.run/data/v2/index.json"
TAG = "[Regression · OLens loses · 2026-10-21]"
ARMS = [("olens_rl_B", "OLens ×4"), ("transcript", "Transcript"), ("mlnla_B", "Multi-layer NLA ×4"), ("jlens_B", "J-lens")]

def passes(b, arm, pid):
    s = []
    for f in sorted(glob.glob(str(C / "harness_runs" / b / arm / f"{pid}.d*.json"))):
        r = json.load(open(f))
        if str(r.get("stopped_by", "")).startswith(("error", "salvaged")) or not r.get("judge"): continue
        s.append((int(f.split(".d")[-1][:-5]), r["judge"]["score"]))
    return dict(s)

def main():
    with urllib.request.urlopen(LIVE_URL, timeout=60) as r:
        raw = r.read()
    try: live = json.loads(gzip.decompress(raw))
    except OSError: live = json.loads(raw)
    rows = [json.loads(l) for f in sorted((C / "analysis/diag/regressions").glob("picks_*.jsonl")) for l in open(f)]
    toks = {}
    for b in {x["batch"] for x in rows}:
        for l in open(C / "datasets" / b / "data/tokens.jsonl"):
            t = json.loads(l); toks[t["prompt_id"]] = t["pieces"]
    picks = []
    for x in rows:
        b, pid = x["batch"], x["pid"]
        bank = json.load(gzip.open(C / "banks/s3d_rl600_all" / f"{pid}.json.gz", "rt"))["readouts"]
        poss = []
        for p in x["positions"]:
            cell = bank[str(p["layer"])].get(str(p["pos"])) or []
            q = next((ln.strip().lstrip("- ").strip()[:220] for s in cell for ln in s.split("\n") if re.search(p["regex"], ln, re.I)), None)
            if not q: sys.exit(f"{pid}: no line at t{p['pos']} L{p['layer']} matches /{p['regex']}/")
            poss.append({"pos": p["pos"], "layer": p["layer"], "label": p["label"], "quote": q, "token": toks[pid][p["pos"]]})
        P = {a: passes(b, a, pid) for a, _ in ARMS}
        d = int(x["failing_olens_draw"]); sc = P["olens_rl_B"].get(d)
        if sc is None or sc >= 7: sys.exit(f"{pid}: OLens d{d} score {sc} is not a failing draw")
        cnt = lambda a: f"{sum(v >= 7 for v in P[a].values())}/{len(P[a])}"
        picks.append({"ds": "chive_fresh", "pid": pid, "arm": "olens_rl_B", "draw": d, "strength": cnt("olens_rl_B"),
                      "title": f"{TAG} {x['title']}",
                      "blurb": f"{x['blurb']} Mechanism: {x['mechanism']}. Fix: {x['fix']}",
                      "positions": poss, "others": ", ".join(f"{lab} {cnt(a)}" for a, lab in ARMS[1:] if P[a]),
                      "set": "regression"})
        print(f"{pid}: OLens {cnt('olens_rl_B')} tr {cnt('transcript')} NLA {cnt('mlnla_B')} · shown d{d} ({sc:.1f}) · {x['mechanism']}")
    old = [p for p in live["picks"] if not str(p.get("title", "")).startswith("[Regression")]
    idx = dict(live, picks=picks + old)
    out = V / "regress_stage" / "v2" / "index.json.gz"; out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(out, "wb", mtime=0, compresslevel=6) as f:
        f.write(json.dumps(idx, ensure_ascii=False, separators=(",", ":")).encode())
    print(f"\n{len(picks)} regression picks + {len(old)} live picks -> {out}")
    print("upload: modal volume put --force --env agam chive-cases-data "
          f"{out} /v2/index.json.gz")

if __name__ == "__main__":
    main()
