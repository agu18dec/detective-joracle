"""Shared loaders, strata, paired bootstrap and plotting for the CHIVE fresh report.

All harness records are read through a small mtime-keyed cache (report/cache/records.pkl) so re-runs
are cheap. Records whose stopped_by starts with "error" or "salvaged" are dropped everywhere.
Only base draws d0-d4 are used.
"""
import glob, gzip, json, os, pickle, random, pathlib, collections
import numpy as np

C = pathlib.Path("/workspace/agam/chive_scale")
REP = C / "report"
GRAPHS = pathlib.Path("/workspace/agam/chive_graphs")
BATCHES = ["fresh1", "fresh2", "fresh3", "fresh4"]
HEADLINE = ["transcript", "olens_rl_B", "olens_k1", "olens_single", "olens_single_k1", "olens_single_k2", "olens_single_k8", "olens_single_k16", "jlens_L2060", "mlnla_k1", "mlnla_B"]
SECONDARY = ["jlens_B"]
ARMS = ["transcript", "olens_rl_B", "olens_k1", "olens_single", "olens_single_k1", "jlens_L2060", "jlens_B", "mlnla_k1", "mlnla_B"]
OLENS = ["olens_k1", "olens_rl_B", "olens_single", "olens_single_k1"]
LABEL = {
    "transcript": "Transcript only (baseline)",
    "olens_k1": "OLens x1",
    "olens_rl_B": "OLens x4",
    "olens_single": "Single-bullet OLens x4",
    "olens_single_k1": "Single-bullet OLens x1",
    "olens_single_k2": "Single-bullet OLens x2",
    "olens_single_k8": "Single-bullet OLens x8",
    "olens_single_k16": "Single-bullet OLens x16",
    "jlens_L2060": "J-lens L20-60",
    "jlens_B": "J-lens L0-62 (secondary)",
    "mlnla_k1": "Multi-layer NLA x1",
    "mlnla_B": "Multi-layer NLA x4",
    "olens_shufc": "OLens shuffled",
    "jlens_shufc": "J-lens shuffled",
    "mlnla_k1_shufc": "NLA shuffled",
}
# ---- palette (dataviz reference instance, light mode; first three slots validated all-pairs)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FAM_COLOR = {"olens": "#2a78d6", "jlens": "#eb6834", "mlnla": "#1baf7a", "transcript": "#898781"}
def family(a):
    for f in ("olens", "jlens", "mlnla"):
        if a.startswith(f): return f
    return "transcript"

def bad(r):
    return str(r.get("stopped_by", "")).startswith(("error", "salvaged"))

# ---------------------------------------------------------------- harness records (cached)
_CACHE = REP / "cache" / "records.pkl"
def _extract(r):
    j = r.get("judge") or {}; cj = r.get("concluder_judge") or {}
    ex = (r.get("experiments") or [])
    e0 = ex[0] if ex else None
    return {"score": j.get("score"), "stopped_by": r.get("stopped_by"), "seed": r.get("seed"),
            "conc_score": cj.get("score") if isinstance(cj, dict) else None,
            "exp": ({"k": e0.get("k"), "n": e0.get("n"), "rate": e0.get("rate")} if e0 else None)}

def load_records(evals, arms, draws=range(5)):
    """-> dict[(eval, arm, pid, draw)] = extracted fields (bad records dropped)."""
    cache = pickle.load(open(_CACHE, "rb")) if _CACHE.exists() else {}
    import concurrent.futures as cf
    out = {}; todo = []; meta = {}
    for ev in evals:
        for a in arms:
            for f in glob.glob(str(C / "harness_runs" / ev / a / "*.json")):
                name = os.path.basename(f)
                try: pid, d = name[:-5].split(".d"); d = int(d)
                except ValueError: continue
                if d not in draws: continue
                meta[f] = (ev, a, pid, d)
    def stat(f):
        try: return f, os.path.getmtime(f)
        except OSError: return f, None
    def read(item):
        f, mt = item
        try: return f, (mt, _extract(json.load(open(f))))
        except Exception: return f, None
    with cf.ThreadPoolExecutor(64) as ex:
        mts = dict(ex.map(stat, list(meta)))
        todo = [(f, mt) for f, mt in mts.items() if mt is not None and (f not in cache or cache[f][0] != mt)]
        for f, v in ex.map(read, todo):
            if v: cache[f] = v
    changed = bool(todo)
    for f, key in meta.items():
        if f not in cache or mts.get(f) is None: continue
        rec = cache[f][1]
        if bad(rec): continue
        out[key] = rec
    if changed:
        tmp = str(_CACHE) + ".tmp"; pickle.dump(cache, open(tmp, "wb")); os.replace(tmp, _CACHE)
    return out

def pass_table(arms, batches=BATCHES, draws=range(5)):
    """-> {(batch,pid): {arm: [bool]*5}} for incidents where the arm has all draws judged."""
    R = load_records(batches, arms, draws)
    T = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (b, a, pid, d), r in R.items():
        if r["score"] is not None: T[(b, pid)][a][d] = r["score"] >= 7
    out = {}
    for k, v in T.items():
        out[k] = {a: [v[a][d] for d in draws] for a in v if len(v[a]) == len(draws)}
    return out

# ---------------------------------------------------------------- strata
def plain_recall():
    """no-tools plain answers' primary-cause recall per incident (3 plain answers)."""
    out = {}
    for b in BATCHES:
        for f in glob.glob(str(C / "points" / b / "tags" / "*.json")):
            pid = f.split("/")[-1][:-5]; prim = json.load(open(f))["primary"]
            gs = [json.load(open(x))["point_scores"] for x in sorted(glob.glob(str(C / "points" / b / "grades/plain" / f"{pid}.d*.json")))[:3]]
            idx = [i for i, t in enumerate(prim) if t]
            if len(gs) == 3 and idx and all(len(g) == len(prim) for g in gs):
                out[(b, pid)] = sum(sum(g[i] - 1 for i in idx) / (9 * len(idx)) for g in gs) / 3
    return out

def top_p():
    out = {}
    for b in BATCHES:
        f = C / "confidence" / b / "confidence.json"
        if f.exists():
            for pid, v in json.load(open(f)).items():
                if v.get("top_p") is not None: out[(b, pid)] = v["top_p"]
    return out

def strata():
    """Fixed strata over ALL incidents (not re-cut per analysis): bottom third by plain primary recall
    ('hardest third') and bottom third by no-tools top_p ('least-confident third')."""
    pr = plain_recall(); tp = top_p()
    h = sorted(pr, key=lambda k: (pr[k], k)); l = sorted(tp, key=lambda k: (tp[k], k))
    return {"overall": None, "hardest third": set(h[:len(h) // 3]), "least-confident third": set(l[:len(l) // 3])}
STRATA = ["overall", "hardest third", "least-confident third"]

def in_stratum(keys, S, name):
    return list(keys) if S[name] is None else [k for k in keys if k in S[name]]

# ---------------------------------------------------------------- paired bootstrap
def boot(vals, ref, n_boot=2000, seed=0):
    """vals/ref: arrays over the same incidents. Returns mean(vals), mean(ref), diff, lo, hi (95%)."""
    v = np.asarray(vals, float); r = np.asarray(ref, float); n = len(v)
    if n == 0: return (np.nan,) * 5
    d = v - r; rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n)); bs = np.sort(d[idx].mean(1))
    return v.mean(), r.mean(), d.mean(), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]

def boot_ci(vals, n_boot=2000, seed=0):
    v = np.asarray(vals, float); n = len(v)
    if n == 0: return (np.nan,) * 3
    rng = np.random.default_rng(seed); bs = np.sort(v[rng.integers(0, n, size=(n_boot, n))].mean(1))
    return v.mean(), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]

# ---------------------------------------------------------------- plotting
def mpl():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.titleweight": "bold"})
    return plt

def forest(ax, rows, title, pct=True, xlabel="difference vs transcript", show_labels=True, ref_text=None, show_n=False):
    """rows: list of dicts {arm, mean, diff, lo, hi, n}. Horizontal dot + 95% CI of the paired difference vs
    transcript per arm; the arm's absolute value (and n if show_n) is printed at the right edge."""
    k = 100 if pct else 1
    rows = [r for r in rows if r["arm"] != "transcript"]
    tv = [r["transcript"] for r in rows if r.get("transcript") is not None and not np.isnan(r["transcript"])]
    if tv:  # explicit transcript-only baseline row (difference 0 by definition; absolute value at the right)
        rows = [dict(arm="transcript", mean=float(np.mean(tv)), diff=0.0, lo=0.0, hi=0.0, n=rows[0].get("n"))] + rows
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    fmt = (lambda m: f"{k * m:.1f}%" if pct else f"{m:.3f}")
    for y, r in zip(ys, rows):
        a = r["arm"]; col = FAM_COLOR[family(a)]
        hollow = a in SECONDARY or a.endswith("shufc")
        if r.get("diff") is None or np.isnan(r["diff"]): continue
        if a == "transcript":
            ax.plot([0], [y], "D", ms=7, color=FAM_COLOR["transcript"], zorder=3)
            ax.text(1.02, y, fmt(r["mean"]) + "  (baseline)", transform=ax.get_yaxis_transform(), va="center", ha="left",
                    color=INK, fontsize=8, fontweight="bold")
            continue
        ax.plot([k * r["lo"], k * r["hi"]], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([k * r["diff"]], [y], "o", ms=7, mfc=SURF if hollow else col, mec=col, mew=2 if hollow else 1.5, zorder=3)
        txt = fmt(r["mean"]) + (f"  n={r['n']}" if show_n else "")
        ax.text(1.02, y, txt, transform=ax.get_yaxis_transform(), va="center", ha="left", color=INK2, fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL.get(r["arm"], r["arm"]) for r in rows] if show_labels else [""] * len(rows))
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=0, labelsize=8)
    ax.set_title(title, loc="left", fontsize=9, pad=16 if ref_text else 6)
    if ref_text: ax.text(0, 1.02, ref_text, transform=ax.transAxes, ha="left", va="bottom", color=INK2, fontsize=8)
    ax.set_xlabel(xlabel + (" (pp)" if pct else ""), color=MUTED, fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)

def save(fig, name, export=None):
    """Write report/fig/<name>.png and, if export is given, /workspace/agam/chive_graphs/<export>.png (atomic replace)."""
    fig.savefig(REP / "fig" / f"{name}.png", dpi=170, bbox_inches="tight", facecolor=SURF)
    if export:
        GRAPHS.mkdir(parents=True, exist_ok=True); tmp = GRAPHS / f".{export}.tmp.png"
        fig.savefig(tmp, dpi=170, bbox_inches="tight", facecolor=SURF); os.replace(tmp, GRAPHS / f"{export}.png")

def write_csv(name, rows, cols):
    import csv
    with open(REP / "data" / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows: w.writerow([("" if r.get(c) is None or (isinstance(r.get(c), float) and np.isnan(r.get(c))) else (round(r[c], 5) if isinstance(r[c], float) else r[c])) for c in cols])
