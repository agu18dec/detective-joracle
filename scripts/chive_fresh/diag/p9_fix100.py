"""Fix test (Agam 2026-10-21): 100 dev incidents (analysis/diag/fix100) x draws 0-2, frozen harness, no experiments.
pass@1 (judge mean >= 7) and refuted-rival assertion (judge v3, final hypothesis), paired incident bootstrap.
  fixes 2/3 (full banks): olens_sup / olens_rr / olens_suprr vs olens_rl_B; mlnla_suprr vs mlnla_B; OLens vs NLA with both
  fix 1 (sparse 223k-cell banks, frozen prompt): omp32_sub223 / osyn8_sub223 vs olens_sub223 (same cells), and the
  coverage cost olens_sub223 vs olens_rl_B
usage: python p9_fix100.py judge   (judge v3 on the new arms)   |   python p9_fix100.py report
"""
import json, sys, subprocess, collections, os
sys.path.insert(0, "/workspace/agam/chive_scale/report"); sys.path.insert(0, "/workspace/agam/chive_scale/analysis/diag")
from common import boot
from dcommon import *
IDS = {b: open(C / "analysis/diag/fix100" / f"{b}.txt").read().split() for b in ("fresh1", "fresh2")}
NEW = ["olens_sup", "olens_rr", "olens_suprr", "mlnla_suprr", "olens_sub223", "omp32_sub223", "osyn8_sub223", "qsynomp_sub223"]
D = range(3)

def passes(arm):
    out = collections.defaultdict(dict)
    for b, ids in IDS.items():
        for pid in ids:
            for d in D:
                f = C / "harness_runs" / b / arm / f"{pid}.d{d}.json"
                if not f.exists(): continue
                try: r = json.loads(f.read_text())
                except Exception: continue
                if str(r.get("stopped_by", "")).startswith(("error", "salvaged")) or not r.get("judge") or r["judge"].get("score") is None: continue
                out[(b, pid)][d] = float(r["judge"]["score"] >= 7)
    return out

def rivals(arm):
    out = collections.defaultdict(dict)
    for b, ids in IDS.items():
        for pid in ids:
            for d in D:
                f = C / "diag" / "judge3" / b / "cls_final" / arm / f"{pid}.d{d}.json"
                if not f.exists(): continue
                j = json.loads(f.read_text())
                if not j.get("nrv"): continue
                rv = {str(x.get("match")) for x in j.get("labels", []) if str(x.get("label", "")).upper() == "RIVAL"}
                out[(b, pid)][d] = len(rv) / j["nrv"]
    return out

def paired(A, B):
    ks = [k for k in A if k in B and set(A[k]) & set(B[k])]
    a = [sum(A[k][d] for d in set(A[k]) & set(B[k])) / len(set(A[k]) & set(B[k])) for k in ks]
    b = [sum(B[k][d] for d in set(A[k]) & set(B[k])) / len(set(A[k]) & set(B[k])) for k in ks]
    return ks, a, b

def judge():
    env = dict(os.environ, J3_THREADS="8")
    for b in IDS:
        arms = [a for a in NEW if (C / "harness_runs" / b / a).exists()]
        subprocess.run(["/workspace/agam/chive/.venv/bin/python", "-u", str(C / "analysis/diag/judge3.py"), b, "--arms", ",".join(arms),
                        "--ids", str(C / "analysis/diag/fix100" / f"{b}.txt"), "--draws", "0,1,2", "--which", "final"], env=env, check=False)

def report():
    pairs = [("olens_sup", "olens_rl_B", "fix 2: support counts"), ("olens_rr", "olens_rl_B", "fix 3: rival-role prompt"),
             ("olens_suprr", "olens_rl_B", "fixes 2+3, OLens"), ("mlnla_suprr", "mlnla_B", "fixes 2+3, NLA"),
             ("olens_suprr", "mlnla_suprr", "OLens - NLA, both with fixes 2+3"), ("olens_rl_B", "mlnla_B", "OLens - NLA, plain (reference)"),
             ("olens_suprr", "mlnla_B", "OLens + fixes 2+3 vs plain NLA"),
             ("olens_sub223", "olens_rl_B", "coverage cost: OLens on 223k cells vs full"),
             ("omp32_sub223", "olens_sub223", "fix 1: AR-picks-4-of-32 vs plain (same cells)"),
             ("osyn8_sub223", "olens_sub223", "fix 1: Opus synthesis-of-8 vs plain (same cells)"),
             ("qsynomp_sub223", "olens_sub223", "fix 1: Qwen synth x4 -> NN-OMP vs plain (same cells)"),
             ("qsynomp_sub223", "mlnla_B", "Qwen synth x4 -> NN-OMP (sparse) vs plain NLA (full)"),
             ("omp32_sub223", "mlnla_B", "AR-picks-4-of-32 (sparse) vs plain NLA (full)"),
             ("osyn8_sub223", "mlnla_B", "Opus synthesis-of-8 (sparse) vs plain NLA (full)")]
    rows = []
    for a, r, lab in pairs:
        line = f"{lab:50s}"
        for met, fn in (("pass@1", passes), ("rivals", rivals)):
            ks, va, vb = paired(fn(a), fn(r))
            if not ks: line += f" | {met}: no data"; continue
            m1, m2, d, lo, hi = boot(va, vb)
            rows.append(dict(label=lab, metric=met, arm=a, ref=r, n=len(ks), arm_mean=m1, ref_mean=m2, diff=d, lo=lo, hi=hi))
            line += f" | {met} n={len(ks)} {100 * m1:.1f} vs {100 * m2:.1f} d={100 * d:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}]"
        print(line)
    write_rows("p9_fix100.csv", rows)

if __name__ == "__main__":
    judge() if sys.argv[1] == "judge" else report()
