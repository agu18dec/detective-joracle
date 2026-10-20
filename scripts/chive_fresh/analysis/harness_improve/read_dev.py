"""Dev read: each experimental arm vs its baseline arm on dev240 (fresh1+2), draws d0-d2, paired by incident.
Bad (salvaged/error) records dropped; an incident counts when both arms have >= 2 good judged draws."""
import json, glob, collections, sys
import numpy as np
sys.path.insert(0, "/workspace/agam/chive_scale/report")
from common import boot
C = "/workspace/agam/chive_scale"
dev = {(b, p) for b in ("fresh1", "fresh2") for p in open(f"{C}/analysis/harness_improve/dev/{b}.txt").read().split()}
BAD = []
def load(arm):
    d = collections.defaultdict(list)
    for b in ("fresh1", "fresh2"):
        for dr in range(3):
            for f in glob.glob(f"{C}/harness_runs/{b}/{arm}/*.d{dr}.json"):
                p = f.split("/")[-1].split(".d")[0]
                if (b, p) not in dev: continue
                try: r = json.load(open(f))
                except Exception: BAD.append(f); continue
                j = r.get("judge") or {}
                if str(r.get("stopped_by", "")).startswith(("error", "salv")) or j.get("score") is None: continue
                d[(b, p)].append(j["score"] >= 7)
    return d
A = {a: load(a) for a in ["transcript", "olens_rl_B", "mlnla_B", "olens_single_k16", "olens_recur", "mlnla_recur", "olens_s16_recur", "olens_filt", "mlnla_filt", "olens_mc", "mlnla_mc", "olens_s16_mcr"]}
def cmp(a, b):
    ks = [k for k in dev if len(A[a].get(k, [])) >= 2 and len(A[b].get(k, [])) >= 2]
    m, r, d, lo, hi = boot([np.mean(A[a][k]) for k in ks], [np.mean(A[b][k]) for k in ks])
    print(f"{a:16s} {100*m:5.1f} vs {b:16s} {100*r:5.1f}  Δ {100*d:+5.1f} [{100*lo:+.1f},{100*hi:+.1f}]  n={len(ks)}")
print("== variant vs its own baseline"); [cmp(a, b) for a, b in [("olens_recur", "olens_rl_B"), ("mlnla_recur", "mlnla_B"), ("olens_s16_recur", "olens_single_k16"), ("olens_filt", "olens_rl_B"), ("mlnla_filt", "mlnla_B"), ("olens_mc", "olens_rl_B"), ("mlnla_mc", "mlnla_B"), ("olens_s16_mcr", "olens_single_k16"), ("olens_s16_mcr", "olens_s16_recur")]]
print("== OLens variants vs NLA"); [cmp(a, b) for a, b in [("olens_rl_B", "mlnla_B"), ("olens_recur", "mlnla_recur"), ("olens_s16_recur", "mlnla_recur"), ("olens_recur", "mlnla_B"), ("olens_s16_recur", "mlnla_B"), ("olens_filt", "mlnla_filt"), ("olens_filt", "mlnla_B"), ("olens_mc", "mlnla_mc"), ("olens_mc", "mlnla_B"), ("olens_s16_mcr", "mlnla_mc"), ("olens_s16_mcr", "mlnla_B")]]
print("== vs transcript"); [cmp(a, "transcript") for a in ["olens_rl_B", "mlnla_B", "olens_recur", "olens_s16_recur", "olens_filt", "olens_mc", "mlnla_mc", "olens_s16_mcr"]]

print("unreadable files:", len(BAD), BAD[:3])
