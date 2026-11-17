"""Fig. 17: real cosine / Neumann basis (default model) — truncation/basis comparison and COVID July forecasts."""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import cosine_galerkin as cg  # noqa: E402

MET, FIG = os.path.join(ROOT, "results", "metrics"), os.path.join(ROOT, "results", "figures")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.fontsize": 6.5, "legend.frameon": False})
runs = defaultdict(list)
for f in glob.glob(os.path.join(MET, "m4", "*_s[0-9].json")):
    d = json.load(open(f)); m = d["meta"]
    runs[(m["regime"], m["M"], m["boundary"], m["model"], m["lambda_pde"], m["R"])].append(d["result"])
base = {os.path.basename(f)[10:-5]: json.load(open(f))["results"] for f in glob.glob(os.path.join(MET, "m4", "baselines_*.json"))}
trunc = json.load(open(os.path.join(MET, "m4_truncation.json")))
old_c = json.load(open(os.path.join(MET, "covid_results.json")))
Z = np.load(os.path.join(ROOT, "data", "processed", "covid_i10_field.npz"))
u, S, s_km = Z["u"], float(Z["scale_S"]), Z["s_km"]
dates = pd.to_datetime(Z["dates"])
N = u.shape[1]
COL = dict(Persistence="0.55", LinearTrend="0.35", A="k", B="tab:orange", C="tab:green", D="tab:blue",
           E="tab:purple", F="tab:red")
lam = lambda nm: 0.0 if nm == "E" else 0.01  # noqa: E731

fig, axs = plt.subplots(2, 3, figsize=(16, 8.4))
# (a) truncation error vs qubits
ax = axs[0, 0]
for lab, keys, q, st in [("cosine (Neumann, real)", ["cosine_M2", "cosine_M4", "cosine_M8", "cosine_M16"], [3, 5, 9, 17], "o-"),
                         ("complex Fourier, periodic", ["M1_periodic", "M2_periodic", "M4_periodic", "M8_periodic"], [3, 5, 9, 17], "s--"),
                         ("complex Fourier, mirrored", ["M1_mirror", "M2_mirror", "M4_mirror", "M8_mirror"], [3, 5, 9, 17], "^:")]:
    for per, a in (("train", 1.0), ("july", 0.55)):
        ax.plot(q, [trunc[k][per] for k in keys], st, alpha=a, label=f"{lab}, {per}")
ax.set_xscale("log", base=2); ax.set_xticks([3, 5, 9, 17]); ax.set_xticklabels(["3", "5", "9", "17"])
ax.set_xlabel("qubits (= real inputs)"); ax.set_ylabel("mean rel. L2 truncation error"); ax.legend(fontsize=5.8)
ax.set_title("(a) I-10 field truncation error vs qubit budget")
# (b) rollout RMSE by day, cosine M=8
ax = axs[0, 1]
days = np.arange(1, 32)
bc = base["covid_M8_cosine"]
for nm in ["Persistence", "LinearTrend", "A"]:
    ax.plot(days, bc[nm]["rollout_field_rmse_by_day"], color=COL[nm], lw=1.6, label=nm)
for nm in ["B", "C", "D", "E", "F"]:
    rows = runs.get(("covid", 8, "cosine", nm, lam(nm), 1), [])
    if rows:
        a = np.array([r["rollout_field_rmse_by_day"] for r in rows])
        ax.plot(days, a.mean(0), color=COL[nm], lw=1.2, label=f"{nm} (n={len(rows)})")
        ax.fill_between(days, a.min(0), a.max(0), color=COL[nm], alpha=0.12)
a = np.array([r["rollout_field_rmse_by_step_vs_observed"] for r in old_c["per_seed"]["D"]["seeds"]]) * S
ax.plot(days, a.mean(0), ":", color="tab:blue", lw=1.4, label="D, earlier M=2 complex Fourier")
ax.axhline(bc["Projection-floor"]["field_rmse_per100k_rollout"], color="k", ls="--", lw=0.7, label="cosine M=8 projection floor")
ax.set_xlabel("July day"); ax.set_ylabel("field RMSE (cases/100k/day)"); ax.legend(ncol=2, fontsize=5.8)
ax.set_title("(b) COVID 31-day rollout, cosine M=8 (9 qubits)")
# (c) July rollout RMSE: cosine M=8 vs cosine M=4 vs earlier variant
ax = axs[0, 2]
nms = ["Persistence", "LinearTrend", "A", "B", "C", "D", "E", "F"]
series = [("cosine M=8 (9q)", 8, "cosine"), ("cosine M=4 (5q)", 4, "cosine"), ("complex M=4 mirror (9q)", 4, "mirror"), ("complex M=2 periodic (5q)", 2, "periodic")]
w = 0.2
for i, (lab, M_, b) in enumerate(series):
    v, e = [], []
    for nm in nms:
        if nm in ("Persistence", "LinearTrend", "A"):
            v.append(base[f"covid_M{M_}_{b}"][nm]["field_rmse_per100k_rollout"]); e.append(0)
        else:
            rows = runs.get(("covid", M_, b, nm, lam(nm), 1), [])
            if not rows and b == "periodic" and nm in old_c["summary"]:
                s_ = old_c["summary"][nm]["field_rmse_per100k_rollout"]; v.append(s_["mean"]); e.append(s_["std"])
            elif rows:
                r = np.array([q["field_rmse_per100k_rollout"] for q in rows]); v.append(r.mean()); e.append(r.std())
            else:
                v.append(np.nan); e.append(0)
    ax.bar(np.arange(len(nms)) + (i - 1.5) * w, v, w, yerr=e, capsize=2, label=lab)
ax.set_xticks(range(len(nms))); ax.set_xticklabels(["Pers.", "Lin.", "A", "B", "C", "D", "E", "F"])
ax.set_ylim(10, None); ax.set_ylabel("July rollout field RMSE"); ax.legend(fontsize=6)
ax.set_title("(c) July rollout: real cosine vs earlier complex Fourier\n(M=2 periodic learned models: earlier λ=0.1 runs)")
# (d)-(f) forecast fields
te = np.where((dates >= "2020-07-01") & (dates <= "2020-07-31"))[0]
xc = cg.grid(N)
ext = [1, 31, s_km[0], s_km[-1]]
vmax = np.percentile(u[te] * S, 99)
im = axs[1, 0].imshow((u[te] * S).T, aspect="auto", origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="magma")
axs[1, 0].set_title("(d) observed July 2020 field"); axs[1, 0].set_ylabel("km along I-10 (LA → Jacksonville)")
for ax, nm, ttl in [(axs[1, 1], "D", "(e) QPINO D rollout, cosine M=8, seed 0"), (axs[1, 2], "A", "(f) Galerkin ODE A rollout, cosine M=8")]:
    pr = None
    if nm == "A":
        pr = bc["A"].get("pred_rollout")
    else:
        rows = sorted(runs.get(("covid", 8, "cosine", nm, 0.01, 1), []), key=lambda r: r.get("seed", 0))
        pr = rows[0].get("pred_rollout") if rows else None
    if pr is not None:
        ax.imshow(cg.reconstruct(np.array(pr), xc).T * S, aspect="auto", origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="magma")
    else:
        ax.text(0.5, 0.5, "not available", ha="center", transform=ax.transAxes)
    ax.set_title(ttl)
for ax in axs[1]:
    ax.set_xlabel("July day")
plt.colorbar(im, ax=axs[1].tolist(), label="cases/100k/day", shrink=0.85)
fig.suptitle("Fig. 17  Real cosine / Neumann Galerkin basis (default model): one real coefficient per qubit, λ_PDE=0.01", fontsize=10)
for ext_ in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig17_cosine_basis.{ext_}"), dpi=160, bbox_inches="tight")
print("wrote fig17_cosine_basis")
