"""Figures for the M=4 / mirror / data-dominated-loss / re-uploading study (fig04d, fig08b, fig13-16)."""
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
from src import galerkin as gk, preprocessing as pp  # noqa: E402
from src.custom_ansatz import get_graph  # noqa: E402

MET, FIG = os.path.join(ROOT, "results", "metrics"), os.path.join(ROOT, "results", "figures")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.fontsize": 7, "legend.frameon": False})
runs = defaultdict(list)
for f in glob.glob(os.path.join(MET, "m4", "*_s[0-9].json")):
    d = json.load(open(f)); m = d["meta"]
    runs[(m["regime"], m["M"], m["boundary"], m["model"], m["lambda_pde"], m["R"])].append(d["result"])
base = {os.path.basename(f)[10:-5]: json.load(open(f))["results"] for f in glob.glob(os.path.join(MET, "m4", "baselines_*.json"))}
trunc = json.load(open(os.path.join(MET, "m4_truncation.json")))
old_c = json.load(open(os.path.join(MET, "covid_results.json")))
old_s = json.load(open(os.path.join(MET, "synthetic_results.json")))
Z = np.load(os.path.join(ROOT, "data", "processed", "covid_i10_field.npz"))
u, x, S = Z["u"], Z["x"], float(Z["scale_S"])
dates = pd.to_datetime(Z["dates"])
N = u.shape[1]
COL = dict(Persistence="0.55", LinearTrend="0.35", A="k", B="tab:orange", C="tab:green", D="tab:blue",
           E="tab:purple", F="tab:red")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def stat(key, metric):
    rows = runs.get(key, [])
    if not rows:
        return np.nan, np.nan, 0
    a = np.array([r[metric] for r in rows], float)
    return a.mean(), a.std(), len(a)


# ---------------------------------------------------------------- fig04d: M=4 coupling graphs
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
names = gk.names(4)
ang = np.pi / 2 - 2 * np.pi * np.arange(9) / 9
pos = np.c_[np.cos(ang), np.sin(ang)]
for ax, g in zip(axs, ["structured", "triad", "nearest"]):
    E = get_graph(g, 9)
    full = set(__import__("itertools").combinations(range(9), 2))
    for j, k in E:
        ax.plot(*pos[[j, k]].T, "-", color="#555", lw=0.8, zorder=1)
    if g == "structured":
        for j, k in sorted(full - set(E)):
            ax.plot(*pos[[j, k]].T, ":", color="tab:red", lw=1.6, zorder=1)
    for j in range(9):
        ax.scatter(*pos[j], s=650, color="tab:blue" if j else "k", zorder=3)
        ax.text(*pos[j], names[j].replace(" ", "\n"), ha="center", va="center", color="w", fontsize=6.5, zorder=4)
    ax.set_title(f"{g}: {len(E)} of 36 ZZ edges" + (" (red dotted = absent)" if g == "structured" else ""))
    ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("Fig. 4d  M=4 (9-qubit) coupling graphs from the Galerkin Hessian: only (Re a3, Im a3) and "
             "(Re a4, Im a4) never multiply", fontsize=10)
save(fig, "fig04d_coupling_graphs_m4")

# ---------------------------------------------------------------- fig13: truncation / boundary
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
ax = axs[0]
Ms = [1, 2, 4, 8]
for i, (b, c) in enumerate([("periodic", "tab:blue"), ("mirror", "tab:orange")]):
    for j, sp in enumerate(["train", "july"]):
        v = [trunc[f"M{m}_{b}"][sp] for m in Ms]
        ax.plot(Ms, v, ("o-" if sp == "train" else "s--"), color=c, label=f"{b}, {sp}")
ax.set_xscale("log", base=2); ax.set_xticks(Ms); ax.set_xticklabels(Ms)
ax.set_xlabel("truncation M"); ax.set_ylabel("mean rel. L2 truncation error (32 bins)"); ax.legend()
ax.set_title("projection error vs M and boundary")
ax = axs[1]
um, xm = pp.mirror_extend(u), gk.uniform_grid(2 * N, 0.5)
for Mk, b, c, ls in [(2, "periodic", "tab:blue", "-"), (4, "periodic", "tab:blue", "--"),
                     (2, "mirror", "tab:orange", "-"), (4, "mirror", "tab:orange", "--")]:
    if b == "mirror":
        rec = gk.reconstruct(gk.field_to_coeffs(um, xm, Mk), xm)[:, :N]
    else:
        rec = gk.reconstruct(gk.field_to_coeffs(u, x, Mk), x)
    e = np.linalg.norm(u - rec, axis=1) / (np.linalg.norm(u, axis=1) + 1e-12)
    ax.plot(dates[7:], e[7:], ls, color=c, lw=1, label=f"M={Mk} {b}")
ax.axvline(pd.Timestamp("2020-06-17"), color="k", lw=0.6); ax.axvline(pd.Timestamp("2020-07-01"), color="k", lw=0.6)
ax.set_ylim(0, 1.05); ax.set_ylabel("rel. L2 truncation error"); ax.legend(ncol=2)
ax.set_title("over time (lines: val / test start)"); ax.tick_params(axis="x", rotation=30)
ax = axs[2]
i15 = int(np.where(dates == pd.Timestamp("2020-07-15"))[0][0])
s_km = Z["s_km"]
ax.plot(s_km, u[i15] * S, "k.-", lw=1, label="observed (7-day mean)")
for Mk, b, c, ls in [(2, "periodic", "tab:blue", "-"), (4, "periodic", "tab:cyan", "-"), (4, "mirror", "tab:orange", "--")]:
    if b == "mirror":
        rec = gk.reconstruct(gk.field_to_coeffs(um[i15], xm, Mk), xm)[:N]
    else:
        rec = gk.reconstruct(gk.field_to_coeffs(u[i15], x, Mk), x)
    ax.plot(s_km, rec * S, ls, color=c, label=f"M={Mk} {b}")
ax.set_xlabel("distance along I-10 from Santa Monica (km)"); ax.set_ylabel("cases / 100k / day")
ax.set_title("2020-07-15 field and its projections"); ax.legend()
fig.suptitle("Fig. 13  Periodic vs mirrored (even, domain 2L) boundary for the I-10 transect, M=2 and M=4", fontsize=10)
save(fig, "fig13_truncation_boundary_m4")

# ---------------------------------------------------------------- fig14: COVID July M=4 mirror
fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
ax = axs[0]
days = np.arange(1, 32)
bm = base["covid_M4_mirror"]
for nm in ["Persistence", "LinearTrend", "A"]:
    ax.plot(days, bm[nm]["rollout_field_rmse_by_day"], color=COL[nm], lw=1.6, label=f"{nm} (M=4 mirror)")
for nm in ["B", "C", "D", "E", "F"]:
    rows = runs.get(("covid", 4, "mirror", nm, 0.0 if nm == "E" else 0.01, 1), [])
    if rows:
        a = np.array([r["rollout_field_rmse_by_day"] for r in rows])
        ax.plot(days, a.mean(0), color=COL[nm], lw=1.2, label=f"{nm} (M=4 mirror, n={len(rows)})")
        ax.fill_between(days, a.min(0), a.max(0), color=COL[nm], alpha=0.12)
od = old_c["per_seed"]["D"]["seeds"]
a = np.array([r["rollout_field_rmse_by_step_vs_observed"] for r in od]) * S
ax.plot(days, a.mean(0), ":", color="tab:blue", lw=1.5, label="D old (M=2 periodic, λ=0.1)")
ax.axhline(bm["Projection-floor"]["field_rmse_per100k_rollout"], color="k", ls="--", lw=0.7, label="M=4 mirror projection floor")
ax.set_xlabel("July day (rollout horizon)"); ax.set_ylabel("field RMSE (cases/100k/day)"); ax.legend(fontsize=6.3, ncol=2)
ax.set_title("31-day rollout error from the observed Jun 30 state")
te = np.where((dates >= "2020-07-01") & (dates <= "2020-07-31"))[0]
xo = xm[:N]
rowsD = runs.get(("covid", 4, "mirror", "D", 0.01, 1), [])
ext = [1, 31, s_km[0], s_km[-1]]
vmax = np.percentile(u[te] * S, 99)
axs[1].imshow((u[te] * S).T, aspect="auto", origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="magma")
axs[1].set_title("observed July field"); axs[1].set_xlabel("July day"); axs[1].set_ylabel("km along I-10")
if rowsD and "pred_rollout" in rowsD[0]:
    f2 = gk.reconstruct(np.array(rowsD[0]["pred_rollout"]), xo) * S
    im = axs[2].imshow(f2.T, aspect="auto", origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="magma")
    plt.colorbar(im, ax=axs[2], label="cases/100k/day")
axs[2].set_title("QPINO D rollout (M=4 mirror, seed 0)"); axs[2].set_xlabel("July day")
fig.suptitle("Fig. 14  JHU July 2020, M=4 mirrored boundary, data-dominated loss (λ_PDE=0.01)", fontsize=10)
save(fig, "fig14_covid_m4_mirror")

# ---------------------------------------------------------------- fig08b: synthetic M=4 rollouts
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
ax = axs[0]
t = 0.1 * np.arange(1, 41)
for nm in ["B", "C", "D", "E", "F"]:
    rows = runs.get(("synthetic", 4, "periodic", nm, 0.0 if nm == "E" else 0.01, 1), [])
    if rows:
        a = np.array([r["rollout_rel_by_step"] for r in rows])
        ax.semilogy(t, a.mean(0), color=COL[nm], label=f"{nm} M=4 (n={len(rows)})")
    o = old_s["per_seed"].get(nm, {}).get("seeds", [])
    if o:
        ax.semilogy(t, np.mean([r["rollout_rel_by_step"] for r in o], 0), ":", color=COL[nm], lw=1, label=f"{nm} M=2 old")
ax.semilogy(t, base["synthetic_M4_periodic"]["Persistence"]["rollout_rel_by_step"], color="0.55", label="persistence")
ax.set_xlabel("rollout time t"); ax.set_ylabel("relative coefficient error"); ax.legend(ncol=2, fontsize=6.3)
ax.set_title("synthetic test rollouts (12 unseen ICs)")
ax = axs[1]
nmz = ["B", "C", "D", "E", "F"]
w = 0.38
for i, (Mk, lab) in enumerate([(2, "M=2 (old, λ=0.1)"), (4, "M=4 (new, λ=0.01)")]):
    vals, errs = [], []
    for nm in nmz:
        if Mk == 2:
            s = old_s["summary"][nm]["rollout_rel_l2"]; vals.append(s["mean"]); errs.append(s["std"])
        else:
            m_, s_, _ = stat(("synthetic", 4, "periodic", nm, 0.0 if nm == "E" else 0.01, 1), "rollout_rel_l2")
            vals.append(m_); errs.append(s_)
    ax.bar(np.arange(5) + (i - 0.5) * w, vals, w, yerr=errs, label=lab, color=["tab:gray", "tab:blue"][i], capsize=2)
ax.set_xticks(range(5)); ax.set_xticklabels(nmz); ax.set_yscale("log"); ax.set_ylabel("40-step rollout rel. L2")
ax.legend(); ax.set_title("old vs new truncation")
fig.suptitle("Fig. 8b  Synthetic Fisher–KPP at M=4 (9 qubits)", fontsize=10)
save(fig, "fig08b_synthetic_m4")

# ---------------------------------------------------------------- fig15: re-uploading sweep
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
Rs = [1, 2, 3]
for reg, b, ax, met, lab in [("synthetic", "periodic", axs[0], "rollout_rel_l2", "40-step rollout rel. L2"),
                              ("covid", "mirror", axs[1], "field_rmse_per100k_rollout", "July rollout field RMSE")]:
    for Mk, c in [(2, "tab:gray"), (4, "tab:blue")]:
        st = [stat((reg, Mk, b, "D", 0.01, R), met) for R in Rs]
        ax.errorbar(Rs, [s[0] for s in st], yerr=[s[1] for s in st], fmt="o-", color=c, capsize=3, label=f"M={Mk}")
    if reg == "covid":
        ax.axhline(base["covid_M4_mirror"]["Persistence"]["field_rmse_per100k_rollout"], color="0.55", ls="--", label="persistence (M=4 mirror)")
        ax.axhline(base["covid_M4_mirror"]["A"]["field_rmse_per100k_rollout"], color="k", ls="--", label="Galerkin ODE A (M=4 mirror)")
    ax.set_xticks(Rs); ax.set_xlabel("re-uploads R"); ax.set_ylabel(lab); ax.legend(); ax.set_title(f"{reg} ({b})")
ax = axs[2]
for reg, b, mk in [("synthetic", "periodic", "o-"), ("covid", "mirror", "s--")]:
    for Mk, c in [(2, "tab:gray"), (4, "tab:blue")]:
        st = [stat((reg, Mk, b, "D", 0.01, R), "sec_per_epoch") for R in Rs]
        ax.plot(Rs, [s[0] for s in st], mk, color=c, label=f"{reg} M={Mk}")
ax.set_yscale("log"); ax.set_xticks(Rs); ax.set_xlabel("re-uploads R"); ax.set_ylabel("s / epoch (1–2 threads, shared box)")
ax.legend(); ax.set_title("training cost per R")
fig.suptitle("Fig. 15  Data re-uploading W_R S(z)…W_1 S(z), λ_PDE=0.01, 3 seeds (synthetic M=4 R=1,3: 2 seeds)", fontsize=10)
save(fig, "fig15_reupload_sweep")

# ---------------------------------------------------------------- fig16: lambda sweep
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
lams = [0.0, 0.001, 0.01]
for ax, (reg, b, met, lab) in zip(axs, [("synthetic", "periodic", "rollout_rel_l2", "40-step rollout rel. L2"),
                                        ("covid", "mirror", "field_rmse_per100k_rollout", "July rollout field RMSE")]):
    st = [stat((reg, 4, b, "E" if l == 0 else "D", l, 1), met) for l in lams]
    ax.errorbar(range(3), [s[0] for s in st], yerr=[s[1] for s in st], fmt="o-", capsize=3, color="tab:blue", label="QPINO M=4")
    st1 = [stat((reg, 4, b, "E" if l == 0 else "D", l, 1), "rel_l2_increment") for l in lams]
    ax2 = ax.twinx(); ax2.plot(range(3), [s[0] for s in st1], "s--", color="tab:red", label="1-step incr. err")
    ax2.set_ylabel("1-step increment error", color="tab:red")
    ax.set_xticks(range(3)); ax.set_xticklabels([str(l) for l in lams]); ax.set_xlabel("λ_PDE")
    ax.set_ylabel(lab); ax.set_title(f"{reg} M=4 {b}")
fig.suptitle("Fig. 16  λ_PDE sweep (data-dominated loss; λ=0 is model E)", fontsize=10)
fig.tight_layout()
save(fig, "fig16_lambda_sweep_m4")
