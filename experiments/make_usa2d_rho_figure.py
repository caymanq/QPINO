"""Figure 20 (report section 17): learned spatial growth rate rho(x,y), July forecasts and errors."""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import make_usa2d_figures as M      # re-uses the light theme, map helper and data (also rewrites fig18/19)
from src import cosine2d_galerkin as c2

plt, us_map, CMAP, ERR, mask, u, dates, te, day, FIG, MET, ACC = (M.plt, M.us_map, M.CMAP, M.ERR, M.mask, M.u, M.dates,
                                                               M.te, M.day, M.FIG, M.MET, M.ACC)
LX, LY = M.Lx / M.LREF, M.Ly / M.LREF
runs = defaultdict(list)
for f in glob.glob(os.path.join(MET, "usa2d_K3*_lam0.0001_R1_s[0-9].json")):
    d = json.load(open(f)); m = d["meta"]
    runs[(m.get("rho", "scalar"), m["model"])].append(d["result"] | {"seed": m["seed"]})
bS = {K: json.load(open(os.path.join(MET, f"baselines_usa2d_K{K}_rhoS4.json"))) for K in (3, 8)}
b3 = bS[3]["results"]


def med(key):
    rows = sorted(runs.get(key, []), key=lambda r: r["rmse_rollout"])
    return rows[len(rows) // 2] if rows else None


def grid(b, K=3):
    c2.setup(mask, K, LX, LY)
    return c2.reconstruct_grid(np.asarray(b))


def rho_grid(r, K):
    c2.setup(mask, K, LX, LY, Kr=4)
    out = np.full(mask.size, np.nan); out[mask.ravel() > 0] = c2.rho_field(r)
    return out.reshape(mask.shape)


rho3 = rho_grid(bS[3]["meta"]["fit"]["r"], 3)
rho8 = rho_grid(bS[8]["meta"]["fit"]["r"], 8)
rinit = rho_grid(bS[3]["meta"]["fit"]["r_local_growth_init"], 3)
RHO = M.LinearSegmentedColormap.from_list("rho", ["#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"]); RHO.set_bad((0, 0, 0, 0))
d26, i26 = day["2020-07-26"], int(np.where(dates == pd.Timestamp("2020-07-26"))[0][0])
Ds, Dr = med(("scalar", "D")), med(("spatial", "D"))
fig = plt.figure(figsize=(22, 8.4))
gs = fig.add_gridspec(2, 8, width_ratios=[1, 1, 1, 1, 1, 0.04, 0.5, 1.1], wspace=0.08, hspace=0.42)
im_r = us_map(fig.add_subplot(gs[0, 0]), rho3, f"learned ρ(x,y), 3×3 A-ρ\n[{np.nanmin(rho3):.4f}, {np.nanmax(rho3):.4f}] /day",
              cmap=RHO, vmin=0, vmax=max(0.02, np.nanmax(rho3)))
a3 = fig.axes[-1]
fig.colorbar(im_r, cax=a3.inset_axes([0.15, -0.1, 0.7, 0.05]), orientation="horizontal").set_label("ρ (1/day)")
v8 = np.nanmax(np.abs(rho8))
a8 = fig.add_subplot(gs[0, 1])
im8 = us_map(a8, rho8, f"ρ(x,y), 8×8 classical, β→0\n[{np.nanmin(rho8):.3f}, {np.nanmax(rho8):.3f}] /day", cmap=ERR, vmin=-v8, vmax=v8)
fig.colorbar(im8, cax=a8.inset_axes([0.15, -0.1, 0.7, 0.05]), orientation="horizontal").set_label("ρ (1/day)")
im = us_map(fig.add_subplot(gs[0, 2]), u[i26], "observed Jul 26")
if Dr:
    us_map(fig.add_subplot(gs[0, 3]), grid(Dr["pred_rollout"])[d26], f"QPINO D, ρ(x,y), Jul 26\n(seed {Dr['seed']}, RMSE {Dr['rollout_rmse_by_day'][d26]:.3f})")
us_map(fig.add_subplot(gs[0, 4]), grid(b3["A"]["pred_rollout"])[d26], f"Galerkin A-ρ(x,y), Jul 26\n(RMSE {b3['A']['rollout_rmse_by_day'][d26]:.3f})")
fig.colorbar(im, cax=fig.add_subplot(gs[0, 5])).set_label("u")
errs = []
if Ds:
    errs.append((Ds["pred_rollout"], f"error QPINO D, scalar ρ\n(seed {Ds['seed']}, λ=1e-4)"))
if Dr:
    errs.append((Dr["pred_rollout"], f"error QPINO D, ρ(x,y)\n(seed {Dr['seed']}, λ=1e-4)"))
errs += [(b3["A-scalar-sameprotocol"]["pred_rollout"], "error Galerkin A, scalar ρ\n(Jul 26)"), (b3["A"]["pred_rollout"], "error Galerkin A-ρ(x,y)\n(Jul 26)")]
floor = grid(b3["Projection-floor"]["coeffs_july"])[d26]
errs.append((None, "error, 3×3 truncation floor\n(Jul 26)"))
for k, (P, lab) in enumerate(errs[:5]):
    f = floor if P is None else grid(P)[d26]
    ime = us_map(fig.add_subplot(gs[1, k]), f - u[i26], lab, cmap=ERR, vmin=-0.4, vmax=0.4)
fig.colorbar(ime, cax=fig.add_subplot(gs[1, 5])).set_label("forecast − observed (u)")
ax = fig.add_subplot(gs[:, 7]); days = np.arange(1, len(te) + 1)
ax.plot(days, b3["Persistence"]["rollout_rmse_by_day"], color="0.6", label="persistence")
ax.plot(days, b3["LinearTrend"]["rollout_rmse_by_day"], color="0.35", label="linear trend")
ax.plot(days, b3["A-scalar-sameprotocol"]["rollout_rmse_by_day"], color="k", ls="-.", label="A, scalar ρ")
ax.plot(days, b3["A"]["rollout_rmse_by_day"], color="k", label="A-ρ(x,y)")
ax.plot(days, b3["Projection-floor"]["rollout_rmse_by_day"], "--", color="k", lw=0.8, label="3×3 floor")
ax.plot(days, bS[8]["results"]["A-scalar-sameprotocol"]["rollout_rmse_by_day"], color="tab:brown", lw=1, label="8×8 classical A, scalar ρ")
ax.plot(days, bS[8]["results"]["Projection-floor"]["rollout_rmse_by_day"], "--", color="tab:brown", lw=0.8, label="8×8 floor")
for (rk, nm), c in ((("scalar", "D"), "tab:blue"), (("spatial", "D"), "tab:cyan"), (("scalar", "B"), "tab:orange"), (("spatial", "B"), "tab:red")):
    rows = runs.get((rk, nm), [])
    if rows:
        a = np.array([q["rollout_rmse_by_day"] for q in rows])
        ax.plot(days, a.mean(0), color=c, lw=1.2, label=f"{nm}, {'ρ(x,y)' if rk == 'spatial' else 'scalar ρ'} (n={len(rows)})")
        ax.fill_between(days, a.min(0), a.max(0), color=c, alpha=0.12)
ax.set_ylim(0.07, 0.2); ax.set_xlabel("July day"); ax.set_ylabel("rollout RMSE in u (land cells)")
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.08)); ax.set_title("July rollout error (λ_PDE = 1e-4)", fontsize=12)
fig.suptitle("Spatially varying growth rate ρ(x,y) in u_t = D∇²u + ρ(x,y)u − βu²: learned ρ, Jul 26 forecasts (rollout from Jun 30) and errors",
             fontsize=14, color=ACC)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig20_usa2d_spatial_rho.{e}"), dpi=150, bbox_inches="tight", facecolor="white")
print("wrote fig20")
