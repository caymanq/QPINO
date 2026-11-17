"""T15 table + fig21 for the regional domain-decomposition QPINO (report section 18)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import make_usa2d_figures as M
from src.regional_galerkin import Decomposition

plt, us_map, ERR, mask, u, dates, te, day, FIG, ACC = M.plt, M.us_map, M.ERR, M.mask, M.u, M.dates, M.te, M.day, M.FIG, M.ACC
ROOT = M.ROOT
OUT = os.path.join(ROOT, "results", "metrics", "usa2d_regional")
S16 = os.path.join(ROOT, "results", "metrics", "usa2d")
RG = np.load(os.path.join(ROOT, "data", "processed", "usa_regions_256x128.npz"))
lab, names = RG["label"].astype(int), [str(s) for s in RG["names"]]
Z = M.Z
NY, NX = mask.shape
LREF = NY * float(Z["dy_km"]) / np.pi
DD = Decomposition(lab, 3, float(Z["dx_km"]) / LREF, float(Z["dy_km"]) / LREF, names)
land = mask.ravel() > 0
bR = json.load(open(os.path.join(OUT, "baselines_regional_K3.json")))
res, meta = bR["results"], bR["meta"]
asm = defaultdict(list)
for f in sorted(glob.glob(os.path.join(OUT, "assembled_K3_*_s[0-9].json"))):
    d = json.load(open(f)); asm[d["meta"]["model"]].append(d["results"] | {"seed": d["meta"]["seed"]})
single = defaultdict(list)
for f in glob.glob(os.path.join(S16, "usa2d_K3_[A-Z]_lam0.0001_R1_s[0-9].json")):
    d = json.load(open(f)); single[d["meta"]["model"]].append(d["result"])
b3 = json.load(open(os.path.join(S16, "baselines_usa2d_K3_rhoS4.json")))["results"]
b3s = json.load(open(os.path.join(S16, "baselines_usa2d_K3.json")))["results"]


def fm(vals, fmt="{:.4f}"):
    a = np.array(vals, float)
    return fmt.format(a[0]) if len(a) == 1 else (fmt + " ± " + fmt).format(a.mean(), a.std())


def reg_rmse(pred_coeffs):
    """rollout RMSE vs observed per region (July); regional (T, R, 9) or single-basis (T, 9) coefficients."""
    P = np.asarray(pred_coeffs)
    if P.ndim == 2:
        M.c2.setup(mask, 3, M.Lx / M.LREF, M.Ly / M.LREF)
        f = M.c2.reconstruct_grid(P).reshape(len(P), -1)
    else:
        f = DD.field_flat(P, NY * NX)
    obs = u[te].reshape(len(te), -1)
    return [float(np.sqrt(((f[:, r_.flat] - obs[:, r_.flat]) ** 2).mean())) for r_ in DD.regions]


def row(lab_, rows, key_viol="pde_violation_rollout", params="–", regional=True):
    g = lambda k: fm([r[k] for r in rows])  # noqa: E731
    key = "pred_rollout" if "pred_rollout" in rows[0] else ("coeffs_july" if "coeffs_july" in rows[0] else None)
    rr = np.array([reg_rmse(r[key]) for r in rows]) if key else None
    regs = " | ".join(fm(rr[:, i]) if rr is not None else "–" for i in range(DD.R))
    v = g(key_viol) if key_viol in rows[0] else "–"
    return f"| {lab_} | {params} | {g('rmse_onestep')} | {g('rmse_rollout')} | {g('rmse_jul12')} | {g('rmse_jul26')} | {g('rmse_rollout_vs_projection')} | {v} | {regs} |"


f = meta["fit"]
t = res["truncation"]
L = ["## T15. Regional domain decomposition (5 regions × 3×3 modes = 45 real coefficients; 5 independent 9-qubit circuits)", "",
     f"Regions (cells): " + ", ".join(f"{n} ({c})" for n, c in zip(names, meta["cells"])) + f"; {meta['n_faces']} interface cell faces. "
     f"cond(G) per region: {', '.join(f'{c:.0f}' for c in meta['cond_G'])}. "
     f"Regional truncation floor: rel. L2 train {t['train']:.3f} / val {t['val']:.3f} / July {t['july']:.3f}, RMSE July {t['rmse_july']:.4f} "
     f"(single 3×3 basis: RMSE July {b3s['truncation']['rmse_july']:.4f}).", "",
     f"Regional Galerkin ODE A fit (rollout protocol, H=7, window {f['fit_window_days']} d chosen on validation): D = {f['D']:.3g} "
     f"({f['D_physical_km2_per_day']:.3g} km²/day), β = {f['beta']:.4f}, ρ_r = " + ", ".join(f"{n} {v:.4f}" for n, v in zip(names, f["rho"])) +
     f" /day. July interface-flux / local-RHS norm ratio (observed coefficients): {res['flux_to_local_rhs_ratio']:.3f}.", "",
     "Rollout from the observed Jun 30 state, RMSE in u vs the full-resolution observed field (land cells); 'vs projection' is against the model's own projection "
     "(regional 45-mode projection for regional rows, single 3×3 projection for single-region rows). PDE violation: regional rows use the regional physics (D, ρ_r, β; "
     "pointwise PDE, 400 points per region); single-region rows use the §17 scalar reference — the two are not on the same scale of physics, compare within a block. "
     "Trained rows: 3 seeds, λ_PDE = 1e-4.", "",
     "| model | params | 1-step RMSE | rollout RMSE | Jul 12 | Jul 26 | rollout vs projection | PDE viol. rollout | " + " | ".join(names) + " |",
     "|---|---|---|---|---|---|---|---|" + "---|" * DD.R]
L.append(row("**regional** truncation floor", [res["Projection-floor"]], params="–"))
L.append(row("regional persistence", [res["Persistence"]], params="0"))
L.append(row("regional linear trend", [res["LinearTrend"]], params="0"))
L.append(row("regional Galerkin A, coupled (flux)", [res["A-regional-coupled"]], params="7"))
L.append(row("regional Galerkin A, decoupled", [res["A-regional-decoupled"]], params="7"))
for nm, lab_ in (("D", "regional QPINO D (5×9 qubits)"), ("B", "regional MLP B")):
    rows = asm.get(nm, [])
    if rows:
        P = f"{rows[0]['n_params_total']}"
        L.append(row(lab_ + ", coupled", [r["coupled"] for r in rows], params=P))
        L.append(row(lab_ + ", decoupled", [r["decoupled"] for r in rows], params=P))
sg = lambda r: [r] if isinstance(r, dict) else r  # noqa: E731
L.append(f"| *single 3×3 basis (§16/§17)* | | | | | | | | " + " | ".join("" for _ in names) + " |")
L.append(row("single-region truncation floor", [b3s["Projection-floor"]], key_viol="-", regional=False))
L.append(row("single-region persistence", [b3s["Persistence"]], key_viol="-", params="0", regional=False))
L.append(row("single-region linear trend", [b3s["LinearTrend"]], key_viol="-", params="0", regional=False))
L.append(row("single-region Galerkin A (rollout protocol)", [b3["A-scalar-sameprotocol"]], key_viol="-", params="3"))
L.append(row("single-region QPINO D (9 qubits), λ=1e-4", single["D"], key_viol="-", params="478"))
L.append(row("single-region MLP B, λ=1e-4", single["B"], key_viol="-", params="485"))
L.append("")
for nm in ("D", "B"):
    rows = asm.get(nm, [])
    if rows:
        tt = np.array([r["train_time_s_per_region"] for r in rows])
        L.append(f"* regional {nm}: training wall time per region {tt.mean():.0f} s (range {tt.min():.0f}–{tt.max():.0f} s, 1 thread, regions run in parallel).")
txt = "\n".join(L).replace("| – | – |", "| – | – |")
# single-region rows have pred_rollout in single 3x3 coefficients: per-region columns computed from the single basis
open(os.path.join(ROOT, "docs", "results_tables_usa2d_regional.md"), "w").write("# Regional domain-decomposition QPINO (report §18)\n\n" + txt + "\n")
print(txt)

# ------------------------------------------------------------------ fig21
NYX = NY * NX
grid = lambda B: DD.field_flat(np.asarray(B), NYX).reshape(np.asarray(B).shape[:-2] + (NY, NX))  # noqa: E731
d26, i26 = day["2020-07-26"], int(np.where(dates == pd.Timestamp("2020-07-26"))[0][0])
med = lambda rows, k: sorted(rows, key=lambda r: r[k]["rmse_rollout"])[len(rows) // 2]  # noqa: E731
rD = med(asm["D"], "coupled")
sD = sorted(single["D"], key=lambda r: r["rmse_rollout"])[1]
fig = plt.figure(figsize=(22, 8.4))
gs = fig.add_gridspec(2, 8, width_ratios=[1, 1, 1, 1, 1, 0.04, 0.5, 1.1], wspace=0.08, hspace=0.42)
from matplotlib.colors import ListedColormap  # noqa: E402
regc = ListedColormap(["#a6cee3", "#fdbf6f", "#b2df8a", "#fb9a99", "#cab2d6"])
ax = fig.add_subplot(gs[0, 0])
us_map(ax, np.where(mask > 0, lab, np.nan).astype(float), "5 regions\n(own 3×3 basis each)", cmap=regc, vmin=-0.5, vmax=4.5)
for r_, reg in enumerate(DD.regions):
    yy, xx = np.unravel_index(reg.flat, mask.shape)
    lon = M.ext[0] + (xx.mean() + 0.5) / NX * (M.ext[1] - M.ext[0]); lat = M.ext[2] + (yy.mean() + 0.5) / NY * (M.ext[3] - M.ext[2])
    ax.text(lon, lat, names[r_].replace("/", "/\n"), ha="center", va="center", fontsize=8, color="#1b2533", weight="bold")
im = us_map(fig.add_subplot(gs[0, 1]), u[i26], "observed\nJul 26")
us_map(fig.add_subplot(gs[0, 2]), grid(res["Projection-floor"]["pred_rollout"])[d26],
       f"regional floor (45 modes)\nJul 26, RMSE {res['Projection-floor']['rollout_rmse_by_day'][d26]:.3f}")
us_map(fig.add_subplot(gs[0, 3]), grid(rD["coupled"]["pred_rollout"])[d26],
       f"regional QPINO D, coupled\nJul 26, seed {rD['seed']}, RMSE {rD['coupled']['rollout_rmse_by_day'][d26]:.3f}")
us_map(fig.add_subplot(gs[0, 4]), grid(res["A-regional-coupled"]["pred_rollout"])[d26],
       f"regional Galerkin A, coupled\nJul 26, RMSE {res['A-regional-coupled']['rollout_rmse_by_day'][d26]:.3f}")
fig.colorbar(im, cax=fig.add_subplot(gs[0, 5])).set_label("u")
M.c2.setup(mask, 3, M.Lx / M.LREF, M.Ly / M.LREF)
sfield = M.c2.reconstruct_grid(np.asarray(sD["pred_rollout"]))[d26]
panels = [(sfield, f"error single-region QPINO D\n(9 qubits, RMSE {sD['rollout_rmse_by_day'][d26]:.3f})"),
          (grid(rD["coupled"]["pred_rollout"])[d26], "error regional QPINO D\n(5×9 qubits, coupled)"),
          (grid(rD["decoupled"]["pred_rollout"])[d26], "error regional QPINO D\n(decoupled)"),
          (grid(res["A-regional-coupled"]["pred_rollout"])[d26], "error regional Galerkin A\n(coupled)"),
          (grid(res["Projection-floor"]["pred_rollout"])[d26], "error regional floor")]
for k, (F, t_) in enumerate(panels):
    ime = us_map(fig.add_subplot(gs[1, k]), F - u[i26], t_ + " (Jul 26)" if "\n" not in t_ else t_, cmap=ERR, vmin=-0.4, vmax=0.4)
fig.colorbar(ime, cax=fig.add_subplot(gs[1, 5])).set_label("forecast − observed (u)")
ax = fig.add_subplot(gs[:, 7]); days = np.arange(1, len(te) + 1)
ax.plot(days, res["Persistence"]["rollout_rmse_by_day"], color="0.6", label="regional persistence")
ax.plot(days, res["LinearTrend"]["rollout_rmse_by_day"], color="0.35", label="regional linear trend")
ax.plot(days, res["A-regional-coupled"]["rollout_rmse_by_day"], color="k", label="regional A, coupled")
ax.plot(days, res["Projection-floor"]["rollout_rmse_by_day"], "--", color="k", lw=0.8, label="regional floor (45 modes)")
ax.plot(days, b3s["Projection-floor"]["rollout_rmse_by_day"], ":", color="k", lw=0.8, label="single 3×3 floor")
for nm, c in (("D", "tab:blue"), ("B", "tab:orange")):
    a = np.array([r["coupled"]["rollout_rmse_by_day"] for r in asm[nm]])
    ax.plot(days, a.mean(0), color=c, lw=1.3, label=f"regional {nm}, coupled (n={len(a)})"); ax.fill_between(days, a.min(0), a.max(0), color=c, alpha=0.12)
a = np.array([r["rollout_rmse_by_day"] for r in single["D"]])
ax.plot(days, a.mean(0), color="tab:blue", ls="--", lw=1, label="single-region D (n=3)")
ax.set_ylim(0.07, 0.26); ax.set_xlabel("July day"); ax.set_ylabel("rollout RMSE in u (land cells)")
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.set_title("July rollout error (λ_PDE = 1e-4)", fontsize=12)
fig.suptitle("Regional domain-decomposition QPINO: 5 regions × 9 qubits (never entangled), classical interface-flux coupling; rollout from Jun 30",
             fontsize=14, color=ACC)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig21_usa2d_regional.{e}"), dpi=150, bbox_inches="tight", facecolor="white")
print("wrote fig21")
