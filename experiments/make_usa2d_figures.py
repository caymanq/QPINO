"""Figures 18-19: 2D contiguous-USA QPINO maps (light YlGnBu theme, state borders), report section 16."""
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
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import cosine2d_galerkin as c2  # noqa: E402

MET, FIG = os.path.join(ROOT, "results", "metrics", "usa2d"), os.path.join(ROOT, "results", "figures")
FG, ACC, STATE = "#1b2533", "#0b5f8a", "#27323f"
CMAP = LinearSegmentedColormap.from_list("ylgnbu", ["#f4f9d2", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"])
CMAP.set_bad((0, 0, 0, 0))
ERR = LinearSegmentedColormap.from_list("err", ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"])
ERR.set_bad((0, 0, 0, 0))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG})
Z = np.load(os.path.join(ROOT, "data", "processed", "us_grid_density_256x128.npz"))
u, mask = Z["u"].astype(float), Z["mask"]
dates = pd.to_datetime(Z["dates"])
NY, NX = mask.shape
ext = [Z["lon_edges"][0], Z["lon_edges"][-1], Z["lat_edges"][0], Z["lat_edges"][-1]]
SL = np.load(os.path.join(ROOT, "data", "processed", "us_state_lines.npz"))
lines = np.split(SL["xy"], np.cumsum(SL["lens"])[:-1])
Lx, Ly = NX * float(Z["dx_km"]), NY * float(Z["dy_km"])
LREF = Ly / np.pi
runs = defaultdict(list)
for f in glob.glob(os.path.join(MET, "usa2d_K*_s[0-9].json")):
    d = json.load(open(f)); m = d["meta"]
    if m.get("lambda_pde") == 1e-4:          # section 17 runs: fig20 (make_usa2d_rho_figure.py)
        continue
    runs[(m["K"], m["model"])].append(d["result"] | {"seed": m["seed"]})
base = {K: json.load(open(os.path.join(MET, f"baselines_usa2d_K{K}.json"))) for K in (3, 4)
        if os.path.exists(os.path.join(MET, f"baselines_usa2d_K{K}.json"))}
te = np.where((dates > "2020-06-30") & (dates <= "2020-07-26"))[0]


def grid_of(b, K):
    c2.setup(mask, K, Lx / LREF, Ly / LREF)
    return c2.reconstruct_grid(np.asarray(b))


def us_map(ax, f, title, cmap=CMAP, vmin=0, vmax=1):
    f = np.where(mask > 0, f, np.nan)
    im = ax.imshow(f, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.add_collection(LineCollection(lines, colors=STATE, linewidths=0.45, alpha=0.6))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(37.0))); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12.5, pad=4)
    return im


def median_seed(K, nm):
    rows = sorted(runs.get((K, nm), []), key=lambda r: r["rmse_rollout"])
    return rows[len(rows) // 2] if rows else None


K = 3
rD = median_seed(K, "D")
bK = base[K]["results"]
day = {d: int(np.where(dates[te] == pd.Timestamp(d))[0][0]) for d in ("2020-07-12", "2020-07-26")}
# ---------------------------------------------------------------- fig 18: Cayman-style 8 panels
fig, axs = plt.subplots(2, 4, figsize=(17.5, 6.4), gridspec_kw=dict(wspace=0.04, hspace=0.12))
obs_days = ["2020-03-15", "2020-04-05", "2020-04-26", "2020-05-17", "2020-06-07", "2020-06-28"]
for ax, d in zip(axs.ravel()[:6], obs_days):
    i = int(np.where(dates == pd.Timestamp(d))[0][0])
    im = us_map(ax, u[i], pd.Timestamp(d).strftime("%b %d, %Y"))
if rD is not None:
    P = grid_of(rD["pred_rollout"], K)
    for ax, d in zip(axs.ravel()[6:], ("2020-07-12", "2020-07-26")):
        us_map(ax, P[day[d]], pd.Timestamp(d).strftime("%b %d, %Y") + "  (forecast)")
cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.015, pad=0.01); cb.set_label("u = log1p(cases/100k) / log1p(4800)")
sub = (f"QPINO D (9 qubits, 3×3 real cosine modes on the land mask), seed {rD['seed']} (median of 3), "
       f"July rollout RMSE {rD['rmse_rollout']:.3f}") if rD else "QPINO D: not available"
fig.suptitle("Observed u(x,y,t) Mar 15 – Jun 28 2020 (256×128) and QPINO forecasts from the Jun 30 state\n" + sub,
             fontsize=14, color=ACC, y=1.01)
for ext_ in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig18_usa2d_qpino_maps.{ext_}"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote fig18")

# ---------------------------------------------------------------- fig 19: companion panels
fig = plt.figure(figsize=(21, 8.2))
gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 0.04, 0.28, 1.15], wspace=0.06, hspace=0.3)
floor = grid_of(bK["Projection-floor"]["coeffs_july"], K)
Arol = grid_of(bK["A"]["pred_rollout"], K)
P = grid_of(rD["pred_rollout"], K) if rD else None
for r, d in enumerate(("2020-07-12", "2020-07-26")):
    i, j = int(np.where(dates == pd.Timestamp(d))[0][0]), day[d]
    lab = pd.Timestamp(d).strftime("%b %d")
    im = us_map(fig.add_subplot(gs[r, 0]), u[i], f"observed {lab}")
    us_map(fig.add_subplot(gs[r, 1]), floor[j], f"truncation floor {lab}\n(3×3 projection of observed, RMSE {bK['Projection-floor']['rollout_rmse_by_day'][j]:.3f})")
    if P is not None:
        us_map(fig.add_subplot(gs[r, 2]), P[j], f"QPINO D forecast {lab}\n(RMSE {rD['rollout_rmse_by_day'][j]:.3f})")
    us_map(fig.add_subplot(gs[r, 3]), Arol[j], f"Galerkin ODE A forecast {lab}\n(RMSE {bK['A']['rollout_rmse_by_day'][j]:.3f})")
ax = fig.add_subplot(gs[0, 6])
days = np.arange(1, len(te) + 1)
col = dict(Persistence="0.55", LinearTrend="0.3", A="k", B="tab:orange", C="tab:green", D="tab:blue", E="tab:purple", F="tab:red")
for nm in ("Persistence", "LinearTrend", "A"):
    ax.plot(days, bK[nm]["rollout_rmse_by_day"], color=col[nm], lw=1.6, label=nm)
ax.plot(days, bK["Persistence-fullres"]["rollout_rmse_by_day"], ":", color="0.4", label="persistence, full-res field")
ax.plot(days, bK["Projection-floor"]["rollout_rmse_by_day"], "--", color="k", lw=0.8, label="3×3 truncation floor")
for nm in ("B", "C", "D", "E", "F"):
    rows = runs.get((K, nm), [])
    if rows:
        a = np.array([q["rollout_rmse_by_day"] for q in rows])
        ax.plot(days, a.mean(0), color=col[nm], lw=1.1, label=f"{nm} (n={len(rows)})")
        ax.fill_between(days, a.min(0), a.max(0), color=col[nm], alpha=0.12)
ax.set_xlabel("July day"); ax.set_ylabel("RMSE in u (land cells)"); ax.legend(fontsize=7.5, ncol=2, frameon=False)
ax.set_title("July rollout error, 3×3 (9 qubits)", fontsize=12)
ax = fig.add_subplot(gs[1, 6])
Ks = [k for k in (3, 4) if k in base]
for i, (key, lab) in enumerate((("train", "train (Mar 15–Jun 16)"), ("july", "July"))):
    ax.bar(np.arange(len(Ks)) + (i - 0.5) * 0.35, [base[k]["results"]["truncation"][key] for k in Ks], 0.35, label=lab)
ax.set_xticks(range(len(Ks))); ax.set_xticklabels([f"{k}×{k}\n({k*k} qubits)" for k in Ks])
ax.set_ylabel("mean rel. L2 truncation error (land)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("2D cosine truncation error", fontsize=12)
cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 4])); cb.set_label("u")
fig.suptitle("July 2020: observed vs 3×3 truncation floor vs QPINO D vs Galerkin ODE A (rollout from Jun 30)", fontsize=14, color=ACC)
for ext_ in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig19_usa2d_july_companion.{ext_}"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote fig19")
