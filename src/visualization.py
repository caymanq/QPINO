"""Publication figures (Phase 10).  All functions read saved metrics/arrays and write PNG+PDF.
Synthetic-validation figures are titled 'SYNTHETIC'; JHU figures 'JHU COVID-19 (exploratory)'."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import galerkin as gk  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 220, "font.size": 9, "axes.titlesize": 10,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.fontsize": 7.5, "legend.frameon": False})
COL = {"Persistence": "#7f7f7f", "LinearTrend": "#bcbd22", "A": "#1f77b4", "A-fit": "#aec7e8",
       "B": "#ff7f0e", "B-data": "#ffbb78", "C": "#9467bd", "D": "#d62728", "D-mlpdec": "#e377c2",
       "E": "#2ca02c", "F": "#8c564b"}
SYN = "SYNTHETIC Fisher-KPP"
JHU = "JHU COVID-19, I-10 transect (exploratory)"


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _date_ax(ax, dates, n=6):
    idx = np.linspace(0, len(dates) - 1, n).astype(int)
    ax.set_xticks(idx)
    ax.set_xticklabels([pd.Timestamp(dates[i]).strftime("%b %d") for i in idx], rotation=0)


# 1 ------------------------------------------------------------------------------------------
def fig_covid_data(Z, counties, waypoints, split_dates, path):
    u, raw, dates, s = Z["u"], Z["raw_daily_per100k"], Z["dates"], Z["s_km"]
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, :2])
    sc = ax.scatter(counties.Long_, counties.Lat, c=counties.s_km, s=np.sqrt(counties.Population) / 25,
                    cmap="viridis", edgecolor="k", lw=0.2)
    ax.plot([w[2] for w in waypoints], [w[1] for w in waypoints], "k-", lw=1)
    for w in waypoints[::3]:
        ax.annotate(w[0].rsplit(" ", 1)[0], (w[2], w[1]), fontsize=6.5, xytext=(2, 4), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="arc length s along I-10 (km)")
    ax.set_title(f"(a) {len(counties)} counties within 100 km of I-10 (marker size ~ sqrt population)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect(1.15)
    ax = fig.add_subplot(gs[0, 2])
    T = len(dates)
    for j, c in zip([3, 6, 13, 18, 22, 30], plt.cm.tab10.colors):
        ax.plot(raw[:, j], color=c, lw=0.5, alpha=0.5)
        ax.plot(u[:, j] * float(Z["scale_S"]), color=c, lw=1.4, label=f"bin {j} (s={s[j]:.0f} km)")
    ax.set_ylabel("new cases / 100k / day"); _date_ax(ax, dates, 5)
    ax.set_title("(b) raw daily (thin) vs 7-day trailing mean (thick)"); ax.legend(ncol=1)
    ax.set_ylim(0, np.nanpercentile(raw, 99.7))
    for k, (lab, A) in enumerate([("(c) raw daily incidence per 100k (negatives clipped)", raw),
                                   ("(d) processed field u(x,t) (7-day mean / training max)", u)]):
        ax = fig.add_subplot(gs[1, k if k == 0 else 1:3] if k else gs[1, 0])
        im = ax.imshow(A.T, aspect="auto", origin="lower", cmap="magma",
                       extent=[0, T - 1, 0, s[-1] + s[0]], vmax=np.nanpercentile(A, 99.5))
        plt.colorbar(im, ax=ax)
        for d, lbl in split_dates:
            i = list(dates).index(d)
            ax.axvline(i, color="c", lw=1, ls="--"); ax.text(i + 0.5, 50, lbl, color="c", fontsize=7)
        _date_ax(ax, dates); ax.set_ylabel("s (km): LA -> Jacksonville"); ax.set_title(lab, fontsize=9)
    fig.suptitle(f"Fig. 1  Raw and processed {JHU}", fontsize=11)
    return save(fig, path)


# 2 ------------------------------------------------------------------------------------------
def fig_m2_reconstruction(Z, periodic, path):
    u, x, dates, s = Z["u"], Z["x"], Z["dates"], Z["s_km"]
    xf = np.linspace(0, 2 * np.pi, 400)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, d in zip(axs.flat[:4], ["2020-04-15", "2020-06-16", "2020-06-30", "2020-07-21"]):
        i = list(dates).index(d)
        a = gk.field_to_coeffs(u[i], x)
        ax.bar(x, u[i], width=2 * np.pi / len(x) * 0.9, color="#bbb", label="observed u (32 bins)")
        ax.plot(xf, gk.reconstruct(a, xf), "r", lw=2, label="M=2 Galerkin reconstruction")
        F = np.fft.fft(u[i]); F[5:len(x) - 4] = 0
        ax.plot(x, np.fft.ifft(F).real, "b--", lw=1, label="M=4 (reference)")
        e = np.linalg.norm(u[i] - gk.reconstruct(a, x)) / np.linalg.norm(u[i])
        ax.set_title(f"{d}: M=2 rel. L2 error {e:.2f}"); ax.set_xlabel("x = 2 pi s / L")
        ax.set_xlim(0, 2 * np.pi)
    axs.flat[0].legend()
    ax = axs.flat[4]
    for Mk, c in zip([1, 2, 4, 8], ["#fdae61", "r", "b", "g"]):
        F = np.fft.fft(u, axis=-1); F[:, Mk + 1:len(x) - Mk] = 0
        e = np.linalg.norm(np.fft.ifft(F, axis=-1).real - u, axis=-1) / (np.linalg.norm(u, axis=-1) + 1e-12)
        ax.plot(e, color=c, label=f"M={Mk}")
    ax.set_ylim(0, 1.05); _date_ax(ax, dates, 5); ax.set_ylabel("relative L2 truncation error")
    ax.set_title("truncation error over time (periodic map)"); ax.legend()
    ax = axs.flat[5]
    labels = ["periodic", "mirror (even ext.)", "detrended periodic"]
    tr = [periodic["M2_trunc_rel_l2_periodic"]["train"], periodic["M2_trunc_rel_l2_mirror"]["train"],
          periodic["M2_trunc_rel_l2_detrended_periodic"]["train"]]
    ju = [periodic["M2_trunc_rel_l2_periodic"]["july"], periodic["M2_trunc_rel_l2_mirror"]["july"],
          periodic["M2_trunc_rel_l2_detrended_periodic"]["july"]]
    xx = np.arange(3)
    ax.bar(xx - 0.2, tr, 0.4, label="train (Mar 8-Jun 16)"); ax.bar(xx + 0.2, ju, 0.4, label="July")
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylabel("mean M=2 rel. L2 error")
    ax.set_title(f"boundary treatment (wrap jump: {periodic['wrap_jump_rel_mean']:.2f} of max)"); ax.legend()
    fig.suptitle(f"Fig. 2  M=2 spatial reconstruction of the {JHU} field", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


# 3 ------------------------------------------------------------------------------------------
def fig_galerkin_vs_full(t, x, U, Ag, trunc, path):
    UM = gk.reconstruct(Ag, x)
    Ap = gk.field_to_coeffs(U, x)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, A, lab in [(axs[0, 0], U, "full PDE (pseudo-spectral N=128)"), (axs[0, 1], UM, "M=2 Galerkin ODE (reconstructed)"),
                       (axs[0, 2], U - UM, "difference")]:
        im = ax.imshow(A.T, aspect="auto", origin="lower", extent=[t[0], t[-1], 0, 2 * np.pi],
                       cmap="RdBu_r" if lab == "difference" else "viridis")
        plt.colorbar(im, ax=ax); ax.set_title(lab); ax.set_xlabel("t"); ax.set_ylabel("x")
    for j, c in enumerate(plt.cm.tab10.colors[:5]):
        axs[1, 0].plot(t, Ap[:, j], color=c, lw=2, alpha=0.5, label=f"{gk.NAMES[j]} projected full")
        axs[1, 0].plot(t, Ag[:, j], "--", color=c, lw=1)
    axs[1, 0].set_title("coefficients: projected full (solid) vs Galerkin (dashed)"); axs[1, 0].legend(ncol=2, fontsize=6)
    axs[1, 0].set_xlabel("t")
    axs[1, 1].plot(t, np.linalg.norm(U - gk.reconstruct(Ap, x), axis=1) / np.linalg.norm(U, axis=1), label="optimal M=2 projection")
    axs[1, 1].plot(t, np.linalg.norm(U - UM, axis=1) / np.linalg.norm(U, axis=1), "--", label="M=2 Galerkin evolution")
    axs[1, 1].set_yscale("log"); axs[1, 1].set_title("relative L2 error vs full solution"); axs[1, 1].legend(); axs[1, 1].set_xlabel("t")
    Ms = [1, 2, 3, 4, 8]
    axs[1, 2].semilogy(Ms, [trunc[f"M{m}_rel_l2"] for m in Ms], "o-")
    axs[1, 2].set_xlabel("retained modes M"); axs[1, 2].set_title("truncation error of full solutions (all trajectories)")
    fig.suptitle(f"Fig. 3  {SYN} (D=0.1, rho=1, beta=1): full vs M=2 Galerkin-truncated evolution", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


# 4 ------------------------------------------------------------------------------------------
def fig_circuits(path_qpino, path_hea, path_graphs, edges_by_name):
    import pennylane as qml
    from . import pennylane_circuits as plc
    E = edges_by_name["structured"]
    dev = qml.device("default.qubit", wires=5)

    @qml.qnode(dev)
    def circ():
        for j in range(5):
            qml.RY(0.0, wires=j, id=f"a{j}z{j}")
        plc.trotter_layer(np.ones(5) * 0.5, np.ones(5) * 0.5, np.ones(len(E)) * 0.5, E, 1.0)
        return [qml.expval(qml.PauliZ(j)) for j in range(5)]

    fig, ax = qml.draw_mpl(circ, decimals=None, style="pennylane")()
    fig.suptitle("Fig. 4a  QPINO circuit: R_y(alpha_j z_j) embedding  +  one first-order Trotter step of\n"
                 "exp(-i tau H_theta), H_theta = sum h Z + sum g X + sum_{(j,k) in E_Galerkin} J ZZ   (repeat L times), measure <Z_j>, <Z_jZ_k>",
                 fontsize=10)
    fig.savefig(path_qpino, bbox_inches="tight", dpi=200); plt.close(fig)

    @qml.qnode(dev)
    def hea():
        for j in range(5):
            qml.RY(0.0, wires=j)
        for _ in range(2):
            for j in range(5):
                qml.RY(0.1, wires=j); qml.RZ(0.1, wires=j)
            for j in range(5):
                qml.CNOT(wires=[j, (j + 1) % 5])
        return [qml.expval(qml.PauliZ(j)) for j in range(5)]

    fig, ax = qml.draw_mpl(hea, decimals=None, style="pennylane")()
    fig.suptitle("Fig. 4b  Baseline C: same R_y embedding + generic hardware-efficient ansatz (2 layers)", fontsize=10)
    fig.savefig(path_hea, bbox_inches="tight", dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, len(edges_by_name), figsize=(3 * len(edges_by_name), 3.2))
    ang = np.pi / 2 - 2 * np.pi * np.arange(5) / 5
    pos = np.stack([np.cos(ang), np.sin(ang)], 1)
    for ax, (nm, edges) in zip(axs, edges_by_name.items()):
        for j, k in edges:
            ax.plot(*pos[[j, k]].T, "-", color="#555", lw=1.5)
        for j in range(5):
            ax.add_patch(plt.Circle(pos[j], 0.2, color="#d62728" if j == 0 else ("#1f77b4" if j in (1, 2) else "#2ca02c"), zorder=3))
            ax.text(*pos[j], gk.NAMES[j].replace(" ", "\n"), ha="center", va="center", color="w", fontsize=6.5, zorder=4)
        ax.set_title(f"{nm}: |E| = {len(edges)}"); ax.set_aspect(1); ax.axis("off")
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    fig.suptitle("Fig. 4c  ZZ coupling graphs. 'structured' = pairs (j,k) with d^2 f/da_j da_k != 0 in the M=2 Galerkin RHS",
                 fontsize=10)
    return save(fig, path_graphs)


# 5 / 6 --------------------------------------------------------------------------------------
def fig_loss_curves(hs, hc, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, H, title in [(axs[0], hs, SYN), (axs[1], hc, JHU)]:
        for name in ["B", "C", "D", "E", "F"]:
            if name not in H:
                continue
            h = H[name]
            ax.semilogy(h["epoch"], h["data"], color=COL[name], lw=1.3, label=f"{name} train")
            ax.semilogy(h["epoch"], h["val_data"], "--", color=COL[name], lw=1, label=f"{name} val")
        ax.set_xlabel("epoch"); ax.set_ylabel("one-step data loss (units of training increment variance)")
        ax.set_title(title); ax.legend(ncol=2)
    fig.suptitle("Fig. 5  Training (solid) and validation (dashed) data loss", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def fig_data_vs_pde(hs, hc, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, H, title in [(axs[0], hs, SYN), (axs[1], hc, JHU)]:
        h = H["D"]
        for k, c, lab in [("data", "k", "data"), ("pde_full", "r", "PDE residual (full, reconstructed field)"),
                          ("pde_gal", "m", "projected Galerkin residual"), ("roll", "b", "rollout"),
                          ("val_data", "gray", "val data")]:
            ax.semilogy(h["epoch"], h[k], color=c, label=lab, lw=1.2, ls="--" if k == "val_data" else "-")
        if "E" in H:
            ax.semilogy(H["E"]["epoch"], H["E"]["pde_full"], ":", color="r", label="PDE residual of model E (no physics loss)")
        ax2 = ax.twinx(); ax2.plot(h["epoch"], h["lam_pde"], color="orange", alpha=0.6); ax2.set_ylabel("lambda_PDE (curriculum)", color="orange")
        ax2.grid(False)
        ax.set_xlabel("epoch"); ax.set_title(f"{title}: model D"); ax.legend(fontsize=6.5)
    fig.suptitle("Fig. 6  Separate data and PDE-residual losses (scaled units)", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


# 7 ------------------------------------------------------------------------------------------
def fig_covid_pred_vs_obs(Z, models, path):
    u, x, dates, idx = Z["u"], Z["x"], Z["dates"], Z["idx_july"]
    S = float(Z["scale_S"])
    obs = u[idx] * S
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.5))
    panels = [("observed (32 bins)", obs), ("observed, M=2 projection", gk.reconstruct(gk.field_to_coeffs(u[idx], x), x) * S)]
    for nm in ["D", "Persistence"]:
        panels.append((f"{nm}: 31-day rollout from Jun 30", gk.reconstruct(Z[f"roll_{nm}"], x) * S))
    vmax = np.percentile(obs, 99)
    for ax, (lab, A) in zip(axs[0], panels):
        im = ax.imshow(A.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax, extent=[1, 31, 0, 3735])
        ax.set_title(lab); ax.set_xlabel("July day"); ax.set_ylabel("s (km)")
    plt.colorbar(im, ax=axs[0, -1], label="cases/100k/day")
    xf = np.linspace(0, 2 * np.pi, 300)
    for ax, d in zip(axs[1], [0, 6, 13, 30]):
        ax.bar(x, obs[d], width=2 * np.pi / len(x) * 0.9, color="#ccc", label="observed")
        ax.plot(xf, gk.reconstruct(gk.field_to_coeffs(u[idx[d]], x), xf) * S, "k--", lw=1, label="obs M=2")
        for nm in models:
            if f"one_{nm}" in Z and d == 0:
                pass
            if f"roll_{nm}" in Z:
                ax.plot(xf, gk.reconstruct(Z[f"roll_{nm}"][d], xf) * S, color=COL.get(nm, "k"), lw=1.2, label=f"{nm} rollout")
        ax.set_title(f"Jul {d + 1} (rollout day {d + 1})"); ax.set_xlabel("x"); ax.set_xlim(0, 2 * np.pi)
    axs[1, 0].legend(fontsize=6)
    fig.suptitle(f"Fig. 7  Predicted vs observed July 2020 fields, {JHU}", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


# 8 ------------------------------------------------------------------------------------------
def fig_multistep(syn, cov, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for nm, v in syn.items():
        r = v["seeds"][0]
        if "rollout_rel_by_step" in r:
            axs[0].semilogy(np.arange(1, len(r["rollout_rel_by_step"]) + 1), r["rollout_rel_by_step"], color=COL.get(nm, "k"), label=nm)
    axs[0].set_xlabel("rollout step (dt = 0.1)"); axs[0].set_ylabel("relative coefficient error")
    axs[0].set_title(f"{SYN}: 40-step rollouts from unseen ICs (seed 0)"); axs[0].legend(ncol=2)
    for nm, v in cov.items():
        r = v["seeds"][0]
        if "rollout_field_rmse_by_step_vs_observed" in r:
            axs[1].plot(np.arange(1, 32), np.array(r["rollout_field_rmse_by_step_vs_observed"]), color=COL.get(nm, "k"), label=nm)
    axs[1].set_xlabel("days after Jun 30"); axs[1].set_ylabel("field RMSE vs observed (u units)")
    axs[1].set_title(f"{JHU}: July rollouts (seed 0)"); axs[1].legend(ncol=2)
    fig.suptitle("Fig. 8  Classical vs quantum multi-step forecasts", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


# 9 ------------------------------------------------------------------------------------------
def fig_fourier_traj(SA, Z, path):
    fig, axs = plt.subplots(2, 5, figsize=(15, 5.8))
    t = SA["t"]; true = SA["test"][0]
    for j in range(5):
        ax = axs[0, j]
        ax.plot(t, true[:, j], "k", lw=2.2, alpha=0.4, label="truth")
        for nm in ["A", "B", "C", "D", "E", "F"]:
            if f"roll_{nm}" in SA:
                ax.plot(t[1:], SA[f"roll_{nm}"][0, :, j], color=COL[nm], lw=1, label=nm)
        ax.set_title(f"SYN {gk.NAMES[j]}"); ax.set_xlabel("t")
    axs[0, 0].legend(fontsize=6)
    A, idx, i0 = Z["A"], Z["idx_july"], int(Z["i_jun30"])
    T = np.arange(len(A))
    for j in range(5):
        ax = axs[1, j]
        ax.plot(T, A[:, j], "k", lw=1.5, alpha=0.6, label="observed")
        for nm in ["Persistence", "LinearTrend", "A", "B", "D", "E"]:
            if f"roll_{nm}" in Z:
                ax.plot(idx, Z[f"roll_{nm}"][:, j], color=COL[nm], lw=1, label=nm)
        ax.axvline(i0, color="c", ls="--", lw=0.8)
        ax.set_title(f"JHU {gk.NAMES[j]}"); _date_ax(ax, Z["dates"], 4)
    axs[1, 0].legend(fontsize=6)
    fig.suptitle("Fig. 9  Fourier-coefficient trajectories: synthetic test IC (top), JHU I-10 with July rollouts (bottom)", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


# 10 -----------------------------------------------------------------------------------------
def fig_spectrum(spec, path):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for R, c in zip(["1", "2", "3"], ["#d62728", "#1f77b4", "#2ca02c"]):
        p = np.mean([spec[R][f"input{j}_power_by_freq"] for j in range(5)], 0)
        axs[0].semilogy(np.arange(len(p)), p + 1e-18, "o-", color=c, label=f"R={R} re-uploads")
        d = spec[R]["power_by_total_degree_output0"]
        k = sorted(int(v) for v in d)
        axs[1].semilogy(k, [d[str(v)] if str(v) in d else d[v] for v in k], "o-", color=c, label=f"R={R}")
    axs[0].axhline(1e-15, color="k", lw=0.5); axs[0].set_xlabel("|omega| along one input angle phi_j = alpha_j z_j")
    axs[0].set_ylabel("mean |c_omega| of decoder output"); axs[0].set_title("1-D spectrum of trained models (cut-off at R)")
    axs[0].legend()
    axs[1].set_xlabel("total degree |omega|_1"); axs[1].set_ylabel("spectral power"); axs[1].set_title("5-D spectrum, output da0/dt")
    axs[1].legend()
    Rs = np.arange(1, 6)
    axs[2].semilogy(Rs, (2 * Rs + 1) ** 5, "ko-", label="frequency vectors (2R+1)^5")
    axs[2].semilogy(Rs, ((2 * Rs + 1) ** 5 + 1) / 2, "s--", color="gray", label="independent (real output)")
    axs[2].set_xlabel("re-uploads R"); axs[2].set_title("accessible coefficient-space spectrum (5 inputs)"); axs[2].legend()
    fig.suptitle("Fig. 10  Quantum Fourier spectrum in COEFFICIENT space (frequencies in phi = alpha z, not spatial wavenumbers k)",
                 fontsize=10.5)
    fig.tight_layout()
    return save(fig, path)


# 11 -----------------------------------------------------------------------------------------
def _bars(ax, d, keys, metric, title, ylabel, color="#d62728", ref=None, log=True):
    keys = [k for k in keys if k in d]
    m = [d[k][metric]["mean"] for k in keys]; s = [d[k][metric]["std"] for k in keys]
    ax.bar(range(len(keys)), m, yerr=s, color=color, alpha=0.8, capsize=2)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=40, ha="right", fontsize=7)
    if ref is not None:
        ax.axhline(ref[1], color="gray", ls="--", lw=1); ax.text(0, ref[1], ref[0], fontsize=7, color="gray", va="bottom")
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8)


def fig_ablations(AC, PL, cov_ref, path):
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.5))
    s = AC["synthetic"]
    _bars(axs[0, 0], s, ["graph=structured", "graph=triad", "graph=full", "graph=nearest", "graph=none (F)", "HEA (C)"],
          "rollout_rel_l2", "SYN: coupling graph / ansatz", "40-step rollout rel. L2")
    _bars(axs[0, 1], s, ["graph=structured", "trotter L=1", "trotter L=2", "trotter L=4", "trotter L=8"],
          "rollout_rel_l2", "SYN: exact vs trained Trotter L", "rollout rel. L2")
    _bars(axs[0, 2], s, ["tau fixed c=2", "tau fixed c=10", "tau fixed c=50", "tau learned c0=10"],
          "dt0.2_rollout_rel_l2", "SYN: tau = c dt, tested at dt=0.2 (trained 0.1)", "rollout rel. L2 @ dt=0.2")
    _bars(axs[0, 3], s, ["graph=structured", "reupload R=2", "reupload R=3", "measure Z only", "decoder MLP",
                         "frozen circuit + linear decoder", "frozen circuit + MLP decoder", "alpha0=0.25", "alpha0=0.5",
                         "alpha0=2", "alpha fixed=1"], "rel_l2_increment", "SYN: re-uploading / readout / alpha",
          "one-step increment rel. L2")
    _bars(axs[1, 0], PL["synthetic_full"], list(PL["synthetic_full"]), "rollout_rel_l2", "SYN (32 traj): loss weights",
          "rollout rel. L2", color="#1f77b4")
    _bars(axs[1, 1], PL["synthetic_lowdata"], list(PL["synthetic_lowdata"]), "rollout_rel_l2",
          "SYN LOW-DATA (2 traj x 10 steps): loss weights", "rollout rel. L2", color="#1f77b4")
    c = AC.get("covid", {})
    if c:
        _bars(axs[1, 2], c, list(c), "field_rmse_per100k_rollout", "JHU: ansatz variants (July rollout)",
              "field RMSE (cases/100k/day)", color="#9467bd", ref=("persistence", cov_ref["Persistence"]), log=False)
        lo = min(v["field_rmse_per100k_rollout"]["mean"] for v in c.values())
        axs[1, 2].set_ylim(lo * 0.9, None)
    cp = PL.get("covid", {})
    if cp:
        _bars(axs[1, 3], cp, list(cp), "field_rmse_per100k_rollout", "JHU: loss weights (July rollout)",
              "field RMSE (cases/100k/day)", color="#2ca02c", ref=("persistence", cov_ref["Persistence"]), log=False)
        lo = min(v["field_rmse_per100k_rollout"]["mean"] for v in cp.values())
        axs[1, 3].set_ylim(lo * 0.9, None)
    fig.suptitle("Fig. 11  Model ablations (mean +- std over seeds)", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


# 12 -----------------------------------------------------------------------------------------
def fig_runtime(SC, syn_summary, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    rv = SC["runtime_vs_qubits"]
    ns = sorted(int(n) for n in rv)
    for key, st in [("structured-like(full)_exact", "o-"), ("structured-like(full)_trotter", "s--"),
                    ("nearest_exact", "^-"), ("nearest_trotter", "v--")]:
        vals = [rv[str(n)].get(key + "_s_per_fwdbwd_batch256", np.nan) for n in ns]
        axs[0].semilogy(ns, vals, st, label=key.replace("_", " "))
    axs[0].set_xlabel("qubits n"); axs[0].set_ylabel("s per forward+backward (batch 256)")
    axs[0].set_title("classical statevector simulation cost (exponential in n)"); axs[0].legend(fontsize=6.5)
    names = [k for k in ["A", "B", "B-data", "C", "D", "D-mlpdec", "E", "F"] if k in syn_summary]
    axs[1].bar(range(len(names)), [syn_summary[k]["n_params"] for k in names], color=[COL[k] for k in names])
    axs[1].set_xticks(range(len(names))); axs[1].set_xticklabels(names)
    ax2 = axs[1].twinx(); ax2.plot(range(len(names)), [syn_summary[k]["train_time_s"] for k in names], "ko-")
    ax2.set_ylabel("training time (s, CPU)"); ax2.grid(False)
    axs[1].set_ylabel("trainable parameters (bars)"); axs[1].set_title("SYN: parameters and training time")
    sh = SC["shot_sensitivity"]
    for key, c in [("synthetic", "#d62728"), ("covid_july", "#9467bd")]:
        if key in sh:
            d = sh[key]
            Ss = sorted(int(k) for k in d if k.isdigit())
            axs[2].loglog(Ss, [d[str(s)]["rms_dev_from_exact"] for s in Ss], "o-", color=c, label=f"{key}: dev. from exact")
            axs[2].loglog(Ss, [d[str(s)]["rms_dev_from_exact"] for s in Ss][0] * np.sqrt(Ss[0] / np.array(Ss)), ":", color=c,
                          label="1/sqrt(shots)")
    axs[2].set_xlabel("shots per circuit"); axs[2].set_ylabel("RMS prediction deviation"); axs[2].legend(fontsize=6.5)
    axs[2].set_title("measurement-shot sensitivity (D, test subset)")
    fig.suptitle("Fig. 12  Runtime and resource comparison", fontsize=11)
    fig.tight_layout()
    return save(fig, path)
