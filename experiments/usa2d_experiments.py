"""2D contiguous-USA QPINO: real 2D cosine/Neumann Galerkin coefficients on the land mask (report section 16).

    python usa2d_experiments.py --K 3 --baselines          # truncation, persistence, linear trend, A
    python usa2d_experiments.py --K 3 --model D --seed 0   # B (MLP), C (HEA), D (QPINO), E, F
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import yaml

from common import C, CKPT, MET, ROOT, ev, gk, save_json, GalerkinOperator, count_params
from src import cosine2d_galerkin as c2
from src.qpino import MLPOperator, QPINO
from src.training import make_dataset, make_norm, set_seed, train

U2 = yaml.safe_load(open(os.path.join(ROOT, "configs", "usa2d.yaml")))
p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=U2["modes_per_axis"])
p.add_argument("--model", default="D")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--R", type=int, default=1)
p.add_argument("--epochs", type=int, default=C["train"]["epochs"])
p.add_argument("--lambda_pde", type=float, default=U2["train"]["lambda_pde"])
p.add_argument("--evolution", default=None)
p.add_argument("--baselines", action="store_true")
p.add_argument("--rho", choices=["scalar", "spatial"], default="scalar")
p.add_argument("--Kr", type=int, default=U2["rho_modes"])
p.add_argument("--eval_only", type=float, default=None, help="load the saved checkpoint; value = training wall time (s)")
p.add_argument("--hinge_tol", type=float, default=U2["train"].get("pde_hinge_tol"))
args = p.parse_args()
gk.set_basis("cosine2d")
K, n = args.K, args.K ** 2
OUT = os.path.join(MET, "usa2d"); os.makedirs(OUT, exist_ok=True)
SPATIAL = args.rho == "spatial"
cfg_name = f"usa2d_K{K}" + (f"_rhoS{args.Kr}" if SPATIAL else "")

# ------------------------------------------------------------------ data
loc = os.path.join(ROOT, "data", "processed", "us_grid_density_256x128.npz")
if not os.path.exists(loc):
    shutil.copy(U2["grid_npz"], loc)
Z = np.load(loc)
u_full, mask = Z["u"].astype(np.float64), Z["mask"]
dates = pd.to_datetime(Z["dates"])
NY, NX = mask.shape
Lx_km, Ly_km = NX * float(Z["dx_km"]), NY * float(Z["dy_km"])
LREF = Ly_km / np.pi                         # non-dimensional length unit (km)
c2.setup(mask, K, Lx_km / LREF, Ly_km / LREF, Kr=args.Kr)
S = c2.active()
land = S["land"]
U = u_full.reshape(len(dates), -1)[:, land]   # (T, P_land)
A = c2.field_to_coeffs(U)                     # (T, n)
tr = np.where(dates <= U2["train_end"])[0]
va = np.where((dates > U2["train_end"]) & (dates <= U2["val_end"]))[0]
te = np.where((dates > U2["val_end"]) & (dates <= U2["test_end"]))[0]
i30 = va[-1]
dt, H = U2["dt"], U2["horizon"]
train_tr, val_tr = [A[tr[0]:tr[-1] + 1]], [A[tr[-1]:va[-1] + 1]]
rng = np.random.default_rng(0)
X_C = S["xy"][rng.choice(len(S["xy"]), U2["n_colloc"], replace=False)]
norm, scales = make_norm(train_tr, dt, X_C)

def scalar_fit():
    path = os.path.join(OUT, f"fit_usa2d_K{K}.json")
    if os.path.exists(path):
        return json.load(open(path))
    T = train_tr[0]
    f = gk.fit_galerkin_parameters(T[:-1], T[1:], dt, w=1 / scales["s_inc"].numpy(), n_starts=12, seed=0)
    f["D_physical_km2_per_day"] = f["D"] * LREF ** 2
    f["K_carrying_capacity_u"] = f["rho"] / f["beta"]
    save_json(f, path)
    return f


def spatial_fit(fs):
    """Spatial growth rate rho(x,y) on Kr x Kr cosine modes, fitted jointly with (D, beta).

    Protocol (chosen after a check that one-step fits of a spatial rho over the whole training
    window extrapolate badly; see report section 17): multi-step (H = 7) rollout objective, fit on
    the last NW days of the training window, shrinkage lam_l2 of the non-constant rho modes; the
    (NW, lam_l2) pair is chosen by the validation-window (Jun 17-30) rollout error.  The scalar
    model is fitted with the identical protocol (const_only) for a like-for-like comparison.
    Initialisation: local logistic growth rates of the observed field, rho_loc = du/dt / u + beta u,
    projected onto the rho basis (also stored)."""
    path = os.path.join(OUT, f"fit_usa2d_K{K}_rhoS{args.Kr}.json")
    if os.path.exists(path):
        return json.load(open(path))
    Ut = U[tr[0]:tr[-1] + 1]
    du, uu = np.diff(Ut, axis=0), Ut[:-1]
    loc = np.where(uu > 0.05, du / np.maximum(uu, 1e-6) + fs["beta"] * uu, np.nan)
    rho_loc = np.nan_to_num(np.nanmean(loc[-28:], 0), nan=fs["rho"])
    r_loc = c2.rho_to_coeffs(rho_loc)
    w = 1 / scales["s_inc"].numpy()
    V = val_tr[0]

    def val_err(f):
        roll = GalerkinOperator(f["D"], np.array(f["r"]), f["beta"]).rollout(torch.tensor(V[0][None]), dt, len(V) - 1)[0].numpy()
        return ev.rel_l2(roll, V[1:])
    out, sweep = {}, []
    for kind in ("scalar", "spatial"):
        best = None
        for NW in U2["rho_fit_windows"]:
            T = train_tr[0][-NW:] if NW else train_tr[0]
            Wn = make_dataset([T], H)["windows"].numpy()
            for lam2 in ([0.0] if kind == "scalar" else U2["rho_lam_l2_grid"]):
                r_init = np.zeros(args.Kr ** 2); r_init[0] = fs["rho"]
                f = c2.fit_spatial_rho(T[:-1], T[1:], dt, w, r_init, max(fs["D"], 1e-8), fs["beta"], lam_smooth=0.0,
                                       lam_l2=lam2, windows=Wn, const_only=(kind == "scalar"), iters=U2["rho_fit_iters"])
                f.update(kind=kind, fit_window_days=NW or len(train_tr[0]), val_rollout_rel_l2=val_err(f))
                sweep.append({k: f[k] for k in ("kind", "fit_window_days", "lam_l2", "loss", "val_rollout_rel_l2", "beta")})
                print(kind, NW, lam2, "val", round(f["val_rollout_rel_l2"], 5), "loss", round(f["loss"], 4), flush=True)
                if best is None or f["val_rollout_rel_l2"] < best["val_rollout_rel_l2"]:
                    best = f
        best["D_physical_km2_per_day"] = best["D"] * LREF ** 2
        rf = c2.rho_field(np.array(best["r"]))
        best.update(rho_field_range=[float(rf.min()), float(rf.max())], rho_field_mean=float(rf.mean()))
        out[kind] = best
    res = dict(out["spatial"]); res["scalar_same_protocol"] = out["scalar"]; res["sweep"] = sweep
    res["r_local_growth_init"] = r_loc.tolist(); res["rho_local_growth_range"] = [float(rho_loc.min()), float(rho_loc.max())]
    save_json(res, path)
    return res


fit_s = scalar_fit()
fit = spatial_fit(fit_s) if SPATIAL else fit_s
phys = (fit["D"], np.array(fit["r"]), fit["beta"]) if SPATIAL else (fit["D"], fit["rho"], fit["beta"])
# physics-violation diagnostic: residual of the full PDE against BOTH reference physics
fit_sp = json.load(open(os.path.join(OUT, f"fit_usa2d_K{K}_rhoS{args.Kr}.json"))) if os.path.exists(
    os.path.join(OUT, f"fit_usa2d_K{K}_rhoS{args.Kr}.json")) else None
REF = {"scalar": (fit_s["D"], fit_s["rho"], fit_s["beta"])}
if fit_sp:
    REF["spatial"] = (fit_sp["D"], np.array(fit_sp["r"]), fit_sp["beta"])
X_EV = S["xy"][np.random.default_rng(1).choice(len(S["xy"]), U2["n_eval_points"], replace=False)]


def pde_violation(a_n, a_np1):
    """RMS (u/day) of (u_{n+1}-u_n)/dt - [D lap u + rho(x,y) u - beta u^2] (trapezoid) on 2000 land cells."""
    a_n, a_np1 = np.asarray(a_n), np.asarray(a_np1)
    out = {}
    for k, (D_, r_, b_) in REF.items():
        ut = (c2.reconstruct(a_np1, X_EV) - c2.reconstruct(a_n, X_EV)) / dt
        f = 0.5 * (gk.full_pde_rhs_on_grid(a_n, X_EV, D_, r_, b_) + gk.full_pde_rhs_on_grid(a_np1, X_EV, D_, r_, b_))
        out[k] = float(np.sqrt(((ut - f) ** 2).mean()))
    return out


DAY = {d: int(np.where(dates[te] == pd.Timestamp(d))[0][0]) for d in ("2020-07-12", "2020-07-26")}


def field(b):
    return np.asarray(b) @ S["phi"].T


def metrics(one, roll):
    a_prev, a_true, u_obs = A[te - 1], A[te], U[te]
    f1, f2, fp = field(one), field(roll), field(a_true)
    e = np.sqrt(((f2 - u_obs) ** 2).mean(-1))
    return dict(rel_l2=ev.rel_l2(one, a_true), rel_l2_increment=ev.rel_l2(one - a_prev, a_true - a_prev),
                rollout_rel_l2=ev.rel_l2(roll, a_true),
                rmse_onestep=float(np.sqrt(((f1 - u_obs) ** 2).mean())),
                rmse_rollout=float(np.sqrt(((f2 - u_obs) ** 2).mean())),
                rmse_rollout_vs_projection=float(np.sqrt(((f2 - fp) ** 2).mean())),
                rmse_jul12=float(e[DAY["2020-07-12"]]), rmse_jul26=float(e[DAY["2020-07-26"]]),
                rollout_rmse_by_day=e.tolist())


@torch.no_grad()
def evaluate(m):
    ctx = lambda: m.cache_unitaries() if hasattr(m, "cache_unitaries") else torch.no_grad()  # noqa: E731  fresh per use
    with ctx():
        one = m(torch.tensor(A[te - 1]), dt).numpy()
        roll = m.rollout(torch.tensor(A[i30][None]), dt, len(te))[0].numpy()
    r = metrics(one, roll)
    r["pred_onestep"], r["pred_rollout"] = one.tolist(), roll.tolist()
    with ctx():
        Ttr, Tva = train_tr[0], val_tr[0]
        p_tr = m(torch.tensor(Ttr[:-1]), dt).numpy()
        p_va = m(torch.tensor(Tva[:-1]), dt).numpy()
    seq = np.vstack([A[i30][None], roll])
    r["pde_violation"] = dict(train_onestep=pde_violation(Ttr[:-1], p_tr), val_onestep=pde_violation(Tva[:-1], p_va),
                              test_rollout=pde_violation(seq[:-1], seq[1:]),
                              observed_test=pde_violation(A[te - 1], A[te]))
    return r


meta = dict(K=K, n_qubits=n, rho=args.rho, Kr=args.Kr if SPATIAL else None, grid=[NY, NX], land_cells=int(land.sum()), L_km=[Lx_km, Ly_km], L_ref_km=LREF,
            phys=list(phys), cond_G=S["cond_G"],
            dates=[str(dates[tr[0]].date()), U2["train_end"], U2["val_end"], str(dates[te[-1]].date())])
if args.baselines:
    res = {}
    proj = field(A)

    def trunc(ix):
        return float(np.mean(np.linalg.norm(proj[ix] - U[ix], axis=1) / np.linalg.norm(U[ix], axis=1)))
    res["truncation"] = dict(train=trunc(tr), val=trunc(va), july=trunc(te),
                             rmse_train=float(np.sqrt(((proj[tr] - U[tr]) ** 2).mean())),
                             rmse_july=float(np.sqrt(((proj[te] - U[te]) ** 2).mean())))
    res["Projection-floor"] = metrics(A[te], A[te])
    res["Persistence"] = metrics(A[te - 1], np.repeat(A[i30][None], len(te), 0))
    pr = np.vstack([A[i30][None], np.repeat(A[i30][None], len(te), 0)])
    res["Persistence"]["pde_violation"] = dict(test_rollout=pde_violation(pr[:-1], pr[1:]),
                                               observed_test=pde_violation(A[te - 1], A[te]))
    res["Persistence"]["pred_rollout"] = np.repeat(A[i30][None], len(te), 0).tolist()
    e_raw = np.sqrt(((U[i30][None] - U[te]) ** 2).mean(-1))
    res["Persistence-fullres"] = dict(rmse_rollout=float(np.sqrt(((U[i30][None] - U[te]) ** 2).mean())),
                                      rmse_onestep=float(np.sqrt(((U[te - 1] - U[te]) ** 2).mean())),
                                      rmse_jul12=float(e_raw[DAY["2020-07-12"]]), rmse_jul26=float(e_raw[DAY["2020-07-26"]]),
                                      rollout_rmse_by_day=e_raw.tolist())
    ltr = ev.linear_trend_rollout(A, i30, len(te), 7)
    res["LinearTrend"] = metrics(ev.linear_trend_predictions(A, te, 7), ltr)
    pr = np.vstack([A[i30][None], ltr])
    res["LinearTrend"]["pde_violation"] = dict(test_rollout=pde_violation(pr[:-1], pr[1:]))
    t0 = time.time()
    res["A"] = evaluate(GalerkinOperator(*phys))
    if SPATIAL:   # scalar rho fitted with the same multi-step / recent-window protocol
        fq = fit["scalar_same_protocol"]
        res["A-scalar-sameprotocol"] = evaluate(GalerkinOperator(fq["D"], np.array(fq["r"]), fq["beta"]))
    res["A"]["eval_time_s"] = time.time() - t0
    res["Projection-floor"]["coeffs_july"] = A[te].tolist()
    meta["fit"] = fit
    save_json(dict(meta=meta, results=res), os.path.join(OUT, f"baselines_{cfg_name}.json"))
    for k, v in res.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)})
    print("fit", fit)
    raise SystemExit

# ------------------------------------------------------------------ model
SPECS = {"D": {}, "E": dict(lambda_pde=0.0), "F": dict(graph="none"), "C": dict(ansatz="hea"), "B": dict(kind="mlp")}
spec = dict(SPECS[args.model])
mcfg = dict(C["model"]); mcfg["reupload"] = args.R
mcfg["evolution"] = args.evolution or ("exact_eigh" if n <= 10 else "statevector")
tcfg = dict(C["train"]); tcfg.update(epochs=args.epochs, seed=args.seed, lambda_pde=args.lambda_pde,
                                     log_every=U2["train"].get("log_every", 10), pde_hinge_tol=args.hinge_tol)
if "lambda_pde" in spec:
    tcfg["lambda_pde"] = spec.pop("lambda_pde")
set_seed(args.seed)
if spec.get("kind") == "mlp":
    m = MLPOperator(norm, U2["mlp_hidden"][n], C["mlp"]["depth"], args.seed, n)
else:
    mcfg.update(spec)
    m = QPINO(norm, c0=U2["c0"], seed=args.seed, n=n, **mcfg)
tag = f"{cfg_name}_{args.model}_lam{tcfg['lambda_pde']:g}_R{args.R}_s{args.seed}"
Dtr, Dva = make_dataset(train_tr, H), make_dataset(val_tr, min(H, len(val_tr[0]) - 1))
t0 = time.time()
if args.eval_only is not None:     # re-evaluate a saved final checkpoint (wall = training time recovered from the logs)
    m.load_state_dict(torch.load(os.path.join(CKPT, "usa2d", f"{tag}.pt"), weights_only=False)["state_dict"])
    hist, wall = {}, args.eval_only
else:
    hist, m = train(m, Dtr, Dva, dt, tcfg, phys, scales, X_C, ckpt_path=os.path.join(CKPT, "usa2d", f"{tag}.pt"))
    wall = time.time() - t0
r = evaluate(m)
r.update(n_params=count_params(m), train_time_s=wall, epochs=args.epochs, sec_per_epoch=wall / args.epochs,
         threads=torch.get_num_threads())
if hasattr(m, "c"):
    r["learned_c"] = float(m.c)
meta.update(model=args.model, lambda_pde=tcfg["lambda_pde"], R=args.R, seed=args.seed, tag=tag,
            evolution=mcfg.get("evolution"))
save_json(dict(meta=meta, result=r, history={k: hist[k] for k in ("epoch", "data", "val_data", "pde_full", "val_pde_full", "lam_pde") if k in hist}),
          os.path.join(OUT, f"{tag}.json"))
print(tag, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
