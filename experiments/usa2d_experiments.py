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
args = p.parse_args()
gk.set_basis("cosine2d")
K, n = args.K, args.K ** 2
OUT = os.path.join(MET, "usa2d"); os.makedirs(OUT, exist_ok=True)
cfg_name = f"usa2d_K{K}"

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
c2.setup(mask, K, Lx_km / LREF, Ly_km / LREF)
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

fit_path = os.path.join(OUT, f"fit_{cfg_name}.json")
if os.path.exists(fit_path):
    fit = json.load(open(fit_path))
else:
    T = train_tr[0]
    fit = gk.fit_galerkin_parameters(T[:-1], T[1:], dt, w=1 / scales["s_inc"].numpy(), n_starts=12, seed=0)
    fit["D_physical_km2_per_day"] = fit["D"] * LREF ** 2
    fit["K_carrying_capacity_u"] = fit["rho"] / fit["beta"]
    save_json(fit, fit_path)
phys = (fit["D"], fit["rho"], fit["beta"])
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
    ctx = m.cache_unitaries() if hasattr(m, "cache_unitaries") else torch.no_grad()
    with ctx:
        one = m(torch.tensor(A[te - 1]), dt).numpy()
        roll = m.rollout(torch.tensor(A[i30][None]), dt, len(te))[0].numpy()
    r = metrics(one, roll)
    r["pred_onestep"], r["pred_rollout"] = one.tolist(), roll.tolist()
    return r


meta = dict(K=K, n_qubits=n, grid=[NY, NX], land_cells=int(land.sum()), L_km=[Lx_km, Ly_km], L_ref_km=LREF,
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
    res["Persistence"]["pred_rollout"] = np.repeat(A[i30][None], len(te), 0).tolist()
    e_raw = np.sqrt(((U[i30][None] - U[te]) ** 2).mean(-1))
    res["Persistence-fullres"] = dict(rmse_rollout=float(np.sqrt(((U[i30][None] - U[te]) ** 2).mean())),
                                      rmse_onestep=float(np.sqrt(((U[te - 1] - U[te]) ** 2).mean())),
                                      rmse_jul12=float(e_raw[DAY["2020-07-12"]]), rmse_jul26=float(e_raw[DAY["2020-07-26"]]),
                                      rollout_rmse_by_day=e_raw.tolist())
    res["LinearTrend"] = metrics(ev.linear_trend_predictions(A, te, 7), ev.linear_trend_rollout(A, i30, len(te), 7))
    t0 = time.time()
    res["A"] = evaluate(GalerkinOperator(*phys))
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
tcfg = dict(C["train"]); tcfg.update(epochs=args.epochs, seed=args.seed, lambda_pde=args.lambda_pde)
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
hist, m = train(m, Dtr, Dva, dt, tcfg, phys, scales, X_C, ckpt_path=os.path.join(CKPT, "usa2d", f"{tag}.pt"))
wall = time.time() - t0
r = evaluate(m)
r.update(n_params=count_params(m), train_time_s=wall, epochs=args.epochs, sec_per_epoch=wall / args.epochs,
         threads=torch.get_num_threads())
if hasattr(m, "c"):
    r["learned_c"] = float(m.c)
meta.update(model=args.model, lambda_pde=tcfg["lambda_pde"], R=args.R, seed=args.seed, tag=tag,
            evolution=mcfg.get("evolution"))
save_json(dict(meta=meta, result=r, history={k: hist[k] for k in ("epoch", "data", "val_data", "pde_full", "lam_pde") if k in hist}),
          os.path.join(OUT, f"{tag}.json"))
print(tag, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
