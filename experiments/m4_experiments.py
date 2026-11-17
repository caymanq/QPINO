"""M = 4 / mirrored-boundary / data-dominated-loss study (new options; the M=2 periodic
results in results/metrics/{synthetic,covid}_results.json are not touched).

One invocation = one (regime, M, boundary, model, lambda_pde, R, seed) training run, written
to results/metrics/m4/<tag>.json so many single-thread processes can run in parallel.
`--baselines` evaluates the untrained references (persistence, linear trend, Galerkin ODE A).

  python experiments/m4_experiments.py --regime covid --M 4 --boundary mirror --model D --seed 0
  python experiments/m4_experiments.py --regime covid --M 4 --boundary mirror --baselines
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from common import (C, CKPT, MET, ROOT, X_C, ev, fk, gk, make_dataset, make_norm, save_json,
                    PersistenceOperator, GalerkinOperator, count_params)
from src import preprocessing as pp
from src.qpino import MLPOperator, QPINO
from src.training import set_seed, train

OVR = yaml.safe_load(open(os.path.join(ROOT, "configs", "m4_mirror.yaml")))
p = argparse.ArgumentParser()
p.add_argument("--regime", choices=["covid", "synthetic"], required=True)
p.add_argument("--M", type=int, default=OVR["galerkin_M"])
p.add_argument("--boundary", choices=["periodic", "mirror"], default="periodic")
p.add_argument("--model", default="D")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lambda_pde", type=float, default=OVR["train"]["lambda_pde"])
p.add_argument("--R", type=int, default=1)
p.add_argument("--epochs", type=int, default=C["train"]["epochs"])
p.add_argument("--baselines", action="store_true")
args = p.parse_args()
n = 2 * args.M + 1
OUT = os.path.join(MET, "m4"); os.makedirs(OUT, exist_ok=True)
cfg_name = f"{args.regime}_M{args.M}_{args.boundary}"


# ------------------------------------------------------------------ data
def covid_data():
    CC = C["covid"]
    Z = np.load(os.path.join(ROOT, "data", "processed", "covid_i10_field.npz"))
    import pandas as pd
    u, x = Z["u"], Z["x"]
    dates = pd.to_datetime(Z["dates"])
    tr, va, te = pp.split_indices(dates, CC["train_end"], CC["val_end"], CC["test_end"])
    N = u.shape[1]
    if args.boundary == "mirror":
        xg = gk.uniform_grid(2 * N, 0.5)
        A = gk.field_to_coeffs(pp.mirror_extend(u), xg, args.M)
        A[:, 2::2] = 0.0                 # even extension: Im a_k = 0 exactly (round-off ~1e-17 removed)
        x_obs = xg[:N]                   # the original 32 bins = first half of the 2L domain
    else:
        A = gk.field_to_coeffs(u, x, args.M)
        x_obs = x
    idx = np.arange(va[-1] + 1, te[-1] + 1)
    return dict(A=A, u=u, x_obs=x_obs, S=float(Z["scale_S"]), dt=CC["dt"], H=CC["horizon"], c0=CC["c0"],
                train=[A[tr[0]:tr[-1] + 1]], val=[A[tr[-1]:va[-1] + 1]], idx=idx, i_jun30=va[-1],
                u_obs=u[te], L_km=3735.0 * (2 if args.boundary == "mirror" else 1))


def synthetic_data():
    S = C["synthetic"]
    n_tot = S["n_train"] + S["n_val"] + S["n_test"]
    t, x, U = fk.generate_trajectories(n_tot, S["D"], S["rho"], S["beta"], S["dt"], S["n_steps"],
                                       N=S["N"], seed=S["data_seed"])
    A = [gk.field_to_coeffs(u_, x, args.M) for u_ in U]
    i1, i2 = S["n_train"], S["n_train"] + S["n_val"]
    return dict(t=t, x=x, U=U, train=A[:i1], val=A[i1:i2], test=A[i2:], U_test=U[i2:], dt=S["dt"],
                H=S["horizon"], c0=S["c0"], phys=(S["D"], S["rho"], S["beta"]))


Dd = covid_data() if args.regime == "covid" else synthetic_data()
dt = Dd["dt"]
norm, scales = make_norm(Dd["train"], dt, X_C)
# Components with zero training variance (Im a_k under mirroring) would get s_inc ~ 1e-12 and
# sigma ~ 1e-8, i.e. 1e24 loss weights.  Give them the median scale of the active components so
# the loss penalises spurious Im outputs at a normal scale (and float ops stay well-conditioned).
_si = scales["s_inc"].clone(); dead = _si < 1e-9
if bool(dead.any()):
    _si[dead] = _si[~dead].median()
    scales["s_inc"] = _si
    norm.sigma[dead] = 1.0
    norm.g_scale.copy_(_si / dt)

# Fisher-KPP parameters: true values (synthetic) or a fit in the new representation (COVID)
fit_path = os.path.join(OUT, f"fit_{cfg_name}.json")
if args.regime == "synthetic":
    phys = Dd["phys"]
else:
    if os.path.exists(fit_path):
        fit = json.load(open(fit_path))
    else:
        T = Dd["train"][0]
        w = 1 / scales["s_inc"].numpy()
        w = np.where(scales["s_inc"].numpy() < 1e-9, 0.0, w)     # Im a_k == 0 under mirroring
        fit = gk.fit_galerkin_parameters(T[:-1], T[1:], dt, w=w, n_starts=12, seed=0)
        fit["D_physical_km2_per_day"] = fit["D"] * (Dd["L_km"] / (2 * np.pi)) ** 2
        fit["K_carrying_capacity_u"] = fit["rho"] / fit["beta"]
        save_json(fit, fit_path)
    phys = (fit["D"], fit["rho"], fit["beta"])


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(m):
    ctx = m.cache_unitaries() if hasattr(m, "cache_unitaries") else torch.no_grad()
    with ctx:
        if args.regime == "covid":
            A, idx = Dd["A"], Dd["idx"]
            one = m(torch.tensor(A[idx - 1]), dt).numpy()
            roll = m.rollout(torch.tensor(A[Dd["i_jun30"]][None]), dt, len(idx))[0].numpy()
            r = covid_metrics(one, roll)
            r["pred_onestep"], r["pred_rollout"] = one.tolist(), roll.tolist()
            return r
        S = len(Dd["test"][0]) - 1
        ds = make_dataset(Dd["test"], 1)
        pred = m(ds["a_n"], dt)
        r = ev.point_metrics(pred, ds["a_np1"], ds["a_n"], X_C, phys, dt)
        a0 = torch.tensor(np.stack([T[0] for T in Dd["test"]]))
        roll = m.rollout(a0, dt, S)
        true = np.stack([T[1:S + 1] for T in Dd["test"]])
        uf = np.stack([u_[1:S + 1] for u_ in Dd["U_test"]])
        r.update(ev.rollout_metrics(roll, true, X_C, uf, Dd["x"]))
        r["pred_rollout_test0"] = roll[0].numpy().tolist()
        keep = ["rel_l2", "rel_l2_increment", "rollout_rel_l2", "rollout_final_rel_l2",
                "rollout_field_rel_l2_vs_truncated", "rollout_field_rel_l2_vs_observed",
                "rollout_positivity_violation_frac", "rollout_finite", "pde_residual_rms_trapezoid",
                "galerkin_residual_rms_trapezoid", "rollout_rel_by_step", "pred_rollout_test0"]
        return {k: r[k] for k in keep if k in r}


def covid_metrics(one, roll):
    A, idx = Dd["A"], Dd["idx"]
    a_prev, a_true = A[idx - 1], A[idx]
    f1, f2 = gk.reconstruct(one, Dd["x_obs"]), gk.reconstruct(roll, Dd["x_obs"])
    e2 = np.sqrt(((f2 - Dd["u_obs"]) ** 2).mean(-1)) * Dd["S"]
    return dict(rel_l2=ev.rel_l2(one, a_true),
                rel_l2_increment=ev.rel_l2(one - a_prev, a_true - a_prev),
                rollout_rel_l2=ev.rel_l2(roll, a_true),
                field_rmse_per100k_onestep=float(np.sqrt(((f1 - Dd["u_obs"]) ** 2).mean())) * Dd["S"],
                field_rmse_per100k_rollout=float(np.sqrt(((f2 - Dd["u_obs"]) ** 2).mean())) * Dd["S"],
                rollout_field_rmse_by_day=e2.tolist(),
                rollout_negative_frac=float((f2 < 0).mean()),
                max_abs_Im_rollout=float(np.abs(roll[:, 2::2]).max()))


meta = dict(regime=args.regime, M=args.M, n_qubits=n, boundary=args.boundary, phys=list(phys),
            config="configs/m4_mirror.yaml (+ default.yaml)")
if args.baselines:
    res = {}
    if args.regime == "covid":
        A, idx, i0 = Dd["A"], Dd["idx"], Dd["i_jun30"]
        res["Persistence"] = covid_metrics(A[idx - 1], np.repeat(A[i0][None], len(idx), 0))
        res["LinearTrend"] = covid_metrics(ev.linear_trend_predictions(A, idx, 7),
                                           ev.linear_trend_rollout(A, i0, len(idx), 7))
        # projection floor: the observed field's own |k|<=M projection (best any model can do)
        res["Projection-floor"] = covid_metrics(A[idx], A[idx])
    else:
        res["Persistence"] = evaluate(PersistenceOperator())
    res["A"] = evaluate(GalerkinOperator(*phys))
    meta["fit"] = None if args.regime == "synthetic" else fit
    save_json(dict(meta=meta, results=res), os.path.join(OUT, f"baselines_{cfg_name}.json"))
    for k, v in res.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)})
    raise SystemExit

# ------------------------------------------------------------------ model
SPECS = {"D": {}, "E": dict(lambda_pde=0.0), "F": dict(graph="none"), "C": dict(ansatz="hea"),
         "B": dict(kind="mlp")}
spec = dict(SPECS[args.model])
mcfg = dict(C["model"]); mcfg["reupload"] = args.R
mcfg["evolution"] = "exact_eigh"   # same U and gradients as matrix_exp (tested), much faster backward
tcfg = dict(C["train"]); tcfg.update(epochs=args.epochs, seed=args.seed, lambda_pde=args.lambda_pde)
if "lambda_pde" in spec:
    tcfg["lambda_pde"] = spec.pop("lambda_pde")
set_seed(args.seed)
if spec.get("kind") == "mlp":
    m = MLPOperator(norm, OVR["mlp"]["hidden"] if args.M == 4 else C["mlp"]["hidden"], C["mlp"]["depth"], args.seed, n)
else:
    mcfg.update(spec)
    m = QPINO(norm, c0=Dd["c0"], seed=args.seed, n=n, **mcfg)
tag = f"{cfg_name}_{args.model}_lam{tcfg['lambda_pde']:g}_R{args.R}_s{args.seed}"
Dtr, Dva = make_dataset(Dd["train"], Dd["H"]), make_dataset(Dd["val"], min(Dd["H"], len(Dd["val"][0]) - 1))
t0 = time.time()
hist, m = train(m, Dtr, Dva, dt, tcfg, phys, scales, X_C, ckpt_path=os.path.join(CKPT, "m4", f"{tag}.pt"))
wall = time.time() - t0
r = evaluate(m)
r.update(n_params=count_params(m), train_time_s=wall, epochs=args.epochs,
         sec_per_epoch=wall / args.epochs, threads=torch.get_num_threads())
if hasattr(m, "c"):
    r["learned_c"] = float(m.c)
meta.update(model=args.model, lambda_pde=tcfg["lambda_pde"], R=args.R, seed=args.seed, tag=tag)
save_json(dict(meta=meta, result=r, history={k: hist[k] for k in ("epoch", "data", "val_data", "pde_full", "lam_pde") if k in hist}),
          os.path.join(OUT, f"{tag}.json"))
print(tag, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
