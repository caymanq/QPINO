"""Shared experiment utilities: data builders, model factory, standard evaluation."""
from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config, evaluation as ev, fisher_kpp as fk, galerkin as gk  # noqa: E402
from src.qpino import GalerkinOperator, MLPOperator, PersistenceOperator, QPINO  # noqa: E402
from src.training import make_dataset, make_norm, save_json, set_seed, train  # noqa: E402

torch.set_default_dtype(torch.float64)
C = config.load()
torch.set_num_threads(int(os.environ.get("QPINO_THREADS", C.get("threads", 4))))
RES = os.path.join(ROOT, "results")
FIG, MET, CKPT = (os.path.join(RES, d) for d in ("figures", "metrics", "checkpoints"))
for d in (FIG, MET, CKPT, os.path.join(ROOT, "data", "processed")):
    os.makedirs(d, exist_ok=True)
X_C = gk.uniform_grid(64)  # PDE collocation points

# --------------------------------------------------------------------------- synthetic data
def synthetic_data(dt=None, n_steps=None, N=None, seed=None, S=None):
    S = S or C["synthetic"]
    dt = dt or S["dt"]
    n_steps = n_steps or int(round(S["n_steps"] * S["dt"] / dt))
    n = S["n_train"] + S["n_val"] + S["n_test"]
    t, x, U = fk.generate_trajectories(n, S["D"], S["rho"], S["beta"], dt, n_steps,
                                       N=N or S["N"], seed=seed if seed is not None else S["data_seed"])
    A = [gk.field_to_coeffs(u, x) for u in U]
    i1, i2 = S["n_train"], S["n_train"] + S["n_val"]
    return dict(t=t, x=x, U=U, A=A, train=A[:i1], val=A[i1:i2], test=A[i2:], U_test=U[i2:], dt=dt)


# --------------------------------------------------------------------------- model factory
MODEL_SPECS = {
    "A": dict(kind="galerkin"),
    "B": dict(kind="mlp"),
    "B-data": dict(kind="mlp", train=dict(lambda_pde=0.0)),
    "C": dict(kind="qpino", model=dict(ansatz="hea")),
    "D": dict(kind="qpino"),
    "D-mlpdec": dict(kind="qpino", model=dict(decoder="mlp")),
    "E": dict(kind="qpino", train=dict(lambda_pde=0.0)),
    "F": dict(kind="qpino", model=dict(graph="none")),
}
DESCR = {
    "Persistence": "a(t+dt) = a(t)",
    "LinearTrend": "a(t+dt) = a(t) + [a(t) - a(t-7)]/7",
    "A": "Classical Galerkin ODE (fitted / true D, rho, beta)",
    "B": "Classical MLP, same loss as D",
    "B-data": "Classical MLP, no PDE loss",
    "C": "Ry embedding + hardware-efficient ansatz, same loss",
    "D": "QPINO: custom Hamiltonian, structured graph, data+PDE loss",
    "D-mlpdec": "QPINO (D) with an MLP decoder",
    "E": "QPINO without physics loss",
    "F": "QPINO without ZZ couplings",
}


def build_model(name, norm, phys, c0, seed=0, model_over=None, train_over=None):
    spec = MODEL_SPECS.get(name, dict(kind="qpino"))
    mcfg = dict(C["model"]); mcfg.update(spec.get("model", {})); mcfg.update(model_over or {})
    tcfg = dict(C["train"]); tcfg.update(spec.get("train", {})); tcfg.update(train_over or {})
    tcfg["seed"] = seed
    if spec["kind"] == "galerkin":
        return GalerkinOperator(*phys), tcfg
    if spec["kind"] == "mlp":
        return MLPOperator(norm, C["mlp"]["hidden"], C["mlp"]["depth"], seed), tcfg
    return QPINO(norm, c0=c0, seed=seed, **mcfg), tcfg


def fit_and_train(name, data_tr, data_va, dt, norm, scales, phys, c0, seed, H, tag,
                  model_over=None, train_over=None, save=True):
    set_seed(seed)
    m, tcfg = build_model(name, norm, phys, c0, seed, model_over, train_over)
    Dtr, Dva = make_dataset(data_tr, H), make_dataset(data_va, min(H, len(data_va[0]) - 1))
    t0 = time.time()
    if isinstance(m, GalerkinOperator):
        hist = {}
    else:
        hist, m = train(m, Dtr, Dva, dt, tcfg, phys, scales, X_C,
                        ckpt_path=os.path.join(CKPT, f"{tag}_{name}_s{seed}.pt") if save else None)
    hist["wall_time_s"] = time.time() - t0
    return m, hist, tcfg


@torch.no_grad()
def evaluate_on_trajs(m, trajs, dt, phys, x_eval, roll_steps=None, U_full=None, x_full=None):
    """One-step metrics on all consecutive pairs and rollout from each trajectory's start."""
    Dte = make_dataset(trajs, 1)
    pred = m(Dte["a_n"], dt)
    out = ev.point_metrics(pred, Dte["a_np1"], Dte["a_n"], x_eval, phys, dt)
    a0 = torch.tensor(np.stack([T[0] for T in trajs]))
    S = roll_steps or len(trajs[0]) - 1
    roll = m.rollout(a0, dt, S)
    true = np.stack([T[1:S + 1] for T in trajs])
    uf = None if U_full is None else np.stack([u[1:S + 1] for u in U_full])
    out.update(ev.rollout_metrics(roll, true, x_eval, uf, x_full))
    return out, roll.numpy()


def count_params(m):
    return m.n_params() if hasattr(m, "n_params") else 0


__all__ = ["C", "FIG", "MET", "CKPT", "X_C", "synthetic_data", "build_model", "fit_and_train",
           "evaluate_on_trajs", "make_norm", "make_dataset", "save_json", "ev", "gk", "fk",
           "PersistenceOperator", "GalerkinOperator", "DESCR", "count_params"]


# --------------------------------------------------------------------------- COVID bundle
def covid_bundle():
    """Load the processed I-10 field written by covid_training.py and the fitted parameters."""
    import json
    import pandas as pd
    from src import preprocessing as pp
    CC = C["covid"]
    Z = np.load(os.path.join(ROOT, "data", "processed", "covid_i10_field.npz"))
    u, x, A = Z["u"], Z["x"], Z["A"]
    dates = pd.to_datetime(Z["dates"])
    tr, va, te = pp.split_indices(dates, CC["train_end"], CC["val_end"], CC["test_end"])
    with open(os.path.join(MET, "covid_results.json")) as f:
        fit = json.load(f)["fitted_params"]
    idx = np.arange(va[-1] + 1, te[-1] + 1)
    return dict(u=u, x=x, A=A, dates=dates, train=[A[tr[0]:tr[-1] + 1]], val=[A[tr[-1]:va[-1] + 1]],
                idx_july=idx, a_prev=A[idx - 1], a_true=A[idx], a_jun30=A[va[-1]], u_obs=u[te],
                dt=CC["dt"], phys=(fit["D"], fit["rho"], fit["beta"]), S=float(Z["scale_S"]), c0=CC["c0"],
                H=CC["horizon"])


@torch.no_grad()
def covid_eval(m, B):
    one = m(torch.tensor(B["a_prev"]), B["dt"]).numpy()
    roll = m.rollout(torch.tensor(B["a_jun30"][None]), B["dt"], len(B["idx_july"]))[0].numpy()
    f1 = gk.reconstruct(one, B["x"]); f2 = gk.reconstruct(roll, B["x"])
    return dict(rel_l2=ev.rel_l2(one, B["a_true"]),
                rel_l2_increment=ev.rel_l2(one - B["a_prev"], B["a_true"] - B["a_prev"]),
                rollout_rel_l2=ev.rel_l2(roll, B["a_true"]),
                field_rmse_per100k_onestep=float(np.sqrt(((f1 - B["u_obs"]) ** 2).mean())) * B["S"],
                field_rmse_per100k_rollout=float(np.sqrt(((f2 - B["u_obs"]) ** 2).mean())) * B["S"])


def agg(rows):
    keys = rows[0].keys()
    return {k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows])))
            for k in keys if isinstance(rows[0][k], (int, float))}
