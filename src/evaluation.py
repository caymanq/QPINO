"""Evaluation metrics: accuracy, physical consistency, stability, shots, runtime."""
from __future__ import annotations

import time

import numpy as np
import torch

from . import galerkin, losses


def _np(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)


def rel_l2(pred, true):
    pred, true = _np(pred), _np(true)
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-15))


def point_metrics(pred, true, a_n, x, phys=None, dt=None, u_full=None, x_full=None):
    """Metrics for one-step (or any) predictions pred vs true (P,5)."""
    pred, true, a_n = _np(pred), _np(true), _np(a_n)
    out = dict(mse=float(((pred - true) ** 2).sum(-1).mean()),
               rmse_coeff=float(np.sqrt(((pred - true) ** 2).mean())),
               rel_l2=rel_l2(pred, true),
               rel_l2_increment=rel_l2(pred - a_n, true - a_n))
    up, ut = galerkin.reconstruct(pred, x), galerkin.reconstruct(true, x)
    out["field_rmse_vs_truncated"] = float(np.sqrt(((up - ut) ** 2).mean()))
    out["field_rel_l2_vs_truncated"] = rel_l2(up, ut)
    if u_full is not None:
        upf = galerkin.reconstruct(pred, x_full)
        out["field_rmse_vs_observed"] = float(np.sqrt(((upf - u_full) ** 2).mean()))
        out["field_rel_l2_vs_observed"] = rel_l2(upf, u_full)
    if phys is not None and dt is not None:
        D, rho, beta = phys
        tn = torch.tensor(a_n); tp = torch.tensor(pred)
        xc = galerkin.uniform_grid(128)
        for sch in ["forward", "trapezoid"]:
            Rg = _np(losses.galerkin_residual(tn, tp, dt, D, rho, beta, sch))
            Rf = _np(losses.full_pde_residual(tn, tp, dt, D, rho, beta, xc, sch))
            out[f"galerkin_residual_rms_{sch}"] = float(np.sqrt((Rg ** 2).mean()))
            out[f"pde_residual_rms_{sch}"] = float(np.sqrt((Rf ** 2).mean()))
    out["positivity_violation_frac"] = float((galerkin.reconstruct(pred, galerkin.uniform_grid(128)) < 0).mean())
    return out


def rollout_metrics(pred_roll, true_roll, x, u_full=None, x_full=None):
    """pred_roll, true_roll: (B, S, 5).  Error per horizon step and aggregates."""
    p, t = _np(pred_roll), _np(true_roll)
    err_t = np.sqrt(((p - t) ** 2).sum(-1).mean(0))
    rel_t = np.linalg.norm(p - t, axis=-1).mean(0) / (np.linalg.norm(t, axis=-1).mean(0) + 1e-15)
    up, ut = galerkin.reconstruct(p, x), galerkin.reconstruct(t, x)
    out = dict(rollout_mse=float(((p - t) ** 2).sum(-1).mean()),
               rollout_rel_l2=rel_l2(p, t), rollout_final_rel_l2=rel_l2(p[:, -1], t[:, -1]),
               rollout_err_by_step=err_t.tolist(), rollout_rel_by_step=rel_t.tolist(),
               rollout_field_rmse_vs_truncated=float(np.sqrt(((up - ut) ** 2).mean())),
               rollout_field_rel_l2_vs_truncated=rel_l2(up, ut),
               rollout_positivity_violation_frac=float((galerkin.reconstruct(p, galerkin.uniform_grid(128)) < 0).mean()),
               rollout_finite=bool(np.isfinite(p).all()))
    if u_full is not None:
        upf = galerkin.reconstruct(p, x_full)
        out["rollout_field_rmse_vs_observed"] = float(np.sqrt(((upf - u_full) ** 2).mean()))
        out["rollout_field_rel_l2_vs_observed"] = rel_l2(upf, u_full)
        out["rollout_field_rmse_by_step_vs_observed"] = np.sqrt(((upf - u_full) ** 2).mean((0, 2))).tolist()
    return out


@torch.no_grad()
def long_term_stability(model, a0, dt, n_steps, K=None):
    """Roll the operator for many steps; report finiteness, max norm, final state."""
    traj = _np(model.rollout(torch.as_tensor(a0), dt, n_steps))
    fin = bool(np.isfinite(traj).all())
    out = dict(n_steps=n_steps, finite=fin,
               max_norm=float(np.nanmax(np.linalg.norm(traj, axis=-1))) if fin else float("inf"),
               final_a0_mean=float(np.nanmean(traj[:, -1, 0])),
               final_nonmean_norm=float(np.nanmean(np.linalg.norm(traj[:, -1, 1:], axis=-1))),
               min_field=float(np.nanmin(galerkin.reconstruct(traj[:, -1], galerkin.uniform_grid(64)))))
    if K is not None:
        out["K_equilibrium"] = float(K)
        out["final_dist_to_equilibrium"] = float(np.nanmean(np.abs(traj[:, -1, 0] - K)) +
                                                 out["final_nonmean_norm"])
    return out, traj


def linear_trend_predictions(A, idx_targets, window=7):
    """One-step linear-trend baseline: a_{n+1} = a_n + (a_n - a_{n-window})/window."""
    A = _np(A)
    return np.stack([A[i - 1] + (A[i - 1] - A[i - 1 - window]) / window for i in idx_targets])


def linear_trend_rollout(A, i0, n_steps, window=7):
    A = _np(A)
    slope = (A[i0] - A[i0 - window]) / window
    return np.stack([A[i0] + s * slope for s in range(1, n_steps + 1)])


@torch.no_grad()
def shot_sensitivity(model, a_n, a_np1, dt, shots_list, repeats=10, seed=0):
    torch.manual_seed(seed)
    exact = _np(model(a_n, dt))
    res = {"exact_mse": float(((exact - _np(a_np1)) ** 2).sum(-1).mean())}
    for S in shots_list:
        model.shots = S
        mses, dev = [], []
        for _ in range(repeats):
            p = _np(model(a_n, dt))
            mses.append(((p - _np(a_np1)) ** 2).sum(-1).mean())
            dev.append(np.sqrt(((p - exact) ** 2).sum(-1).mean()))
        model.shots = None
        res[str(S)] = dict(mse_mean=float(np.mean(mses)), mse_std=float(np.std(mses)),
                           rms_dev_from_exact=float(np.mean(dev)))
    return res


def time_forward_backward(model, a, dt, reps=20):
    t0 = time.time()
    for _ in range(reps):
        model.zero_grad()
        model(a, dt).pow(2).sum().backward()
    return (time.time() - t0) / reps
