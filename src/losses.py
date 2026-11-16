"""Physics-informed loss terms.

All coefficient-space errors are measured in units of the training-set one-step increment
scale s_inc (per component), i.e. ||(a_hat - a)/s_inc||^2: a value of ~1 means "as large as a
typical one-day/one-step change".  Residuals are scaled by s_inc/dt for the same reason.
Raw (unscaled) metrics are reported separately in evaluation.py.
"""
from __future__ import annotations

import numpy as np
import torch

from . import galerkin


def data_loss(a_hat, a_true, s_inc):
    return (((a_hat - a_true) / s_inc) ** 2).sum(-1).mean()


def field_loss(a_hat, a_true, x):
    return ((galerkin.reconstruct(a_hat, x) - galerkin.reconstruct(a_true, x)) ** 2).mean()


def _rhs_ref(a_n, a_hat, D, rho, beta, scheme):
    if scheme == "forward":
        return galerkin.galerkin_rhs(a_n, D, rho, beta)
    if scheme == "trapezoid":
        return 0.5 * (galerkin.galerkin_rhs(a_n, D, rho, beta) + galerkin.galerkin_rhs(a_hat, D, rho, beta))
    if scheme == "midpoint":
        return galerkin.galerkin_rhs(0.5 * (a_n + a_hat), D, rho, beta)
    raise ValueError(scheme)


def galerkin_residual(a_n, a_hat, dt, D, rho, beta, scheme="forward"):
    """R_k = (a_hat_k - a_k)/dt - f_k(.)  (projected Galerkin residual, 5 real components)."""
    return (a_hat - a_n) / dt - _rhs_ref(a_n, a_hat, D, rho, beta, scheme)


def full_pde_residual(a_n, a_hat, dt, D, rho, beta, x_c, scheme="forward"):
    """R(x_i) = (u_hat(x_i,t+dt) - u(x_i,t))/dt - [D u_xx + rho u - beta u^2] on collocation
    points, with spatial derivatives evaluated analytically from the Fourier representation.
    u^2 here is the pointwise square of the reconstructed field and therefore contains the
    |k| = 3,4 modes that the projected (Galerkin) residual discards."""
    u_n = galerkin.reconstruct(a_n, x_c)
    u_h = galerkin.reconstruct(a_hat, x_c)
    ut = (u_h - u_n) / dt
    f = lambda a: galerkin.full_pde_rhs_on_grid(a, x_c, D, rho, beta)  # noqa: E731
    if scheme == "forward":
        r = f(a_n)
    elif scheme == "trapezoid":
        r = 0.5 * (f(a_n) + f(a_hat))
    else:
        r = f(0.5 * (a_n + a_hat))
    return ut - r


def pde_losses(a_n, a_hat, dt, phys, s_inc, s_field, x_c, scheme="forward"):
    D, rho, beta = phys
    Rg = galerkin_residual(a_n, a_hat, dt, D, rho, beta, scheme)
    Rf = full_pde_residual(a_n, a_hat, dt, D, rho, beta, x_c, scheme)
    L_gal = ((Rg * dt / s_inc) ** 2).sum(-1).mean()
    L_full = ((Rf * dt / s_field) ** 2).mean()
    return L_full, L_gal


def rollout_loss(model, a_windows, dt, s_inc):
    """a_windows: (B, H+1, 5).  Mean over b and s of ||a_hat_{n+s} - a_{n+s}||^2."""
    H = a_windows.shape[1] - 1
    pred = model.rollout(a_windows[:, 0], dt, H)
    return (((pred - a_windows[:, 1:]) / s_inc) ** 2).sum(-1).mean(), pred


def regularization(model, a_hat, x_c, K=None, pred_roll=None, bound=None, w=None):
    """Negativity + saturation + parameter-size + stability penalties (each returned)."""
    w = w or {}
    u = galerkin.reconstruct(a_hat, x_c)
    terms = {"neg": (torch.relu(-u) ** 2).mean()}
    terms["sat"] = (torch.relu(u - 1.2 * K) ** 2).mean() if K is not None and np.isfinite(K) else u.sum() * 0
    terms["param"] = model.hamiltonian_norm() if hasattr(model, "hamiltonian_norm") else u.sum() * 0
    if pred_roll is not None and bound is not None:
        terms["stab"] = (torch.relu(pred_roll.norm(dim=-1) - bound) ** 2).mean()
    else:
        terms["stab"] = u.sum() * 0
    total = sum(w.get(k, 0.0) * v for k, v in terms.items())
    return total, terms
