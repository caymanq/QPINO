"""Phase-12 check 8: the PDE / Galerkin residuals of accurate reference solutions vanish as
numerical resolution improves (dt -> 0 for the time discretisation; M -> inf for the
unresolved-mode part of the full reconstructed residual)."""
import numpy as np
import torch

from src import fisher_kpp as fk
from src import galerkin as gk
from src import losses

PH = (0.1, 1.0, 1.0)


def _gal_traj(dt, T=1.0):
    a0 = np.array([0.25, 0.06, -0.03, 0.04, 0.02])
    t = np.arange(0, T + 1e-12, dt)
    return torch.tensor(gk.integrate_galerkin(a0, t, *PH))


def test_galerkin_residual_converges(record):
    out = {}
    for sch, order_expect in [("forward", 1.0), ("trapezoid", 2.0), ("midpoint", 2.0)]:
        rms = []
        for dt in [0.1, 0.05, 0.025, 0.0125]:
            A = _gal_traj(dt)
            R = losses.galerkin_residual(A[:-1], A[1:], dt, *PH, scheme=sch)
            rms.append(float(R.pow(2).mean().sqrt()))
        order = np.log2(rms[-2] / rms[-1])
        out[f"{sch}_rms_dt0.1"], out[f"{sch}_rms_dt0.0125"], out[f"{sch}_order"] = rms[0], rms[-1], order
        assert rms[0] > rms[1] > rms[2] > rms[3]
        assert abs(order - order_expect) < 0.25
    record(**out)


def _truncate(u, Mk):
    U = np.fft.fft(u, axis=-1)
    N = u.shape[-1]
    U[..., Mk + 1:N - Mk] = 0
    return np.fft.ifft(U, axis=-1).real


def _full_residual_M(U, dt, Mk, x):
    """Trapezoid full Fisher-KPP residual of the |k|<=M truncation of a reference solution."""
    UM = _truncate(U, Mk)
    f = fk.rhs(UM, *PH)
    R = (UM[1:] - UM[:-1]) / dt - 0.5 * (f[1:] + f[:-1])
    return float(np.sqrt((R ** 2).mean()))


def test_full_pde_residual_converges(record):
    """Full residual on the reconstructed field of an accurate full-PDE solution.
    (a) For M=2 it plateaus at the unresolved-mode (|k|=3,4 of beta u^2) level as dt -> 0;
    (b) it -> 0 when both dt -> 0 and the truncation M grows."""
    N = 128
    x = 2 * np.pi * np.arange(N) / N
    u0 = 0.25 + 0.12 * np.cos(x) + 0.08 * np.sin(2 * x)
    xc = gk.uniform_grid(128)
    rms_M2 = []
    for dt in [0.1, 0.05, 0.025, 0.0125]:
        t = np.arange(0, 1.0 + 1e-12, dt)
        A = torch.tensor(gk.field_to_coeffs(fk.solve(u0, t, *PH), x))
        R = losses.full_pde_residual(A[:-1], A[1:], dt, *PH, xc, scheme="trapezoid")
        rms_M2.append(float(R.pow(2).mean().sqrt()))
    res_M = {}
    for Mk, dt in [(2, 0.0125), (4, 0.0125), (8, 0.00625), (16, 0.003125)]:
        t = np.arange(0, 1.0 + 1e-12, dt)
        res_M[Mk] = _full_residual_M(fk.solve(u0, t, *PH, rtol=1e-12, atol=1e-14), dt, Mk, x)
    # Galerkin-exact trajectory: full residual = only the discarded |k|=3,4 part of beta u^2
    A = _gal_traj(0.0125)
    Rg = losses.full_pde_residual(A[:-1], A[1:], 0.0125, *PH, xc, scheme="trapezoid")
    u = gk.reconstruct(A, xc)
    U2 = np.fft.fft(u.numpy() ** 2, axis=-1) / 128
    unresolved = np.zeros_like(U2); unresolved[:, 3:5] = U2[:, 3:5]; unresolved[:, -4:-2] = U2[:, -4:-2]
    unres_rms = float(np.sqrt(((PH[2] * np.fft.ifft(unresolved * 128, axis=-1).real) ** 2).mean()))
    record(full_rms_M2_by_dt=rms_M2, **{f"full_rms_M{k}": v for k, v in res_M.items()},
           galerkin_traj_full_residual=float(Rg.pow(2).mean().sqrt()), unresolved_mode_rms=unres_rms)
    assert abs(res_M[2] - rms_M2[-1]) / rms_M2[-1] < 0.05  # two independent implementations agree
    assert res_M[2] > res_M[4] > res_M[8] > res_M[16] and res_M[16] < 1e-4
    assert abs(float(Rg.pow(2).mean().sqrt()) - unres_rms) / unres_rms < 0.05
