"""Classical reference solver for u_t = D u_xx + rho u - beta u^2 on a periodic domain.

Method of lines: Fourier pseudo-spectral (default) or 2nd-order central finite differences
in space; scipy.integrate.solve_ivp in time (DOP853, rtol=1e-9, atol=1e-11 by default).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

DEFAULT_TOL = dict(rtol=1e-9, atol=1e-11)


def wavenumbers(N, L=2 * np.pi):
    return 2 * np.pi * np.fft.fftfreq(N, d=L / N)


def laplacian(u, L=2 * np.pi, method="spectral"):
    N = u.shape[-1]
    if method == "spectral":
        k = wavenumbers(N, L)
        return np.real(np.fft.ifft(-(k ** 2) * np.fft.fft(u, axis=-1), axis=-1))
    dx = L / N
    return (np.roll(u, -1, -1) - 2 * u + np.roll(u, 1, -1)) / dx ** 2


def rhs(u, D, rho, beta, L=2 * np.pi, method="spectral"):
    return D * laplacian(u, L, method) + rho * u - beta * u ** 2


def solve(u0, t_eval, D, rho, beta, L=2 * np.pi, method="spectral", integrator="DOP853", **tol):
    """Return u(t_eval) with shape (len(t_eval), N)."""
    opts = dict(DEFAULT_TOL)
    opts.update(tol)
    sol = solve_ivp(lambda t, y: rhs(y, D, rho, beta, L, method), (t_eval[0], t_eval[-1]),
                    np.asarray(u0, float), t_eval=t_eval, method=integrator, **opts)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def step(u, dt, D, rho, beta, **kw):
    """Classical flow map u(x,t) -> u(x,t+dt)."""
    return solve(u, np.array([0.0, dt]), D, rho, beta, **kw)[-1]


# ---------------------------------------------------------------- analytic checks
def logistic_solution(u0, t, rho, beta):
    """Spatially uniform solution: u = K u0 e^{rho t} / (K + u0 (e^{rho t} - 1)), K = rho/beta."""
    K = rho / beta
    e = np.exp(rho * t)
    return K * u0 * e / (K + u0 * (e - 1))


def linear_mode_solution(x, t, k, amp, D, rho):
    """For beta = 0: u = amp cos(kx) exp((rho - D k^2) t)."""
    return amp * np.cos(k * x)[None, :] * np.exp((rho - D * k ** 2) * np.asarray(t))[:, None]


# ---------------------------------------------------------------- synthetic data
def random_initial_condition(x, rng, c0_range=(0.05, 0.35), kmax=2):
    """u0 = c0 + sum_{k<=kmax} A_k cos(k x + phi_k), positive by construction."""
    c0 = rng.uniform(*c0_range)
    u = np.full_like(x, c0)
    for k in range(1, kmax + 1):
        A = rng.uniform(0.0, c0 / (2.2 * kmax) * 2)
        u += A * np.cos(k * x + rng.uniform(0, 2 * np.pi))
    return np.clip(u, 1e-4, None)


def generate_trajectories(n_traj, D, rho, beta, dt, n_steps, N=128, seed=0, kmax=2):
    """Full-resolution Fisher-KPP trajectories from random low-mode positive ICs.
    Returns (t, x, U) with U shape (n_traj, n_steps+1, N)."""
    rng = np.random.default_rng(seed)
    x = 2 * np.pi * np.arange(N) / N
    t = dt * np.arange(n_steps + 1)
    U = np.stack([solve(random_initial_condition(x, rng, kmax=kmax), t, D, rho, beta)
                  for _ in range(n_traj)])
    return t, x, U
