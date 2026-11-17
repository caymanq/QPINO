"""Cosine (Neumann / no-flux) Fourier-Galerkin basis for Fisher-KPP.

Physical domain [0, L] with u_x = 0 at both ends is mapped to x = pi * s / L in [0, pi]:

    u_M(x, t) = sum_{k=0}^{M} b_k(t) cos(k x),     b_k real  (M+1 coefficients = qubits)

Projection (orthogonality on [0, pi]):  b_0 = (1/pi) int u dx,  b_k = (2/pi) int u cos(kx) dx.
Using cos(px) cos(qx) = 1/2 [cos((p-q)x) + cos((p+q)x)] and cos(-jx) = cos(jx):

    (u_M^2)_k = sum_{p,q=0}^{M} T_kpq b_p b_q,
    T_kpq = 1/2 [ delta(|p-q| = k) + delta(p+q = k) ]           (same formula for k = 0 and k > 0)

The k=0 vs k>0 normalisations cancel: a cos(jx) term with j = k projects onto b_k with weight 1
for every k (cos(0 x) = 1 projects onto b_0 with weight 1, cos(kx) onto b_k with (2/pi) int cos^2 = 1).
E.g. (u^2)_0 = b_0^2 + 1/2 sum_{p>=1} b_p^2.  Galerkin system (physical D_phys = D (L/pi)^2):

    db_k/dt = (rho - D k^2) b_k - beta sum_{p,q} T_kpq b_p b_q,    k = 0..M.

Products b_p b_q with p+q > M (and the |p-q| = k terms are always resolved) generate modes
k > M that the truncation discards.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch


def triad_tensor(M: int) -> np.ndarray:
    """T[k, p, q] = 1/2 [delta(|p-q|, k) + delta(p+q, k)] for k, p, q = 0..M."""
    T = np.zeros((M + 1,) * 3)
    for k, p, q in itertools.product(range(M + 1), repeat=3):
        T[k, p, q] = 0.5 * ((abs(p - q) == k) + (p + q == k))
    return T


_TC = {}


def _T(M, like):
    key = (M, torch.is_tensor(like))
    if key not in _TC:
        T = triad_tensor(M)
        _TC[key] = torch.tensor(T) if torch.is_tensor(like) else T
    return _TC[key]


def m_of(b) -> int:
    return b.shape[-1] - 1


def grid(N: int) -> np.ndarray:
    """Cell centres x_j = pi (j + 1/2) / N on [0, pi] (the 32 I-10 bins for N = 32)."""
    return np.pi * (np.arange(N) + 0.5) / N


def field_to_coeffs(u, x=None, M: int = 8):
    """Discrete cosine projection on cell centres (exact DCT-II normalisation):
    b_0 = mean(u), b_k = (2/N) sum_j u_j cos(k x_j).  Exact for |k| <= M < N."""
    N = u.shape[-1]
    x = grid(N) if x is None else x
    k = np.arange(M + 1)
    C = np.cos(np.outer(x, k)) * (2.0 / N)
    C[:, 0] = 1.0 / N
    if torch.is_tensor(u):
        return u @ torch.as_tensor(C)
    return np.asarray(u) @ C


def _cos(x, like, M):
    k = np.arange(M + 1)
    C = np.cos(np.outer(np.asarray(x), k))           # (len(x), M+1)
    return torch.as_tensor(C) if torch.is_tensor(like) else C


def reconstruct(b, x):
    """u_M(x) = sum_k b_k cos(k x).  b: (..., M+1) -> (..., len(x))."""
    return b @ _cos(x, b, m_of(b)).T


def reconstruct_uxx(b, x):
    M = m_of(b)
    k2 = np.arange(M + 1) ** 2.0
    C = _cos(x, b, M) * (-(torch.as_tensor(k2) if torch.is_tensor(b) else k2))
    return b @ C.T


def quad(b):
    """(u_M^2)_k for k = 0..M via the triad tensor."""
    T = _T(m_of(b), b)
    if torch.is_tensor(b):
        return torch.einsum("kpq,...p,...q->...k", T, b, b)
    return np.einsum("kpq,...p,...q->...k", T, b, b)


def quad_explicit(b):
    """Same as quad, written as the explicit double loop over cos p cos q products."""
    M = m_of(b)
    out = [0 * b[..., 0] for _ in range(M + 1)]
    for p in range(M + 1):
        for q in range(M + 1):
            pr = 0.5 * b[..., p] * b[..., q]
            if abs(p - q) <= M:
                out[abs(p - q)] = out[abs(p - q)] + pr
            if p + q <= M:
                out[p + q] = out[p + q] + pr
    return torch.stack(out, -1) if torch.is_tensor(b) else np.stack(out, -1)


def galerkin_rhs(b, D, rho, beta):
    M = m_of(b)
    k2 = np.arange(M + 1) ** 2.0
    if torch.is_tensor(b):
        k2 = torch.as_tensor(k2)
    return (rho - D * k2) * b - beta * quad(b)


def galerkin_rhs_by_projection(b, D, rho, beta, N: int = 256):
    """Evaluate D u_xx + rho u - beta u^2 for u = u_M on a fine cell-centred grid and project
    back with the discrete cosine transform (no aliasing for N > 2M)."""
    x = grid(N)
    u = reconstruct(b, x)
    f = D * reconstruct_uxx(b, x) + rho * u - beta * u ** 2
    return field_to_coeffs(f, x, m_of(b))


def coupling_hessian(M: int, D=0.1, rho=1.0, beta=1.0):
    """sum_k |T_kpq + T_kqp| -> non-zero (p, q) define the coupling graph."""
    T = triad_tensor(M)
    return np.abs(T + T.transpose(0, 2, 1)).sum(0)


def coupling_stats(M: int):
    T = triad_tensor(M)
    n = M + 1
    H = coupling_hessian(M)
    edges = [(p, q) for p, q in itertools.combinations(range(n), 2) if H[p, q] > 0]
    # per-equation sparsity: how many distinct pairs {p<=q} enter equation k
    per_k = [int(sum(1 for p in range(n) for q in range(p, n) if T[k, p, q] + T[k, q, p] > 0)) for k in range(n)]
    return dict(n_qubits=n, n_edges=len(edges), n_pairs=n * (n - 1) // 2,
                tensor_nonzeros=int((T > 0).sum()), tensor_size=int(T.size),
                pairs_per_equation=per_k, total_pairs_incl_diag=n * (n + 1) // 2)


# ---------------------------------------------------------------- Neumann reference solvers
def neumann_fd_rhs(u, D, rho, beta, dx):
    """2nd-order cell-centred finite differences with reflecting ghost cells (u_x = 0)."""
    up = np.concatenate([u[..., :1], u, u[..., -1:]], -1)
    return D * (up[..., 2:] - 2 * u + up[..., :-2]) / dx ** 2 + rho * u - beta * u ** 2


def solve_neumann_fd(u0, t_eval, D, rho, beta, rtol=1e-10, atol=1e-12):
    from scipy.integrate import solve_ivp
    N = len(u0)
    dx = np.pi / N
    sol = solve_ivp(lambda t, y: neumann_fd_rhs(y, D, rho, beta, dx), (t_eval[0], t_eval[-1]),
                    np.asarray(u0, float), t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def solve_neumann_spectral(u0, t_eval, D, rho, beta, **tol):
    """Exact Neumann solution via the even extension on [0, 2 pi) and the periodic
    pseudo-spectral solver (an even IC stays even; u_x = 0 at 0 and pi)."""
    from . import fisher_kpp as fk
    ue = np.concatenate([u0, u0[::-1]])
    return fk.solve(ue, t_eval, D, rho, beta, **tol)[:, :len(u0)]


def random_neumann_ic(x, rng, c0_range=(0.05, 0.35), kmax=3):
    """u0 = c0 + sum_{k=1..kmax} A_k cos(k x), A_k ~ U(-c0/kmax, c0/kmax), positive."""
    c0 = rng.uniform(*c0_range)
    u = np.full_like(x, c0)
    for k in range(1, kmax + 1):
        u += rng.uniform(-c0 / kmax, c0 / kmax) * np.cos(k * x)
    return np.clip(u, 1e-4, None)


def generate_neumann_trajectories(n_traj, D, rho, beta, dt, n_steps, N=128, seed=0, kmax=3):
    rng = np.random.default_rng(seed)
    x = grid(N)
    t = dt * np.arange(n_steps + 1)
    U = np.stack([solve_neumann_spectral(random_neumann_ic(x, rng, kmax=kmax), t, D, rho, beta,
                                         rtol=1e-10, atol=1e-12) for _ in range(n_traj)])
    return t, x, U
