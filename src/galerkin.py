"""M = 2 Fourier-Galerkin truncation of the Fisher-KPP equation.

Field on the periodic domain x in [0, 2*pi):

    u_M(x,t) = sum_{k=-2}^{2} a_k(t) e^{ikx},   a_{-k} = conj(a_k).

Real coefficient vector (the 5 qubit features):

    a = (a0, b1, c1, b2, c2) = (a_0, Re a_1, Im a_1, Re a_2, Im a_2)

Galerkin system  da_k/dt = (rho - D k^2) a_k - beta * sum_{p+q=k} a_p a_q,
expanded into the five real equations (derivation in docs/QPINO_technical_report.md):

    da0/dt = rho a0      - beta [a0^2 + 2(b1^2 + c1^2 + b2^2 + c2^2)]
    db1/dt = (rho-D)  b1 - 2 beta [a0 b1 + b1 b2 + c1 c2]
    dc1/dt = (rho-D)  c1 - 2 beta [a0 c1 + b1 c2 - c1 b2]
    db2/dt = (rho-4D) b2 - beta [2 a0 b2 + b1^2 - c1^2]
    dc2/dt = (rho-4D) c2 - beta [2 a0 c2 + 2 b1 c1]

General M (added for the M=4 study): the real vector is
    a = (a0, Re a1, Im a1, ..., Re aM, Im aM)   (2M+1 entries = qubits),
and every function below infers M from the last dimension of `a` (len = 2M+1).  M = 2 keeps
the explicit equations above; M = 4 has its own explicit expansion (galerkin_rhs_m4_explicit,
derived with sympy, see docs/QPINO_technical_report.md section 14); any other M uses the
convolution form galerkin_rhs_conv.  All three agree to round-off (tests/test_galerkin_m4.py).

All functions accept torch tensors (differentiable) and most accept numpy arrays.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

M = 2
MODES = np.arange(-M, M + 1)
NAMES = ["a0", "Re a1", "Im a1", "Re a2", "Im a2"]
K_OF_COMPONENT = np.array([0, 1, 1, 2, 2])
K2 = np.array([0.0, 1.0, 1.0, 4.0, 4.0])


def m_of(a) -> int:
    """Truncation order M inferred from a real coefficient vector of length 2M+1."""
    n = a.shape[-1]
    if n % 2 != 1:
        raise ValueError(f"coefficient vector length {n} is not 2M+1")
    return (n - 1) // 2


def names(Mo: int = M):
    out = ["a0"]
    for k in range(1, Mo + 1):
        out += [f"Re a{k}", f"Im a{k}"]
    return out


def k_of_component(Mo: int = M):
    return np.array([0] + [k for k in range(1, Mo + 1) for _ in range(2)])


def uniform_grid(N: int, offset: float = 0.0) -> np.ndarray:
    """x_j = 2*pi*(j + offset)/N, j = 0..N-1 (offset=0.5 gives bin centres)."""
    return 2 * np.pi * (np.arange(N) + offset) / N


# ----------------------------------------------------------------------------
# Projection and reconstruction
# ----------------------------------------------------------------------------
def complex_coeffs(u, x=None, kmax: int = M):
    """a_k = (1/N) sum_j u(x_j) e^{-ikx_j}, k = 0..kmax (uniform periodic grid)."""
    is_t = torch.is_tensor(u)
    N = u.shape[-1]
    if x is None:
        x = uniform_grid(N)
    k = np.arange(kmax + 1)
    E = np.exp(-1j * np.outer(x, k)) / N  # (N, kmax+1)
    if is_t:
        Et = torch.as_tensor(E, dtype=torch.complex128)
        return u.to(torch.complex128) @ Et
    return np.asarray(u) @ E


def field_to_coeffs(u, x=None, Mo: int = M):
    """Project a real field (..., N) onto the 2M+1 real Galerkin coordinates (default M=2)."""
    ak = complex_coeffs(u, x, Mo)
    parts = [ak[..., 0].real]
    for k in range(1, Mo + 1):
        parts += [ak[..., k].real, ak[..., k].imag]
    if torch.is_tensor(ak):
        return torch.stack(parts, dim=-1)
    return np.stack(parts, axis=-1)


def _trigk(x, like, k):
    if torch.is_tensor(like):
        x = torch.as_tensor(x, dtype=like.dtype)
        return torch.cos(k * x), torch.sin(k * x)
    x = np.asarray(x)
    return np.cos(k * x), np.sin(k * x)


def reconstruct(a, x):
    """u_M(x) = a0 + 2 sum_k (b_k cos kx - c_k sin kx).  a: (...,2M+1) -> (...,len(x))"""
    u = a[..., 0:1]
    for k in range(1, m_of(a) + 1):
        ck, sk = _trigk(x, a, k)
        u = u + 2 * (a[..., 2 * k - 1:2 * k] * ck - a[..., 2 * k:2 * k + 1] * sk)
    return u


def reconstruct_uxx(a, x):
    """Analytic second derivative of u_M: -2 sum_k k^2 (b_k cos kx - c_k sin kx)."""
    u = 0 * a[..., 0:1]
    for k in range(1, m_of(a) + 1):
        ck, sk = _trigk(x, a, k)
        u = u - 2 * k * k * (a[..., 2 * k - 1:2 * k] * ck - a[..., 2 * k:2 * k + 1] * sk)
    return u


def to_complex_modes(a):
    """Real vector (...,2M+1) -> complex modes (..., 2M+1) ordered k = -M..M."""
    Mo = m_of(a)
    if torch.is_tensor(a):
        pos = [torch.complex(a[..., 2 * k - 1], a[..., 2 * k]) for k in range(1, Mo + 1)]
        return torch.stack([p.conj() for p in pos[::-1]] + [a[..., 0].to(torch.complex128)] + pos, dim=-1)
    a = np.asarray(a)
    pos = [a[..., 2 * k - 1] + 1j * a[..., 2 * k] for k in range(1, Mo + 1)]
    return np.stack([p.conj() for p in pos[::-1]] + [a[..., 0] + 0j] + pos, axis=-1)


def from_complex_modes(ac):
    """Complex modes ordered k=-M..M -> real (2M+1)-vector (uses k>=0 entries)."""
    Mo = (ac.shape[-1] - 1) // 2
    parts = [ac[..., Mo].real]
    for k in range(1, Mo + 1):
        parts += [ac[..., Mo + k].real, ac[..., Mo + k].imag]
    if torch.is_tensor(ac):
        return torch.stack(parts, dim=-1)
    return np.stack(parts, axis=-1)


# ----------------------------------------------------------------------------
# Right-hand sides
# ----------------------------------------------------------------------------
def quad_conv_complex(ac):
    """(u^2)_k = sum_{p+q=k, |p|,|q|<=M} a_p a_q for k=-M..M (complex, k-ordered)."""
    Mo = (ac.shape[-1] - 1) // 2
    out = []
    for k in range(-Mo, Mo + 1):
        s = 0
        for p in range(-Mo, Mo + 1):
            q = k - p
            if -Mo <= q <= Mo:
                s = s + ac[..., p + Mo] * ac[..., q + Mo]
        out.append(s)
    if torch.is_tensor(ac):
        return torch.stack(out, dim=-1)
    return np.stack(out, axis=-1)


def galerkin_rhs_complex(ac, D, rho, beta):
    """Complex Galerkin RHS for all k=-M..M (used to check conjugate symmetry)."""
    Mo = (ac.shape[-1] - 1) // 2
    k2 = np.arange(-Mo, Mo + 1).astype(float) ** 2
    if torch.is_tensor(ac):
        k2 = torch.as_tensor(k2)
    return (rho - D * k2) * ac - beta * quad_conv_complex(ac)


def galerkin_rhs_conv(a, D, rho, beta):
    """General-M real Galerkin RHS via the complex convolution (any M)."""
    return from_complex_modes(galerkin_rhs_complex(to_complex_modes(a), D, rho, beta))


def galerkin_rhs(a, D, rho, beta):
    """Real Galerkin RHS.  Explicit equations for M=2 (5 entries) and M=4 (9 entries),
    convolution form for any other M."""
    n = a.shape[-1]
    if n == 5:
        return galerkin_rhs_m2_explicit(a, D, rho, beta)
    if n == 9:
        return galerkin_rhs_m4_explicit(a, D, rho, beta)
    return galerkin_rhs_conv(a, D, rho, beta)


def galerkin_rhs_m4_explicit(a, D, rho, beta):
    """Explicit expanded real M=4 Galerkin equations (9 components; sympy-derived).
    q_k = (u_M^2)_k restricted to |k|<=4; rhs_k = (rho - D k^2) a_k - beta q_k."""
    a0, b1, c1, b2, c2, b3, c3, b4, c4 = [a[..., i] for i in range(9)]
    q0 = a0 ** 2 + 2 * (b1 ** 2 + c1 ** 2 + b2 ** 2 + c2 ** 2 + b3 ** 2 + c3 ** 2 + b4 ** 2 + c4 ** 2)
    qb1 = 2 * (a0 * b1 + b1 * b2 + b2 * b3 + b3 * b4 + c1 * c2 + c2 * c3 + c3 * c4)
    qc1 = 2 * (a0 * c1 + b1 * c2 - b2 * c1 + b2 * c3 - b3 * c2 + b3 * c4 - b4 * c3)
    qb2 = 2 * a0 * b2 + b1 ** 2 - c1 ** 2 + 2 * (b1 * b3 + b2 * b4 + c1 * c3 + c2 * c4)
    qc2 = 2 * (a0 * c2 + b1 * c1 + b1 * c3 + b2 * c4 - b3 * c1 - b4 * c2)
    qb3 = 2 * (a0 * b3 + b1 * b2 + b1 * b4 - c1 * c2 + c1 * c4)
    qc3 = 2 * (a0 * c3 + b1 * c2 + b1 * c4 + b2 * c1 - b4 * c1)
    qb4 = 2 * a0 * b4 + 2 * b1 * b3 + b2 ** 2 - 2 * c1 * c3 - c2 ** 2
    qc4 = 2 * (a0 * c4 + b1 * c3 + b2 * c2 + b3 * c1)
    lin = [rho, rho - D, rho - D, rho - 4 * D, rho - 4 * D, rho - 9 * D, rho - 9 * D, rho - 16 * D, rho - 16 * D]
    q = [q0, qb1, qc1, qb2, qc2, qb3, qc3, qb4, qc4]
    v = [a0, b1, c1, b2, c2, b3, c3, b4, c4]
    f = [lin[i] * v[i] - beta * q[i] for i in range(9)]
    if torch.is_tensor(a):
        return torch.stack(f, dim=-1)
    return np.stack(f, axis=-1)


def galerkin_rhs_m2_explicit(a, D, rho, beta):
    """Explicit expanded real M=2 Galerkin equations.  a: (...,5)."""
    a0, b1, c1, b2, c2 = [a[..., i] for i in range(5)]
    f0 = rho * a0 - beta * (a0 ** 2 + 2 * (b1 ** 2 + c1 ** 2 + b2 ** 2 + c2 ** 2))
    f1 = (rho - D) * b1 - 2 * beta * (a0 * b1 + b1 * b2 + c1 * c2)
    f2 = (rho - D) * c1 - 2 * beta * (a0 * c1 + b1 * c2 - c1 * b2)
    f3 = (rho - 4 * D) * b2 - beta * (2 * a0 * b2 + b1 ** 2 - c1 ** 2)
    f4 = (rho - 4 * D) * c2 - beta * (2 * a0 * c2 + 2 * b1 * c1)
    if torch.is_tensor(a):
        return torch.stack([f0, f1, f2, f3, f4], dim=-1)
    return np.stack([f0, f1, f2, f3, f4], axis=-1)


def galerkin_rhs_by_projection(a, D, rho, beta, N: int = 64):
    """Direct spectral projection: evaluate D u_xx + rho u - beta u^2 for u = u_M on a fine
    grid and project back onto |k|<=M.  With N >= 3*M+1 there is no aliasing, so this must
    equal galerkin_rhs exactly (to round-off)."""
    x = uniform_grid(N)
    u = reconstruct(a, x)
    uxx = reconstruct_uxx(a, x)
    f = D * uxx + rho * u - beta * u ** 2
    return field_to_coeffs(f, x, m_of(a))


def full_pde_rhs_on_grid(a, x, D, rho, beta):
    """D u_xx + rho u - beta u^2 for the reconstructed u_M on collocation points x.
    Includes the |k|=3,4 modes generated by u_M^2 that the Galerkin system discards."""
    u = reconstruct(a, x)
    return D * reconstruct_uxx(a, x) + rho * u - beta * u ** 2


def coupling_hessian(D=0.1, rho=1.0, beta=1.0, Mo: int = M):
    """|d^2 f_i / d a_j d a_k| summed over i.  Non-zero off-diagonal entries define the
    'Galerkin convolution' coupling graph used by the structured ansatz."""
    n = 2 * Mo + 1
    a = torch.zeros(n, dtype=torch.float64)
    Hs = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        h = torch.autograd.functional.hessian(lambda v: galerkin_rhs(v, D, rho, beta)[i], a)
        Hs += h.abs()
    return Hs.numpy()


# ----------------------------------------------------------------------------
# Time integration
# ----------------------------------------------------------------------------
SOLVE_IVP_KW = dict(method="DOP853", rtol=1e-10, atol=1e-12)


def integrate_galerkin(a0, t_eval, D, rho, beta, **kw):
    """Integrate the (2M+1)-D real Galerkin ODE with scipy (DOP853, rtol=1e-10, atol=1e-12)."""
    opts = dict(SOLVE_IVP_KW)
    opts.update(kw)
    sol = solve_ivp(lambda t, y: galerkin_rhs(y, D, rho, beta), (t_eval[0], t_eval[-1]),
                    np.asarray(a0, float), t_eval=t_eval, **opts)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def rk4_step(a, dt, D, rho, beta, substeps: int = 8):
    """Differentiable classical RK4 flow map for the Galerkin system (torch or numpy)."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = galerkin_rhs(a, D, rho, beta)
        k2 = galerkin_rhs(a + 0.5 * h * k1, D, rho, beta)
        k3 = galerkin_rhs(a + 0.5 * h * k2, D, rho, beta)
        k4 = galerkin_rhs(a + h * k3, D, rho, beta)
        a = a + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return a


def fit_galerkin_parameters(a_n, a_np1, dt, w=None, n_starts: int = 6, seed: int = 0,
                            init=None, iters: int = 400, fit_D: bool = True):
    """Least-squares fit of (D, rho, beta) > 0 to one-step pairs a_n -> a_{n+1} using the
    RK4 Galerkin flow map.  Parameters are log-parameterised (strictly positive).
    Returns dict(D, rho, beta, loss) of the best of several starts."""
    a_n = torch.as_tensor(a_n, dtype=torch.float64)
    a_np1 = torch.as_tensor(a_np1, dtype=torch.float64)
    if w is None:
        w = torch.ones(a_n.shape[-1], dtype=torch.float64)
    w = torch.as_tensor(w, dtype=torch.float64)
    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_starts):
        if init is not None and s == 0:
            p0 = np.log(np.asarray(init, float))
        else:
            p0 = np.array([rng.uniform(-6, 1), rng.uniform(-5, 1), rng.uniform(-5, 1)])
        logp = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([logp], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            D = torch.exp(logp[0]) if fit_D else torch.tensor(1e-12, dtype=torch.float64)
            pred = rk4_step(a_n, dt, D, torch.exp(logp[1]), torch.exp(logp[2]), substeps=4)
            loss = (((pred - a_np1) * w) ** 2).mean()
            loss.backward()
            return loss

        try:
            opt.step(closure)
            with torch.no_grad():
                D = torch.exp(logp[0]) if fit_D else torch.tensor(0.0, dtype=torch.float64)
                pred = rk4_step(a_n, dt, D, torch.exp(logp[1]), torch.exp(logp[2]), substeps=4)
                loss = float((((pred - a_np1) * w) ** 2).mean())
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(loss) and (best is None or loss < best["loss"]):
            p = torch.exp(logp.detach()).numpy()
            best = dict(D=float(p[0]) if fit_D else 0.0, rho=float(p[1]), beta=float(p[2]), loss=loss)
    return best
