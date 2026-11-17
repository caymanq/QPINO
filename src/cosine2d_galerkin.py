"""2D real cosine / Neumann Galerkin basis on the contiguous-USA land mask.

Basis on the bounding rectangle [0, Lx] x [0, Ly] (non-dimensional: lengths in units of Ly/pi, so
y in [0, pi], x in [0, pi Lx/Ly]):

    phi_kl(x, y) = cos(k pi x / Lx) cos(l pi y / Ly),   k, l = 0..K-1,   one real b_kl per qubit
    (component index j = k*K + l;  K = 3 -> 9 qubits, K = 4 -> 16 qubits).

Land mask, handled honestly: the field only exists on land cells Omega (17,609 of 32,768 cells at
256 x 128).  Coefficients are the L2(Omega) least-squares projection  b = G^{-1} Phi^T W u  with the
land mass matrix G_ij = <phi_i, phi_j>_Omega (ocean / Canada / Mexico cells carry no data and no
weight).  The dynamics are the weak (Galerkin) form of Fisher-KPP on Omega,

    G db/dt = -D K b + rho G b - beta C(b, b),
    K_ij = <grad phi_i, grad phi_j>_Omega,   C_i(b, b) = <phi_i, (Phi b)^2>_Omega,

whose natural boundary condition is no-flux across the (staircase) coastline / border, i.e. the
Neumann condition on the land domain itself, not on the rectangle.  On the full rectangle
(mask == 1, fine grid) G is diagonal and the system reduces to the tensor-product cosine triads
(tested in tests/test_cosine2d.py).  All integrals are midpoint quadrature on the cells.
"""
from __future__ import annotations

import numpy as np
import torch

_S = {}   # active geometry: set by setup()


def modes(K):
    return [(k, l) for k in range(K) for l in range(K)]


def basis(xy, K, Lx, Ly):
    """phi_j at points xy (P, 2) -> (P, K*K), and its gradient (P, K*K, 2)."""
    x, y = xy[:, 0], xy[:, 1]
    km = np.array([k for k, _ in modes(K)], float) * np.pi / Lx
    lm = np.array([l for _, l in modes(K)], float) * np.pi / Ly
    cx, cy = np.cos(np.outer(x, km)), np.cos(np.outer(y, lm))
    sx, sy = np.sin(np.outer(x, km)), np.sin(np.outer(y, lm))
    phi = cx * cy
    grad = np.stack([-km * sx * cy, -lm * cx * sy], -1)
    lap = -(km ** 2 + lm ** 2) * phi
    return phi, grad, lap


def cell_centres(NX, NY, Lx, Ly):
    xc = (np.arange(NX) + 0.5) * Lx / NX
    yc = (np.arange(NY) + 0.5) * Ly / NY
    X, Y = np.meshgrid(xc, yc)               # (NY, NX), row 0 = south
    return X, Y


def setup(mask, K, Lx, Ly):
    """Precompute land-mask operators.  mask: (NY, NX) with 1 = land."""
    NY, NX = mask.shape
    X, Y = cell_centres(NX, NY, Lx, Ly)
    land = mask.ravel() > 0
    xy = np.c_[X.ravel()[land], Y.ravel()[land]]
    w = np.full(len(xy), (Lx / NX) * (Ly / NY))
    phi, grad, lap = basis(xy, K, Lx, Ly)
    G = (phi * w[:, None]).T @ phi
    Kst = np.einsum("p,pid,pjd->ij", w, grad, grad)
    C3 = np.einsum("p,pi,pj,pk->ijk", w, phi, phi, phi)
    Gi = np.linalg.inv(G)
    _S.clear()
    _S.update(K=K, n=K * K, Lx=Lx, Ly=Ly, mask=mask, land=land, xy=xy, w=w, phi=phi, G=G, Kst=Kst,
              C3=C3, A_lin=Gi @ Kst, A_quad=np.einsum("ai,ijk->ajk", Gi, C3), cond_G=float(np.linalg.cond(G)),
              proj=Gi @ (phi * w[:, None]).T)
    _S["t"] = {k: torch.tensor(_S[k]) for k in ("A_lin", "A_quad")}
    return _S


def active():
    return _S


def field_to_coeffs(u, x=None, K=None):
    """u: (..., NY, NX) full grid (ocean ignored) or (..., P_land) -> (..., K*K) land LS projection."""
    u = np.asarray(u)
    if u.shape[-2:] == _S["mask"].shape:
        u = u.reshape(*u.shape[:-2], -1)[..., _S["land"]]
    return u @ _S["proj"].T


def _phi_at(x):
    phi, _, lap = basis(np.asarray(x), _S["K"], _S["Lx"], _S["Ly"])
    return phi, lap


_PC = {}


def _cached(x, like):
    key = (id(x), torch.is_tensor(like))
    if key not in _PC:
        phi, lap = _phi_at(x)
        _PC[key] = (torch.tensor(phi), torch.tensor(lap)) if torch.is_tensor(like) else (phi, lap)
    return _PC[key]


def reconstruct(b, x):
    """u at points x (P, 2) (non-dimensional coordinates)."""
    phi, _ = _cached(x, b)
    return b @ phi.T


def reconstruct_lap(b, x):
    _, lap = _cached(x, b)
    return b @ lap.T


def reconstruct_grid(b):
    """(..., n) -> (..., NY, NX) with NaN outside the land mask."""
    b = np.asarray(b)
    out = np.full(b.shape[:-1] + (_S["mask"].size,), np.nan)
    out[..., _S["land"]] = b @ _S["phi"].T
    return out.reshape(b.shape[:-1] + _S["mask"].shape)


def galerkin_rhs(b, D, rho, beta):
    """db/dt = -D G^-1 K b + rho b - beta G^-1 C(b, b)   (weak form on the land domain)."""
    if torch.is_tensor(b):
        Al, Aq = _S["t"]["A_lin"], _S["t"]["A_quad"]
        return -D * b @ Al.T + rho * b - beta * torch.einsum("ajk,...j,...k->...a", Aq, b, b)
    return -D * b @ _S["A_lin"].T + rho * b - beta * np.einsum("ajk,...j,...k->...a", _S["A_quad"], b, b)


def rect_rhs_tensor_product(b, D, rho, beta, K, Lx, Ly):
    """Reference for the full rectangle: tensor-product cosine triads (1D T in each direction)."""
    from .cosine_galerkin import triad_tensor
    T = triad_tensor(K - 1)
    B = np.asarray(b).reshape(K, K)
    k = np.arange(K)
    lam = (k[:, None] * np.pi / Lx) ** 2 + (k[None, :] * np.pi / Ly) ** 2
    quad = np.einsum("kpq,lrs,pr,qs->kl", T, T, B, B)
    return ((rho - D * lam) * B - beta * quad).ravel()


def coupling_hessian(n):
    Hs = np.abs(_S["A_quad"] + _S["A_quad"].transpose(0, 2, 1)).sum(0)
    return Hs
