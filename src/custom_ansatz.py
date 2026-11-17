"""Trainable Hamiltonian-inspired ansatz and baseline circuits.

    H_theta = sum_j h_j Z_j + sum_j g_j X_j + sum_{(j,k) in E} J_jk Z_j Z_k
    U_theta(tau) = exp(-i H_theta tau)                       (exact, torch.linalg.matrix_exp)
                 ~ [prod_j e^{-i h_j Z_j dtau} prod_j e^{-i g_j X_j dtau}
                    prod_{jk} e^{-i J_jk Z_j Z_k dtau}]^L,  dtau = tau / L   (1st-order Trotter)

h_j : local phase rotation of feature qubit j (precession about z)
g_j : single-qubit mixing (transverse field) - turns R_y amplitudes into phases and back
J_jk: Ising interactions between feature qubits, restricted to a coupling graph E
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn

from . import galerkin
from .quantum_embedding import CDT, X, Z, kron_all, pauli_on, z_signs

N_QUBITS = 5


# ---------------------------------------------------------------- coupling graphs
def _M(n):
    return galerkin.n_to_M(n)


def galerkin_graph(n=N_QUBITS, tol=1e-12):
    """Edges (j,k) for which a_j a_k appears in some Galerkin equation (Hessian != 0).
    n = 2M+1 qubits (M=2: 9 of 10 pairs; M=4: 34 of 36 pairs)."""
    Hs = galerkin.coupling_hessian(Mo=_M(n))
    return [(j, k) for j, k in itertools.combinations(range(n), 2) if Hs[j, k] > tol]


def triad_graph(n=N_QUBITS):
    """Only the k>=1 mode-transfer couplings (drops the a0 'mean-field' edges)."""
    return [(j, k) for (j, k) in galerkin_graph(n) if j > 0 and k > 0]


def full_graph(n=N_QUBITS):
    return list(itertools.combinations(range(n), 2))


def nn_graph(n=N_QUBITS):
    return [(j, j + 1) for j in range(n - 1)]


GRAPHS = {"structured": galerkin_graph, "triad": triad_graph, "full": full_graph,
          "nearest": nn_graph, "none": lambda n=N_QUBITS: []}


def get_graph(name, n=N_QUBITS):
    return GRAPHS[name](n)


# ---------------------------------------------------------------- Hamiltonian ansatz
class HamiltonianAnsatz(nn.Module):
    def __init__(self, edges, n=N_QUBITS, init_scale=0.5, seed=0, use_x=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n, self.edges, self.use_x = n, list(edges), use_x
        self.h = nn.Parameter(init_scale * torch.randn(n, generator=g, dtype=torch.float64))
        self.g = nn.Parameter(init_scale * torch.randn(n, generator=g, dtype=torch.float64))
        self.J = nn.Parameter(init_scale * torch.randn(len(self.edges), generator=g, dtype=torch.float64))
        s = z_signs(n)
        self.register_buffer("zdiag", torch.tensor(s.T.copy()))  # (n, 2^n)
        zz = np.stack([s[:, j] * s[:, k] for j, k in self.edges]) if self.edges else np.zeros((0, 2 ** n))
        self.register_buffer("zzdiag", torch.tensor(zz))
        self.register_buffer("Xmats", torch.tensor(np.stack([pauli_on(X, j, n) for j in range(n)])))

    def n_params(self):
        return 2 * self.n + len(self.edges)

    def hamiltonian(self):
        diag = self.h @ self.zdiag
        if len(self.edges):
            diag = diag + self.J @ self.zzdiag
        H = torch.diag(diag).to(CDT)
        if self.use_x:
            H = H + torch.einsum("j,jab->ab", self.g.to(CDT), self.Xmats)
        return H

    def unitary(self, tau, method="exact", L=1):
        if method == "exact":
            return torch.linalg.matrix_exp(-1j * self.hamiltonian() * tau)
        if method == "exact_eigh":
            return expm_hermitian(self.hamiltonian(), tau)
        return self.trotter_unitary(tau, L)

    def trotter_unitary(self, tau, L):
        dtau = tau / L
        dz = torch.exp(-1j * (self.h @ self.zdiag).to(CDT) * dtau)
        if len(self.edges):
            dzz = torch.exp(-1j * (self.J @ self.zzdiag).to(CDT) * dtau)
        else:
            dzz = torch.ones_like(dz)
        if self.use_x:
            xs = [torch.cos(gj * dtau) * torch.eye(2, dtype=CDT) - 1j * torch.sin(gj * dtau) *
                  torch.tensor(X, dtype=CDT) for gj in self.g]
            UX = xs[0]
            for m in xs[1:]:
                UX = torch.kron(UX, m)
        else:
            UX = torch.eye(2 ** self.n, dtype=CDT)
        # matrix order  prod_Z . prod_X . prod_ZZ  (ZZ acts first on the state)
        step = dz[:, None] * UX * dzz[None, :]
        U = torch.eye(2 ** self.n, dtype=CDT)
        for _ in range(L):
            U = step @ U
        return U


# ---------------------------------------------------------------- hardware-efficient baseline
def _cnot(n, c, t):
    dim = 2 ** n
    P = np.zeros((dim, dim))
    for b in range(dim):
        bit_c = (b >> (n - 1 - c)) & 1
        b2 = b ^ (1 << (n - 1 - t)) if bit_c else b
        P[b2, b] = 1
    return P


class HardwareEfficientAnsatz(nn.Module):
    """Generic HEA: n_layers x [RY(theta) RZ(phi) on every qubit -> CNOT ring].
    Data-independent and tau-independent (it has no notion of evolution time)."""

    def __init__(self, n=N_QUBITS, n_layers=2, seed=0, init_scale=0.5):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n, self.n_layers = n, n_layers
        self.theta = nn.Parameter(init_scale * torch.randn(n_layers, n, 2, generator=g, dtype=torch.float64))
        ring = np.eye(2 ** n)
        for j in range(n):
            ring = _cnot(n, j, (j + 1) % n) @ ring
        self.register_buffer("ring", torch.tensor(ring, dtype=CDT))

    def n_params(self):
        return self.theta.numel()

    @staticmethod
    def _ry(t):
        c, s = torch.cos(t / 2), torch.sin(t / 2)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(CDT)

    @staticmethod
    def _rz(t):
        e = torch.exp(-0.5j * t.to(CDT))
        z = torch.zeros((), dtype=CDT)
        return torch.stack([torch.stack([e, z]), torch.stack([z, e.conj()])])

    def unitary(self, tau=None, method=None, L=None):
        U = torch.eye(2 ** self.n, dtype=CDT)
        for l in range(self.n_layers):
            layer = None
            for j in range(self.n):
                gj = self._rz(self.theta[l, j, 1]) @ self._ry(self.theta[l, j, 0])
                layer = gj if layer is None else torch.kron(layer, gj)
            U = self.ring @ layer @ U
        return U


# ---------------------------------------------------------------- fast exact exponential
class HermitianExpm(torch.autograd.Function):
    """U = exp(-i tau H) for Hermitian H via eigh, with the exact Daleckii-Krein backward.

    Forward: H = V diag(w) V^dagger,  U = V diag(e^{-i tau w}) V^dagger.
    Backward: the Frechet derivative is L(E) = V[(V^dagger E V) o Gamma]V^dagger with
    Gamma_ij = (phi(w_i) - phi(w_j)) / (w_i - w_j)  (-> phi'(w_i) when w_i ~ w_j),
    phi(w) = e^{-i tau w}; for a real loss the adjoint is V[(V^dagger G V) o conj(Gamma)]V^dagger.
    Same value and gradient as torch.linalg.matrix_exp (tests/test_galerkin_m4.py), but the
    backward costs a few matrix products instead of a 2n x 2n matrix exponential whose cost
    grows with the gradient norm.  Used for the 9-qubit runs (evolution="exact_eigh")."""

    @staticmethod
    def forward(ctx, H, tau):
        w, V = torch.linalg.eigh(H)
        ph = torch.exp(-1j * tau.to(w.dtype) * w)
        U = (V * ph[None, :]) @ V.conj().T
        ctx.save_for_backward(w, V, ph, tau, H, U)
        return U

    @staticmethod
    def backward(ctx, G):
        w, V, ph, tau, H, U = ctx.saved_tensors
        dw = w[:, None] - w[None, :]
        close = dw.abs() < 1e-9
        gam = torch.where(close, (-1j * tau * ph)[:, None].expand_as(dw).to(ph.dtype),
                          (ph[:, None] - ph[None, :]) / torch.where(close, torch.ones_like(dw), dw))
        Gt = V.conj().T @ G @ V
        gH = V @ (Gt * gam.conj()) @ V.conj().T
        # dU/dtau = -i H U  ->  dL/dtau = Re tr(G^dagger (-i H U))
        gtau = (G.conj() * (-1j * H @ U)).sum().real.to(tau.dtype)
        return gH, gtau.reshape(tau.shape)


def expm_hermitian(H, tau):
    if not torch.is_tensor(tau):
        tau = torch.tensor(float(tau), dtype=torch.float64)
    return HermitianExpm.apply(H, tau)
