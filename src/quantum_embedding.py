"""Exact (torch-native) statevector primitives and the R_y data embedding.

Conventions: n qubits, qubit 0 is the most significant bit of the basis index (same as
PennyLane's default.qubit).  States are complex128 tensors of shape (B, 2**n).

Embedding:  H_emb(z) = 1/2 sum_j alpha_j z_j Y_j,  S(z) = exp(-i H_emb) = (x)_j R_y(alpha_j z_j),
            R_y(phi)|0> = cos(phi/2)|0> + sin(phi/2)|1>.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

CDT = torch.complex128
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)


def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def pauli_on(P, j, n):
    return kron_all([P if i == j else I2 for i in range(n)])


def z_signs(n):
    """(2**n, n) matrix of eigenvalues of Z_j on each basis state."""
    b = np.arange(2 ** n)[:, None]
    bits = (b >> (n - 1 - np.arange(n))[None, :]) & 1
    return 1.0 - 2.0 * bits


def measurement_matrix(n, include_zz=True):
    """Columns: Z_0..Z_{n-1} then Z_jZ_k (j<k).  <O> = |psi|^2 @ matrix."""
    s = z_signs(n)
    cols = [s[:, j] for j in range(n)]
    names = [f"Z{j}" for j in range(n)]
    if include_zz:
        for j, k in itertools.combinations(range(n), 2):
            cols.append(s[:, j] * s[:, k])
            names.append(f"Z{j}Z{k}")
    return torch.tensor(np.stack(cols, 1), dtype=torch.float64), names


def ry_matrix(phi):
    """Batch of R_y(phi) matrices (..., 2, 2)."""
    c, s = torch.cos(phi / 2), torch.sin(phi / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def ry_product_state(angles):
    """S(z)|0...0> for angles (B, n) -> (B, 2**n) complex (a real product state)."""
    c, s = torch.cos(angles / 2), torch.sin(angles / 2)
    B, n = angles.shape
    state = torch.stack([c[:, 0], s[:, 0]], -1)
    for j in range(1, n):
        v = torch.stack([c[:, j], s[:, j]], -1)
        state = (state[:, :, None] * v[:, None, :]).reshape(B, -1)
    return state.to(CDT)


def apply_1q(state, G, wire, n):
    """Apply a (batched) single-qubit gate G (B,2,2) or (2,2) to `wire`."""
    B = state.shape[0]
    psi = state.reshape(B, 2 ** wire, 2, 2 ** (n - wire - 1))
    G = G.to(CDT)
    if G.dim() == 2:
        out = torch.einsum("ts,blsr->bltr", G, psi)
    else:
        out = torch.einsum("bts,blsr->bltr", G, psi)
    return out.reshape(B, -1)


def apply_ry_layer(state, angles, n):
    for j in range(n):
        state = apply_1q(state, ry_matrix(angles[:, j]), j, n)
    return state


def embedding_hamiltonian(z, alpha):
    """Dense H_emb(z) = 1/2 sum_j alpha_j z_j Y_j (numpy; used for verification only)."""
    n = len(z)
    return 0.5 * sum(alpha[j] * z[j] * pauli_on(Y, j, n) for j in range(n))


def expectations(state, meas):
    probs = (state.conj() * state).real
    return probs @ meas, probs


# ---------------------------------------------------------------- Fourier spectrum
def accessible_frequencies(n_layers, n_dims=5):
    """Frequency set of an n_layers-times re-uploaded R_y embedding (alpha_j = 1):
    each Y_j/2 generator has eigenvalue differences {-1,0,1}; L re-uploads give
    {-L..L} per input dimension and the Cartesian product over dimensions."""
    per_dim = list(range(-n_layers, n_layers + 1))
    return per_dim, (2 * n_layers + 1) ** n_dims


def numerical_spectrum_1d(f, n_layers, n_grid=64):
    """|c_omega| of the 1-D periodic function phi -> f(phi), phi in [0, 4*pi) (period 4*pi
    in the angle because R_y uses phi/2 ... but expectation values are 2*pi periodic)."""
    phi = 2 * np.pi * np.arange(n_grid) / n_grid
    vals = np.array([f(p) for p in phi])
    c = np.fft.fft(vals) / n_grid
    return np.fft.fftfreq(n_grid, 1 / n_grid), np.abs(c)
