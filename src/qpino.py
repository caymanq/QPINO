"""QPINO model and classical operator baselines.

QPINO one-step map (5 real Galerkin coefficients a -> a'):
    z      = (a - mu) / sigma                       (training-set standardisation)
    psi_0  = S(z)|0>^5,  S(z) = (x)_j R_y(alpha_j z_j)
    psi    = U_theta(tau) psi_0,  tau = c * dt      (single embedding)
           = W_R S(z) ... W_1 S(z)|0>              (data re-uploading, R layers)
    m      = (<Z_j>, [<Z_j Z_k>])
    G      = s_G * D_eta(m)                        (classical decoder, s_G = derivative scale)
    a'     = a + dt * G                            (residual update -> identity at dt = 0)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import galerkin
from .custom_ansatz import HamiltonianAnsatz, HardwareEfficientAnsatz, get_graph
from .quantum_embedding import (apply_ry_layer, measurement_matrix, ry_product_state)

torch.set_default_dtype(torch.float64)


def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


class Standardizer(nn.Module):
    def __init__(self, mu, sigma, g_scale):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float64))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float64))
        self.register_buffer("g_scale", torch.as_tensor(g_scale, dtype=torch.float64))


class QPINO(nn.Module):
    def __init__(self, norm: Standardizer, ansatz="hamiltonian", graph="structured",
                 evolution="exact", trotter_L=4, tau_mode="learned", c0=1.0,
                 reupload=1, measure_zz=True, decoder="linear", alpha0=1.0,
                 train_alpha=True, hidden=16, seed=0, init_scale=0.5, n=5):
        super().__init__()
        torch.manual_seed(seed)
        if ansatz == "hea":
            tau_mode = "fixed"  # the HEA has no evolution time, so c would be an unused parameter
        self.n, self.norm = n, norm
        self.ansatz_type, self.evolution, self.trotter_L = ansatz, evolution, trotter_L
        self.tau_mode, self.reupload, self.graph_name = tau_mode, reupload, graph
        if ansatz == "hamiltonian":
            edges = get_graph(graph)
            self.layers = nn.ModuleList([HamiltonianAnsatz(edges, n, init_scale, seed + 17 * r)
                                         for r in range(reupload)])
        elif ansatz == "hea":
            self.layers = nn.ModuleList([HardwareEfficientAnsatz(n, 2, seed + 17 * r, init_scale)
                                         for r in range(reupload)])
        else:
            raise ValueError(ansatz)
        a = torch.full((n,), float(alpha0))
        self.alpha = nn.Parameter(a) if train_alpha else None
        if not train_alpha:
            self.register_buffer("alpha_fixed", a)
        c_raw = torch.tensor(_inv_softplus(c0))
        if tau_mode == "learned":
            self.c_raw = nn.Parameter(c_raw)
        else:
            self.register_buffer("c_raw", c_raw)
        meas, self.meas_names = measurement_matrix(n, measure_zz)
        self.register_buffer("meas", meas)
        nf = meas.shape[1]
        if decoder == "linear":
            self.decoder = nn.Linear(nf, 5)
        else:
            self.decoder = nn.Sequential(nn.Linear(nf, hidden), nn.Tanh(), nn.Linear(hidden, 5))
        self.decoder_type = decoder
        self.shots = None  # None = exact expectation values

    # ------------------------------------------------------------------ helpers
    @property
    def alphas(self):
        return self.alpha if self.alpha is not None else self.alpha_fixed

    @property
    def c(self):
        return F.softplus(self.c_raw)

    def tau(self, dt):
        return self.c * dt

    def quantum_params(self):
        return sum(l.n_params() for l in self.layers) + (self.n if self.alpha is not None else 0) + \
            (1 if self.tau_mode == "learned" else 0)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def unitaries(self, dt):
        tau = self.tau(dt)
        return [l.unitary(tau, self.evolution, self.trotter_L) for l in self.layers]

    def state(self, a, dt):
        z = (a - self.norm.mu) / self.norm.sigma
        angles = self.alphas * z
        Us = self.unitaries(dt)
        psi = ry_product_state(angles)
        psi = psi @ Us[0].T
        for U in Us[1:]:
            psi = apply_ry_layer(psi, angles, self.n)
            psi = psi @ U.T
        return psi

    def measure(self, a, dt):
        psi = self.state(a, dt)
        probs = (psi.conj() * psi).real
        if self.shots is None:
            return probs @ self.meas
        # finite-shot estimate: sample basis states from the Born distribution
        p = probs.detach().clamp_min(0)
        p = p / p.sum(1, keepdim=True)
        counts = torch.stack([torch.bincount(torch.multinomial(pi, self.shots, replacement=True),
                                             minlength=p.shape[1]) for pi in p]).to(torch.float64)
        return (counts / self.shots) @ self.meas

    def rhs(self, a, dt):
        m = self.measure(a, dt)
        return self.norm.g_scale * self.decoder(m)

    def forward(self, a, dt):
        return a + dt * self.rhs(a, dt)

    def rollout(self, a0, dt, n_steps):
        out, a = [], a0
        for _ in range(n_steps):
            a = self.forward(a, dt)
            out.append(a)
        return torch.stack(out, 1)

    def hamiltonian_norm(self):
        if self.ansatz_type != "hamiltonian":
            return sum((l.theta ** 2).sum() for l in self.layers)
        return sum((l.h ** 2).sum() + (l.g ** 2).sum() + (l.J ** 2).sum() for l in self.layers)


class MLPOperator(nn.Module):
    """Baseline B: classical residual MLP a -> a + dt * s_G * MLP(z)."""

    def __init__(self, norm: Standardizer, hidden=8, depth=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.norm = norm
        layers, d = [], 5
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, 5))
        self.net = nn.Sequential(*layers)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def rhs(self, a, dt):
        return self.norm.g_scale * self.net((a - self.norm.mu) / self.norm.sigma)

    def forward(self, a, dt):
        return a + dt * self.rhs(a, dt)

    def rollout(self, a0, dt, n_steps):
        out, a = [], a0
        for _ in range(n_steps):
            a = self.forward(a, dt)
            out.append(a)
        return torch.stack(out, 1)

    def hamiltonian_norm(self):
        return sum((p ** 2).sum() for p in self.parameters()) * 0


class GalerkinOperator(nn.Module):
    """Baseline A: integrate the classical 5-D Galerkin ODE (RK4, 20 substeps per step;
    agrees with solve_ivp DOP853 rtol=1e-10 to <1e-9 for the step sizes used here)."""

    def __init__(self, D, rho, beta, substeps=20):
        super().__init__()
        self.D, self.rho, self.beta, self.substeps = D, rho, beta, substeps

    def n_params(self):
        return 3

    def forward(self, a, dt):
        return galerkin.rk4_step(a, dt, self.D, self.rho, self.beta, self.substeps)

    def rhs(self, a, dt):
        return (self.forward(a, dt) - a) / dt

    def rollout(self, a0, dt, n_steps):
        out, a = [], a0
        for _ in range(n_steps):
            a = self.forward(a, dt)
            out.append(a)
        return torch.stack(out, 1)


class PersistenceOperator(nn.Module):
    def n_params(self):
        return 0

    def forward(self, a, dt):
        return a.clone()

    def rollout(self, a0, dt, n_steps):
        return a0[:, None, :].repeat(1, n_steps, 1)
