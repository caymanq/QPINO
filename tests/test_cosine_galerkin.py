"""Cosine / Neumann Galerkin basis: triad tensor, projection, Neumann solvers, convergence in M,
coupling graph, equivalence with the mirrored Fourier projection, 9-qubit PennyLane check."""
import numpy as np
import pytest
import torch

from src import cosine_galerkin as cg
from src import galerkin as gk
from src import pennylane_circuits as plc
from src import preprocessing as pp
from src.custom_ansatz import get_graph
from src.qpino import QPINO, Standardizer

PH = (0.1, 1.0, 1.0)


@pytest.fixture
def cosine_basis():
    gk.set_basis("cosine")
    yield
    gk.set_basis("fourier")


def test_triad_tensor_vs_explicit_vs_projection(record):
    rng = np.random.default_rng(0)
    out = {}
    for M in (4, 8):
        b = rng.normal(size=(30, M + 1)) * 0.3
        out[f"M{M}_einsum_vs_loop"] = float(np.abs(cg.quad(b) - cg.quad_explicit(b)).max())
        out[f"M{M}_rhs_vs_projection"] = float(np.abs(cg.galerkin_rhs(b, *PH) - cg.galerkin_rhs_by_projection(b, *PH)).max())
    b = rng.normal(size=9)
    q0 = cg.quad(b)[0]
    out["k0_normalisation_err"] = float(abs(q0 - (b[0] ** 2 + 0.5 * (b[1:] ** 2).sum())))
    T = cg.triad_tensor(2)                         # hand check of a few entries
    assert T[0, 1, 1] == 0.5 and T[2, 1, 1] == 0.5 and T[1, 0, 1] == 1.0 and T[1, 2, 1] == 0.5 and T[0, 0, 0] == 1.0
    record(**out)
    assert max(out.values()) < 1e-13


def test_neumann_solvers_agree(record):
    """2nd-order FD (reflecting ghost cells) converges to the even-extension spectral solution."""
    t = np.linspace(0, 1, 6)
    errs = []
    for N in (32, 64, 128):
        x = cg.grid(N)
        u0 = 0.25 + 0.1 * np.cos(x) - 0.06 * np.cos(3 * x) + 0.03 * np.cos(5 * x)
        us = cg.solve_neumann_spectral(u0, t, *PH, rtol=1e-12, atol=1e-14)
        uf = cg.solve_neumann_fd(u0, t, *PH)
        errs.append(float(np.abs(us - uf).max()))
    order = np.log2(errs[1] / errs[2])
    record(fd_err_N32=errs[0], fd_err_N64=errs[1], fd_err_N128=errs[2], fd_order=order)
    assert errs[0] > errs[1] > errs[2] and abs(order - 2) < 0.2


def test_cosine_galerkin_converges_to_neumann_solution(record, cosine_basis):
    """Galerkin ODE at M vs the cosine projection of a fine-grid (N=256) Neumann solution."""
    N = 256
    x = cg.grid(N)
    u0 = cg.random_neumann_ic(x, np.random.default_rng(3), kmax=3)
    t = np.linspace(0, 4, 41)
    U = cg.solve_neumann_spectral(u0, t, *PH, rtol=1e-12, atol=1e-14)
    res = {}
    for M in (2, 4, 8, 16):
        Bp = cg.field_to_coeffs(U, x, M)
        Bg = gk.integrate_galerkin(Bp[0], t, *PH)
        res[M] = float(np.linalg.norm(cg.reconstruct(Bg, x) - U) / np.linalg.norm(U))
    record(**{f"field_rel_err_M{M}": v for M, v in res.items()})
    # M=16 sits at the time-integration tolerance floor (rtol 1e-10 / 1e-12), not below M=8
    assert res[2] > res[4] > res[8] and max(res[8], res[16]) < 1e-8


def test_cosine_coupling_graph(record, cosine_basis):
    s4, s8 = cg.coupling_stats(4), cg.coupling_stats(8)
    E = get_graph("structured", 9)
    record(M8_edges=s8["n_edges"], M8_pairs=s8["n_pairs"], M8_tensor_nnz=s8["tensor_nonzeros"],
           M8_tensor_size=s8["tensor_size"], M4_edges=s4["n_edges"], M8_pairs_per_eq=str(s8["pairs_per_equation"]))
    # every pair p != q couples through k = |p - q| <= M: the pairwise graph is complete
    assert len(E) == 36 and s8["n_edges"] == 36 and s4["n_edges"] == 10 and s8["tensor_nonzeros"] == 109


def test_cosine_equals_mirrored_fourier_projection(record):
    """Cosine projection of the 32-bin field == the mirrored (even) Fourier projection."""
    rng = np.random.default_rng(4)
    u = rng.random((4, 32)) + np.linspace(0, 1, 32)
    xm = gk.uniform_grid(64, 0.5)
    out = {}
    for M in (4, 8):
        rc = cg.reconstruct(cg.field_to_coeffs(u, cg.grid(32), M), cg.grid(32))
        rm = gk.reconstruct_fourier(gk.field_to_coeffs_fourier(pp.mirror_extend(u), xm, M), xm)[:, :32]
        out[f"M{M}"] = float(np.abs(rc - rm).max())
    record(**out)
    assert max(out.values()) < 1e-12


def test_cosine_qpino_matches_pennylane(record, cosine_basis):
    norm = Standardizer(np.zeros(9), np.ones(9), np.ones(9))
    m = QPINO(norm, n=9, seed=21, alpha0=1.1)
    a = torch.randn(2, 9) * 0.4
    mt = m.measure(a, 0.6).detach().numpy()
    l = m.layers[0]
    ang = (m.alphas * a).detach().numpy()
    q = plc.make_qnode(n=9)
    pl = np.array([q([ang[i]], [(l.h.detach().numpy(), l.g.detach().numpy(), l.J.detach().numpy())],
                     l.edges, float(m.tau(0.6).detach())) for i in range(2)])
    err = float(np.abs(mt - pl).max())
    record(max_err=err, n_edges=len(l.edges))
    assert err < 1e-10 and len(l.edges) == 36
