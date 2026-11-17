"""2D cosine / Neumann Galerkin basis on a land mask (report section 16)."""
import numpy as np
import pytest
import torch

from src import cosine2d_galerkin as c2
from src import galerkin as gk
from src import pennylane_circuits as plc
from src.custom_ansatz import get_graph
from src.qpino import QPINO, Standardizer

PH = (0.05, 1.0, 1.0)


def _blob_mask(NY=48, NX=96):
    Y, X = np.mgrid[0:NY, 0:NX]
    m = (((X - 50) / 42.0) ** 2 + ((Y - 24) / 20.0) ** 2 < 1) & ~((X > 70) & (Y > 36))
    return m.astype(float)


@pytest.fixture
def basis2d():
    gk.set_basis("cosine2d")
    yield
    gk.set_basis("fourier")


def test_rectangle_reduces_to_tensor_product_triads(record):
    out = {}
    for K in (3, 4):
        c2.setup(np.ones((64, 128)), K, 2 * np.pi, np.pi)
        b = np.random.default_rng(K).normal(size=K * K) * 0.3
        out[f"K{K}"] = float(np.abs(c2.galerkin_rhs(b, *PH) - c2.rect_rhs_tensor_product(b, *PH, K, 2 * np.pi, np.pi)).max())
    record(**out)
    assert max(out.values()) < 1e-12


def test_rectangle_weak_form_equals_projected_strong_form(record):
    """Neumann rectangle: weak form == L2 projection of D lap u + rho u - beta u^2 (fine grid)."""
    c2.setup(np.ones((128, 256)), 3, 2 * np.pi, np.pi)
    S = c2.active()
    b = np.random.default_rng(1).normal(size=9) * 0.3
    u = b @ S["phi"].T
    lap = c2.reconstruct_lap(b, S["xy"])
    strong = PH[0] * lap + PH[1] * u - PH[2] * u ** 2
    proj = c2.field_to_coeffs(strong)
    err = float(np.abs(proj - c2.galerkin_rhs(b, *PH)).max())
    record(max_err=err)
    assert err < 1e-12


def test_masked_projection_and_reaction_terms(record):
    mask = _blob_mask()
    S = c2.setup(mask, 3, 2 * np.pi, np.pi)
    rng = np.random.default_rng(2)
    b = rng.normal(size=9) * 0.3
    u = b @ S["phi"].T
    rec = float(np.abs(c2.field_to_coeffs(u) - b).max())                 # exact for fields in the span
    noise = rng.normal(size=len(u))
    r = noise - (c2.field_to_coeffs(noise) @ S["phi"].T)
    orth = float(np.abs((S["phi"] * S["w"][:, None]).T @ r).max())     # residual orthogonal on land
    # D = 0: rhs == land L2 projection of rho u - beta u^2
    rx = float(np.abs(c2.galerkin_rhs(b, 0.0, 1.0, 1.0) - c2.field_to_coeffs(u - u ** 2)).max())
    gt = c2.galerkin_rhs(torch.tensor(b), *PH).numpy()
    tn = float(np.abs(gt - c2.galerkin_rhs(b, *PH)).max())
    record(reconstruct_err=rec, residual_orthogonality=orth, reaction_err=rx, torch_vs_numpy=tn, cond_G=S["cond_G"])
    assert max(rec, orth, rx, tn) < 1e-10


def test_masked_coupling_graph_complete_and_pennylane(record, basis2d):
    S = c2.setup(_blob_mask(), 3, 2 * np.pi, np.pi)
    E = get_graph("structured", 9)
    norm = Standardizer(np.zeros(9), np.ones(9), np.ones(9))
    m = QPINO(norm, n=9, seed=5, alpha0=1.1)
    a = torch.randn(2, 9) * 0.4
    mt = m.measure(a, 0.6).detach().numpy()
    l = m.layers[0]
    ang = (m.alphas * a).detach().numpy()
    q = plc.make_qnode(n=9)
    pl = np.array([q([ang[i]], [(l.h.detach().numpy(), l.g.detach().numpy(), l.J.detach().numpy())],
                     l.edges, float(m.tau(0.6).detach())) for i in range(2)])
    err = float(np.abs(mt - pl).max())
    record(n_edges=len(E), max_err=err)
    assert len(E) == 36 and err < 1e-10
