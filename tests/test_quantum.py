"""Phase-12 checks 5-7, 9, 10: unitarity, differentiability, gradients, identity, training."""
import numpy as np
import scipy.linalg as sla
import torch

from src import fisher_kpp as fk
from src import galerkin as gk
from src import pennylane_circuits as plc
from src.custom_ansatz import HamiltonianAnsatz, HardwareEfficientAnsatz, get_graph
from src.qpino import QPINO, Standardizer
from src.quantum_embedding import embedding_hamiltonian, kron_all, ry_matrix, ry_product_state
from src.training import make_dataset, make_norm, train

torch.set_default_dtype(torch.float64)


def _norm():
    return Standardizer(np.zeros(5), np.ones(5), np.ones(5))


def _unitarity_err(U):
    U = U.detach()
    return float((U.conj().T @ U - torch.eye(U.shape[0], dtype=U.dtype)).abs().max())


def test_embedding_equals_ry_product(record):
    """S(z) = exp(-i H_emb(z)) equals (x)_j R_y(alpha_j z_j); R_y(phi)|0> amplitudes."""
    z = np.random.default_rng(1).normal(size=5); alpha = np.array([0.7, 1.1, 1.3, 0.9, 1.6])
    S = sla.expm(-1j * embedding_hamiltonian(z, alpha))
    Sp = kron_all([ry_matrix(torch.tensor(alpha[j] * z[j])).numpy() for j in range(5)])
    psi = ry_product_state(torch.tensor(alpha * z)[None]).numpy()[0]
    e = max(np.abs(S - Sp).max(), np.abs(S[:, 0] - psi).max())
    record(max_err=e)
    assert e < 1e-12


def test_all_circuits_unitary(record):
    """(5) exact, Trotter (L=1..8), all coupling graphs, HEA, re-uploaded products."""
    errs = {}
    for g in ["structured", "triad", "full", "nearest", "none"]:
        A = HamiltonianAnsatz(get_graph(g), seed=3)
        errs[f"exact_{g}"] = _unitarity_err(A.unitary(0.83))
        for L in [1, 2, 8]:
            errs[f"trotter{L}_{g}"] = _unitarity_err(A.unitary(0.83, "trotter", L))
    errs["hea"] = _unitarity_err(HardwareEfficientAnsatz(seed=2).unitary())
    m = QPINO(_norm(), reupload=3, seed=4)
    psi = m.state(torch.randn(6, 5), 0.5)
    errs["reupload3_state_norm"] = float((psi.abs().pow(2).sum(1) - 1).abs().max())
    record(max_err=max(errs.values()), **errs)
    assert max(errs.values()) < 1e-12


def test_trotter_converges_first_order(record):
    A = HamiltonianAnsatz(get_graph("structured"), seed=5)
    Ue = A.unitary(1.0).detach()
    e = [float((A.unitary(1.0, "trotter", L).detach() - Ue).abs().max()) for L in (4, 8, 16, 32)]
    order = np.log2(e[2] / e[3])
    record(err_L4=e[0], err_L8=e[1], err_L16=e[2], err_L32=e[3], observed_order=order)
    assert e[0] > e[1] > e[2] > e[3] and 0.8 < order < 1.2


def test_torch_matches_pennylane(record):
    """Fast torch simulator == PennyLane default.qubit (exact exp, Trotter, re-upload, HEA)."""
    m = QPINO(_norm(), reupload=1, seed=7, alpha0=1.3)
    a = torch.randn(3, 5) * 0.5
    q = plc.make_qnode()
    errs = []
    for meth, L in [("exact", 1), ("trotter", 3)]:
        m.evolution, m.trotter_L = meth, L
        mt = m.measure(a, 0.6).detach().numpy()
        lay = m.layers[0]
        hp = (lay.h.detach().numpy(), lay.g.detach().numpy(), lay.J.detach().numpy())
        ang = (m.alphas * a).detach().numpy()
        pl = np.array([q([ang[i]], [hp], lay.edges, float(m.tau(0.6)), meth, L) for i in range(3)])
        errs.append(np.abs(mt - pl).max())
    m2 = QPINO(_norm(), reupload=2, seed=8)
    mt = m2.measure(a, 0.6).detach().numpy()
    hps = [(l.h.detach().numpy(), l.g.detach().numpy(), l.J.detach().numpy()) for l in m2.layers]
    ang = (m2.alphas * a).detach().numpy()
    pl = np.array([q([ang[i], ang[i]], hps, m2.layers[0].edges, float(m2.tau(0.6))) for i in range(3)])
    errs.append(np.abs(mt - pl).max())
    m3 = QPINO(_norm(), ansatz="hea", seed=9)
    mt = m3.measure(a, 0.6).detach().numpy()
    qh = plc.make_hea_qnode()
    ang = (m3.alphas * a).detach().numpy()
    pl = np.array([qh(ang[i], m3.layers[0].theta.detach().numpy()) for i in range(3)])
    errs.append(np.abs(mt - pl).max())
    record(exact=errs[0], trotter=errs[1], reupload2=errs[2], hea=errs[3])
    assert max(errs) < 1e-10


def test_measurement_gradcheck(record):
    """(6) <Z>, <ZZ> are differentiable: autograd == central finite differences (float64)."""
    m = QPINO(_norm(), seed=11)
    a = torch.randn(2, 5) * 0.4
    names = [n for n, _ in m.named_parameters()]
    max_rel = 0.0
    for pname, p in m.named_parameters():
        if pname.startswith("decoder"):
            continue
        m.zero_grad()
        loss = m.measure(a, 0.7).pow(2).sum() if pname != "x" else None
        loss.backward()
        g = p.grad.clone().reshape(-1)
        fd = torch.zeros_like(g)
        flat = p.data.reshape(-1)
        for i in range(flat.numel()):
            old = flat[i].item()
            flat[i] = old + 1e-6; lp = m.measure(a, 0.7).pow(2).sum().item()
            flat[i] = old - 1e-6; lm = m.measure(a, 0.7).pow(2).sum().item()
            flat[i] = old
            fd[i] = (lp - lm) / 2e-6
        rel = float((g - fd).abs().max() / (fd.abs().max() + 1e-12))
        max_rel = max(max_rel, rel)
    # input gradient (d m / d a), also through torch.autograd.gradcheck
    ok = torch.autograd.gradcheck(lambda v: m.measure(v, 0.7), (a.clone().requires_grad_(True),))
    record(max_rel_err_param_grads=max_rel, n_param_tensors=len(names), input_gradcheck=ok)
    assert max_rel < 1e-6 and ok


def test_gradients_reach_all_parameters(record):
    """(7) gradients propagate through decoder into every quantum and classical parameter."""
    for cfg in [dict(), dict(decoder="mlp"), dict(reupload=2), dict(ansatz="hea"),
                dict(evolution="trotter", trotter_L=2)]:
        m = QPINO(_norm(), seed=2, **cfg)
        m(torch.randn(8, 5), 0.3).pow(2).sum().backward()
        norms = {n: float(p.grad.abs().max()) for n, p in m.named_parameters() if p.requires_grad}
        assert all(v > 0 for v in norms.values()), (cfg, norms)
    record(min_grad=min(norms.values()))


def test_identity_at_zero_dt(record):
    """(9) residual update gives an exact identity at dt = 0 (U = I, decoder finite)."""
    errs = []
    for cfg in [dict(), dict(reupload=3), dict(ansatz="hea"), dict(evolution="trotter")]:
        m = QPINO(_norm(), seed=1, **cfg)
        a = torch.randn(10, 5)
        errs.append(float((m(a, 0.0) - a).abs().max()))
        assert torch.isfinite(m.rhs(a, 0.0)).all()
    record(max_err=max(errs))
    assert max(errs) == 0.0


def test_fourier_spectrum_degree(record):
    """Output of an R-times re-uploaded circuit, as a function of one input angle, contains
    only integer frequencies |omega| <= R (Schuld-Sweke-Meyer)."""
    out = {}
    for R in [1, 2, 3]:
        m = QPINO(_norm(), reupload=R, seed=R, alpha0=1.0, train_alpha=False)
        phi = torch.linspace(0, 2 * np.pi, 65)[:-1]
        a = torch.zeros(64, 5); a[:, 2] = phi
        f = m.measure(a, 1.0)[:, 2].detach().numpy()
        c = np.abs(np.fft.fft(f)) / 64
        freqs = np.fft.fftfreq(64, 1 / 64)
        out[f"R{R}_max_power_beyond_R"] = float(c[np.abs(freqs) > R].max())
        out[f"R{R}_power_at_R"] = float(c[np.abs(freqs) == R].max())
        assert c[np.abs(freqs) > R].max() < 1e-12 and c[np.abs(freqs) == R].max() > 1e-6
    record(**out)


def test_training_loop_smoke(record):
    """(10) full loss (data + PDE + rollout + reg) trains on a tiny synthetic data set."""
    t, x, U = fk.generate_trajectories(4, 0.1, 1.0, 1.0, 0.1, 12, N=32, seed=0)
    trajs = [gk.field_to_coeffs(u, x) for u in U]
    xc = gk.uniform_grid(32)
    norm, scales = make_norm(trajs[:3], 0.1, xc)
    tr, va = make_dataset(trajs[:3], 3), make_dataset(trajs[3:], 3)
    m = QPINO(norm, c0=10.0, seed=0)
    cfg = dict(epochs=60, lr=0.03, lambda_data=1.0, lambda_pde=0.1, lambda_roll=0.5, lambda_reg=1e-3,
               reg_weights={"neg": 1.0, "param": 1e-3}, log_every=5, scheme="forward")
    h, m = train(m, tr, va, 0.1, cfg, (0.1, 1.0, 1.0), scales, xc)
    record(initial_total=h["total"][0], final_total=h["total"][-1], initial_data=h["data"][0],
           final_data=h["data"][-1])
    assert np.isfinite(h["total"]).all() and h["total"][-1] < 0.5 * h["total"][0]
