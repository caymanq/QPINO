"""Phase-12 checks 1-4: Fourier transform, Galerkin equations, conjugate symmetry, solver."""
import numpy as np
import torch

from src import fisher_kpp as fk
from src import galerkin as gk

RNG = np.random.default_rng(0)
PH = dict(D=0.13, rho=0.9, beta=1.3)


def test_fourier_roundtrip(record):
    """(1) projection of an exactly band-limited field recovers its coefficients and the
    inverse transform recovers the field; numpy and torch paths agree; bin-centre grid too."""
    errs = []
    for offset in [0.0, 0.5]:
        x = gk.uniform_grid(32, offset)
        a = RNG.normal(size=(10, 5))
        u = gk.reconstruct(a, x)
        a2 = gk.field_to_coeffs(u, x)
        a3 = gk.field_to_coeffs(torch.tensor(u), x).numpy()
        errs += [np.abs(a2 - a).max(), np.abs(a3 - a).max(), np.abs(gk.reconstruct(a2, x) - u).max()]
    # agreement with numpy FFT on the standard grid
    x = gk.uniform_grid(16)
    u = RNG.normal(size=16)
    ak = np.fft.fft(u) / 16
    a = gk.field_to_coeffs(u, x)
    errs.append(np.abs(a - [ak[0].real, ak[1].real, ak[1].imag, ak[2].real, ak[2].imag]).max())
    record(max_abs_error=max(errs))
    assert max(errs) < 1e-12


def test_galerkin_equations_vs_projection(record):
    """(2) explicit real equations == complex convolution == direct spectral projection of
    D u_xx + rho u - beta u^2 (N=64 grid, no aliasing since 64 >= 3M+1)."""
    a = RNG.normal(size=(50, 5))
    f_exp = gk.galerkin_rhs(a, **PH)
    f_cplx = gk.from_complex_modes(gk.galerkin_rhs_complex(gk.to_complex_modes(a), **PH))
    f_proj = gk.galerkin_rhs_by_projection(a, **PH, N=64)
    e1, e2 = np.abs(f_exp - f_cplx).max(), np.abs(f_exp - f_proj).max()
    record(explicit_vs_convolution=e1, explicit_vs_projection=e2)
    assert e1 < 1e-12 and e2 < 1e-12


def test_conjugate_symmetry(record):
    """(3) the complex RHS of the 5 complex modes satisfies f_{-k} = conj(f_k) and a0 stays
    real; integrating the complex system keeps a_{-k} = conj(a_k)."""
    a = RNG.normal(size=(20, 5))
    ac = gk.to_complex_modes(a)
    f = gk.galerkin_rhs_complex(ac, **PH)
    sym = max(np.abs(f[:, 0] - f[:, 4].conj()).max(), np.abs(f[:, 1] - f[:, 3].conj()).max(),
              np.abs(f[:, 2].imag).max())
    # explicit complex RK4 integration of all 5 complex modes (no symmetry imposed)
    y = gk.to_complex_modes(np.array([0.2, 0.05, -0.03, 0.04, 0.02]))
    for _ in range(500):
        k1 = gk.galerkin_rhs_complex(y, **PH); k2 = gk.galerkin_rhs_complex(y + 0.005 * k1, **PH)
        k3 = gk.galerkin_rhs_complex(y + 0.005 * k2, **PH); k4 = gk.galerkin_rhs_complex(y + 0.01 * k3, **PH)
        y = y + 0.01 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    drift = max(abs(y[0] - y[4].conj()), abs(y[1] - y[3].conj()), abs(y[2].imag))
    record(rhs_symmetry_error=sym, integrated_symmetry_drift=drift)
    assert sym < 1e-13 and drift < 1e-12


def test_solver_linear_modes(record):
    """(4a) beta=0: u = A cos(kx) e^{(rho-Dk^2)t} exactly (spectral) and O(dx^2) (FD)."""
    N = 64
    x = 2 * np.pi * np.arange(N) / N
    t = np.linspace(0, 2, 5)
    u0 = 0.3 * np.cos(3 * x)
    ex = fk.linear_mode_solution(x, t, 3, 0.3, 0.2, 0.5)
    e_sp = np.abs(fk.solve(u0, t, 0.2, 0.5, 0.0) - ex).max()
    e_fd = [np.abs(fk.solve(0.3 * np.cos(3 * xx), t, 0.2, 0.5, 0.0, method="fd") -
                   fk.linear_mode_solution(xx, t, 3, 0.3, 0.2, 0.5)).max()
            for xx in [2 * np.pi * np.arange(n) / n for n in (32, 64, 128)]]
    order = np.log2(e_fd[0] / e_fd[1]), np.log2(e_fd[1] / e_fd[2])
    record(spectral_max_err=e_sp, fd_err_32=e_fd[0], fd_err_64=e_fd[1], fd_err_128=e_fd[2],
           fd_order_1=order[0], fd_order_2=order[1])
    assert e_sp < 1e-8
    assert 1.8 < order[0] < 2.2 and 1.8 < order[1] < 2.2


def test_solver_logistic(record):
    """(4b) uniform IC: logistic solution K u0 e^{rho t}/(K + u0(e^{rho t}-1))."""
    t = np.linspace(0, 8, 9)
    u = fk.solve(np.full(16, 0.05), t, 0.3, 1.0, 2.0)
    err = np.abs(u - fk.logistic_solution(0.05, t, 1.0, 2.0)[:, None]).max()
    record(max_err=err)
    assert err < 1e-8


def test_solver_self_convergence(record):
    """(4c) nonlinear self-convergence: spectral N=32/64 vs N=256 at t=2."""
    def run(N):
        x = 2 * np.pi * np.arange(N) / N
        return fk.solve(0.3 + 0.2 * np.cos(x) + 0.1 * np.sin(2 * x), np.array([0, 2.0]), 0.1, 1.0, 1.0)[-1]
    ref = run(256)[::8]
    e32, e64 = np.abs(run(32) - ref).max(), np.abs(run(64)[::2] - ref).max()
    record(err_N32=e32, err_N64=e64)
    assert e64 < 1e-8 and e32 < 1e-6


def test_galerkin_integrator_agreement(record):
    """RK4 flow map (baseline A) vs solve_ivp DOP853."""
    a0 = np.array([0.3, 0.05, -0.02, 0.03, 0.01])
    t = np.array([0.0, 0.1])
    ref = gk.integrate_galerkin(a0, t, **PH)[-1]
    rk = gk.rk4_step(a0, 0.1, PH["D"], PH["rho"], PH["beta"], substeps=20)
    record(err=np.abs(ref - rk).max())
    assert np.abs(ref - rk).max() < 1e-10


def test_truncation_error_decreases_with_M(record):
    """M=2 truncation of a full Fisher-KPP solution vs higher truncations (diagnostic)."""
    N = 128
    x = 2 * np.pi * np.arange(N) / N
    u = fk.solve(0.2 + 0.08 * np.cos(x) + 0.06 * np.sin(2 * x), np.array([0, 3.0]), 0.1, 1.0, 1.0)[-1]
    U = np.fft.fft(u)
    errs = {}
    for Mk in [1, 2, 3, 4]:
        V = U.copy(); V[Mk + 1:N - Mk] = 0
        errs[Mk] = np.linalg.norm(np.fft.ifft(V).real - u) / np.linalg.norm(u)
    record(**{f"rel_err_M{k}": v for k, v in errs.items()})
    assert errs[1] > errs[2] > errs[3] > errs[4]
