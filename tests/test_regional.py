"""Regional domain decomposition (report section 18): flux conservation, interface consistency,
decoupled limit equals independent regions."""
import numpy as np
import torch

from src import cosine2d_galerkin as c2
from src.regional_galerkin import Decomposition


def _labels(NY=24, NX=48):
    Y, X = np.mgrid[0:NY, 0:NX]
    land = ((X - 24) / 22.0) ** 2 + ((Y - 12) / 11.0) ** 2 < 1
    lab = np.where(X < 16, 0, np.where(X < 32, 1, 2))
    lab = np.where((lab == 1) & (Y > 14), 3, lab)
    return np.where(land, lab, -1)


def test_flux_conservation_and_constant_field(record):
    lab = _labels()
    dd = Decomposition(lab, 3, 0.1, 0.12)
    B = np.random.default_rng(0).normal(size=(5, dd.R, 9)) * 0.3
    raw = dd.flux(B, 0.7, galerkin=False)              # <phi_i, flux>_r ; phi_00 = 1 -> mass flux
    mass_flux = raw[..., 0]                            # (5, R) since phi_0 == 1 on every region
    tot = float(np.abs(mass_flux.sum(-1)).max())
    # Galerkin form: d/dt (region mass) from coupling sums to zero too
    dm = np.stack([(dd.flux(B, 0.7)[..., r, :] @ reg.phi.T) @ reg.w for r, reg in enumerate(dd.regions)], -1)
    tot_g = float(np.abs(dm.sum(-1)).max())
    # a globally constant field has zero interface flux
    Bc = np.zeros((dd.R, 9)); Bc[:, 0] = 0.4
    zc = float(np.abs(dd.flux(Bc, 0.7)).max())
    record(sum_mass_flux=tot, sum_galerkin_mass_rate=tot_g, constant_field_flux=zc, n_faces=dd.n_faces)
    assert tot < 1e-12 and tot_g < 1e-12 and zc < 1e-12 and dd.n_faces > 0


def test_interface_flux_matches_global_fv_laplacian(record):
    """S_r = projection of the cross-interface part of the global 5-point Laplacian (explicit build)."""
    lab = _labels()
    NY, NX = lab.shape
    dx, dy = 0.1, 0.12
    dd = Decomposition(lab, 3, dx, dy)
    B = np.random.default_rng(1).normal(size=(dd.R, 9)) * 0.3
    u = dd.field_flat(B, NY * NX).reshape(NY, NX)
    L = np.zeros((NY, NX))
    for i in range(NY):
        for j in range(NX):
            if lab[i, j] < 0:
                continue
            for di, dj, h in ((0, 1, dx), (0, -1, dx), (1, 0, dy), (-1, 0, dy)):
                ii, jj = i + di, j + dj
                if 0 <= ii < NY and 0 <= jj < NX and lab[ii, jj] >= 0 and lab[ii, jj] != lab[i, j]:
                    L[i, j] += (u[ii, jj] - u[i, j]) / h ** 2
    err = 0.0
    S = dd.flux(B, 1.0, galerkin=False)
    for r, reg in enumerate(dd.regions):
        ref = (reg.phi * reg.w[:, None]).T @ L.ravel()[reg.flat]
        err = max(err, float(np.abs(ref - S[r]).max()))
    record(max_err=err)
    assert err < 1e-10


def test_decoupled_limit_matches_independent_regions(record):
    lab = _labels()
    dd = Decomposition(lab, 3, 0.1, 0.12)
    B = np.random.default_rng(2).normal(size=(dd.R, 9)) * 0.3
    rho = np.array([0.5, 0.8, 1.1, 0.9])
    f = dd.rhs(B, 0.05, rho, 1.0, coupled=False)
    err = 0.0
    for r, reg in enumerate(dd.regions):
        c2.setup(reg.crop_mask(lab), 3, reg.Lx, reg.Ly)
        err = max(err, float(np.abs(c2.galerkin_rhs(B[r], 0.05, rho[r], 1.0) - f[r]).max()))
    # D = 0: coupled == decoupled exactly; torch == numpy
    e0 = float(np.abs(dd.rhs(B, 0.0, rho, 1.0) - dd.rhs(B, 0.0, rho, 1.0, coupled=False)).max())
    et = float(np.abs(dd.rhs(torch.tensor(B), 0.05, rho, 1.0).numpy() - dd.rhs(B, 0.05, rho, 1.0)).max())
    record(independent_err=err, D0_err=e0, torch_err=et)
    assert max(err, e0, et) < 1e-12
