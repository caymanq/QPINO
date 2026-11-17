"""Joint regional training (report section 19): gradients flow across regions through the
classical interface flux, and the joint rollout loss equals a manual numpy rollout."""
import numpy as np
import torch

from src import cosine2d_galerkin as c2
from src.qpino import QPINO, MLPOperator, Standardizer
from src.regional_galerkin import Decomposition
from src.regional_training import JointRegional, rollout_loss, step_weights
from tests.test_regional import _labels


def _joint(flux, qpino_region0=False):
    lab = _labels()
    dd = Decomposition(lab, 3, 0.1, 0.12)
    rng = np.random.default_rng(1)
    models = []
    for r in range(dd.R):
        norm = Standardizer(rng.normal(size=9) * 0.1, np.ones(9) * 0.3, np.ones(9) * 0.05)
        if qpino_region0 and r == 0:
            reg = dd.regions[0]
            c2.setup(reg.crop_mask(lab), 3, reg.Lx, reg.Ly)
            models.append(QPINO(norm, c0=1.0, seed=0, n=9, evolution="exact_eigh"))
        else:
            models.append(MLPOperator(norm, 6, 2, seed=r, n=9))
    return dd, JointRegional(models, dd, D=0.002, flux=flux)


def test_flux_gradients_flow_between_regions(record):
    out = {}
    for flux in (True, False):
        dd, J = _joint(flux, qpino_region0=True)
        r_src, r_tgt = [(a, s) for (a, s) in dd.neighbours if a != s][0]
        B0 = torch.tensor(np.random.default_rng(2).normal(size=(3, dd.R, 9)) * 0.3)
        with J.cache():
            P = J.rollout(B0, 1.0, 3)
        loss = (P[:, :, r_src, :] ** 2).sum()        # loss on region r_src only
        J.zero_grad(); loss.backward()
        g = sum(float(p.grad.abs().sum()) for p in J.models[r_tgt].parameters() if p.grad is not None)
        out[flux] = g
        if flux:   # finite-difference check of one parameter of the *neighbour* region
            p = [q for q in J.models[r_tgt].parameters() if q.requires_grad][0]
            i = int(torch.argmax(p.grad.abs().flatten()))
            ga = float(p.grad.flatten()[i]); eps = 1e-6
            with torch.no_grad():
                vals = []
                for sgn in (1, -1):
                    p.view(-1)[i] += sgn * eps
                    with J.cache():
                        vals.append(float((J.rollout(B0, 1.0, 3)[:, :, r_src, :] ** 2).sum()))
                    p.view(-1)[i] -= sgn * eps
            gfd = (vals[0] - vals[1]) / (2 * eps)
            out["fd_rel_err"] = abs(gfd - ga) / max(abs(ga), 1e-12)
    record(neighbour_grad_flux_on=out[True], neighbour_grad_flux_off=out[False], fd_rel_err=out["fd_rel_err"])
    assert out[True] > 1e-8 and out[False] == 0.0 and out["fd_rel_err"] < 1e-5


def test_rollout_loss_matches_manual_rollout(record):
    dd, J = _joint(True)
    rng = np.random.default_rng(3)
    W = torch.tensor(rng.normal(size=(4, 6, dd.R, 9)) * 0.3)
    s_inc = torch.tensor(rng.uniform(0.5, 2.0, size=(dd.R, 9)))
    alpha, dt = 0.5, 1.0
    L, _ = rollout_loss(J, W, dt, s_inc, alpha)
    with torch.no_grad():
        Wn, sn = W.numpy(), s_inc.numpy()
        B, errs = Wn[:, 0], []
        for s in range(1, Wn.shape[1]):
            nxt = np.stack([J.models[r](torch.tensor(B[:, r]), dt).numpy() for r in range(dd.R)], -2)
            B = nxt + dt * dd.flux(B, J.D)                       # numpy flux
            errs.append((((B - Wn[:, s]) / sn) ** 2).sum((-1, -2)))
        w = step_weights(Wn.shape[1] - 1, alpha).numpy()
        Lm = float((np.stack(errs, 1) * w).mean())
        # flux off == independent region models
        J.flux = False
        P = J.rollout(W[:, 0], dt, 2).numpy()
        Bi = Wn[:, 0]
        for _ in range(2):
            Bi = np.stack([J.models[r](torch.tensor(Bi[:, r]), dt).numpy() for r in range(dd.R)], -2)
        dec = float(np.abs(P[:, -1] - Bi).max())
    L = L.detach()
    record(loss=float(L), manual=Lm, abs_diff=abs(float(L) - Lm), decoupled_max_diff=dec)
    assert abs(float(L) - Lm) < 1e-10 * max(1.0, Lm) and dec < 1e-12
