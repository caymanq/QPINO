"""Joint training of the 5 regional operators with the classical interface flux inside the
training rollouts (report section 19).

Joint step (flux on):   B^{t+1}_r = N_r(B^t_r) + dt * S_r(B^t)      (S: interface diffusive flux)
Flux off:               B^{t+1}_r = N_r(B^t_r)                        (= independent regions)
The regional circuits are still never entangled with each other: every N_r is its own 9-qubit
model; regions interact only through the classical, differentiable flux S(B), so gradients from
region s's loss reach region r's parameters through S whenever r and s are neighbours.
"""
from __future__ import annotations

import contextlib
import copy
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class JointRegional(nn.Module):
    def __init__(self, models, DD, D, flux=True):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.DD, self.D, self.flux = DD, float(D), bool(flux)

    @contextlib.contextmanager
    def cache(self):
        with contextlib.ExitStack() as st:
            for m in self.models:
                if hasattr(m, "cache_unitaries"):
                    st.enter_context(m.cache_unitaries())
            yield

    def step(self, B, dt, flux=None):
        """B: (batch, R, n) -> (batch, R, n)."""
        out = torch.stack([m(B[:, r], dt) for r, m in enumerate(self.models)], -2)
        if self.flux if flux is None else flux:
            out = out + dt * self.DD.flux(B, self.D)
        return out

    def rollout(self, B0, dt, H, flux=None):
        out, B = [], B0
        for _ in range(H):
            B = self.step(B, dt, flux)
            out.append(B)
        return torch.stack(out, 1)                     # (batch, H, R, n)

    def hamiltonian_norm(self):
        return sum(m.hamiltonian_norm() if hasattr(m, "hamiltonian_norm") else 0.0 for m in self.models)


def step_weights(H, alpha):
    """Per-step rollout weights w_s ∝ s^-alpha (alpha=0: uniform), normalised to mean 1."""
    w = np.arange(1, H + 1, dtype=float) ** (-alpha)
    return torch.tensor(w / w.mean())


def rollout_loss(J, windows, dt, s_inc, alpha=0.0):
    """windows: (batch, H+1, R, n); s_inc: (R, n).  Mean over batch and steps (weighted) of
    sum_{r,i} ((B_hat - B)/s_inc)^2."""
    H = windows.shape[1] - 1
    pred = J.rollout(windows[:, 0], dt, H)
    e = (((pred - windows[:, 1:]) / s_inc) ** 2).sum((-1, -2))   # (batch, H)
    return (e * step_weights(H, alpha)).mean(), pred


class RegionalPhysics:
    """Pointwise Fisher-KPP residual per region at collocation points (interface flux excluded),
    trapezoid in time, with the regional reference physics (D, rho_r, beta)."""

    def __init__(self, DD, D, rho, beta, colloc, basis_fn, K=3):
        self.D, self.rho, self.beta = D, np.asarray(rho, float), beta
        self.phi, self.lap = [], []
        for r, reg in enumerate(DD.regions):
            phi, _, lap = basis_fn(reg.xy[colloc[r]], K, reg.Lx, reg.Ly)
            self.phi.append(torch.tensor(phi)); self.lap.append(torch.tensor(lap))

    def residual(self, Bn, Bn1, dt):
        """list over regions of residual arrays (..., P_r), u/day."""
        out = []
        for r in range(len(self.phi)):
            P, Lp = self.phi[r], self.lap[r]
            un, un1 = Bn[..., r, :] @ P.T, Bn1[..., r, :] @ P.T
            f = lambda b, u: self.D * (b @ Lp.T) + self.rho[r] * u - self.beta * u ** 2  # noqa: E731
            out.append((un1 - un) / dt - 0.5 * (f(Bn[..., r, :], un) + f(Bn1[..., r, :], un1)))
        return out

    def rms(self, Bn, Bn1, dt):
        res = self.residual(Bn, Bn1, dt)
        return float(torch.sqrt(sum((x ** 2).sum() for x in res) / sum(x.numel() for x in res)))

    def fields(self, B):
        return [B[..., r, :] @ self.phi[r].T for r in range(len(self.phi))]


def joint_losses(J, ds, dt, s_inc, phys, s_field, cfg, H, alpha, win_idx=None):
    a_n, a_np1 = ds["a_n"], ds["a_np1"]
    one = J.step(a_n, dt)
    out = dict(data=(((one - a_np1) / s_inc) ** 2).sum((-1, -2)).mean())
    res = phys.residual(a_n, one, dt)
    out["pde"] = sum((x ** 2).mean() for x in res) / len(res) / s_field ** 2
    W = ds["windows"][H]
    if win_idx is not None:
        W = W[win_idx]
    out["roll"], pred = rollout_loss(J, W, dt, s_inc, alpha)
    neg = sum((torch.relu(-u) ** 2).mean() for u in phys.fields(one)) / len(res)
    bound = cfg.get("stab_bound_factor", 3.0) * float(a_n.norm(dim=-1).max())
    stab = (torch.relu(pred.norm(dim=-1) - bound) ** 2).mean()
    rw = cfg.get("reg_weights", {})
    out["reg"] = rw.get("neg", 1.0) * neg + rw.get("stab", 1.0) * stab + rw.get("param", 1e-4) * J.hamiltonian_norm()
    out["total"] = (cfg["lambda_data"] * out["data"] + cfg["lambda_pde"] * out["pde"] +
                    cfg["lambda_roll"] * out["roll"] + cfg["lambda_reg"] * out["reg"])
    return out


def make_joint_dataset(A, horizons):
    """A: (T, R, n) training series -> one-step pairs and windows for every horizon."""
    t = lambda v: torch.tensor(np.asarray(v))  # noqa: E731
    return dict(a_n=t(A[:-1]), a_np1=t(A[1:]),
                windows={H: t(np.stack([A[i:i + H + 1] for i in range(len(A) - H)])) for H in horizons})


def val_score(J, B0, V, dt, s_inc):
    """Normalised rollout error over the validation window from the observed state B0 (R, n)."""
    with torch.no_grad(), J.cache():
        P = J.rollout(B0[None], dt, len(V))[0]
    return float((((P - V) / s_inc) ** 2).sum((-1, -2)).mean())


def train_joint(J, ds, val, dt, s_inc, phys, s_field, cfg, stages, alpha=0.0, seed=0, log_every=5, win_batch=32, select_from=None):
    """stages: list of (H, epochs) curriculum.  Model selection over all logged checkpoints
    from epoch E/6 on (identical for every curriculum of the same total length) by the validation-window rollout; the stage (= training
    horizon) of the selected checkpoint is recorded."""
    torch.manual_seed(seed); np.random.seed(seed)
    params = [p for p in J.parameters() if p.requires_grad]
    E = sum(e for _, e in stages)
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E, eta_min=cfg["lr"] * 0.05)
    hist = {k: [] for k in ("epoch", "H", "total", "data", "roll", "pde", "reg", "val", "pde_rms_train_onestep",
                            "pde_rms_val_rollout", "time")}
    best, best_state, best_meta = np.inf, copy.deepcopy(J.state_dict()), None
    t0, ep = time.time(), 0
    select_from = E // 6 if select_from is None else select_from   # same selection window for every curriculum
    B0, V = val
    for si, (H, ne) in enumerate(stages):
        nW = ds["windows"][H].shape[0]
        for k in range(ne):
            idx = torch.randperm(nW)[:win_batch] if win_batch and win_batch < nW else None
            opt.zero_grad()
            with J.cache():
                out = joint_losses(J, ds, dt, s_inc, phys, s_field, cfg, H, alpha, idx)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 10.0))
            opt.step(); sched.step()
            if ep % log_every == 0 or ep == E - 1:
                with torch.no_grad(), J.cache():
                    o = joint_losses(J, ds, dt, s_inc, phys, s_field, cfg, H, alpha)
                    one = J.step(ds["a_n"], dt)
                    P = J.rollout(B0[None], dt, len(V))[0]
                vs = float((((P - V) / s_inc) ** 2).sum((-1, -2)).mean())
                seq = torch.cat([B0[None], P])
                for kk in ("total", "data", "roll", "pde", "reg"):
                    hist[kk].append(float(o[kk]))
                hist["epoch"].append(ep); hist["H"].append(H); hist["val"].append(vs); hist["time"].append(time.time() - t0)
                hist["pde_rms_train_onestep"].append(phys.rms(ds["a_n"], one, dt))
                hist["pde_rms_val_rollout"].append(phys.rms(seq[:-1], seq[1:], dt))
                if ep >= select_from and np.isfinite(vs) and vs < best:
                    best, best_state, best_meta = vs, copy.deepcopy(J.state_dict()), dict(epoch=ep, stage=si, H=H)
            ep += 1
    J.load_state_dict(best_state)
    hist["train_time_s"] = time.time() - t0
    hist["best_val"], hist["selected"] = best, best_meta
    return hist, J
