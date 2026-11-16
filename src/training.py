"""Training loop for QPINO and baselines (data + PDE residual + rollout + regularisation)."""
from __future__ import annotations

import copy
import json
import os
import random
import time

import numpy as np
import torch

from . import galerkin, losses
from .qpino import Standardizer

torch.set_default_dtype(torch.float64)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(trajs, H):
    """trajs: list of (T,5) arrays -> one-step pairs and length-(H+1) rollout windows."""
    a_n, a_np1, wins = [], [], []
    for A in trajs:
        A = np.asarray(A)
        a_n.append(A[:-1]); a_np1.append(A[1:])
        for i in range(len(A) - H):
            wins.append(A[i:i + H + 1])
    t = lambda v: torch.tensor(np.asarray(v))  # noqa: E731
    return dict(a_n=t(np.concatenate(a_n)), a_np1=t(np.concatenate(a_np1)),
                windows=t(np.stack(wins)) if wins else None, H=H)


def make_norm(trajs, dt, x_c):
    A = np.concatenate([np.asarray(T)[:-1] for T in trajs])
    dA = np.concatenate([np.diff(np.asarray(T), axis=0) for T in trajs])
    mu, sigma = A.mean(0), A.std(0) + 1e-8
    s_inc = dA.std(0) + 1e-3 * np.abs(dA).max() + 1e-12
    U = galerkin.reconstruct(np.concatenate([np.asarray(T) for T in trajs]), x_c)
    dU = np.concatenate([np.diff(galerkin.reconstruct(np.asarray(T), x_c), axis=0) for T in trajs])
    s_field = dU.std() + 1e-12
    norm = Standardizer(mu, sigma, s_inc / dt)
    return norm, dict(s_inc=torch.tensor(s_inc), s_field=float(s_field), mu=mu, sigma=sigma,
                      u_max=float(U.max()))


def pde_weight(epoch, cfg):
    """Curriculum: data-dominated start, then linear ramp of lambda_PDE to its target."""
    lam = cfg["lambda_pde"]
    e0, e1 = cfg.get("pde_ramp", [0.2, 0.6])
    E = cfg["epochs"]
    f = (epoch / E - e0) / max(e1 - e0, 1e-9)
    return lam * float(np.clip(f, 0.0, 1.0))


def compute_losses(model, ds, dt, cfg, phys, scales, x_c, lam_pde, idx=None):
    a_n, a_np1 = ds["a_n"], ds["a_np1"]
    if idx is not None:
        a_n, a_np1 = a_n[idx], a_np1[idx]
    a_hat = model(a_n, dt)
    s_inc = scales["s_inc"]
    out = {"data": losses.data_loss(a_hat, a_np1, s_inc)}
    out["field"] = losses.field_loss(a_hat, a_np1, x_c) / scales["s_field"] ** 2
    L_full, L_gal = losses.pde_losses(a_n, a_hat, dt, phys, s_inc, scales["s_field"], x_c,
                                      cfg.get("scheme", "forward"))
    out["pde_full"], out["pde_gal"] = L_full, L_gal
    target = cfg.get("pde_target", "full")
    L_pde = {"full": L_full, "galerkin": L_gal, "both": 0.5 * (L_full + L_gal)}[target]
    pred_roll = None
    if cfg["lambda_roll"] > 0 and ds["windows"] is not None:
        out["roll"], pred_roll = losses.rollout_loss(model, ds["windows"], dt, s_inc)
    else:
        out["roll"] = torch.zeros(())
    K = phys[1] / phys[2] if phys[2] > 0 else None
    bound = cfg.get("stab_bound_factor", 3.0) * float(ds["a_n"].norm(dim=-1).max())
    reg, terms = losses.regularization(model, a_hat, x_c, K, pred_roll, bound, cfg.get("reg_weights", {}))
    out["reg"] = reg
    out["total"] = (cfg["lambda_data"] * out["data"] + lam_pde * L_pde +
                    cfg["lambda_roll"] * out["roll"] + cfg["lambda_reg"] * reg)
    return out


def train(model, train_ds, val_ds, dt, cfg, phys, scales, x_c, ckpt_path=None, verbose=False):
    set_seed(cfg.get("seed", 0))
    params = [p for p in model.parameters() if p.requires_grad]
    hist = {k: [] for k in ["total", "data", "field", "pde_full", "pde_gal", "roll", "reg",
                            "val_data", "val_roll", "val_pde_full", "val_pde_gal", "lam_pde", "time"]}
    if not params:
        return hist, model
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"], eta_min=cfg["lr"] * 0.05)
    best, best_state = np.inf, copy.deepcopy(model.state_dict())
    t0 = time.time()
    P = train_ds["a_n"].shape[0]
    bs = cfg.get("batch_size") or P
    for ep in range(cfg["epochs"]):
        lam = pde_weight(ep, cfg)
        perm = torch.randperm(P)
        for i in range(0, P, bs):
            opt.zero_grad()
            out = compute_losses(model, train_ds, dt, cfg, phys, scales, x_c, lam,
                                 idx=perm[i:i + bs] if bs < P else None)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 10.0))
            opt.step()
        sched.step()
        if ep % cfg.get("log_every", 10) == 0 or ep == cfg["epochs"] - 1:
            with torch.no_grad():
                out = compute_losses(model, train_ds, dt, cfg, phys, scales, x_c, lam)
                vout = compute_losses(model, val_ds, dt, cfg, phys, scales, x_c, lam)
            for k in ["total", "data", "field", "pde_full", "pde_gal", "roll", "reg"]:
                hist[k].append(float(out[k]))
            hist["val_data"].append(float(vout["data"]))
            hist["val_roll"].append(float(vout["roll"]))
            hist["val_pde_full"].append(float(vout["pde_full"]))
            hist["val_pde_gal"].append(float(vout["pde_gal"]))
            hist["lam_pde"].append(lam)
            hist["time"].append(time.time() - t0)
            hist.setdefault("epoch", []).append(ep)
            vscore = float(vout["data"]) + (float(vout["roll"]) if cfg["lambda_roll"] > 0 else 0.0)
            # model selection only after the PDE curriculum has reached its target weight
            # (the same selection window is used for every run, with or without physics loss)
            if ep >= cfg["epochs"] * cfg.get("select_after", cfg.get("pde_ramp", [0.2, 0.6])[1]) * 0.999:
                if np.isfinite(vscore) and vscore < best:
                    best, best_state = vscore, copy.deepcopy(model.state_dict())
            if verbose and ep % (10 * cfg.get("log_every", 10)) == 0:
                print(f"ep {ep:5d} tot {out['total']:.4g} data {out['data']:.4g} pde {out['pde_full']:.4g} "
                      f"roll {out['roll']:.4g} | val {vout['data']:.4g} {vout['roll']:.4g}", flush=True)
    model.load_state_dict(best_state)
    hist["train_time_s"] = time.time() - t0
    hist["best_val_score"] = best
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ckpt_path)
    return hist, model


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if torch.is_tensor(o):
            return o.detach().cpu().tolist()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
