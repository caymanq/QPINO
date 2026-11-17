"""Regional domain-decomposition QPINO for the contiguous USA (report section 18).

5 regions (state groups, data/processed/usa_regions_256x128.npz); each region: its own 3x3 cosine
basis on its bounding box (land-masked, natural no-flux), its own 9-qubit QPINO (real coefficients,
one qubit per mode); circuits are never entangled across regions.  Regions are coupled classically
at every time step through the interface diffusive flux S_r(b) (src/regional_galerkin.py).

    python usa2d_regional.py --baselines                      # floor, persistence, LT, regional A
    python usa2d_regional.py --model D --region 2 --seed 0    # train one regional model
    python usa2d_regional.py --assemble --model D --seed 0    # coupled (and decoupled) rollout
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import yaml

from common import C, CKPT, MET, ROOT, ev, gk, save_json, count_params
from src import cosine2d_galerkin as c2
from src.qpino import MLPOperator, QPINO
from src.regional_galerkin import Decomposition
from src.training import make_dataset, make_norm, set_seed, train

U2 = yaml.safe_load(open(os.path.join(ROOT, "configs", "usa2d.yaml")))
p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=3)
p.add_argument("--model", default="D")
p.add_argument("--region", type=int, default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=C["train"]["epochs"])
p.add_argument("--lambda_pde", type=float, default=U2["train"]["lambda_pde"])
p.add_argument("--baselines", action="store_true")
p.add_argument("--assemble", action="store_true")
args = p.parse_args()
K, n = args.K, args.K ** 2
OUT = os.path.join(MET, "usa2d_regional"); os.makedirs(OUT, exist_ok=True)
CK = os.path.join(CKPT, "usa2d_regional"); os.makedirs(CK, exist_ok=True)

Z = np.load(os.path.join(ROOT, "data", "processed", "us_grid_density_256x128.npz"))
RG = np.load(os.path.join(ROOT, "data", "processed", "usa_regions_256x128.npz"))
lab, names = RG["label"].astype(int), [str(s) for s in RG["names"]]
u_full, mask = Z["u"].astype(np.float64), Z["mask"]
dates = pd.to_datetime(Z["dates"])
NY, NX = mask.shape
LREF = NY * float(Z["dy_km"]) / np.pi
dx, dy = float(Z["dx_km"]) / LREF, float(Z["dy_km"]) / LREF
DD = Decomposition(lab, K, dx, dy, names)
R = DD.R
Uf = u_full.reshape(len(dates), -1)
land = mask.ravel() > 0
A = DD.coeffs(Uf)                                   # (T, R, n)
tr = np.where(dates <= U2["train_end"])[0]
va = np.where((dates > U2["train_end"]) & (dates <= U2["val_end"]))[0]
te = np.where((dates > U2["val_end"]) & (dates <= U2["test_end"]))[0]
i30, dt, H = va[-1], U2["dt"], U2["horizon"]
DAY = {d: int(np.where(dates[te] == pd.Timestamp(d))[0][0]) for d in ("2020-07-12", "2020-07-26")}
Uobs = Uf[:, land]
s_inc = np.stack([np.diff(A[tr[0]:tr[-1] + 1, r], axis=0).std(0) for r in range(R)])
s_inc = s_inc + 1e-3 * np.abs(s_inc).max(1, keepdims=True) + 1e-12


def field(B):
    return DD.field_flat(B, NY * NX)[..., land]


# ------------------------------------------------------------- regional Galerkin ODE fit
def fit_regional():
    """Global D, beta; per-region scalar rho_r (piecewise-constant spatial growth); multi-step
    (H=7) rollout objective on the last NW training days, NW chosen on the validation window."""
    path = os.path.join(OUT, f"fit_regional_K{K}.json")
    if os.path.exists(path):
        return json.load(open(path))
    w = torch.tensor(1 / s_inc)
    V = torch.tensor(A[tr[-1]:va[-1] + 1])
    best, sweep = None, []
    for NW in U2["rho_fit_windows"]:
        T = A[tr[0]:tr[-1] + 1][-NW:] if NW else A[tr[0]:tr[-1] + 1]
        Wn = torch.tensor(np.stack([T[i:i + H + 1] for i in range(len(T) - H)]))
        lD = torch.tensor(np.log(1e-4), requires_grad=True)
        lb = torch.tensor(np.log(0.04), requires_grad=True)
        rr = torch.full((R,), 0.03, requires_grad=True)
        opt = torch.optim.LBFGS([lD, lb, rr], lr=0.5, max_iter=U2["rho_fit_iters"], line_search_fn="strong_wolfe")

        def loss():
            a, L = Wn[:, 0], 0.0
            for s_ in range(1, H + 1):
                a = DD.rk4(a, dt, torch.exp(lD), rr, torch.exp(lb), substeps=4)
                L = L + (((a - Wn[:, s_]) * w) ** 2).mean() / H
            return L

        def closure():
            opt.zero_grad(); L = loss(); L.backward(); return L
        opt.step(closure)
        with torch.no_grad():
            Lf = float(loss())
            a, vr = V[0], []
            for s_ in range(1, len(V)):
                a = DD.rk4(a, dt, torch.exp(lD), rr, torch.exp(lb), substeps=8)
                vr.append(a.numpy())
            verr = ev.rel_l2(np.array(vr).reshape(len(vr), -1), V[1:].numpy().reshape(len(vr), -1))
        f = dict(D=float(torch.exp(lD)), beta=float(torch.exp(lb)), rho=rr.detach().numpy().tolist(), loss=Lf,
                 fit_window_days=NW or len(tr), val_rollout_rel_l2=verr)
        sweep.append(f); print("NW", NW, {k: v for k, v in f.items() if k != "rho"}, flush=True)
        if best is None or verr < best["val_rollout_rel_l2"]:
            best = dict(f)
    best["sweep"] = sweep
    best["D_physical_km2_per_day"] = best["D"] * LREF ** 2
    best["regions"] = names
    save_json(best, path)
    return best


fit = fit_regional()
D_, rho_r, beta_ = fit["D"], np.array(fit["rho"]), fit["beta"]
X_EV = {}
for r, reg in enumerate(DD.regions):
    k = np.random.default_rng(r).choice(len(reg.xy), min(400, len(reg.xy)), replace=False)
    X_EV[r] = k


def pde_violation(Bn, Bn1):
    """RMS (u/day) of the pointwise Fisher-KPP residual with the regional physics (D, rho_r, beta),
    trapezoid in time, 400 cells per region (interface flux not included: pointwise PDE only)."""
    Bn, Bn1 = np.asarray(Bn), np.asarray(Bn1)
    sq, cnt = 0.0, 0
    for r, reg in enumerate(DD.regions):
        phi, _, lap = c2.basis(reg.xy[X_EV[r]], K, reg.Lx, reg.Ly)
        un, un1 = Bn[..., r, :] @ phi.T, Bn1[..., r, :] @ phi.T
        fn = lambda b, u: D_ * (b @ lap.T) + rho_r[r] * u - beta_ * u ** 2  # noqa: E731
        res = (un1 - un) / dt - 0.5 * (fn(Bn[..., r, :], un) + fn(Bn1[..., r, :], un1))
        sq += float((res ** 2).sum()); cnt += res.size
    return float(np.sqrt(sq / cnt))


def metrics(one, roll):
    u_obs, fp = Uobs[te], field(A[te])
    f1, f2 = field(one), field(roll)
    e = np.sqrt(((f2 - u_obs) ** 2).mean(-1))
    seq = np.concatenate([A[i30][None], roll])
    flat = lambda X: X.reshape(len(X), -1)  # noqa: E731
    return dict(rel_l2_increment=ev.rel_l2(flat(one - A[te - 1]), flat(A[te] - A[te - 1])),
                rollout_rel_l2=ev.rel_l2(flat(roll), flat(A[te])),
                rmse_onestep=float(np.sqrt(((f1 - u_obs) ** 2).mean())),
                rmse_rollout=float(np.sqrt(((f2 - u_obs) ** 2).mean())),
                rmse_rollout_vs_projection=float(np.sqrt(((f2 - fp) ** 2).mean())),
                rmse_jul12=float(e[DAY["2020-07-12"]]), rmse_jul26=float(e[DAY["2020-07-26"]]),
                rollout_rmse_by_day=e.tolist(), pde_violation_rollout=pde_violation(seq[:-1], seq[1:]),
                pred_rollout=roll.tolist())


meta = dict(K=K, R=R, regions=names, cells=[len(r.flat) for r in DD.regions], n_faces=DD.n_faces,
            cond_G=[r.cond_G for r in DD.regions], bbox=[[int(r.i0), int(r.i1), int(r.j0), int(r.j1)] for r in DD.regions],
            fit=fit, lambda_pde=args.lambda_pde)

if args.baselines:
    res = {}
    proj = field(A)
    def trunc(ix):
        return float(np.mean(np.linalg.norm(proj[ix] - Uobs[ix], axis=1) / np.linalg.norm(Uobs[ix], axis=1)))
    res["truncation"] = dict(train=trunc(tr), val=trunc(va), july=trunc(te),
                             rmse_train=float(np.sqrt(((proj[tr] - Uobs[tr]) ** 2).mean())),
                             rmse_july=float(np.sqrt(((proj[te] - Uobs[te]) ** 2).mean())))
    res["Projection-floor"] = metrics(A[te], A[te])
    res["Projection-floor"]["pde_violation_observed"] = pde_violation(A[te - 1], A[te])
    res["Persistence"] = metrics(A[te - 1], np.repeat(A[i30][None], len(te), 0))
    Af = A.reshape(len(A), -1)
    res["LinearTrend"] = metrics(ev.linear_trend_predictions(Af, te, 7).reshape(-1, R, n),
                                 ev.linear_trend_rollout(Af, i30, len(te), 7).reshape(-1, R, n))
    for nm, cpl in (("A-regional-coupled", True), ("A-regional-decoupled", False)):
        one = DD.rk4(A[te - 1], dt, D_, rho_r, beta_, substeps=20, coupled=cpl)
        a, roll = A[i30], []
        for _ in te:
            a = DD.rk4(a, dt, D_, rho_r, beta_, substeps=20, coupled=cpl); roll.append(a)
        res[nm] = metrics(one, np.array(roll))
    # interface flux magnitude relative to the local RHS over July (observed coefficients)
    fl = DD.flux(A[te], D_); lo = DD.local_rhs(A[te], D_, rho_r, beta_)
    res["flux_to_local_rhs_ratio"] = float(np.linalg.norm(fl) / np.linalg.norm(lo))
    save_json(dict(meta=meta, results=res), os.path.join(OUT, f"baselines_regional_K{K}.json"))
    for k, v in res.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} if isinstance(v, dict) else v)
    raise SystemExit

SPECS = {"D": {}, "C": dict(ansatz="hea"), "B": dict(kind="mlp")}
if args.region is not None:      # ---------------------------------------- train one regional model
    r = args.region
    reg = DD.regions[r]
    gk.set_basis("cosine2d")
    c2.setup(reg.crop_mask(lab), K, reg.Lx, reg.Ly)
    S = c2.active()
    assert np.allclose(S["G"], reg.G)
    T = A[:, r]
    train_tr, val_tr = [T[tr[0]:tr[-1] + 1]], [T[tr[-1]:va[-1] + 1]]
    X_C = S["xy"][np.random.default_rng(0).choice(len(S["xy"]), min(U2["n_colloc"], len(S["xy"])), replace=False)]
    norm, scales = make_norm(train_tr, dt, X_C)
    phys = (D_, float(rho_r[r]), beta_)       # local weak form (the coupling is added classically)
    spec = dict(SPECS[args.model])
    mcfg = dict(C["model"]); mcfg["evolution"] = "exact_eigh"
    tcfg = dict(C["train"]); tcfg.update(epochs=args.epochs, seed=args.seed, lambda_pde=args.lambda_pde,
                                         log_every=U2["train"].get("log_every", 10))
    set_seed(args.seed)
    if spec.get("kind") == "mlp":
        m = MLPOperator(norm, U2["mlp_hidden"][n], C["mlp"]["depth"], args.seed, n)
    else:
        mcfg.update(spec); m = QPINO(norm, c0=U2["c0"], seed=args.seed, n=n, **mcfg)
    tag = f"regional_K{K}_{args.model}_r{r}_s{args.seed}"
    t0 = time.time()
    hist, m = train(m, make_dataset(train_tr, H), make_dataset(val_tr, min(H, len(val_tr[0]) - 1)), dt, tcfg, phys, scales, X_C)
    wall = time.time() - t0
    torch.save(m, os.path.join(CK, f"{tag}.pt"))
    save_json(dict(meta=dict(region=r, name=names[r], model=args.model, seed=args.seed, tag=tag, phys=list(phys),
                             lambda_pde=args.lambda_pde),
                   result=dict(n_params=count_params(m), train_time_s=wall, sec_per_epoch=wall / args.epochs,
                               final_train_pde_full=hist["pde_full"][-1], final_val_pde_full=hist["val_pde_full"][-1]),
                   history={k: hist[k] for k in ("epoch", "data", "val_data", "pde_full", "val_pde_full") if k in hist}),
              os.path.join(OUT, f"{tag}.json"))
    print(tag, wall, flush=True)
    raise SystemExit

if args.assemble:                # ---------------------------------------- coupled / decoupled rollout
    ms = [torch.load(os.path.join(CK, f"regional_K{K}_{args.model}_r{r}_s{args.seed}.pt"), weights_only=False) for r in range(R)]
    gk.set_basis("cosine2d")
    res = {}

    @torch.no_grad()
    def step(Bt, coupled):
        out = []
        for r, m in enumerate(ms):
            reg = DD.regions[r]
            c2.setup(reg.crop_mask(lab), K, reg.Lx, reg.Ly)
            ctx = m.cache_unitaries() if hasattr(m, "cache_unitaries") else torch.no_grad()
            with ctx:
                x = torch.tensor(np.atleast_2d(Bt[..., r, :]))
                out.append(m(x, dt).numpy().reshape(Bt[..., r, :].shape))
        Bn = np.stack(out, -2)
        return Bn + dt * DD.flux(Bt, D_) if coupled else Bn
    for nm, cpl in (("coupled", True), ("decoupled", False)):
        t0 = time.time()
        one = step(A[te - 1], cpl)
        a, roll = A[i30], []
        for _ in te:
            a = step(a, cpl); roll.append(a)
        res[nm] = metrics(one, np.array(roll)); res[nm]["eval_time_s"] = time.time() - t0
    rows = [json.load(open(os.path.join(OUT, f"regional_K{K}_{args.model}_r{r}_s{args.seed}.json")))["result"] for r in range(R)]
    res["n_params_total"] = sum(q["n_params"] for q in rows)
    res["train_time_s_per_region"] = [q["train_time_s"] for q in rows]
    res["train_pde_full_per_region"] = [q["final_train_pde_full"] for q in rows]
    save_json(dict(meta=meta | dict(model=args.model, seed=args.seed), results=res),
              os.path.join(OUT, f"assembled_K{K}_{args.model}_s{args.seed}.json"))
    print({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in res.items() if isinstance(v, dict)})
