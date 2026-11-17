"""Joint regional QPINO training with the interface flux inside the training rollouts and a long
rollout curriculum (report section 19).

    python usa2d_regional_joint.py --model D --seed 0 --flux 1 --curr long --alpha 0
flux 1/0 : classical interface flux inside training rollouts on/off (off = independent regions)
curr     : long = 7 -> 14 -> 21-day rollout windows (300 epochs each); short = 7 days (900 epochs, the §18 length)
alpha    : per-step rollout weights w_s ∝ s^-alpha
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from common import C, CKPT, MET, ROOT, ev, gk, save_json, count_params
from src import cosine2d_galerkin as c2
from src.qpino import MLPOperator, QPINO
from src.regional_galerkin import Decomposition
from src.regional_training import JointRegional, RegionalPhysics, make_joint_dataset, train_joint
from src.training import make_norm, set_seed

U2 = yaml.safe_load(open(os.path.join(ROOT, "configs", "usa2d.yaml")))
p = argparse.ArgumentParser()
p.add_argument("--model", default="D")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--flux", type=int, default=1)
p.add_argument("--curr", default="long")
p.add_argument("--alpha", type=float, default=0.0)
p.add_argument("--epochs_per_stage", type=int, default=300)
p.add_argument("--lambda_pde", type=float, default=U2["train"]["lambda_pde"])
p.add_argument("--tag_suffix", default="")
args = p.parse_args()
K, n = 3, 9
OUT = os.path.join(MET, "usa2d_regional_joint"); os.makedirs(OUT, exist_ok=True)
CK = os.path.join(CKPT, "usa2d_regional_joint"); os.makedirs(CK, exist_ok=True)
Z = np.load(os.path.join(ROOT, "data", "processed", "us_grid_density_256x128.npz"))
RG = np.load(os.path.join(ROOT, "data", "processed", "usa_regions_256x128.npz"))
lab, names = RG["label"].astype(int), [str(s) for s in RG["names"]]
u_full, mask = Z["u"].astype(np.float64), Z["mask"]
dates = pd.to_datetime(Z["dates"])
NY, NX = mask.shape
LREF = NY * float(Z["dy_km"]) / np.pi
DD = Decomposition(lab, K, float(Z["dx_km"]) / LREF, float(Z["dy_km"]) / LREF, names)
R = DD.R
Uf = u_full.reshape(len(dates), -1)
land = mask.ravel() > 0
A = DD.coeffs(Uf)
tr = np.where(dates <= U2["train_end"])[0]
va = np.where((dates > U2["train_end"]) & (dates <= U2["val_end"]))[0]
te = np.where((dates > U2["val_end"]) & (dates <= U2["test_end"]))[0]
i30, dt = va[-1], U2["dt"]
DAY = {d: int(np.where(dates[te] == pd.Timestamp(d))[0][0]) for d in ("2020-07-12", "2020-07-26")}
Uobs = Uf[:, land]
fit = json.load(open(os.path.join(MET, "usa2d_regional", f"fit_regional_K{K}.json")))
D_, rho_r, beta_ = fit["D"], np.array(fit["rho"]), fit["beta"]

# ------------------------------------------------------------------ models (one per region)
gk.set_basis("cosine2d")
SPECS = {"D": {}, "C": dict(ansatz="hea"), "B": dict(kind="mlp")}
models, s_inc = [], []
Atr = A[tr[0]:tr[-1] + 1]
set_seed(args.seed)
for r, reg in enumerate(DD.regions):
    c2.setup(reg.crop_mask(lab), K, reg.Lx, reg.Ly)
    S = c2.active()
    X_C = S["xy"][np.random.default_rng(0).choice(len(S["xy"]), min(U2["n_colloc"], len(S["xy"])), replace=False)]
    norm, scales = make_norm([Atr[:, r]], dt, X_C)
    s_inc.append(scales["s_inc"].numpy())
    spec = dict(SPECS[args.model]); mcfg = dict(C["model"]); mcfg["evolution"] = "exact_eigh"
    if spec.get("kind") == "mlp":
        models.append(MLPOperator(norm, U2["mlp_hidden"][n], C["mlp"]["depth"], args.seed, n))
    else:
        mcfg.update(spec); models.append(QPINO(norm, c0=U2["c0"], seed=args.seed, n=n, **mcfg))
s_inc = torch.tensor(np.stack(s_inc))
J = JointRegional(models, DD, D_, flux=bool(args.flux))
colloc = {r: np.random.default_rng(r).choice(len(reg.xy), min(400, len(reg.xy)), replace=False) for r, reg in enumerate(DD.regions)}
phys = RegionalPhysics(DD, D_, rho_r, beta_, colloc, c2.basis)
with torch.no_grad():
    F = phys.fields(torch.tensor(Atr))
    s_field = float(torch.cat([torch.diff(f, dim=0).flatten() for f in F]).std())
stages = [(7, args.epochs_per_stage), (14, args.epochs_per_stage), (21, args.epochs_per_stage)] if args.curr == "long" \
    else [(7, 3 * args.epochs_per_stage)]
ds = make_joint_dataset(Atr, sorted({h for h, _ in stages}))
val = (torch.tensor(A[tr[-1]]), torch.tensor(A[va]))
tcfg = dict(C["train"]); tcfg.update(lambda_pde=args.lambda_pde)
tag = f"joint_{args.model}_flux{args.flux}_{args.curr}_a{args.alpha:g}_s{args.seed}{args.tag_suffix}"
t0 = time.time()
hist, J = train_joint(J, ds, val, dt, s_inc, phys, s_field, tcfg, stages, alpha=args.alpha, seed=args.seed,
                      log_every=U2["train"].get("joint_log_every", 5))
wall = time.time() - t0
torch.save(J.state_dict(), os.path.join(CK, f"{tag}.pt"))


# ------------------------------------------------------------------ evaluation (same metrics as §18)
def field(B):
    return DD.field_flat(B, NY * NX)[..., land]


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
                rollout_rmse_by_day=e.tolist(),
                pde_violation_rollout=phys.rms(torch.tensor(seq[:-1]), torch.tensor(seq[1:]), dt),
                pred_rollout=roll.tolist())


res = {}
with torch.no_grad(), J.cache():
    for nm, fl in (("coupled", True), ("decoupled", False)):
        one = J.step(torch.tensor(A[te - 1]), dt, flux=fl).numpy()
        roll = J.rollout(torch.tensor(A[i30][None]), dt, len(te), flux=fl)[0].numpy()
        res[nm] = metrics(one, roll)
    one_tr = J.step(ds["a_n"], dt)
    res["pde_violation_train_onestep"] = phys.rms(ds["a_n"], one_tr, dt)
    Pv = J.rollout(val[0][None], dt, len(va))[0]
    sv = torch.cat([val[0][None], Pv])
    res["pde_violation_val_rollout"] = phys.rms(sv[:-1], sv[1:], dt)
    res["val_rollout_rmse"] = float(np.sqrt(((field(Pv.numpy()) - Uobs[va]) ** 2).mean()))
res.update(n_params=count_params(J), train_time_s=wall, epochs=sum(e for _, e in stages), selected=hist["selected"],
           best_val=hist["best_val"])
meta = dict(model=args.model, seed=args.seed, flux=args.flux, curr=args.curr, stages=stages, alpha=args.alpha, tag=tag,
            lambda_pde=args.lambda_pde, phys=dict(D=D_, rho=rho_r.tolist(), beta=beta_), regions=names, win_batch=32)
save_json(dict(meta=meta, results=res, history={k: v for k, v in hist.items() if isinstance(v, list)}),
          os.path.join(OUT, f"{tag}.json"))
print(tag, {k: round(v, 4) for k, v in res["coupled"].items() if isinstance(v, float)}, hist["selected"], round(wall), flush=True)
