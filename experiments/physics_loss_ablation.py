"""Loss-weight / residual-type ablation for the QPINO (model D architecture).

Regimes: synthetic full data (32 trajectories), synthetic LOW data (2 trajectories, where a
physics prior can matter), and COVID (exploratory).
Run:  python experiments/physics_loss_ablation.py [--epochs N] [--seeds 0 1]
"""
import argparse
import os
import time

import numpy as np
import torch

from common import (C, MET, X_C, agg, covid_bundle, covid_eval, evaluate_on_trajs, make_dataset,
                    make_norm, save_json, synthetic_data, count_params)
from src.qpino import QPINO, MLPOperator
from src.training import train

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=C["train"]["epochs"])
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
p.add_argument("--covid-seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--settings", nargs="+", default=None)
p.add_argument("--regimes", nargs="+", default=["synthetic", "covid"])
p.add_argument("--out", default="physics_loss_ablation.json")
args = p.parse_args()
S = C["synthetic"]
phys_syn = (S["D"], S["rho"], S["beta"])
t0 = time.time()
data = synthetic_data()

SETTINGS = {
    "lam_pde=0 (E)": dict(lambda_pde=0.0),
    "lam_pde=0.01": dict(lambda_pde=0.01),
    "lam_pde=0.1 (D default)": dict(),
    "lam_pde=1": dict(lambda_pde=1.0),
    "lam_pde=10": dict(lambda_pde=10.0),
    "lam_pde=1, no curriculum": dict(lambda_pde=1.0, pde_ramp=[0.0, 1e-9]),
    "PDE-only (lam_data=0, lam_roll=0)": dict(lambda_data=0.0, lambda_roll=0.0, lambda_pde=1.0, pde_ramp=[0.0, 1e-9]),
    "lam_roll=0": dict(lambda_roll=0.0),
    "lam_roll=2": dict(lambda_roll=2.0),
    "residual=galerkin (projected)": dict(pde_target="galerkin"),
    "residual=both": dict(pde_target="both"),
    "scheme=forward": dict(scheme="forward"),
    "scheme=midpoint": dict(scheme="midpoint"),
    "no regularization": dict(lambda_reg=0.0),
}


def run(over, tr, va, test_fn, dt, phys, c0, H, seed, mlp=False):
    nrm, sc = make_norm(tr, dt, X_C)
    m = MLPOperator(nrm, C["mlp"]["hidden"], C["mlp"]["depth"], seed) if mlp else QPINO(nrm, c0=c0, seed=seed, **C["model"])
    tcfg = dict(C["train"]); tcfg.update(epochs=args.epochs, seed=seed); tcfg.update(over)
    hist, m = train(m, make_dataset(tr, H), make_dataset(va, min(H, len(va[0]) - 1)), dt, tcfg, phys, sc, X_C)
    r = test_fn(m)
    r.update(train_time_s=hist["train_time_s"], final_pde_full=hist["pde_full"][-1], final_pde_gal=hist["pde_gal"][-1],
             final_data=hist["data"][-1])
    return r


def syn_test(m):
    r, _ = evaluate_on_trajs(m, data["test"], data["dt"], phys_syn, X_C)
    return {k: r[k] for k in ["rel_l2_increment", "rel_l2", "rollout_rel_l2", "pde_residual_rms_trapezoid",
                              "galerkin_residual_rms_trapezoid", "rollout_positivity_violation_frac"]}


out = {"synthetic_full": {}, "synthetic_lowdata": {}, "covid": {}}
low_tr = [T[:11] for T in data["train"][:2]]  # 2 trajectories x 10 steps (t <= 1)
SEL = args.settings or list(SETTINGS)
for name, over in SETTINGS.items():
    if "synthetic" not in args.regimes or name not in SEL:
        continue
    rows = [run(over, data["train"], data["val"], syn_test, data["dt"], phys_syn, S["c0"], S["horizon"], s) for s in args.seeds]
    out["synthetic_full"][name] = dict(agg(rows), per_seed=rows)
    rows = [run(over, low_tr, data["val"], syn_test, data["dt"], phys_syn, S["c0"], S["horizon"], s) for s in args.seeds]
    out["synthetic_lowdata"][name] = dict(agg(rows), per_seed=rows)
    print(f"[syn] {name:36s} full: incr {out['synthetic_full'][name]['rel_l2_increment']['mean']:.4f} "
          f"roll {out['synthetic_full'][name]['rollout_rel_l2']['mean']:.4f} | low: incr "
          f"{out['synthetic_lowdata'][name]['rel_l2_increment']['mean']:.4f} roll {out['synthetic_lowdata'][name]['rollout_rel_l2']['mean']:.4f}",
          flush=True)
# MLP in the low-data regime with / without physics (is the physics benefit model-specific?)
for name, over in [("MLP lam_pde=0", dict(lambda_pde=0.0)), ("MLP lam_pde=1", dict(lambda_pde=1.0))]:
    if "synthetic" not in args.regimes or (args.settings is not None and name not in args.settings):
        continue
    rows = [run(over, low_tr, data["val"], syn_test, data["dt"], phys_syn, S["c0"], S["horizon"], s, mlp=True) for s in args.seeds]
    out["synthetic_lowdata"][name] = dict(agg(rows), per_seed=rows)
    print(f"[syn-low] {name:32s} incr {out['synthetic_lowdata'][name]['rel_l2_increment']['mean']:.4f} "
          f"roll {out['synthetic_lowdata'][name]['rollout_rel_l2']['mean']:.4f}", flush=True)

if "covid" in args.regimes and os.path.exists(os.path.join(MET, "covid_results.json")):
    B = covid_bundle()
    for name in ["lam_pde=0 (E)", "lam_pde=0.01", "lam_pde=0.1 (D default)", "lam_pde=1", "lam_pde=10",
                 "PDE-only (lam_data=0, lam_roll=0)", "lam_roll=0", "lam_roll=2", "residual=galerkin (projected)",
                 "scheme=forward"]:
        if name not in SEL:
            continue
        rows = [run(SETTINGS[name], B["train"], B["val"], lambda m: covid_eval(m, B), B["dt"], B["phys"], B["c0"], B["H"], s)
                for s in args.covid_seeds]
        out["covid"][name] = dict(agg(rows), per_seed=rows)
        print(f"[covid] {name:36s} incr {out['covid'][name]['rel_l2_increment']['mean']:.3f} "
              f"field1 {out['covid'][name]['field_rmse_per100k_onestep']['mean']:.3f} "
              f"roll {out['covid'][name]['field_rmse_per100k_rollout']['mean']:.3f}", flush=True)
out["epochs"], out["total_time_s"] = args.epochs, time.time() - t0
save_json(out, f"{MET}/{args.out}")
print(f"done in {time.time() - t0:.0f}s")
