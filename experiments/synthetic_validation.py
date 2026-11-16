"""Primary physical validation on synthetic Fisher-KPP data (known D, rho, beta).

Run:  python experiments/synthetic_validation.py [--epochs N] [--seeds 0 1 2]
Outputs: results/metrics/synthetic_*.json, results/checkpoints/synthetic_*.pt,
         results/metrics/synthetic_arrays.npz (for figures)
"""
import argparse
import time

import numpy as np
import torch

from common import (C, MET, X_C, ev, evaluate_on_trajs, fit_and_train, fk, gk, make_norm,
                    PersistenceOperator, GalerkinOperator, save_json, synthetic_data, count_params)

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=C["train"]["epochs"])
p.add_argument("--seeds", type=int, nargs="+", default=C["seeds"])
p.add_argument("--models", nargs="+", default=["B", "B-data", "C", "D", "D-mlpdec", "E", "F"])
args = p.parse_args()
S = C["synthetic"]
phys = (S["D"], S["rho"], S["beta"])
H = S["horizon"]
t0 = time.time()

# ------------------------------------------------------------------ data
data = synthetic_data()
dt = data["dt"]
norm, scales = make_norm(data["train"], dt, X_C)
print(f"synthetic data: {len(data['train'])}/{len(data['val'])}/{len(data['test'])} trajectories, "
      f"{len(data['t'])} times, dt={dt}", flush=True)

# truncation diagnostics: full PDE (N=128) vs M=2 projection vs Galerkin ODE
x = data["x"]
trunc = {}
U = np.stack(data["U"])
for Mk in [1, 2, 3, 4, 8]:
    F_ = np.fft.fft(U, axis=-1); F_[..., Mk + 1:U.shape[-1] - Mk] = 0
    UM = np.fft.ifft(F_, axis=-1).real
    trunc[f"M{Mk}_rel_l2"] = float(np.linalg.norm(UM - U) / np.linalg.norm(U))
gal_vs_proj = []
for A, u in zip(data["test"], data["U_test"]):
    Ag = gk.integrate_galerkin(A[0], data["t"], *phys)
    gal_vs_proj.append(np.linalg.norm(Ag - A) / np.linalg.norm(A))
trunc["galerkin_ode_vs_projected_full_rel_l2"] = float(np.mean(gal_vs_proj))
print("truncation:", trunc, flush=True)

# parameter identification from training coefficients
A_n = np.concatenate([T[:-1] for T in data["train"]]); A_np1 = np.concatenate([T[1:] for T in data["train"]])
fit = gk.fit_galerkin_parameters(A_n, A_np1, dt, w=1 / scales["s_inc"].numpy(), n_starts=6)
print("fitted params (true 0.1, 1, 1):", fit, flush=True)

# ------------------------------------------------------------------ baselines without training
results, rolls = {}, {}
for name, m in [("Persistence", PersistenceOperator()), ("A", GalerkinOperator(*phys)),
                ("A-fit", GalerkinOperator(fit["D"], fit["rho"], fit["beta"]))]:
    r, roll = evaluate_on_trajs(m, data["test"], dt, phys, X_C, U_full=data["U_test"], x_full=x)
    r["n_params"] = count_params(m); results[name] = {"seeds": [r]}; rolls[name] = roll

# ------------------------------------------------------------------ trained models
cfg_over = dict(epochs=args.epochs)
hists, models = {}, {}
for name in args.models:
    results[name] = {"seeds": []}
    for seed in args.seeds:
        m, hist, tcfg = fit_and_train(name, data["train"], data["val"], dt, norm, scales, phys,
                                      S["c0"], seed, H, "synthetic", train_over=cfg_over)
        r, roll = evaluate_on_trajs(m, data["test"], dt, phys, X_C, U_full=data["U_test"], x_full=x)
        r.update(n_params=count_params(m), train_time_s=hist.get("train_time_s"))
        if hasattr(m, "c"):
            r["learned_c"] = float(m.c)
        stab, _ = ev.long_term_stability(m, np.stack([T[0] for T in data["test"]]), dt, 400, K=S["rho"] / S["beta"])
        r["long_term"] = stab
        results[name]["seeds"].append(r)
        if seed == args.seeds[0]:
            hists[name], rolls[name], models[name] = hist, roll, m
        print(f"{name:9s} s{seed}: 1-step rel(incr) {r['rel_l2_increment']:.4f} rel {r['rel_l2']:.2e} "
              f"rollout rel {r['rollout_rel_l2']:.4f} stab finite {stab['finite']} "
              f"dist_eq {stab.get('final_dist_to_equilibrium', 0):.3g} t={r['train_time_s']:.0f}s", flush=True)

for name in ["Persistence", "A", "A-fit"]:
    results[name]["seeds"][0]["long_term"] = ev.long_term_stability(
        {"Persistence": PersistenceOperator(), "A": GalerkinOperator(*phys),
         "A-fit": GalerkinOperator(fit["D"], fit["rho"], fit["beta"])}[name],
        np.stack([T[0] for T in data["test"]]), dt, 400, K=S["rho"] / S["beta"])[0]


def summarize(res):
    keys = ["mse", "rel_l2", "rel_l2_increment", "field_rmse_vs_truncated", "field_rmse_vs_observed",
            "rollout_rel_l2", "rollout_final_rel_l2", "rollout_field_rmse_vs_observed",
            "galerkin_residual_rms_forward", "galerkin_residual_rms_trapezoid",
            "pde_residual_rms_forward", "pde_residual_rms_trapezoid", "positivity_violation_frac",
            "rollout_positivity_violation_frac"]
    out = {}
    for name, v in res.items():
        rows = v["seeds"]
        out[name] = {k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows])))
                     for k in keys if k in rows[0]}
        out[name]["n_params"] = rows[0]["n_params"]; out[name]["n_seeds"] = len(rows)
        out[name]["train_time_s"] = float(np.mean([r.get("train_time_s") or 0 for r in rows]))
        out[name]["long_term_finite_all"] = all(r["long_term"]["finite"] for r in rows if "long_term" in r)
        out[name]["long_term_dist_to_equilibrium"] = float(np.mean([r["long_term"].get("final_dist_to_equilibrium", np.nan)
                                                                    for r in rows if "long_term" in r]))
    return out


summary = summarize(results)

# ------------------------------------------------------------------ generalisation (seed-0 models)
gen = {}
for dt2 in [0.05, 0.2]:
    d2 = synthetic_data(dt=dt2)  # same seed -> same initial conditions
    gen[f"dt={dt2}"] = {}
    for name in ["Persistence", "A"] + list(models):
        m = models.get(name) or {"Persistence": PersistenceOperator(), "A": GalerkinOperator(*phys)}[name]
        r, _ = evaluate_on_trajs(m, d2["test"], dt2, phys, X_C)
        gen[f"dt={dt2}"][name] = {k: r[k] for k in ["rel_l2_increment", "rel_l2", "rollout_rel_l2"]}
# resolution: project the same test trajectories from coarser / finer grids
res_gen = {}
for N2 in [8, 16, 32, 64, 256]:
    if N2 == 256:
        _, x2, U2 = fk.generate_trajectories(len(data["A"]), *phys, dt, S["n_steps"], N=256, seed=S["data_seed"])
        U2 = U2[-S["n_test"]:]
    else:
        stride = S["N"] // N2; U2 = np.stack(data["U_test"])[..., ::stride]; x2 = x[::stride]
    A2 = [gk.field_to_coeffs(u, x2) for u in U2]
    alias = float(np.mean([np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(A2, data["test"])]))
    res_gen[f"N={N2}"] = {"coeff_change_vs_N128": alias}
    for name in ["Persistence", "A"] + list(models):
        m = models.get(name) or {"Persistence": PersistenceOperator(), "A": GalerkinOperator(*phys)}[name]
        r, _ = evaluate_on_trajs(m, A2, dt, phys, X_C)
        res_gen[f"N={N2}"][name] = {k: r[k] for k in ["rel_l2_increment", "rollout_rel_l2"]}
gen["resolution"] = res_gen

# time-window generalisation: train D only on t <= 2, test on t in [2, 4] of the test ICs
half = S["n_steps"] // 2
tw = {}
for name in ["D", "B"]:
    m, hist, _ = fit_and_train(name, [T[:half + 1] for T in data["train"]], [T[:half + 1] for T in data["val"]],
                               dt, norm, scales, phys, S["c0"], 0, H, "synthetic_window", train_over=cfg_over)
    r_in, _ = evaluate_on_trajs(m, [T[:half + 1] for T in data["test"]], dt, phys, X_C)
    r_out, _ = evaluate_on_trajs(m, [T[half:] for T in data["test"]], dt, phys, X_C)
    tw[name] = dict(in_window=dict(rel_l2_increment=r_in["rel_l2_increment"], rollout_rel_l2=r_in["rollout_rel_l2"]),
                    later_window=dict(rel_l2_increment=r_out["rel_l2_increment"], rollout_rel_l2=r_out["rollout_rel_l2"]))
r_out, _ = evaluate_on_trajs(PersistenceOperator(), [T[half:] for T in data["test"]], dt, phys, X_C)
tw["Persistence"] = dict(later_window=dict(rel_l2_increment=r_out["rel_l2_increment"], rollout_rel_l2=r_out["rollout_rel_l2"]))
gen["time_window"] = tw

save_json(dict(config=S, truncation=trunc, fitted_params=fit, true_params=dict(D=phys[0], rho=phys[1], beta=phys[2]),
               summary=summary, per_seed=results, generalization=gen, epochs=args.epochs, seeds=args.seeds,
               total_time_s=time.time() - t0), f"{MET}/synthetic_results.json")
save_json(hists, f"{MET}/synthetic_histories.json")
np.savez_compressed(f"{MET}/synthetic_arrays.npz", t=data["t"], x=x, U0=data["U"][0], test=np.stack(data["test"]),
                    U_test0=data["U_test"][0], **{f"roll_{k}": v for k, v in rolls.items()})
print(f"done in {time.time() - t0:.0f}s")
