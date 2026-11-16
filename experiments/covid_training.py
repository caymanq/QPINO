"""Exploratory JHU COVID-19 experiment: I-10 corridor transect, train Mar 1 - Jun 16 2020,
validate Jun 17 - Jun 30, test = July 2020 (one-step-ahead and a 31-day rollout from Jun 30).

Run:  python experiments/covid_training.py [--epochs N] [--seeds 0 1 2]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import (C, MET, ROOT, X_C, ev, fit_and_train, fk, gk, make_dataset, make_norm,
                    GalerkinOperator, PersistenceOperator, save_json, count_params)
from src import data_loader as dl, preprocessing as pp

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=C["train"]["epochs"])
p.add_argument("--seeds", type=int, nargs="+", default=C["seeds"])
p.add_argument("--models", nargs="+", default=["B", "B-data", "C", "D", "D-mlpdec", "E", "F"])
args = p.parse_args()
CC = C["covid"]
t0 = time.time()

# ------------------------------------------------------------------ data
dl.download(f"{ROOT}/{CC['raw_dir']}", fallback_dir=CC["fallback_dir"])
summary = dl.dataset_summary(f"{ROOT}/{CC['raw_dir']}")
meta, cum = dl.load_us(f"{ROOT}/{CC['raw_dir']}")
F = pp.build_field(meta, cum, n_bins=CC["n_bins"], band_km=CC["band_km"], start=CC["start"],
                   end=CC["test_end"], train_end=CC["train_end"], smooth=CC["smooth_days"])
u, x, dates = F["u"], F["x"], F["dates"]
tr, va, te = pp.split_indices(dates, CC["train_end"], CC["val_end"], CC["test_end"])
A = gk.field_to_coeffs(u, x)
dt = CC["dt"]
print(f"{len(F['counties'])} counties, L={F['L_km']:.0f} km, {CC['n_bins']} bins "
      f"({F['L_km'] / CC['n_bins']:.0f} km), S={F['scale_S']:.2f} cases/100k/day", flush=True)
np.savez_compressed(f"{ROOT}/data/processed/covid_i10_field.npz", u=u, x=x, A=A,
                    incidence_per100k=F["incidence_per100k"], raw_daily_per100k=F["raw_daily_per100k"],
                    dates=np.array([str(d.date()) for d in dates]), s_km=F["s_centres_km"],
                    bin_pop=F["bin_pop"], scale_S=F["scale_S"])
F["counties"][["FIPS", "Admin2", "Province_State", "Lat", "Long_", "Population", "s_km", "dist_km", "bin"]] \
    .to_csv(f"{ROOT}/data/processed/covid_i10_counties.csv")

# ------------------------------------------------------------------ periodicity / truncation analysis
def trunc_err(field, xx):
    a = gk.field_to_coeffs(field, xx)
    return np.linalg.norm(field - gk.reconstruct(a, xx), axis=-1) / (np.linalg.norm(field, axis=-1) + 1e-12)

N = u.shape[1]
um = pp.mirror_extend(u); xm = gk.uniform_grid(2 * N, 0.5)
ramp = u[:, :1] + (u[:, -1:] - u[:, :1]) * (np.arange(N) / (N - 1))[None]
periodic_analysis = {
    "wrap_jump_rel_mean": float(np.nanmean(pp.wrap_jump(u[1:]))),
    "wrap_jump_rel_train_mean": float(np.nanmean(pp.wrap_jump(u[tr[7:]]))),
    "wrap_jump_rel_july_mean": float(np.nanmean(pp.wrap_jump(u[te]))),
    "M2_trunc_rel_l2_periodic": {"train": float(np.nanmean(trunc_err(u[tr[7:]], x))),
                                 "val": float(np.nanmean(trunc_err(u[va], x))),
                                 "july": float(np.nanmean(trunc_err(u[te], x)))},
    "M2_trunc_rel_l2_mirror": {"train": float(np.nanmean(trunc_err(um[tr[7:]], xm))),
                               "july": float(np.nanmean(trunc_err(um[te], xm)))},
    "M2_trunc_rel_l2_detrended_periodic": {"train": float(np.nanmean(trunc_err(u[tr[7:]] - ramp[tr[7:]], x))),
                                           "july": float(np.nanmean(trunc_err(u[te] - ramp[te], x)))},
}
for Mk in [1, 2, 4, 8]:
    Fq = np.fft.fft(u[te], axis=-1); Fq[:, Mk + 1:N - Mk] = 0
    periodic_analysis[f"july_trunc_rel_l2_M{Mk}"] = float(np.mean(np.linalg.norm(np.fft.ifft(Fq, axis=-1).real - u[te], axis=-1)
                                                                  / np.linalg.norm(u[te], axis=-1)))
print("periodicity/truncation:", periodic_analysis, flush=True)

# ------------------------------------------------------------------ splits and normalisation
train_traj = [A[tr[0]:tr[-1] + 1]]
val_traj = [A[tr[-1]:va[-1] + 1]]
test_traj = A[va[-1]:te[-1] + 1]  # Jun 30 .. Jul 31 (32 points, 31 one-step pairs)
norm, scales = make_norm(train_traj, dt, X_C)
u_test_obs = u[te]                # observed full 32-bin field Jul 1..31

# ------------------------------------------------------------------ Fisher-KPP parameter fit
An, Anp1 = A[tr[0]:tr[-1]], A[tr[0] + 1:tr[-1] + 1]
w = 1 / scales["s_inc"].numpy()
fit = gk.fit_galerkin_parameters(An, Anp1, dt, w=w, n_starts=12, seed=0)
profile = {}
for Dfix in [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]:
    best = None
    for s in range(4):
        rng = np.random.default_rng(s)
        logp = torch.tensor([rng.uniform(-5, 0), rng.uniform(-4, 1)], requires_grad=True)
        opt = torch.optim.LBFGS([logp], lr=0.5, max_iter=300, line_search_fn="strong_wolfe")
        a_n, a_np1 = torch.tensor(An), torch.tensor(Anp1)

        def cl():
            opt.zero_grad()
            l = (((gk.rk4_step(a_n, dt, Dfix, torch.exp(logp[0]), torch.exp(logp[1]), 4) - a_np1) * torch.tensor(w)) ** 2).mean()
            l.backward(); return l
        try:
            opt.step(cl); l = float(cl())
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(l) and (best is None or l < best[0]):
            best = (l, float(torch.exp(logp[0])), float(torch.exp(logp[1])))
    profile[str(Dfix)] = dict(loss=best[0], rho=best[1], beta=best[2])
L_km = F["L_km"]
fit["D_physical_km2_per_day"] = fit["D"] * (L_km / (2 * np.pi)) ** 2
fit["K_carrying_capacity_u"] = fit["rho"] / fit["beta"]
fit["profile_over_D"] = profile
print("fitted Fisher-KPP:", {k: v for k, v in fit.items() if k != "profile_over_D"}, flush=True)
phys = (fit["D"], fit["rho"], fit["beta"])

# ------------------------------------------------------------------ evaluation helpers
idx_july = np.arange(va[-1] + 1, te[-1] + 1)        # absolute indices of Jul 1..31
a_prev = torch.tensor(A[idx_july - 1]); a_true = A[idx_july]


def evaluate_preds(one, roll):
    """one: (31,5) one-step preds; roll: (31,5) rollout from Jun 30."""
    r = ev.point_metrics(one, a_true, a_prev.numpy(), X_C, phys, dt, u_test_obs, x)
    r.update(ev.rollout_metrics(roll[None], a_true[None], X_C, u_test_obs[None], x))
    r["field_rmse_per100k_onestep"] = r["field_rmse_vs_observed"] * F["scale_S"]
    r["field_rmse_per100k_rollout"] = r["rollout_field_rmse_vs_observed"] * F["scale_S"]
    rb = r["rollout_field_rmse_by_step_vs_observed"]
    r["rollout_field_rmse_day7"], r["rollout_field_rmse_day14"], r["rollout_field_rmse_day31"] = rb[6], rb[13], rb[30]
    return r


results, preds = {}, {}
# persistence / linear trend (coefficient space)
pers_one = A[idx_july - 1]; pers_roll = np.repeat(A[va[-1]][None], 31, 0)
lt_one = ev.linear_trend_predictions(A, idx_july, 7); lt_roll = ev.linear_trend_rollout(A, va[-1], 31, 7)
for name, (o, r) in {"Persistence": (pers_one, pers_roll), "LinearTrend": (lt_one, lt_roll)}.items():
    results[name] = {"seeds": [evaluate_preds(o, r)]}; preds[name] = (o, r)
# same baselines on the FULL 32-bin field (no M=2 truncation) - the honest field-level reference
full_ref = {}
for name, (fo, fr) in {"Persistence-fullfield": (u[idx_july - 1], np.repeat(u[va[-1]][None], 31, 0)),
                       "LinearTrend-fullfield": (u[idx_july - 1] + (u[idx_july - 1] - u[idx_july - 8]) / 7,
                                                 u[va[-1]] + np.arange(1, 32)[:, None] * (u[va[-1]] - u[va[-1] - 7]) / 7)}.items():
    full_ref[name] = dict(field_rmse_vs_observed=float(np.sqrt(((fo - u_test_obs) ** 2).mean())),
                          rollout_field_rmse_vs_observed=float(np.sqrt(((fr - u_test_obs) ** 2).mean())),
                          field_rel_l2_vs_observed=ev.rel_l2(fo, u_test_obs),
                          rollout_field_rel_l2_vs_observed=ev.rel_l2(fr, u_test_obs))
# A: Galerkin ODE with fitted params; A-fullgrid: finite-difference Fisher-KPP on the 32-bin field
gal = GalerkinOperator(*phys)
with torch.no_grad():
    results["A"] = {"seeds": [evaluate_preds(gal(a_prev, dt).numpy(), gal.rollout(torch.tensor(A[va[-1]][None]), dt, 31)[0].numpy())]}
    preds["A"] = (gal(a_prev, dt).numpy(), gal.rollout(torch.tensor(A[va[-1]][None]), dt, 31)[0].numpy())
fo = np.stack([fk.step(u[i - 1], dt, *phys, method="fd") for i in idx_july])
fr = fk.solve(u[va[-1]], np.arange(0, 32) * dt, *phys, method="fd")[1:]
full_ref["A-fullgrid (FD Fisher-KPP on 32 bins)"] = dict(
    field_rmse_vs_observed=float(np.sqrt(((fo - u_test_obs) ** 2).mean())),
    rollout_field_rmse_vs_observed=float(np.sqrt(((fr - u_test_obs) ** 2).mean())),
    field_rel_l2_vs_observed=ev.rel_l2(fo, u_test_obs), rollout_field_rel_l2_vs_observed=ev.rel_l2(fr, u_test_obs))
for k in full_ref:
    full_ref[k]["field_rmse_per100k_onestep"] = full_ref[k]["field_rmse_vs_observed"] * F["scale_S"]
    full_ref[k]["field_rmse_per100k_rollout"] = full_ref[k]["rollout_field_rmse_vs_observed"] * F["scale_S"]

hists = {}
H = CC["horizon"]
for name in args.models:
    results[name] = {"seeds": []}
    for seed in args.seeds:
        m, hist, _ = fit_and_train(name, train_traj, val_traj, dt, norm, scales, phys, CC["c0"], seed, H,
                                   "covid", train_over=dict(epochs=args.epochs))
        with torch.no_grad():
            o = m(a_prev, dt).numpy(); r = m.rollout(torch.tensor(A[va[-1]][None]), dt, 31)[0].numpy()
        res = evaluate_preds(o, r)
        res.update(n_params=count_params(m), train_time_s=hist.get("train_time_s"))
        if hasattr(m, "c"):
            res["learned_c"] = float(m.c)
        # validation-period (Jun 17-30) one-step error, used for honest model selection
        with torch.no_grad():
            vd = make_dataset(val_traj, 1); res["val_rel_l2_increment"] = ev.rel_l2(m(vd["a_n"], dt) - vd["a_n"], vd["a_np1"] - vd["a_n"])
        results[name]["seeds"].append(res)
        if seed == args.seeds[0]:
            hists[name], preds[name] = hist, (o, r)
        print(f"{name:9s} s{seed}: 1-step coeff rel {res['rel_l2']:.4f} incr {res['rel_l2_increment']:.3f} | "
              f"field rmse/100k 1-step {res['field_rmse_per100k_onestep']:.2f} roll {res['field_rmse_per100k_rollout']:.2f}",
              flush=True)

# mirrored (even-extension) representation: sensitivity of the periodic modelling choice
A_m = gk.field_to_coeffs(um, xm)
tr_m, va_m = [A_m[tr[0]:tr[-1] + 1]], [A_m[tr[-1]:va[-1] + 1]]
norm_m, sc_m = make_norm(tr_m, dt, X_C)
An_m, Anp1_m = A_m[tr[0]:tr[-1]], A_m[tr[0] + 1:tr[-1] + 1]
s_m = sc_m["s_inc"].numpy().copy(); s_m[[2, 4]] = 1.0
fit_m = gk.fit_galerkin_parameters(An_m, Anp1_m, dt, w=1 / s_m, n_starts=12, seed=0)
mirror = {"fitted_params": fit_m, "imag_parts_max_abs": float(np.abs(A_m[:, [2, 4]]).max())}
mD, _, _ = fit_and_train("D", tr_m, va_m, dt, norm_m, sc_m, (fit_m["D"], fit_m["rho"], fit_m["beta"]), CC["c0"], 0, H,
                         "covid_mirror", train_over=dict(epochs=args.epochs))
with torch.no_grad():
    o = mD(torch.tensor(A_m[idx_july - 1]), dt).numpy()
    r = mD.rollout(torch.tensor(A_m[va[-1]][None]), dt, 31)[0].numpy()
for nm, (oo, rr) in {"D-mirror": (o, r), "Persistence-mirror": (A_m[idx_july - 1], np.repeat(A_m[va[-1]][None], 31, 0))}.items():
    f1 = gk.reconstruct(oo, xm)[:, :N]; f2 = gk.reconstruct(rr, xm)[:, :N]
    mirror[nm] = dict(field_rmse_per100k_onestep=float(np.sqrt(((f1 - u_test_obs) ** 2).mean())) * F["scale_S"],
                      field_rmse_per100k_rollout=float(np.sqrt(((f2 - u_test_obs) ** 2).mean())) * F["scale_S"])
print("mirror:", mirror, flush=True)


def summarize(res):
    keys = ["mse", "rel_l2", "rel_l2_increment", "field_rmse_vs_truncated", "field_rmse_vs_observed",
            "field_rmse_per100k_onestep", "rollout_mse", "rollout_rel_l2", "rollout_final_rel_l2",
            "rollout_field_rmse_vs_observed", "field_rmse_per100k_rollout", "rollout_field_rmse_day7",
            "rollout_field_rmse_day14", "rollout_field_rmse_day31", "galerkin_residual_rms_forward",
            "pde_residual_rms_forward", "pde_residual_rms_trapezoid", "positivity_violation_frac",
            "rollout_positivity_violation_frac", "val_rel_l2_increment"]
    out = {}
    for name, v in res.items():
        rows = v["seeds"]
        out[name] = {k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows])))
                     for k in keys if k in rows[0]}
        out[name]["n_seeds"] = len(rows)
        out[name]["n_params"] = rows[0].get("n_params", {"Persistence": 0, "LinearTrend": 0, "A": 3}.get(name))
        out[name]["train_time_s"] = float(np.mean([r.get("train_time_s") or 0 for r in rows]))
    return out


save_json(dict(dataset=summary, field=dict(n_counties=len(F["counties"]), L_km=L_km, n_bins=CC["n_bins"],
                                           bin_km=L_km / CC["n_bins"], scale_S_per100k=F["scale_S"],
                                           n_negative_clipped=F["n_negative_clipped"],
                                           negative_mass_clipped=F["negative_mass_clipped"],
                                           band_km=CC["band_km"], states=sorted(F["counties"].Province_State.unique()),
                                           split=dict(train=[str(dates[tr[0]].date()), str(dates[tr[-1]].date())],
                                                      val=[str(dates[va[0]].date()), str(dates[va[-1]].date())],
                                                      test=[str(dates[te[0]].date()), str(dates[te[-1]].date())]),
                                           bin_pop_min=float(F["bin_pop"].min())),
               periodicity=periodic_analysis, fitted_params=fit, summary=summarize(results), full_field_references=full_ref,
               mirror_sensitivity=mirror, per_seed=results, epochs=args.epochs, seeds=args.seeds,
               total_time_s=time.time() - t0), f"{MET}/covid_results.json")
save_json(hists, f"{MET}/covid_histories.json")
np.savez_compressed(f"{MET}/covid_arrays.npz", u=u, x=x, A=A, dates=np.array([str(d.date()) for d in dates]),
                    idx_july=idx_july, i_jun30=va[-1], scale_S=F["scale_S"],
                    **{f"one_{k}": v[0] for k, v in preds.items()}, **{f"roll_{k}": v[1] for k, v in preds.items()})
print(f"done in {time.time() - t0:.0f}s")
