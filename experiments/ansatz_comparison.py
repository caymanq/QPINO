"""Ansatz ablations: coupling graph, HEA, Trotter L, tau scaling, re-uploading, measurement
set, decoder, embedding scale alpha; plus the Fourier-spectrum analysis of trained models.

Run:  python experiments/ansatz_comparison.py [--epochs N] [--seeds 0 1] [--skip-covid]
"""
import argparse
import os
import time

import numpy as np
import torch

from common import (C, CKPT, MET, X_C, agg, covid_bundle, covid_eval, evaluate_on_trajs, ev,
                    fit_and_train, gk, make_dataset, make_norm, save_json, synthetic_data, count_params)
from src.qpino import QPINO
from src.training import train
from src.quantum_embedding import accessible_frequencies

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=C["train"]["epochs"])
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
p.add_argument("--covid-seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--skip-covid", action="store_true")
args = p.parse_args()
S = C["synthetic"]
phys = (S["D"], S["rho"], S["beta"])
t0 = time.time()
data = synthetic_data()
dt = data["dt"]
norm, scales = make_norm(data["train"], dt, X_C)
gen_sets = {dt2: synthetic_data(dt=dt2) for dt2 in [0.05, 0.2]}

VARIANTS = {
    # coupling graph
    "graph=structured": {}, "graph=triad": dict(graph="triad"), "graph=full": dict(graph="full"),
    "graph=nearest": dict(graph="nearest"), "graph=none (F)": dict(graph="none"),
    "HEA (C)": dict(ansatz="hea"),
    # evolution / Trotter
    "trotter L=1": dict(evolution="trotter", trotter_L=1), "trotter L=2": dict(evolution="trotter", trotter_L=2),
    "trotter L=4": dict(evolution="trotter", trotter_L=4), "trotter L=8": dict(evolution="trotter", trotter_L=8),
    # tau = c dt
    "tau fixed c=2": dict(tau_mode="fixed", c0=2.0), "tau fixed c=10": dict(tau_mode="fixed", c0=10.0),
    "tau fixed c=50": dict(tau_mode="fixed", c0=50.0), "tau learned c0=10": dict(tau_mode="learned", c0=10.0),
    # re-uploading
    "reupload R=2": dict(reupload=2), "reupload R=3": dict(reupload=3),
    # measurement / decoder
    "measure Z only": dict(measure_zz=False), "decoder MLP": dict(decoder="mlp"),
    "frozen circuit + MLP decoder": dict(decoder="mlp", _freeze=True),
    "frozen circuit + linear decoder": dict(_freeze=True),
    # embedding scale (frequency scale of the encoding)
    "alpha0=0.25": dict(alpha0=0.25), "alpha0=0.5": dict(alpha0=0.5), "alpha0=2": dict(alpha0=2.0),
    "alpha fixed=1": dict(train_alpha=False),
}


def run(variant, over, bundle, seed, covid=False):
    over = dict(over)
    freeze = over.pop("_freeze", False)
    c0 = over.pop("c0", bundle["c0"] if covid else S["c0"])
    if covid and "c0" in VARIANTS[variant]:
        c0 = VARIANTS[variant]["c0"] / 10.0  # same tau at the COVID dt = 1 day
    tr, va = bundle["train"], bundle["val"]
    d = bundle["dt"] if covid else dt
    nrm, sc = make_norm(tr, d, X_C)
    ph = bundle["phys"] if covid else phys
    m = QPINO(nrm, c0=c0, seed=seed, **{**C["model"], **over})
    if freeze:
        for n_, p_ in m.named_parameters():
            if not n_.startswith("decoder"):
                p_.requires_grad_(False)
    H = bundle["H"] if covid else S["horizon"]
    tcfg = dict(C["train"]); tcfg.update(epochs=args.epochs, seed=seed)
    torch.manual_seed(seed)
    hist, m = train(m, make_dataset(tr, H), make_dataset(va, min(H, len(va[0]) - 1)), d, tcfg, ph, sc, X_C)
    if covid:
        r = covid_eval(m, bundle)
    else:
        r0, _ = evaluate_on_trajs(m, data["test"], dt, phys, X_C)
        r = {k: r0[k] for k in ["rel_l2_increment", "rel_l2", "rollout_rel_l2", "pde_residual_rms_trapezoid",
                                 "galerkin_residual_rms_trapezoid"]}
        for dt2, d2 in gen_sets.items():
            g, _ = evaluate_on_trajs(m, d2["test"], dt2, phys, X_C)
            r[f"dt{dt2}_rel_l2_increment"] = g["rel_l2_increment"]; r[f"dt{dt2}_rollout_rel_l2"] = g["rollout_rel_l2"]
    r.update(n_params=count_params(m), quantum_params=m.quantum_params() if not freeze else 0,
             train_time_s=hist["train_time_s"], c=float(m.c), final_train_data=hist["data"][-1])
    return r, m


out = {"synthetic": {}, "covid": {}}
trained = {}
for v, over in VARIANTS.items():
    rows = []
    for s in args.seeds:
        r, m = run(v, over, dict(train=data["train"], val=data["val"]), s)
        rows.append(r)
        if s == args.seeds[0]:
            trained[v] = m
    out["synthetic"][v] = dict(agg(rows), per_seed=rows)
    print(f"[syn] {v:32s} incr {out['synthetic'][v]['rel_l2_increment']['mean']:.4f}±{out['synthetic'][v]['rel_l2_increment']['std']:.4f} "
          f"roll {out['synthetic'][v]['rollout_rel_l2']['mean']:.4f} dt0.2 roll {out['synthetic'][v]['dt0.2_rollout_rel_l2']['mean']:.3f} "
          f"t {rows[0]['train_time_s']:.0f}s", flush=True)

# post-hoc Trotterisation of the exact-trained structured model
m = trained["graph=structured"]
post = {}
Te = make_dataset(data["test"], 1)
with torch.no_grad():
    U_ex = m.layers[0].unitary(m.tau(dt)).detach()
    base = m(Te["a_n"], dt)
    for L in [1, 2, 4, 8, 16, 32, 64]:
        m.evolution, m.trotter_L = "trotter", L
        pr = m(Te["a_n"], dt)
        r, _ = evaluate_on_trajs(m, data["test"], dt, phys, X_C)
        post[L] = dict(unitary_max_abs_err=float((m.layers[0].unitary(m.tau(dt), "trotter", L) - U_ex).abs().max()),
                       pred_rms_dev_from_exact=float((pr - base).pow(2).mean().sqrt()),
                       rel_l2_increment=r["rel_l2_increment"], rollout_rel_l2=r["rollout_rel_l2"])
    m.evolution = "exact"
out["posthoc_trotter_structured"] = post
out["tau_structured"] = dict(learned_c=float(m.c), tau_train=float(m.tau(dt)),
                             H_norm=float(torch.linalg.matrix_norm(m.layers[0].hamiltonian(), 2)))

# Fourier spectra of trained models along each input angle (other inputs at the training mean)
spec = {}
for R, v in [(1, "graph=structured"), (2, "reupload R=2"), (3, "reupload R=3")]:
    m = trained[v]
    n_grid = 64
    spec[R] = {"theory_freqs_per_dim": accessible_frequencies(R)[0], "theory_n_freq_vectors": accessible_frequencies(R)[1]}
    for j in range(5):
        phi = torch.linspace(0, 2 * np.pi, n_grid + 1)[:-1]
        z = torch.zeros(n_grid, 5); z[:, j] = phi / m.alphas[j].detach()
        a = norm.mu + z * norm.sigma
        with torch.no_grad():
            Gv = m.rhs(a, dt).numpy()  # (n_grid, 5) decoder outputs
        c = np.abs(np.fft.fft(Gv, axis=0)) / n_grid
        spec[R][f"input{j}_power_by_freq"] = c[:8].sum(1).tolist()
        spec[R][f"input{j}_max_beyond_R"] = float(c[np.abs(np.fft.fftfreq(n_grid, 1 / n_grid)) > R].max())
    # 5-D spectrum on a (2R+2)^5 grid: distribution of power over total degree |omega|_1
    ng = 2 * R + 2
    g1 = torch.arange(ng) * 2 * np.pi / ng
    mesh = torch.stack(torch.meshgrid(*[g1] * 5, indexing="ij"), -1).reshape(-1, 5)
    a = norm.mu + (mesh / m.alphas.detach()) * norm.sigma
    with torch.no_grad():
        Gv = m.rhs(a, dt).numpy()[:, 0].reshape([ng] * 5)
    c = np.abs(np.fft.fftn(Gv)) / ng ** 5
    fr = np.fft.fftfreq(ng, 1 / ng)
    deg = sum(np.abs(np.meshgrid(*[fr] * 5, indexing="ij"))[i] for i in range(5))
    linf = np.max(np.abs(np.meshgrid(*[fr] * 5, indexing="ij")), axis=0)
    spec[R]["power_by_total_degree_output0"] = {int(d_): float((c[deg == d_] ** 2).sum()) for d_ in np.unique(deg)}
    spec[R]["max_coeff_beyond_R_linf"] = float(c[linf > R].max())
    spec[R]["n_nonzero_freq_vectors(>1e-10)"] = int((c > 1e-10).sum())
out["fourier_spectrum"] = spec

if not args.skip_covid and os.path.exists(os.path.join(MET, "covid_results.json")):
    B = covid_bundle()
    for v in ["graph=structured", "graph=triad", "graph=full", "graph=nearest", "graph=none (F)", "HEA (C)",
              "trotter L=1", "trotter L=4", "tau fixed c=2", "tau fixed c=10", "tau learned c0=10",
              "reupload R=2", "reupload R=3", "measure Z only", "decoder MLP", "frozen circuit + MLP decoder",
              "alpha0=0.25", "alpha0=2"]:
        rows = [run(v, VARIANTS[v], B, s, covid=True)[0] for s in args.covid_seeds]
        out["covid"][v] = dict(agg(rows), per_seed=rows)
        print(f"[covid] {v:32s} incr {out['covid'][v]['rel_l2_increment']['mean']:.3f} "
              f"field1 {out['covid'][v]['field_rmse_per100k_onestep']['mean']:.3f} "
              f"roll {out['covid'][v]['field_rmse_per100k_rollout']['mean']:.3f}", flush=True)
out["epochs"], out["seeds"], out["total_time_s"] = args.epochs, args.seeds, time.time() - t0
save_json(out, f"{MET}/ansatz_comparison.json")
print(f"done in {time.time() - t0:.0f}s")
