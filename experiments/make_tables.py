"""Render markdown result tables from results/metrics/*.json -> docs/results_tables.md"""
import json
import os

from common import MET, ROOT, DESCR

J = lambda f: json.load(open(os.path.join(MET, f))) if os.path.exists(os.path.join(MET, f)) else None  # noqa: E731
L = []


def ms(v, k, f="{:.4f}", scale=1.0):
    if k not in v:
        return "-"
    m, s = v[k]["mean"] * scale, v[k]["std"] * scale
    return (f.format(m) + (" ± " + f.format(s) if s > 0 else ""))


S = J("synthetic_results.json")
if S:
    L += ["## Synthetic Fisher-KPP validation (12 unseen initial conditions, dt = 0.1, 40-step rollouts)", "",
          "| model | description | params | 1-step rel. L2 (increment) | 1-step rel. L2 (state) | 40-step rollout rel. L2 | "
          "PDE residual RMS (full, trap.) | Galerkin residual RMS (trap.) | 400-step: dist. to u=K | train s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in S["summary"].items():
        L.append(f"| {k} | {DESCR.get(k, k if k != 'A-fit' else 'Galerkin ODE, fitted params')} | {v['n_params']} | "
                 f"{ms(v, 'rel_l2_increment')} | {ms(v, 'rel_l2', '{:.2e}')} | {ms(v, 'rollout_rel_l2')} | "
                 f"{ms(v, 'pde_residual_rms_trapezoid', '{:.2e}')} | {ms(v, 'galerkin_residual_rms_trapezoid', '{:.2e}')} | "
                 f"{v['long_term_dist_to_equilibrium']:.3f} | {v['train_time_s']:.0f} |")
    L += ["", f"Fitted (D, rho, beta) from training coefficients: {S['fitted_params']['D']:.4f}, {S['fitted_params']['rho']:.4f}, "
          f"{S['fitted_params']['beta']:.4f} (true 0.1, 1, 1).", ""]
    g = S["generalization"]
    L += ["### Generalisation (seed-0 models): time step and resolution", "",
          "| model | rollout @ dt=0.05 | rollout @ dt=0.2 | 1-step incr. @ dt=0.05 | 1-step incr. @ dt=0.2 | rollout from N=8 grid | rollout from N=256 grid |",
          "|---|---|---|---|---|---|---|"]
    for k in g["dt=0.05"]:
        L.append(f"| {k} | {g['dt=0.05'][k]['rollout_rel_l2']:.4f} | {g['dt=0.2'][k]['rollout_rel_l2']:.4f} | "
                 f"{g['dt=0.05'][k]['rel_l2_increment']:.4f} | {g['dt=0.2'][k]['rel_l2_increment']:.4f} | "
                 f"{g['resolution']['N=8'][k]['rollout_rel_l2']:.4f} | {g['resolution']['N=256'][k]['rollout_rel_l2']:.4f} |")
    tw = g["time_window"]
    L += ["", "Time-window generalisation (train on t <= 2 only, test on t in [2, 4] of unseen ICs): " +
          "; ".join(f"{k}: later-window 1-step incr. {v['later_window']['rel_l2_increment']:.3f}, rollout {v['later_window']['rollout_rel_l2']:.3f}"
                    for k, v in tw.items()), ""]

Cv = J("covid_results.json")
if Cv:
    sc = Cv["field"]["scale_S_per100k"]
    L += ["## JHU COVID-19, I-10 transect: July 2020 test (exploratory)", "",
          "One-step = predict day t+1 from the observed day t (31 July targets). Rollout = 31 days autonomously from the observed Jun 30 state. "
          "Field errors are against the OBSERVED 32-bin field (cases/100k/day, 7-day mean) and therefore include the M=2 truncation error.", "",
          "| model | params | 1-step coeff rel. L2 | 1-step increment rel. L2 | 1-step field RMSE | rollout coeff rel. L2 | rollout field RMSE | day-7 field RMSE | day-31 field RMSE | val (Jun 17-30) increment rel. L2 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in Cv["summary"].items():
        L.append(f"| {k} | {v.get('n_params', '-')} | {ms(v, 'rel_l2')} | {ms(v, 'rel_l2_increment', '{:.3f}')} | "
                 f"{ms(v, 'field_rmse_per100k_onestep', '{:.2f}')} | {ms(v, 'rollout_rel_l2', '{:.3f}')} | "
                 f"{ms(v, 'field_rmse_per100k_rollout', '{:.2f}')} | {ms(v, 'rollout_field_rmse_day7', '{:.2f}', sc)} | "
                 f"{ms(v, 'rollout_field_rmse_day31', '{:.2f}', sc)} | {ms(v, 'val_rel_l2_increment', '{:.3f}')} |")
    L += ["", "Full-resolution (no M=2 truncation) field references:", "", "| reference | 1-step field RMSE | rollout field RMSE |", "|---|---|---|"]
    for k, v in Cv["full_field_references"].items():
        L.append(f"| {k} | {v['field_rmse_per100k_onestep']:.2f} | {v['field_rmse_per100k_rollout']:.2f} |")
    fp = Cv["fitted_params"]
    L += ["", f"Fitted Fisher-KPP (Galerkin one-step least squares on Mar 1 - Jun 16): D = {fp['D']:.2e} (x in [0, 2pi]) = "
          f"{fp['D_physical_km2_per_day']:.0f} km^2/day, rho = {fp['rho']:.4f} /day, beta = {fp['beta']:.4f} /day, K = rho/beta = "
          f"{fp['K_carrying_capacity_u']:.3f} (u units) = {fp['K_carrying_capacity_u'] * sc:.1f} cases/100k/day. Scaled fit loss {fp['loss']:.3f} "
          "(persistence = 1 by construction of the scaling, roughly).", "", "Profile over fixed D: " +
          "; ".join(f"D={k}: loss {v['loss']:.3f}, rho {v['rho']:.3f}, beta {v['beta']:.3g}" for k, v in fp["profile_over_D"].items()), ""]

for fname, title, cols in [("ansatz_comparison.json", "Ansatz ablations", None), ("physics_loss_ablation.json", "Physics-loss ablations", None)]:
    A = J(fname)
    if not A:
        continue
    for regime, d in A.items():
        if not isinstance(d, dict) or not d or regime in ("posthoc_trotter_structured", "fourier_spectrum", "tau_structured"):
            continue
        first = next(iter(d.values()))
        if not isinstance(first, dict) or "rel_l2_increment" not in first:
            continue
        keys = [k for k in ["rel_l2_increment", "rollout_rel_l2", "dt0.05_rollout_rel_l2", "dt0.2_rollout_rel_l2",
                            "pde_residual_rms_trapezoid", "field_rmse_per100k_onestep", "field_rmse_per100k_rollout",
                            "n_params", "train_time_s"] if k in first]
        L += [f"## {title}: {regime}", "", "| variant | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for k, v in d.items():
            L.append(f"| {k} | " + " | ".join(ms(v, c, "{:.4g}") for c in keys) + " |")
        L.append("")
    if "posthoc_trotter_structured" in A:
        L += ["### Post-hoc Trotterisation of the exact-trained structured QPINO (seed 0)", "",
              "| L | max abs unitary error | RMS prediction deviation | 1-step incr. rel. L2 | rollout rel. L2 |", "|---|---|---|---|---|"]
        for Lk, v in A["posthoc_trotter_structured"].items():
            L.append(f"| {Lk} | {v['unitary_max_abs_err']:.3e} | {v['pred_rms_dev_from_exact']:.3e} | {v['rel_l2_increment']:.4f} | {v['rollout_rel_l2']:.4f} |")
        L += ["", f"Learned tau scale: {A['tau_structured']}", ""]

SC = J("scalability.json")
if SC:
    L += ["## Resources", "", "| circuit | depth | gates | 2-qubit gates | Hamiltonian params |", "|---|---|---|---|---|"]
    for k, v in SC["circuit_resources"].items():
        L.append(f"| {k} | {v['depth']} | {v['n_gates']} | {v['n_two_qubit']} | {v['n_hamiltonian_params']} |")
    L += ["", "Runtime (s per forward+backward, batch 256) vs qubits:", "", "| n | " + " | ".join(next(iter(SC['runtime_vs_qubits'].values())).keys()) + " |",
          "|---" * (1 + len(next(iter(SC['runtime_vs_qubits'].values())))) + "|"]
    for n, v in SC["runtime_vs_qubits"].items():
        L.append(f"| {n} | " + " | ".join(f"{x:.3g}" for x in v.values()) + " |")
    L += ["", f"Per-sample forward: {SC['per_sample_forward_s']}", "", f"Circuit evaluations: {SC['circuit_evaluations']}", ""]
    sh = SC["shot_sensitivity"]
    for key in ["synthetic", "covid_july"]:
        if key in sh:
            L += [f"Shot sensitivity ({key}): exact MSE {sh[key]['exact_mse']:.3e}; " +
                  "; ".join(f"{s} shots: MSE {v['mse_mean']:.3e}, dev {v['rms_dev_from_exact']:.2e}" for s, v in sh[key].items() if s.isdigit()), ""]
    if "pennylane_20000shots_max_abs_dev_from_exact" in sh:
        L += [f"PennyLane 20000-shot check: max |<O>_shots - <O>_exact| = {sh['pennylane_20000shots_max_abs_dev_from_exact']:.4f} "
              f"(1/sqrt(shots) = {sh['expected_1sigma_at_20000_shots']:.4f}).", ""]

V = J("verification_numbers.json")
if V:
    L += ["## Phase-12 verification numbers (pytest)", ""]
    for k, v in V.items():
        L.append(f"- **{k}**: " + ", ".join(f"{a}={b:.3g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
open(os.path.join(ROOT, "docs", "results_tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L)[:4000])
