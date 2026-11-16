"""Build all 12 Phase-10 figures from saved results.  Run after the experiments."""
import json
import os

import numpy as np
import pandas as pd

from common import C, FIG, MET, ROOT, gk, fk
from src import visualization as vz
from src.custom_ansatz import get_graph
from src.preprocessing import I10_WAYPOINTS

J = lambda f: json.load(open(os.path.join(MET, f)))  # noqa: E731
made = []
Z = dict(np.load(os.path.join(ROOT, "data/processed/covid_i10_field.npz")))
Z["dates"] = pd.to_datetime(Z["dates"])
cnt = pd.read_csv(os.path.join(ROOT, "data/processed/covid_i10_counties.csv"))
CR = J("covid_results.json")
made.append(vz.fig_covid_data(Z, cnt, I10_WAYPOINTS, [(pd.Timestamp("2020-06-17"), "val"), (pd.Timestamp("2020-07-01"), "test")],
                              f"{FIG}/fig01_covid_data.png"))
made.append(vz.fig_m2_reconstruction(Z, CR["periodicity"], f"{FIG}/fig02_m2_reconstruction.png"))
S = C["synthetic"]
rng = np.random.default_rng(7)
x = 2 * np.pi * np.arange(128) / 128
t = np.linspace(0, 4, 81)
U = fk.solve(fk.random_initial_condition(x, rng), t, S["D"], S["rho"], S["beta"])
Ag = gk.integrate_galerkin(gk.field_to_coeffs(U[0], x), t, S["D"], S["rho"], S["beta"])
SR = J("synthetic_results.json")
made.append(vz.fig_galerkin_vs_full(t, x, U, Ag, SR["truncation"], f"{FIG}/fig03_galerkin_vs_full.png"))
made.append(vz.fig_circuits(f"{FIG}/fig04a_circuit_qpino.png", f"{FIG}/fig04b_circuit_hea.png", f"{FIG}/fig04c_coupling_graphs.png",
                            {g: get_graph(g) for g in ["structured", "triad", "full", "nearest", "none"]}))
hs, hc = J("synthetic_histories.json"), J("covid_histories.json")
made.append(vz.fig_loss_curves(hs, hc, f"{FIG}/fig05_loss_curves.png"))
made.append(vz.fig_data_vs_pde(hs, hc, f"{FIG}/fig06_data_vs_pde_losses.png"))
ZC = dict(np.load(os.path.join(MET, "covid_arrays.npz")))
ZC["dates"] = pd.to_datetime(ZC["dates"])
made.append(vz.fig_covid_pred_vs_obs(ZC, ["Persistence", "LinearTrend", "A", "B", "D"], f"{FIG}/fig07_covid_pred_vs_obs.png"))
made.append(vz.fig_multistep({k: v for k, v in SR["per_seed"].items() if k != "B-data"},
                             {k: v for k, v in CR["per_seed"].items() if k in ["Persistence", "LinearTrend", "A", "B", "C", "D", "E", "F"]},
                             f"{FIG}/fig08_multistep_forecasts.png"))
SA = dict(np.load(os.path.join(MET, "synthetic_arrays.npz")))
made.append(vz.fig_fourier_traj(SA, ZC, f"{FIG}/fig09_fourier_trajectories.png"))
if os.path.exists(os.path.join(MET, "ansatz_comparison.json")):
    AC = J("ansatz_comparison.json")
    made.append(vz.fig_spectrum(AC["fourier_spectrum"], f"{FIG}/fig10_frequency_spectrum.png"))
    if os.path.exists(os.path.join(MET, "physics_loss_ablation.json")):
        PL = J("physics_loss_ablation.json")
        made.append(vz.fig_ablations(AC, PL, {"Persistence": CR["summary"]["Persistence"]["field_rmse_per100k_rollout"]["mean"]},
                                     f"{FIG}/fig11_ablations.png"))
if os.path.exists(os.path.join(MET, "scalability.json")):
    made.append(vz.fig_runtime(J("scalability.json"), SR["summary"], f"{FIG}/fig12_runtime_resources.png"))
print("\n".join(made))
