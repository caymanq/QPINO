## Synthetic Fisher-KPP validation (12 unseen initial conditions, dt = 0.1, 40-step rollouts)

| model | description | params | 1-step rel. L2 (increment) | 1-step rel. L2 (state) | 40-step rollout rel. L2 | PDE residual RMS (full, trap.) | Galerkin residual RMS (trap.) | 400-step: dist. to u=K | train s |
|---|---|---|---|---|---|---|---|---|---|
| Persistence | a(t+dt) = a(t) | 0 | 1.0000 | 2.75e-02 | 0.6855 | 1.88e-01 | 8.38e-02 | 0.815 | 0 |
| A | Classical Galerkin ODE (fitted / true D, rho, beta) | 3 | 0.0011 | 3.12e-05 | 0.0004 | 2.75e-03 | 2.55e-05 | 0.000 | 0 |
| A-fit | Galerkin ODE, fitted params | 3 | 0.0014 | 3.88e-05 | 0.0006 | 2.77e-03 | 1.15e-04 | 0.000 | 0 |
| B | Classical MLP, same loss as D | 165 | 0.0471 ± 0.0099 | 1.30e-03 ± 2.73e-04 | 0.0145 ± 0.0032 | 1.02e-02 ± 1.71e-03 | 3.97e-03 ± 8.45e-04 | 0.075 | 12 |
| B-data | Classical MLP, no PDE loss | 165 | 0.0474 ± 0.0101 | 1.31e-03 ± 2.79e-04 | 0.0144 ± 0.0032 | 1.03e-02 ± 1.74e-03 | 4.00e-03 ± 8.61e-04 | 0.079 | 12 |
| C | Ry embedding + hardware-efficient ansatz, same loss | 105 | 0.0857 ± 0.0046 | 2.36e-03 ± 1.28e-04 | 0.0324 ± 0.0043 | 1.78e-02 ± 5.83e-04 | 7.20e-03 ± 3.80e-04 | 0.138 | 47 |
| D | QPINO: custom Hamiltonian, structured graph, data+PDE loss | 105 | 0.1005 ± 0.0167 | 2.77e-03 ± 4.59e-04 | 0.0435 ± 0.0110 | 2.05e-02 ± 2.69e-03 | 8.44e-03 ± 1.35e-03 | 1.014 | 40 |
| D-mlpdec | QPINO (D) with an MLP decoder | 366 | 0.0387 ± 0.0106 | 1.07e-03 ± 2.91e-04 | 0.0151 ± 0.0065 | 8.65e-03 ± 1.77e-03 | 3.24e-03 ± 8.78e-04 | 0.012 | 41 |
| E | QPINO without physics loss | 105 | 0.1003 ± 0.0167 | 2.76e-03 ± 4.61e-04 | 0.0435 ± 0.0111 | 2.04e-02 ± 2.71e-03 | 8.43e-03 ± 1.36e-03 | 1.015 | 44 |
| F | QPINO without ZZ couplings | 96 | 0.3089 ± 0.0194 | 8.50e-03 ± 5.33e-04 | 0.1336 ± 0.0300 | 5.85e-02 ± 3.57e-03 | 2.61e-02 ± 1.60e-03 | 5.810 | 38 |

Fitted (D, rho, beta) from training coefficients: 0.0980, 0.9985, 0.9981 (true 0.1, 1, 1).

### Generalisation (seed-0 models): time step and resolution

| model | rollout @ dt=0.05 | rollout @ dt=0.2 | 1-step incr. @ dt=0.05 | 1-step incr. @ dt=0.2 | rollout from N=8 grid | rollout from N=256 grid |
|---|---|---|---|---|---|---|
| Persistence | 0.6843 | 0.6879 | 1.0000 | 1.0000 | 0.6855 | 0.6855 |
| A | 0.0004 | 0.0004 | 0.0011 | 0.0011 | 0.0004 | 0.0004 |
| B | 0.0181 | 0.0204 | 0.0531 | 0.0562 | 0.0184 | 0.0184 |
| B-data | 0.0182 | 0.0205 | 0.0533 | 0.0564 | 0.0184 | 0.0184 |
| C | 0.0373 | 0.0401 | 0.0910 | 0.0972 | 0.0380 | 0.0380 |
| D | 0.1582 | 0.2124 | 0.4925 | 0.5642 | 0.0302 | 0.0302 |
| D-mlpdec | 0.1527 | 0.4120 | 0.7269 | 0.8127 | 0.0130 | 0.0130 |
| E | 0.1583 | 0.2121 | 0.4927 | 0.5645 | 0.0303 | 0.0303 |
| F | 0.1400 | 0.2962 | 0.3726 | 0.5678 | 0.0914 | 0.0914 |

Time-window generalisation (train on t <= 2 only, test on t in [2, 4] of unseen ICs): D: later-window 1-step incr. 0.113, rollout 0.020; B: later-window 1-step incr. 0.128, rollout 0.017; Persistence: later-window 1-step incr. 1.000, rollout 0.230

## JHU COVID-19, I-10 transect: July 2020 test (exploratory)

One-step = predict day t+1 from the observed day t (31 July targets). Rollout = 31 days autonomously from the observed Jun 30 state. Field errors are against the OBSERVED 32-bin field (cases/100k/day, 7-day mean) and therefore include the M=2 truncation error.

| model | params | 1-step coeff rel. L2 | 1-step increment rel. L2 | 1-step field RMSE | rollout coeff rel. L2 | rollout field RMSE | day-7 field RMSE | day-31 field RMSE | val (Jun 17-30) increment rel. L2 |
|---|---|---|---|---|---|---|---|---|---|
| Persistence | 0 | 0.0423 | 1.000 | 13.70 | 0.388 | 19.71 | 14.61 | 19.61 | - |
| LinearTrend | 0 | 0.0402 | 0.951 | 13.70 | 0.298 | 18.42 | 13.98 | 28.06 | - |
| A | 3 | 0.0434 | 1.025 | 13.71 | 0.327 | 18.01 | 14.38 | 16.59 | - |
| B | 165 | 0.0512 ± 0.0012 | 1.210 ± 0.029 | 13.76 ± 0.00 | 0.456 ± 0.091 | 22.26 ± 2.75 | 16.23 ± 2.10 | 23.14 ± 3.18 | 1.427 ± 0.124 |
| B-data | 165 | 0.0428 ± 0.0010 | 1.012 ± 0.023 | 13.71 ± 0.01 | 0.313 ± 0.011 | 18.98 ± 0.18 | 13.80 ± 0.07 | 20.69 ± 0.21 | 0.854 ± 0.023 |
| C | 105 | 0.0502 ± 0.0029 | 1.188 ± 0.069 | 13.75 ± 0.02 | 0.418 ± 0.046 | 21.45 ± 1.55 | 15.99 ± 1.73 | 23.38 ± 2.88 | 1.191 ± 0.131 |
| D | 105 | 0.0479 ± 0.0038 | 1.132 ± 0.090 | 13.73 ± 0.02 | 0.412 ± 0.060 | 21.33 ± 1.38 | 15.01 ± 1.75 | 23.35 ± 1.69 | 1.211 ± 0.064 |
| D-mlpdec | 366 | 0.0614 ± 0.0019 | 1.450 ± 0.046 | 13.82 ± 0.02 | 0.507 ± 0.116 | 24.29 ± 4.59 | 16.20 ± 1.90 | 28.58 ± 11.12 | 1.574 ± 0.174 |
| E | 105 | 0.0423 ± 0.0003 | 0.999 ± 0.007 | 13.70 ± 0.00 | 0.381 ± 0.069 | 19.86 ± 1.89 | 14.63 ± 0.47 | 20.15 ± 3.06 | 0.964 ± 0.023 |
| F | 96 | 0.0513 ± 0.0038 | 1.214 ± 0.091 | 13.76 ± 0.03 | 0.440 ± 0.047 | 21.79 ± 1.92 | 15.58 ± 1.09 | 24.35 ± 5.34 | 1.247 ± 0.090 |

Full-resolution (no M=2 truncation) field references:

| reference | 1-step field RMSE | rollout field RMSE |
|---|---|---|
| Persistence-fullfield | 3.31 | 18.93 |
| LinearTrend-fullfield | 3.59 | 31.77 |
| A-fullgrid (FD Fisher-KPP on 32 bins) | 3.70 | 16.95 |

Fitted Fisher-KPP (Galerkin one-step least squares on Mar 1 - Jun 16): D = 1.49e-03 (x in [0, 2pi]) = 528 km^2/day, rho = 0.0352 /day, beta = 0.1038 /day, K = rho/beta = 0.339 (u units) = 28.0 cases/100k/day. Scaled fit loss 0.991 (persistence = 1 by construction of the scaling, roughly).

Profile over fixed D: D=0.0001: loss 0.991, rho 0.035, beta 0.115; D=0.001: loss 0.991, rho 0.035, beta 0.108; D=0.01: loss 1.000, rho 0.033, beta 0.0376; D=0.03: loss 1.093, rho 0.044, beta 1.62e-08; D=0.1: loss 2.093, rho 0.087, beta 4.1e-09

## Phase-12 verification numbers (pytest)

- **test_fourier_roundtrip**: max_abs_error=1.78e-15
- **test_galerkin_equations_vs_projection**: explicit_vs_convolution=7.11e-15, explicit_vs_projection=1.78e-14
- **test_conjugate_symmetry**: rhs_symmetry_error=1.78e-15, integrated_symmetry_drift=3.48e-37
- **test_solver_linear_modes**: spectral_max_err=2.36e-11, fd_err_32=0.00432, fd_err_64=0.00107, fd_err_128=0.000266, fd_order_1=2.02, fd_order_2=2
- **test_solver_logistic**: max_err=7.25e-11
- **test_solver_self_convergence**: err_N32=7.15e-10, err_N64=1.31e-11
- **test_galerkin_integrator_agreement**: err=5.27e-15
- **test_truncation_error_decreases_with_M**: rel_err_M1=0.0162, rel_err_M2=0.00215, rel_err_M3=0.000297, rel_err_M4=3.92e-05
- **test_galerkin_residual_converges**: forward_rms_dt0.1=0.00153, forward_rms_dt0.0125=0.00019, forward_order=1, trapezoid_rms_dt0.1=2.57e-05, trapezoid_rms_dt0.0125=4.02e-07, trapezoid_order=2, midpoint_rms_dt0.1=2.81e-05, midpoint_rms_dt0.0125=4.39e-07, midpoint_order=2
- **test_full_pde_residual_converges**: full_rms_M2_by_dt=[0.008104365744683881, 0.008107612214322867, 0.008108469713275356, 0.0081086869564751], full_rms_M2=0.00811, full_rms_M4=0.000413, full_rms_M8=5.43e-07, full_rms_M16=6.34e-08, galerkin_traj_full_residual=0.0099, unresolved_mode_rms=0.0099
- **test_embedding_equals_ry_product**: max_err=1.67e-16
- **test_all_circuits_unitary**: max_err=7.77e-15, exact_structured=6.88e-15, trotter1_structured=6.66e-16, trotter2_structured=1.11e-15, trotter8_structured=3.22e-15, exact_triad=3.55e-15, trotter1_triad=6.66e-16, trotter2_triad=8.88e-16, trotter8_triad=5.11e-15, exact_full=7.77e-15, trotter1_full=5.55e-16, trotter2_full=1.11e-15, trotter8_full=3.77e-15, exact_nearest=6e-15, trotter1_nearest=5.55e-16, trotter2_nearest=1.33e-15, trotter8_nearest=3.33e-15, exact_none=2.89e-15, trotter1_none=6.66e-16, trotter2_none=8.88e-16, trotter8_none=3.33e-15, hea=1.11e-15, reupload3_state_norm=2.22e-15
- **test_trotter_converges_first_order**: err_L4=0.21, err_L8=0.105, err_L16=0.0524, err_L32=0.0261, observed_order=1
- **test_torch_matches_pennylane**: exact=8.1e-15, trotter=2.55e-15, reupload2=3.75e-15, hea=8.88e-16
- **test_measurement_gradcheck**: max_rel_err_param_grads=8.13e-08, n_param_tensors=7, input_gradcheck=1
- **test_gradients_reach_all_parameters**: min_grad=0.0104
- **test_identity_at_zero_dt**: max_err=0
- **test_fourier_spectrum_degree**: R1_max_power_beyond_R=7.13e-17, R1_power_at_R=0.487, R2_max_power_beyond_R=5.78e-17, R2_power_at_R=0.096, R3_max_power_beyond_R=3.98e-17, R3_power_at_R=0.0155
- **test_training_loop_smoke**: initial_total=728, final_total=12.8, initial_data=219, final_data=4.25
