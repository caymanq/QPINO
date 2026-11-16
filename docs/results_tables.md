## Synthetic Fisher-KPP validation (12 unseen initial conditions, dt = 0.1, 40-step rollouts)

| model | description | params | 1-step rel. L2 (increment) | 1-step rel. L2 (state) | 40-step rollout rel. L2 | PDE residual RMS (full, trap.) | Galerkin residual RMS (trap.) | 400-step: dist. to u=K | train s |
|---|---|---|---|---|---|---|---|---|---|
| Persistence | a(t+dt) = a(t) | 0 | 1.0000 | 2.75e-02 | 0.6855 | 1.88e-01 | 8.38e-02 | 0.815 | 0 |
| A | Classical Galerkin ODE (fitted / true D, rho, beta) | 3 | 0.0011 | 3.12e-05 | 0.0004 | 2.75e-03 | 2.55e-05 | 0.000 | 0 |
| A-fit | Galerkin ODE, fitted params | 3 | 0.0014 | 3.88e-05 | 0.0006 | 2.77e-03 | 1.15e-04 | 0.000 | 0 |
| B | Classical MLP, same loss as D | 165 | 0.0471 ± 0.0099 | 1.30e-03 ± 2.73e-04 | 0.0145 ± 0.0032 | 1.02e-02 ± 1.71e-03 | 3.97e-03 ± 8.45e-04 | 0.075 | 12 |
| B-data | Classical MLP, no PDE loss | 165 | 0.0474 ± 0.0101 | 1.31e-03 ± 2.79e-04 | 0.0144 ± 0.0032 | 1.03e-02 ± 1.74e-03 | 4.00e-03 ± 8.61e-04 | 0.079 | 28 |
| C | Ry embedding + hardware-efficient ansatz, same loss | 105 | 0.0857 ± 0.0046 | 2.36e-03 ± 1.28e-04 | 0.0324 ± 0.0043 | 1.78e-02 ± 5.83e-04 | 7.20e-03 ± 3.80e-04 | 0.138 | 226 |
| D | QPINO: custom Hamiltonian, structured graph, data+PDE loss | 105 | 0.1001 ± 0.0167 | 2.76e-03 ± 4.59e-04 | 0.0434 ± 0.0110 | 2.04e-02 ± 2.70e-03 | 8.41e-03 ± 1.35e-03 | 1.015 | 39 |
| D-mlpdec | QPINO (D) with an MLP decoder | 366 | 0.0391 ± 0.0111 | 1.08e-03 ± 3.07e-04 | 0.0154 ± 0.0067 | 8.72e-03 ± 1.88e-03 | 3.27e-03 ± 9.26e-04 | 0.012 | 40 |
| E | QPINO without physics loss | 105 | 0.1003 ± 0.0167 | 2.76e-03 ± 4.61e-04 | 0.0435 ± 0.0111 | 2.04e-02 ± 2.71e-03 | 8.43e-03 ± 1.36e-03 | 1.015 | 202 |
| F | QPINO without ZZ couplings | 96 | 0.3089 ± 0.0194 | 8.50e-03 ± 5.33e-04 | 0.1336 ± 0.0300 | 5.85e-02 ± 3.57e-03 | 2.61e-02 ± 1.60e-03 | 5.810 | 36 |

Fitted (D, rho, beta) from training coefficients: 0.0980, 0.9985, 0.9981 (true 0.1, 1, 1).

### Generalisation (seed-0 models): time step and resolution

| model | rollout @ dt=0.05 | rollout @ dt=0.2 | 1-step incr. @ dt=0.05 | 1-step incr. @ dt=0.2 | rollout from N=8 grid | rollout from N=256 grid |
|---|---|---|---|---|---|---|
| Persistence | 0.6843 | 0.6879 | 1.0000 | 1.0000 | 0.6855 | 0.6855 |
| A | 0.0004 | 0.0004 | 0.0011 | 0.0011 | 0.0004 | 0.0004 |
| B | 0.0181 | 0.0204 | 0.0531 | 0.0562 | 0.0184 | 0.0184 |
| B-data | 0.0182 | 0.0205 | 0.0533 | 0.0564 | 0.0184 | 0.0184 |
| C | 0.0373 | 0.0401 | 0.0910 | 0.0972 | 0.0380 | 0.0380 |
| D | 0.1582 | 0.2124 | 0.4924 | 0.5644 | 0.0302 | 0.0302 |
| D-mlpdec | 0.1523 | 0.4133 | 0.7287 | 0.8119 | 0.0133 | 0.0133 |
| E | 0.1583 | 0.2121 | 0.4926 | 0.5644 | 0.0303 | 0.0303 |
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
| B-data | 165 | 0.0512 ± 0.0013 | 1.210 ± 0.030 | 13.76 ± 0.00 | 0.473 ± 0.084 | 22.74 ± 2.50 | 16.38 ± 2.10 | 23.34 ± 2.82 | 1.418 ± 0.128 |
| C | 105 | 0.0502 ± 0.0029 | 1.188 ± 0.069 | 13.75 ± 0.02 | 0.418 ± 0.046 | 21.45 ± 1.55 | 15.99 ± 1.73 | 23.38 ± 2.88 | 1.191 ± 0.131 |
| D | 105 | 0.0479 ± 0.0038 | 1.132 ± 0.090 | 13.73 ± 0.02 | 0.412 ± 0.060 | 21.33 ± 1.38 | 15.01 ± 1.75 | 23.35 ± 1.69 | 1.211 ± 0.064 |
| D-mlpdec | 366 | 0.0614 ± 0.0019 | 1.450 ± 0.046 | 13.82 ± 0.02 | 0.507 ± 0.116 | 24.29 ± 4.59 | 16.20 ± 1.90 | 28.58 ± 11.12 | 1.574 ± 0.174 |
| E | 105 | 0.0480 ± 0.0038 | 1.134 ± 0.090 | 13.73 ± 0.02 | 0.413 ± 0.062 | 21.35 ± 1.43 | 15.07 ± 1.82 | 23.30 ± 1.66 | 1.214 ± 0.064 |
| F | 96 | 0.0513 ± 0.0038 | 1.214 ± 0.091 | 13.76 ± 0.03 | 0.440 ± 0.047 | 21.79 ± 1.92 | 15.58 ± 1.09 | 24.35 ± 5.34 | 1.247 ± 0.090 |

Full-resolution (no M=2 truncation) field references:

| reference | 1-step field RMSE | rollout field RMSE |
|---|---|---|
| Persistence-fullfield | 3.31 | 18.93 |
| LinearTrend-fullfield | 3.59 | 31.77 |
| A-fullgrid (FD Fisher-KPP on 32 bins) | 3.70 | 16.95 |

Fitted Fisher-KPP (Galerkin one-step least squares on Mar 1 - Jun 16): D = 1.49e-03 (x in [0, 2pi]) = 528 km^2/day, rho = 0.0352 /day, beta = 0.1038 /day, K = rho/beta = 0.339 (u units) = 28.0 cases/100k/day. Scaled fit loss 0.991 (persistence = 1 by construction of the scaling, roughly).

Profile over fixed D: D=0.0001: loss 0.991, rho 0.035, beta 0.115; D=0.001: loss 0.991, rho 0.035, beta 0.108; D=0.01: loss 1.000, rho 0.033, beta 0.0376; D=0.03: loss 1.093, rho 0.044, beta 1.62e-08; D=0.1: loss 2.093, rho 0.087, beta 4.1e-09

## Ansatz ablations: synthetic

| variant | rel_l2_increment | rollout_rel_l2 | dt0.05_rollout_rel_l2 | dt0.2_rollout_rel_l2 | pde_residual_rms_trapezoid | n_params | train_time_s |
|---|---|---|---|---|---|---|---|
| graph=triad | 0.2716 ± 0.003176 | 0.08604 ± 0.0077 | 0.1475 ± 0.0365 | 0.2667 ± 0.03273 | 0.05164 ± 0.0006127 | 101 | 59.12 ± 2.432 |
| graph=full | 0.08175 ± 0.004873 | 0.0347 ± 0.005744 | 0.1831 ± 0.01442 | 0.2153 ± 0.02363 | 0.0172 ± 0.001111 | 106 | 314.6 ± 235.3 |
| graph=nearest | 0.1188 ± 0.06159 | 0.06114 ± 0.04458 | 0.3099 ± 0.1268 | 0.4846 ± 0.3008 | 0.02385 ± 0.01058 | 100 | 50.18 ± 0.3449 |
| graph=none (F) | 0.3014 ± 0.01987 | 0.1209 ± 0.02953 | 0.1377 ± 0.002238 | 0.343 ± 0.04672 | 0.05709 ± 0.003662 | 96 | 49.08 ± 0.9171 |
| HEA (C) | 0.08793 ± 0.004196 | 0.03473 ± 0.00325 | 0.03403 ± 0.003276 | 0.03684 ± 0.00322 | 0.0176 ± 0.0006296 | 105 | 58.73 ± 0.2837 |
| trotter L=1 | 0.07468 ± 0.00098 | 0.03089 ± 0.001167 | 0.1769 ± 0.003043 | 0.3815 ± 0.1872 | 0.01495 ± 7.456e-05 | 105 | 199.8 ± 149 |
| trotter L=2 | 0.06374 ± 0.006919 | 0.02746 ± 0.003509 | 0.2302 ± 0.05245 | 0.571 ± 0.3612 | 0.01344 ± 0.001595 | 105 | 50.3 ± 1.383 |
| trotter L=4 | 0.07265 ± 0.001414 | 0.03543 ± 0.002814 | 0.1672 ± 0.03981 | 0.4044 ± 0.06343 | 0.01541 ± 0.0002459 | 105 | 46.7 ± 1.384 |
| trotter L=8 | 0.08051 ± 0.01378 | 0.04067 ± 0.01463 | 0.2653 ± 0.06959 | 0.2553 ± 0.01785 | 0.01681 ± 0.002889 | 105 | 45.79 ± 0.3301 |
| graph=structured | 0.09 ± 0.01053 | 0.03642 ± 0.006177 | 0.1604 ± 0.002191 | 0.1681 ± 0.04436 | 0.01868 ± 0.001435 | 105 | 59.93 ± 4.459 |
| tau fixed c=2 | 0.1014 ± 0.01955 | 0.04175 ± 0.01152 | 0.1387 ± 0.008586 | 0.1714 ± 0.05367 | 0.02044 ± 0.003343 | 104 | 314.2 ± 235.4 |
| tau fixed c=10 | 0.08877 ± 0.0001622 | 0.03946 ± 0.001258 | 0.1566 ± 0.02456 | 0.2019 ± 0.05596 | 0.01862 ± 0.0002958 | 104 | 50.89 ± 0.2928 |
| tau fixed c=50 | 0.09227 ± 0.0001278 | 0.05473 ± 0.01732 | 0.278 ± 0.1782 | 0.3674 ± 0.1258 | 0.01945 ± 0.0001249 | 104 | 49.67 ± 0.9211 |
| tau learned c0=10 | 0.09 ± 0.01053 | 0.03642 ± 0.006177 | 0.1604 ± 0.002191 | 0.1681 ± 0.04436 | 0.01868 ± 0.001435 | 105 | 49.6 ± 0.08297 |
| reupload R=2 | 0.08425 ± 0.004935 | 0.03645 ± 0.01026 | 0.3017 ± 0.03612 | 0.433 ± 0.01967 | 0.01686 ± 0.000911 | 124 | 279.2 ± 152 |
| reupload R=3 | 0.1117 ± 0.0236 | 0.0518 ± 0.002479 | 0.228 ± 0.02815 | 0.47 ± 0.05838 | 0.02157 ± 0.004211 | 143 | 311.2 ± 121.9 |
| measure Z only | 0.1005 ± 0.01024 | 0.03329 ± 0.0001296 | 0.1561 ± 0.01466 | 0.1504 ± 0.05199 | 0.02216 ± 0.001468 | 55 | 60.07 ± 7.085 |
| decoder MLP | 0.03186 ± 0.003394 | 0.01069 ± 0.002209 | 0.2679 ± 0.1157 | 0.4364 ± 0.02207 | 0.00753 ± 0.0006913 | 366 | 315.6 ± 235 |
| frozen circuit + MLP decoder | 0.2655 ± 0.00417 | 0.1199 ± 0.0001113 | 0.1974 ± 0.08252 | 0.2173 ± 0.08429 | 0.05682 ± 0.003294 | 341 | 36.66 ± 0.1057 |
| frozen circuit + linear decoder | 0.2509 ± 0.008917 | 0.1112 ± 0.007755 | 0.1229 ± 0.001545 | 0.1583 ± 0.04613 | 0.05197 ± 0.001533 | 80 | 32.5 ± 0.09696 |
| alpha0=0.25 | 0.07725 ± 0.005614 | 0.02789 ± 0.002322 | 0.2446 ± 0.004819 | 0.3329 ± 0.0262 | 0.01627 ± 0.001251 | 105 | 49.57 ± 0.08524 |
| alpha0=0.5 | 0.08616 ± 0.02042 | 0.0351 ± 0.01124 | 0.264 ± 0.08415 | 0.5222 ± 0.1802 | 0.01787 ± 0.003745 | 105 | 199.5 ± 149.9 |
| alpha0=2 | 0.09007 ± 0.02324 | 0.04621 ± 0.01507 | 0.2396 ± 0.007934 | 0.2833 ± 0.03128 | 0.01809 ± 0.004088 | 105 | 53.89 ± 0.0708 |
| alpha fixed=1 | 0.184 ± 0.02657 | 0.1258 ± 0.05057 | 0.1138 ± 0.001291 | 0.269 ± 0.005092 | 0.03848 ± 0.003404 | 100 | 46.05 ± 0.9368 |

## Ansatz ablations: covid

| variant | rel_l2_increment | rollout_rel_l2 | field_rmse_per100k_onestep | field_rmse_per100k_rollout | n_params | train_time_s |
|---|---|---|---|---|---|---|
| graph=structured | 1.132 ± 0.08986 | 0.4124 ± 0.06015 | 13.73 ± 0.01877 | 21.33 ± 1.38 | 105 | 25.66 ± 0.2115 |
| graph=triad | 1.208 ± 0.05088 | 0.4604 ± 0.1409 | 13.75 ± 0.008902 | 22.71 ± 4.142 | 101 | 25.19 ± 0.8865 |
| graph=full | 1.124 ± 0.04471 | 0.457 ± 0.09151 | 13.73 ± 0.01256 | 22.59 ± 2.392 | 106 | 23.73 ± 0.1698 |
| graph=nearest | 1.205 ± 0.1014 | 0.3893 ± 0.05538 | 13.75 ± 0.02296 | 20.55 ± 1.03 | 100 | 22.82 ± 0.1784 |
| graph=none (F) | 1.214 ± 0.09053 | 0.4396 ± 0.04712 | 13.76 ± 0.02659 | 21.79 ± 1.918 | 96 | 22.87 ± 0.5037 |
| HEA (C) | 1.188 ± 0.06922 | 0.4184 ± 0.04576 | 13.75 ± 0.01912 | 21.45 ± 1.548 | 105 | 114.9 ± 113.9 |
| trotter L=1 | 1.192 ± 0.0755 | 0.4004 ± 0.1341 | 13.74 ± 0.02099 | 21.31 ± 3.476 | 105 | 18.29 ± 0.1549 |
| trotter L=4 | 1.276 ± 0.03004 | 0.4837 ± 0.0869 | 13.77 ± 0.01382 | 24.68 ± 3.507 | 105 | 19.17 ± 0.1685 |
| tau fixed c=2 | 1.33 ± 0.05905 | 0.3496 ± 0.03203 | 13.79 ± 0.01794 | 19.14 ± 0.3464 | 104 | 22.07 ± 0.2129 |
| tau fixed c=10 | 1.183 ± 0.009984 | 0.4249 ± 0.08485 | 13.75 ± 0.006322 | 21.77 ± 3.083 | 104 | 22.21 ± 0.1041 |
| tau learned c0=10 | 1.132 ± 0.08986 | 0.4124 ± 0.06015 | 13.73 ± 0.01877 | 21.33 ± 1.38 | 105 | 55.47 ± 32.29 |
| reupload R=2 | 1.121 ± 0.04516 | 0.4312 ± 0.03443 | 13.73 ± 0.009072 | 21.25 ± 0.8598 | 124 | 50.91 ± 0.5257 |
| reupload R=3 | 1.101 ± 0.05186 | 0.3294 ± 0.02563 | 13.73 ± 0.01452 | 18.7 ± 0.5799 | 143 | 77.27 ± 1.149 |
| measure Z only | 1.123 ± 0.04095 | 0.3773 ± 0.07664 | 13.73 ± 0.01103 | 19.84 ± 1.228 | 55 | 22.74 ± 0.01504 |
| decoder MLP | 1.45 ± 0.0457 | 0.5066 ± 0.1159 | 13.82 ± 0.02342 | 24.29 ± 4.591 | 366 | 23.39 ± 0.1957 |
| frozen circuit + MLP decoder | 1.467 ± 0.09427 | 0.3979 ± 0.07194 | 13.82 ± 0.02869 | 21.77 ± 2.586 | 341 | 13.56 ± 0.182 |
| alpha0=0.25 | 1.155 ± 0.06243 | 0.3488 ± 0.07055 | 13.74 ± 0.01281 | 19.58 ± 1.523 | 105 | 22.81 ± 0.3057 |
| alpha0=2 | 1.191 ± 0.0182 | 0.3913 ± 0.04316 | 13.75 ± 0.004753 | 19.8 ± 1.16 | 105 | 71.67 ± 69.07 |

### Post-hoc Trotterisation of the exact-trained structured QPINO (seed 0)

| L | max abs unitary error | RMS prediction deviation | 1-step incr. rel. L2 | rollout rel. L2 |
|---|---|---|---|---|
| 1 | 8.221e-01 | 6.922e-03 | 0.8391 | 0.4114 |
| 2 | 1.106e+00 | 3.912e-03 | 0.4749 | 0.3735 |
| 4 | 9.044e-01 | 6.387e-03 | 0.7697 | 0.3967 |
| 8 | 5.158e-01 | 3.783e-03 | 0.4614 | 0.2515 |
| 16 | 2.638e-01 | 1.931e-03 | 0.2462 | 0.1167 |
| 32 | 1.322e-01 | 9.609e-04 | 0.1414 | 0.0585 |
| 64 | 6.605e-02 | 4.774e-04 | 0.0991 | 0.0388 |

Learned tau scale: {'learned_c': 13.222803352091514, 'tau_train': 1.3222803352091514, 'H_norm': 10.219759382001664}

## Physics-loss ablations: synthetic_full

| variant | rel_l2_increment | rollout_rel_l2 | pde_residual_rms_trapezoid | train_time_s |
|---|---|---|---|---|
| lam_pde=0 (E) | 0.09018 ± 0.01055 | 0.03648 ± 0.0062 | 0.01871 ± 0.00144 | 58.93 ± 0.6545 |
| lam_pde=0.01 | 0.09017 ± 0.01055 | 0.03648 ± 0.006198 | 0.01871 ± 0.00144 | 315.3 ± 262.9 |
| lam_pde=0.1 (D default) | 0.09 ± 0.01053 | 0.03642 ± 0.006177 | 0.01868 ± 0.001435 | 49.64 ± 0.5215 |
| lam_pde=1 | 0.08835 ± 0.01031 | 0.0358 ± 0.006037 | 0.01839 ± 0.001389 | 49.92 ± 0.3312 |
| lam_pde=10 | 0.07876 ± 0.008619 | 0.0322 ± 0.004671 | 0.01674 ± 0.001048 | 178.4 ± 124 |
| lam_pde=1, no curriculum | 0.07155 ± 0.002838 | 0.02922 ± 0.002527 | 0.0157 ± 0.0006637 | 50.08 ± 0.9073 |
| PDE-only (lam_data=0, lam_roll=0) | 0.07767 ± 2.903e-05 | 0.02858 ± 0.0007463 | 0.02015 ± 0.0001138 | 18.32 ± 0.2978 |
| lam_roll=0 | 0.1051 ± 0.01003 | 0.0414 ± 0.003862 | 0.02137 ± 0.001714 | 22.87 ± 2.297 |
| lam_roll=2 | 0.0754 ± 0.003905 | 0.03059 ± 0.001344 | 0.01613 ± 0.001004 | 53.21 ± 3.687 |
| residual=galerkin (projected) | 0.08998 ± 0.01063 | 0.03636 ± 0.00627 | 0.01867 ± 0.001452 | 48.7 ± 0.7534 |
| residual=both | 0.09 ± 0.01057 | 0.0364 ± 0.006219 | 0.01868 ± 0.001443 | 49.9 ± 0.2324 |
| scheme=forward | 0.08999 ± 0.01051 | 0.03643 ± 0.00619 | 0.01867 ± 0.001431 | 194.4 ± 149.1 |
| scheme=midpoint | 0.08999 ± 0.01053 | 0.03641 ± 0.006189 | 0.01867 ± 0.001436 | 51.93 ± 0.2903 |
| no regularization | 0.09034 ± 0.01083 | 0.03664 ± 0.00628 | 0.01874 ± 0.001495 | 49.27 ± 0.3887 |

## Physics-loss ablations: synthetic_lowdata

| variant | rel_l2_increment | rollout_rel_l2 | pde_residual_rms_trapezoid | train_time_s |
|---|---|---|---|---|
| lam_pde=0 (E) | 0.4462 ± 0.03495 | 0.2816 ± 0.01645 | 0.09775 ± 0.0003955 | 16.92 ± 0.3894 |
| lam_pde=0.01 | 0.4462 ± 0.03496 | 0.2816 ± 0.01648 | 0.09775 ± 0.0003963 | 17.36 ± 0.129 |
| lam_pde=0.1 (D default) | 0.4461 ± 0.03492 | 0.2815 ± 0.01648 | 0.09774 ± 0.0004038 | 16.88 ± 0.4106 |
| lam_pde=1 | 0.4465 ± 0.03568 | 0.2819 ± 0.01524 | 0.09784 ± 0.0002643 | 42.35 ± 25.9 |
| lam_pde=10 | 0.4386 ± 0.03501 | 0.2697 ± 0.009729 | 0.09666 ± 0.0003612 | 17.85 ± 0.4505 |
| lam_pde=1, no curriculum | 0.3146 ± 0.04954 | 0.1127 ± 0.02649 | 0.06226 ± 0.009114 | 16.19 ± 0.2803 |
| PDE-only (lam_data=0, lam_roll=0) | 0.3434 ± 0.01841 | 0.1919 ± 0.03467 | 0.06913 ± 0.004805 | 5.532 ± 0.03286 |
| lam_roll=0 | 0.4515 ± 0.0005091 | 0.2876 ± 0.04923 | 0.09723 ± 0.006275 | 8.738 ± 0.1346 |
| lam_roll=2 | 0.3852 ± 0.1293 | 0.1944 ± 0.1089 | 0.07858 ± 0.02444 | 281.1 ± 262.4 |
| residual=galerkin (projected) | 0.4457 ± 0.03505 | 0.2804 ± 0.01628 | 0.09766 ± 0.0003742 | 17.02 ± 0.08028 |
| residual=both | 0.4459 ± 0.03498 | 0.281 ± 0.01637 | 0.0977 ± 0.0003896 | 17.74 ± 0.8111 |
| scheme=forward | 0.4475 ± 0.03614 | 0.2833 ± 0.01468 | 0.09798 ± 0.0001724 | 17.79 ± 0.1615 |
| scheme=midpoint | 0.4461 ± 0.03491 | 0.2815 ± 0.01646 | 0.09774 ± 0.0004051 | 16.95 ± 0.2114 |
| no regularization | 0.4447 ± 0.03317 | 0.2811 ± 0.01693 | 0.09751 ± 0.0006956 | 16.42 ± 0.1071 |
| MLP lam_pde=0 | 0.5592 ± 0.03318 | 0.2928 ± 0.09876 | 0.1105 ± 0.007622 | 4.033 ± 0.07276 |
| MLP lam_pde=1 | 0.5622 ± 0.03337 | 0.2949 ± 0.09957 | 0.1112 ± 0.007773 | 4.382 ± 0.284 |

## Physics-loss ablations: covid

| variant | rel_l2_increment | rollout_rel_l2 | field_rmse_per100k_onestep | field_rmse_per100k_rollout | train_time_s |
|---|---|---|---|---|---|
| lam_pde=0 (E) | 1.134 ± 0.09018 | 0.4132 ± 0.06186 | 13.73 ± 0.01889 | 21.35 ± 1.433 | 26.16 ± 0.07278 |
| lam_pde=0.01 | 1.134 ± 0.09015 | 0.4134 ± 0.06194 | 13.73 ± 0.01888 | 21.35 ± 1.434 | 26.04 ± 0.6609 |
| lam_pde=0.1 (D default) | 1.132 ± 0.08986 | 0.4124 ± 0.06015 | 13.73 ± 0.01877 | 21.33 ± 1.38 | 24.36 ± 0.1141 |
| lam_pde=1 | 1.11 ± 0.09504 | 0.3795 ± 0.02853 | 13.73 ± 0.01883 | 20.57 ± 0.7422 | 23.41 ± 0.2009 |
| lam_pde=10 | 1.071 ± 0.066 | 0.3731 ± 0.03826 | 13.72 ± 0.01475 | 20.38 ± 1.085 | 24.05 ± 1.015 |
| PDE-only (lam_data=0, lam_roll=0) | 0.9953 ± 0.008826 | 0.34 ± 0.03806 | 13.7 ± 0.001843 | 18.83 ± 0.8236 | 86.34 ± 112.4 |
| lam_roll=0 | 1.414 ± 0.1571 | 0.4093 ± 0.0727 | 13.82 ± 0.03819 | 20.71 ± 1.292 | 6.451 ± 0.1739 |
| lam_roll=2 | 1.138 ± 0.0549 | 0.4283 ± 0.06417 | 13.74 ± 0.01306 | 21.67 ± 1.749 | 23.8 ± 0.638 |
| residual=galerkin (projected) | 1.12 ± 0.09467 | 0.4007 ± 0.05129 | 13.73 ± 0.01896 | 21.06 ± 1.104 | 22.88 ± 0.03554 |
| scheme=forward | 1.132 ± 0.08986 | 0.4128 ± 0.06059 | 13.73 ± 0.01877 | 21.34 ± 1.39 | 22.62 ± 0.3196 |

## Resources

| circuit | depth | gates | 2-qubit gates | Hamiltonian params |
|---|---|---|---|---|
| structured_L1 | 9 | 24 | 9 | 19 |
| structured_L4 | 30 | 81 | 36 | 19 |
| triad_L1 | 7 | 20 | 5 | 15 |
| triad_L4 | 25 | 65 | 20 | 15 |
| full_L1 | 10 | 25 | 10 | 20 |
| full_L4 | 31 | 85 | 40 | 20 |
| nearest_L1 | 7 | 19 | 4 | 14 |
| nearest_L4 | 19 | 61 | 16 | 14 |
| none_L1 | 3 | 15 | 0 | 10 |
| none_L4 | 9 | 45 | 0 | 10 |
| HEA_2layers | 15 | 35 | 10 | 20 |

Runtime (s per forward+backward, batch 256) vs qubits:

| n | statevector_bytes | structured-like(full)_exact_s_per_fwdbwd_batch256 | structured-like(full)_trotter_s_per_fwdbwd_batch256 | nearest_exact_s_per_fwdbwd_batch256 | nearest_trotter_s_per_fwdbwd_batch256 |
|---|---|---|---|---|---|
| 3 | 128 | 0.00232 | 0.000895 | 0.000836 | 0.000744 |
| 4 | 256 | 0.00122 | 0.000958 | 0.000958 | 0.00086 |
| 5 | 512 | 0.00189 | 0.00153 | 0.00184 | 0.00126 |
| 6 | 1.02e+03 | 0.00634 | 0.00203 | 0.00651 | 0.00184 |
| 7 | 2.05e+03 | 0.0344 | 0.00487 | 0.0334 | 0.00585 |
| 8 | 4.1e+03 | 0.244 | 0.0238 | 0.209 | 0.0214 |
| 9 | 8.19e+03 | 1.67 | 0.146 | 1.55 | 0.132 |
| 10 | 1.64e+04 | 13.1 | 1.01 | 13.5 | 0.985 |
| 11 | 3.28e+04 | 8.45 | 7.68 |

Per-sample forward: {'pennylane_default_qubit': 0.006598126888275146, 'torch_statevector_batched': 1.2589693069458006e-05}

Circuit evaluations: {'synthetic_per_epoch_simulator': 7040, 'synthetic_total_simulator': 10560000, 'hardware_parameter_shift_per_epoch': 359040, 'hardware_shots_per_epoch_at_1e3_shots': 359040000, 'note': 'parameter-shift needs 2 evaluations per quantum parameter per sample (generalised shift rules needed for multi-eigenvalue generators such as exp(-i tau H_theta) when not decomposed into gates)'}

Shot sensitivity (synthetic): exact MSE 2.393e-06; 64 shots: MSE 1.478e-05, dev 3.42e-03; 256 shots: MSE 5.061e-06, dev 1.62e-03; 1024 shots: MSE 3.104e-06, dev 8.74e-04; 4096 shots: MSE 2.548e-06, dev 4.17e-04; 16384 shots: MSE 2.446e-06, dev 2.03e-04; 65536 shots: MSE 2.403e-06, dev 1.03e-04

Shot sensitivity (covid_july): exact MSE 4.425e-04; 64 shots: MSE 4.542e-04, dev 3.07e-03; 256 shots: MSE 4.442e-04, dev 1.56e-03; 1024 shots: MSE 4.425e-04, dev 7.90e-04; 4096 shots: MSE 4.420e-04, dev 3.96e-04; 16384 shots: MSE 4.426e-04, dev 1.95e-04; 65536 shots: MSE 4.425e-04, dev 9.47e-05

PennyLane 20000-shot check: max |<O>_shots - <O>_exact| = 0.0156 (1/sqrt(shots) = 0.0071).

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
