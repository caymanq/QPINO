# 2D contiguous-USA QPINO — results tables (report §16)

Field: u = log1p(cumulative cases/100k)/log1p(4800) on the 256×128 grid (17,609 land cells), JHU CSSE. Train Mar 15–Jun 16 2020, validate Jun 17–30, test Jul 1–26 (one-step from the observed previous day; rollout from the observed Jun 30 state). RMSE in u over land cells against the full-resolution observed field. λ_PDE = 0.01, 3 seeds.

## T12. Truncation error of the 2D real cosine basis on the land mask

| modes | qubits | rel. L2 train | rel. L2 val | rel. L2 July | RMSE train | RMSE July | cond(G) |
|---|---|---|---|---|---|---|---|
| 3×3 | 9 | 0.372 | 0.213 | 0.165 | 0.1378 | 0.1157 | 64 |
| 4×4 | 16 | 0.338 | 0.196 | 0.153 | 0.1238 | 0.1073 | 668 |

## T13. July 2020 RMSE (u, land cells)

| modes | model | params | 1-step incr. rel. L2 | 1-step RMSE | rollout RMSE Jul 1–26 | Jul 12 | Jul 26 | rollout RMSE vs projection | s/epoch |
|---|---|---|---|---|---|---|---|---|---|
| 3×3 | truncation floor | – | 0 | 0.1157 | 0.1157 | 0.1188 | 0.1014 | 0 | – |
| 3×3 | Persistence | 0 | 1.000 | 0.1158 | 0.1318 | 0.1285 | 0.1443 | 0.0630 | – |
| 3×3 | LinearTrend | 0 | 0.566 | 0.1158 | 0.1164 | 0.1190 | 0.1039 | 0.0125 | – |
| 3×3 | A | 3 | 1.004 | 0.1158 | 0.1186 | 0.1208 | 0.1088 | 0.0258 | – |
| 3×3 | B | 485 ± 0 | 6.760 ± 2.492 | 0.1164 ± 0.0004 | 0.1188 ± 0.0010 | 0.1202 ± 0.0008 | 0.1112 ± 0.0031 | 0.0263 ± 0.0045 | 0.01 ± 0.00 |
| 3×3 | C | 459 ± 0 | 5.875 ± 1.079 | 0.1164 ± 0.0003 | 0.1376 ± 0.0109 | 0.1354 ± 0.0125 | 0.1544 ± 0.0114 | 0.0725 ± 0.0196 | 0.30 ± 0.01 |
| 3×3 | D | 478 ± 0 | 3.939 ± 0.459 | 0.1161 ± 0.0000 | 0.1355 ± 0.0076 | 0.1327 ± 0.0075 | 0.1542 ± 0.0168 | 0.0691 ± 0.0159 | 0.36 ± 0.01 |
| 3×3 | E | 478 ± 0 | 3.944 ± 0.454 | 0.1161 ± 0.0000 | 0.1344 ± 0.0079 | 0.1328 ± 0.0076 | 0.1459 ± 0.0174 | 0.0667 ± 0.0168 | 0.36 ± 0.01 |
| 3×3 | F | 442 ± 0 | 6.041 ± 1.269 | 0.1170 ± 0.0005 | 0.1563 ± 0.0249 | 0.1493 ± 0.0163 | 0.1808 ± 0.0426 | 0.1015 ± 0.0369 | 0.36 ± 0.01 |
| 4×4 | truncation floor | – | 0 | 0.1073 | 0.1073 | 0.1093 | 0.0952 | 0 | – |
| 4×4 | Persistence | 0 | 1.000 | 0.1073 | 0.1248 | 0.1200 | 0.1410 | 0.0638 | – |
| 4×4 | LinearTrend | 0 | 0.688 | 0.1073 | 0.1090 | 0.1098 | 0.1013 | 0.0192 | – |
| 4×4 | A | 3 | 0.722 | 0.1073 | 0.1146 | 0.1143 | 0.1149 | 0.0403 | – |
| 4×4 | B | 816 ± 0 | 6.587 ± 0.673 | 0.1078 ± 0.0001 | 0.1221 ± 0.0066 | 0.1172 ± 0.0050 | 0.1368 ± 0.0113 | 0.0571 ± 0.0140 | 0.00 ± 0.00 |

Reference without truncation: persistence of the full-resolution Jun 30 field gives rollout RMSE 0.0762 (Jul 12 0.0615, Jul 26 0.1195) and one-step RMSE 0.0098. Every Galerkin model is bounded below by the truncation floor.
* fitted A (3×3): D = 1.06e-13 (non-dim; 8.84e-08 km²/day), ρ = 0.0315/day, β = 0.0356, K = ρ/β = 0.884, fit loss 1.024
* fitted A (4×4): D = 1.36e-11 (non-dim; 1.14e-05 km²/day), ρ = 0.0289/day, β = 0.0284, K = ρ/β = 1.015, fit loss 1.016

## 16-qubit (4×4) runtime benchmark (statevector, no dense operators, 1 thread)

| Trotter L | batch | fwd+bwd (s) | est. s/epoch (93 one-step + 609 rollout evaluations) | est. run (1500 epochs) |
|---|---|---|---|---|
| 4 | 16 | 4.93 | 216 | 90 h |
| 1 | 32 | 3.82 | 84 | 35 h |

## T14. Spatial growth rate ρ(x,y) (report §17): July 2020 RMSE and physics-violation diagnostic

λ_PDE = 1e-4 for all trained rows except the §16 rows (λ = 0.01). ρ(x,y) = Σ r_m ψ_m with a 4×4 cosine basis (16 coefficients). PDE violation = RMS (u/day) of the trapezoid full-PDE residual on 2000 land points; 'own' uses the model's reference physics (scalar fit for scalar rows, spatial fit for ρ(x,y) rows); rollout column gives scalar-ref / spatial-ref. Observed July data: 0.0022 / 0.0024 (3×3 projection, scalar / spatial ref).

| modes | model | params | 1-step RMSE | rollout RMSE | Jul 12 | Jul 26 | rollout vs projection | PDE viol. val 1-step (own) | PDE viol. test rollout (scalar / spatial ref) |
|---|---|---|---|---|---|---|---|---|---|
| 3×3 | truncation floor | – | 0.1157 | 0.1157 | 0.1188 | 0.1014 | 0 | – | 0.0022 / 0.0024 |
| 3×3 | Persistence | 0 | 0.1158 | 0.1318 | 0.1285 | 0.1443 | 0.0630 | – | 0.0056 / 0.0032 |
| 3×3 | Linear trend | 0 | 0.1158 | 0.1164 | 0.1190 | 0.1039 | 0.0125 | – | 0.0025 / 0.0021 |
| 3×3 | A, scalar ρ (§16 one-step fit) | 3 | 0.1158 | 0.1186 | 0.1208 | 0.1088 | 0.0258 | 0.0001 | 0.0000 / 0.0029 |
| 3×3 | A, scalar ρ (rollout protocol) | 3 | 0.1158 | 0.1189 | 0.1206 | 0.1101 | 0.0272 | 0.0029 | 0.0025 / 0.0018 |
| 3×3 | **A-ρ(x,y)** (rollout protocol) | 18 | 0.1158 | 0.1184 | 0.1203 | 0.1088 | 0.0251 | 0.0013 | 0.0029 / 0.0014 |
| 3×3 | D (9-qubit QPINO), scalar ρ, λ=0.01 (§16) | 478 ± 0 | 0.1161 ± 0.0000 | 0.1355 ± 0.0076 | 0.1327 ± 0.0075 | 0.1542 ± 0.0168 | 0.0691 ± 0.0159 | – | – |
| 3×3 | D (9-qubit QPINO), scalar ρ, λ=1e-4 | 478 ± 0 | 0.1161 ± 0.0000 | 0.1348 ± 0.0081 | 0.1324 ± 0.0076 | 0.1474 ± 0.0172 | 0.0673 ± 0.0173 | 0.0090 ± 0.0010 | 0.0089 ± 0.0015 / 0.0084 ± 0.0019 |
| 3×3 | **D (9-qubit QPINO), ρ(x,y), λ=1e-4** | 478 ± 0 | 0.1161 ± 0.0000 | 0.1330 ± 0.0074 | 0.1318 ± 0.0077 | 0.1421 ± 0.0153 | 0.0639 ± 0.0158 | 0.0093 ± 0.0005 | 0.0087 ± 0.0013 / 0.0082 ± 0.0017 |
| 3×3 | C (HEA QPINO), scalar ρ, λ=0.01 (§16) | 459 ± 0 | 0.1164 ± 0.0003 | 0.1376 ± 0.0109 | 0.1354 ± 0.0125 | 0.1544 ± 0.0114 | 0.0725 ± 0.0196 | – | – |
| 3×3 | C (HEA QPINO), scalar ρ, λ=1e-4 | 459 ± 0 | 0.1164 ± 0.0003 | 0.1373 ± 0.0097 | 0.1354 ± 0.0126 | 0.1553 ± 0.0112 | 0.0723 ± 0.0176 | 0.0057 ± 0.0007 | 0.0086 ± 0.0012 / 0.0084 ± 0.0010 |
| 3×3 | **C (HEA QPINO), ρ(x,y), λ=1e-4** | 459 ± 0 | 0.1164 ± 0.0003 | 0.1372 ± 0.0097 | 0.1354 ± 0.0126 | 0.1552 ± 0.0111 | 0.0722 ± 0.0177 | 0.0057 ± 0.0005 | 0.0086 ± 0.0012 / 0.0084 ± 0.0010 |
| 3×3 | B (MLP), scalar ρ, λ=0.01 (§16) | 485 ± 0 | 0.1164 ± 0.0004 | 0.1188 ± 0.0010 | 0.1202 ± 0.0008 | 0.1112 ± 0.0031 | 0.0263 ± 0.0045 | – | – |
| 3×3 | B (MLP), scalar ρ, λ=1e-4 | 485 ± 0 | 0.1164 ± 0.0004 | 0.1190 ± 0.0010 | 0.1205 ± 0.0008 | 0.1110 ± 0.0031 | 0.0274 ± 0.0048 | 0.0054 ± 0.0026 | 0.0044 ± 0.0012 / 0.0040 ± 0.0013 |
| 3×3 | **B (MLP), ρ(x,y), λ=1e-4** | 485 ± 0 | 0.1164 ± 0.0003 | 0.1184 ± 0.0011 | 0.1202 ± 0.0009 | 0.1094 ± 0.0023 | 0.0246 ± 0.0051 | 0.0061 ± 0.0033 | 0.0038 ± 0.0003 / 0.0035 ± 0.0008 |
| 8×8 (classical) | truncation floor | – | 0.0911 | 0.0911 | 0.0917 | 0.0809 | 0 | – | – |
| 8×8 (classical) | classical A, scalar ρ (one-step fit) | 3 | 0.0913 | 0.1162 | 0.1092 | 0.1414 | 0.0721 | 0.0002 | 0.0002 / 0.0307 |
| 8×8 (classical) | classical A, scalar ρ (rollout protocol) | 3 | 0.0912 | 0.0982 | 0.0957 | 0.0993 | 0.0365 | 0.0065 | 0.0060 / 0.0258 |
| 8×8 (classical) | classical A-ρ(x,y) (rollout protocol) | 18 | 0.0951 | 0.5861 | 0.3452 | 1.2131 | 0.5790 | 0.0011 | 0.1080 / 0.0039 |

* ρ(x,y) fit (3×3): window 28 d, λ_L2 = 100, D = 1e-08, β = 0.0079, ρ range on land [0.0002, 0.0179]/day (mean 0.0093), val rollout rel. L2 0.0263; scalar same protocol: window 56 d, ρ = 0.0143, β = 0.0153, val rollout rel. L2 0.0462.
* ρ(x,y) fit (8×8): window 28 d, λ_L2 = 10, D = 3.58e-12, β = 0.0000, ρ range on land [-0.1242, 0.1060]/day (mean -0.0083), val rollout rel. L2 0.1052; scalar same protocol: window 28 d, ρ = 0.0042, β = 0.0000, val rollout rel. L2 0.1687.
* local log-growth initialiser range (3×3 projection, last 28 training days): [-0.241, 0.712]/day.
