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
