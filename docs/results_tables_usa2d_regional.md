# Regional domain-decomposition QPINO (report §18)

## T15. Regional domain decomposition (5 regions × 3×3 modes = 45 real coefficients; 5 independent 9-qubit circuits)

Regions (cells): West (5782), Southwest/Texas (3065), Midwest (4661), Southeast (2983), Northeast (1118); 381 interface cell faces. cond(G) per region: 29, 898, 30, 106, 234. Regional truncation floor: rel. L2 train 0.300 / val 0.172 / July 0.135, RMSE July 0.0949 (single 3×3 basis: RMSE July 0.1157).

Regional Galerkin ODE A fit (rollout protocol, H=7, window 56 d chosen on validation): D = 8.8e-06 (7.37 km²/day), β = 0.0060, ρ_r = West 0.0083, Southwest/Texas 0.0127, Midwest 0.0097, Southeast 0.0080, Northeast 0.0076 /day. July interface-flux / local-RHS norm ratio (observed coefficients): 0.141.

Rollout from the observed Jun 30 state, RMSE in u vs the full-resolution observed field (land cells); 'vs projection' is against the model's own projection (regional 45-mode projection for regional rows, single 3×3 projection for single-region rows). PDE violation: regional rows use the regional physics (D, ρ_r, β; pointwise PDE, 400 points per region); single-region rows use the §17 scalar reference — the two are not on the same scale of physics, compare within a block. Trained rows: 3 seeds, λ_PDE = 1e-4.

| model | params | 1-step RMSE | rollout RMSE | Jul 12 | Jul 26 | rollout vs projection | PDE viol. rollout | West | Southwest/Texas | Midwest | Southeast | Northeast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **regional** truncation floor | – | 0.0949 | 0.0949 | 0.0961 | 0.0840 | 0.0000 | 0.0027 | 0.1188 | 0.0830 | 0.0965 | 0.0556 | 0.0533 |
| regional persistence | 0 | 0.0950 | 0.1149 | 0.1086 | 0.1350 | 0.0649 | 0.0036 | 0.1402 | 0.1125 | 0.1112 | 0.0795 | 0.0580 |
| regional linear trend | 0 | 0.0949 | 0.0972 | 0.0971 | 0.0919 | 0.0211 | 0.0020 | 0.1210 | 0.0876 | 0.0987 | 0.0565 | 0.0538 |
| regional Galerkin A, coupled (flux) | 7 | 0.0949 | 0.0997 | 0.0986 | 0.0979 | 0.0307 | 0.0002 | 0.1255 | 0.0905 | 0.0982 | 0.0591 | 0.0570 |
| regional Galerkin A, decoupled | 7 | 0.0949 | 0.0999 | 0.0987 | 0.0984 | 0.0313 | 0.0000 | 0.1256 | 0.0912 | 0.0982 | 0.0589 | 0.0571 |
| regional QPINO D (5×9 qubits), coupled | 2390 | 0.0960 ± 0.0003 | 0.1348 ± 0.0139 | 0.1279 ± 0.0171 | 0.1705 ± 0.0151 | 0.0949 ± 0.0192 | 0.0111 ± 0.0014 | 0.1533 ± 0.0029 | 0.1360 ± 0.0195 | 0.1159 ± 0.0117 | 0.1121 ± 0.0335 | 0.1383 ± 0.0616 |
| regional QPINO D (5×9 qubits), decoupled | 2390 | 0.0960 ± 0.0003 | 0.1369 ± 0.0153 | 0.1278 ± 0.0177 | 0.1713 ± 0.0165 | 0.0977 ± 0.0208 | 0.0113 ± 0.0013 | 0.1555 ± 0.0068 | 0.1363 ± 0.0202 | 0.1164 ± 0.0104 | 0.1087 ± 0.0380 | 0.1575 ± 0.0679 |
| regional MLP B, coupled | 2425 | 0.0961 ± 0.0003 | 0.1311 ± 0.0251 | 0.1191 ± 0.0170 | 0.1698 ± 0.0533 | 0.0870 ± 0.0354 | 0.0080 ± 0.0016 | 0.1367 ± 0.0025 | 0.1500 ± 0.0312 | 0.1249 ± 0.0294 | 0.1148 ± 0.0677 | 0.0606 ± 0.0006 |
| regional MLP B, decoupled | 2425 | 0.0961 ± 0.0003 | 0.1308 ± 0.0245 | 0.1192 ± 0.0168 | 0.1684 ± 0.0511 | 0.0867 ± 0.0347 | 0.0079 ± 0.0016 | 0.1368 ± 0.0027 | 0.1516 ± 0.0335 | 0.1228 ± 0.0261 | 0.1142 ± 0.0663 | 0.0619 ± 0.0016 |
| *single 3×3 basis (§16/§17)* | | | | | | | |  |  |  |  |  |
| single-region truncation floor | – | 0.1157 | 0.1157 | 0.1188 | 0.1014 | 0.0000 | – | 0.1385 | 0.1036 | 0.1133 | 0.0803 | 0.1082 |
| single-region persistence | 0 | 0.1158 | 0.1318 | 0.1285 | 0.1443 | 0.0630 | – | 0.1520 | 0.1210 | 0.1298 | 0.1069 | 0.1143 |
| single-region linear trend | 0 | 0.1158 | 0.1164 | 0.1190 | 0.1039 | 0.0125 | – | – | – | – | – | – |
| single-region Galerkin A (rollout protocol) | 3 | 0.1158 | 0.1189 | 0.1206 | 0.1101 | 0.0272 | – | 0.1396 | 0.1071 | 0.1146 | 0.0917 | 0.1153 |
| single-region QPINO D (9 qubits), λ=1e-4 | 478 | 0.1161 ± 0.0000 | 0.1348 ± 0.0081 | 0.1324 ± 0.0076 | 0.1474 ± 0.0172 | 0.0673 ± 0.0173 | – | 0.1576 ± 0.0120 | 0.1141 ± 0.0030 | 0.1265 ± 0.0038 | 0.1021 ± 0.0037 | 0.1626 ± 0.0328 |
| single-region MLP B, λ=1e-4 | 485 | 0.1164 ± 0.0004 | 0.1190 ± 0.0010 | 0.1205 ± 0.0008 | 0.1110 ± 0.0031 | 0.0274 ± 0.0048 | – | 0.1410 ± 0.0032 | 0.1061 ± 0.0017 | 0.1149 ± 0.0012 | 0.0862 ± 0.0063 | 0.1197 ± 0.0094 |

* regional D: training wall time per region 495 s (range 478–539 s, 1 thread, regions run in parallel).
* regional B: training wall time per region 9 s (range 9–10 s, 1 thread, regions run in parallel).
