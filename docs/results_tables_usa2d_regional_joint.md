# Joint regional training (report §19)

## T16. Joint regional training: flux inside the training rollouts and a 7→14→21-day curriculum (report §19)

Rollout from the observed Jun 30 state (coupled rollout for every row), RMSE in u vs the observed full-resolution field; 'vs floor' = against the regional 45-mode projection of the observed field (the 0.0949 floor itself is the projection's error vs observed). PDE violation = pointwise regional Fisher–KPP residual (u/day, interface flux excluded). λ_PDE = 1e-4. 'selected' = training horizon and epoch of the validation-selected checkpoint per seed.

| model | seeds | 1-step RMSE | rollout RMSE | Jul 12 | Jul 26 | rollout vs floor | vs floor, Jul 26 | PDE viol. rollout | PDE viol. train 1-step | selected (H@epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| regional truncation floor | 1 | 0.0949 | 0.0949 | 0.0961 | 0.0840 | 0.0000 | 0.0000 | 0.0027 | – | – |
| regional persistence | 1 | 0.0950 | 0.1149 | 0.1086 | 0.1350 | 0.0649 | 0.1057 | 0.0036 | – | – |
| regional linear trend | 1 | 0.0949 | 0.0972 | 0.0971 | 0.0919 | 0.0211 | 0.0373 | 0.0020 | – | – |
| regional Galerkin A (coupled) | 1 | 0.0949 | 0.0997 | 0.0986 | 0.0979 | 0.0307 | 0.0503 | 0.0002 | – | – |
| §18 regional D (independent training, flux only in rollout) | 3 | 0.0960 ± 0.0003 | 0.1348 ± 0.0139 | 0.1279 ± 0.0171 | 0.1705 ± 0.0151 | 0.0949 ± 0.0192 | 0.1481 ± 0.0173 | 0.0111 ± 0.0014 | – | – |
| §18 regional B (independent training, flux only in rollout) | 3 | 0.0961 ± 0.0003 | 0.1311 ± 0.0251 | 0.1191 ± 0.0170 | 0.1698 ± 0.0533 | 0.0870 ± 0.0354 | 0.1447 ± 0.0606 | 0.0080 ± 0.0016 | – | – |
| **§19 HEA C: flux in training, 7→14→21 d** | 1 | 0.0951 | 0.1146 | 0.1068 | 0.1316 | 0.0643 | 0.1013 | 0.0046 | 0.0078 | 7d@150 |
| **§19 MLP B: flux in training, 7→14→21 d** | 3 | 0.0952 ± 0.0001 | 0.1058 ± 0.0019 | 0.1030 ± 0.0015 | 0.1106 ± 0.0033 | 0.0467 ± 0.0044 | 0.0719 ± 0.0051 | 0.0031 ± 0.0001 | 0.0078 ± 0.0001 | 7d@150, 7d@225, 7d@275 |
| §19 MLP B: no flux in training, 7→14→21 d | 3 | 0.0953 ± 0.0001 | 0.1075 ± 0.0026 | 0.1033 ± 0.0012 | 0.1163 ± 0.0067 | 0.0503 ± 0.0056 | 0.0801 ± 0.0097 | 0.0037 ± 0.0004 | 0.0079 ± 0.0001 | 7d@165, 14d@315, 14d@340 |
| §19 MLP B: flux in training, 7 d only | 3 | 0.0952 ± 0.0001 | 0.1057 ± 0.0018 | 0.1028 ± 0.0014 | 0.1110 ± 0.0034 | 0.0465 ± 0.0043 | 0.0726 ± 0.0053 | 0.0031 ± 0.0001 | 0.0078 ± 0.0001 | 7d@150, 7d@225, 7d@355 |
| §19 MLP B: no flux in training, 7 d only | 3 | 0.0953 ± 0.0001 | 0.1093 ± 0.0037 | 0.1035 ± 0.0015 | 0.1233 ± 0.0112 | 0.0538 ± 0.0078 | 0.0896 ± 0.0155 | 0.0040 ± 0.0006 | 0.0080 ± 0.0002 | 7d@165, 7d@260, 7d@450 |

Step-weighting pilot (seed 0, flux in training, 7→14→21 d; w_s ∝ s^-α), selected by the validation rollout:

| model | α | val rollout RMSE (Jun 17–30) | July rollout RMSE |
|---|---|---|---|
| HEA C | 0 | 0.1143 | 0.1146 |
| HEA C | 1 | 0.1116 | 0.1121 |
| MLP B | 0 | 0.1105 | 0.1031 |
| MLP B | 1 | 0.1115 | 0.1102 |

* training wall time, MLP B long: 24 s per run (5 regions jointly, 1 thread; n=7)
* training wall time, MLP B short: 16 s per run (5 regions jointly, 1 thread; n=6)
* training wall time, HEA C long: 1074 s per run (5 regions jointly, 1 thread; n=2)
