**Reading T10 (synthetic Neumann Fisher–KPP, cosine M=8).** With known physics, the classical Galerkin ODE A is exact up to integration error (rollout 2.3e-8). Among learned models with matched parameter counts, the classical MLP B is best: rollout 0.0194 ± 0.0021, one-step 0.037. QPINO D reaches 0.067 ± 0.013 (one-step 0.100 ± 0.021), the HEA C 0.058 ± 0.024, and D without physics loss (E, 2 seeds) 0.066 ± 0.023. Removing the $ZZ$ couplings (F, 2 seeds) degrades the rollout to 0.112 ± 0.031 and the one-step error to 0.171 ± 0.013. The **couplings matter**, as in the earlier variant, while λ_PDE = 0.01 has no resolvable effect (D vs E). At 5 qubits (cosine M=4), D reaches 0.0303 ± 0.0010 and C 0.0346 ± 0.0109, against 0.0434 ± 0.0110 and 0.0324 ± 0.0043 for the earlier M=2 periodic runs. Those used a different (periodic) synthetic dataset and λ_PDE = 0.1, so this comparison is indicative only. Training cost per synthetic run, 1 thread: ~1400 s at 9 qubits (D, C, E, F; 0.92–0.95 s/epoch), 37–50 s at 5 qubits, 13–16 s for the MLP.

## 15.6 Figure

Fig. 17 (`results/figures/fig17_cosine_basis.png|pdf`) has six panels:
* (a) truncation error against qubit budget for the cosine, periodic-Fourier and mirrored-Fourier bases;
* (b) COVID 31-day rollout RMSE by day for cosine M=8, all models and references;
* (c) July rollout RMSE of every model in cosine M=8 and M=4 versus the earlier complex-Fourier configurations;
* (d)–(f) the observed July field, a QPINO D rollout forecast and the Galerkin ODE A forecast.

## 15.7 Caveats specific to the cosine runs

* **Trimmed:** synthetic cosine M=8 E and F seed 2 (`results/metrics/m4/TRIMMED.md`). E and F synthetic use 2 seeds. Everything else ran with 3 seeds.
* **Seed sensitivity:** on COVID, the cosine M=8 learned models have seed spreads of ±1.7–4.4 RMSE units. With 3 seeds, no ranking among B–F is resolved.
* **The coupling graph is complete** in the cosine basis, so the "physics-derived graph" carries no structural information here. Only the F (no-ZZ) ablation tests the couplings.
* **Basis comparison:** cosine M=4 is numerically identical to the M=4 even-mirror projection. The gain from real inputs is qubit efficiency (5 instead of 9 qubits for the same field), not a better field representation.
* **D is unidentifiable** on the COVID field: the fitted D moves from 2e-3 (M=4) to 7e-10 (M=8).
* The `max_abs_Im_rollout` field in the cosine JSONs is meaningless (no Im components) and is not reported.
* All quantum circuits are classically simulated (exact statevector). **No quantum advantage** is shown or claimed.
