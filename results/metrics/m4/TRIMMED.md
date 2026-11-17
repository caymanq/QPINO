# Trimmed cells (compute budget)
Trimmed on 2026-11-16 ~18:40 MST to keep the remaining grid under ~3 h on the 8-core box
(priority: COVID M4-mirror D/E/F/C/B at R=1 > re-uploading sweep > lambda sweep):
* synthetic M=4 periodic, seed 2, for D, E, F and D(lambda_pde=0.001) -> these use seeds 0,1 only
* synthetic M=4 periodic D R=3 seed 2 (killed at ~20% of training) -> R=3 synthetic uses seeds 0,1
