# Trimmed cells (compute budget)
Trimmed on 2026-11-16 ~18:40 MST to keep the remaining grid under ~3 h on the 8-core box
(priority: COVID M4-mirror D/E/F/C/B at R=1 > re-uploading sweep > lambda sweep):
* synthetic M=4 periodic, seed 2, for D, E, F and D(lambda_pde=0.001) -> these use seeds 0,1 only
* synthetic M=4 periodic D R=3 seed 2 (killed at ~20% of training) -> R=3 synthetic uses seeds 0,1

# Superseded runs (bug fix, 2026-11-16 ~19:30 MST)
All COVID *mirror* trained runs finished before 19:30 (M=4 and M=2, 29 JSONs) were moved to
`superseded_mirror_bug/` and rerun. Cause: the zero-variance Im a_k components were meant to get
normal-scale normalisation, but the detection threshold (s_inc < 1e-9) never fired, so those
inputs were standardised with sigma = 1e-8 and their loss scale was ~100x too small. Tiny spurious
Im outputs were then amplified into large rotation angles, training data loss rose 10-100x and
rollouts were unstable. Synthetic and periodic runs have no zero-variance components and are unaffected.

# Cosine-basis grid trims (2026-11-16 ~22:26 MST)
To wrap up inside the budget, the lowest-priority cells (synthetic extras) were trimmed:
* synthetic cosine M=8, model E seed 2 and model F seed 2 were not run -> E and F synthetic use seeds 0, 1.
All other cosine cells (COVID M=8 D/E/F/C/B x 3 seeds, D R=2/R=3 x 3 seeds, COVID cosine M=4 D/C x 3,
synthetic cosine M=8 D/C/B x 3 and cosine M=4 D/C x 3) ran in full.
