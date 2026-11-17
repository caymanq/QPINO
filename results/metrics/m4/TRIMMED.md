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
