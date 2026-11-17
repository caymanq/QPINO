"""Truncation error of the I-10 field for M = 2 / 4 (and 8), periodic vs mirrored boundary.

Periodic: x_j = 2 pi (j+1/2)/32 on the transect itself (LA identified with Jacksonville).
Mirror:   even reflection [u, reverse(u)] on 64 points (domain 2L) -> cosine series,
          no wrap jump.  Errors are measured on the original 32 bins only.
Writes results/metrics/m4_truncation.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, galerkin as gk, preprocessing as pp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = config.load()["covid"]
Z = np.load(os.path.join(ROOT, "data", "processed", "covid_i10_field.npz"))
u, x = Z["u"], Z["x"]
dates = pd.to_datetime(Z["dates"])
tr, va, te = pp.split_indices(dates, C["train_end"], C["val_end"], C["test_end"])
N = u.shape[1]
um, xm = pp.mirror_extend(u), gk.uniform_grid(2 * N, 0.5)


def rel(Mk, mirror, idx):
    if mirror:
        a = gk.field_to_coeffs(um[idx], xm, Mk); rec = gk.reconstruct(a, xm)[:, :N]
    else:
        a = gk.field_to_coeffs(u[idx], x, Mk); rec = gk.reconstruct(a, x)
    f = u[idx]
    return float(np.nanmean(np.linalg.norm(f - rec, axis=-1) / (np.linalg.norm(f, axis=-1) + 1e-12)))


splits = {"train": tr[7:], "val": va, "july": te}   # first 7 train days are the smoothing burn-in
out = {"note": "mean relative L2 error of the |k|<=M projection, measured on the 32 original bins",
       "wrap_jump_rel": {k: float(np.nanmean(pp.wrap_jump(u[v]))) for k, v in splits.items()},
       "wrap_jump_rel_mirror": float(np.abs(um[:, -1] - um[:, 0]).max())}
for Mk in (1, 2, 4, 8):
    for mir in (False, True):
        out[f"M{Mk}_{'mirror' if mir else 'periodic'}"] = {k: rel(Mk, mir, v) for k, v in splits.items()}
# cosine / Neumann basis on the 32 bins (x_j = pi (j+1/2)/32): M+1 real coefficients
from src import cosine_galerkin as cg  # noqa: E402
xc = cg.grid(N)
for Mk in (2, 4, 8, 16):
    out[f"cosine_M{Mk}"] = {}
    for k, v in splits.items():
        f = u[v]; rec = cg.reconstruct(cg.field_to_coeffs(f, xc, Mk), xc)
        out[f"cosine_M{Mk}"][k] = float(np.nanmean(np.linalg.norm(f - rec, axis=-1) / (np.linalg.norm(f, axis=-1) + 1e-12)))
out["cosine_qubits"] = {f"M{Mk}": Mk + 1 for Mk in (2, 4, 8, 16)}
# effective resolved wavelengths
L = 3735.0
out["wavelengths_km"] = {"periodic_M2": [L, L / 2], "periodic_M4": [L / k for k in range(1, 5)],
                         "mirror_M2": [2 * L, L], "mirror_M4": [2 * L / k for k in range(1, 5)],
                         "cosine_M8": [2 * L / k for k in range(1, 9)]}
out["dof"] = {"periodic_M2": 5, "periodic_M4": 9, "mirror_M2": "3 nonzero of 5 (Im = 0)",
              "mirror_M4": "5 nonzero of 9 (Im = 0)", "mirror_M8": "9 nonzero of 17"}
im = np.abs(gk.field_to_coeffs(um, xm, 4)[:, 2::2]).max()
out["mirror_max_abs_Im_coeff"] = float(im)
with open(os.path.join(ROOT, "results", "metrics", "m4_truncation.json"), "w") as f:
    json.dump(out, f, indent=2)
for k, v in out.items():
    print(k, v)
