"""T12-T13: 2D contiguous-USA tables -> docs/results_tables_usa2d.md."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics", "usa2d")
runs = defaultdict(list)
for f in glob.glob(os.path.join(MET, "usa2d_K*_s[0-9].json")):
    d = json.load(open(f)); runs[(d["meta"]["K"], d["meta"]["model"])].append(d["result"])
base = {K: json.load(open(os.path.join(MET, f"baselines_usa2d_K{K}.json"))) for K in (3, 4)}


def ms(K, nm, k, fmt="{:.4f}"):
    rows = runs.get((K, nm), [])
    if not rows:
        return "n/a"
    a = np.array([r[k] for r in rows], float)
    return (fmt + " ± " + fmt).format(a.mean(), a.std()) + ("" if len(a) == 3 else f" (n={len(a)})")


L = ["# 2D contiguous-USA QPINO — results tables (report §16)", "",
     "Field: u = log1p(cumulative cases/100k)/log1p(4800) on the 256×128 grid (17,609 land cells), JHU CSSE. "
     "Train Mar 15–Jun 16 2020, validate Jun 17–30, test Jul 1–26 (one-step from the observed previous day; rollout from the observed Jun 30 state). "
     "RMSE in u over land cells against the full-resolution observed field. λ_PDE = 0.01, 3 seeds.", "",
     "## T12. Truncation error of the 2D real cosine basis on the land mask", "",
     "| modes | qubits | rel. L2 train | rel. L2 val | rel. L2 July | RMSE train | RMSE July | cond(G) |", "|---|---|---|---|---|---|---|---|"]
for K in (3, 4):
    t, m = base[K]["results"]["truncation"], base[K]["meta"]
    L.append(f"| {K}×{K} | {K*K} | {t['train']:.3f} | {t['val']:.3f} | {t['july']:.3f} | {t['rmse_train']:.4f} | {t['rmse_july']:.4f} | {m['cond_G']:.0f} |")
L += ["", "## T13. July 2020 RMSE (u, land cells)", "",
      "| modes | model | params | 1-step incr. rel. L2 | 1-step RMSE | rollout RMSE Jul 1–26 | Jul 12 | Jul 26 | rollout RMSE vs projection | s/epoch |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for K in (3, 4):
    b = base[K]["results"]
    L.append(f"| {K}×{K} | truncation floor | – | 0 | {b['Projection-floor']['rmse_onestep']:.4f} | {b['Projection-floor']['rmse_rollout']:.4f} | "
             f"{b['Projection-floor']['rmse_jul12']:.4f} | {b['Projection-floor']['rmse_jul26']:.4f} | 0 | – |")
    for nm, P in (("Persistence", 0), ("LinearTrend", 0), ("A", 3)):
        r = b[nm]
        L.append(f"| {K}×{K} | {nm} | {P} | {r['rel_l2_increment']:.3f} | {r['rmse_onestep']:.4f} | {r['rmse_rollout']:.4f} | "
                 f"{r['rmse_jul12']:.4f} | {r['rmse_jul26']:.4f} | {r['rmse_rollout_vs_projection']:.4f} | – |")
    for nm in ("B", "C", "D", "E", "F"):
        if (K, nm) in runs:
            L.append(f"| {K}×{K} | {nm} | {ms(K, nm, 'n_params', '{:.0f}')} | {ms(K, nm, 'rel_l2_increment', '{:.3f}')} | {ms(K, nm, 'rmse_onestep')} | "
                     f"{ms(K, nm, 'rmse_rollout')} | {ms(K, nm, 'rmse_jul12')} | {ms(K, nm, 'rmse_jul26')} | "
                     f"{ms(K, nm, 'rmse_rollout_vs_projection')} | {ms(K, nm, 'sec_per_epoch', '{:.2f}')} |")
pf = base[3]["results"]["Persistence-fullres"]
L += ["", f"Reference without truncation: persistence of the full-resolution Jun 30 field gives rollout RMSE {pf['rmse_rollout']:.4f} "
      f"(Jul 12 {pf['rmse_jul12']:.4f}, Jul 26 {pf['rmse_jul26']:.4f}) and one-step RMSE {pf['rmse_onestep']:.4f}. Every Galerkin model is bounded below by the truncation floor."]
for K in (3, 4):
    f = base[K]["meta"]["fit"]
    L.append(f"* fitted A ({K}×{K}): D = {f['D']:.3g} (non-dim; {f['D_physical_km2_per_day']:.3g} km²/day), ρ = {f['rho']:.4f}/day, β = {f['beta']:.4f}, K = ρ/β = {f['K_carrying_capacity_u']:.3f}, fit loss {f['loss']:.3f}")
B16 = [json.load(open(os.path.join(MET, f))) for f in ("bench_16q_L4.json", "bench_16q_L1.json")]
L += ["", "## 16-qubit (4×4) runtime benchmark (statevector, no dense operators, 1 thread)", "",
      "| Trotter L | batch | fwd+bwd (s) | est. s/epoch (93 one-step + 609 rollout evaluations) | est. run (1500 epochs) |", "|---|---|---|---|---|"]
for b in B16:
    L.append(f"| {b['L']} | {b['B']} | {b['fwd_s'] + b['bwd_s']:.2f} | {b['est_epoch_s']:.0f} | {b['est_run_min_1500_epochs'] / 60:.0f} h |")
open(os.path.join(ROOT, "docs", "results_tables_usa2d.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
