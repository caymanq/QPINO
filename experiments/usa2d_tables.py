"""T12-T13: 2D contiguous-USA tables -> docs/results_tables_usa2d.md."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics", "usa2d")
runs, runs17 = defaultdict(list), defaultdict(list)
for f in glob.glob(os.path.join(MET, "usa2d_K*_s[0-9].json")):
    d = json.load(open(f)); m = d["meta"]
    if m.get("lambda_pde") == 1e-4:       # section 17 runs (lambda 1e-4, scalar / spatial rho) go to T14
        runs17[(m["K"], m.get("rho", "scalar"), m["model"])].append(d["result"])
    else:
        runs[(m["K"], m["model"])].append(d["result"])
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

# ------------------------------------------------------------------ section 17: spatial rho, lambda 1e-4
bS = {K: json.load(open(os.path.join(MET, f"baselines_usa2d_K{K}_rhoS4.json"))) for K in (3, 8)}
b8 = json.load(open(os.path.join(MET, "baselines_usa2d_K8.json")))


def m17(rows, k, fmt="{:.4f}"):
    if not rows:
        return "n/a"
    a = np.array([k(r) if callable(k) else r[k] for r in rows], float)
    if np.isnan(a).any():
        return "–"
    if len(rows) == 1:
        return fmt.format(a[0])
    return (fmt + " ± " + fmt).format(a.mean(), a.std()) + ("" if len(a) == 3 else f" (n={len(a)})")


def pv(r, split, ref):
    v = r.get("pde_violation", {}).get(split)
    return v[ref] if v else float("nan")


def row(K, lab, P, rows, own):
    f = lambda k, fmt="{:.4f}": m17(rows, k, fmt)  # noqa: E731
    return (f"| {K}×{K} | {lab} | {P if isinstance(P, str) else f('n_params', '{:.0f}')} | {f('rmse_onestep')} | {f('rmse_rollout')} | {f('rmse_jul12')} | "
            f"{f('rmse_jul26')} | {f('rmse_rollout_vs_projection')} | {f(lambda r: pv(r, 'val_onestep', own), '{:.4f}')} | "
            f"{f(lambda r: pv(r, 'test_rollout', 'scalar'), '{:.4f}')} / {f(lambda r: pv(r, 'test_rollout', 'spatial'), '{:.4f}')} |").replace("– / –", "–")


L += ["", "## T14. Spatial growth rate ρ(x,y) (report §17): July 2020 RMSE and physics-violation diagnostic", "",
      "λ_PDE = 1e-4 for all trained rows except the §16 rows (λ = 0.01). ρ(x,y) = Σ r_m ψ_m with a 4×4 cosine basis (16 coefficients). "
      "PDE violation = RMS (u/day) of the trapezoid full-PDE residual on 2000 land points; 'own' uses the model's reference physics "
      "(scalar fit for scalar rows, spatial fit for ρ(x,y) rows); rollout column gives scalar-ref / spatial-ref. "
      "Observed July data: " + f"{bS[3]['results']['Persistence']['pde_violation']['observed_test']['scalar']:.4f} / "
      f"{bS[3]['results']['Persistence']['pde_violation']['observed_test']['spatial']:.4f} (3×3 projection, scalar / spatial ref).", "",
      "| modes | model | params | 1-step RMSE | rollout RMSE | Jul 12 | Jul 26 | rollout vs projection | PDE viol. val 1-step (own) | PDE viol. test rollout (scalar / spatial ref) |",
      "|---|---|---|---|---|---|---|---|---|---|"]
b3 = bS[3]["results"]
L.append(f"| 3×3 | truncation floor | – | {b3['Projection-floor']['rmse_onestep']:.4f} | {b3['Projection-floor']['rmse_rollout']:.4f} | "
         f"{b3['Projection-floor']['rmse_jul12']:.4f} | {b3['Projection-floor']['rmse_jul26']:.4f} | 0 | – | "
         f"{pv(b3['Persistence'], 'observed_test', 'scalar'):.4f} / {pv(b3['Persistence'], 'observed_test', 'spatial'):.4f} |")
for nm, lab, P in (("Persistence", "Persistence", "0"), ("LinearTrend", "Linear trend", "0")):
    L.append(row(3, lab, P, [b3[nm]], "scalar"))
L.append(row(3, "A, scalar ρ (§16 one-step fit)", "3", [base[3]["results"]["A"]], "scalar"))
L.append(row(3, "A, scalar ρ (rollout protocol)", "3", [b3["A-scalar-sameprotocol"]], "scalar"))
L.append(row(3, "**A-ρ(x,y)** (rollout protocol)", "18", [b3["A"]], "spatial"))
for nm, lab in (("D", "D (9-qubit QPINO)"), ("C", "C (HEA QPINO)"), ("B", "B (MLP)")):
    L.append(row(3, f"{lab}, scalar ρ, λ=0.01 (§16)", None, runs.get((3, nm), []), "scalar"))
    L.append(row(3, f"{lab}, scalar ρ, λ=1e-4", None, runs17.get((3, "scalar", nm), []), "scalar"))
    L.append(row(3, f"**{lab}, ρ(x,y), λ=1e-4**", None, runs17.get((3, "spatial", nm), []), "spatial"))
b8s = bS[8]["results"]
L.append(f"| 8×8 (classical) | truncation floor | – | {b8s['Projection-floor']['rmse_onestep']:.4f} | {b8s['Projection-floor']['rmse_rollout']:.4f} | "
         f"{b8s['Projection-floor']['rmse_jul12']:.4f} | {b8s['Projection-floor']['rmse_jul26']:.4f} | 0 | – | – |")
L.append(row(8, "classical A, scalar ρ (one-step fit)", "3", [b8["results"]["A"]], "scalar").replace("| 8×8 |", "| 8×8 (classical) |"))
L.append(row(8, "classical A, scalar ρ (rollout protocol)", "3", [b8s["A-scalar-sameprotocol"]], "scalar").replace("| 8×8 |", "| 8×8 (classical) |"))
L.append(row(8, "classical A-ρ(x,y) (rollout protocol)", "18", [b8s["A"]], "spatial").replace("| 8×8 |", "| 8×8 (classical) |"))
L += [""]
for K in (3, 8):
    f = bS[K]["meta"]["fit"]; s_ = f["scalar_same_protocol"]
    L.append(f"* ρ(x,y) fit ({K}×{K}): window {f['fit_window_days']} d, λ_L2 = {f['lam_l2']:g}, D = {f['D']:.3g}, β = {f['beta']:.4f}, "
             f"ρ range on land [{f['rho_field_range'][0]:.4f}, {f['rho_field_range'][1]:.4f}]/day (mean {f['rho_field_mean']:.4f}), "
             f"val rollout rel. L2 {f['val_rollout_rel_l2']:.4f}; scalar same protocol: window {s_['fit_window_days']} d, ρ = {s_['rho_field_mean']:.4f}, "
             f"β = {s_['beta']:.4f}, val rollout rel. L2 {s_['val_rollout_rel_l2']:.4f}.")
L.append(f"* local log-growth initialiser range (3×3 projection, last 28 training days): [{bS[3]['meta']['fit']['rho_local_growth_range'][0]:.3f}, "
         f"{bS[3]['meta']['fit']['rho_local_growth_range'][1]:.3f}]/day.")
open(os.path.join(ROOT, "docs", "results_tables_usa2d.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
