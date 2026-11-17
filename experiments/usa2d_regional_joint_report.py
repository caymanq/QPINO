"""T16 table + fig22: joint regional training with flux in the training rollouts and a long
rollout curriculum (report section 19)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import make_usa2d_figures as M
from src.regional_galerkin import Decomposition

plt, us_map, ERR, mask, u, dates, te, day, FIG, ACC = M.plt, M.us_map, M.ERR, M.mask, M.u, M.dates, M.te, M.day, M.FIG, M.ACC
ROOT = M.ROOT
OUT = os.path.join(ROOT, "results", "metrics", "usa2d_regional_joint")
R18 = os.path.join(ROOT, "results", "metrics", "usa2d_regional")
RG = np.load(os.path.join(ROOT, "data", "processed", "usa_regions_256x128.npz"))
lab, names = RG["label"].astype(int), [str(s) for s in RG["names"]]
Z = M.Z
NY, NX = mask.shape
LREF = NY * float(Z["dy_km"]) / np.pi
DD = Decomposition(lab, 3, float(Z["dx_km"]) / LREF, float(Z["dy_km"]) / LREF, names)
land = mask.ravel() > 0
Uf = u.reshape(len(dates), -1)
Aall = DD.coeffs(Uf)
proj = DD.field_flat(Aall[te], NY * NX)[:, land]
bR = json.load(open(os.path.join(R18, "baselines_regional_K3.json")))["results"]
a18 = defaultdict(list)
for f in sorted(glob.glob(os.path.join(R18, "assembled_K3_*_s[0-9].json"))):
    d = json.load(open(f)); a18[d["meta"]["model"]].append(d["results"]["coupled"])
runs = defaultdict(list)
for f in sorted(glob.glob(os.path.join(OUT, "joint_*_s[0-9].json"))):
    d = json.load(open(f)); m = d["meta"]
    runs[(m["model"], m["flux"], m["curr"], m["alpha"])].append(d["results"] | {"seed": m["seed"], "history": d["history"]})


def vsproj_day(pred):
    f = DD.field_flat(np.asarray(pred), NY * NX)[:, land]
    return np.sqrt(((f - proj) ** 2).mean(-1))


def fm(v, fmt="{:.4f}"):
    a = np.array(v, float)
    return fmt.format(a[0]) if len(a) == 1 else (fmt + " ± " + fmt).format(a.mean(), a.std())


def row(label, rows, extra=True):
    g = lambda k: fm([r[k] for r in rows])  # noqa: E731
    d26 = fm([vsproj_day(r["pred_rollout"])[-1] for r in rows])
    s = (f"| {label} | {len(rows)} | {g('rmse_onestep')} | {g('rmse_rollout')} | {g('rmse_jul12')} | {g('rmse_jul26')} | "
         f"{g('rmse_rollout_vs_projection')} | {d26} | {g('pde_violation_rollout')} |")
    return s


def jrow(label, key):
    rows = runs.get(key, [])
    if not rows:
        return f"| {label} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    c = [r["coupled"] for r in rows]
    sel = ", ".join(f"{r['selected']['H']}d@{r['selected']['epoch']}" for r in sorted(rows, key=lambda r: r["seed"]))
    return row(label, c) + f" {fm([r['pde_violation_train_onestep'] for r in rows])} | {sel} |"


L = ["## T16. Joint regional training: flux inside the training rollouts and a 7→14→21-day curriculum (report §19)", "",
     "Rollout from the observed Jun 30 state (coupled rollout for every row), RMSE in u vs the observed full-resolution field; 'vs floor' = against the "
     "regional 45-mode projection of the observed field (the 0.0949 floor itself is the projection's error vs observed). PDE violation = pointwise regional "
     "Fisher–KPP residual (u/day, interface flux excluded). λ_PDE = 1e-4. 'selected' = training horizon and epoch of the validation-selected checkpoint per seed.", "",
     "| model | seeds | 1-step RMSE | rollout RMSE | Jul 12 | Jul 26 | rollout vs floor | vs floor, Jul 26 | PDE viol. rollout | PDE viol. train 1-step | selected (H@epoch) |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
L.append(row("regional truncation floor", [bR["Projection-floor"]]) + " – | – |")
L.append(row("regional persistence", [bR["Persistence"]]) + " – | – |")
L.append(row("regional linear trend", [bR["LinearTrend"]]) + " – | – |")
L.append(row("regional Galerkin A (coupled)", [bR["A-regional-coupled"]]) + " – | – |")
for nm in ("D", "B"):
    L.append(row(f"§18 regional {nm} (independent training, flux only in rollout)", a18[nm]) + " – | – |")
lbl = {"D": "QPINO D", "C": "HEA C", "B": "MLP B"}
MAIN = {"D": 1.0, "C": 1.0, "B": 0.0}      # step weighting chosen on validation (seed-0 pilot)
ABL = {"D": 0.0, "C": 1.0, "B": 0.0}       # D ablation grid was launched at alpha=0 before the D pilot finished
for nm in ("D", "C", "B"):
    L.append(jrow(f"**§19 {lbl[nm]}: flux in training, 7→14→21 d (α={MAIN[nm]:g}, main)**", (nm, 1, "long", MAIN[nm])))
    a = ABL[nm]
    if a != MAIN[nm]:
        L.append(jrow(f"§19 {lbl[nm]}: flux in training, 7→14→21 d (α={a:g})", (nm, 1, "long", a)))
    L.append(jrow(f"§19 {lbl[nm]}: no flux in training, 7→14→21 d (α={a:g})", (nm, 0, "long", a)))
    L.append(jrow(f"§19 {lbl[nm]}: flux in training, 7 d only (α={a:g})", (nm, 1, "short", a)))
    L.append(jrow(f"§19 {lbl[nm]}: no flux in training, 7 d only (α={a:g})", (nm, 0, "short", a)))
L = [x for x in L if "| 0 | n/a" not in x]
L += ["", "Step-weighting pilot (seed 0, flux in training, 7→14→21 d; w_s ∝ s^-α), selected by the validation rollout:", "",
      "| model | α | val rollout RMSE (Jun 17–30) | July rollout RMSE |", "|---|---|---|---|"]
for nm in ("D", "C", "B"):
    for a in (0.0, 1.0):
        rr = [r for r in runs.get((nm, 1, "long", a), []) if r["seed"] == 0]
        if rr:
            L.append(f"| {lbl[nm]} | {a:g} | {rr[0]['val_rollout_rmse']:.4f} | {rr[0]['coupled']['rmse_rollout']:.4f} |")
tt = defaultdict(list)
for (nm, fl, cu, a), rows in runs.items():
    tt[(nm, cu)] += [r["train_time_s"] for r in rows]
L += [""] + [f"* training wall time, {lbl[k[0]]} {k[1]}: {np.mean(v):.0f} s per run (5 regions jointly, 1 thread; n={len(v)})" for k, v in sorted(tt.items())]
txt = "\n".join(L)
open(os.path.join(ROOT, "docs", "results_tables_usa2d_regional_joint.md"), "w").write("# Joint regional training (report §19)\n\n" + txt + "\n")
print(txt)

# ------------------------------------------------------------------ fig22
grid = lambda B: DD.field_flat(np.asarray(B), NY * NX).reshape(np.asarray(B).shape[:-2] + (NY, NX))  # noqa: E731
d26, i26 = day["2020-07-26"], int(np.where(dates == pd.Timestamp("2020-07-26"))[0][0])
med = lambda rows: sorted(rows, key=lambda r: r["rmse_rollout"])[len(rows) // 2]  # noqa: E731
mainD = [r["coupled"] | {"seed": r["seed"]} for r in runs.get(("D", 1, "long", 1.0), [])]
mainC = [r["coupled"] | {"seed": r["seed"]} for r in runs.get(("C", 1, "long", 1.0), [])]
mainB = [r["coupled"] | {"seed": r["seed"]} for r in runs.get(("B", 1, "long", 0.0), [])]
fig = plt.figure(figsize=(22, 8.6))
gs = fig.add_gridspec(2, 8, width_ratios=[1, 1, 1, 1, 1, 0.04, 0.8, 1.1], wspace=0.08, hspace=0.42)
im = us_map(fig.add_subplot(gs[0, 0]), u[i26], "observed\nJul 26")
us_map(fig.add_subplot(gs[0, 1]), grid(bR["Projection-floor"]["pred_rollout"])[d26], f"regional floor (45 modes)\nJul 26, RMSE {bR['Projection-floor']['rollout_rmse_by_day'][d26]:.3f}")
r18 = med(a18["D"])
us_map(fig.add_subplot(gs[0, 2]), grid(r18["pred_rollout"])[d26], f"§18 regional QPINO D\nJul 26, RMSE {r18['rollout_rmse_by_day'][d26]:.3f}")
panels = [(mainD, "§19 QPINO D, flux in training"), (mainB, "§19 MLP B, flux in training")]
for k, (rows, t_) in enumerate(panels):
    if rows:
        m_ = med(rows)
        us_map(fig.add_subplot(gs[0, 3 + k]), grid(m_["pred_rollout"])[d26], f"{t_}\nJul 26, seed {m_['seed']}, RMSE {m_['rollout_rmse_by_day'][d26]:.3f}")
fig.colorbar(im, cax=fig.add_subplot(gs[0, 5])).set_label("u")
errs = [(r18, "error §18 regional D"), (med(mainD) if mainD else None, "error §19 QPINO D"),
        (med(mainC) if mainC else None, "error §19 HEA C"), (med(mainB) if mainB else None, "error §19 MLP B"),
        (bR["A-regional-coupled"], "error regional Galerkin A")]
for k, (r_, t_) in enumerate(errs):
    if r_ is not None:
        ime = us_map(fig.add_subplot(gs[1, k]), grid(r_["pred_rollout"])[d26] - u[i26], t_ + "\n(Jul 26)", cmap=ERR, vmin=-0.4, vmax=0.4)
fig.colorbar(ime, cax=fig.add_subplot(gs[1, 5])).set_label("forecast − observed (u)")
ax = fig.add_subplot(gs[0, 7]); days = np.arange(1, len(te) + 1)
ax.plot(days, vsproj_day(bR["LinearTrend"]["pred_rollout"]), color="0.35", label="regional linear trend")
ax.plot(days, vsproj_day(bR["A-regional-coupled"]["pred_rollout"]), color="k", label="regional A")
ax.plot(days, vsproj_day(bR["Persistence"]["pred_rollout"]), color="0.65", label="regional persistence")
for rows, c, ls, l_ in ((a18["D"], "tab:blue", "--", "§18 D"), (a18["B"], "tab:orange", "--", "§18 B"),
                        (mainD, "tab:blue", "-", "§19 D"), (mainC, "tab:green", "-", "§19 C"), (mainB, "tab:orange", "-", "§19 B")):
    if rows:
        a = np.array([vsproj_day(r["pred_rollout"]) for r in rows])
        ax.plot(days, a.mean(0), color=c, ls=ls, lw=1.3, label=f"{l_} (n={len(a)})"); ax.fill_between(days, a.min(0), a.max(0), color=c, alpha=0.1)
ax.set_xlabel("July day"); ax.set_ylabel("RMSE vs regional projection"); ax.set_title("rollout drift away from the 45-mode projection", fontsize=11)
ax.legend(fontsize=7, frameon=False, ncol=2)
ax = fig.add_subplot(gs[1, 7])
for rr, c in ((runs.get(("D", 1, "long", 1.0), []), "tab:blue"), (runs.get(("B", 1, "long", 0.0), []), "tab:orange")):
    for r_ in rr:
        h = r_["history"]; ax.semilogy(h["epoch"], h["val"], color=c, lw=0.8, alpha=0.8)
for x in (300, 600):
    ax.axvline(x, color="0.5", lw=0.6, ls=":")
ax.text(150, ax.get_ylim()[1] * 0.8, "H=7", ha="center", fontsize=8); ax.text(450, ax.get_ylim()[1] * 0.8, "H=14", ha="center", fontsize=8)
ax.text(750, ax.get_ylim()[1] * 0.8, "H=21", ha="center", fontsize=8)
ax.set_xlabel("epoch"); ax.set_ylabel("validation rollout loss (Jun 17–30)"); ax.set_title("curriculum: validation score (D blue, B orange)", fontsize=11)
fig.suptitle("§19 Joint regional training with the interface flux inside the training rollouts and a 7→14→21-day curriculum (rollout from Jun 30)",
             fontsize=14, color=ACC)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig22_usa2d_regional_joint.{e}"), dpi=150, bbox_inches="tight", facecolor="white")
print("wrote fig22")
