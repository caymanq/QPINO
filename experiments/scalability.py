"""Resources, runtime scaling and measurement-shot sensitivity.

Run:  python experiments/scalability.py
"""
import json
import os
import time

import numpy as np
import torch

from common import (C, CKPT, MET, X_C, covid_bundle, ev, make_dataset, make_norm, save_json, synthetic_data)
from src import pennylane_circuits as plc
from src.custom_ansatz import HamiltonianAnsatz, HardwareEfficientAnsatz, full_graph, get_graph, nn_graph
from src.qpino import QPINO
from src.quantum_embedding import measurement_matrix, ry_product_state

t0 = time.time()
out = {}

# ---------------------------------------------------------------- runtime vs qubit number
scal = {}
for n in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
    row = {"statevector_bytes": 16 * 2 ** n}
    for name, edges in [("structured-like(full)", full_graph(n)), ("nearest", nn_graph(n))]:
        A = HamiltonianAnsatz(edges, n=n, seed=0)
        meas, _ = measurement_matrix(n, include_zz=False)
        ang = torch.randn(256, n)
        for meth, L in [("exact", 1), ("trotter", 4)]:
            if meth == "exact" and n > 10:
                continue
            reps = 5 if n <= 8 else 2
            t1 = time.time()
            for _ in range(reps):
                A.zero_grad()
                psi = ry_product_state(ang) @ A.unitary(1.0, meth, L).T
                ((psi.conj() * psi).real @ meas).sum().backward()
            row[f"{name}_{meth}_s_per_fwdbwd_batch256"] = (time.time() - t1) / reps
    scal[n] = row
    print(n, {k: f"{v:.3g}" for k, v in row.items()}, flush=True)
out["runtime_vs_qubits"] = scal

# torch vs PennyLane per-sample cost (5 qubits)
m = QPINO(make_norm(synthetic_data()["train"], 0.1, X_C)[0], c0=10.0)
q = plc.make_qnode()
lay = m.layers[0]
hp = (lay.h.detach().numpy(), lay.g.detach().numpy(), lay.J.detach().numpy())
a = torch.randn(20, 5) * 0.1 + m.norm.mu
ang = (m.alphas * (a - m.norm.mu) / m.norm.sigma).detach().numpy()
t1 = time.time()
for i in range(20):
    q([ang[i]], [hp], lay.edges, float(m.tau(0.1)), "trotter", 4)
pl_t = (time.time() - t1) / 20
m.evolution, m.trotter_L = "trotter", 4
t1 = time.time()
with torch.no_grad():
    for _ in range(20):
        m.measure(a, 0.1)
tt = (time.time() - t1) / 20 / 20
out["per_sample_forward_s"] = dict(pennylane_default_qubit=pl_t, torch_statevector_batched=tt)
print("per-sample forward: pennylane", pl_t, "torch", tt, flush=True)

# ---------------------------------------------------------------- circuit resources
res = {}
hp_any = lambda E: (np.ones(5), np.ones(5), np.ones(len(E)))  # noqa: E731
for g in ["structured", "triad", "full", "nearest", "none"]:
    E = get_graph(g)
    for L in [1, 4]:
        sp = plc.trotter_specs(*hp_any(E), E, 1.0, L)
        res[f"{g}_L{L}"] = dict(sp, n_hamiltonian_params=10 + len(E), n_edges=len(E))
dev_hea = HardwareEfficientAnsatz()
import pennylane as qml  # noqa: E402
qh = plc.make_hea_qnode()
sp = qml.specs(qh)(np.zeros(5), dev_hea.theta.detach().numpy())["resources"]
res["HEA_2layers"] = dict(depth=int(sp.depth), n_gates=int(sp.num_gates), gate_types=dict(sp.gate_types),
                          n_two_qubit=10, n_hamiltonian_params=20)
out["circuit_resources"] = res

# circuit-evaluation counts per training run (exact simulator vs hardware estimate)
S = C["synthetic"]
P_syn = S["n_train"] * S["n_steps"]
W_syn = S["n_train"] * (S["n_steps"] - S["horizon"] + 1)
n_eval_epoch = P_syn + W_syn * S["horizon"]
n_qparams = 10 + 9 + 5 + 1
out["circuit_evaluations"] = dict(
    synthetic_per_epoch_simulator=n_eval_epoch,
    synthetic_total_simulator=n_eval_epoch * C["train"]["epochs"],
    hardware_parameter_shift_per_epoch=n_eval_epoch * (2 * n_qparams + 1),
    hardware_shots_per_epoch_at_1e3_shots=n_eval_epoch * (2 * n_qparams + 1) * 1000,
    note="parameter-shift needs 2 evaluations per quantum parameter per sample (generalised shift rules "
         "needed for multi-eigenvalue generators such as exp(-i tau H_theta) when not decomposed into gates)")

# ---------------------------------------------------------------- shot sensitivity
from src.training import make_dataset as mk  # noqa: E402
shots = [64, 256, 1024, 4096, 16384, 65536]
sens = {}
data = synthetic_data()
norm, _ = make_norm(data["train"], data["dt"], X_C)
ck = os.path.join(CKPT, "synthetic_D_s0.pt")
if os.path.exists(ck):
    mD = QPINO(norm, c0=S["c0"], **C["model"])
    mD.load_state_dict(torch.load(ck)["state_dict"])
    Te = mk(data["test"], 1)
    sub = torch.arange(0, Te["a_n"].shape[0], 8)[:60]
    sens["synthetic"] = ev.shot_sensitivity(mD, Te["a_n"][sub], Te["a_np1"][sub], data["dt"], shots, repeats=10)
    # PennyLane finite-shot spot check (3 samples, 20000 shots)
    lay = mD.layers[0]
    hp = (lay.h.detach().numpy(), lay.g.detach().numpy(), lay.J.detach().numpy())
    qs = plc.make_qnode(shots=20000)
    with torch.no_grad():
        exact = mD.measure(Te["a_n"][sub[:3]], data["dt"]).numpy()
        ang = (mD.alphas * (Te["a_n"][sub[:3]] - norm.mu) / norm.sigma).numpy()
    pls = np.array([qs([ang[i]], [hp], lay.edges, float(mD.tau(data["dt"]))) for i in range(3)])
    sens["pennylane_20000shots_max_abs_dev_from_exact"] = float(np.abs(pls - exact).max())
    sens["expected_1sigma_at_20000_shots"] = float(1 / np.sqrt(20000))
if os.path.exists(os.path.join(CKPT, "covid_D_s0.pt")):
    B = covid_bundle()
    nrm, _ = make_norm(B["train"], B["dt"], X_C)
    mC = QPINO(nrm, c0=B["c0"], **C["model"])
    mC.load_state_dict(torch.load(os.path.join(CKPT, "covid_D_s0.pt"))["state_dict"])
    sens["covid_july"] = ev.shot_sensitivity(mC, torch.tensor(B["a_prev"]), torch.tensor(B["a_true"]), B["dt"], shots, repeats=10)
out["shot_sensitivity"] = sens
print("shots:", json.dumps(sens, indent=1)[:1500], flush=True)

# ---------------------------------------------------------------- training runtimes from the main runs
rt = {}
for f in ["synthetic_results.json", "covid_results.json"]:
    pth = os.path.join(MET, f)
    if os.path.exists(pth):
        J = json.load(open(pth))
        rt[f] = {k: dict(train_time_s=v.get("train_time_s"), n_params=v.get("n_params")) for k, v in J["summary"].items()}
out["training_runtime"] = rt
out["total_time_s"] = time.time() - t0
save_json(out, f"{MET}/scalability.json")
print(f"done in {time.time() - t0:.0f}s")
