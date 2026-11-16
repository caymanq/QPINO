"""PennyLane reference implementations of the QPINO circuits (default.qubit statevector).

Used to verify the fast torch-native simulator and for finite-shot checks, and to draw the
circuit diagrams.  Gate conventions: RZ(phi)=exp(-i phi Z/2), RX(phi)=exp(-i phi X/2),
IsingZZ(phi)=exp(-i phi ZZ/2), so exp(-i h Z dtau) = RZ(2 h dtau) etc.
"""
from __future__ import annotations

import itertools

import numpy as np
import pennylane as qml


def hamiltonian(h, g, J, edges):
    coeffs, ops = [], []
    for j in range(len(h)):
        coeffs.append(float(h[j])); ops.append(qml.PauliZ(j))
    for j in range(len(g)):
        coeffs.append(float(g[j])); ops.append(qml.PauliX(j))
    for (j, k), Jjk in zip(edges, J):
        coeffs.append(float(Jjk)); ops.append(qml.PauliZ(j) @ qml.PauliZ(k))
    return qml.Hamiltonian(coeffs, ops)


def embedding(angles):
    for j, a in enumerate(angles):
        qml.RY(float(a), wires=j)


def trotter_layer(h, g, J, edges, dtau):
    # matrix order prod_Z prod_X prod_ZZ -> ZZ gates are applied first in time
    for (j, k), Jjk in zip(edges, J):
        qml.IsingZZ(2 * float(Jjk) * dtau, wires=[j, k])
    for j in range(len(g)):
        qml.RX(2 * float(g[j]) * dtau, wires=j)
    for j in range(len(h)):
        qml.RZ(2 * float(h[j]) * dtau, wires=j)


def observables(n, include_zz=True):
    obs = [qml.PauliZ(j) for j in range(n)]
    if include_zz:
        obs += [qml.PauliZ(j) @ qml.PauliZ(k) for j, k in itertools.combinations(range(n), 2)]
    return obs


def make_qnode(n=5, shots=None, include_zz=True):
    dev = qml.device("default.qubit", wires=n, shots=shots)

    @qml.qnode(dev)
    def circuit(angles_layers, ham_layers, edges, tau, method="exact", L=1):
        """angles_layers: angles re-uploaded before each layer; ham_layers: [(h,g,J), ...]."""
        for angles, (h, g, J) in zip(angles_layers, ham_layers):
            embedding(angles)
            if method == "exact":
                qml.exp(hamiltonian(h, g, J, edges), -1j * float(tau))
            else:
                for _ in range(L):
                    trotter_layer(h, g, J, edges, float(tau) / L)
        return [qml.expval(o) for o in observables(n, include_zz)]

    return circuit


def make_hea_qnode(n=5, shots=None, include_zz=True):
    dev = qml.device("default.qubit", wires=n, shots=shots)

    @qml.qnode(dev)
    def circuit(angles, theta):
        embedding(angles)
        for l in range(theta.shape[0]):
            for j in range(n):
                qml.RY(float(theta[l, j, 0]), wires=j)
                qml.RZ(float(theta[l, j, 1]), wires=j)
            for j in range(n):
                qml.CNOT(wires=[j, (j + 1) % n])
        return [qml.expval(o) for o in observables(n, include_zz)]

    return circuit


def trotter_specs(h, g, J, edges, tau, L, n=5):
    """Gate counts / depth of the embedding + L-step Trotter circuit (ZZ decomposed into
    CNOT-RZ-CNOT for the 2-qubit-gate count)."""
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def c():
        embedding(np.zeros(n))
        for _ in range(L):
            trotter_layer(h, g, J, edges, tau / L)
        return qml.expval(qml.PauliZ(0))

    sp = qml.specs(c)()
    res = sp["resources"]
    return dict(depth=int(res.depth), n_gates=int(res.num_gates),
                gate_types=dict(res.gate_types), n_two_qubit=int(L * len(edges)),
                cnot_equiv=int(2 * L * len(edges)))
