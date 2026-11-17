"""Runtime benchmark: 16-qubit QPINO D / HEA C step on a statevector (no 2^n x 2^n matrices).
Used to decide whether the 4x4 (16-qubit) 2D runs fit the ~30 min/run limit (report section 16)."""
import itertools, json, os, sys, time
import numpy as np, torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("QPINO_THREADS", 1)))
n, B, L = 16, int(os.environ.get("B", 93)), int(os.environ.get("L", 4))
dim = 2 ** n
bits = ((np.arange(dim)[:, None] >> (n - 1 - np.arange(n))) & 1)
Zs = torch.tensor(1 - 2 * bits, dtype=torch.float64)                     # (dim, n)
E = list(itertools.combinations(range(n), 2))
ZZ = torch.stack([Zs[:, j] * Zs[:, k] for j, k in E], 1)                 # (dim, 120)
h = torch.randn(n, requires_grad=True); g = torch.randn(n, requires_grad=True); J = torch.randn(len(E), requires_grad=True)
ang = torch.randn(B, n)

def ry_state(a):
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    psi = torch.ones(B, 1, dtype=torch.complex128)
    for j in range(n):
        psi = torch.einsum("bi,bj->bij", psi, torch.stack([c[:, j], s[:, j]], 1).to(torch.complex128)).reshape(B, -1)
    return psi

def xlayer(psi, th):
    for j in range(n):
        v = psi.reshape(B, 2 ** j, 2, -1)
        v = torch.cos(th[j]) * v - 1j * torch.sin(th[j]) * v.flip(2)
        psi = v.reshape(B, dim)
    return psi

def evolve(psi, tau):
    dt_ = tau / L
    pz, pzz = torch.exp(-1j * dt_ * (Zs @ h)), torch.exp(-1j * dt_ * (ZZ @ J))
    for _ in range(L):
        psi = pz * xlayer(pzz * psi, g * dt_)
    return psi

t0 = time.time()
psi = evolve(ry_state(ang), 1.0)
p = (psi.conj() * psi).real
feat = torch.cat([p @ Zs, p @ ZZ], 1)
loss = (feat ** 2).sum()
t1 = time.time()
loss.backward()
t2 = time.time()
import resource
out = dict(n=n, B=B, L=L, fwd_s=t1 - t0, bwd_s=t2 - t1, peak_rss_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
# one QPINO epoch = 1 one-step batch (93) + rollout loss (87 windows x 7 steps = 609 evaluations)
out["est_epoch_s"] = (t2 - t0) * (93 + 609) / B
out["est_run_min_1500_epochs"] = out["est_epoch_s"] * 1500 / 60
print(json.dumps(out))
