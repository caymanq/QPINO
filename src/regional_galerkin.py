"""Regional domain decomposition of the 2D cosine-Galerkin Fisher-KPP system (report section 18).

The land mask is split into R regions (state groups).  Region r has its own Kr x Kr... (here K x K)
cosine basis on its bounding box, land-masked to the region's cells, with the weak-form operators
of src/cosine2d_galerkin.py (natural no-flux on the region boundary):

    G_r db_r/dt = -D K_r b_r + rho_r G_r b_r - beta C_r(b_r, b_r) + S_r(b),

and neighbouring regions are coupled classically through the diffusive flux across every
interface face (5-point finite-volume Laplacian, cell p in region r, neighbour cell q in s != r):

    S_r,i(b) = D * sum_{faces (p,q)} phi_i(p) (u_s(q) - u_r(p)) * A_cell / h_face^2,

i.e. the cross-interface part of the global discrete Laplacian, projected onto region r's basis.
Each face contributes equal and opposite amounts to the two regions' masses (exact conservation;
tests/test_regional.py).  Lengths are non-dimensional (L_ref = Ly/pi, as in section 16).
"""
from __future__ import annotations

import numpy as np
import torch

from . import cosine2d_galerkin as c2


class Region:
    def __init__(self, label_grid, r, K, dx, dy, name=""):
        NY, NX = label_grid.shape
        self.r, self.K, self.n, self.name = r, K, K * K, name
        cells = np.argwhere(label_grid == r)                    # (P, 2) [row, col]
        self.rows, self.cols = cells[:, 0], cells[:, 1]
        self.i0, self.i1 = self.rows.min(), self.rows.max() + 1
        self.j0, self.j1 = self.cols.min(), self.cols.max() + 1
        self.Lx, self.Ly = (self.j1 - self.j0) * dx, (self.i1 - self.i0) * dy
        self.flat = self.rows * NX + self.cols                    # indices into the flattened grid
        xy = np.c_[(self.cols - self.j0 + 0.5) * dx, (self.rows - self.i0 + 0.5) * dy]
        self.xy = xy
        self.w = np.full(len(xy), dx * dy)
        phi, grad, _ = c2.basis(xy, K, self.Lx, self.Ly)
        self.phi = phi
        G = (phi * self.w[:, None]).T @ phi
        self.G, self.Gi = G, np.linalg.inv(G)
        self.cond_G = float(np.linalg.cond(G))
        self.A_lin = self.Gi @ np.einsum("p,pid,pjd->ij", self.w, grad, grad)
        self.A_quad = np.einsum("ai,ijk->ajk", self.Gi, c2._tri(self.w, phi, phi, phi))
        self.proj = self.Gi @ (phi * self.w[:, None]).T
        self.t = {k: torch.tensor(getattr(self, k)) for k in ("A_lin", "A_quad")}

    def crop_mask(self, label_grid):
        return (label_grid[self.i0:self.i1, self.j0:self.j1] == self.r).astype(float)

    def coeffs(self, U_flat):
        """Land LS projection of fields (..., NY*NX) onto this region's basis."""
        return np.asarray(U_flat)[..., self.flat] @ self.proj.T

    def field(self, b):
        return np.asarray(b) @ self.phi.T


class Decomposition:
    def __init__(self, label_grid, K, dx, dy, names=None):
        self.label = label_grid
        R = int(label_grid.max()) + 1
        self.regions = [Region(label_grid, r, K, dx, dy, (names or [""] * R)[r]) for r in range(R)]
        self.R, self.n = R, K * K
        NY, NX = label_grid.shape
        pos = {}
        for reg in self.regions:
            for k, f in enumerate(reg.flat):
                pos[int(f)] = (reg.r, k)
        faces = []      # (region r, local cell p, region s, local cell q, weight A/h^2)
        for i in range(NY):
            for j in range(NX):
                a = label_grid[i, j]
                if a < 0:
                    continue
                for di, dj, h in ((0, 1, dx), (1, 0, dy)):
                    ii, jj = i + di, j + dj
                    if ii < NY and jj < NX and label_grid[ii, jj] >= 0 and label_grid[ii, jj] != a:
                        r, p = pos[i * NX + j]; s, q = pos[ii * NX + jj]
                        faces.append((r, p, s, q, dx * dy / h ** 2))
        self.faces = np.array(faces)
        self.n_faces = len(faces)
        # per ordered pair (r <- s): sparse coupling  S_r = D * [M_rs b_s - N_r b_r]
        self.M, self.N = {}, {r: np.zeros((self.n, self.n)) for r in range(R)}
        for (r, p, s, q, wt) in faces:
            r, p, s, q = int(r), int(p), int(s), int(q)
            for (a, ia, b_, ib) in ((r, p, s, q), (s, q, r, p)):
                ra, rb = self.regions[a], self.regions[b_]
                self.M.setdefault((a, b_), np.zeros((self.n, self.n)))
                self.M[(a, b_)] += wt * np.outer(ra.phi[ia], rb.phi[ib])
                self.N[a] += wt * np.outer(ra.phi[ia], ra.phi[ia])
        self.neighbours = sorted(set((a, b_) for (a, b_) in self.M))
        # Galerkin-form coupling (G_r^-1 applied)
        self.Mg = {k: self.regions[k[0]].Gi @ v for k, v in self.M.items()}
        self.Ng = {r: self.regions[r].Gi @ v for r, v in self.N.items()}
        self.t = dict(M={k: torch.tensor(v) for k, v in self.Mg.items()}, N={k: torch.tensor(v) for k, v in self.Ng.items()})

    # ------------------------------------------------------------- coupling and dynamics
    def flux(self, B, D, galerkin=True):
        """Interface source for all regions. B: (..., R, n) -> (..., R, n).
        galerkin=False returns the raw weak-form source <phi_i, flux> (before G^-1)."""
        tt = torch.is_tensor(B)
        M, N = (self.t["M"], self.t["N"]) if (tt and galerkin) else ((self.Mg, self.Ng) if galerkin else (self.M, self.N))
        outs = []
        for r in range(self.R):
            acc = -B[..., r, :] @ N[r].T
            for (a, s) in self.neighbours:
                if a == r:
                    acc = acc + B[..., s, :] @ M[(a, s)].T
            outs.append(D * acc)
        return torch.stack(outs, -2) if tt else np.stack(outs, -2)

    def local_rhs(self, B, D, rho, beta):
        """Independent regional weak-form RHS (no coupling). rho: scalar or per-region (R,)."""
        tt = torch.is_tensor(B)
        outs = []
        for r, reg in enumerate(self.regions):
            b = B[..., r, :]
            rr = rho[r] if np.ndim(rho) else rho
            Al, Aq = (reg.t["A_lin"], reg.t["A_quad"]) if tt else (reg.A_lin, reg.A_quad)
            ein = torch.einsum if tt else np.einsum
            outs.append(-D * b @ Al.T + rr * b - beta * ein("ajk,...j,...k->...a", Aq, b, b))
        return torch.stack(outs, -2) if tt else np.stack(outs, -2)

    def rhs(self, B, D, rho, beta, coupled=True):
        f = self.local_rhs(B, D, rho, beta)
        return f + self.flux(B, D) if coupled else f

    def rk4(self, B, dt, D, rho, beta, substeps=8, coupled=True):
        h = dt / substeps
        for _ in range(substeps):
            k1 = self.rhs(B, D, rho, beta, coupled)
            k2 = self.rhs(B + 0.5 * h * k1, D, rho, beta, coupled)
            k3 = self.rhs(B + 0.5 * h * k2, D, rho, beta, coupled)
            k4 = self.rhs(B + h * k3, D, rho, beta, coupled)
            B = B + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return B

    # ------------------------------------------------------------- data helpers
    def coeffs(self, U_flat):
        return np.stack([reg.coeffs(U_flat) for reg in self.regions], -2)

    def field_flat(self, B, size):
        B = np.asarray(B)
        out = np.full(B.shape[:-2] + (size,), np.nan)
        for r, reg in enumerate(self.regions):
            out[..., reg.flat] = B[..., r, :] @ reg.phi.T
        return out

    def masses(self, B):
        """Integral of the reconstructed u over each region (weak-form mass)."""
        return np.stack([(np.asarray(B)[..., r, :] @ reg.phi.T) @ reg.w for r, reg in enumerate(self.regions)], -1)
