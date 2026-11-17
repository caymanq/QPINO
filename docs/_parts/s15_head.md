# 15. Default model: real cosine / Neumann Galerkin basis (one real coefficient per qubit)

This is the **default model** of the repository (`configs/default.yaml: galerkin.basis = cosine, M = 8`). The classical data fed to the circuit are **only real Galerkin coefficients, one real coefficient per mode on one qubit**. All baselines A–F, persistence and linear trend in the cosine runs use exactly the same real inputs $\mathbf b=(b_0,\dots,b_M)$ (same standardisation, same splits, same loss with $\lambda_{\mathrm{PDE}}=0.01$ and the same ramp, same model-selection window). The complex-Fourier (Re/Im) periodic and mirrored runs of §§2, 5.3, 7–9 and 14 are kept as the **earlier complex-Fourier variant**.

## 15.1 Model and explicit triad tensor

On $s\in[0,L]$ with no-flux boundaries $u_s(0)=u_s(L)=0$, rescale $x=\pi s/L\in[0,\pi]$ and expand
$$u(x,t)=\sum_{k=0}^{M}b_k(t)\cos(kx),\qquad b_k\in\mathbb R .$$
Every term satisfies the Neumann condition, and $\{\cos kx\}$ is orthogonal on $[0,\pi]$ with $\int_0^\pi\cos^2(kx)\,dx=\pi$ for $k=0$ and $\pi/2$ for $k>0$, so the projection is $b_0=\frac1\pi\int_0^\pi u\,dx$, $b_k=\frac2\pi\int_0^\pi u\cos kx\,dx$ ($k\ge1$). With $\cos px\cos qx=\tfrac12[\cos(p-q)x+\cos(p+q)x]$ and $\cos(-mx)=\cos(mx)$,
$$u^2=\sum_{p,q=0}^{M}b_pb_q\cos px\cos qx=\sum_{k\ge0}\Big(\sum_{p,q}T_{kpq}\,b_pb_q\Big)\cos kx,\qquad T_{kpq}=\tfrac12\big[\delta_{|p-q|,k}+\delta_{p+q,k}\big].$$
The $k=0$ vs $k>0$ normalisation is built in: the projection weights $\frac1\pi$ and $\frac2\pi$ exactly compensate the norms, so the same $T$ serves all $k$. For example $(u^2)_0=b_0^2+\tfrac12\sum_{p\ge1}b_p^2$, because the mean of $\cos^2$ is $\frac12$. Galerkin truncation keeps $k\le M$ (terms with $p+q>M$ are dropped). With $\partial_x^2\cos kx=-k^2\cos kx$ (dimensional wavenumber $k\pi/L$),
$$\boxed{\ \dot b_k=\big(\rho-D\,k^2\big)\,b_k-\beta\sum_{p,q=0}^{M}T_{kpq}\,b_pb_q,\qquad k=0,\dots,M\ }$$
($D$ in units of $(L/\pi)^2$ per day; $D_{\mathrm{phys}}=D(L/\pi)^2$ with $L=3735$ km). Explicitly at $M=4$:

$$\begin{aligned}
\dot b_0&=\rho b_0-\beta\big[b_0^2+\tfrac12(b_1^2+b_2^2+b_3^2+b_4^2)\big]\\
\dot b_1&=(\rho-D)b_1-\beta\big[2b_0b_1+b_1b_2+b_2b_3+b_3b_4\big]\\
\dot b_2&=(\rho-4D)b_2-\beta\big[2b_0b_2+\tfrac12b_1^2+b_1b_3+b_2b_4\big]\\
\dot b_3&=(\rho-9D)b_3-\beta\big[2b_0b_3+b_1b_2+b_1b_4\big]\\
\dot b_4&=(\rho-16D)b_4-\beta\big[2b_0b_4+b_1b_3+\tfrac12b_2^2\big]
\end{aligned}$$

and in general
$$(u^2)_0=b_0^2+\tfrac12\sum_{p=1}^{M}b_p^2,\qquad (u^2)_k=\underbrace{\sum_{p=0}^{M-k}b_pb_{p+k}}_{\text{difference triads}}+\underbrace{\tfrac12\sum_{p=0}^{k}b_pb_{k-p}}_{\text{sum triads}}\quad(k\ge1),$$
so the mean mode always enters as $2b_0b_k$. At $M=8$ the tensor has **109 non-zero entries of $9^3=729$** (15 %); the number of distinct pairs $\{p,q\}$ entering equation $k$ is $[9,8,8,7,7,6,6,5,5]$ for $k=0..8$. Code: `src/cosine_galerkin.py` (`triad_tensor`, `quad` (einsum), `quad_explicit` (double loop), `galerkin_rhs`, `galerkin_rhs_by_projection`); dispatched through `src/galerkin.py` with `set_basis("cosine")`.

## 15.2 Verification (6 new tests in `tests/test_cosine_galerkin.py`)

| test | result |
|---|---|
| einsum tensor vs explicit double loop vs fine-grid ($N=256$) DCT projection of $Du_{xx}+\rho u-\beta u^2$ | max diff $1.1\times10^{-16}$ (loop) / $1.6\times10^{-15}$ (projection) at $M=8$; $k=0$ normalisation error 0 |
| Neumann reference solvers: 2nd-order FD with reflecting ghost cells vs even-extension spectral | errors $4.4\times10^{-4},1.1\times10^{-4},2.8\times10^{-5}$ at $N=32,64,128$; observed order 2.00 |
| cosine Galerkin ODE vs cosine projection of the fine-grid ($N=256$) Neumann solution, $t\in[0,4]$ | field rel. error $4.5\times10^{-3}$ ($M=2$), $4.2\times10^{-5}$ ($M=4$), $7.7\times10^{-10}$ ($M=8$), $3.0\times10^{-9}$ ($M=16$, time-integration tolerance floor) — spectral convergence |
| cosine projection of the 32-bin field = even-mirror Fourier projection | max diff $2.0\times10^{-15}$ ($M=8$) |
| coupling graph | 36 of 36 pairs at $M=8$, 10 of 10 at $M=4$; 109 tensor non-zeros |
| 9-qubit cosine QPINO, torch simulator vs PennyLane `default.qubit` | max diff $5.4\times10^{-15}$ |

## 15.3 Coupling graph and its selectivity

The structured Hamiltonian graph has an edge $(p,q)$ whenever $\partial^2\dot b_k/\partial b_p\partial b_q\ne0$ for some $k$. For the cosine basis **every pair couples**: for $p\ne q$ the difference triad $k=|p-q|\le M$ always exists. The graph is therefore **complete** (36/36 edges at $M=8$, 10/10 at $M=4$) and **not selective at the pair level**, unlike the complex-Fourier graph (9 of 10 edges at $M=2$, 34 of 36 at $M=4$). The selectivity of the cosine system lives only in the tensor (15 % non-zero; each equation sees 5–9 pairs). A pairwise $Z_pZ_q$ Hamiltonian cannot express which output $k$ a pair should feed. That job is left to the readout/decoder, so in the cosine basis "structured" = "full" and the graph ablation has nothing to test. Model F (no $ZZ$) remains the informative structural ablation.

## 15.4 Data

* **COVID:** the original 32-bin I-10 field (LA → Jacksonville, $L=3735$ km) with Neumann boundaries. The cosine projection is identical to the even-mirror Fourier projection (test above), but each mode now uses one qubit instead of a Re/Im pair whose Im part is identically 0. The splits are unchanged: train Mar 1–Jun 16, validate Jun 17–30, test July 2020 (one-step and a 31-day rollout from the observed Jun 30 state).
* **Synthetic:** a Neumann Fisher–KPP dataset (`cosine_galerkin.generate_neumann_trajectories`). Random smooth non-negative cosine ICs evolve under the full-grid Neumann solver ($N=256$, spectral even extension) with the same $(D,\rho,\beta)$ and $\Delta t$ as the periodic synthetic set, and are projected onto $M+1$ cosine modes. The test set is 12 unseen ICs × 40 steps; rollout errors are measured against the projection of the full PDE solution.
