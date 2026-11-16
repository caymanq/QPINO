---
title: "QPINO: a Physics-Informed Quantum Neural Operator for M = 2 Galerkin Fisher–KPP dynamics, with an exploratory JHU COVID-19 application"
author: "Cayman Quinn (BYU) — technical report generated from the code and results in this repository"
date: "November 2026"
---

**Status.** Everything below was produced by the code in this repository (branch `qpino-fisher-kpp`). All numbers are read from `results/metrics/*.json`; full tables are in [`docs/results_tables.md`](results_tables.md). All quantum circuits are *classically simulated* (exact 5-qubit statevector). No quantum advantage is claimed or implied.

# 1. Question and summary

**Research question.** Can a custom quantum Hamiltonian $H_\theta$, through the unitary $U_\theta(\tau)=e^{-iH_\theta\tau}$ together with an $R_y$ data embedding, $Z$-basis measurement and a classical decoder, approximate the nonlinear, dissipative one-step evolution operator of the $M=2$ Galerkin-truncated Fisher–KPP system?

SUMMARY_PLACEHOLDER

# 2. The Fisher–KPP model and its M = 2 Galerkin truncation

## 2.1 Model

$$\partial_t u = D\,\partial_x^2 u + \rho u - \beta u^2,\qquad D,\rho,\beta>0,\qquad x\in[0,2\pi)\ \text{(periodic)}.$$

$D$ (effective diffusion), $\rho$ (growth) and $\beta$ (saturation; carrying capacity $K=\rho/\beta$) are treated as **phenomenological** parameters. For the COVID field they are *not* interpreted as literal mobility or transmission rates. The model does not conserve $\int u\,dx$, and no conservation constraint is imposed.

## 2.2 Derivation

Insert $u_M(x,t)=\sum_{k=-2}^{2}a_k(t)e^{ikx}$ with $a_{-k}=a_k^*$ (real field). Then $\partial_x^2 e^{ikx}=-k^2e^{ikx}$ and
$$u_M^2=\sum_{p=-2}^{2}\sum_{q=-2}^{2}a_pa_q e^{i(p+q)x},$$
which contains wavenumbers $|p+q|\le 4$. Galerkin projection onto $e^{ikx}$, $|k|\le2$ (multiply by $e^{-ikx}/2\pi$ and integrate; $\frac{1}{2\pi}\int_0^{2\pi}e^{i(m-k)x}dx=\delta_{mk}$) keeps only $p+q=k$:
$$\boxed{\dot a_k=(\rho-Dk^2)a_k-\beta\sum_{p+q=k,\ |p|,|q|\le2}a_pa_q,\qquad k=-2,\dots,2.}$$
The discarded $|p+q|=3,4$ terms are the **unresolved modes** produced by the nonlinearity.

**Convolution sums.** Writing $a_1=b_1+ic_1$, $a_2=b_2+ic_2$:

* $k=0$: pairs $(0,0),(\pm1,\mp1),(\pm2,\mp2)$: $\ (u^2)_0=a_0^2+2|a_1|^2+2|a_2|^2$.
* $k=1$: pairs $(0,1),(1,0),(-1,2),(2,-1)$: $\ (u^2)_1=2a_0a_1+2a_1^*a_2$, with $a_1^*a_2=(b_1b_2+c_1c_2)+i(b_1c_2-c_1b_2)$.
* $k=2$: pairs $(0,2),(2,0),(1,1)$: $\ (u^2)_2=2a_0a_2+a_1^2$, with $a_1^2=(b_1^2-c_1^2)+2ib_1c_1$.

**Explicit real equations** for $\mathbf a=(a_0,b_1,c_1,b_2,c_2)=(a_0,\mathrm{Re}a_1,\mathrm{Im}a_1,\mathrm{Re}a_2,\mathrm{Im}a_2)$ (implemented in `src/galerkin.py::galerkin_rhs`):
$$
\begin{aligned}
\dot a_0 &= \rho a_0-\beta\left[a_0^2+2(b_1^2+c_1^2+b_2^2+c_2^2)\right]\\
\dot b_1 &= (\rho-D)b_1-2\beta\left[a_0b_1+b_1b_2+c_1c_2\right]\\
\dot c_1 &= (\rho-D)c_1-2\beta\left[a_0c_1+b_1c_2-c_1b_2\right]\\
\dot b_2 &= (\rho-4D)b_2-\beta\left[2a_0b_2+b_1^2-c_1^2\right]\\
\dot c_2 &= (\rho-4D)c_2-\beta\left[2a_0c_2+2b_1c_1\right]
\end{aligned}
$$

**Conjugate symmetry.** If $a_{-k}=a_k^*$ then $(u^2)_{-k}=\sum_{p+q=-k}a_pa_q=\sum_{p'+q'=k}a_{-p'}a_{-q'}=\big((u^2)_k\big)^*$, and $(\rho-Dk^2)$ is real and even in $k$, so $\dot a_{-k}=\dot a_k^*$ and $\dot a_0\in\mathbb R$: the real-field constraint is invariant under the flow. Verified numerically (Section 6) by integrating all five *complex* modes with no symmetry imposed.

**Reconstruction and derivatives** (analytic, used in the PDE residual):
$u_M=a_0+2(b_1\cos x-c_1\sin x)+2(b_2\cos2x-c_2\sin2x)$, $\ \partial_x^2u_M=-2(b_1\cos x-c_1\sin x)-8(b_2\cos2x-c_2\sin2x)$.
Projection: $a_k=\frac1N\sum_j u(x_j)e^{-ikx_j}$ on a uniform grid (exact for band-limited fields when $N\ge 2M+1$; the Galerkin RHS is recovered exactly from a grid projection when $N\ge3M+1$ because $u_M^2$ has $|k|\le4$).

**Coupling structure.** The Hessian $\partial^2\dot a_i/\partial a_j\partial a_k$ is constant (the RHS is quadratic). Its non-zero off-diagonal pattern is the set of coefficient pairs that interact through the convolution: $(a_0,b_1),(a_0,c_1),(a_0,b_2),(a_0,c_2),(b_1,c_1),(b_1,b_2),(b_1,c_2),(c_1,b_2),(c_1,c_2)$ — **9 of the 10 possible pairs**; only $(b_2,c_2)$ never appears. This defines the "structured" $ZZ$ graph (computed programmatically in `custom_ansatz.galerkin_graph`). Removing the five $a_0$ "mean-field" edges leaves the 5-edge "triad" graph of $k=1\leftrightarrow k=2$ mode-transfer couplings.

## 2.3 Reference solvers

* Full PDE: method of lines, Fourier pseudo-spectral Laplacian (or 2nd-order central differences), `scipy.integrate.solve_ivp(method="DOP853", rtol=1e-9, atol=1e-11)` (`src/fisher_kpp.py`).
* Galerkin ODE (baseline A): `solve_ivp(DOP853, rtol=1e-10, atol=1e-12)`, and a differentiable RK4 flow map with 20 sub-steps per step (agrees with DOP853 to $<10^{-10}$, test `test_galerkin_integrator_agreement`).

# 3. Quantum model

## 3.1 Embedding: why $R_y$

Standardise $\mathbf z=(\mathbf a-\mu)/\sigma$ (training-set statistics). $H_{\rm emb}(\mathbf z)=\frac12\sum_j\alpha_jz_jY_j$; the terms act on different qubits and commute, so
$S(\mathbf z)=e^{-iH_{\rm emb}}=\bigotimes_jR_y(\alpha_jz_j)$ (verified to $10^{-12}$ against `scipy.linalg.expm`), with
$$R_y(\phi)|0\rangle=\cos\tfrac\phi2|0\rangle+\sin\tfrac\phi2|1\rangle .$$
Reasons for $R_y$: (i) it produces **real** amplitudes, so $|\psi_{\rm emb}\rangle$ is a real product state whose computational-basis amplitudes are the products $\prod_j\{\cos\frac{\phi_j}2,\sin\frac{\phi_j}2\}$, i.e. the embedding moves probability between $|0\rangle$ and $|1\rangle$ on each qubit by an amount that encodes the size of the coefficient; (ii) a single qubit then has $\langle Z\rangle=\cos\phi$, $\langle X\rangle=\sin\phi$, $\langle Y\rangle=0$ — the coefficient value appears as a point on the $x$–$z$ great circle of the Bloch sphere; (iii) because $Y/2$ has eigenvalues $\pm\frac12$, every expectation value is a trigonometric polynomial with integer frequencies in $\phi_j$ (next section). Note that $\cos\phi$ is even: measuring $Z$ right after the embedding loses the **sign** of $z_j$. The transverse terms $g_jX_j$ of $H_\theta$ are what rotate $\sin\phi_j$ (sign-carrying) information into the $Z$ basis — with $g=0$ the ansatz is diagonal, commutes with every $Z$-basis observable, and the circuit would do nothing to the measured statistics.

## 3.2 Fourier expressibility (coefficient space)

For one embedding layer, $\langle O\rangle(\boldsymbol\phi)=\langle0|S^\dagger U^\dagger OUS|0\rangle$ and each $R_y(\phi_j)$ appears once in the ket and once in the bra. With generator eigenvalues $\lambda_\pm=\pm\frac12$ the accessible frequency differences per input are $\Omega_j=\{\lambda_a-\lambda_b\}=\{-1,0,1\}$, so
$$f(\mathbf z)=\sum_{\boldsymbol\omega\in\{-1,0,1\}^5}c_{\boldsymbol\omega}(\theta)\,e^{i\boldsymbol\omega\cdot(\boldsymbol\alpha\odot\mathbf z)} ,$$
$3^5=243$ frequency vectors (122 independent for a real output). With $R$ data re-uploads $W_R S\cdots W_1S$ the spectrum per input becomes $\{-R,\dots,R\}$ (sums of $R$ copies of $\Omega_j$), i.e. $(2R+1)^5=243,\ 3125,\ 16807$ frequency vectors for $R=1,2,3$. The trainable $\alpha_j$ rescale (but do not add) frequencies. These are **coefficient-space frequencies**: they describe how a decoded output varies with the *value* of a Galerkin coefficient, i.e. with the angle $\phi_j=\alpha_jz_j$. They are unrelated to the **spatial wavenumbers** $k\in\{-2,\dots,2\}$, which label *which* coefficients exist. The Galerkin RHS is a quadratic *polynomial* in $\mathbf a$; a degree-$R$ trigonometric polynomial represents it only approximately (e.g. $\sin\phi\approx\phi$, $\cos\phi\approx1-\phi^2/2$, products $\sin\phi_j\sin\phi_k$ from $\langle Z_jZ_k\rangle$-type terms), so accuracy depends on staying in a moderate-angle regime — tested by the $\alpha$ ablation. The numerical spectrum of trained models (Fig. 10) confirms the cut-off: Fourier power beyond $|\omega|=R$ is at round-off.

## 3.3 Custom Hamiltonian ansatz (every term)

$$H_\theta=\sum_{j=0}^4h_jZ_j+\sum_{j=0}^4g_jX_j+\sum_{(j,k)\in E}J_{jk}Z_jZ_k,\qquad U_\theta(\tau)=e^{-iH_\theta\tau}\ (\hbar=1).$$

* $h_jZ_j$ — **local phase evolution**: precession of qubit $j$ about $z$. On its own it changes no $Z$-basis probability, but it rotates the $x$–$y$ components ($\sin\phi_j$ information) so that, combined with $X$ mixing, the measured $\langle Z_j\rangle$ can become a general first-harmonic function $c_0+c_1\cos\phi_j+c_2\sin\phi_j$.
* $g_jX_j$ — **single-qubit mixing** (transverse field): rotates population between $|0\rangle$ and $|1\rangle$; required for any non-trivial $Z$-basis output (Section 3.1).
* $J_{jk}Z_jZ_k$ — **interactions between quantum features**: conditional phases that, together with $X$ mixing, make $\langle Z_j\rangle$ depend on $z_k$ and create entanglement. These are the only terms able to generate the cross-coefficient products that appear in the Galerkin convolution (e.g. $a_0b_1$, $b_1b_2+c_1c_2$). $E$ is the 9-edge Galerkin graph (structured), compared with the 10-edge fully connected graph, a 4-edge nearest-neighbour chain $0\!-\!1\!-\!2\!-\!3\!-\!4$, the 5-edge triad graph and $E=\varnothing$ (baseline F). We do **not** assume that two-body $ZZ$ terms reproduce the classical quadratic convolution; adequacy is tested experimentally.

**Time.** $\tau$ is the quantum evolution parameter; $\Delta t$ is the classical step. We use $\tau=c\,\Delta t$ with $c$ fixed or learned ($c=\mathrm{softplus}(\tilde c)>0$). Because the data use a single $\Delta t$, $c$ is degenerate with the overall scale of $H_\theta$ during training (only $cH_\theta$ enters); it matters only when the trained model is applied at a different $\Delta t$ (Section 8.4).

**Exact vs Trotter.** Exact: `torch.linalg.matrix_exp` on the $32\times32$ matrix. First-order product formula:
$U_\theta(\tau)\approx\big[\prod_je^{-ih_jZ_j\delta\tau}\prod_je^{-ig_jX_j\delta\tau}\prod_{(j,k)}e^{-iJ_{jk}Z_jZ_k\delta\tau}\big]^L$, $\delta\tau=\tau/L$ (the $ZZ$ factors act first). In gates: `IsingZZ(2Jδτ)`, `RX(2gδτ)`, `RZ(2hδτ)`.

**Re-uploading.** $U(\mathbf z,\tau)=W_R(\theta_R,\tau)S(\mathbf z)\cdots W_1(\theta_1,\tau)S(\mathbf z)$ with independent $\theta_r$. Because $S(\mathbf z)$ does not commute with $W_r$, this circuit is **not** in general evolution under any single time-independent Hamiltonian; it is a data-dependent product of unitaries and loses the "Hamiltonian surrogate" interpretation in exchange for a richer spectrum.

## 3.4 Measurement, decoder, update

$m_j=\langle Z_j\rangle$ and $m_{jk}=\langle Z_jZ_k\rangle$ (15 features). Decoder $D_\eta$: **linear** (default, so that the non-linearity must come from the circuit) or a 1-hidden-layer tanh MLP (variant D-mlpdec). Output scale $s_G=s_{\rm inc}/\Delta t$ (training increment scale). Residual update $\hat{\mathbf a}(t+\Delta t)=\mathbf a(t)+\Delta t\,G(\mathbf a,\Delta t)$, which gives an exact identity at $\Delta t=0$ (there $U=I$ and $G$ stays finite; verified). Gradients: PyTorch autograd through `matrix_exp`/Trotter products (exact statevector); on hardware this would be replaced by parameter-shift / generalised shift rules.

# 4. Loss

$\mathcal L=\lambda_{\rm data}\mathcal L_{\rm data}+\lambda_{\rm PDE}\mathcal L_{\rm PDE}+\lambda_{\rm roll}\mathcal L_{\rm roll}+\lambda_{\rm reg}\mathcal L_{\rm reg}$, defaults $(1,\,0.1,\,0.5,\,1)$.

* $\mathcal L_{\rm data}=\frac1B\sum_b\|(\hat{\mathbf a}_{n+1}-\mathbf a_{n+1})/s_{\rm inc}\|^2$ — the spec's squared error with a fixed diagonal metric $s_{\rm inc}$ = per-component training std of one-step increments (otherwise $a_0$ would dominate the $\sim\!20\times$ smaller $k=1,2$ coefficients). A value $\approx1$ means "as large as a typical one-step change". A reconstructed-field loss on the grid is also computed and logged.
* $\mathcal L_{\rm PDE}$ (default target): full reconstructed-field residual on $N_c=64$ collocation points, $R(x_i)=\frac{\hat u_M(x_i,t+\Delta t)-u_M(x_i,t)}{\Delta t}-\mathcal N[u]$, with $\mathcal N[u]=D\partial_x^2u+\rho u-\beta u^2$ evaluated analytically from the Fourier representation and $u^2$ taken **pointwise** (so it contains the $|k|=3,4$ modes). The model is a discrete map, not a continuous trajectory, so no autograd time derivative is used. Schemes: forward ($\mathcal N$ at $t$), **trapezoid** (default, average of $\mathcal N$ at $t$ and $t+\Delta t$), midpoint ($\mathcal N$ at the average state).
* Projected Galerkin residual $R_k=\frac{\hat a_k-a_k}{\Delta t}-f_k(\cdot)$ with the same schemes, always reported separately (and selectable as the training target).
* $\mathcal L_{\rm roll}$: $H$-step autonomous rollouts ($H=5$ synthetic, $7$ COVID) over all training windows.
* $\mathcal L_{\rm reg}$: negativity $\overline{\mathrm{relu}(-\hat u)^2}$ (weight 1), saturation $\overline{\mathrm{relu}(\hat u-1.2K)^2}$ (implemented; weight 0 by default because the COVID $K$ is poorly constrained), Hamiltonian-parameter $L_2$ ($10^{-4}$), rollout stability $\overline{\mathrm{relu}(\|\hat{\mathbf a}_{n+s}\|-3\max\|\mathbf a\|)^2}$ (weight 1).
* **Curriculum:** $\lambda_{\rm PDE}$ is 0 for the first 20 % of epochs and ramps linearly to its target by 60 %. **Model selection** (best validation one-step + rollout loss) is restricted to the last 40 % of epochs for *every* run, with or without physics loss (an earlier version allowed physics-free runs to select from all epochs; this confounded the physics ablation and was fixed before the reported runs).
* Optimiser: Adam, lr 0.02, cosine decay to 5 %, 1500 full-batch epochs, gradient clipping 10, float64, seeds 0/1/2.
