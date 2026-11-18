# Eqs. (6)–(9) on ER networks: annealed $pK$ scaling check

## 1. Research question

Ariaratnam and Strogatz (Phys. Rev. Lett. 86, 4278, 2001; arXiv:nlin/0012017) give an incoherent-branch Hopf condition and a death-equilibrium existence condition for the all-to-all Winfree model. The completed Stage 1 Erdős–Rényi scans are used here to ask whether those $p=1$ boundaries remain consistent with the saved finite-graph runs after the annealed replacement $K\to pK$. The relation under test is $pK_c(p,\delta)\approx K_c(1,\delta)$. It is a mean-field scaling hypothesis. It is not a theorem for a finite ER graph, and it is not a statement that a classifier label is the same object as a linear instability or a self-consistent equilibrium.

## 2. Original Eqs. (6)–(9) and their assumptions

The formulas below are the ones already implemented for the $p=1$ checks in `validate_eq7.py` and `validate_death_boundary.py`, as recorded in their reports. This script calls those functions. It does not introduce a new transcription of the 2001 paper. Where the paper's display equation was reduced to a numerical condition in that earlier code, the reduction used here is that code.

**Eq. (6), Hopf condition.** The stored solver treats the boundary of the stationary incoherent state as an implicit condition on the imaginary axis. A candidate is kept only when it is a unique root, the imaginary frequency is nonzero, and $K<1-\delta$. The real part is reduced to the cubic already coded for the Landau pole, and the imaginary part is a subtracted principal-value integral. These roots diagnose the truncation. They were not the $p=1$ comparison baseline, and they are not refit here. The solver returns no valid root at $\delta=0.225$ and $\delta=0.250$.

**Eq. (7), small-$\delta$ truncation.** The printed product form, truncated before $O(\delta^7)$, is

\[K_H^{(7)}(\delta)=\frac{8\delta}{\pi}\left[1+\frac{16\delta^2}{\pi^2}+\frac{16(\pi^2+80)\delta^4}{\pi^4}\right].\]

The additive expansion that drops the outer factor from the second and third terms is not this formula. At $\delta=0.10$ the two expressions differ by 0.01318. The pre-set comparison window is $0<\delta\le 0.20$. The series remains below the incoherent-branch edge $K=1-\delta$ until $\delta=0.2524$, and the dashed extension in that interval is exploratory. It is not a global exact boundary. $\delta=0$ is degenerate because every natural frequency equals 1.

**Eq. (8), death self-consistency.** For $\sigma\ge 1+\delta$,

\[F_\delta(\sigma)=1+\frac{1}{2\delta}\int_{1-\delta}^{1+\delta}\sqrt{1-\frac{\omega^2}{\sigma^2}}\,d\omega,\qquad K(\sigma,\delta)=\frac{\sigma}{F_\delta(\sigma)},\]

and $K_{\mathrm{exist}}(\delta)=\min K$. At $\delta=0$ the factor collapses to $1+\sqrt{1-1/\sigma^2}$, whose minimum is the exact value $4/(3\sqrt{3})$. Critical points solve $F-\sigma F'=0$. A positive slope at an interior root identifies a local minimum.

**Eq. (9), endpoint, and the saddle-node.** The minimum is not assigned by using the endpoint alone. The sign of $F-\sigma F'$ at $\sigma=1+\delta$ changes at $\delta_*=0.2956$. Below $\delta_*$ the minimum is an interior saddle-node, at $\sigma_{\mathrm{SN}}>1+\delta$. Above $\delta_*$ there is no interior root and the minimum is the endpoint $\sigma=1+\delta$. Both candidates are retained. The existence envelope is whichever candidate is lower. Existence of a stopped equilibrium is not the statement that the saved initial condition converges to it.

The incoherent branch used by Eq. (6) and Eq. (7) additionally requires $K<1-\delta$. Eq. (5), $K=1-\delta$, is that branch edge. It is drawn as a reference and is not retested here.

## 3. Derivation of the $pK$ scaling

The saved model is

\[\dot\theta_i=\omega_i-\frac{K}{N}\sin\theta_i\sum_{j=1}^{N}A_{ij}(1+\cos\theta_j),\]

with $N=1000$, a zero diagonal, and $\omega\in[1-\delta,1+\delta]$ on the deterministic grid. Replacing $A_{ij}$ by its expectation $p$ for $i\neq j$ produces a mean interaction $pK(N-1)/N$. The hypothesis tested here drops the factor $(N-1)/N$ and replaces $K$ by $pK$ in the $p=1$ condition. No degree correction and no spectral correction are added.

For the explicit Hopf truncation this is $K_H^{(p)}(\delta)=K_H^{(7)}(\delta)/p$ on the same $\delta$ window. The window is unchanged because $pK_H^{(p)}=K_H^{(7)}$ lies below $1-\delta$ exactly when the $p=1$ series does.

For death, the auxiliary amplitude $\sigma$ stays inside $F_\delta$. The scaled self-consistency is $pK=\sigma/F_\delta(\sigma)$, so $K_{\mathrm{Death,exist}}^{(p)}(\delta)=p^{-1}\min_{\sigma\ge 1+\delta}\sigma/F_\delta(\sigma)$. Multiplying $F_\delta$ by $p$ again would double-count the mean-field factor. The saddle-node and endpoint candidates scale by the same $1/p$. Their crossing $\delta_*$ does not depend on $K$ and is therefore the same at every $p$.

Eq. (5) scales in the same way: $K_{\mathrm{Eq5}}^{(p)}(\delta)=(1-\delta)/p$. A one-line finite-$N$ shift of size $1/999$ is smaller than the coupling step 0.05 and is not fitted.

## 4. Dataset audit

The Stage 1 index `/Users/mohan/Desktop/SRS/helper/stage1_runs/index.sqlite` was reread for this check. Status counts: completed 6962, pending 1, running 1. The Eq. (5) label cache records {'completed': 6962, 'pending': 1, 'running': 1}. Labels below come from that cache, version 1, built with archived Classifier v1.0.0-pilot. Config SHA-256 `5e77fae526cd6060b9f097db70d327f241c75a4579d7e9606c2070f360d1be44` matches the archive manifest: True. `classify.py` SHA-256 in the cache is `2d3488de136f0420cc6d570f6c23640b7759b4f5c728957c2b0da4d2b517d58b`; the manifest checksum is `b8dcc227ced4404952a1e93776abcf51ecd8bc77ad73e9db984fa477eb907bad`. The Eq. (5) note already traced that worktree difference to phase-diagram drawing helpers. This script does not reclassify and does not change thresholds.

Incomplete rows, excluded from every onset: `bc7f4faf661a90d2f2b43455b7538fcd` (running, p=0.2, delta=1.0, K=1.95); `6bfc891de871edc9c0b416ec405f843e` (pending, p=0.2, delta=1.0, K=2.0).

Completed runs in the cache: 6962. Distinct $p$ values are 0.2, 0.4, 0.8, and 1.0. There is no $p=0.05$ and no $p=0.60$. Maximum $|ω|$ discrepancy recorded by the cache is 0.000e+00. Network seed 8250318217634423533, nested graphs True, adjacency mismatches 0 of 6962 metadata files. Git HEAD at the cache build was `910f9c2700396a0cc081f0df5c96089b84bb8b2c`. Git HEAD for this script is `910f9c2700396a0cc081f0df5c96089b84bb8b2c`.

| p | delta columns | K counts | smallest Kmax | largest Kmax |
| --- | --- | --- | --- | --- |
| 0.2 | 41 | 1 cols × 39 K, 37 cols × 41 K, 3 cols × 61 K | 1.9 | 3.0 |
| 0.4 | 41 | 38 cols × 41 K, 3 cols × 61 K | 2.0 | 3.0 |
| 0.8 | 41 | 38 cols × 41 K, 3 cols × 61 K | 2.0 | 3.0 |
| 1.0 | 41 | 38 cols × 41 K, 3 cols × 61 K | 2.0 | 3.0 |

Coupling runs from 0 in steps of 0.05. Thirty-eight widths at each $p$ stop at $K=2$. The widths $\delta\in\{0.05,0.20,0.80\}$ continue to $K=3$, except that $p=0.2$, $\delta=1$ stops at $K=1.90$ because $K=1.95$ is running and $K=2.00$ is pending. A prediction above the scanned $K$ is out of range. It is not counted as evidence against the formula.

## 5. $p=1$ baseline reproduction

The $p=1$ labels, the Eq. (7) series, the Eq. (6) roots, and the death-existence values were compared with the published validation notes before any $p<1$ residual was interpreted. The ODE was not rerun.

| Result | Check | Detail |
| --- | --- | --- |
| PASS | Eq. (7) series at delta=0.025 | computed 0.06373, published 0.0637 |
| PASS | Eq. (7) series at delta=0.050 | computed 0.12785, published 0.12785 |
| PASS | Eq. (7) series at delta=0.075 | computed 0.19282, published 0.1928 |
| PASS | Eq. (7) series at delta=0.100 | computed 0.25915, published 0.25915 |
| PASS | Eq. (7) series at delta=0.125 | computed 0.32752, published 0.3275 |
| PASS | Eq. (7) series at delta=0.150 | computed 0.39876, published 0.39876 |
| PASS | Eq. (7) series at delta=0.175 | computed 0.47393, published 0.4739 |
| PASS | Eq. (7) series at delta=0.200 | computed 0.55435, published 0.55435 |
| PASS | p=1 completed profiles | found 1741 |
| PASS | p=1 Incoherence count | found 329, previous report 329 |
| PASS | p=1 Partial Locking count | found 1, previous report 1 |
| PASS | p=1 Locking count | found 88, previous report 88 |
| PASS | p=1 Partial Death count | found 348, previous report 348 |
| PASS | p=1 Death count | found 975, previous report 975 |
| PASS | p=1 Hopf label exit at delta=0.025 | status=sustained exit K_exit=0.1 after=Locking inside=True |
| PASS | p=1 Hopf label exit at delta=0.050 | status=sustained exit K_exit=0.15 after=Locking inside=True |
| PASS | p=1 Hopf label exit at delta=0.075 | status=sustained exit K_exit=0.2 after=Locking inside=True |
| PASS | p=1 Hopf label exit at delta=0.100 | status=sustained exit K_exit=0.3 after=Locking inside=True |
| PASS | p=1 Hopf label exit at delta=0.125 | status=sustained exit K_exit=0.35 after=Locking inside=True |
| PASS | p=1 Hopf label exit at delta=0.150 | status=sustained exit K_exit=0.45 after=Locking inside=False |
| PASS | p=1 Hopf label exit at delta=0.175 | status=sustained exit K_exit=0.5 after=Locking inside=True |
| PASS | p=1 Hopf label exit at delta=0.200 | status=non-monotonic K_exit=None after=Locking inside=False |
| PASS | Eq. (7) is the product form | absolute gap at delta=0.10 is 0.01318 |
| PASS | Eq. (7) crosses 1-delta near 0.252 | crossing 0.2524 |
| PASS | Eq. (6) root at delta=0.050 | K=0.12785, Im=-4.46e-16, ReGap=1.69e-15 |
| PASS | Eq. (6) root at delta=0.100 | K=0.25920, Im=-1.95e-16, ReGap=1.11e-16 |
| PASS | Eq. (6) root at delta=0.150 | K=0.39991, Im=1.46e-16, ReGap=2.89e-15 |
| PASS | Eq. (6) root at delta=0.200 | K=0.56905, Im=-1.32e-13, ReGap=-2.22e-16 |
| PASS | Eq. (6) root at delta=0.220 | K=0.67792, Im=-3.29e-16, ReGap=4.44e-16 |
| PASS | Eq. (6) has no valid root at delta=0.225 | stored None |
| PASS | Eq. (6) has no valid root at delta=0.250 | stored None |
| PASS | p=1 death anchor delta=0.000 | mechanism=saddle-node K=0.7698 observed=0.80 previous=Locking |
| PASS | p=1 death anchor delta=0.200 | mechanism=saddle-node K=0.7902 observed=0.80 previous=Incoherence |
| PASS | p=1 death anchor delta=0.225 | mechanism=saddle-node K=0.7956 observed=0.80 previous=Partial Locking |
| PASS | p=1 death anchor delta=0.275 | mechanism=saddle-node K=0.8082 observed=0.85 previous=Partial Death |
| PASS | p=1 death anchor delta=0.300 | mechanism=end-point K=0.8155 observed=0.85 previous=Partial Death |
| PASS | p=1 death anchor delta=0.400 | mechanism=end-point K=0.8495 observed=0.90 previous=Partial Death |
| PASS | p=1 death anchor delta=0.500 | mechanism=end-point K=0.8889 observed=0.90 previous=Partial Death |
| PASS | p=1 death anchor delta=1.000 | mechanism=end-point K=1.1202 observed=1.15 previous=Partial Death |
| PASS | p=1 death branch follows delta* | delta*=0.2956 |
| PASS | p=1 death residual sign | n=41 min=0.0028 |
| PASS | p=1 death residual range | min=0.0028 median=0.0273 max=0.0507 |
| PASS | p=1 death late set | late=[0.4, 0.525, 0.75] |
| PASS | delta=0 death closed form | K=0.769800 exact=0.769800 |

Quadrature and search-range lines from the original death solver: quad d=0.500 sigma=2.000 |F_analytic-F_quad|=2.22e-16 quad_err=9.4e-15; quad d=0.500 sigma=1.500 |F_analytic-F_quad|=0.00e+00 quad_err=4.8e-11; quad d=0.100 sigma=1.500 |F_analytic-F_quad|=2.22e-16 quad_err=1.7e-15; quad d=0.025 sigma=1.200 |F_analytic-F_quad|=3.55e-15 quad_err=3.1e-16; quad d=1.000 sigma=2.500 |F_analytic-F_quad|=0.00e+00 quad_err=1.4e-12; delta->0 gap at sigma=1.5: 1.79e-09; worst quadrature gap: 3.55e-15; search range K(d=0.2): 0.7902228854 at x80, 0.7902228854 at x200, diff=-2.22e-16; delta=0 K=0.7698003589 exact 4/(3*sqrt(3))=0.7698003589 diff=-2.22e-16. The baseline matches the earlier $p=1$ notes on the state counts, the eight main-window Hopf label exits, the Eq. (6) roots, the death anchors, the saddle-node/end-point split, and the sign and scale of the death residuals. The Eq. (7) series, not the Eq. (6) root, remains the comparison baseline inside $0<\delta\le 0.20$.

## 6. Hopf phase-diagram consistency

A Hopf threshold is a linear-stability statement about the stationary incoherent state. The quantity available here is a W3 classifier label. $K_{\mathrm{leave}}$ is the first scanned coupling whose label is not Incoherence. $K_{\mathrm{exit}}$ is the sustained-exit coupling from the $p=1$ Eq. (7) note: the column must start in Incoherence, and Incoherence must not return. Neither number is a Hopf eigenvalue. The plotted and tabulated residual is $\Delta(pK)=pK_{\mathrm{proxy}}-K_H^{(7)}(\delta)$. It is a label-proxy residual.

Coverage of the scaled Eq. (7) curve. `theory not applicable` includes $\delta=0$, every width outside $0<\delta\le 0.20$, and any series value that has crossed $1-\delta$.

| p | In range | Near upper boundary | Out of range | Insufficient K resolution | Theory not applicable |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 6 | 1 | 1 | 0 | 33 |
| 0.4 | 8 | 0 | 0 | 0 | 33 |
| 0.8 | 8 | 0 | 0 | 0 | 33 |
| 1.0 | 8 | 0 | 0 | 0 | 33 |

Main-window columns. The residual uses the first non-Incoherence label. A sustained Locking exit is reported separately and is still only a label change.

| p | delta | K_H^(7) | K_H(p) | Scan | K_leave | First label | Exit rule | K_exit | pK_leave - K_H^(7) | Mixed or return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 0.025 | 0.0637 | 0.319 | in range | 0.20 | Partial Locking | sustained exit | 0.20 | -0.024 | yes |
| 0.2 | 0.050 | 0.1279 | 0.639 | in range | 0.50 | Partial Locking | sustained exit | 0.50 | -0.028 | yes |
| 0.2 | 0.075 | 0.1928 | 0.964 | in range | 0.80 | Partial Locking | non-monotonic | — | -0.033 | yes |
| 0.2 | 0.100 | 0.2592 | 1.296 | in range | 0.75 | Partial Locking | non-monotonic | — | -0.109 | yes |
| 0.2 | 0.125 | 0.3275 | 1.638 | in range | 1.25 | Partial Locking | non-monotonic | — | -0.078 | yes |
| 0.2 | 0.150 | 0.3988 | 1.994 | near upper boundary | 2.00 | Partial Locking | sustained exit | 2.00 | 0.001 | no |
| 0.2 | 0.175 | 0.4739 | 2.370 | out of range | — | — | remains incoherence | — | — | no |
| 0.2 | 0.200 | 0.5544 | 2.772 | in range | 2.90 | Partial Locking | sustained exit | 2.90 | 0.026 | no |
| 0.4 | 0.025 | 0.0637 | 0.159 | in range | 0.15 | Partial Locking | sustained exit | 0.15 | -0.004 | yes |
| 0.4 | 0.050 | 0.1279 | 0.320 | in range | 0.25 | Partial Locking | sustained exit | 0.25 | -0.028 | yes |
| 0.4 | 0.075 | 0.1928 | 0.482 | in range | 0.35 | Partial Locking | sustained exit | 0.35 | -0.053 | yes |
| 0.4 | 0.100 | 0.2592 | 0.648 | in range | 0.55 | Partial Locking | sustained exit | 0.55 | -0.039 | yes |
| 0.4 | 0.125 | 0.3275 | 0.819 | in range | 0.65 | Partial Locking | non-monotonic | — | -0.068 | yes |
| 0.4 | 0.150 | 0.3988 | 0.997 | in range | 0.85 | Partial Locking | non-monotonic | — | -0.059 | yes |
| 0.4 | 0.175 | 0.4739 | 1.185 | in range | 1.20 | Partial Locking | non-monotonic | — | 0.006 | yes |
| 0.4 | 0.200 | 0.5544 | 1.386 | in range | 1.50 | Partial Locking | sustained exit | 1.50 | 0.046 | yes |
| 0.8 | 0.025 | 0.0637 | 0.080 | in range | 0.10 | Locking | sustained exit | 0.10 | 0.016 | no |
| 0.8 | 0.050 | 0.1279 | 0.160 | in range | 0.15 | Partial Locking | sustained exit | 0.15 | -0.008 | yes |
| 0.8 | 0.075 | 0.1928 | 0.241 | in range | 0.20 | Partial Locking | sustained exit | 0.20 | -0.033 | yes |
| 0.8 | 0.100 | 0.2592 | 0.324 | in range | 0.25 | Partial Locking | sustained exit | 0.25 | -0.059 | yes |
| 0.8 | 0.125 | 0.3275 | 0.409 | in range | 0.35 | Partial Locking | sustained exit | 0.35 | -0.048 | yes |
| 0.8 | 0.150 | 0.3988 | 0.498 | in range | 0.40 | Partial Locking | non-monotonic | — | -0.079 | yes |
| 0.8 | 0.175 | 0.4739 | 0.592 | in range | 0.45 | Partial Locking | non-monotonic | — | -0.114 | yes |
| 0.8 | 0.200 | 0.5544 | 0.693 | in range | 0.55 | Partial Locking | non-monotonic | — | -0.114 | yes |
| 1.0 | 0.025 | 0.0637 | 0.064 | in range | 0.10 | Locking | sustained exit | 0.10 | 0.036 | no |
| 1.0 | 0.050 | 0.1279 | 0.128 | in range | 0.15 | Locking | sustained exit | 0.15 | 0.022 | no |
| 1.0 | 0.075 | 0.1928 | 0.193 | in range | 0.20 | Locking | sustained exit | 0.20 | 0.007 | no |
| 1.0 | 0.100 | 0.2592 | 0.259 | in range | 0.30 | Locking | sustained exit | 0.30 | 0.041 | no |
| 1.0 | 0.125 | 0.3275 | 0.328 | in range | 0.35 | Locking | sustained exit | 0.35 | 0.022 | no |
| 1.0 | 0.150 | 0.3988 | 0.399 | in range | 0.45 | Locking | sustained exit | 0.45 | 0.051 | no |
| 1.0 | 0.175 | 0.4739 | 0.474 | in range | 0.50 | Locking | sustained exit | 0.50 | 0.026 | no |
| 1.0 | 0.200 | 0.5544 | 0.554 | in range | 0.65 | Locking | non-monotonic | — | 0.096 | yes |

State sequences in that window:

- p=0.2, delta=0.025: Inc[0.00,0.15] → PL[0.20,0.30] → Lock[0.35,2.00]
- p=0.2, delta=0.050: Inc[0.00,0.45] → PL[0.50,0.70] → Lock[0.75,3.00]
- p=0.2, delta=0.075: Inc[0.00,0.75] → PL[0.80,1.00] → Inc@1.05 → PL@1.10 → Lock[1.15,2.00]
- p=0.2, delta=0.100: Inc[0.00,0.70] → PL@0.75 → Inc[0.80,1.00] → PL[1.05,1.20] → Inc@1.25 → PL[1.30,1.45] → Inc@1.50 → PL@1.55 → Lock[1.60,2.00]
- p=0.2, delta=0.125: Inc[0.00,1.20] → PL[1.25,1.35] → Inc[1.40,1.60] → PL[1.65,2.00]
- p=0.2, delta=0.150: Inc[0.00,1.95] → PL@2.00
- p=0.2, delta=0.175: Inc[0.00,2.00]
- p=0.2, delta=0.200: Inc[0.00,2.85] → PL[2.90,3.00]
- p=0.4, delta=0.025: Inc[0.00,0.10] → PL@0.15 → Lock[0.20,1.90] → Death[1.95,2.00]
- p=0.4, delta=0.050: Inc[0.00,0.20] → PL[0.25,0.30] → Lock[0.35,1.85] → PL@1.90 → Death[1.95,3.00]
- p=0.4, delta=0.075: Inc[0.00,0.30] → PL[0.35,0.50] → Lock[0.55,1.85] → PL@1.90 → PD@1.95 → Death@2.00
- p=0.4, delta=0.100: Inc[0.00,0.50] → PL[0.55,0.70] → Lock[0.75,1.80] → PL[1.85,1.95] → Death@2.00
- p=0.4, delta=0.125: Inc[0.00,0.60] → PL[0.65,0.70] → Inc@0.75 → PL[0.80,0.90] → Lock[0.95,1.80] → PL[1.85,1.95] → Death@2.00
- p=0.4, delta=0.150: Inc[0.00,0.80] → PL@0.85 → Inc[0.90,1.00] → PL[1.05,1.20] → Lock[1.25,1.65] → PL[1.70,1.95] → Death@2.00
- p=0.4, delta=0.175: Inc[0.00,1.15] → PL[1.20,1.40] → Inc[1.45,1.50] → PL[1.55,1.60] → Inc@1.65 → PL[1.70,1.95] → Death@2.00
- p=0.4, delta=0.200: Inc[0.00,1.45] → PL[1.50,1.95] → PD@2.00 → Death[2.05,3.00]
- p=0.8, delta=0.025: Inc[0.00,0.05] → Lock[0.10,0.95] → Death[1.00,2.00]
- p=0.8, delta=0.050: Inc[0.00,0.10] → PL@0.15 → Lock[0.20,0.95] → Death[1.00,3.00]
- p=0.8, delta=0.075: Inc[0.00,0.15] → PL@0.20 → Lock[0.25,0.95] → Death[1.00,2.00]
- p=0.8, delta=0.100: Inc[0.00,0.20] → PL[0.25,0.30] → Lock[0.35,0.95] → Death[1.00,2.00]
- p=0.8, delta=0.125: Inc[0.00,0.30] → PL[0.35,0.40] → Lock[0.45,0.95] → Death[1.00,2.00]
- p=0.8, delta=0.150: Inc[0.00,0.35] → PL@0.40 → Inc@0.45 → PL@0.50 → Lock[0.55,0.90] → PL@0.95 → Death[1.00,2.00]
- p=0.8, delta=0.175: Inc[0.00,0.40] → PL@0.45 → Inc@0.50 → PL[0.55,0.65] → Lock[0.70,0.85] → PL[0.90,0.95] → Death[1.00,2.00]
- p=0.8, delta=0.200: Inc[0.00,0.50] → PL@0.55 → Inc[0.60,0.70] → PL[0.75,0.85] → Inc@0.90 → PL@0.95 → Death[1.00,3.00]
- p=1.0, delta=0.025: Inc[0.00,0.05] → Lock[0.10,0.75] → Death[0.80,2.00]
- p=1.0, delta=0.050: Inc[0.00,0.10] → Lock[0.15,0.75] → Death[0.80,3.00]
- p=1.0, delta=0.075: Inc[0.00,0.15] → Lock[0.20,0.75] → Death[0.80,2.00]
- p=1.0, delta=0.100: Inc[0.00,0.25] → Lock[0.30,0.75] → Death[0.80,2.00]
- p=1.0, delta=0.125: Inc[0.00,0.30] → Lock[0.35,0.75] → Death[0.80,2.00]
- p=1.0, delta=0.150: Inc[0.00,0.40] → Lock[0.45,0.75] → Death[0.80,2.00]
- p=1.0, delta=0.175: Inc[0.00,0.45] → Lock[0.50,0.75] → Death[0.80,2.00]
- p=1.0, delta=0.200: Inc[0.00,0.60] → Lock@0.65 → Inc[0.70,0.75] → Death[0.80,3.00]

## 7. Direct Hopf evidence

The completed metadata were read for every saved run. Order-parameter flags: {'sample_order_parameter_false': 6962, 'order_parameter_dt_none': 6962}. Missing JSON files: 0. Missing NPZ files: 0. Array headers: r_times=[0] snapshot_times=[8] theta_grid=[1000] on 6962 files. NPZ keys in one completed archive: omega, oscillator_index, rotation_numbers, rotation_number, window_start, window_end, theta_final, theta_grid, t_final, t_grid, t_origin, dt, step_index, snapshot_times, snapshot_theta, r_times, r_values, K, p, Delta, N, network_seed, initial_phase_seed, trial_id. There is no stored mean-field $H(t)$, and `r_times` is empty in every completed archive: the run configuration did not sample $R(t)$. `theta_grid` is one phase vector. `snapshot_theta` has eight phases. In the archives inspected for the Hopf window those times are $t=0,2000,\ldots,14000$.

Eight phases separated by 2000 time units are not a trajectory that can show a Hopf oscillation. An oscillator period is of order $2\pi$, so a sample every 2000 time units aliases any cycle and cannot establish periodicity, growth of an oscillatory amplitude, or a clean distinction between a limit cycle and a slow transient. The only macroscopic quantity computed from these snapshots is the order-parameter magnitude $|R|$ at $t\ge 10000$, averaged over those three late samples. It is reported below as a descriptive check of whether the first non-Incoherence label coincides with a large ordered snapshot. It is not a Hopf eigenvalue.

Late-snapshot $|R|$ across the first label change in the Eq. (7) window. `Before` is the saved cell immediately below that label, and `At leave` is the first non-Incoherence cell. Each number is the mean of $|R|$ at $t=10000$, $12000$, and $14000$.

| p | Columns | Median late |R| before | Median late |R| at leave | Leave cells with late |R|>0.5 |
| --- | --- | --- | --- | --- |
| 0.2 | 7 | 0.071 | 0.146 | 2/7 |
| 0.4 | 8 | 0.141 | 0.177 | 2/8 |
| 0.8 | 8 | 0.096 | 0.143 | 1/8 |
| 1.0 | 8 | 0.144 | 0.882 | 8/8 |

Examples of the same three-point average:

| p | delta | K before | late |R| | K leave | Label | late |R| |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 0.050 | 0.45 | 0.071 | 0.50 | Partial Locking | 0.051 |
| 0.4 | 0.050 | 0.20 | 0.031 | 0.25 | Partial Locking | 0.091 |
| 0.8 | 0.100 | 0.20 | 0.077 | 0.25 | Partial Locking | 0.107 |
| 1.0 | 0.100 | 0.25 | 0.127 | 0.30 | Locking | 0.879 |
| 1.0 | 0.200 | 0.60 | 0.508 | 0.65 | Locking | 0.951 |

At $p=1$ every Locking cell in this window has late $|R|>0.5$ (median $0.882$). The preceding cell is usually near $0.1$, except at $\delta=0.20$, where the Incoherence cell just below the single Locking point is already ordered. At $p\le 0.8$ the first Partial Locking label usually does not raise late $|R|$ above $0.5$. That disagreement means the early Partial Locking label is not, by itself, evidence that a macroscopic oscillation has begun. The sampling interval still cannot show whether $|R|$ oscillates between the stored times.

## 8. Death existence comparison

The observed onset is the first W3 label with near-zero fraction exactly 1 at $|\rho|<0.001$. The residual is $\Delta(pK)=pK_{\mathrm{Death,obs}}-K_{\mathrm{exist}}^{(1)}(\delta)$. It compares an attractor label with an existence threshold. A positive residual can mean that another attractor is still selected, that the finite-time run has not settled, or that the grid point above the threshold is the first available sample. A negative residual, Death while the stopped equilibrium does not yet exist, is the one-sided discrepancy singled out for inspection.

Coverage of the scaled existence envelope:

| p | In range | Near upper boundary | Out of range | Insufficient K resolution | Death before existence | In range but no Death |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 0 | 0 | 41 | 0 | 0 | 0 |
| 0.4 | 8 | 3 | 30 | 0 | 0 | 0 |
| 0.8 | 41 | 0 | 0 | 0 | 0 | 0 |
| 1.0 | 41 | 0 | 0 | 0 | 0 | 0 |

At every first Death cell the W3 near-zero count is 1000. Cells whose W1 or W2 count is below 1000: 0. Every first Death cell is inside the cut on W1, W2, and W3, so the onset is not produced by reading only W3. Agreement of the three windows still does not identify the saved state with the self-consistent equilibrium.

Columns with an early onset, a delay of more than one $K$ step, a missing in-range onset, or a Death label that does not persist:

| p | delta | Scan | K_exist(p) | K_Death | Previous | Steps | Residual pK | Persists |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 0.800 | in range | 2.559 | 2.75 | Partial Death | 3 | 0.077 | yes |

## 9. Saddle-node versus endpoint

The mechanism is a property of $F_\delta$ and does not change with $p$. It switches at $\delta_*=0.2956$. On the saddle-node side the endpoint value lies above the envelope. Using Eq. (9) alone would place the existence boundary too high wherever the two candidates differ by more than the grid. That separation is large only at very small $\delta$: at $\delta=0$ the endpoint is $K=1$ and the envelope is $0.7698$. By $\delta=0.10$ the gap has fallen to about $0.016$, and by $\delta=0.20$ it is about $0.001$. A count of which candidate lies closer to the observed onset is then decided by the $0.05$ bin, not by the mechanism. The mechanism itself remains the sign of $F-\sigma F'$ at the endpoint.

| p | Mechanism | In-range onsets | State before Death | Within 0 or 1 K step | Envelope closer than endpoint |
| --- | --- | --- | --- | --- | --- |
| 0.2 | saddle-node | 0 | — | — | — |
| 0.2 | end-point | 0 | — | — | — |
| 0.4 | saddle-node | 7 | Locking 2, Partial Death 2, Partial Locking 3 | 7/7 | 3/7 |
| 0.4 | end-point | 1 | Partial Death 1 | 0/1 | — |
| 0.8 | saddle-node | 12 | Locking 6, Partial Death 3, Partial Locking 3 | 12/12 | 2/12 |
| 0.8 | end-point | 29 | Partial Death 29 | 29/29 | — |
| 1.0 | saddle-node | 12 | Incoherence 1, Locking 8, Partial Death 2, Partial Locking 1 | 12/12 | 2/12 |
| 1.0 | end-point | 29 | Partial Death 29 | 29/29 | — |

## 10. Grid resolution and coverage

The coupling step is 0.05. In the $pK$ coordinate one step has length $0.05p$: 0.010 at $p=0.2$, 0.020 at $p=0.4$, 0.040 at $p=0.8$, and 0.050 at $p=1$. A residual smaller than that length is what the upper edge of a bin produces when the threshold sits between two scanned points. It is not sub-grid precision. Near-upper columns are those whose first scanned point above the prediction is the last point in the column, or whose prediction equals that last point. They are drawn as open markers and are not pooled into the common-interval medians. Out-of-range columns have no residual.

## 11. Comparison across $p$

Medians are not taken over every saved width. For the Hopf proxy, the common in-range window uses p=0.2, 0.4, 0.8, 1.0 and 6 shared widths. For death, the common in-range window uses p=0.4, 0.8, 1.0 and 8 shared widths. If the four probabilities do not all qualify, the smallest $p$ is dropped until the intersection is nonempty. That partial overlap is labeled as such.

Hopf label proxy on each $p$ separately, restricted to the Eq. (7) window and to in-range columns. These four medians are not one cross-$p$ average.

| p | In-range widths | Sustained Locking exit inside the gap | Median leave residual | Min | Max |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 6 | 0/0 | -0.030 | -0.109 | 0.026 |
| 0.4 | 8 | 0/0 | -0.034 | -0.068 | 0.046 |
| 0.8 | 8 | 1/1 | -0.053 | -0.114 | 0.016 |
| 1.0 | 8 | 6/7 | 0.031 | 0.007 | 0.096 |

Shared Hopf widths, p=0.2, 0.4, 0.8, 1.0: 0.025, 0.050, 0.075, 0.100, 0.125, 0.200.

| delta | p with a leave | Range of pK_leave | First label by p |
| --- | --- | --- | --- |
| 0.025 | 4 | 0.060 | 0.2:PL, 0.4:PL, 0.8:Lock, 1.0:Lock |
| 0.050 | 4 | 0.050 | 0.2:PL, 0.4:PL, 0.8:PL, 1.0:Lock |
| 0.075 | 4 | 0.060 | 0.2:PL, 0.4:PL, 0.8:PL, 1.0:Lock |
| 0.100 | 4 | 0.150 | 0.2:PL, 0.4:PL, 0.8:PL, 1.0:Lock |
| 0.125 | 4 | 0.100 | 0.2:PL, 0.4:PL, 0.8:PL, 1.0:Lock |
| 0.200 | 4 | 0.210 | 0.2:PL, 0.4:PL, 0.8:PL, 1.0:Lock |

Death onset on each $p$ separately, in-range columns only.

| p | In-range onsets | 0 or 1 K step | More than one K step | Before existence | Median residual | Max residual |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 0 | 0/0 | 0 | 0 | — | — |
| 0.4 | 8 | 7/8 | 1 | 0 | 0.024 | 0.077 |
| 0.8 | 41 | 41/41 | 0 | 0 | 0.031 | 0.057 |
| 1.0 | 41 | 41/41 | 0 | 0 | 0.027 | 0.051 |

Shared death widths, p=0.4, 0.8, 1.0: 0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.200, 0.800.

| delta | p with Death | Range of pK_Death | pK_Death by p |
| --- | --- | --- | --- |
| 0.000 | 3 | 0.020 | 0.4:0.780, 0.8:0.800, 1.0:0.800 |
| 0.025 | 3 | 0.020 | 0.4:0.780, 0.8:0.800, 1.0:0.800 |
| 0.050 | 3 | 0.020 | 0.4:0.780, 0.8:0.800, 1.0:0.800 |
| 0.075 | 3 | 0.000 | 0.4:0.800, 0.8:0.800, 1.0:0.800 |
| 0.100 | 3 | 0.000 | 0.4:0.800, 0.8:0.800, 1.0:0.800 |
| 0.125 | 3 | 0.000 | 0.4:0.800, 0.8:0.800, 1.0:0.800 |
| 0.200 | 3 | 0.020 | 0.4:0.820, 0.8:0.800, 1.0:0.800 |
| 0.800 | 3 | 0.050 | 0.4:1.100, 0.8:1.080, 1.0:1.050 |

## 12. Main conclusions and unresolved questions

1. The scaled Eq. (7) series is bracketed by a sustained Incoherence-to-Locking label in 7 of 30 in-range columns. Six of those are the $p=1$ widths already reported, and $\delta=0.15$ at $p=1$ still misses the stored gap by $0.001$ in $K$. On the other in-range columns the first label is Partial Locking, and 18 of 30 in-range leave residuals are negative: the label changes at a $pK$ below $K_H^{(7)}$. Panel A shows that failure to collapse. It is a statement about the classifier proxy, not about the Hopf eigenvalue.

2. Inside the Eq. (7) window and inside the scan, every recorded first departure from Incoherence is Locking or Partial Locking. Partial Death does not take that role on these columns. Of the 23 coarse snapshot pairs at $p\le 0.8$, 5 have late $|R|>0.5$ on the Partial Locking or Locking cell. The early Partial Locking labels are therefore not a macroscopic ordering transition in the stored snapshots.

3. No finely sampled $R(t)$ or $H(t)$ is stored. The eight phase snapshots cannot resolve a cycle. Every quantitative Hopf statement in this note is a classifier-proxy comparison, supplemented only by the coarse $|R|$ check in section 7. The Hopf instability itself was not measured.

4. Death before the existence envelope occurs in 0 columns. No saved Death label precedes the scaled existence threshold. Delays of more than one $K$ step on in-range columns: 1.

5. The saddle-node branch, not the Eq. (9) endpoint, determines the envelope for $\delta<\delta_*$. The endpoint remains the envelope only for $\delta>\delta_*$. The two candidates are drawn separately in panel C.

6. Out-of-range counts for the scaled Eq. (7) curve, among widths where the series itself is applicable: p=0.2: 1. Out-of-range counts for the death envelope: p=0.2: 41, p=0.4: 30. These scans end before the predicted coupling. They are coverage limits.

7. All four probabilities have a nonempty shared in-range Eq. (7) interval. The cross-$p$ comparison in section 11 is restricted to that interval.

8. The saved design has one network, one initial phase, and a finite observation time. At the first Death cell, W1, W2, and W3 are all entirely inside $|\rho|<0.001$, so that onset is not an artifact of reading only W3. The $K/N$ normalization and the difference between existence and attraction can still move an onset by a fraction of a grid step. The archives do not separate those effects from an error in the annealed $pK$ hypothesis. A finer $K$ scan or a linear-stability calculation would be required before a label residual is called a Hopf error.


## Source record

Theory implementation: `stage2_classifier/theoretical_validation/eq7_hopf/validate_eq7.py` SHA-256 `f1d95700ece8c931c869d67f549e5686c97f9d4b1aee5b64742507c9dcdb2e3c`, `stage2_classifier/theoretical_validation/death_boundary/validate_death_boundary.py` SHA-256 `6ab10460ab7f10f7cd6aaea816a8c4298f113fd23e3eac1878aa05239250f376`. Classifier config SHA-256 `5e77fae526cd6060b9f097db70d327f241c75a4579d7e9606c2070f360d1be44`. This script SHA-256 `732c19deffe81a6b1350a1ec0cbf0f0fdaf9e3447f55ba2489cfbc178ef788ee`. Label cache `/tmp/eq5_er_scaling_cache.pkl`, version 1. No ODE was rerun, no threshold was changed, and no new network correction was fitted.

