# Degree-based annealed Hopf boundary for an ER network

Question: does degree heterogeneity correct the naive \(pK\) scaling of the all-drifting Hopf boundary by a significant amount?

The calculation extends the \(p=1\) linear response already checked against Ariaratnam and Strogatz, Phys. Rev. Lett. 86, 4278 (2001), Eq. (6). No Winfree ODE is integrated. Classifier v1.0.0-pilot is not used. \(K_H(p=1)/p\) is a comparison baseline, not the Hopf boundary of a quenched ER network.

## Characteristic equation

For the undirected ER graph the degree is \(\mathrm{Binomial}(N-1,p)\). The neighbor distribution is

\[
q(k)=\frac{k\,P_{\mathrm{deg}}(k)}{\langle k\rangle}.
\]

For \(0<p<1\) this is a point mass of the excess binomial: if \(X\sim\mathrm{Binomial}(N-2,p)\), then \(q\) is the law of \(X+1\), and

\[
\mathbb{E}_q[k]=\frac{\langle k^2\rangle}{\langle k\rangle}=1+(N-2)p.
\]

At \(p=1\), \(q\) is a point mass at \(k=N-1\).

In the degree-based annealed approximation the input to a degree-\(k\) oscillator is

\[
H_k=\frac{k}{N}\sum_{k'}q(k')G_{k'}.
\]

On the all-drifting stationary branch \(G_k^*=1\), so \(H_k^*=k/N\) and the local coefficient is \(a_k=Kk/N\). A collective perturbation \(s=\sum q\,\delta G\) then shifts every class by \(\delta a_k=(Kk/N)s=a_k s\).

The susceptibility \(\chi(\lambda;a,\delta)\) is the unit-\(\delta a\) response validated at \(p=1\). It is evaluated at the local coefficient \(a_k\), not at the global \(K\). The global \(K\) enters only by setting \(a_k=Kk/N\) and by the feedback prefactor. Closing \(s=\sum q\,\delta G\) gives

\[
D(\lambda;K,p,\delta,N)
=1-\sum_k q(k)\,a_k\,\chi(\lambda;a_k,\delta)
=1-\frac{K}{N}\sum_k k\,q(k)\,\chi(\lambda;a_k,\delta).
\]

The two writings are the same identity. The candidate equation is kept as stated.

At \(p=1\) the sum collapses to one class, \(a=K(N-1)/N\), and \(D=1-a\chi(\lambda;a)\). That is the validated all-to-all characteristic equation at coupling \(a\). Therefore

\[
K_H(p=1,N)=K_H^{\mathrm{Eq.(6)}}(\delta)\,\frac{N}{N-1},
\]

and \(N\to\infty\) recovers Eq. (6). The factor \(N/(N-1)\) is the missing self-pulse of the zero-diagonal adjacency. It is not degree heterogeneity: at \(p=1\) there is no degree spread.

The formula uses \(G_k^*=1\) in every class that carries neighbor weight. It is not extended onto a pinned/drifting mixed branch. A class with \(a_k\ge 1-\delta\) is omitted only when its neighbor mass is at most \(10^{-8}\); the retained weights are not renormalized. A larger pinned mass marks the case invalid and leaves \(K_H\) blank.

## Numerical method

\(N=1000\), \(p\in\{1,0.8,0.6,0.4,0.2\}\), \(\delta\in\{0.05,0.10,0.20\}\). The closed-form observable and the \(\mathrm{Re}\,\lambda\to 0^+\) continuation are the Step 1 routines. Fourier–Galerkin is used only as a truncation check, on the same frequency nodes as the closed form. Roots are not computed on the imaginary axis. For \(\varepsilon\in\{10^{-2},3\times 10^{-3},10^{-3},3\times 10^{-4},10^{-4}\}\), \(D(\varepsilon+i\Omega)=0\) is solved from three starts near the homogeneous guess and clustered at \(\Delta K<2\times 10^{-4}\), \(\Delta\Omega<2\times 10^{-3}\). The accepted value is the quadratic intercept in \(\varepsilon\). Its uncertainty is the absolute difference between that intercept and the linear intercept through the two smallest accepted \(\varepsilon\).

The degree window starts at three standard deviations and grows until the discarded binomial mass is below the requested tolerance. Production runs discard less than \(10^{-17}\).

A case is reported only when all of the following hold: one continued root, relative continuation uncertainty below \(10^{-4}\), residual below \(10^{-7}\), exact pinned neighbor mass at most \(10^{-8}\), degree-tail shift below \(10^{-5}\), and Fourier agreement below \(10^{-6}\). A pure imaginary root of this scalar equation is not a proof of a nonlinear Hopf bifurcation, and it is not a statement about one quenched ER realization.

## Checks

The \(p=1\) recovery gate asks for \(K_H^{\mathrm{degree}}=K_H^{\mathrm{Eq.(6)}}\,N/(N-1)\) within a relative tolerance \(10^{-4}\). The three gaps are \(4.9\times 10^{-7}\), \(2.5\times 10^{-8}\), and \(7.3\times 10^{-7}\). They reproduce the Step 1 difference between the response extrapolation and the Eq. (6) solver. The gate passes.

Degree truncation at \(\varepsilon=10^{-3}\): moving the discarded mass from about \(6\times 10^{-7}\) to below \(10^{-17}\) changes \(K\) by at most \(1.5\times 10^{-7}\) relative. The production window and the \(10^{-8}\) window differ by less than \(10^{-11}\) relative. At \(p=1\) the window is a single degree.

Fourier truncation at the mean neighbor coefficient agrees with the closed form to about \(10^{-11}\) already at \(M=16\). The most demanding retained class in this grid, degree 259 at \(p=0.2\), \(\delta=0.20\), has relative errors \(4\times 10^{-5}\) at \(M=16\), \(8\times 10^{-10}\) at \(M=32\), and \(7\times 10^{-14}\) at \(M=48\).

Residuals at \(\varepsilon=10^{-4}\) are between \(10^{-16}\) and \(2\times 10^{-13}\). Continuation uncertainties are between \(3\times 10^{-7}\) and \(3.5\times 10^{-6}\) in \(K\). Every accepted \(\varepsilon\) has one cluster of the three starts.

One point is outside the branch. At \(p=0.2\), \(\delta=0.20\), \(\varepsilon=10^{-2}\), \(D\) has no root with pinned neighbor mass at most \(10^{-8}\). The search reaches the boundary \(K=2.920\) with \(|D|\approx 8\times 10^{-3}\). The four smaller values of \(\varepsilon\) are interior, monotone, and are the points used in the extrapolation. At the extrapolated root the exact pinned mass is \(4.2\times 10^{-10}\).

## Comparison with \(pK\) scaling

\[
K_H^{\mathrm{scaling}}(p,\delta)=\frac{K_H^{\mathrm{Eq.(6)}}(\delta)}{p}.
\]

The homogeneous annealed boundary, with the same mean neighbor degree and no spread, is

\[
K_H^{\mathrm{hom}}=K_H^{\mathrm{Eq.(6)}}\,\frac{N}{\mathbb{E}_q[k]}.
\]

Its relative offset from \(pK\) scaling is exactly

\[
\frac{K_H^{\mathrm{hom}}-K_H^{\mathrm{scaling}}}{K_H^{\mathrm{scaling}}}=\frac{2p-1}{1+(N-2)p}.
\]

This equals \(1/(N-1)\) only at \(p=1\). It is \(O(1/N)\) and is present even if every neighbor has the same degree. The degree-spread correction is the separate ratio

\[
\frac{K_H^{\mathrm{degree}}-K_H^{\mathrm{hom}}}{K_H^{\mathrm{hom}}}.
\]

All 15 cases are `valid_all_drifting` and `converged`.

| \(p\) | \(\delta\) | \(K_H^{\mathrm{scaling}}\) | \(K_H^{\mathrm{degree}}\) | relative deviation | spread correction | \(\Omega_H\) |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.05 | 0.127852 | 0.127980 | \(+1.001\times 10^{-3}\) | \(+4.9\times 10^{-7}\) | 0.990511 |
| 1.0 | 0.10 | 0.259205 | 0.259464 | \(+1.001\times 10^{-3}\) | \(-2.5\times 10^{-8}\) | 0.960259 |
| 1.0 | 0.20 | 0.569049 | 0.569620 | \(+1.002\times 10^{-3}\) | \(+7.3\times 10^{-7}\) | 0.785837 |
| 0.8 | 0.05 | 0.159815 | 0.159936 | \(+7.54\times 10^{-4}\) | \(+3.6\times 10^{-6}\) | 0.990505 |
| 0.8 | 0.10 | 0.324006 | 0.324254 | \(+7.64\times 10^{-4}\) | \(+1.4\times 10^{-5}\) | 0.960233 |
| 0.8 | 0.20 | 0.711312 | 0.711927 | \(+8.65\times 10^{-4}\) | \(+1.14\times 10^{-4}\) | 0.785681 |
| 0.6 | 0.05 | 0.213087 | 0.213160 | \(+3.42\times 10^{-4}\) | \(+8.8\times 10^{-6}\) | 0.990494 |
| 0.6 | 0.10 | 0.432008 | 0.432168 | \(+3.70\times 10^{-4}\) | \(+3.7\times 10^{-5}\) | 0.960190 |
| 0.6 | 0.20 | 0.948416 | 0.949019 | \(+6.36\times 10^{-4}\) | \(+3.03\times 10^{-4}\) | 0.785423 |
| 0.4 | 0.05 | 0.319630 | 0.319477 | \(-4.80\times 10^{-4}\) | \(+1.9\times 10^{-5}\) | 0.990474 |
| 0.4 | 0.10 | 0.648012 | 0.647742 | \(-4.17\times 10^{-4}\) | \(+8.2\times 10^{-5}\) | 0.960105 |
| 0.4 | 0.20 | 1.422624 | 1.422880 | \(+1.80\times 10^{-4}\) | \(+6.80\times 10^{-4}\) | 0.784905 |
| 0.2 | 0.05 | 0.639260 | 0.637381 | \(-2.94\times 10^{-3}\) | \(+5.0\times 10^{-5}\) | 0.990413 |
| 0.2 | 0.10 | 1.296025 | 1.292431 | \(-2.77\times 10^{-3}\) | \(+2.19\times 10^{-4}\) | 0.959850 |
| 0.2 | 0.20 | 2.845247 | 2.841882 | \(-1.18\times 10^{-3}\) | \(+1.81\times 10^{-3}\) | 0.783350 |

The largest relative deviation from \(K_H(p=1)/p\) is 0.294%, at \(p=0.2\), \(\delta=0.05\). Almost all of it is the finite-\(N\) mean-degree factor, \(-0.299\%\) at this \(p\). The largest correction coming from the spread of \(q(k)\) is 0.181%, at \(p=0.2\), \(\delta=0.20\). At \(p=1\) that spread correction is at the \(10^{-7}\) level of the Step 1 residual.

The spread correction increases as \(p\) decreases and as \(\delta\) increases. The signed deviation from \(K_H(p=1)/p\) does not: the finite-\(N\) factor \((2p-1)/(1+(N-2)p)\) changes sign at \(p=1/2\). On the right-hand panel of `er_annealed_hopf.png` the dashed curve is that factor and the markers are the degree theory. The vertical gap between a marker and the curve is the spread correction.

For this ER law the degree coefficient of variation is \(\sqrt{(1-p)/((N-1)p)}\), about 6.3% at \(p=0.2\). A correction of a few tenths of a percent is the size expected from a smooth function of that narrow distribution. Degree heterogeneity in the annealed theory does not move the Hopf boundary by a significant amount relative to \(pK\) scaling.

## What this does and does not establish

Reliable for the annealed all-drifting theory at these 15 points:

- the characteristic equation, including the \(K/N\) normalization, the weight \(q(k)\), and the split between \(a_k\) and \(K\);
- recovery of Eq. (6) at \(p=1\), with the explicit finite-\(N\) factor \(N/(N-1)\);
- the tabulated roots, residuals, continuation uncertainties, tail tests, and Fourier tests;
- the conclusion that the degree-spread correction is at most 0.181% here, so this approximation nearly coincides with \(pK\) scaling once the \(O(1/N)\) mean-degree factor is removed.

Not established:

- a nonlinear Hopf theorem, for the annealed equation or for a finite network;
- validity on the pinned/drifting mixed branch;
- the Hopf boundary of a quenched ER realization. The scalar \(H_k\) is an average over neighbor degrees. It does not transfer to one adjacency matrix at \(p<1\).

Because the annealed degree correction is this small, it cannot account for a substantially larger shift of a simulated ER boundary away from \(pK\) scaling. Quenched network structure is a separate question and is the appropriate next test.
