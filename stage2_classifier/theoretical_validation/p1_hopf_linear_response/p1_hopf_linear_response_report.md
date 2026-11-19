# p = 1 linear-response Hopf baseline

Check of the all-drifting linear response against the all-to-all Hopf boundary in Ariaratnam and Strogatz, Phys. Rev. Lett. 86, 4278 (2001). The comparison target is their discrete-spectrum condition, Eq. (6). Eq. (7) is the small-δ truncation of that condition. Classifier labels are not used. The Winfree ODE is not integrated.

## Model and finite-N coupling

The continuum equation in the paper is

\[
\dot\theta_i=\omega_i+\frac{\kappa}{N}\sum_{j=1}^{N}P(\theta_j)R(\theta_i),
\qquad P(\theta)=1+\cos\theta,\quad R(\theta)=-\sin\theta,
\]

with \(g(\omega)=1/(2\gamma)\) on \([1-\gamma,1+\gamma]\). Here \(\gamma=\delta\) and \(\kappa=K\).

The production adjacency (`stage1.model.make_adjacency`) is an undirected ER graph with a zero diagonal. At \(p=1\) every off-diagonal entry is 1. A direct check at \(N=8\) gives a zero diagonal, unit off-diagonal entries, and a symmetric matrix. The ODE is still scaled by \(K/N\), so each oscillator misses the self-pulse that the paper sum includes. The relative factor is \((N-1)/N\). For the \(N=1000\) runs this is 0.999. The Hopf comparison below is against the continuum limit, where that diagonal term vanishes. It is not inserted as a finite-\(N\) shift of \(K\).

## Derivation check

On the all-drifting branch every natural frequency satisfies \(\omega>K\). The stationary density is the paper's Eq. (4),

\[
f^*(\theta\mid\omega)=\frac{\sqrt{\omega^2-K^2}}{2\pi(\omega-K\sin\theta)}.
\]

For \(P=1+\cos\theta\), \(\int_0^{2\pi}\cos\theta\,f^*\,d\theta=0\), so \(\int_0^{2\pi}(1+\cos\theta)\,f^*\,d\theta=1\). The mean field is therefore \(H^*=1\) for any frequency density supported above \(K\), and the self-consistent coefficient is \(a=KH^*=K\). This is why the incoherent branch ends at the paper's Eq. (5), \(K=1-\delta\). Eq. (5) is the edge of the branch, not a second Hopf curve.

Linearizing the continuity equation about this density, and scaling the unknown so that it is the response to a unit perturbation of the mean-field coefficient, gives

\[
\lambda u+\partial_\theta[(\omega-K\sin\theta)u]=\partial_\theta[\sin\theta\,f^*].
\]

The characteristic function is \(D(\lambda;K,\delta)=1-K\chi\), with

\[
\chi=\int g(\omega)\int_0^{2\pi}(1+\cos\theta)\,u(\theta,\omega;\lambda)\,d\theta\,d\omega.
\]

The sign and the factor of \(K\) match the paper: \(D=0\) is Eq. (6). With the Fourier convention \(u=\sum c_n e^{in\theta}\), the recurrence

\[
(\lambda+in\omega)c_n-\frac{nK}{2}(c_{n-1}-c_{n+1})=s_n
\]

is the mode form of that PDE. The forcing coefficients are

\[
s_n=\frac{in\omega\,d_n}{2\pi K}\ (n\neq 0),\qquad
d_n=(-ir)^n\ (n>0),\quad d_n=(ir)^{-n}\ (n<0),
\]

where \(r=K/(\omega+\sqrt{\omega^2-K^2})\). The one-frequency observable reduces exactly to the integrand of Eq. (6) divided by \(K^2\):

\[
\int_0^{2\pi}(1+\cos\theta)\,u\,d\theta
=\frac{\lambda\omega(\omega-\sqrt{\omega^2-K^2})}{K^2(\lambda^2+\omega^2-K^2)}.
\]

Thus \(K\chi=1\) if and only if Eq. (6) holds.

This identity was checked before any boundary value was accepted.

| Check | Result |
| --- | --- |
| FFT of \(\partial_\theta(\sin\theta\,f^*)\) against \(s_n\) | absolute error \(3\times 10^{-17}\) |
| \(K\to 0\) observable against \(\lambda/(2(\lambda^2+\omega^2))\) | relative error \(2\times 10^{-12}\) |
| Fourier–Galerkin (\(M=40\)) against the Eq. (6) integrand, two finite-\(K\) samples | relative error \(6\times 10^{-16}\) and \(9\times 10^{-16}\) |
| Independent second-order collocation of the PDE, Richardson \(N=512,1024\) | relative difference from Galerkin \(2\times 10^{-8}\) and \(8\times 10^{-7}\) |

The collocation difference is the second-order truncation, not a sign or normalization mismatch. No correction to the response equation or to the recurrence was required.

## Continuous spectrum

The paper states that the linearized operator has a purely imaginary continuous spectrum, and a discrete spectrum given by Eq. (6). It also states that Eq. (6) has no root with \(\mathrm{Re}(\lambda)<0\). The Hopf boundary is obtained by solving Eq. (6) and then taking \(\mathrm{Re}(\lambda)\to 0^+\).

A probe \(\lambda=i\Omega\) sits on that continuous spectrum whenever \(\Omega=\sqrt{\omega^2-K^2}\) for some \(\omega\) in \([1-\delta,1+\delta]\). A finite Fourier matrix or a real-frequency quadrature at \(\varepsilon=0\) then mixes a principal-value integral with an uncontrolled pole. Those zeros were not used.

The calculation here solves \(D(\varepsilon+i\Omega;K,\delta)=0\) for \(\varepsilon>0\), where the resolvent is off the continuous spectrum, and extrapolates the root to \(\varepsilon=0\). That is the same right-hand limit used by the paper. The existing solver `validate_eq7.solve_eq6` evaluates the same limit by a Landau pole plus a subtracted principal-value integral. The two implementations are compared below. They are not the same code path.

## Which widths were used

The requested widths were tested against Eq. (5) and Eq. (6) before the response root was accepted. All three lie inside the all-drifting branch, with a wide gap to \(K=1-\delta\). None was replaced.

| \(\delta\) | \(1-\delta\) | Eq. (6) root | Eq. (7) | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0.05 | 0.95 | 0.127852 | 0.127852 | all-drifting Hopf root |
| 0.10 | 0.90 | 0.259205 | 0.259152 | all-drifting Hopf root |
| 0.20 | 0.80 | 0.569049 | 0.554350 | all-drifting Hopf root |
| 0.25 | 0.75 | none | 0.737832 | no root inside the branch |

\(\delta=0.25\) is a negative control. The printed truncation 0.7378 is below \(1-\delta=0.75\), but it is not a root of Eq. (6). The response search found no zero of \(D\) at any tested \(\varepsilon\). On a \(36\times 36\) scan at \(\varepsilon=10^{-2}\), the smallest \(|D|\) was 0.131. The Hopf fields for this width are left empty. The truncation is not used to fill them.

## Numerical method

For each frequency the Fourier system is solved at truncation \(M\), with the zero mode removed because \(\lambda\neq 0\) forces it to vanish. Frequencies are integrated with Gauss panels graded around the complex Landau pole. At a probe with \(\varepsilon=10^{-3}\), that rule agrees with adaptive quadrature of the closed-form integrand to a relative error of \(2\times 10^{-14}\).

Roots are computed at \(\varepsilon\in\{10^{-2},3\times 10^{-3},10^{-3},3\times 10^{-4},10^{-4}\}\) by a bounded nonlinear least-squares solve for \((K,\Omega)\). Several initial guesses, including the Eq. (7) value and the valley of \(|D|\), collapse to one root. \(K(\varepsilon)\) and \(\Omega(\varepsilon)\) are extrapolated to \(\varepsilon=0\) with a quadratic fit in \(\varepsilon\). The reported uncertainty is the difference between that intercept and the linear intercept through the two smallest \(\varepsilon\). A root is kept only when the residual is below \(10^{-7}\), the intercept lies at least \(10^{-4}\) below \(1-\delta\), the fit deviation is below \(10^{-4}\), and the \(M=24\) and \(M=32\) roots at \(\varepsilon=10^{-3}\) differ by less than \(10^{-5}\).

## Convergence

At \(\varepsilon=10^{-3}\), raising \(M\) from 12 to 32 does not change \(K\) in the first eight digits at any of the three widths. At \(\delta=0.20\), \(M=8\) differs from \(M=32\) by \(8\times 10^{-6}\); from \(M=12\) upward the difference is absorbed by the solver tolerance. The \(M=24\) versus \(M=32\) gap at \(\delta=0.20\) is \(3\times 10^{-15}\).

The continuation itself is smooth. For example at \(\delta=0.05\), \(K\) falls from 0.14620 at \(\varepsilon=10^{-2}\) to 0.12802 at \(\varepsilon=10^{-4}\), and the extrapolated intercept is 0.127852. Residuals on the accepted roots are at most \(2\times 10^{-13}\). Extrapolation uncertainties are \(7\times 10^{-7}\), \(3\times 10^{-7}\), and \(7\times 10^{-7}\) at \(\delta=0.05\), 0.10, and 0.20.

A direct \(\varepsilon=0\) root was not computed, for the spectral reason above.

## Comparison with the paper

`paper_K_H` is the Eq. (6) root from `validate_eq7.solve_eq6`. `response_K_H` is the extrapolated Fourier–Galerkin root. The pass threshold fixed before accepting the table was a relative error below \(10^{-3}\) at every valid width.

| \(\delta\) | Eq. (6) \(K_H\) | Response \(K_H\) | Absolute error | Relative error | Eq. (7) relative to Eq. (6) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.127852095 | 0.127852158 | \(+6.3\times 10^{-8}\) | \(4.9\times 10^{-7}\) | \(2.9\times 10^{-6}\) |
| 0.10 | 0.259204992 | 0.259204985 | \(-6.5\times 10^{-9}\) | \(2.5\times 10^{-8}\) | \(2.0\times 10^{-4}\) |
| 0.20 | 0.569049493 | 0.569049909 | \(+4.2\times 10^{-7}\) | \(7.3\times 10^{-7}\) | \(2.6\times 10^{-2}\) |
| 0.25 |  |  |  |  | no Eq. (6) root |

The corresponding imaginary frequencies also agree. At \(\delta=0.20\) the Eq. (6) frequency is 0.785838 and the response extrapolation is 0.785837.

Eq. (7) tracks Eq. (6) at \(\delta=0.05\) and is already 2.6% low at \(\delta=0.20\). That gap is the \(O(\delta^7)\) truncation. The paper's own statement of the approximation is for small \(\gamma\), with about a 4% band cited for couplings up to 0.6. It is not a failure of the response calculation. The response root follows Eq. (6), not the truncation.

## Verdict

The p = 1 baseline passes inside the all-drifting regime. The response equation and the Fourier recurrence reproduce Eq. (6), and the \(\varepsilon\to 0^+\) root agrees with the existing Eq. (6) solver to a relative error below \(10^{-6}\) at \(\delta=0.05\), 0.10, and 0.20. All three roots satisfy \(K_H<1-\delta\).

The substantive numerical caution is the continuous spectrum. It was handled by the right-hand limit, not by accepting an imaginary-axis matrix eigenvalue. No coefficient in the response formula had to be changed.

## Unresolved points

- The finite-\(N\) missing self-pulse, of relative size \((N-1)/N\), was not measured as a shift of this continuum boundary.
- The scan covers one Hopf pair in the window \(0.25<\Omega<2\). It is not a proof that the right half-plane contains no other discrete eigenvalues. The paper's statement that there is no eigenvalue with negative real part was used as published, not re-derived.
- \(\delta=0.25\) has no all-drifting Hopf root. The same formulas must not be continued onto the mixed pinned/drifting branch.
- A scalar mean field \(H(t)\) is special to all-to-all coupling. Agreement at \(p=1\) does not by itself supply the susceptibility of one Erdős–Rényi realization at \(p<1\).

## Files

- `linear_response_hopf.py` — checks, continuation, CSV, and figure
- `p1_hopf_linear_response.csv` — comparison table; empty fields are unresolved, not guesses
- `p1_hopf_linear_response.png` — Eq. (6), Eq. (7), the response root, and the \(\delta=0.25\) exclusion
- `p1_hopf_linear_response_diagnostics.json` — identity checks, \(\varepsilon\)-continuation, and truncation sweep
