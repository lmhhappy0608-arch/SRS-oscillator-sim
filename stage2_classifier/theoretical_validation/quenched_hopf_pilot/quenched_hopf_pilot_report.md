# Quenched ER pilot: collective pulse near the Hopf prediction

Question: does a finite quenched ER network show an oscillatory instability near the theoretical all-drifting Hopf boundary, without using the five-state classifier?

This is finite-network dynamical evidence for or against that prediction. It is not a measurement of a quenched Hopf eigenvalue. No critical coupling is fitted.

## What a finite network does and does not have

The continuum Hopf boundary is a discrete eigenvalue of the stationary incoherent density. On the all-drifting branch that density is constant in time and has collective cosine \(C=\langle\cos\theta\rangle=0\).

A network of \(N=1000\) oscillators with a fixed adjacency and a fixed frequency grid does not have this density as a fixed point. If every oscillator is still drifting, the corresponding object is a quasiperiodic, or weakly irregular, trajectory whose collective pulse fluctuates. Those fluctuations are \(O(N^{-1/2})\) when the oscillators are incoherent, and they can be larger when a weakly damped collective mode is continuously excited. A spectral line near the theoretical frequency \(\Omega_H\) is therefore not, by itself, evidence of instability. The \(p=1\), \(\delta=0.05\) runs show that line on both sides of the boundary. The amplitude does not.

The reference initial condition used here is the product of Adler densities, \(\dot\theta=\omega_i-a_i\sin\theta\), with \(a_i=Kk_i/N\). That is the continuum profile only when \(a_i<\omega_i\) for every \(i\) and the pulse equals 1. Every run in this pilot had zero pinned oscillators. The largest local coefficient was \(0.766\), below the lowest frequency \(0.80\) on the \(\delta=0.20\) grid. The construction is still an approximation to a quenched stationary state, not the state itself.

The archived Stage 1 trajectories cannot be reused. They use one network seed, one uniform initial phase, a \(K\) step of \(0.05\), and eight phase snapshots from \(t=0\) to \(14000\). Their order-parameter series is empty. The nearest stored couplings are not the \(\pm 2\%\) grid around \(K_H\).

## Setup

The ODE is the production Winfree model, integrated with the existing RK4 step at \(dt=0.1\):

\[
\dot\theta_i=\omega_i-\frac{K}{N}\sin\theta_i\sum_j A_{ij}(1+\cos\theta_j).
\]

\(A\) is the undirected ER graph with a zero diagonal. \(A\) is held fixed across \(K\). Frequencies are the validated quantile grid \(\omega_i=\mathrm{linspace}(1-\delta,1+\delta,N)\), not an i.i.d. uniform draw. That grid is the \(N\)-point discretization of the uniform density used in Steps 1 and 2. `frequency_seed` is blank and `frequency_rule` records the grid. An i.i.d. frequency sample would add a second quenched disorder that those steps did not contain.

\(N=1000\), \(T=1200\), samples of \(C(t)\) every \(0.2\). The scan is centered on the degree-based boundary \(K_H^{\mathrm{degree}}\) from Step 2, at relative offsets \(-0.10,-0.05,-0.02,0,+0.02,+0.05,+0.10\). The \(pK\) boundary \(K_H(p=1)/p\) differs from \(K_H^{\mathrm{degree}}\) by at most \(0.3\%\) on this set, which is smaller than the finest offset, so the two predictions are the same vertical line in the scan figure. Three phase realizations are used at \(p=1\), where the graph is the complete graph without self-loops. Three network realizations are used at \(p=0.8,0.4,0.2\). Seeds are SHA-256 derivations of `quenched-hopf-pilot-v1` and are stored in the CSV.

The observable is \(C(t)=N^{-1}\sum_i\cos\theta_i(t)\). The mean of the network fields \(H_i(t)=N^{-1}\sum_j A_{ij}(1+\cos\theta_j)\) was recorded as a check. For an undirected graph the degree-weighted pulse \(B(t)=\sum_k q(k)G_k(t)\) is exactly proportional to the average of \(H_i\), so it is not an independent diagnostic and was not given a separate column.

A halved step, \(dt=0.05\), changes \(C(t)\) over \(t\in[0,40]\) by at most \(2.3\times 10^{-8}\) at the \(p=1\), \(\delta=0.05\), \(+10\%\) point and by at most \(1.4\times 10^{-6}\) at the stiffest point, \(p=0.2\), \(\delta=0.20\), \(+10\%\). The late-window RMS and the last-quarter RMS differ by less than \(50\%\) on every incoherent-product run.

## How a run is labeled

Labels were frozen from one \(p=1\) probe before the three-realization scan. For each incoherent-product realization, compare the demeaned RMS of \(C\) on \([600,1200]\) for \(K<K_H\) with the same RMS for \(K>K_H\).

- The realization separates if every above-boundary RMS exceeds every below-boundary RMS and the ratio of the two medians is at least 3.
- \(p=1\) is `confirmed` only when all three realizations separate and the spectral peak at the \(+2\%\) point lies within \(0.06\) of \(\Omega_H\). This confirms the diagnostic against the known continuum boundary. It does not produce a new critical value.
- \(p<1\) is `suggestive` when at least two realizations separate. It is never `confirmed`.
- Otherwise the group is `unresolved`.

A pure imaginary root, a classifier label, and a large \(C(t)\) without this amplitude jump are not used.

## p = 1 control

Both \(\delta=0.05\) and \(\delta=0.20\) are `confirmed`. All three realizations separate.

At \(\delta=0.05\) the below-boundary late RMS stays in about \(0.04\)–\(0.11\), including a narrow spectral peak within \(0.01\) of \(\Omega_H\approx 0.991\). Crossing the boundary raises the late RMS to about \(0.46\)–\(0.61\). The \(+2\%\) peak remains within \(0.004\) of \(\Omega_H\). The same saturated RMS, \(0.614\), is reached from a uniform initial phase.

At \(\delta=0.20\) the below-boundary RMS is about \(0.02\)–\(0.04\) and the spectrum is not concentrated at \(\Omega_H\). Above the boundary the RMS is about \(0.45\)–\(0.50\). The peak sits \(0.032\) below \(\Omega_H\) at \(+2\%\) and \(0.095\) below it at \(+10\%\), with a harmonic near twice that frequency. The late mean of \(C\) also shifts by about \(+0.13\). The oscillation is real, and it is no longer the linear eigenfrequency once the amplitude is large. A uniform initial phase reaches the same RMS, \(0.496\) against \(0.495\).

The change in amplitude lies between the \(-2\%\) and \(+2\%\) samples. No finer critical coupling is reported.

## Quenched networks

Every \((p,\delta)\) group at \(p=0.8,0.4,0.2\) is `suggestive`. In seven of the eight groups, all three networks meet the factor-of-3 separation. The exception is \(p=0.4\), \(\delta=0.20\), realization 0: the ranges still do not overlap, but the median ratio is \(2.50\) because the \(-2\%\) amplitude is already \(0.23\). The pre-registered factor of 3 fails for that one network, so its rows are marked `suggestive` rather than `collective_oscillation`.

The shape of the scan is the same as \(p=1\). At \(\delta=0.05\) the late RMS stays below about \(0.16\) for \(K<K_H\) and reaches about \(0.39\)–\(0.61\) above it. At \(\delta=0.20\) the subthreshold floor is higher and noisier on the sparse graphs, about \(0.07\)–\(0.18\), and the saturated RMS is about \(0.42\)–\(0.50\). The \(+2\%\) frequency offset is \(+0.004\) at \(\delta=0.05\) for every \(p\), and about \(-0.02\) to \(-0.04\) at \(\delta=0.20\). The \(+10\%\) offset is about \(-0.09\) to \(-0.10\), again shared by \(p=1\) and the ER graphs. Degree heterogeneity and quenched edges do not move the collective frequency outside the nonlinear shift already present at \(p=1\).

Uniform initial phases at \(\pm 10\%\), one network per \((p,\delta)\), reproduce the incoherent-product amplitude to three digits on the high side. The high-amplitude oscillation is an attractor of these finite systems, not an artifact of the Adler initial condition.

## Perturbation pairs

For the same \(p=1\) network, frequencies, and \(K\), a second initial phase was the Adler sample plus \(0.05\sin\theta\). Above the boundary the two trajectories reach the same late RMS. Their difference does not measure a growth rate. At \(\delta=0.05\), \(+10\%\), the late RMS of \(C_1-C_2\) stays near \(0.31\) while each signal has RMS \(0.61\): the residual is a phase offset on a common oscillation. At \(\delta=0.20\), \(+10\%\), that difference falls to \(0.08\) and is flat from the first \(200\) time units to the late window, so the two trajectories have approached the same phase of the attractor. Below the boundary the difference stays at the fluctuation level, \(0.03\)–\(0.05\).

The same pattern appears for one \(p=0.2\), \(\delta=0.20\) pair: the perturbed and unperturbed late RMS values are \(0.444\) and \(0.444\). A distance between two finite-\(N\) trajectories is not a Hopf eigenvalue of the incoherent density. No growth rate is reported.

## What this does not identify

The classifier transitions in the earlier \(p=1\) audits are changes in rotation-number labels: incoherence, locking, partial death, and death. Those labels can track the small-\(\delta\) Hopf truncation, the incoherent-branch edge \(K=1-\delta\), or the death boundary, depending on \(\delta\). The collective-pulse amplitude in this pilot is a different object, and it moves at the Eq. (6) / degree-theory location. A mismatch between a classifier boundary and the Hopf curve can therefore be a comparison of two different dynamical boundaries. This pilot does not recompute classifier labels and does not measure that mismatch again.

Not established: a quenched linear eigenvalue, a precise finite-network critical \(K\), stability of the mixed pinned/drifting branch, or the effect of replacing the frequency grid by i.i.d. uniform draws.

## Next check

The amplitude evidence is strong enough to justify a linear-response calculation on a quenched network, and it is not strong enough to replace that calculation. The smallest useful step is one \(p=1\) system and one \(p=0.2\) network, at a single \(K\) a few percent below and above \(K_H\). Project a small collective cosine perturbation off a reference trajectory and fit the difference only while it remains small. Do not use the late-time distance of two saturated trajectories. An i.i.d. frequency draw is a separate, optional control.
