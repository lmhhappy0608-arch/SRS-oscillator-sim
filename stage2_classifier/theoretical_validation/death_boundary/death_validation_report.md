# Death existence boundary (Eq. 8–9)

This note compares the Ariaratnam & Strogatz death-equilibrium threshold with the W3 Death labels of Classifier v1 on the saved \(p=1\) profiles. \(K_{\mathrm{exist}}\) is the coupling at which a fully stopped equilibrium begins to exist. It is not the coupling at which every initial condition must enter Death.

## A. Data and methodology

The Stage 1 index has 6964 rows, 6962 of them completed. Of the completed rows, 1741 have \(p=1\). Those 1741 are the profiles used here. All have \(N=1000\) and trial 0. Two index rows are not completed; their status is not used.

\(K\) runs from 0 to 3 in steps of 0.05, and \(\delta\) from 0 to 1 in steps of 0.025, but the rectangle is incomplete. Every \(\delta\) has all couplings \(K\le 2\). Only \(\delta\in\{0.05,0.20,0.80\}\) also have \(K=2.05,\ldots,3\). The other 760 cells are missing and are left blank. They lie above the death onset and do not change it.

| Item | Recorded value |
| --- | --- |
| Model | `winfree_er_v1` |
| Classifier | Archived `v1.0.0-pilot`, `stage2_classifier/classifier_v1/classify.py` and `config.json`. The stored classification tables are not the source: they do not cover this grid. Labels were recomputed in memory with `classify_window`. |
| Near zero | \(\lvert\rho\rvert<0.001\), strict. Death requires this for every oscillator. Priority is Death, Locking, Partial Death, Partial Locking, Incoherence. |
| Window | W3 \(=[12000,14000]\). Rotation number \((\theta(14000)-\theta(12000))/2000\). |
| Solver | RK4, stored `dt=0.1`, \(t\) from 0 to 14000, on all 1741 rows. |
| Randomness | `master_seed=0`, recipe `sha256-v1`, mode `paired`, one shared network seed and one shared initial-phase seed. `frequency_seed` is null. Frequencies are \(\omega_i=1-\delta+2\delta(i-1)/999\). |
| Network | At \(p=1\), the undirected complete graph with a zero diagonal (499500 upper edges). Regenerating it from the shared seed matches the adjacency hash of one stored run. Coupling is \(K/N\), with no self-pulse. |

A continuum interaction normalized by \(N\) rather than \(N-1\) differs by the relative factor \((N-1)/N=0.999\). That difference is not removed by shifting \(K_{\mathrm{exist}}\).

Eq. (8) is evaluated as
\[
F_\delta(\sigma)=1+\frac{1}{2\delta}\int_{1-\delta}^{1+\delta}\sqrt{1-\frac{\omega^2}{\sigma^2}}\,d\omega,
\qquad
K(\sigma,\delta)=\frac{\sigma}{F_\delta(\sigma)},
\]
for \(\sigma\ge 1+\delta\), and \(K_{\mathrm{exist}}=\min K\). The integral uses its antiderivative. Critical points solve \(F-\sigma F'=0\) with `brentq`, in the coordinate \(\sigma=(1+\delta)\cosh u\), which keeps the square root regular at the upper frequency. `minimize_scalar` on the same interval checks that the chosen critical value is the minimum. The search extends to \(80(1+\delta)\); repeating it to \(200(1+\delta)\) at \(\delta=0.2\) changes \(K\) by \(2\times 10^{-16}\). `quad` agrees with the antiderivative to at most \(4\times 10^{-15}\) on five probes. For \(\delta=0\) the factor collapses to \(1+\sqrt{1-1/\sigma^2}\), and the minimizer is the exact value \(4/(3\sqrt{3})=0.769800\). No scanned \(\delta\) is unresolved. The branch is whatever the minimization returns. It is not assigned by a hand-set cutoff.

## B. Theoretical boundary

\(F-\sigma F'\) at \(\sigma=1+\delta\) changes from negative to positive at
\[
\delta_*=0.2956.
\]
Below that value, \(K\) decreases as \(\sigma\) leaves \(1+\delta\) and the minimum is an interior root. That is the saddle-node branch: the death solution appears at \(\sigma_{\mathrm{SN}}>1+\delta\). Above \(\delta_*\), the slope at the endpoint is positive, there is no interior root, and the minimum is \(\sigma=1+\delta\). That is the end-point branch, Eq. (9). Each of the 12 saddle-node widths has exactly one interior root, and that root is a local minimum. Each of the 29 end-point widths has none.

On the scanned grid the split falls between \(\delta=0.275\) (saddle-node, \(K_{\mathrm{exist}}=0.8082\)) and \(\delta=0.300\) (end-point, \(K_{\mathrm{exist}}=0.8155\)). At \(\delta=0\), \(\sigma_{\mathrm{SN}}=2/\sqrt{3}\) and \(K_{\mathrm{exist}}=0.7698\). At \(\delta=1\), \(K_{\mathrm{exist}}=1.1202\) with \(\sigma=2\).

## C. Comparison with Classifier v1

For every \(\delta\), Death occurs, the first Death cell is not the leftmost grid point, and every later cell through the top of that column stays Death. There is no reversal and no left-censored onset. \(K_{\mathrm{observed}}-K_{\mathrm{exist}}\) is positive for all 41 widths: minimum \(+0.0028\), median \(+0.0273\), maximum \(+0.0507\).

For 38 widths, \(K_{\mathrm{exist}}\) lies in the open-closed gap \((K_{\mathrm{previous}}, K_{\mathrm{observed}}]\). The observed onset is then the first grid point above the threshold, and the positive difference is the distance from the threshold up to that grid point. The other three widths are one grid point later; they are the cases in Section D.

| \(\delta\) | Mechanism | \(K_{\mathrm{exist}}\) | \(K_{\mathrm{observed}}\) | Previous state | \(\Delta K\) |
| --- | --- | --- | --- | --- | --- |
| 0.000 | saddle-node | 0.7698 | 0.80 | Locking | +0.030 |
| 0.200 | saddle-node | 0.7902 | 0.80 | Incoherence | +0.010 |
| 0.225 | saddle-node | 0.7956 | 0.80 | Partial Locking | +0.004 |
| 0.275 | saddle-node | 0.8082 | 0.85 | Partial Death | +0.042 |
| 0.300 | end-point | 0.8155 | 0.85 | Partial Death | +0.035 |
| 0.400 | end-point | 0.8495 | 0.90 | Partial Death | +0.051 |
| 0.500 | end-point | 0.8889 | 0.90 | Partial Death | +0.011 |
| 1.000 | end-point | 1.1202 | 1.15 | Partial Death | +0.030 |

Saddle-node widths are \(\delta=0,0.025,\ldots,0.275\) (12 values). For \(\delta\le 0.175\) the cell before Death is Locking, the near-zero fraction there is 0, and Death begins at \(K=0.80\). End-point widths are \(\delta=0.300,\ldots,1\) (29 values). The cell before Death is Partial Death on all 29, and the oscillators with \(\lvert\rho\rvert\ge 0.001\) are exactly the highest-frequency suffix.

## D. Disagreements and interpretation

**The route changes earlier than the existence mechanism.** Locking precedes Death for every \(\delta\le 0.175\). At \(\delta=0.200\), still on the saddle-node branch, the column is Incoherence, Locking at the single point \(K=0.65\), Incoherence again at \(K=0.70\) and \(0.75\) (near-zero fraction 0), then Death at \(K=0.80\). At \(\delta=0.225\) the only Partial Locking cell in the \(p=1\) set sits at \(K=0.75\), again with near-zero fraction 0, and Death follows at \(K=0.80\). At \(\delta=0.250\) and \(0.275\) the predecessor is already Partial Death, while the existence calculation is still saddle-node. From \(\delta=0.300\) onward the predecessor is Partial Death and the existence branch is end-point. Direct evidence is that pair of classifications: the label sequence and the sign of \(F-\sigma F'\) at \(\sigma=1+\delta\). The two switches are about one grid step apart and were not aligned by hand. A saddle-node birth of the death solution does not require the preceding attractor to be the locked state. On this grid the locked predecessor is confined to \(\delta\le 0.175\); between there and \(\delta_*\) the saved state is already leaving locking while the death solution is still born in the interior.

**Three end-point widths are one grid point later than the first point above \(K_{\mathrm{exist}}\).** In each, \(K_{\mathrm{exist}}\) sits just below a grid value, that grid value is not Death, and the next one is.

| \(\delta\) | \(K_{\mathrm{exist}}\) | Not yet Death | Alive oscillators | Their \(\lvert\rho\rvert\) | Death at |
| --- | --- | --- | --- | --- | --- |
| 0.400 | 0.8495 | \(K=0.85\) | 3, highest frequencies, \(\omega\in[1.398,1.400]\) | 0.038, 0.060, 0.078 | 0.90 |
| 0.525 | 0.8993 | \(K=0.90\) | 1, \(\omega=\omega_{\max}=1.525\) | 0.041 | 0.95 |
| 0.750 | 0.9999 | \(K=1.00\) | 2, highest frequencies, \(\omega\in[1.748,1.750]\) | 0.063, 0.097 | 1.05 |

The near-zero fraction on those three cells is 0.997, 0.999, and 0.998, so the population is almost stopped, but the remaining rotation numbers are several times 0.01. They are not an artifact of the strict cut at 0.001. Direct evidence stops there: the W3 averages of the top oscillators are still \(O(10^{-2})\) at a coupling only \(10^{-4}\) to \(10^{-3}\) above the existence threshold, and they are below 0.001 at the next grid point. The calculation says a stopped equilibrium exists at \(K_{\mathrm{exist}}\). It does not say that this initial condition has reached it, or that a frequency essentially equal to \(\sigma\) has a negligible finite-time rotation. Those are open readings. The \(K/N\) normalization is a relative shift of size \(1/N\), about \(10^{-3}\) in \(K\) near this boundary, which is the same order as the undershoot of the grid point and much smaller than the extra 0.05 step. It is too small, by itself, to place the whole delay on the normalization, and this test did not measure a finite-\(N\) correction.

**End-point predecessor fractions are high only when the previous grid point is already above \(K_{\mathrm{exist}}\).** Where the previous point is still below the threshold, a large high-frequency suffix is still rotating. At \(\delta=0.500\), \(K=0.85\) is below \(K_{\mathrm{exist}}=0.8889\), the near-zero fraction is 0.806, and 194 oscillators, all at the high-frequency end, have \(\lvert\rho\rvert\ge 0.001\). Death is the next cell, \(K=0.90\). That pattern matches an existence threshold: below it, \(\omega_{\max}\) cannot be stopped; above the next grid point, the saved profile is entirely within the 0.001 cut. It is not a measurement of the self-consistent phase.

## E. Conclusions

1. On all 41 widths the Classifier v1 Death region begins at the first scanned coupling above \(K_{\mathrm{exist}}\), or one grid point after that. The median difference between the recorded onset and the existence threshold is \(+0.027\), the size expected from reporting the upper edge of a 0.05 bin. No onset is early, and none is late by two grid steps. Once Death appears, it persists. That is agreement at the resolution of the grid, not pointwise identity, and it is agreement about existence versus the observed attractor rather than a proof that the saved state is the paper's equilibrium.

2. The small-\(\delta\) and large-\(\delta\) routes differ. For \(\delta\le 0.175\), on the saddle-node branch, Death is preceded by Locking and the near-zero fraction jumps from 0 to 1. For every end-point width, \(\delta\ge 0.300\), Death is preceded by Partial Death and the oscillators that are still rotating form the high-frequency end. The changeover in the labels occupies \(\delta=0.200\)–\(0.275\), where the existence branch is still saddle-node but the predecessor is Incoherence, Partial Locking, or Partial Death. The existence mechanism itself changes at \(\delta_*=0.296\), between the scanned values 0.275 and 0.300.

3. The only onsets outside the bracketing gap are \(\delta=0.400\), \(0.525\), and \(0.750\). Each misses full Death at a grid point \(10^{-4}\) to \(10^{-3}\) above \(K_{\mathrm{exist}}\) because one to three highest-frequency oscillators still have \(\lvert\rho\rvert\sim 0.04\)–\(0.10\).

4. The usable evidence is the independent match between \(K_{\mathrm{exist}}\) and the first persistent Death label, the split between a locking predecessor at small \(\delta\) and a high-frequency partial-death predecessor at large \(\delta\), and the fact that the thresholds were not adjusted to produce that match. The numerical solution of Eq. (8) is stable under the quadrature, root, and range checks above.

5. The comparison does not establish that the finite-time, finite-\(N\) trajectory is the self-consistent equilibrium, that the death solution is globally attracting, or that one initial phase represents an ensemble. It does not explain why the top oscillators at the three late grid points are still rotating. Locking, Partial Locking, and the location of the incoherent-branch edge are outside this test.

The figure is `death_validation_figure.png`. Gray cells have no saved run. The curves are the existence boundary, divided at \(\delta_*\).
