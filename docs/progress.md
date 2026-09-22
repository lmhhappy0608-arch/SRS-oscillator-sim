# Research progress

Chronological log of what was done, what it showed, and what comes next. Append new sections at the end. Do not start a second repository.

## 1. Model reproduction

Winfree dynamics on ER graphs, scanned over coupling \(K\), frequency width \(\delta\), and edge probability \(p\).

- Code: `code/Ariaratnam2001.py`, grid generation in `code/generate_raw_grid.py`
- Parameters: \(N=1000\), typical \(T=10000\), \(T_0=8000\)
- Output: per-\(p\) phase diagrams `winfree_phase_diagram_p_*.png`

Early work used the macroscopic order parameter \(R\). Dense networks and narrow frequencies synchronize more easily. Averaging \(R\) mixes qualitatively different outcomes.

## 2. From \(R\) to rotation number

Each oscillator has rotation number

\[
\rho_i=\frac{\theta_i(T)-\theta_i(T_0)}{T-T_0}.
\]

The profile \(\rho(\omega)\) separates death, locking, and drift. Plots: `code/plot_rotation_vs_frequency.py`.

## 3. Old five-state classification

Rules live in `classify_state()` and **are still in use; they have not been replaced**:

| State | Old rule (summary) |
|---|---|
| Death | Fraction with \(\|\rho\|<0.01\) is \(\ge 0.95\) |
| Partial Death | At least 4 consecutive oscillators near 0 after sorting by \(\omega\) |
| Locking | All \(\rho\) nearly equal, or a nonzero plateau covers \(\ge 0.95\) |
| Partial Locking | A nonzero plateau exists (at least 15 consecutive points, \(\rho\) span \(\le 0.02\)) |
| Incoherence | No such plateau |

The consecutive-count cut, the 0.02 window, and the 0.95 coverage are presets. The lock center \(c\) is unknown a priori; only Death is known to sit at 0.

## 4. Numerical cross-check

C++ RK4 / RK45 vs Python, in `cpp/`. This checks that the rotation numbers are not a solver artifact.

## 5. Three-line mixture (not yet a new classifier)

Each oscillator is compared to three geometric lines: \(\rho=0\) (death), \(\rho=c\) (horizontal lock), \(\rho\approx\omega\) (drift).

- Fit: `classification/code/fit_rotation_mixture.py`
- Residuals: `classification/code/analyze_mixture_residuals.py`
- Plan: `classification/MIXTURE_STATS_PLAN.md`

Each cell gets weights \((w_{\mathrm{death}},w_{\mathrm{lock}},w_{\mathrm{drift}})\). This is an observation tool and **does not change `classify_state()`**.

## 6. Current: observe plateaus; estimate \(c\) and noise

Do not preset “how many consecutive oscillators lie in a window.” Death sits around 0; the lock value \(c\) is unknown; noise is read from the data.

- Code: `classification/code/observe_plateaus.py`
- Figures: `classification/figures/plateau_observe/`
- Table: `classification/tables/plateau_observe_gallery.csv`

Find tight clusters in \(\rho\) (no consecutive-\(\omega\) requirement):

- \(c=\) median of the cluster
- \(\sigma=\) standard deviation of the cluster (noise at the current \(\Delta T=2000\))
- \(f=\) fraction of oscillators in the cluster

Observed scales at \(p=0.8\), trial 0:

- Pure Death: \(c=0\), \(\sigma\sim 10^{-10}\)
- Pure Lock: \(c\) depends on \(K\) (e.g. 0.93 vs 0.76), \(\sigma\sim 10^{-5}\)
- Partial: the same band, but \(f<1\)
- Incoherence: no tight horizontal cluster

The words `death` / `lock` on the figures are a gallery menu, not new class labels.

## 7. Next (not done yet)

- Hold a few trials that already show a plateau and lengthen \(T\); check whether \(\sigma\) shrinks
- Rewrite the plateau definition from those observations, then decide whether to touch the five-state classifier
