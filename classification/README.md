# Classification experiments

Observation tools for the five-state problem. These scripts **do not replace** `classify_state()` in `code/plot_rotation_vs_frequency.py`.

This page is the run guide for **estimating the unknown lock center \(c\)** and reading the current-window noise from existing \((\omega,\rho)\) trials.

## Requirements

- Python 3 with NumPy and Matplotlib
- Raw trial files in `raw_trials/` at the repository root (not in git). Filename example:
  `raw_trials/K_0.70_delta_0.20_p_0.80_trial_00.npz`

Run every command below from `classification/`.

```bash
cd classification
```

## Estimate \(c\) and noise

```bash
python3 code/observe_plateaus.py
```

This is the default gallery: ten representative cells at \(p=0.8\), trial 0 (pure death / lock / partial death / partial lock / incoherence). It writes

| Output | Path |
|---|---|
| Per-trial \(\rho(\omega)\) + histogram | `figures/plateau_observe/rho_hist_K_*_delta_*_p_*_trial_*.png` |
| Combined gallery | `figures/plateau_observe/plateau_gallery_p_0.80.png` |
| Table of \(c\), \(\sigma\), \(f\) | `tables/plateau_observe_gallery.csv` |

One cell only:

```bash
python3 code/observe_plateaus.py --k 0.7 --delta 0.20 --p 0.8 --trial 0
```

Several \(K\) or \(\delta\) values (Cartesian product):

```bash
python3 code/observe_plateaus.py --k 0.4 0.7 --delta 0.08 0.20 --p 0.8 --trial 0
```

Optional paths: `--raw-dir`, `--figure-dir`, `--table-dir`.

## How \(c\) is estimated

Operational estimator for inspection, not a theorem from Ariaratnam–Strogatz or Ermentrout–Kopell, and not a new class label.

1. Use raw \(\rho\) (no cleaning). Sort in \(\rho\), not in \(\omega\).
2. Break the sorted list where a consecutive gap exceeds
   \[
   \varepsilon = \tfrac12\,\frac{\omega_{\max}-\omega_{\min}}{N-1}.
   \]
3. Drop blocks smaller than 5% of \(N\) (display cutoff).
4. For each remaining block:
   - \(c=\) median of \(\rho\)
   - \(\sigma=\) standard deviation of \(\rho\) (noise at the current \(T\))
   - \(f=\) fraction of oscillators in the block

The dashed line on the figure is this median. A consecutive run in \(\omega\) is recorded afterwards; it does not decide whether a cluster is kept.

On a clean lock trial there is one pile and one \(c\). On \((K,\delta,p)=(0.70,0.20,0.8)\) the same visual band can be split into two nearby medians (about \(0.769\) and \(0.773\)) because small gaps in \(\rho\) exceed \(\varepsilon\). Read that figure as one band near \(c\approx 0.77\), not two lock speeds.

## How to read the figures

Left: \(\rho\) vs \(\omega\). Gray line is \(\rho=\omega\). Dashed line is estimated \(c\).

Right: histogram of \(\rho\). A sharp peak is a horizontal pile; a broad cloud with no peak is incoherence.

Figure titles such as `death` / `lock` are a gallery menu only. They are not new classification labels.

## Other scripts (not the current \(c\) estimator)

| File | Role |
|---|---|
| `code/fit_rotation_mixture.py` | Three-line mixture (death / lock / drift) |
| `code/analyze_mixture_residuals.py` | Residual diagnostics |
| `code/plot_weight_simplex.py` | Mixture-weight simplex |

```bash
python3 code/analyze_mixture_residuals.py
python3 code/analyze_mixture_residuals.py --p 0.2 0.4 0.6 0.8 1.0
```
