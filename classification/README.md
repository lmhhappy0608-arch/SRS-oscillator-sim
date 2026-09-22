# Classification experiments

Observation and trial tools for the five-state classification. These scripts **do not replace** the old `classify_state()` in `code/plot_rotation_vs_frequency.py`.

## Scripts

| File | Role |
|---|---|
| `code/observe_plateaus.py` | Current: estimate \(c\), \(\sigma\), \(f\) from raw \(\rho\) |
| `code/fit_rotation_mixture.py` | Three-line mixture (death / lock / drift) |
| `code/analyze_mixture_residuals.py` | Residual diagnostics |
| `code/plot_weight_simplex.py` | Mixture-weight simplex |

Run from `classification/`:

```bash
python3 code/observe_plateaus.py
```

Raw solutions are still read from `raw_trials/` at the repository root.
