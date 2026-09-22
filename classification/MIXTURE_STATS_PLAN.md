# Statistical refinement of the three-component mixture

At the oscillator level there are still only three geometric lines: death \(\rho=0\), one horizontal lock \(\rho=c\), and drift \(\rho\approx\omega\). Statistics should answer two questions: where on the \((K,\delta)\) grid those three lines fail, and how to separate Death / Partial / Locking from the weights without returning to a 0.95 cutoff.

When a mixture weight is zero the parameters are not identifiable, so \(2(\ell_1-\ell_0)\) is not a standard \(\chi^2\). Compare 2 / 3 / 4 lines with BIC or a bootstrap. Do not report an ordinary likelihood-ratio \(p\)-value.

## Suggested order

| Step | Do | Look at | Used to refine |
| --- | --- | --- | --- |
| 1 | Per cell: RMSE, max residual, residual shape vs \(\omega\) | Whether high residuals form a band | Where three lines are not enough |
| 2 | Plot weights \((w_d,w_l,w_r)\) on the simplex / entropy \(H(w)\) | 3 clumps vs 5 | Whether Partial is its own cluster or a continuum |
| 3 | GMM / k-means on the weights only; choose \(k\) by silhouette | Whether centers sit on the three vertices and two edges | Data-driven split of the five macroscopic states |
| 4 | Add a model only on high-residual cells: a second horizontal line, or a free-slope branch | Keep if \(\Delta\mathrm{BIC}<0\) | Slipping, double plateaus, bent branches |
| 5 | Across \(p=0.2,\ldots,1.0\), compare locked area and mixing bandwidth | Whether they change monotonically | How density reshapes the phase diagram |

## Reading macroscopic states from the weights

A coarse phase has one weight near 1. Death / Locking / Incoherence are the three vertices of the simplex. There is no need to set 0.95; cluster in weight space, or require the largest weight to stand clearly above the other two.

Partial states sit on an edge. Partial Death \(\approx\) the death–drift edge (\(w_{\mathrm{lock}}\approx 0\), the other two both visible). Partial Locking \(\approx\) the lock–drift edge. The degree of mixing is the smaller of those two weights, not “at least 15 points.”

## What residuals say

Sort by \(\omega\) and look at the residual shape:

| Residual shape | Likely missed case | Next model |
| --- | --- | --- |
| A horizontal residual plateau | A second locked speed | Add \(\rho=c_2\) |
| A middle stretch off both 0 and \(\omega\) | Slipping / pulled frequencies | Free-slope segment |
| Large at the ends, small in the middle | Missed plateau, or a biased \(c\) | Check `lock_center` |
| Small everywhere, no structure | Three lines suffice | Do not add a component |

When comparing to the old five-color map, do not treat \(\mathrm{argmax}(w)\) as a new five-way label and then build a confusion matrix. First box-plot \((w_d,w_l,w_r)\) by old label: Death should sit at \((1,0,0)\); Partial Death should spread along the death–drift edge. If old Partial Locking does not form a clump in weight space, it is a threshold-made color block, not a fifth phase to keep.

## Layout and first command

Classification code, figures, and tables live under `classification/`:

- Raw solutions: `raw_trials/` at the repo root (not in git)
- Mixture fit: `classification/code/fit_rotation_mixture.py`
- Residual step: `classification/code/analyze_mixture_residuals.py`
- Weight figures: `classification/figures/rotation_mixture/`
- Residual figures: `classification/figures/mixture_residuals/`
- Tables: `classification/tables/`

From `classification/`:

```bash
python3 code/analyze_mixture_residuals.py
python3 code/analyze_mixture_residuals.py --p 0.2 0.4 0.6 0.8 1.0
```

Each cell writes RMSE, max \(|\rho-\hat\rho|\), lag-1 autocorrelation of residuals vs \(\omega\), RMSE on low / mid / high \(\omega\), and residual tilt and curvature. Use the maps to see whether high residuals form a band; use the example plots to see residual shape vs \(\omega\).
