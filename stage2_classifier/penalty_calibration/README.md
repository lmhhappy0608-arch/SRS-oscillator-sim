# Stage 2C — Method C penalty separation and calibration

This directory separates the original Method C penalty into a segment-count
coefficient `alpha` and an extra linear-model coefficient `beta`, checks that
the split matches the original objective, and calibrates the pair on synthetic
rotation-number profiles. Real Stage 1 profiles are used only as diagnostics.
No five-state label is assigned.

## What was reused

Original Method C is unchanged:

`/Users/mohan/Desktop/SRS/data_examine/experiments/plateau_detection_pilot/pilot.py`

SHA-256 `fbe18c40e05a495352aa3552be7fcc7c28228012d7a0d58e1da3a5d403ed13c1`.

`segment_profile` is not edited. The separated mode calls the existing
`cost_audit.dp_penalized` recurrence with caller-supplied penalties. The
wrapper is `method_c_separated.py`. Stage 2B outputs under
`stage2_classifier/method_c_diagnostics/` are not overwritten.

## Objectives

Original, as implemented in `segment_profile`:

```
J_old = sum_G [ RSS_m(G) + tau * sigma_hat^2 * ln(N) * (1 + p_m) ]
```

`p_m = 1` for a constant segment and `p_m = 2` for a linear segment. So a
constant segment is charged `2 * tau * unit` and a linear segment is charged
`3 * tau * unit`, with `unit = sigma_hat^2 * ln(N)`.

Separated:

```
J_new = sum_G RSS_m(G) + sigma_hat^2 * ln(N) * (alpha * M + beta * L)
```

`M` is the number of segments and `L` is the number of linear segments.
`alpha > 0` and `beta >= 0`. A constant segment costs `alpha * unit` and a
linear segment costs `(alpha + beta) * unit`.

These are the same optimization when `alpha = 2 * tau` and `beta = tau`,
including the same `sigma_hat`, minimum segment length, frequency order, and
tie-break. Equal model costs keep the constant model (`cost1 < cost0` is
strict). Equal path costs keep the earliest start (`np.argmin`).

`sigma_hat = 1.4826 * MAD(second difference of rho) / sqrt(6)`, floored at
`1e-12`. Minimum segment length stays 4 and was not retuned. Constant pieces
use the sample mean. Linear pieces use ordinary least squares. There is no
smoothing and no extra acceptance threshold inside the DP.

Penalties live on the scale of `sigma_hat^2 * ln(N)`. A larger `alpha` is a
larger per-segment charge only after that scale is fixed. Do not compare a
raw `alpha` with a raw `tau` without the map above.

## Equivalence

The run compared original `tau in {0.5, 1, 2}` with `alpha = 2*tau`,
`beta = tau` on a noiseless linear–constant–linear profile, a slope-0.30
Gaussian profile, and W3 of C15, C23, C01, C21, and C30 (11 comparisons).
Segment boundaries, segment types, fitted values, and objective values
matched, with absolute gaps of 0 on these cases. Calibration continued only
because that check passed.

## Synthetic benchmark

Omega is `linspace(0.8, 1.2, 1000)`. Pieces are continuous at junctions.
Master seed is `20261101` (`numpy` `SeedSequence`). Calibration seeds are
`0,1,2,3`. Validation seeds are `10,11,12`. Noiseless profiles use seed 0
once. The same noisy realization is never in both splits.

Calibration structures, as `(length, slope)` pieces:

- `lcl_long`, `lcl_medium`, `lcl_short`: linear–constant–linear, plateau
  lengths 400, 120, and 40.
- `clc`: constant–linear–constant.
- `two_plateaus`, `three_plateaus`.
- `pure_flat`.
- `slope_0p01`, `slope_0p05`, `slope_0p30`, `slope_1`.
- `short_shallow`: a slope-0.30 piece of length 80 between steeper slopes.

Validation-only structures: `lcl_len180`, `lcl_right`, `slope_0p15`,
`slope_0p50`, `lcl_len25` (plateau length 25), `lcl_long_ar1`,
`slope_0p30_ar1`, and `mismatch_quad_plateau`.

Noise levels are `0`, `0.002`, `0.01`, and `0.03`. These scales are
referenced to Stage 2B `sigma_hat` values (about 0.013 on C15 and 0.020 on
C23). They are not an estimate of the ODE's noise model, and real profile
variation is not treated as independent noise. Gaussian draws are iid.
Validation also includes stationary AR(1) noise with `phi = 0.9`, with the
innovation scaled so the marginal standard deviation equals the listed sigma,
and one quadratic background with an embedded constant. The quadratic case
is marked `model_mismatch` and is excluded from boundary scores.

Detectability uses an OLS slope t-statistic with the generating marginal
sigma. `t >= 3` is tagged distinguishable. `t < 3` is tagged unresolved.
A nonzero slope at sigma 0 is distinguishable. A zero slope is a plateau.
The AR(1) t-statistic ignores correlation and is tagged as such. Unresolved
oscillators predicted constant are neither true positives nor false
positives. Pure-unresolved profiles have `score_included = 0`.

234 profiles: 156 calibration and 78 validation. One fit took about 0.015 s.
The 25-cell grid was about 5850 fits and was not reduced.

## Metrics and score

Constant precision, recall, and F1 are oscillator-level. A true positive is
predicted constant on a true plateau. A false positive is predicted constant
on a distinguishable linear oscillator. A false negative is predicted linear
on a true plateau. False-constant coverage is `FP / N`. Missed-plateau
coverage is `FN / n_plateau` (0 when there is no plateau). Segment-count
error is segment-level. Boundary error is the mean symmetric nearest-cut
distance in oscillators; an empty-versus-nonempty cut set uses the full
span. Model-mismatch rows do not enter boundary metrics.

The composite score was fixed before the grid was read:

```
0.30 * F1
+ 0.20 * (1 - false_constant_coverage)
+ 0.15 * (1 - missed_plateau_coverage)
+ 0.15 * type_accuracy
+ 0.10 * (1 - count_penalty)
+ 0.10 * (1 - boundary_penalty)
```

Count penalty saturates at `|predicted_n - true_n| / max(true_n, 1)`.
Boundary penalty saturates at mean cut distance / 20 oscillators.
False-constant and missed-plateau flags use coverage 0.05 and, for plateaus
of at least 20 oscillators, missed fraction 0.50. Over-segmentation means
`predicted_n >= true_n + 2`.

Tie-break among cells: higher score, then lower false-constant coverage,
then lower missed-plateau coverage, then smaller grid distance to
`(alpha=2, beta=1)`. The robust set is cells whose score is within 0.02 of
the selected cell and whose false-constant coverage is within +0.02 of that
cell. The grid is extended once, by a factor of two, only if the winner sits
on the outer edge. This run did not extend.

## Stopping rule

A candidate default is frozen only if all of the following hold: equivalence
passed; easy plateaus (`lcl_long`, `lcl_medium`, `clc`, `two_plateaus`,
`three_plateaus`, `pure_flat`) at sigma `<= 0.01` have F1 at least 0.80 and
missed coverage at most 0.25; selected false-constant coverage is at most
0.15; the validation score drops by at most 0.15; fewer than two
single-metric winners lie outside the one-step neighborhood; the selected
cell is not on the outer boundary. Otherwise the status string is
`Calibration inconclusive`. Weights were not changed after seeing the grid.
C15 and C23 were not used to choose `alpha` or `beta`.

## This run

Status: `Calibration inconclusive`. Defaults are not frozen.

Selected cell on 152 score-included calibration profiles: `alpha=2`,
`beta=1` (original Method C at `tau=1`). Score 0.893, constant F1 0.879,
false-constant coverage 0.019, missed-plateau coverage 0.105. Easy-plateau
F1 was 0.952. The cell is interior.

The robust set is one connected component of 7 cells:
`(2, 0.5)`, `(2, 1)`, `(4, 0.5)`, `(4, 1)`, `(4, 2)`, `(8, 1)`, `(8, 2)`.
Scores are 0.873–0.893. This is a tested candidate set on this grid, not a
probability of a dynamical state, and not a support fraction.

Validation score at the selected cell is 0.740, a drop of 0.152. The drop
is concentrated in AR(1) profiles (mean score about 0.40), plateaus of
length 25 at sigma `>= 0.01` (mean about 0.43), and a few noiseless rows
whose `sigma_hat` hits the `1e-12` floor so the penalty underflows relative
to accumulated path cost. Gaussian holdouts with ordinary plateau lengths
and sigma `> 0`, excluding `lcl_len25`, have mean score about 0.95. That
split is descriptive. It does not change the pre-specified status.

Low `alpha`, especially with large `beta`, produces false constants. The
worst calibration cell is `(0.5, 4)`, with false-constant coverage about
0.60 and over-segmentation rate about 0.93. That cell also wins the
missed-plateau-only metric, which is why single-metric agreement fails.
Lowest false-constant coverage is `(8, 0.25)`, outside the robust set.

## Real profiles

Sixteen representative cases from Stage 2A. None entered the score.
Diagnostic: C15, C23. Examined: C01, C02, C05, C09, C13, C17, C22, C25,
C26, C30. Held out from even this diagnostic reading's role label: C07,
C20, C21, C28. All sixteen were still segmented so the table is complete.
The task text's C23 id was missing one character; the completed
representative run is `69ef3196fff7d8f950eb2eedad86c537`.

At `alpha=2`, `beta=1`, W3 reproduces Stage 2B at `tau=1`: C15 has 14
segments and 7 constants, none of them inside omega `[0.93, 1.04]`; C23 has
3 linear segments and 0 constants. Constant segments are fit pieces, not
locking groups. See `real_profile_diagnostics.csv` for window mismatch,
constant Jaccard, and coverage differences. Same coverage does not mean the
same oscillators.

## Rerun

From `/Users/mohan/Desktop/SRS/helper`:

```
python3 stage2_classifier/penalty_calibration/run_calibration.py
```

The script reads Stage 1 profiles through `load_profiles`. It does not run
the Winfree ODE, write the SQLite index, or edit `pilot.py`.

## Outputs

| File | Contents |
| --- | --- |
| `config.json` | Grid, seeds, case roles, method hash |
| `candidate_parameters.json` | Equivalence, selected cell, robust set, checks |
| `synthetic_benchmark_summary.csv` | One row per synthetic profile |
| `penalty_grid_results.csv` | Calibration profile-by-cell metrics |
| `validation_results.csv` | Held-out profile-by-cell metrics |
| `real_profile_diagnostics.csv` | 16 real cases, full grid, three windows |
| `figures/fig1_calibration_heatmaps.png` | F1, false-constant, missed-plateau |
| `figures/fig2_synthetic_examples.png` | Success and failure examples |
| `figures/fig3_real_comparison.png` | Original-equivalent, selected, high false-constant |
| `figures/fig4_parameter_robustness.png` | Score region and slices |
