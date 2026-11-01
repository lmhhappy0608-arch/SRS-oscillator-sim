# Method C diagnostics on two Stage 1 profiles

This directory records how the existing constant–linear segmentation (Method C) behaves on two completed Stage 1 runs. It does not rerun the Winfree ODE, does not scan the 732-run pilot, and does not assign five-state labels.

## Reused implementation

Method C is `segment_profile` and `estimate_sigma` in

`/Users/mohan/Desktop/SRS/data_examine/experiments/plateau_detection_pilot/pilot.py`

The same file is in `worktree-cursor` (identical SHA-256; see `config.json`). The script imports that module. It does not copy the dynamic program.

`pilot.py` imports `find_rho_peaks` at load time. This diagnostic inserts a stub for that name and does not call the rho-clustering classifier. A later script, `penalty_split.py`, separates the changepoint penalty from the slope penalty. That variant is not used here.

## Objective

For a frequency-ordered profile of length `n`, each segment `G` is either a constant (`m = 0`) or a straight line in omega (`m = 1`):

```
RSS_0(G) = sum (rho - mean(rho))^2
RSS_1(G) = residual sum of squares of the OLS line rho ~ omega
objective = sum_G  RSS_m(G) + tau * sigma_hat^2 * ln(n) * (1 + p_m)
```

with `p_0 = 1` and `p_1 = 2`. The leading `1` charges a new segment. `p_m` charges the number of fitted parameters. Both charges share the same `tau`, so the changepoint penalty and the segment-type penalty are not separate inputs. Their difference is one parameter unit:

```
penalty_linear - penalty_constant = tau * sigma_hat^2 * ln(n)
```

Inside the dynamic program a block becomes linear only when `RSS_1 + penalty_linear` is strictly smaller than `RSS_0 + penalty_constant`. A tie stays constant.

`sigma_hat` is `1.4826 * MAD(second difference of rho) / sqrt(6)`, with a floor of `1e-12`. It is estimated on the profile being segmented. The code does not name this criterion BIC or AIC, and `segment_profile` does not smooth the series or apply another threshold. The published default is `tau = 1` and minimum segment length `4`. A display cutoff used by the original pilot (`max(10, ceil(0.05 n))`) is not applied to the tables here.

Penalty numbers from another algorithm are not on this scale. The same `tau` is also not the same numeric penalty on two profiles, because the penalty tracks `sigma_hat^2`.

## Cases

| Case | Run ID | K | p | Delta |
| --- | --- | --- | --- | --- |
| C15 | `ba238fc07fe835f9a65ee9de85e4d601` | 1.40 | 0.40 | 0.20 |
| C23 | `69ef3196fff7d8f950eb2eedad86c537` | 2.75 | 0.20 | 0.20 |

The task note spelled the C23 id as `69ef3196fff7d8f950eb2eead86c537`. That id is not in `stage1_runs`. The id above is the completed representative case with the same K, p, and Delta. Both runs have N = 1000 and trial 0. Windows are W1 [8000, 10000], W2 [10000, 12000], and W3 [12000, 14000].

Profiles are read with `stage1.query.load_profiles`. Segmentation sees each window in increasing omega. These two runs were already sorted that way.

The interval omega ∈ [0.93, 1.04] is a diagnostic region specified for C15. It is also evaluated on C23 so the columns match. It is not a boundary returned by Method C.

## Grids

Published baseline: `tau = 1`, minimum length `4`.

Penalty multipliers relative to that baseline, on W3, other settings fixed:

`0.125, 0.25, 0.5, 1, 2, 4, 8`

Minimum lengths `4, 5, 10, 20, 30` at `tau = 1` and, separately, at `tau = 0.25`. The second value is where C23 still has many short constant segments, so the length constraint has something to change. Length bins of 10, 20, and 30 oscillators in the figures are display groups only.

Window comparison uses `tau = 0.25` (weak), `1` (baseline), and `4` (strong) on W1, W2, and W3. Each window uses its own `sigma_hat`. W1 and W2 are also fit with `sigma_hat` frozen at the W3 value, so the numeric penalty matches W3. That comparison is the `sigma_mode` column.

A window pair is labeled `profile_close` only when the maximum absolute rotation-number difference is at most `0.001`. Segmentations are labeled as agreeing when the constant/linear label differs on at most 2% of oscillators. Both cutoffs are reporting labels for this directory. On these two runs the maximum window difference is about 0.013–0.018, so the comparison note stays `undecidable_profile_change_or_algorithm_sensitivity` even when the segment labels match. The mismatch fraction and boundary shift are in the CSV.

## How to run

From the repository root:

```bash
cd /Users/mohan/Desktop/SRS/helper
python3 stage2_classifier/method_c_diagnostics/run_diagnostics.py
```

The Method C file must be at the path stored in `config.json`. One run here performed 54 segmentations in under a second. Nothing is written to `stage1_runs`.

## Outputs

| File | Contents |
| --- | --- |
| `config.json` | Source hash, objective, grids, sigma values, run ids |
| `penalty_sensitivity.csv` | One row per case and tau on W3, minimum length 4 |
| `min_segment_sensitivity.csv` | Minimum lengths at tau = 1 and tau = 0.25 |
| `window_consistency.csv` | Weak, baseline, and strong tau on three windows |
| `segment_details.csv` | One row per segment, with run id, penalty, slope or level, and RSS |
| `summary.json` | Compact copy of the summary columns |
| `figures/fig1_penalty_comparison.png` | Raw rho and the fit at five tau values |
| `figures/fig2_penalty_sensitivity.png` | Segment counts, constant coverage, and RSS versus tau |
| `figures/fig3_window_comparison.png` | W1, W2, W3 at three tau values |
| `figures/fig4_c15_diagnostic_region.png` | Zoom of [0.93, 1.04] with the fixed-interval mean and OLS |
| `figures/fig5_c23_short_segments.png` | Constant-segment locations for C23 |

In `segment_details.csv`, `fitted_constant` is the segment mean and is blank for a linear segment. `fitted_slope` and `fitted_intercept` describe the OLS line and are blank for a constant segment. `fit_residual_rss` is the residual sum of squares that enters the objective. Constant segments in these files are fit pieces, not locking groups.

Figure 4 draws two different calculations. The black mean and the dotted line use only the oscillators inside [0.93, 1.04] on that window. The blue and orange curves are Method C fits of the full 1000-point profile at tau = 0.25, 1, and 4.

On the wide axes of Figures 1 and 3, a linear segment with slope near 0.3 can look nearly horizontal. The segment type and slope are in the CSV and in the Figure 4 panel titles.
