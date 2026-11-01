# Pilot exploratory analysis

Descriptive summaries of the saved Stage 1 pilot runs. The script reads
`stage1_runs` through `load_profiles` and does not integrate the ODE or modify
the SQLite index.

Near-zero tolerances and local-slope cutoffs are exploratory parameters stored
in `analysis_config.json`. They are not definitions of dynamical states.
Empty CSV fields are undefined values, not zeros. `frequency_seed` is empty
because the pilot frequencies are a deterministic grid.

Window keys are `w1` = 8000-10000, `w2` = 10000-12000, and `w3` = 12000-14000.
`D12`, `D23`, and `D13` are absolute differences between those windows.
Local slopes are least-squares slopes of rho against omega after one shared
frequency sort, on windows of 11 and 31 oscillators.

```bash
python3 -m stage1.explore
```

Outputs:

- `features.csv`: one row per run
- `representative_cases.csv`: diversity sample for visual review
- `review_candidates.csv`: high window variation, slope-scale sensitivity, neighboring-K changes, dispersion, and near-zero tolerance sensitivity
- `figures/`: parameter-space maps and one contact sheet
- `cases/`: one two-panel figure per representative case
- `summary.json`: numeric quantiles and the largest adjacent-K feature changes
