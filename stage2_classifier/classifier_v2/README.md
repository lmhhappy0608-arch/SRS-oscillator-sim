# Classifier v2.0.0

Archived Classifier v1.0.0-pilot, plus Hybrid A. The only new acceptance
rule is that a Method C linear segment which already passes Rule 3 is a
plateau when

```
S = |slope| * (omega_end - omega_start) <= 1e-4
```

There is no segment merging. Death, Locking, Partial Death, the near-zero
cut, Rule 3, alpha, beta, the minimum segment length, and the state
priority are unchanged. W3 assigns the state.

`tau_S = 1e-4` is the smaller of the two operating values compared after
the incremental-error recalibration. It is not an identified optimum.

The run reads completed Stage 1 profiles. It does not integrate the ODE
and does not write the Stage 1 index. Two indexed runs were not completed
when this version was built, so the archived count is 6962 rather than
the planned 7324. Those archived tables and figures are unchanged.

Later Stage 1 cells were labeled with the same v2.0.0 rule and were not
folded back into `classification_results.csv`. The union of that archive
and `theoretical_validation/full_grid_boundaries/extended_classification.csv`
is `classification_labels_all.csv`: 14325 completed runs, one row per
`run_id`. `cohort` is `archived` or `extended`. The extended rows carry
the state labels only; segment diagnostics remain in the 6962-row archive.
`state_summary_all.csv` counts the union.

```
python3 stage2_classifier/classifier_v2/classify.py
```
