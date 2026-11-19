# Stage 2 classifier records

Local archives for structure detection and classifier versions. Nothing in
these directories reruns the Winfree ODE or edits Stage 1 storage.

| Path | Contents |
| --- | --- |
| [method_c_diagnostics](method_c_diagnostics/README.md) | Stage 2B Method C sensitivity on C15 and C23 |
| [penalty_calibration](penalty_calibration/README.md) | Stage 2C separated-penalty calibration. Status: Calibration inconclusive |
| [rule_decisions](rule_decisions/README.md) | Notes from the rule-setting discussion |
| [classifier_v2](classifier_v2/README.md) | Current classifier, `v2.0.0`. Full-grid labels are in `classification_labels_all.csv` |
| [classifier_v2_research](classifier_v2_research/hybrid_method_c/hybrid_method_c_report.md) | Recalibration that selected `tau_S = 1e-4` |
| [theoretical_validation](theoretical_validation/README.md) | Equation checks, annealed ER boundaries, and the phase-diagram overlays |

`classifier_v2/` is the classifier used for the full Stage 1 grid and for
the equation checks. Later classifier versions must be new directories.
