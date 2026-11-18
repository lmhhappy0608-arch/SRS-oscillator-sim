# Stage 2 classifier records

Local archives for structure detection and classifier versions. Nothing in
these directories reruns the Winfree ODE or edits Stage 1 storage.

| Path | Contents |
| --- | --- |
| [method_c_diagnostics](method_c_diagnostics/README.md) | Stage 2B Method C sensitivity on C15 and C23 |
| [penalty_calibration](penalty_calibration/README.md) | Stage 2C separated-penalty calibration. Status: Calibration inconclusive |
| [rule_decisions](rule_decisions/README.md) | Notes from the rule-setting discussion |
| [classifier_v1](classifier_v1/README.md) | Classifier `v1.0.0-pilot`, kept as the archive. The 732-run classification and the earlier equation checks import this directory |
| [classifier_v2](classifier_v2/README.md) | Current classifier, `v2.0.0`. v1 plus Hybrid A at `tau_S = 1e-4`. Full-grid labels are in `classification_labels_all.csv` |
| [classifier_v2_research](classifier_v2_research/hybrid_method_c/hybrid_method_c_report.md) | Recalibration that selected `tau_S = 1e-4` |
| [theoretical_validation](theoretical_validation/README.md) | Equation checks, annealed ER boundaries, and the phase-diagram overlays |

`classifier_v2/` is the classifier used for the full Stage 1 grid. The
completed `classifier_v1/` tables stay in place: the p=1 checks of Eqs. (5),
(7), and (8)–(9) still import that code. Later classifier versions must be
new directories.
