# Stage 2 classifier records

Local archives for structure detection and classifier versions. Nothing in
these directories reruns the Winfree ODE or edits Stage 1 storage.

| Path | Contents |
| --- | --- |
| [penalty_calibration/method_c_separated.py](penalty_calibration/method_c_separated.py) | Method C wrapper called by Classifier v2 |
| [rule_decisions](rule_decisions/README.md) | Notes from the rule-setting discussion |
| [classifier_v2](classifier_v2/README.md) | Current classifier, `v2.0.0`. Full-grid labels are in `classification_labels_all.csv` |
| [theoretical_validation](theoretical_validation/README.md) | Equation checks, annealed ER boundaries, and the phase-diagram overlays |

`classifier_v2/` is the classifier used for the full Stage 1 grid and for
the equation checks. Later classifier versions must be new directories.
