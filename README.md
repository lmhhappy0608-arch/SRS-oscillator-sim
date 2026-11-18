# Winfree oscillators on ER networks (SRS)

Winfree model on ER random graphs. The current five-state classifier is
Classifier v2.0.0 in `stage2_classifier/classifier_v2/`. It is Classifier
v1 plus one Hybrid A plateau rule, `tau_S = 1e-4`.

Stage 1 stores every completed profile in `stage1_runs/`, including the
parameter sets added after the first v2 archive: all of p=0.6, p=0.4 for
K>2, and p=0.2 through K=8. That directory is local and is not in git.
Labels for all 14325 completed runs are in
`stage2_classifier/classifier_v2/classification_labels_all.csv`.

Equation checks and the annealed phase-diagram overlays are in
`stage2_classifier/theoretical_validation/`.

## Layout

| Path | Contents |
| --- | --- |
| `code/` | Python simulation and the older summer classifier |
| `cpp/` | C++ RK4 / RK45 cross-check |
| `stage1/` | Parameter scan. Writes only to `stage1_runs/` |
| `stage2_classifier/classifier_v2/` | Current classifier, archived 6962-run tables, and the full label table |
| `stage2_classifier/classifier_v1/` | Pilot archive. Earlier equation checks still import it |
| `stage2_classifier/theoretical_validation/` | Equation checks and overlay figures |
| `docs/progress.md` | Research timeline |

## How to run

Stage 1 scan and the current classifier:

```bash
python3 -m stage1 scan --config stage1/configs/stage1_pilot.json --workers 1
python3 stage2_classifier/classifier_v2/classify.py
```

The pilot config now plans only the additional p=0.2 strip, K from 3.05 to
8. Completed runs already in `stage1_runs/` are skipped.

Python 3 with NumPy, SciPy, and Matplotlib. See `cpp/README.md` for the
C++ cross-check.
