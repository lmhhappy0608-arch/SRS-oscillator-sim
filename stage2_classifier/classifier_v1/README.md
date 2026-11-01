# Classifier v1.0.0-pilot

- Experiment: five-state classification of the Stage 1 pilot
- Version: `v1.0.0-pilot`
- Archive status: complete
- Parameter status: decision rules adopted; `alpha = 2`, `beta = 1` provisional
- ODE simulation: not rerun

This directory was completed in place. The earlier files here were an
incomplete unlabeled stop, not a finished result.

## Parameter summary

W3 assigns the state. Priority is Death, Locking, Partial Death, Partial
Locking, Incoherence.

| Rule | Value |
| --- | --- |
| Near zero | `abs(rho) < 0.001` |
| Death | near-zero fraction = 1 |
| Locking | whole-profile rho range `<= 0.01` |
| Partial Death | at least 15 consecutive near-zero oscillators |
| Valid nonzero plateau | at least 15 oscillators, relative omega span `>= 0.02`, rho span `<= 0.02`, `abs(mean rho) >= 0.001` |
| Partial Locking | at least one valid nonzero plateau |
| Incoherence | none of the above |
| Method C | `alpha = 2`, `beta = 1`, minimum segment length 4 |

Disagreement of the three window states, or of their validated-plateau
counts, sets `temporal_review_flag`. W1 and W2 do not replace W3.

## Output files

| File | Role |
| --- | --- |
| `config.json` | Thresholds, sources, and status |
| `algorithm_specification.md` | Full procedure |
| `classify.py` | Reproducible runner |
| `classification_results.csv` | One row per pilot run |
| `segmentation_details.csv` | Segments and Rule 3 flags, linked by `run_id` |
| `state_summary.csv` | Counts computed from the result table |
| `review_candidates.csv` | Rows with `review_flag = 1` |
| `figures/` | Distribution, state against K, representative fits |
| `run_manifest.json` | Timestamp, counts, checksums |
| `CHANGELOG.md` | What this completed archive contains |

## How to reproduce

From `/Users/mohan/Desktop/SRS/helper`:

```
python3 stage2_classifier/classifier_v1/classify.py
```

The command reads `stage1_runs` through the Stage 1 index and rewrites the
CSV files, figures, manifest, and `archive_validation.json` in this directory.

## Known limitations

Alpha and beta are provisional. Rule 3 was not selected by a gap in the
pilot histograms. Window disagreement is recorded as review and is not
removed from the W3 label. See `algorithm_specification.md`.

## Next version

Do not overwrite this completed directory. Put a later rule change in a new
directory and list every changed value in that directory's changelog.
