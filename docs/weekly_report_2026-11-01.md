# Weekly report — week ending 2026-11-01

Repository: `lmhhappy0608-arch/SRS-oscillator-sim ‘

## What was finished

### 1. Stage 1 pilot: 732 rotation profiles

Config: `stage1/configs/stage1_pilot.json`.

- \(N=1000\), \(T=14000\), one trial, RK4
- Windows: W1 \([8000,10000]\), W2 \([10000,12000]\), W3 \([12000,14000]\)
- \(K\) from 0 to 3, step 0.05 (61 values)
- \(p\in\{1, 0.8, 0.4, 0.2\}\)
- \(\Delta\in\{0.05, 0.2, 0.8\}\)
- \(61\times 4\times 3=732\) completed runs

Raw profiles stay in `stage1_runs/` (about 105 MB). That directory is listed in the local `.gitignore` and should not be pushed. Descriptive features and case figures are in `stage1_analysis/pilot_exploration/`.

### 2. Segment model before any five-state label

The plateau finder is the existing Method C segmentation, not a new package. A profile is split into constant and linear pieces by the penalized dynamic program `cost_audit.dp_penalized`. The separated objective is

\[
J=\sum \mathrm{RSS}+\hat\sigma^2\ln(N)\,(\alpha M+\beta L).
\]

\(M\) is the number of pieces and \(L\) is the number of linear pieces. This is a BIC-style change-point penalty. The repository name is Method C. Minimum piece length 4 is the algorithm constraint already in Method C, not a physical locking size.

| Step | Directory | Result |
| --- | --- | --- |
| 2B | `stage2_classifier/method_c_diagnostics/` | Sensitivity on two saved profiles (C15, C23). No labels. |
| 2C | `stage2_classifier/penalty_calibration/` | Separated \((\alpha,\beta)\) matches original \(\tau\) when \(\alpha=2\tau\) and \(\beta=\tau\). Status written in `candidate_parameters.json`: **Calibration inconclusive**. The cell \(\alpha=2\), \(\beta=1\) equals the original \(\tau=1\) and is used as a provisional setting. It is not a frozen theoretical value. |

### 3. Decision rules, then Classifier v1

Rules were set from the 732 saved profiles where a gap existed, and by explicit choice where the histograms had no gap. The live specification is `stage2_classifier/classifier_v1/algorithm_specification.md`. \(\alpha=2\), \(\beta=1\) stay provisional.

W3 assigns the state. W1 and W2 do not override it. A row is flagged for review when the three window states differ, or when the counts of validated nonzero plateaus differ.

| Rule | Adopted value |
| --- | --- |
| Near zero | \(\lvert\rho\rvert<0.001\) |
| Death | near-zero fraction \(=1\) (not 0.95) |
| Locking | whole-profile \(\max\rho-\min\rho\le 0.01\) |
| Partial Death | at least 15 consecutive near-zero oscillators |
| Valid nonzero plateau | Method C constant piece, length \(\ge 15\), relative \(\omega\) span \(\ge 0.02\), \(\rho\) span \(\le 0.02\), \(\lvert\mathrm{mean}\,\rho\rvert\ge 0.001\) |
| Partial Locking | at least one valid nonzero plateau |
| Priority | Death → Locking → Partial Death → Partial Locking → Incoherence |

The relative-\(\omega\) cut of 0.02 was specified directly. The constant-segment histograms did not contain an empty gap that selected it.

### 4. v1 labels on all 732 runs

Directory: `stage2_classifier/classifier_v1/`. Version `v1.0.0-pilot`. `archive_validation.json` records `passed: true`, 732 classified, 0 failed.

Counts from `state_summary.csv` (W3):

| State | Count |
| --- | --- |
| Death | 294 |
| Incoherence | 189 |
| Partial Death | 115 |
| Locking | 107 |
| Partial Locking | 27 |

`review_flag=1` on 95 rows. Of those, 82 differ only in the validated-plateau count across windows, and 13 differ in both state and plateau count. The flag is a window record, not a probability.

Phase diagrams are `figures/phase_diagram_p_*.png`. Each \(p\) has only three \(\Delta\) columns, so the blocks look smooth because the \(\Delta\) grid is coarse, not because the classifier interpolated a boundary.

### 5. Same 732 profiles under the summer classifier

`stage2_classifier/classifier_v1/classify_summer.py` calls the unmodified `classify_state` in `code/Ariaratnam2001.py` (near zero 0.01, Death at fraction \(\ge 0.95\), consecutive near-zero run 4, priority Death → Partial Death → Locking → Partial Locking → Incoherence). The label used for comparison is also W3.

`classifier_change_summary.csv`: 700 rows keep the same state, 32 change. The 32 parameter points are in `changed_parameters_v1_summer.csv`. All have \(N=1000\).

| v1 → Summer | Count |
| --- | --- |
| Incoherence → Locking | 2 |
| Incoherence → Partial Locking | 7 |
| Partial Locking → Incoherence | 7 |
| Partial Locking → Locking | 3 |
| Partial Death → Death | 6 |
| Incoherence → Partial Death | 7 |

Side-by-side figures: `figures/phase_diagram_compare_p_{1.00,0.80,0.40,0.20}.png`. Left panel is v1, right panel is the summer classifier.

## What this week does not establish

- \(\alpha=2\), \(\beta=1\) are not a calibrated optimum. Stage 2C remains inconclusive.
- The three-\(\Delta\) diagrams are not a replacement for the summer \((K,\delta)\) grid, which used many more frequency widths and stopped at \(K=2\), \(T=10000\).
- The 32 disagreements are measured label differences on the same saved W3 profiles. They are not yet a statement about which rule is closer to the dynamics.

## Where the files are

| Path | Role |
| --- | --- |
| `stage1/` | Scan code and pilot config |
| `stage1_runs/` | 732 NPZ profiles and the index. Local only |
| `stage1_analysis/pilot_exploration/` | Features and case figures from the saved runs |
| `stage2_classifier/method_c_diagnostics/` | Stage 2B |
| `stage2_classifier/penalty_calibration/` | Stage 2C |
| `stage2_classifier/classifier_v1/` | Rules, v1 labels, summer comparison, figures |
| `docs/weekly_report_2026-11-01.md` | This note |

`docs/progress.md` and the root `README.md` still describe the summer `classify_state` as the classifier in use. That text is older than this week’s archive.
