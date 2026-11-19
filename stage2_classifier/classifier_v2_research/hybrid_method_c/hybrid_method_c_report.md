# Hybrid Method C validation

## 1. Research objective

The question is whether plateaus missed by archived Method C can be recovered by validating nearly flat linear segments, without a large false-positive rate on slow or nonlinear drift.
Classifier v1.0.0-pilot is unchanged. Method C still uses alpha = 2, beta = 1, and minimum segment length 4. Only constant segments pass Rule 3 in the baseline. Death, Locking, Partial Death, the near-zero cut, and the state priority are unchanged.
The Winfree ODE was not rerun. Stage 1 files were read only. No theoretical phase boundary was used as a label, and the five named diagnostic cells were not used to choose parameters.

## 2. Reused code and data

- Method C dynamic program: `stage2_classifier/penalty_calibration/method_c_separated.py`, calling the archived `cost_audit.dp_penalized`.
- Rule 3 and the state priority: `stage2_classifier/classifier_v1/config.json`.
- Synthetic profiles, families, and the development / validation / test split: `catalog()` in `stage2_classifier/classifier_v2_research/calibrate_plateau_detector.py`.
- Real profiles: completed Stage 1 NPZ files, window W3 = [12000, 14000], frequency-sorted before segmentation.

The v2 candidate detector was not reused for the decision. Its development split had already been used to freeze that candidate's own thresholds. Those thresholds are not inputs here. The same synthetic generator is reused because it already contains an embedded plateau, slow slopes, quadratic drift, a near-global plateau with a tail, multiple plateaus, and noisy plateaus.
The older Stage 2C generator was not scored. It does not have the same held-out lengths and slopes.
Saved `segmentation_details.csv` covers the 732-run pilot and does not contain the p = 1 focus cells. Those cells were segmented again in memory.

Leakage check: tau_S and the piece guard were chosen on development profiles only. Validation was not searched. The test split was scored once after the representative point was fixed. Real Winfree profiles, including the five named cells, were excluded from the score. v1 state labels were not used as ground truth. There is no separate tiny real validation set, because real profiles have no plateau mask that is independent of the classifier.

## 3. Hybrid A

Hybrid A keeps the Method C partition. A constant segment is accepted exactly by archived Rule 3: at least 15 oscillators, relative omega span at least 0.02, rotation-number range R_I at most 0.02, and absolute mean at least 0.001.
A linear segment is no longer rejected for its label. It must pass the same geometry and also S_I <= tau_S, where b_I is the ordinary-least-squares slope Cov(omega, rho) / Var(omega) and S_I = |b_I| * (max omega - min omega) on that segment.
No adjacent segments are merged.

## 4. Hybrid B

Hybrid B starts from the same admissible single segments and also considers contiguous runs of Method C segments. A run of two or more segments is accepted only after recomputing member count, relative omega span, R_I, mean rotation number, slope, and S_I. The union must pass geometry and S_I <= tau_S. Two individually valid segments are not kept as their union unless that union passes.
A segment whose own S_I exceeds the piece guard cannot enter a multi-segment run. A constant segment that passes Rule 3 can still stand alone. Among overlapping admissible runs, a run contained in a larger admissible run is dropped. Remaining overlaps keep the smaller S_I, then the longer run, then the earlier start.
This blocks a steep tail piece and a steep slope piece from being absorbed. It does not block a very shallow chunk whose own S_I is below the cap. That case is measured as a false positive, not removed by an extra rule.

## 5. Calibration

The searched trend caps are a geometric grid from 1e-4 to the archived range cap 0.02. The piece guard uses the same grid plus a setting with no piece guard. Eligibility is fixed before the scores are read:

1. False-positive profile rate at most 0.05 on linear drift, slow slope, and nonlinear drift; long-plateau recall at least 0.80; near-global recall at least 0.80; median boundary error at most 25 oscillators.
2. The same false-positive cap and recall floors, without the boundary constraint.
3. False-positive rate at most 0.05 and long recall at least 0.50.
4. False-positive rate at most 0.10 and long recall at least 0.50.

The first stage with any grid point is the eligible pool. The stable subset stays within 0.02 long-recall and 0.02 mean false-positive rate of the best point in that pool. The representative tau_S is the smallest stable value. If the no-guard piece setting is stable at that tau_S, it is used.
Development profiles that are selectable, iid, and single-window are the only selection set. The objective is generating-mask recovery against drift false positives. It is not agreement with v1 and it is not a count of Partial Locking labels.

Selection status: **eligible**. Stage: false-positive rate <= 0.10, long recall >= 0.50.
Eligible grid points: 28. Stable grid points: 28.
Stable tau_S values: 0.0001, 0.000213, 0.000454, 0.000969.
Stable piece guards: 0.0001, 0.000213, 0.000454, 0.000969, 0.00206, 0.0044, 0.00938, 0.0200, none.
Representative point: tau_S = 0.0001, piece guard = none.

### Development metrics at the representative point

| Method | Macro F1 | Long recall | Near-global recall | FP drift | FP slow | FP nonlinear | Boundary | Miss recovery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid A | 0.6892 | 0.9130 | 1.0000 | 0 | 0 | 0.0833 | 1.5000 | 0.0714 |
| Hybrid B | 0.6980 | 1.0000 | 1.0000 | 0 | 0 | 0.0833 | 0.5000 | 0.2143 |

Baseline matches 0.6111 of the development generating plateaus at IoU >= 0.5. Miss recovery is the fraction of those baseline misses that the hybrid matches. Boundary error is the median endpoint error, in oscillators, on easy long matches.

## 6. Synthetic benchmark

Validation and test use different lengths, slopes, and seeds from development. Test numbers below did not move the representative point.

| Split | Method | Macro F1 | Long recall | Near-global | FP drift | FP slow | FP nonlinear | Stress recall | Boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | A | 0.9053 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0 | 0.7500 |
| validation | B | 0.9119 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0 | 0.5000 |
| test | A | 0.9117 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0 | 0.5000 |
| test | B | 0.9137 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0 | 0.5000 |

### Family F1 on validation

| Family | Hybrid A | Hybrid B |
| --- | --- | --- |
| embedded | 0.6572 | 0.6586 |
| flat | n/a | n/a |
| tails | 0.9968 | 0.9968 |
| multi | 0.9672 | 0.9923 |
| nearglobal | 0.9997 | 0.9997 |

### Embedded length recall, low noise, development

| Length | Hybrid A | Hybrid B |
| --- | --- | --- |
| 40 | 1.0000 | 1.0000 |
| 100 | 1.0000 | 1.0000 |
| 200 | 0.5000 | 1.0000 |
| 500 | 1.0000 | 1.0000 |
| 900 | 1.0000 | 1.0000 |

### Noiseless slope segments

Hybrid A

| Family | Slope | Full S | Full R | Segments | Profiles fired | Mean members |
| --- | --- | --- | --- | --- | --- | --- |
| drift | 0.300 | 0.210 | 0.210 | 1 | 0/3 | 0 |
| drift | 1.000 | 0.700 | 0.700 | 5 | 0/3 | 0 |
| slow | 0.020 | 0.014 | 0.014 | 1 | 0/3 | 0 |
| slow | 0.050 | 0.035 | 0.035 | 1 | 0/3 | 0 |
| slow | 0.100 | 0.070 | 0.070 | 5 | 0/3 | 0 |

Hybrid B

| Family | Slope | Full S | Full R | Segments | Profiles fired | Mean members |
| --- | --- | --- | --- | --- | --- | --- |
| drift | 0.300 | 0.210 | 0.210 | 1 | 0/3 | 0 |
| drift | 1.000 | 0.700 | 0.700 | 5 | 0/3 | 0 |
| slow | 0.020 | 0.014 | 0.014 | 1 | 0/3 | 0 |
| slow | 0.050 | 0.035 | 0.035 | 1 | 0/3 | 0 |
| slow | 0.100 | 0.070 | 0.070 | 5 | 0/3 | 0 |

Stress recall is the hit rate on positive plateaus of length at least 80 with noise 0.01. Those misses are the archived range cap: the observed range exceeds 0.02, so neither hybrid can accept the interval. The validation embedded F1 near 0.66 is the average of clean profiles, which are recovered, and this noise level, which is not.

### Sensitivity

FP columns count any accepted plateau, including an archived constant segment. Added columns count profiles that Baseline does not already accept. The block at tau_S = 0.000969 lists every piece guard; elsewhere Hybrid B is shown with the guard off.

| Method | tau_S | Piece guard | Long recall | FP slow | Added slow | FP nonlinear | Added nonlinear | FP drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.0001 | none | 0.9130 | 0 | 0 | 0.0833 | 0 | 0 |
| B | 0.0001 | none | 1.0000 | 0 | 0 | 0.0833 | 0 | 0 |
| A | 0.000213 | none | 0.9130 | 0 | 0 | 0.0833 | 0 | 0 |
| B | 0.000213 | none | 1.0000 | 0 | 0 | 0.0833 | 0 | 0 |
| A | 0.000454 | none | 0.9130 | 0 | 0 | 0.0833 | 0 | 0 |
| B | 0.000454 | none | 1.0000 | 0 | 0 | 0.0833 | 0 | 0 |
| A | 0.000969 | none | 0.9130 | 0 | 0 | 0.0833 | 0 | 0 |
| B | 0.000969 | 0.0001 | 1.0000 | 0 | 0 | 0.0833 | 0 | 0 |
| B | 0.000969 | 0.000213 | 1.0000 | 0 | 0 | 0.3333 | 0.2500 | 0 |
| B | 0.000969 | 0.000454 | 1.0000 | 0 | 0 | 0.3333 | 0.2500 | 0 |
| B | 0.000969 | 0.000969 | 1.0000 | 0 | 0 | 0.3333 | 0.2500 | 0 |
| B | 0.000969 | 0.00206 | 1.0000 | 0 | 0 | 0.3333 | 0.2500 | 0 |
| B | 0.000969 | 0.0044 | 1.0000 | 0 | 0 | 0.3333 | 0.2500 | 0 |
| B | 0.000969 | 0.00938 | 1.0000 | 0 | 0 | 0.3333 | 0.2500 | 0 |
| B | 0.000969 | 0.0200 | 1.0000 | 0 | 0 | 0.3333 | 0.2500 | 0 |
| B | 0.000969 | none | 1.0000 | 0 | 0 | 0.3333 | 0.2500 | 0 |
| A | 0.00206 | none | 0.9130 | 0.1667 | 0.1667 | 0.0833 | 0 | 0 |
| B | 0.00206 | none | 1.0000 | 0.1667 | 0.1667 | 0.3333 | 0.2500 | 0 |
| A | 0.0044 | none | 0.9130 | 0.1667 | 0.1667 | 0.0833 | 0 | 0 |
| B | 0.0044 | none | 1.0000 | 0.1667 | 0.1667 | 0.3333 | 0.2500 | 0 |
| A | 0.00938 | none | 0.9130 | 0.1667 | 0.1667 | 0.0833 | 0 | 0 |
| B | 0.00938 | none | 1.0000 | 0.1667 | 0.1667 | 0.3333 | 0.2500 | 0 |
| A | 0.0200 | none | 0.9130 | 0.3333 | 0.3333 | 0.0833 | 0 | 0 |
| B | 0.0200 | none | 1.0000 | 0.3333 | 0.3333 | 0.5833 | 0.5000 | 0 |

## 7. Real Winfree diagnostic cases

These rows describe W3. They are not a test score. A state change to Partial Locking is not evidence that the cell is frequency-locked.

| p | delta | K | Segments | Constant | Linear | Baseline | Hybrid A | Hybrid B | Base n | A n | B n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 0.200 | 0.60 | 58 | 0 | 58 | Incoherence | Partial Locking | Partial Locking | 0 | 16 | 1 |
| 1.00 | 0.200 | 0.65 | 43 | 0 | 43 | Locking | Locking | Locking | 0 | 17 | 1 |
| 1.00 | 0.200 | 0.70 | 132 | 0 | 132 | Incoherence | Partial Locking | Partial Locking | 0 | 2 | 6 |
| 1.00 | 0.200 | 0.75 | 137 | 0 | 137 | Incoherence | Partial Locking | Partial Locking | 0 | 3 | 15 |
| 1.00 | 0.150 | 0.40 | 45 | 13 | 32 | Incoherence | Incoherence | Incoherence | 0 | 0 | 0 |
| 1.00 | 0.100 | 0.30 | 38 | 0 | 38 | Locking | Locking | Locking | 0 | 18 | 2 |
| 1.00 | 0.050 | 0.10 | 115 | 57 | 58 | Incoherence | Incoherence | Incoherence | 0 | 0 | 0 |
| 1.00 | 0.200 | 0.10 | 3 | 0 | 3 | Incoherence | Incoherence | Incoherence | 0 | 0 | 0 |
| 1.00 | 0.200 | 0.55 | 44 | 16 | 28 | Incoherence | Incoherence | Incoherence | 0 | 0 | 0 |
| 1.00 | 0.225 | 0.75 | 55 | 25 | 30 | Partial Locking | Partial Locking | Partial Locking | 1 | 1 | 1 |
| 0.40 | 0.200 | 1.50 | 52 | 16 | 36 | Partial Locking | Partial Locking | Partial Locking | 4 | 4 | 4 |
| 0.40 | 0.200 | 1.70 | 39 | 9 | 30 | Partial Locking | Partial Locking | Partial Locking | 3 | 6 | 4 |
| 0.80 | 0.200 | 0.70 | 30 | 12 | 18 | Incoherence | Incoherence | Incoherence | 0 | 0 | 0 |
| 0.80 | 0.200 | 1.40 | 1 | 1 | 0 | Death | Death | Death | 0 | 0 | 0 |

### New plateaus

**p=1.00, delta=0.200, K=0.60.**
- Hybrid A: members 24 (24 outside Baseline), indices [5, 29), omega [0.8020, 0.8112], R=2.599e-06, S=2.598e-06, b=-0.0002821, mean=0.7257, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 46 (46 outside Baseline), indices [29, 75), omega [0.8116, 0.8296], R=5.134e-06, S=5.133e-06, b=-0.0002849, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 45 (45 outside Baseline), indices [75, 120), omega [0.8300, 0.8476], R=5.187e-06, S=5.187e-06, b=-0.0002944, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 30 (30 outside Baseline), indices [120, 150), omega [0.8480, 0.8597], R=3.522e-06, S=3.522e-06, b=-0.0003034, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 40 (40 outside Baseline), indices [150, 190), omega [0.8601, 0.8757], R=4.863e-06, S=4.863e-06, b=-0.0003114, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 55 (55 outside Baseline), indices [190, 245), omega [0.8761, 0.8977], R=6.939e-06, S=6.94e-06, b=-0.000321, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 133 (133 outside Baseline), indices [245, 378), omega [0.8981, 0.9510], R=1.74e-05, S=1.744e-05, b=-0.0003299, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 43 (43 outside Baseline), indices [397, 440), omega [0.9590, 0.9758], R=5.335e-06, S=5.336e-06, b=-0.0003173, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 43 (43 outside Baseline), indices [440, 483), omega [0.9762, 0.9930], R=5.069e-06, S=5.07e-06, b=-0.0003015, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 32 (32 outside Baseline), indices [548, 580), omega [1.0194, 1.0318], R=2.893e-06, S=2.893e-06, b=-0.0002331, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 28 (28 outside Baseline), indices [580, 608), omega [1.0322, 1.0430], R=2.204e-06, S=2.205e-06, b=-0.0002039, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 23 (23 outside Baseline), indices [608, 631), omega [1.0434, 1.0523], R=1.547e-06, S=1.547e-06, b=-0.0001757, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 31 (31 outside Baseline), indices [631, 662), omega [1.0527, 1.0647], R=1.706e-06, S=1.707e-06, b=-0.0001421, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 33 (33 outside Baseline), indices [672, 705), omega [1.0691, 1.0819], R=1.059e-06, S=1.06e-06, b=-8.271e-05, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 26 (26 outside Baseline), indices [786, 812), omega [1.1147, 1.1247], R=1.161e-06, S=1.161e-06, b=0.0001159, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 25 (25 outside Baseline), indices [834, 859), omega [1.1339, 1.1435], R=2.124e-06, S=2.124e-06, b=0.000221, mean=0.7256, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid B: members 966 (966 outside Baseline), indices [0, 966), omega [0.8000, 1.1864], R=7.777e-05, S=7.782e-05, b=-0.0002014, mean=0.7256, pieces=51, tail=false. adjacent merge recovered pieces that fail on their own.

**p=1.00, delta=0.200, K=0.65.**
- Hybrid A: members 30 (30 outside Baseline), indices [164, 194), omega [0.8657, 0.8773], R=3.251e-06, S=3.249e-06, b=-0.0002798, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 57 (57 outside Baseline), indices [194, 251), omega [0.8777, 0.9001], R=4.849e-06, S=4.838e-06, b=-0.0002158, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 48 (48 outside Baseline), indices [251, 299), omega [0.9005, 0.9193], R=3.071e-06, S=3.067e-06, b=-0.000163, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 27 (27 outside Baseline), indices [299, 326), omega [0.9197, 0.9301], R=1.417e-06, S=1.417e-06, b=-0.0001361, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 36 (36 outside Baseline), indices [374, 410), omega [0.9497, 0.9638], R=1.366e-06, S=1.365e-06, b=-9.74e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 33 (33 outside Baseline), indices [410, 443), omega [0.9642, 0.9770], R=1.097e-06, S=1.097e-06, b=-8.56e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 42 (42 outside Baseline), indices [443, 485), omega [0.9774, 0.9938], R=1.235e-06, S=1.234e-06, b=-7.517e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 27 (27 outside Baseline), indices [497, 524), omega [0.9990, 1.0094], R=6.751e-07, S=6.75e-07, b=-6.484e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 26 (26 outside Baseline), indices [524, 550), omega [1.0098, 1.0198], R=6.012e-07, S=6.011e-07, b=-6.005e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 39 (39 outside Baseline), indices [550, 589), omega [1.0202, 1.0354], R=8.384e-07, S=8.381e-07, b=-5.508e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 29 (29 outside Baseline), indices [589, 618), omega [1.0358, 1.0470], R=5.694e-07, S=5.693e-07, b=-5.078e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 29 (29 outside Baseline), indices [618, 647), omega [1.0474, 1.0587], R=5.359e-07, S=5.358e-07, b=-4.779e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 53 (53 outside Baseline), indices [647, 700), omega [1.0591, 1.0799], R=9.29e-07, S=9.284e-07, b=-4.459e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 55 (55 outside Baseline), indices [708, 763), omega [1.0835, 1.1051], R=9.059e-07, S=9.051e-07, b=-4.186e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 114 (114 outside Baseline), indices [763, 877), omega [1.1055, 1.1508], R=1.983e-06, S=1.969e-06, b=-4.353e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 83 (83 outside Baseline), indices [877, 960), omega [1.1512, 1.1840], R=2.044e-06, S=2.014e-06, b=-6.135e-05, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 40 (40 outside Baseline), indices [960, 1000), omega [1.1844, 1.2000], R=2.048e-06, S=1.958e-06, b=-0.0001254, mean=0.6472, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid B: members 1000 (1000 outside Baseline), indices [0, 1000), omega [0.8000, 1.2000], R=7.51e-05, S=4.769e-05, b=-0.0001192, mean=0.6472, pieces=43, tail=false. adjacent merge recovered pieces that fail on their own.

**p=1.00, delta=0.200, K=0.70.**
- Hybrid A: members 21 (21 outside Baseline), indices [500, 521), omega [1.0002, 1.0082], R=2.25e-05, S=2.25e-05, b=-0.00281, mean=0.5485, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 24 (24 outside Baseline), indices [583, 607), omega [1.0334, 1.0426], R=3.165e-05, S=3.165e-05, b=-0.003436, mean=0.5483, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid B: members 500 (500 outside Baseline), indices [0, 500), omega [0.8000, 0.9998], R=0.0002371, S=7.845e-05, b=-0.0003927, mean=0.5486, pieces=71, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 69 (69 outside Baseline), indices [553, 622), omega [1.0214, 1.0486], R=9.202e-05, S=9.199e-05, b=-0.003379, mean=0.5484, pieces=5, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 51 (51 outside Baseline), indices [640, 691), omega [1.0563, 1.0763], R=8.101e-05, S=8.1e-05, b=-0.004046, mean=0.5482, pieces=7, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 46 (46 outside Baseline), indices [720, 766), omega [1.0883, 1.1063], R=8.73e-05, S=8.728e-05, b=-0.004844, mean=0.5481, pieces=6, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 44 (44 outside Baseline), indices [775, 819), omega [1.1103, 1.1275], R=9.489e-05, S=9.487e-05, b=-0.00551, mean=0.548, pieces=3, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 181 (181 outside Baseline), indices [819, 1000), omega [1.1279, 1.2000], R=0.01573, S=5.686e-05, b=-0.0007889, mean=0.5478, pieces=33, tail=false. R is much larger than the fitted trend. adjacent merge recovered pieces that fail on their own.

**p=1.00, delta=0.200, K=0.75.**
- Hybrid A: members 28 (28 outside Baseline), indices [716, 744), omega [1.0867, 1.0975], R=2.812e-05, S=2.811e-05, b=-0.002601, mean=0.4151, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 45 (45 outside Baseline), indices [744, 789), omega [1.0979, 1.1155], R=4.551e-05, S=4.547e-05, b=-0.002581, mean=0.4151, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 22 (22 outside Baseline), indices [789, 811), omega [1.1159, 1.1243], R=2.224e-05, S=2.223e-05, b=-0.002644, mean=0.415, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid B: members 33 (33 outside Baseline), indices [17, 50), omega [0.8068, 0.8196], R=8.771e-05, S=8.774e-05, b=-0.006848, mean=0.4168, pieces=7, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 26 (26 outside Baseline), indices [75, 101), omega [0.8300, 0.8400], R=8.253e-05, S=8.254e-05, b=-0.008246, mean=0.4167, pieces=4, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 26 (26 outside Baseline), indices [114, 140), omega [0.8456, 0.8557], R=9.074e-05, S=9.076e-05, b=-0.009067, mean=0.4165, pieces=5, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 23 (23 outside Baseline), indices [148, 171), omega [0.8593, 0.8681], R=8.406e-05, S=8.407e-05, b=-0.009544, mean=0.4164, pieces=2, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 26 (26 outside Baseline), indices [217, 243), omega [0.8869, 0.8969], R=9.637e-05, S=9.639e-05, b=-0.009629, mean=0.4162, pieces=4, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 21 (21 outside Baseline), indices [255, 276), omega [0.9021, 0.9101], R=7.349e-05, S=7.349e-05, b=-0.009178, mean=0.416, pieces=2, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 23 (23 outside Baseline), indices [276, 299), omega [0.9105, 0.9193], R=7.727e-05, S=7.728e-05, b=-0.008773, mean=0.4159, pieces=3, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 26 (26 outside Baseline), indices [322, 348), omega [0.9289, 0.9389], R=7.727e-05, S=7.728e-05, b=-0.00772, mean=0.4158, pieces=3, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 33 (33 outside Baseline), indices [369, 402), omega [0.9477, 0.9606], R=8.387e-05, S=8.386e-05, b=-0.006545, mean=0.4156, pieces=5, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 41 (41 outside Baseline), indices [402, 443), omega [0.9610, 0.9770], R=9.213e-05, S=9.21e-05, b=-0.005751, mean=0.4155, pieces=8, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 49 (49 outside Baseline), indices [449, 498), omega [0.9798, 0.9990], R=9.25e-05, S=9.245e-05, b=-0.00481, mean=0.4154, pieces=8, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 60 (60 outside Baseline), indices [521, 581), omega [1.0086, 1.0322], R=8.844e-05, S=8.836e-05, b=-0.00374, mean=0.4153, pieces=8, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 72 (72 outside Baseline), indices [594, 666), omega [1.0378, 1.0663], R=8.642e-05, S=8.63e-05, b=-0.003036, mean=0.4152, pieces=8, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 81 (81 outside Baseline), indices [744, 825), omega [1.0979, 1.1299], R=8.405e-05, S=8.376e-05, b=-0.002615, mean=0.415, pieces=3, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 60 (60 outside Baseline), indices [857, 917), omega [1.1431, 1.1668], R=9.365e-05, S=9.132e-05, b=-0.003866, mean=0.4149, pieces=12, tail=false. adjacent merge recovered pieces that fail on their own.

**p=1.00, delta=0.150, K=0.40.** No plateau beyond Baseline.

**p=1.00, delta=0.100, K=0.30.**
- Hybrid A: members 34 (34 outside Baseline), indices [175, 209), omega [0.9350, 0.9416], R=1.253e-05, S=1.253e-05, b=-0.001897, mean=0.943, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 45 (45 outside Baseline), indices [226, 271), omega [0.9452, 0.9541], R=1.482e-05, S=1.482e-05, b=-0.001682, mean=0.943, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 33 (33 outside Baseline), indices [271, 304), omega [0.9543, 0.9607], R=9.955e-06, S=9.954e-06, b=-0.001554, mean=0.943, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 42 (42 outside Baseline), indices [304, 346), omega [0.9609, 0.9691], R=1.185e-05, S=1.185e-05, b=-0.001444, mean=0.943, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 29 (29 outside Baseline), indices [346, 375), omega [0.9693, 0.9749], R=7.56e-06, S=7.56e-06, b=-0.001349, mean=0.943, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 21 (21 outside Baseline), indices [375, 396), omega [0.9751, 0.9791], R=5.152e-06, S=5.152e-06, b=-0.001287, mean=0.943, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 49 (49 outside Baseline), indices [396, 445), omega [0.9793, 0.9889], R=1.16e-05, S=1.159e-05, b=-0.001206, mean=0.943, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 38 (38 outside Baseline), indices [457, 495), omega [0.9915, 0.9989], R=8.083e-06, S=8.083e-06, b=-0.001091, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 48 (48 outside Baseline), indices [495, 543), omega [0.9991, 1.0085], R=9.512e-06, S=9.511e-06, b=-0.001011, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 50 (50 outside Baseline), indices [543, 593), omega [1.0087, 1.0185], R=9.093e-06, S=9.092e-06, b=-0.0009269, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 50 (50 outside Baseline), indices [593, 643), omega [1.0187, 1.0285], R=8.322e-06, S=8.321e-06, b=-0.0008482, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 64 (64 outside Baseline), indices [643, 707), omega [1.0287, 1.0413], R=9.656e-06, S=9.655e-06, b=-0.0007655, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 31 (31 outside Baseline), indices [707, 738), omega [1.0415, 1.0475], R=4.208e-06, S=4.208e-06, b=-0.0007005, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 65 (65 outside Baseline), indices [738, 803), omega [1.0477, 1.0606], R=8.17e-06, S=8.169e-06, b=-0.0006376, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 55 (55 outside Baseline), indices [803, 858), omega [1.0608, 1.0716], R=6.053e-06, S=6.053e-06, b=-0.0005599, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 23 (23 outside Baseline), indices [858, 881), omega [1.0718, 1.0762], R=2.24e-06, S=2.24e-06, b=-0.0005085, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 45 (45 outside Baseline), indices [881, 926), omega [1.0764, 1.0852], R=4.062e-06, S=4.063e-06, b=-0.0004612, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 74 (74 outside Baseline), indices [926, 1000), omega [1.0854, 1.1000], R=5.34e-06, S=5.352e-06, b=-0.0003662, mean=0.9429, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid B: members 209 (209 outside Baseline), indices [17, 226), omega [0.9034, 0.9450], R=9.454e-05, S=9.387e-05, b=-0.002254, mean=0.9431, pieces=18, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 625 (625 outside Baseline), indices [375, 1000), omega [0.9751, 1.1000], R=9.614e-05, S=9.515e-05, b=-0.0007616, mean=0.9429, pieces=14, tail=false. adjacent merge recovered pieces that fail on their own.

**p=1.00, delta=0.050, K=0.10.** No plateau beyond Baseline.

**p=1.00, delta=0.200, K=0.10.** No plateau beyond Baseline.

**p=1.00, delta=0.200, K=0.55.** No plateau beyond Baseline.

**p=1.00, delta=0.225, K=0.75.** No plateau beyond Baseline.

**p=0.40, delta=0.200, K=1.50.**
- Hybrid A: no added plateau.
- Hybrid B: members 95 (44 outside Baseline), indices [669, 764), omega [1.0679, 1.1055], R=0.0004377, S=9.421e-05, b=-0.002503, mean=0.727, pieces=6, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 23 (23 outside Baseline), indices [768, 791), omega [1.1075, 1.1163], R=0.0003623, S=1.187e-05, b=-0.001348, mean=0.727, pieces=2, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 29 (29 outside Baseline), indices [828, 857), omega [1.1315, 1.1427], R=0.0004333, S=3.284e-05, b=-0.002929, mean=0.7268, pieces=5, tail=false. adjacent merge recovered pieces that fail on their own.

**p=0.40, delta=0.200, K=1.70.**
- Hybrid A: members 108 (108 outside Baseline), indices [10, 118), omega [0.8040, 0.8468], R=7.389e-05, S=4.964e-05, b=-0.001159, mean=0.5971, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 217 (217 outside Baseline), indices [118, 335), omega [0.8472, 0.9337], R=4.114e-05, S=1.458e-05, b=-0.0001685, mean=0.5971, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid A: members 379 (379 outside Baseline), indices [335, 714), omega [0.9341, 1.0855], R=9.016e-05, S=2.657e-05, b=0.0001755, mean=0.5971, pieces=1, tail=false. linear segment passed geometry and the trend cap.
- Hybrid B: members 761 (720 outside Baseline), indices [0, 761), omega [0.8000, 1.1043], R=0.0001549, S=1.079e-05, b=-3.546e-05, mean=0.5971, pieces=6, tail=false. adjacent merge recovered pieces that fail on their own.
- Hybrid B: members 28 (28 outside Baseline), indices [875, 903), omega [1.1504, 1.1612], R=0.0001079, S=1.045e-05, b=0.0009665, mean=0.5971, pieces=3, tail=false. adjacent merge recovered pieces that fail on their own.

**p=0.80, delta=0.200, K=0.70.** No plateau beyond Baseline.

**p=0.80, delta=0.200, K=1.40.** No plateau beyond Baseline.

### Representative shapes

- Global locking shape: p=1.00, delta=0.200, K=0.65, range=7.51e-05, min 80-point range=1.32e-06. Method C returned 43 segments (0 constant, 43 linear). States: Locking / Locking / Locking.
- Linear drift: p=1.00, delta=0.200, K=0.10, range=0.4021, min 80-point range=0.03146. Method C returned 3 segments (0 constant, 3 linear). States: Incoherence / Incoherence / Incoherence.
- Locally flat, globally drifting: p=1.00, delta=0.200, K=0.60, range=0.09347, min 80-point range=6.089e-07. Method C returned 58 segments (0 constant, 58 linear). States: Incoherence / Partial Locking / Partial Locking.
- Flat prefix with a high-frequency tail: p=0.40, delta=0.200, K=1.70, range=0.2464, min 80-point range=2.591e-05. Method C returned 39 segments (9 constant, 30 linear). States: Partial Locking / Partial Locking / Partial Locking.
Roles not found in the pre-specified probe pool: More than one flat window.

The figure `hybrid_method_c_benchmark.png` shows one validation embedded plateau, one validation slow slope, one validation quadratic, and the four requested real cells. Shaded blue intervals are Hybrid A. Hatched orange intervals are Hybrid B. Gray lines are Method C cuts. Pale green, on the synthetic panels, is the generating plateau.

## 8. False-positive analysis

Hybrid A: No noiseless development slope profile was accepted.
Hybrid B: No noiseless development slope profile was accepted.
Validation profile rates, including archived constant segments, are slow 0/0, nonlinear 0/0, drift 0/0 for A/B. Rates added beyond Baseline are slow 0/0, nonlinear 0/0, drift 0/0.
A new real plateau is a false-positive candidate when it is a short locally flat piece of a curve, or when its rotation-number range is carried by a few oscillators rather than by the fitted trend. The named cells are diagnostics, not labels.

## 9. Comparison with archived Method C

Development positive plateaus: 108. Baseline hits: 66. Missed because a linear segment already passes geometry and tau_S: 9. Missed because the covering pieces are below the archived size or span cut: 0. Missed because the true interval itself fails archived geometry: 33. Mixed or rejected by the trend cap: 0.

On the synthetic positive families, Baseline recall of generating plateaus is 0.6111 on development, 0.8333 on validation, and 0.8182 on test. Global Locking does not depend on these plateaus: a profile whose full range is at most 0.01 stays Locking whether or not a linear segment is relabeled.

## 10. Recommendation and limitations

Of 42 development positive plateaus missed by Baseline, 9 are covered by linear segments that pass geometry and the trend cap, 0 are broken into pieces below the archived size or span cut, 33 fail the archived geometry on the true interval itself, and 0 are mixed.
On the validation split, Hybrid A long-plateau recall is 1.0000 against a baseline positive-plateau recall of 0.8333. The fraction of baseline misses recovered by Hybrid A is 0.3333 (6 missed generating plateaus). On the locked test split the same recovery fraction is 0 for Hybrid A and 0 for Hybrid B.
Hybrid B does not improve validation long-plateau recall or validation miss recovery over Hybrid A. On development it does raise long recall from the Hybrid A value to 1, including the length-200 embedded profiles, by joining linear pieces that already pass individually. That is a coverage gain, not evidence that merging is required on the held-out splits.
Validation false-positive profile rates for Hybrid A are drift 0, slow 0, nonlinear 0. The part of those rates that is not already a Baseline constant segment is drift 0, slow 0, nonlinear 0. Hybrid B added rates are drift 0, slow 0, nonlinear 0.
The development stable tau_S values are 0.0001, 0.000213, 0.000454, 0.000969. The representative value is the smallest of these. It sits on the lower edge of the grid, so a still smaller cap was not tested. At that edge the piece guard does not change the synthetic scores, and the pre-registered rule therefore leaves the guard off. The guard is not identified. It starts to matter once tau_S approaches 0.001, where a loose guard lets merges accept pieces of a quadratic.
Alpha, beta, minimum segment length, Rule 3 geometry, and the state priority were held fixed. N is 1000 in both the saved grid and the synthetic design, so the member-count cut has no second-N check. Noise of 0.01 pushes many true plateaus over the archived range cap 0.02. Neither hybrid can recover those without changing that cap, and the stress recall is the measure of that limit.
These results do not support writing either hybrid into Classifier v2. Hybrid A recovers some linear segments that Baseline drops, and at tau_S <= 0.000454 it does not add slow-slope or nonlinear false positives beyond Baseline on the development set. The same operating point still turns several real incoherent profiles into Partial Locking because Method C has cut an almost flat body into linear pieces. Those state changes are diagnostics. They were not used to choose tau_S, and they are not a confirmed lock. Hybrid B is not clearly better on the validation split, and on the real curved profiles it builds extra local plateaus, including one interval whose range is carried by a 4-oscillator spike. A full-database shadow pass is cheap, but it would not answer the scientific question that this benchmark leaves open. It was not run.

### Stage C cost, not run

The Stage 1 index has 6964 rows, 6962 completed. Archived segmentation details do not cover this rectangle. Mean Method C time on the synthetic profiles was 0.015 s per window. Mean time on the 14 real W3 profiles, including acceptance, was 0.023 s. A full shadow pass over 6964 runs and 3 windows is about 5.2 minutes if the synthetic timing holds, or 8.0 minutes using the real-profile timing. Only plateau acceptance and the existing state rules would be reapplied. The ODE would not be rerun. Memory stays at one profile at a time.

Stage C was not run. No hybrid rule was written into Classifier v1 or the v2 candidate.
