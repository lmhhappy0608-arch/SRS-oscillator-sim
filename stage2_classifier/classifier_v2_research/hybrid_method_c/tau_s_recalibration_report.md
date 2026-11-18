# Hybrid A tau_S recalibration

Hybrid A is otherwise fixed: archived Method C cuts, archived Rule 3, no segment merging, and the existing state priority. Only `tau_S` is recalibrated. The previous choice `tau_S = 1e-4` is not reused as an answer. Real Winfree labels are not ground truth, and the number of new Partial Locking labels is not an objective. Synthetic curves are generated profiles, not Winfree trajectories.

## Audit of the previous selection

The old search grid in `hybrid_method_c.py` is `geomspace(1e-4, 0.02, 8)`:

| tau_S | FP drift | FP slow | FP nonlinear | Long recall | First stage reached |
|---|---:|---:|---:|---:|---:|
| 0.0001 | 0.000 | 0.000 | 0.083 | 0.913 | 3 |
| 0.000213166 | 0.000 | 0.000 | 0.083 | 0.913 | 3 |
| 0.000454399 | 0.000 | 0.000 | 0.083 | 0.913 | 3 |
| 0.000968625 | 0.000 | 0.000 | 0.083 | 0.913 | 3 |
| 0.00206478 | 0.000 | 0.167 | 0.083 | 0.913 | -1 |
| 0.00440142 | 0.000 | 0.167 | 0.083 | 0.913 | -1 |
| 0.00938235 | 0.000 | 0.167 | 0.083 | 0.913 | -1 |
| 0.02 | 0.000 | 0.333 | 0.083 | 0.913 | -1 |

Stage codes follow the old script: 0 is the strict joint rule, 2 is false-positive rate at most 0.05 with long recall at least 0.50, 3 is false-positive rate at most 0.10 with long recall at least 0.50, and -1 means even stage 3 failed. The old selector took the first stage that any grid point passed, then the smallest stable `tau_S` inside it. Recomputing that grid on the original development catalog agrees with the recorded facts:

- `1e-4` is the lower edge. No smaller cap was on the grid.
- The nonlinear false-positive rate stays at 0.083 because one noisy quadratic has a Constant segment that passes Rule 3. Recorded Constant hits: seed 1: n=23, R=0.01907, S=0.001158. A linear trend cap cannot remove a Constant segment.
- That single Constant hit keeps the false-positive rate above 0.05, so the first three stages never open. The selector therefore lands in the loosest stage for every tau that has not yet started accepting slow slopes.
- Inside that stage the smallest stable value is the grid edge, `1e-4`.

This is an evaluation-design problem, not a segmentation bug. The old false-positive rate mixes baseline Constant errors with new Linear errors, and the smallest-stable rule then returns the untested boundary of the grid. The piece guard is not identified at that edge; it only belongs to the rejected merging rule.

## What the original benchmark can separate

On the original development catalog, geometry-valid linear segments of true plateaus have `S_I` at numerical zero or they do not exist. Noise of 0.01 makes the plateau fail the archived range cap, so `tau_S` cannot recover it. Noiseless slow slopes remain one segment whose `S_I` is the full rotation-number change, about `0.002` and above.

| tau_S | Recovered / recoverable | Recall | Incremental FP | FPR | Off-plateau segments | Precision | Boundary | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-06 | 3/3 | 1 | 0/47 | 0 | 0 | 1 | 3.5 | 0.859 |
| 1e-05 | 3/3 | 1 | 0/47 | 0 | 0 | 1 | 3.5 | 0.859 |
| 0.0001 | 3/3 | 1 | 0/47 | 0 | 0 | 1 | 3.5 | 0.859 |
| 0.001 | 3/3 | 1 | 0/47 | 0 | 0 | 1 | 3.5 | 0.859 |
| 0.01 | 3/3 | 1 | 3/47 | 0.0638 | 0 | 1 | 3.5 | 0.859 |

From `1e-6` through `1e-3` the incremental counts in the table do not move. On the old grid, the slow-slope false-profile rate is still zero at `1e-4` and rises to 0.167 at `0.002`. The original benchmark therefore cannot choose among `5e-5`, `1e-4`, `2e-4`, and `5e-4`. It mostly scores Constant segments and full-span slopes. The validation and test splits of that catalog were already inspected in earlier rounds, so they are reported only as history and are not used to place the new cap.

## Calibration benchmark

Because the original catalog is flat across the relevant decade, a separate calibration catalog was generated. It is not a modification of the old test set. Development, validation, and test use disjoint seeds and shifted numerical values. The master seed is `20261109`. Development uses `delta = 0.20`, validation `0.15`, and test `0.25`.

- Fragmented plateau: five length-140 blocks with a designed segment trend, separated by short steep connectors. The blocks are the truth.
- Fragmented drift: a monotone curve plus a sine wobble at 80 percent of the slope that would reverse it, so Method C can cut locally flatter pieces. There is no truth plateau.
- Slow slope: one designed trend from about `1e-5` to a few times `1e-3`, with `1e-6` noise so the noise scale does not collapse to the numerical floor.
- Nonlinear: shallow quadratics and sine bends, with no truth plateau.
- Plateau with tail: a body of designed trend and a separate tail whose trend is `5e-4`, a few times `1e-3`, or several times `1e-2`. Only the body is truth. A body trend of 0 is the Constant-segment control.
- Noisy plateau: a flat truth at several noise scales, including values that violate the Rule 3 range cap.

The selection rule was fixed before validation or test scores were consulted:

1. On development, a threshold is clean when it has zero incremental false profiles and zero accepted linear segments whose oscillators lie mostly outside the generated plateau.
2. Among clean thresholds, keep those that recover the most geometry-recoverable baseline misses.
3. If their median boundary errors differ by more than five oscillators, drop the worse ones.
4. Call the remaining set meaningful only if it recovers at least half of those recoverable misses.
5. Emit one value only when one threshold remains. Otherwise report the equivalent interval.
6. Validation may shrink that interval when it adds an incremental false profile or a mostly-off-plateau segment. The held-out test is scored after the decision and does not move it.
7. Linear segments added to a curve the baseline already flagged are reported separately. They do not enter the clean-threshold rule, because a Constant-segment error cannot be removed by `tau_S`.

## Development and validation

Development, calibration catalog:

| tau_S | Recovered / recoverable | Recall | Incremental FP | FPR | Off-plateau segments | Precision | Boundary | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-06 | 0/52 | 0 | 0/23 | 0 | 0 | na | na | 0.557 |
| 1.33352e-06 | 0/52 | 0 | 0/23 | 0 | 0 | na | na | 0.557 |
| 1e-05 | 7/52 | 0.135 | 8/23 | 0.348 | 0 | 1 | 0.5 | 0.659 |
| 5e-05 | 18/52 | 0.346 | 15/23 | 0.652 | 0 | 0.996 | 0.5 | 0.763 |
| 0.0001 | 35/52 | 0.673 | 16/23 | 0.696 | 0 | 0.999 | 0.5 | 0.871 |
| 0.0002 | 42/52 | 0.808 | 17/23 | 0.739 | 0 | 0.999 | 0.5 | 0.935 |
| 0.0005 | 52/52 | 1 | 19/23 | 0.826 | 3 | 0.999 | 0.5 | 0.952 |
| 0.001 | 52/52 | 1 | 20/23 | 0.87 | 6 | 0.999 | 0.5 | 0.941 |
| 0.01 | 52/52 | 1 | 23/23 | 1 | 37 | 0.999 | 0.5 | 0.903 |

Validation uses the same columns. It can shrink the frontier. It does not invent the rule.

| tau_S | Recovered / recoverable | Recall | Incremental FP | FPR | Off-plateau segments | Precision | Boundary | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-06 | 0/42 | 0 | 0/20 | 0 | 0 | na | na | 0.527 |
| 1.33352e-06 | 0/42 | 0 | 0/20 | 0 | 0 | na | na | 0.527 |
| 1e-05 | 0/42 | 0 | 7/20 | 0.35 | 0 | na | na | 0.527 |
| 5e-05 | 16/42 | 0.381 | 12/20 | 0.6 | 0 | 0.999 | 0.5 | 0.757 |
| 0.0001 | 32/42 | 0.762 | 14/20 | 0.7 | 0 | 0.996 | 0.5 | 0.925 |
| 0.0002 | 34/42 | 0.81 | 15/20 | 0.75 | 0 | 0.999 | 0.5 | 0.932 |
| 0.0005 | 42/42 | 1 | 16/20 | 0.8 | 0 | 0.999 | 0.5 | 0.957 |
| 0.001 | 42/42 | 1 | 18/20 | 0.9 | 6 | 0.999 | 0.5 | 0.933 |
| 0.01 | 42/42 | 1 | 20/20 | 1 | 35 | 0.999 | 0.5 | 0.886 |

Development decision: **clean_but_weak_recovery**. The only thresholds with zero incremental false profiles and zero mostly-off-plateau segments are the two adjacent grid points `[1e-06, 1.33352e-06]`. They recover **0 of 52** geometry-recoverable baseline misses (Wilson 95 percent interval 0 to 0.069). This is not a proposed operating range.

The first development threshold with an incremental false profile is `1.778e-06` (1 of 23 profiles, Wilson 0.008 to 0.210). The first with a mostly-off-plateau segment is `5e-4`. Incremental false profiles begin by family at: nonlinear at `1.778e-06`, fragmented drift at `3.162e-06`, slow slope at `1e-05`.

Eleven of the 34 development curves with no generated plateau already have a Rule 3 Constant segment, all of them shallow bends. Those 11 stay in the baseline count at every cap. By `1e-4` the cap has also added 155 linear segments on those already-flagged curves. That count is recorded separately and was not used as a penalty, because a Constant error cannot be removed by `tau_S`.

There is no threshold that raises recall while the incremental false-profile count stays at zero. The first recovered shelves appear at `1e-5`: 7 of 52, and only after 8 of 23 clean negative profiles have already been flagged (Wilson 0.188 to 0.551 on that rate). Recall then climbs with the false-profile count:

| tau_S | Recovered shelves | Incremental false profiles | Off-plateau segments |
|---|---:|---:|---:|
| 1e-06 to 1.33e-06 | 0/52 | 0/23 | 0 |
| 1e-05 | 7/52 | 8/23 | 0 |
| 5e-05 | 18/52 | 15/23 | 0 |
| 1e-04 | 35/52 | 16/23 | 0 |
| 2e-04 | 42/52 | 17/23 | 0 |
| 5e-04 | 52/52 | 19/23 | 3 |

At `1e-4` the Wilson interval is 0.538 to 0.785 for recall and 0.491 to 0.844 for the incremental false-profile rate. The 16 false profiles are 5 slow slopes, all 8 fragmented drifts, and 3 bends. Matched shelves at this cap still have median precision about 0.999 and median boundary error 0.5 oscillator. The loss is whole false profiles, not a smeared edge. Tail segments begin to pass only at about `5e-4`.

Validation has the same shape on its own, shifted trend values: zero recovery on the two clean grid points, then 32 of 42 recovered at `1e-4` with 14 of 20 incremental false profiles. It does not shrink the empty-recovery band, and it does not produce a second band that is both clean and useful. Flat iid noise never becomes a linear-segment problem: small noise is already a Constant plateau, and noise of 0.01 fails the archived range cap. Those misses are Rule 3 misses.

Validation stability: validation is also free of incremental false profiles on `[1e-06, 1.33352e-06]`, and those points still recover nothing.

## Held-out calibration test, scored once

| tau_S | Recovered / recoverable | Recall | Incremental FP | FPR | Off-plateau segments | Precision | Boundary | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-06 | 0/52 | 0 | 0/19 | 0 | 0 | na | na | 0.506 |
| 1.33352e-06 | 0/52 | 0 | 0/19 | 0 | 0 | na | na | 0.506 |
| 1e-05 | 0/52 | 0 | 5/19 | 0.263 | 0 | na | na | 0.506 |
| 5e-05 | 32/52 | 0.615 | 11/19 | 0.579 | 0 | 0.999 | 0.5 | 0.898 |
| 0.0001 | 32/52 | 0.615 | 11/19 | 0.579 | 0 | 0.999 | 0.5 | 0.898 |
| 0.0002 | 42/52 | 0.808 | 13/19 | 0.684 | 0 | 0.999 | 0.5 | 0.929 |
| 0.0005 | 52/52 | 1 | 15/19 | 0.789 | 6 | 0.999 | 0.5 | 0.937 |
| 0.001 | 52/52 | 1 | 15/19 | 0.789 | 6 | 0.999 | 0.5 | 0.937 |
| 0.01 | 52/52 | 1 | 19/19 | 1 | 42 | 0.999 | 0.5 | 0.887 |

The held-out split was scored once after the decision. It agrees that the two clean grid points recover 0 of 52 shelves and flag 0 of 19 clean negative profiles. At the old candidate `1e-4` it recovers 32 of 52 (Wilson 0.480 to 0.735) and flags 11 of 19 (Wilson 0.363 to 0.769). On this split, `5e-05` and `1e-4` land on the same counts. That tie was not used to move the decision. Full recovery, 52 of 52, arrives at `5e-4` together with 15 of 19 false profiles and 6 tail segments.

## Saved Winfree profiles

All 6964 saved profiles were segmented once. Higher-priority states do not change with `tau_S`. The only possible label change is Incoherence to Partial Locking, and only when a geometry-valid linear segment has `S_I` at or below the cap.

| tau_S | Linear plateaus | n < 30 | Incoherence to Partial Locking |
|---|---:|---:|---:|
| 1e-06 | 101 | 35 | 0 |
| 1.33352e-06 | 136 | 52 | 1 |
| 1e-05 | 838 | 235 | 1 |
| 5e-05 | 1563 | 351 | 6 |
| 0.0001 | 1781 | 358 | 6 |
| 0.0002 | 1934 | 359 | 9 |
| 0.0005 | 1999 | 361 | 11 |
| 0.001 | 2015 | 365 | 14 |
| 0.01 | 2264 | 440 | 62 |

At `1e-4` this reproduces the previous screen: 1781 linear plateaus and 6 label changes. The same screen shows that the synthetic zero-error band is not empty on the saved runs. At `1e-6` it still accepts 101 linear pieces, none of which changes a label. The next grid point, `1.33e-6`, accepts the flattest piece of `p=1, delta=0.20, K=0.60` (`S_I = 1.06e-6`, 33 oscillators, relative span 0.032) and changes that one label. The two synthetic grid points are therefore not equivalent on the database. Because neither recovers a generated shelf, that difference is not used to pick one of them.

The six label changes at `1e-4`, and the five shelves that appear between `1e-4` and `5e-4`, were read from the curves in the previous gallery. They are diagnostics. Short fragments remain visible in the table: `K=0.70` and `K=0.75` are carried by pieces of 21 and 22 oscillators sitting on the relative-span floor of 0.02. The `K=0.70` piece also differs from the first window by a mean absolute rotation of 0.0016. The five `p=0.4` shelves are longer (140 to 266 oscillators) with `S_I` from `1.17e-4` to `3.46e-4` and window-to-window rotation differences of about `1e-4` to `5e-4`. The `p=0.4, delta=0.20, K=1.45` piece is locally flat and long, while the whole profile still spans 0.36 in rotation number. `p=1, delta=0.15, K=0.40` stays out until about `8e-4`. None of these nine cells is a ground-truth label.

| p | delta | K | Baseline state | Segments | Minimum linear S | Profile range | Flattest linear segment |
|---|---:|---:|---|---:|---:|---:|---|
| 0.4 | 0.15 | 1 | Incoherence | 77 | 0.000116593 | 0.2449 | n=209, rel=0.208, R=0.000178, W1 S=0.000169, W2 S=0.000224, mean |W3-W1|=0.000539 |
| 0.4 | 0.175 | 1.45 | Incoherence | 8 | 0.000202168 | 0.1194 | n=160, rel=0.159, R=0.000285, W1 S=0.000197, W2 S=0.0002, mean |W3-W1|=0.000106 |
| 0.4 | 0.175 | 1.5 | Incoherence | 10 | 0.000198966 | 0.122 | n=140, rel=0.139, R=0.000434, W1 S=9.84e-05, W2 S=0.000101, mean |W3-W1|=0.000209 |
| 0.4 | 0.175 | 1.65 | Incoherence | 9 | 0.000197089 | 0.1507 | n=143, rel=0.142, R=0.000466, W1 S=0.000135, W2 S=0.00019, mean |W3-W1|=0.000154 |
| 0.4 | 0.2 | 1.45 | Incoherence | 62 | 0.000345642 | 0.3564 | n=266, rel=0.265, R=0.000523, W1 S=0.000182, W2 S=0.000171, mean |W3-W1|=0.00016 |
| 1 | 0.15 | 0.4 | Incoherence | 45 | 0.000808143 | 0.3255 | n=77, rel=0.076, R=0.00123, W1 S=0.000867, W2 S=0.000275, mean |W3-W1|=0.000455 |
| 1 | 0.2 | 0.6 | Incoherence | 58 | 1.05976e-06 | 0.09347 | n=33, rel=0.032, R=1.06e-06, W1 S=2.92e-06, W2 S=5.43e-07, mean |W3-W1|=1.03e-05 |
| 1 | 0.2 | 0.7 | Incoherence | 132 | 2.25038e-05 | 0.01573 | n=21, rel=0.020, R=2.25e-05, W1 S=2.63e-06, W2 S=2.23e-05, mean |W3-W1|=0.0016 |
| 1 | 0.2 | 0.75 | Incoherence | 137 | 2.22309e-05 | 0.1939 | n=22, rel=0.021, R=2.22e-05, W1 S=2.4e-05, W2 S=2.23e-05, mean |W3-W1|=1.5e-05 |

`p=1, delta=0.15, K=0.40` has a visible flat step whose flattest geometry-valid segment has `S_I` near `8e-4`. Accepting it requires a cap in that neighborhood. The same cap also accepts slower members of the synthetic drift families. Window gaps on the new pieces remain finite-window disagreements, not long-time locking.

## Recommendation

Status: **provisional, no value and no operating interval for Classifier v2**.

The selection rule does find two adjacent caps, `[1e-06, 1.33e-06]`, with zero incremental false profiles. They recover none of the 52 recoverable baseline misses. Every cap that recovers a generated shelf also flags slow slopes, locally flat pieces of a monotone drift, or both. At the previous candidate `1e-4`, development recovery is 35/52 and the incremental false-profile count is 16/23. The remaining shelves are all recovered by about `3.2e-4`, at which point 19 of 23 clean negative profiles are flagged. Designed tails start passing at `5e-4`, after that recovery is already complete.

No single value is recommended, and the zero-error band is not a recommended configuration. `tau_S` is not written into Classifier v2.

A shelf segment and a drift segment with the same `S_I` are the same object under Hybrid A. Once those trends overlap, the cap cannot keep one and reject the other. The calibration says that overlap begins below the trends of the shelves this rule was meant to recover.

## Limitations

- The calibration curves are constructed segments, not integrated Winfree trajectories. A pass rate on them is not a Winfree state accuracy.
- The only zero-error band is two adjacent points of a discrete log grid. A finer grid cannot create a gap that the segment trends do not have.
- Eleven baseline Constant false profiles sit on shallow bends. Their extra linear pieces are listed and were excluded from the acceptance rule.
- Rule 3 still rejects short pieces and any piece whose rotation-number range exceeds 0.02, whatever `tau_S` is. Noise at 0.01 on a flat plateau is mostly in that group.
- The original validation and test catalogs were already inspected and are not a fresh final test.
- The new test split was scored once, but it is small and uses the same six generators with shifted numbers.
- Every saved run has `N=1000`.
- Three finite windows are not a long-time frequency lock.
