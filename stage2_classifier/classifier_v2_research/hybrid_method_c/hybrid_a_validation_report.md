# Hybrid A validation

Hybrid A keeps the archived Method C segmentation and Rule 3. A constant segment is accepted exactly as before. A linear segment is accepted only when it also satisfies `S_I <= 0.0001`. Segments are not merged. Death, Locking, Partial Death, and the state priority are unchanged. The Winfree ODE was not rerun, and the Stage 1 index was opened read-only.

`0.0001` is the candidate under test. It was not re-optimized here. The five named diagnostic cells are sanity checks. They are not ground truth and are not used as a false-positive rate.

Three kinds of evidence are kept separate below: numerical plateau structure on saved rotation numbers, synthetic profiles with a known flat interval, and long-time frequency locking. Agreement of W1 `[8000, 10000]`, W2 `[10000, 12000]`, and W3 `[12000, 14000]` is a finite-window check. It does not establish long-time locking. The state is assigned from W3.

## Database audit

The index contains 6964 runs, not the expected 7324. All 6964 output files are present and have finite natural frequencies and three rotation-number windows. The status field says 6962 completed, one running, and one pending. The two non-completed rows are `p=0.2, delta=1, K=1.95` and `p=0.2, delta=1, K=2`. Their saved profiles are valid, so they are included. Both lie at `delta=1`, where Hybrid A accepts no linear plateau, so they do not affect the plateau counts.

Every profile has `N=1000`. The coupling values are `p in {0.2, 0.4, 0.8, 1}`. No dependence on `N` can be estimated.

A startup check compared the fast dynamic program with archived `segment_profile` on three completed profiles. The cuts matched. Across the full index, Hybrid A never dropped a constant plateau, and it never changed a Death, Locking, or Partial Death label.

## Full-database screening

At `tau_S = 0.0001`:

| Quantity | Count |
|---|---:|
| Constant plateaus, unchanged Rule 3 | 630 |
| Accepted linear plateaus | 1781 |
| Runs containing a linear plateau | 425 |
| Linear segments failing geometry or the nonzero test | 118375 |
| Linear segments passing geometry and nonzero, but `S_I > 0.0001` | 658 |

The 1781 linear plateaus sit on three kinds of run:

| Host state before Hybrid A | Linear plateaus | Runs |
|---|---:|---:|
| Locking | 1610 | 361 |
| Partial Locking | 139 | 58 |
| Incoherence, which becomes Partial Locking | 32 | 6 |

Death and Partial Death contain none. A linear plateau inside Locking cannot change the label, because the full profile already has rotation-number range at most 0.01.

Accepted linear plateaus are numerically very flat. Median length is 77 members (minimum 21). Median `S_I` is `1.12e-5` and the maximum is just under `1e-4`. Median rotation-number range `R_I` is `1.46e-5` and the maximum is `5.69e-4`. Median relative frequency span is 0.076. Of the 1781 plateaus, 801 have relative span at least 0.10, including 9 of the 32 plateaus on newly labeled runs.

On the same index interval, 1653 of 1781 linear plateaus also pass the Hybrid A rule on both W1 and W2. Another 81 pass on exactly one earlier window. The remaining 47 pass on W3 only.

## State transitions

| Baseline | Hybrid A | Runs |
|---|---|---:|
| Death | Death | 1809 |
| Locking | Locking | 522 |
| Partial Death | Partial Death | 1717 |
| Partial Locking | Partial Locking | 124 |
| Incoherence | Incoherence | 2786 |
| Incoherence | Partial Locking | 6 |

No other transition occurs. In particular, no Partial Locking run returns to Incoherence.

All 1781 linear plateaus occur at `delta <= 0.20`. For `delta >= 0.225` the linear-plateau count is zero. By `p`, the counts are 965 at `p=1`, 316 at `p=0.4`, 284 at `p=0.8`, and 216 at `p=0.2`. The six label changes are three at `p=1, delta=0.20` (`K=0.60, 0.70, 0.75`), one at `p=0.8, delta=0.20, K=0.90`, and two at `p=0.2` (`delta=0.075, K=1.05` and `delta=0.10, K=1.50`).

The six new Partial Locking runs have a median of 64.5 Method C segments. Runs that stay Incoherence have a median of 7. The new labels are therefore concentrated on profiles that Method C cuts finely, and on `delta <= 0.20`. They are not spread through the high-delta drifting region.

## Where the new plateaus are flat

A linear segment was flagged as a continuing-slope fragment when its relative frequency span was below 0.10, the line fitted to its neighboring segments had trend above 0.001, and the segment lay on that line to within `max(0.002, 2 R_I)`. This flag fired on 0 of 1781 accepted linear plateaus. Neighboring segments are flat on the same scale as the accepted piece. That screen does not by itself prove the piece is a locking plateau. It does say the accepted pieces are not local scraps of a visible slope.

The four sampled drift controls, with absolute correlation at least 0.98 and rotation-number range from 0.21 to 1.89, produce no geometry-passing linear candidate at all. On an unsplit drift, `S_I` of a long piece is essentially the rotation-number range, which fails both Rule 3 and `tau_S`.

## Stratified sample

The sample has 25 runs. New Partial Locking has only 6 runs in the database, so that category is complete rather than filled out to 8. The plateau-like Incoherence category has 5 runs, and all 5 are included. The other categories were drawn across `p` and `delta`. The named sanity cells that were not already in the new-label list are extra and are not used to estimate an error rate.

| Category | p | delta | K | Baseline | Hybrid A | Segments | Linear plateaus | Members covered |
|---|---:|---:|---:|---|---|---:|---:|---:|
| New Partial Locking | 1.0 | 0.20 | 0.60 | Incoherence | Partial Locking | 58 | 16 | 657 |
| New Partial Locking | 1.0 | 0.20 | 0.70 | Incoherence | Partial Locking | 132 | 2 | 45 |
| New Partial Locking | 1.0 | 0.20 | 0.75 | Incoherence | Partial Locking | 137 | 3 | 95 |
| New Partial Locking | 0.8 | 0.20 | 0.90 | Incoherence | Partial Locking | 71 | 4 | 620 |
| New Partial Locking | 0.2 | 0.075 | 1.05 | Incoherence | Partial Locking | 12 | 5 | 967 |
| New Partial Locking | 0.2 | 0.10 | 1.50 | Incoherence | Partial Locking | 9 | 2 | 375 |
| Plateau-like Incoherence | 0.4 | 0.20 | 1.45 | Incoherence | Incoherence | 62 | 0 | 0 |
| Plateau-like Incoherence | 0.4 | 0.15 | 1.00 | Incoherence | Incoherence | 77 | 0 | 0 |
| Plateau-like Incoherence | 0.4 | 0.175 | 1.45 | Incoherence | Incoherence | 8 | 0 | 0 |
| Plateau-like Incoherence | 0.4 | 0.175 | 1.50 | Incoherence | Incoherence | 10 | 0 | 0 |
| Plateau-like Incoherence | 0.4 | 0.175 | 1.65 | Incoherence | Incoherence | 9 | 0 | 0 |
| Near tau_S | 0.2 | 0.05 | 1.00 | Locking | Locking | 4 | 1 | 395 |
| Near tau_S | 0.4 | 0.10 | 1.05 | Locking | Locking | 2 | 0 | 0 |
| Near tau_S | 0.8 | 0.15 | 0.95 | Partial Locking | Partial Locking | 55 | 1 | 71 |
| Near tau_S | 0.8 | 0.20 | 0.95 | Partial Locking | Partial Locking | 71 | 2 | 244 |
| Drift control | 0.2 | 0.10 | 1.00 | Incoherence | Incoherence | 2 | 0 | 0 |
| Drift control | 0.4 | 0.30 | 1.00 | Incoherence | Incoherence | 3 | 0 | 0 |
| Drift control | 0.8 | 0.60 | 0.50 | Incoherence | Incoherence | 30 | 0 | 0 |
| Drift control | 1.0 | 0.90 | 0.10 | Incoherence | Incoherence | 28 | 0 | 0 |
| Previous Partial Locking | 0.2 | 0.05 | 0.70 | Partial Locking | Partial Locking | 10 | 4 | 970 |
| Previous Partial Locking | 0.4 | 0.20 | 1.50 | Partial Locking | Partial Locking | 52 | 0 | 278 |
| Previous Partial Locking | 0.8 | 0.225 | 0.80 | Partial Locking | Partial Locking | 17 | 0 | 119 |
| Previous Partial Locking | 1.0 | 0.225 | 0.75 | Partial Locking | Partial Locking | 55 | 0 | 25 |
| Sanity, already Locking | 1.0 | 0.20 | 0.65 | Locking | Locking | 43 | 17 | 768 |
| Sanity, stays Incoherence | 1.0 | 0.15 | 0.40 | Incoherence | Incoherence | 45 | 0 | 0 |

Labels below are structural readings of the saved curves. They are not a correct/incorrect score against the archived classifier.

### New Partial Locking

`p=1, delta=0.20, K=0.60` is a broad shelf at rotation number about 0.726, followed by a sharp high-frequency tail. The 16 accepted pieces cover 657 members of that shelf. Each piece has `R_I` and `S_I` of a few times `1e-6`. W1 and W2 agree at about `1e-5`. The tail is not accepted. This is a genuine shelf that Method C cut into linear pieces. Reading: genuine plateau structure.

`p=0.8, delta=0.20, K=0.90` is the same kind of shelf, at about 0.502, covering 620 members, with a noisy tail left out. The longest piece fails the same rule on W1 (`S` about `3.3e-4`, mean absolute member difference about 0.002). W2 passes. Reading: genuine shelf on W3, with a real window disagreement.

`p=0.2, delta=0.075, K=1.05` accepts 967 of 1000 members at about 0.973. The profile range is 0.032, so it misses Locking only because of a high-frequency spike. All five pieces pass W1 and W2. Reading: genuine broad shelf.

`p=0.2, delta=0.10, K=1.50` has the same visual shelf, but only 375 members are accepted, almost all in one segment of length 347. Two linear segments elsewhere on the shelf fail the trend test. Windows pass. Reading: genuine shelf, incompletely covered.

`p=1, delta=0.20, K=0.75` has a long shelf and a steep tail. Method C makes 137 linear segments, and 134 fail the geometry rule. The three accepted pieces are adjacent, lengths 22, 28, and 45, and they sit on the shelf rather than on the tail. Their ranges are about `2e-5` to `5e-5`. Reading: ambiguous. The shelf is real, but Partial Locking is triggered by three short fragments of it.

`p=1, delta=0.20, K=0.70` is almost flat: the W3 range is 0.0157, just above the Locking cap, with a small rise at the high-frequency end. W1 sits about 0.001 away from W3. Only two segments are accepted, lengths 21 and 24. They are locally flat, not a visible slope. Reading: ambiguous. The label is carried by two short scraps of a nearly global shelf.

Across these 32 plateaus, none shows a visible systematic slope. Five of them, the two at `K=0.70` and the three at `K=0.75`, are short fragments that do not cover the shelf they sit on.

### Plateau-like Incoherence

All five stay Incoherence because their flattest geometry-passing linear segment has `S_I` between `1.17e-4` and `3.46e-4`.

`p=0.4, delta=0.20, K=1.45` has a wide shelf between noisy ends. `p=0.4, delta=0.15, K=1.00` has a shelf after a noisy ramp and before a noisy rise. The three runs at `p=0.4, delta=0.175, K=1.45, 1.50, 1.65` are almost entirely flat, with a few high-frequency spikes; the shelf is rejected as one piece whose trend is about `2e-4`. Reading for all five: ambiguous missed shelves, not slope false alarms. These are exactly the five runs that would become Partial Locking if `tau_S` were raised to `0.0005`.

### Near the candidate threshold

`p=0.2, delta=0.05, K=1.00` is already Locking, with full range `1.9e-4`. One linear piece of 351 members is accepted. W1 and W2 tilt in opposite directions at that same tiny scale. The state does not change.

`p=0.4, delta=0.10, K=1.05` is Locking and accepts nothing. Its smallest candidate `S_I` is `1.27e-4`.

`p=0.8, delta=0.15, K=0.95` and `p=0.8, delta=0.20, K=0.95` are already Partial Locking because of a constant segment on a shelf. Hybrid A adds one or two short linear pieces on that same shelf. At `K=0.95, delta=0.20`, one added piece differs from W2 by about 0.018 in mean absolute rotation number even though the piece itself is flat on each window. Reading: the added pieces are ambiguous fragments; they do not create the Partial Locking label.

### Drift controls

The four controls are single rising curves, with two to 30 segments and no accepted plateau. Method C does not manufacture a `tau_S`-passing scrap on these profiles. Reading: clear drifts, correctly left as Incoherence by Hybrid A.

### Previous Partial Locking and the two extra sanity cells

`p=0.2, delta=0.05, K=0.70` was already Partial Locking from a constant segment. Hybrid A adds linear pieces that fill 970 members of the same shelf.

`p=0.4, delta=0.20, K=1.50` stays Partial Locking because of constant segments, including one of length 202 on the shelf and three short constant segments near the noisy end. Hybrid A adds no linear plateau.

`p=0.8, delta=0.225, K=0.80` and `p=1, delta=0.225, K=0.75` stay Partial Locking because of one constant segment each, of lengths 119 and 25, sitting on a bend in an otherwise rising curve. Hybrid A accepts no linear piece. These two labels are archived Rule 3 behavior. They are not a Hybrid A result.

`p=1, delta=0.20, K=0.65` stays Locking. Its range is `7.5e-5`. Seventeen linear pieces cover 768 members of a curve that is already globally flat. The linear rule is redundant with Locking.

`p=1, delta=0.15, K=0.40` stays Incoherence. The curve is two slopes with a flat step between them. The flattest geometry-passing linear segment has `S_I = 8.1e-4`, so it is rejected at every tested threshold through `0.0005`. This cell was previously described as an Incoherence example. Visually it contains a shelf. It is not used as ground truth. It is evidence that `tau_S = 0.0001` also misses shelves.

## Windows

For the broad new shelves at `K=0.60`, `K=1.05`, and `K=1.50`, member-wise absolute differences between W3 and the earlier windows are about `1e-5` to `6e-4`, and the same intervals pass the rule on W1 and W2. That is numerical agreement among three finite windows.

It is not uniform. The long piece at `p=0.8, K=0.90` fails W1. The `K=0.70` profile is offset between windows by about 0.001. One short piece at `p=0.8, delta=0.20, K=0.95` differs from W2 by about 0.018. None of these comparisons is a long trajectory at fixed parameters. The saved data stop at time 14000.

## Synthetic benchmark

The existing dev, validation, and test catalogs were segmented with archived Method C and scored with Hybrid A, without merging. Dev was the split used when this candidate threshold was chosen, so the confirmation numbers below are validation plus test. A true interval counts as detected when some accepted interval has intersection-over-union at least 0.5. Boundary error is the mean absolute endpoint error on low-noise intervals of length at least 80.

Validation plus test, at `tau_S = 0.0001`, compared with constant segments only:

| Family | Profiles | Truth intervals | Hybrid A detected | Missed | Constant-only detected |
|---|---:|---:|---:|---:|---:|
| Embedded plateau | 26 | 26 | 16 | 10 | 16 |
| Embedded, noise 0.01 | 10 | 10 | 0 | 10 | 0 |
| Flat plateau | 2 | 2 | 2 | 0 | 2 |
| Plateau with drifting tails | 8 | 8 | 8 | 0 | 8 |
| Near-global plateau with a short tail | 12 | 12 | 12 | 0 | 12 |
| Slow linear slope | 8 | 0 | 0 false profiles | 0 | 0 |
| Quadratic drift | 4 | 0 | 0 false profiles | 0 | 0 |
| Linear drift | 22 | 0 | 2 false profiles | 0 | 2 |

The ten missed embedded intervals are the noise-0.01 profiles. Both rules miss them because the rotation-number range exceeds the archived cap of 0.02. On the intervals that do match, the median boundary error is 2.25 oscillators for embedded plateaus, 1.0 for tails, and 0.25 for the near-global family.

Slow slopes and quadratic drift produce no accepted plateau at `0.00005`, `0.0001`, `0.0002`, or `0.0005`, on dev as well as on validation and test. The two linear-drift false profiles are noisy (`sigma = 0.01`) and are accepted by the constant rule as well. Their accepted pieces are constant segments, not linear segments. One noisy quadratic on dev is also a constant-rule acceptance, unchanged by Hybrid A.

Hybrid A does report additional linear intervals on some noiseless embedded and tail profiles. Those intervals have `R_I = 0`. They are pieces of the true flat interval. The interval score allows only one predicted piece to match a truth interval, so the extra pieces are counted as unmatched even though their oscillators lie inside the truth plateau. They are not false plateaus on the sloping sides. The same pattern appears on dev. It is a segmentation-and-scoring effect, not evidence that `tau_S = 0.0001` accepts a slow slope.

Held-out true-interval detection is therefore the same for Hybrid A and for constant-only Method C. Hybrid A does not recover the noise-0.01 plateaus, and it does not add slope false alarms on this catalog through `tau_S = 0.0005`.

## Sensitivity of tau_S

Geometry-passing, nonzero linear segments were re-counted at four thresholds. The state change uses the smallest such `S_I` on each Incoherence run.

| tau_S | Linear plateaus | Of which n < 30 | Median n | Incoherence to Partial Locking |
|---:|---:|---:|---:|---:|
| 0.00005 | 1563 | 351 | 60 | 6 |
| 0.0001 | 1781 | 358 | 77 | 6 |
| 0.0002 | 1934 | 359 | 95 | 9 |
| 0.0005 | 1999 | 361 | 102 | 11 |

The six labels at `0.0001` are already present at `0.00005`. Raising the threshold to `0.0002` adds three Incoherence runs, and `0.0005` adds the other two. Those five runs are the plateau-like sample: visible shelves whose smallest `S_I` is `1.17e-4` to `3.46e-4`. The plateaus added between thresholds are long. Their median lengths are 282, 336, and 429. The count of segments shorter than 30 members moves only from 351 to 361.

So `0.0001` is not an unstable point in the short-scrap sense, and the synthetic slow-slope and quadratic profiles stay negative through `0.0005`. It is sensitive in the label sense: doubling it changes three database labels, and multiplying it by five changes five. Those extra labels are shelves that the visual sample also regards as plausible plateaus. The present value is a conservative edge of that group, not a threshold whose label set is locally flat. The sanity shelf at `p=1, delta=0.15, K=0.40` remains excluded even at `0.0005`.

## Failure cases

1. Short fragments decide a label. At `p=1, delta=0.20, K=0.70`, two segments of lengths 21 and 24 change Incoherence to Partial Locking. At `K=0.75`, three segments of lengths 22 to 45 do the same, while most of the visually flat body was cut into pieces that fail the archived length or span rule.

2. Similar shelves are rejected. Five Incoherence runs with a visible shelf have minimum `S_I` only 1.2 to 3.5 times the candidate threshold. The sanity curve at `p=1, delta=0.15, K=0.40` has a flat step with minimum `S_I = 8.1e-4` and stays Incoherence.

3. Window disagreement. The longest new piece at `p=0.8, delta=0.20, K=0.90` fails W1. The `K=0.70` windows are offset. One short addition at `p=0.8, delta=0.20, K=0.95` differs from W2 by about 0.018.

4. Archived Rule 3, not Hybrid A, still accepts a short flat bend on a rising curve. The clearest sampled case is `p=1, delta=0.225, K=0.75`: one constant segment of 25 members on a smooth drift, with no linear plateau added.

5. Noise at `sigma = 0.01` still defeats both rules on embedded synthetic plateaus, because `R_I` exceeds 0.02.

## Recommendation

Do not install Hybrid A as the Classifier v2 plateau rule on this evidence.

The numerical evidence says the candidate rule accepts 1781 very flat linear pieces and changes six labels, all at `delta <= 0.20`. Four of those six are broad shelves with a tail. Two are short fragments of a shelf. Plain drifting controls are not accepted. The synthetic evidence says the same threshold does not accept slow slopes or quadratic drift, and it does not detect any held-out truth interval that constant-only Method C misses. The long-time evidence is absent: the archive has three finite windows, and they already disagree on some of the new pieces.

The specific gaps are: no `N` other than 1000; no trajectory beyond time 14000; the six new labels include the three pre-specified diagnostic cells, so they are not an independent confirmation set; visually similar shelves remain Incoherence at a slightly larger `S_I`; and the held-out synthetic catalog shows no detection gain over the constant rule once Rule 3 geometry is held fixed.

## Answers

1. Hybrid A accepts 1781 linear plateaus. They occur on 425 runs. Thirty-two of them lie on the six runs whose label changes. The other 1749 lie on runs that were already Locking or Partial Locking.

2. Those plateaus change 6 runs from Incoherence to Partial Locking. No other state changes.

3. None of those 32 plateaus is an obvious local-slope false alarm. Every one has `R_I` and `S_I` below about `1.3e-4` and lies on a visually flat shelf. Five of the 32, at `K=0.70` and `K=0.75`, are short fragments that do not cover the surrounding shelf. A continuing-slope screen fired on 0 of the 1781 plateaus. This is not a claim that all 1781 were inspected visually.

4. The six label changes are stable between `0.00005` and `0.0001`. They are not stable under loosening: `0.0002` produces 9 label changes and `0.0005` produces 11. The added plateaus are long, not a burst of segments shorter than 30 members. Synthetic slow slopes and quadratic drifts remain negative through `0.0005`. The candidate is a conservative edge, and the Partial Locking count does move when the threshold is doubled.

5. Hybrid A is not clearly better than constant-only Method C. On the held-out synthetic catalog its true-interval detections match the constant rule exactly. On the saved Winfree profiles it does pick up broad shelves that Method C marked as linear segments, which is why four of the six new labels look like real shelves. It also leaves neighboring shelves undetected and lets two labels be decided by short fragments.

6. The evidence is not sufficient to adopt Hybrid A as the Classifier v2 plateau mechanism. The numerical shelf evidence and the synthetic negative controls support a conservative reading of `tau_S = 0.0001`. They do not identify the shelves that should be Partial Locking, and they do not show long-time frequency locking.
