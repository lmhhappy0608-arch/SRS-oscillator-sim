# Adopted decision rules

These choices are the working Rule 1–7 definitions. They do not replace
`classifier_v1/` (`v1.0.0-pilot`, archive status `incomplete`). No five-state
labels have been written from this file.

Method C remains separated penalties with provisional `alpha = 2`, `beta = 1`,
minimum segment length 4.

## Rule 1

W3 `[12000, 14000]` assigns the state. W1 `[8000, 10000]` and W2
`[10000, 12000]` are a temporal-stability check. Disagreement sets
`temporal_review_flag`. There is no majority vote, and W1/W2 do not replace
the W3 label.

## Rule 2

A Method C constant segment is a candidate nonzero plateau and still has to
pass Rule 3. A Method C linear segment is not searched again for a local
plateau and is not merged. Near-zero oscillators are detected without using
Method C. A profile that Method C splits entirely into linear pieces can still
be Locking under Rule 7.

## Rule 3

A constant segment `G` is valid when all three hold:

- at least 15 oscillators;
- `(max omega in G - min omega in G) / (max omega - min omega) >= 0.02`;
- `max rho in G - min rho in G <= 0.02`.

The old absolute omega-span cut of 0.03 is not used. This triple was specified
directly. The W3 histograms of constant-segment length, omega span, and rho
span had no empty gap that selected these numbers.

On the 732 saved W3 profiles, 2294 constant segments exist. 535 of them pass
Rule 3, on 459 profiles. Segments with absolute mean level below 0.001 account
for 108 of 199 passes at Delta 0.05, 109 of 152 at Delta 0.2, and all 184 at
Delta 0.8.

## Rules 4–7

- Near zero: `abs(rho) < 0.001`.
- Death: that condition holds for every oscillator (`f0 = 1`).
- Partial-death run: at least 15 consecutive near-zero oscillators. Whether
  this label is assigned before Locking or a validated plateau is still open.
- Locking: `max(rho) - min(rho) <= 0.01` on the whole profile. That condition
  already includes every oscillator, so there is no extra coverage fraction.

## Still open

State priority, the boundary between Partial Locking and Incoherence, and any
review flag other than `temporal_review_flag`.
