# Classifier v1.0.0-pilot algorithm specification

This version assigns one W3 state to every completed Stage 1 pilot run.
`alpha = 2` and `beta = 1` stay provisional. The decision rules below are
the rules used for this classification. An earlier incomplete stop in this
same directory did not assign states and has been replaced.

## A. Definitions

The rotation number `rho` is the saved finite-time value. This classifier
does not recompute it.

An oscillator is near zero when `abs(rho) < 0.001`.

The near-zero fraction `f0` is the fraction of oscillators that are near
zero. Death requires `f0 = 1`.

A partial-death run is at least 15 consecutive near-zero oscillators after
sorting by increasing `omega`.

Locking uses the whole profile: `max(rho) - min(rho) <= 0.01`. Every
oscillator is inside that band, so there is no extra coverage fraction.

A candidate plateau is a constant segment from Method C. A linear segment is
not searched again. A candidate is a validated nonzero plateau when all of
the following hold:

- at least 15 oscillators;
- `(max omega in the segment - min omega in the segment) / (max omega - min omega) >= 0.02`;
- `max rho - min rho` inside the segment is at most 0.02;
- `abs(mean rho) >= 0.001`.

Segments that pass the first three tests with `abs(mean rho) < 0.001` are
recorded as validated near-zero plateaus. They do not by themselves assign
Partial Locking.

`locked_fraction` is 1 when the W3 state is Locking. Otherwise it is the
fraction of oscillators covered by validated nonzero plateaus.

## B. Method C

```
J = sum_G RSS_m(G) + sigma_hat^2 * ln(N) * (alpha * M + beta * L)
```

with `alpha = 2` and `beta = 1`. `M` is the number of segments and `L` is
the number of linear segments. Constant pieces use the mean. Linear pieces
use an ordinary-least-squares line. The dynamic program is the unchanged
`cost_audit.dp_penalized` routine. Minimum segment length is 4.

```
sigma_hat = max(1.4826 * MAD(second difference of rho) / sqrt(6), 1e-12)
```

Equal model costs keep the constant model. Equal path costs keep the
earliest start.

## C. Procedure for one window

Input: frequency-sorted `omega` and one rotation-number row.

1. Sort by `omega` with a stable argsort when the saved order is not
   nondecreasing. Windows are selected by their bounds: W1 `[8000, 10000]`,
   W2 `[10000, 12000]`, W3 `[12000, 14000]`.
2. Compute the near-zero mask, `f0`, the longest near-zero run, and the
   profile range.
3. Segment with Method C at `alpha = 2`, `beta = 1`.
4. Mark Rule 3 segments and the nonzero subset.
5. Assign the first matching state:
   - Death, if `f0 = 1`;
   - Locking, if the profile range is at most 0.01;
   - Partial Death, if the longest near-zero run has at least 15 oscillators;
   - Partial Locking, if at least one validated nonzero plateau exists;
   - Incoherence otherwise.

Output: the state, fractions, segment table, and fit.

## D. Across windows

`assigned_state` is the W3 state. W1 and W2 are not a vote and do not
replace W3.

`temporal_review_flag` is set when the three states are not the same, or
when the three validated-nonzero-plateau counts are not the same. The
reason string is `temporal_state_disagreement`,
`temporal_plateau_count_disagreement`, or both. A failed profile is stored
with `execution_status = failed` and `review_reason = classification_failed`.

## E. What this version does not use

The summer `classify_state` thresholds are not applied. In particular there
is no 95% coverage rule, no absolute omega span of 0.03, and no identity
test against `rho = omega`. A constant segment shorter than 15 oscillators
remains a fit piece and is not a validated plateau.

## F. Known limits

`alpha` and `beta` were not frozen by Stage 2C. The Rule 3 numbers were
specified directly; constant-segment length, omega span, and rho span on
the pilot have no empty gap that selects 15, 0.02, and 0.02. Plateaus of
about 25 oscillators were unreliable in the synthetic benchmark at
`sigma >= 0.01`. The noise estimator hits its floor on nearly flat profiles
and underestimates positively correlated residuals. A review flag records
window disagreement. It is not a probability of a state.
