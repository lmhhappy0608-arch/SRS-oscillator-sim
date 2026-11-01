# Classifier changelog

## v1.0.0-pilot — 2026-11-01

Status: complete. This directory was finished in place.

The earlier files in `stage2_classifier/classifier_v1/` were an incomplete
unlabeled stop. They were not a finished classification, and this version
replaces them.

This completed archive:

- Uses W3 to assign the state. W1 and W2 set `temporal_review_flag` when
  the states differ or the validated nonzero plateau counts differ.
- Uses near-zero tolerance `0.001`, Death at `f0 = 1`, Locking at profile
  rho range `<= 0.01`, and Partial Death at 15 consecutive near-zero
  oscillators.
- Validates a nonzero plateau as a Method C constant segment with at least
  15 oscillators, relative omega span at least 0.02, rho span at most 0.02,
  and absolute mean level at least 0.001.
- Uses priority Death, Locking, Partial Death, Partial Locking, Incoherence.
- Keeps `alpha = 2` and `beta = 1` provisional.
- Classifies the 732 saved pilot profiles. The ODE was not rerun.

Do not overwrite this completed directory. A later change belongs in a new
directory.
