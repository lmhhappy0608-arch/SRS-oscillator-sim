# Classifier changelog

## v2.0.0 — 2026-11-09

Status: complete. This directory is the first installed Classifier v2.

The earlier slope-homogeneous detector, with its own split threshold,
`r_max = 0.05`, and `s_max = 0.005`, is not this version. Its proposal,
shadow table, and frozen parameter file have been removed.

Relative to archived Classifier v1.0.0-pilot:

- A Method C linear segment that passes Rule 3 is a validated plateau when
  `S = |slope| * omega_span <= 1e-4`.
- Constant segments, Rule 3, alpha, beta, minimum segment length, Death,
  Locking, Partial Death, and the state priority are unchanged.
- Adjacent segments are not merged.

The ODE was not rerun. Classification uses the completed saved profiles.
