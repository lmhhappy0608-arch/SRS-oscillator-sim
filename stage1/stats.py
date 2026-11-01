"""Descriptive summaries of saved rotation profiles.

These quantities are not state labels. A tolerance is always returned with
the statistic that depends on it. Nothing here assigns Death, Locking,
Partial Locking, Partial Death, or Incoherence.
"""

from __future__ import annotations

import numpy as np


def _as_profile(rotation_number: np.ndarray) -> np.ndarray:
    profile = np.asarray(rotation_number, dtype=np.float64)
    if profile.ndim != 1:
        raise ValueError("expected one rotation-number profile of shape (N,)")
    return profile


def near_zero_fraction(rotation_number: np.ndarray, tolerance: float) -> dict:
    """Fraction of oscillators with ``|rho| < tolerance``.

    ``tolerance`` is part of the result. The fraction is descriptive only.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    profile = _as_profile(rotation_number)
    return {
        "quantity": "near_zero_fraction",
        "tolerance": float(tolerance),
        "value": float(np.mean(np.abs(profile) < float(tolerance))),
        "interpretation": "descriptive",
    }


def rotation_range(rotation_number: np.ndarray) -> dict:
    profile = _as_profile(rotation_number)
    return {
        "quantity": "rotation_range",
        "min": float(np.min(profile)),
        "max": float(np.max(profile)),
        "span": float(np.max(profile) - np.min(profile)),
        "interpretation": "descriptive",
    }


def rotation_distribution(rotation_number: np.ndarray, bins: int = 20) -> dict:
    if bins < 1:
        raise ValueError("bins must be positive")
    profile = _as_profile(rotation_number)
    counts, edges = np.histogram(profile, bins=int(bins))
    return {
        "quantity": "rotation_distribution",
        "bins": int(bins),
        "counts": counts.astype(np.int64),
        "edges": edges.astype(np.float64),
        "interpretation": "descriptive",
    }


def sort_by_frequency(omega: np.ndarray, *arrays: np.ndarray) -> dict:
    """Return a frequency sort that moves every array with the same permutation.

    The permutation contains the original oscillator indices. This does not
    rewrite saved runs.
    """
    frequencies = np.asarray(omega, dtype=np.float64)
    if frequencies.ndim != 1:
        raise ValueError("omega must have shape (N,)")
    order = np.argsort(frequencies, kind="mergesort")
    sorted_arrays = []
    for array in arrays:
        values = np.asarray(array)
        if values.shape[-1] != order.size:
            raise ValueError("each array's last axis must index the same oscillators as omega")
        sorted_arrays.append(values[..., order])
    return {
        "permutation": order.astype(np.int64),
        "oscillator_index": order.astype(np.int64),
        "omega": frequencies[order],
        "arrays": tuple(sorted_arrays),
        "interpretation": "descriptive",
    }


def local_slope_statistics(omega: np.ndarray, rotation_number: np.ndarray) -> dict:
    """Finite differences after sorting ``omega`` and ``rho`` together."""
    profile = _as_profile(rotation_number)
    ordered = sort_by_frequency(omega, profile)
    omega_sorted = ordered["omega"]
    rho_sorted = ordered["arrays"][0]
    gaps = np.diff(omega_sorted)
    if gaps.size == 0 or np.any(gaps <= 0.0):
        return {
            "quantity": "local_slope",
            "defined": False,
            "reason": "natural frequencies are not strictly increasing",
            "interpretation": "descriptive",
        }
    slopes = np.diff(rho_sorted) / gaps
    return {
        "quantity": "local_slope",
        "defined": True,
        "oscillator_index_left": ordered["oscillator_index"][:-1],
        "oscillator_index_right": ordered["oscillator_index"][1:],
        "slope": slopes,
        "median_slope": float(np.median(slopes)),
        "interpretation": "descriptive",
    }


def window_difference(earlier: np.ndarray, later: np.ndarray) -> dict:
    first = _as_profile(earlier)
    second = _as_profile(later)
    if first.shape != second.shape:
        raise ValueError("window profiles must share the oscillator axis")
    delta = second - first
    return {
        "quantity": "window_difference",
        "difference": delta,
        "max_abs": float(np.max(np.abs(delta))),
        "interpretation": "descriptive",
    }


def order_parameter_statistics(order_parameter: np.ndarray) -> dict:
    values = np.asarray(order_parameter, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("order_parameter must be a non-empty one-dimensional sample")
    return {
        "quantity": "order_parameter",
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "final": float(values[-1]),
        "count": int(values.size),
        "interpretation": "descriptive",
    }
