"""Separated-penalty wrapper around the existing Method C dynamic program.

The original ``segment_profile`` implementation is not modified.
``cost_audit.dp_penalized`` is the same recurrence with the two segment
penalties supplied by the caller. This module only chooses those two numbers.

Original objective, with ``p_constant = 1`` and ``p_linear = 2``::

    J_old = sum_G RSS_m(G) + tau * sigma_hat^2 * ln(N) * (1 + p_m)

Separated objective::

    J_new = sum_G RSS_m(G) + sigma_hat^2 * ln(N) * (alpha * M + beta * L)

``M`` is the number of segments and ``L`` is the number of linear segments.
The penalties that implement ``J_new`` are

    pen_constant = alpha * sigma_hat^2 * ln(N)
    pen_linear   = (alpha + beta) * sigma_hat^2 * ln(N)

so each segment pays ``alpha`` and each linear segment pays an extra ``beta``.
``alpha = 2 * tau`` and ``beta = tau`` reproduce ``J_old``.

Ties inside the shared dynamic program keep the constant model when the two
model costs are equal, and ``argmin`` keeps the earliest admissible start
when two starts have the same total cost.
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np

METHOD_C_DIR = Path(
    "/Users/mohan/Desktop/SRS/data_examine/experiments/plateau_detection_pilot"
)
METHOD_C_FILE = METHOD_C_DIR / "pilot.py"


def load_method_c():
    """Import Method C without executing the rho-clustering classifier."""
    if not METHOD_C_FILE.is_file():
        raise FileNotFoundError(f"Method C file is missing: {METHOD_C_FILE}")
    if "observe_plateaus" not in sys.modules:
        stub = types.ModuleType("observe_plateaus")

        def find_rho_peaks(*_args, **_kwargs):
            raise RuntimeError("penalty calibration does not call find_rho_peaks")

        stub.find_rho_peaks = find_rho_peaks
        sys.modules["observe_plateaus"] = stub
    sys.path.insert(0, str(METHOD_C_DIR))
    pilot = importlib.import_module("pilot")
    cost_audit = importlib.import_module("cost_audit")
    return pilot, cost_audit


def penalty_unit(n: int, sigma2: float) -> float:
    return float(sigma2 * math.log(n))


def separated_penalties(n: int, sigma2: float, alpha: float, beta: float) -> tuple[float, float, float]:
    """Return ``(unit, pen_constant, pen_linear)`` for ``J_new``."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    unit = penalty_unit(n, sigma2)
    pen_constant = float(alpha * unit)
    pen_linear = float((alpha + beta) * unit)
    return unit, pen_constant, pen_linear


def original_penalties(n: int, sigma2: float, tau: float) -> tuple[float, float, float]:
    """Penalties written in the same order as ``pilot.segment_profile``."""
    log_n = math.log(n)
    pen_constant = tau * sigma2 * log_n * (1.0 + 1.0)
    pen_linear = tau * sigma2 * log_n * (1.0 + 2.0)
    unit = sigma2 * log_n
    return float(unit), float(pen_constant), float(pen_linear)


def equivalent_alpha_beta(tau: float) -> tuple[float, float]:
    return 2.0 * float(tau), float(tau)


def fit_values(omega: np.ndarray, y: np.ndarray, segments: list[tuple[int, int, int]]) -> np.ndarray:
    """Same fitted curve as ``pilot.segment_profile``: mean, or ``np.polyfit``."""
    fitted = np.empty(y.size, dtype=float)
    for start, end, mode in segments:
        if mode == 0:
            fitted[start:end] = float(np.mean(y[start:end]))
        else:
            slope, intercept = np.polyfit(omega[start:end], y[start:end], 1)
            fitted[start:end] = intercept + slope * omega[start:end]
    return fitted


def segment_rss(cost_audit, omega: np.ndarray, y: np.ndarray, segments) -> float:
    rss = 0.0
    for start, end, mode in segments:
        stats = cost_audit.block_stats(omega, y, start, end)
        rss += stats["rss_linear"] if mode == 1 else stats["rss_constant"]
    return float(rss)


def segment_original(pilot, omega: np.ndarray, y: np.ndarray, sigma2: float, tau: float, min_len: int):
    segments, fitted, objective = pilot.segment_profile(omega, y, sigma2, tau, min_len)
    return list(segments), fitted, float(objective)


def segment_separated(
    cost_audit,
    omega: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    alpha: float,
    beta: float,
    min_len: int,
):
    """Run the existing DP under ``J_new`` and attach the original fitted curve."""
    _unit, pen0, pen1 = separated_penalties(len(y), sigma2, alpha, beta)
    segments, objective = cost_audit.dp_penalized(omega, y, pen0, pen1, min_len)
    fitted = fit_values(omega, y, segments)
    rss = segment_rss(cost_audit, omega, y, segments)
    n_linear = sum(mode == 1 for _start, _end, mode in segments)
    penalty = pen0 * (len(segments) - n_linear) + pen1 * n_linear
    if abs((rss + penalty) - objective) > 1e-6 * max(1.0, abs(objective)):
        raise RuntimeError("separated objective does not match RSS + alpha*M + beta*L")
    return list(segments), fitted, float(objective), float(pen0), float(pen1), float(rss)


def segmentation_equal(left, right) -> bool:
    return [(int(a), int(b), int(c)) for a, b, c in left] == [
        (int(a), int(b), int(c)) for a, b, c in right
    ]
