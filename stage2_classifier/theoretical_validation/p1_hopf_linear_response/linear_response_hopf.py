"""p=1 linear-response check of the all-to-all Winfree Hopf boundary.

The continuum benchmark is Ariaratnam and Strogatz, Phys. Rev. Lett. 86, 4278
(2001). Equation (6) is the discrete-spectrum condition. Equation (7) is its
small-delta truncation. This script does not integrate the Winfree ODE, does
not read classifier labels, and does not write into the archived classifier.

The response formula is evaluated from Re(lambda) > 0 and extrapolated to the
imaginary axis. A pure imaginary probe sits on the continuous spectrum of the
transport operator, so a finite Fourier matrix on that axis is not used as a
Hopf root.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-p1-hopf-response")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EQ7_DIR = ROOT / "stage2_classifier" / "theoretical_validation" / "eq7_hopf"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PI = float(np.pi)
TARGET_DELTAS = (0.05, 0.10, 0.20)
CONTROL_DELTAS = (0.25,)
EPSILONS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
M_SWEEP = (8, 12, 16, 24, 32)
M_PRODUCTION = 32
N_OUTER = 64
N_INNER = 160

CSV_PATH = HERE / "p1_hopf_linear_response.csv"
FIGURE_PATH = HERE / "p1_hopf_linear_response.png"
JSON_PATH = HERE / "p1_hopf_linear_response_diagnostics.json"

CSV_FIELDS = [
    "delta",
    "paper_K_H",
    "response_K_H",
    "absolute_error",
    "relative_error",
    "convergence_status",
    "validity_status",
    "eq7_K_H",
    "eq7_absolute_error",
    "eq7_relative_error",
    "paper_Omega",
    "response_Omega",
    "all_drifting_edge",
    "extrapolation_uncertainty",
    "paper_source",
]


def series_eq7(delta: float) -> float:
    """Printed Eq. (7). Same product form as validate_eq7.series_eq7."""
    delta = float(delta)
    factor = 1.0 + 16.0 * delta**2 / PI**2 + 16.0 * (PI**2 + 80.0) * delta**4 / PI**4
    return (8.0 * delta / PI) * factor


def all_drifting_edge(delta: float) -> float:
    """Eq. (5): the incoherent branch exists only for K < 1 - delta."""
    return 1.0 - float(delta)


def radius_ratio(omega: float, coupling: float) -> float:
    beta = float(np.sqrt(omega * omega - coupling * coupling))
    return coupling / (omega + beta)


def stationary_fourier(n: int, radius: float) -> complex:
    """Fourier coefficient of beta / (omega - K sin theta), with d_0 = 1."""
    if n >= 0:
        return complex((-1j * radius) ** n)
    return complex((1j * radius) ** (-n))


def forcing_coefficient(n: int, omega: float, coupling: float) -> complex:
    """Fourier coefficient of d/dtheta (sin(theta) f*).

    Synthesis convention: q(theta) = sum s_n exp(i n theta).
    """
    if n == 0 or coupling == 0.0:
        return 0j
    radius = radius_ratio(omega, coupling)
    return 1j * n * omega * stationary_fourier(n, radius) / (2.0 * PI * coupling)


def closed_form_observable(lam: complex, omega: np.ndarray, coupling: float) -> np.ndarray:
    """One-frequency response predicted by the Eq. (6) integrand over K^2.

    int_0^{2pi} (1+cos theta) u dtheta
        = lambda * omega * (omega - beta) / (K^2 * (lambda^2 + omega^2 - K^2)).
    """
    omega = np.asarray(omega, dtype=np.float64)
    beta = np.sqrt(omega * omega - coupling * coupling)
    denominator = lam * lam + omega * omega - coupling * coupling
    return lam * omega * (omega - beta) / (coupling * coupling * denominator)


def _leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(n))
    return nodes, weights


def _pole_location(lam: complex, coupling: float) -> complex:
    root = np.sqrt(np.complex128(coupling * coupling - lam * lam))
    if float(np.real(root)) < 0.0:
        root = -root
    return complex(root)


def _dyadic_panels(lower: float, upper: float, center: float, imag_distance: float, n_outer: int, n_inner: int) -> list[tuple[float, float, int]]:
    """Panels graded away from the Landau pole. The nearest panel is a few widths of Im(omega_pole)."""
    half = max(4.0 * imag_distance, 1.0e-6)
    inner_left = max(lower, center - half)
    inner_right = min(upper, center + half)
    panels: list[tuple[float, float, int]] = []
    if inner_right - inner_left > 1.0e-14:
        panels.append((inner_left, inner_right, n_inner))
    edge = inner_left
    width = half
    for _ in range(24):
        if edge - lower <= 1.0e-12:
            break
        width *= 2.0
        start = max(lower, edge - width)
        if edge - start > 1.0e-14:
            panels.append((start, edge, n_outer))
        edge = start
    edge = inner_right
    width = half
    for _ in range(24):
        if upper - edge <= 1.0e-12:
            break
        width *= 2.0
        stop = min(upper, edge + width)
        if stop - edge > 1.0e-14:
            panels.append((edge, stop, n_outer))
        edge = stop
    if not panels:
        panels.append((lower, upper, n_inner))
    return panels


def quadrature_rule(delta: float, lam: complex, coupling: float, n_outer: int, n_inner: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss nodes on [1-delta, 1+delta], graded around the complex Landau pole."""
    lower = 1.0 - float(delta)
    upper = 1.0 + float(delta)
    pole = _pole_location(lam, coupling)
    center = float(np.clip(pole.real, lower, upper))
    imag_distance = max(abs(pole.imag), 1.0e-8)
    panels = _dyadic_panels(lower, upper, center, imag_distance, n_outer, n_inner)
    nodes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for start, end, count in panels:
        raw, weight = _leggauss(count)
        nodes.append(0.5 * (end - start) * raw + 0.5 * (start + end))
        weights.append(0.5 * (end - start) * weight / (2.0 * float(delta)))
    omega = np.concatenate(nodes)
    measure = np.concatenate(weights)
    if abs(float(measure.sum()) - 1.0) > 1.0e-9:
        raise RuntimeError(f"frequency weights sum to {measure.sum()}, not 1")
    if np.any(omega <= coupling):
        raise RuntimeError("quadrature entered the pinned region omega <= K")
    return omega, measure


def galerkin_observable(lam: complex, omega: np.ndarray, coupling: float, truncation: int) -> np.ndarray:
    """Solve the Fourier recurrence for each frequency and return the observable."""
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if np.any(omega <= coupling):
        raise ValueError("all-drifting response requires omega > K at every node")
    negative = np.arange(-truncation, 0, dtype=int)
    positive = np.arange(1, truncation + 1, dtype=int)
    modes = np.concatenate([negative, positive])
    mode_values = modes.astype(np.float64)
    index = {int(mode): position for position, mode in enumerate(modes)}
    size = modes.size
    coupling_matrix = np.zeros((size, size), dtype=np.complex128)
    for mode in modes:
        row = index[int(mode)]
        lower = int(mode) - 1
        upper = int(mode) + 1
        if lower in index:
            coupling_matrix[row, index[lower]] = -0.5 * float(mode) * coupling
        if upper in index:
            coupling_matrix[row, index[upper]] = 0.5 * float(mode) * coupling
    beta = np.sqrt(omega * omega - coupling * coupling)
    radius = coupling / (omega + beta)
    observable = np.empty(omega.size, dtype=np.complex128)
    diagonal_index = np.arange(size)
    for sample, frequency in enumerate(omega):
        system = coupling_matrix.copy()
        system[diagonal_index, diagonal_index] += lam + 1j * mode_values * frequency
        coefficients_d = np.empty(size, dtype=np.complex128)
        sample_radius = float(radius[sample])
        for position, mode in enumerate(modes):
            coefficients_d[position] = stationary_fourier(int(mode), sample_radius)
        rhs = 1j * mode_values * frequency * coefficients_d / (2.0 * PI * coupling)
        solved = np.linalg.solve(system, rhs)
        observable[sample] = PI * (solved[index[1]] + solved[index[-1]])
    return observable


def chi_galerkin(
    lam: complex,
    coupling: float,
    delta: float,
    truncation: int,
    n_outer: int = N_OUTER,
    n_inner: int = N_INNER,
) -> complex:
    omega, weights = quadrature_rule(delta, lam, coupling, n_outer, n_inner)
    observable = galerkin_observable(lam, omega, coupling, truncation)
    return complex(np.sum(observable * weights))


def chi_closed_quad(lam: complex, coupling: float, delta: float) -> tuple[complex, float]:
    """Adaptive integral of the closed-form observable. Reference, not the reported root."""
    lower, upper = 1.0 - float(delta), 1.0 + float(delta)
    scale = 1.0 / (2.0 * float(delta))

    def real_part(frequency: float) -> float:
        value = closed_form_observable(lam, np.array([frequency]), coupling)[0]
        return float(np.real(value)) * scale

    def imag_part(frequency: float) -> float:
        value = closed_form_observable(lam, np.array([frequency]), coupling)[0]
        return float(np.imag(value)) * scale

    real, real_error = quad(real_part, lower, upper, epsabs=1.0e-12, limit=800)
    imag, imag_error = quad(imag_part, lower, upper, epsabs=1.0e-12, limit=800)
    return complex(real, imag), float(real_error + imag_error)


def characteristic_galerkin(lam: complex, coupling: float, delta: float, truncation: int) -> complex:
    return 1.0 - coupling * chi_galerkin(lam, coupling, delta, truncation)


def finite_difference_observable(lam: complex, omega: float, coupling: float, n_theta: int) -> complex:
    """Second-order periodic collocation of the response PDE, independent of the recurrence."""
    if omega <= coupling:
        raise ValueError("omega must exceed K")
    theta = np.linspace(0.0, 2.0 * PI, n_theta, endpoint=False)
    step = 2.0 * PI / n_theta
    beta = float(np.sqrt(omega * omega - coupling * coupling))
    velocity = omega - coupling * np.sin(theta)
    density = beta / (2.0 * PI * velocity)
    forcing = np.sin(theta) * density
    rhs = (np.roll(forcing, -1) - np.roll(forcing, 1)) / (2.0 * step)
    system = np.zeros((n_theta, n_theta), dtype=np.complex128)
    index = np.arange(n_theta)
    system[index, index] = lam
    forward = (index + 1) % n_theta
    backward = (index - 1) % n_theta
    system[index, forward] += velocity[forward] / (2.0 * step)
    system[index, backward] -= velocity[backward] / (2.0 * step)
    perturbation = np.linalg.solve(system, rhs)
    return complex(np.sum((1.0 + np.cos(theta)) * perturbation) * step)


def fft_forcing(omega: float, coupling: float, truncation: int, n_theta: int = 8192) -> dict[int, complex]:
    theta = np.linspace(0.0, 2.0 * PI, n_theta, endpoint=False)
    beta = float(np.sqrt(omega * omega - coupling * coupling))
    velocity = omega - coupling * np.sin(theta)
    density = beta / (2.0 * PI * velocity)
    coefficients = np.fft.fft(np.sin(theta) * density) / n_theta
    transformed: dict[int, complex] = {}
    for mode in range(-truncation, truncation + 1):
        slot = mode if mode >= 0 else n_theta + mode
        transformed[mode] = complex(1j * mode * coefficients[slot])
    return transformed


def adjacency_audit() -> dict:
    """p=1 in the production solver is the complete graph without self-loops."""
    from stage1.model import P1_DEFINITION, make_adjacency

    size = 8
    adjacency = make_adjacency(size, 1.0, 1)
    off_diagonal = adjacency[~np.eye(size, dtype=bool)]
    audit = {
        "definition": P1_DEFINITION,
        "n_checked": size,
        "diagonal_max": float(np.max(np.abs(np.diag(adjacency)))),
        "off_diagonal_min": float(np.min(off_diagonal)),
        "symmetric": bool(np.array_equal(adjacency, adjacency.T)),
        "paper_sum_includes_self": True,
        "continuum_normalization": (
            "The continuum theory replaces the sum by an integral. "
            "The missing diagonal is a relative factor (N-1)/N and is not inserted into the Hopf comparison."
        ),
    }
    if audit["diagonal_max"] != 0.0 or audit["off_diagonal_min"] != 1.0 or not audit["symmetric"]:
        raise RuntimeError(f"unexpected p=1 adjacency: {audit}")
    return audit


def relative_error(left: complex, right: complex) -> float:
    scale = max(abs(right), 1.0e-12)
    return abs(left - right) / scale


def run_identity_checks() -> dict:
    """Gate the root search on an independent check of the recurrence and the Eq. (6) factor."""
    checks: list[dict] = []

    frequency = 1.10
    coupling = 0.30
    spectral = fft_forcing(frequency, coupling, 12)
    forcing_abs = []
    forcing_rel_significant = []
    for mode in range(-12, 13):
        analytic = forcing_coefficient(mode, frequency, coupling)
        absolute = abs(spectral[mode] - analytic)
        forcing_abs.append(absolute)
        if abs(analytic) > 1.0e-8:
            forcing_rel_significant.append(absolute / abs(analytic))
    max_abs = float(max(forcing_abs))
    max_rel = float(max(forcing_rel_significant)) if forcing_rel_significant else 0.0
    checks.append(
        {
            "name": "forcing_fourier",
            "max_absolute_error": max_abs,
            "max_relative_error_above_1e-8": max_rel,
            "passed": bool(max_abs < 1.0e-9 and max_rel < 1.0e-8),
        }
    )

    lam = 0.15 + 1.2j
    small_k = 1.0e-6
    small_omega = 1.05
    small_observable = galerkin_observable(lam, np.array([small_omega]), small_k, 10)[0]
    limit = lam / (2.0 * (lam * lam + small_omega * small_omega))
    small_error = relative_error(small_observable, limit)
    checks.append(
        {
            "name": "k_to_zero_observable",
            "relative_error": float(small_error),
            "passed": bool(small_error < 1.0e-7),
        }
    )

    samples = (
        (0.02 + 0.95j, 1.15, 0.45),
        (0.01 + 0.98j, 1.00, 0.25),
    )
    sample_reports = []
    for sample_lambda, sample_omega, sample_k in samples:
        closed = complex(closed_form_observable(sample_lambda, np.array([sample_omega]), sample_k)[0])
        low = galerkin_observable(sample_lambda, np.array([sample_omega]), sample_k, 24)[0]
        high = galerkin_observable(sample_lambda, np.array([sample_omega]), sample_k, 40)[0]
        fd_256 = finite_difference_observable(sample_lambda, sample_omega, sample_k, 256)
        fd_512 = finite_difference_observable(sample_lambda, sample_omega, sample_k, 512)
        fd_1024 = finite_difference_observable(sample_lambda, sample_omega, sample_k, 1024)
        fd_richardson = (4.0 * fd_1024 - fd_512) / 3.0
        sample_reports.append(
            {
                "lambda": sample_lambda,
                "omega": sample_omega,
                "K": sample_k,
                "galerkin_24_vs_40": float(relative_error(low, high)),
                "galerkin_40_vs_closed_form": float(relative_error(high, closed)),
                "finite_difference_1024_vs_galerkin": float(relative_error(fd_1024, high)),
                "richardson_vs_galerkin": float(relative_error(fd_richardson, high)),
                "finite_difference_256": fd_256,
                "passed": bool(
                    relative_error(low, high) < 1.0e-8
                    and relative_error(high, closed) < 1.0e-8
                    and relative_error(fd_richardson, high) < 5.0e-4
                ),
            }
        )
    checks.append(
        {
            "name": "finite_k_identity",
            "samples": sample_reports,
            "passed": all(item["passed"] for item in sample_reports),
        }
    )

    probe = 1.0e-3 + 0.9j
    probe_k = 0.30
    probe_delta = 0.20
    adaptive, adaptive_error = chi_closed_quad(probe, probe_k, probe_delta)
    nodes, weights = quadrature_rule(probe_delta, probe, probe_k, N_OUTER, N_INNER)
    split = complex(np.sum(closed_form_observable(probe, nodes, probe_k) * weights))
    quadrature_error = relative_error(split, adaptive)
    checks.append(
        {
            "name": "split_quadrature",
            "relative_error": float(quadrature_error),
            "adaptive_error_estimate": adaptive_error,
            "passed": bool(quadrature_error < 1.0e-8 and adaptive_error < 1.0e-8),
        }
    )

    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def load_eq7_module():
    if str(EQ7_DIR) not in sys.path:
        sys.path.insert(0, str(EQ7_DIR))
    return importlib.import_module("validate_eq7")


def paper_hopf(delta: float, eq7_module) -> dict:
    """Existing Eq. (6) root. Empty when that solver finds no unique incoherent-branch root."""
    solved = eq7_module.solve_eq6(float(delta))
    truncation = float(series_eq7(delta))
    edge = all_drifting_edge(delta)
    record = {
        "delta": float(delta),
        "status": solved["status"],
        "eq7_K_H": truncation,
        "all_drifting_edge": edge,
        "eq7_inside_branch": bool(0.0 < truncation < edge),
        "K": None,
        "Omega": None,
        "imag_residual": None,
        "real_gap": None,
    }
    if solved["status"] == "ok" and len(solved["roots"]) == 1:
        root = solved["roots"][0]
        record["K"] = float(root["K"])
        record["Omega"] = float(root["Omega"])
        record["imag_residual"] = float(root["imag_residual"])
        record["real_gap"] = float(root["real_gap"])
        record["branch_ok"] = bool(root["branch_ok"])
    return record


def _cluster_roots(found: list[tuple[float, float, float]]) -> list[dict]:
    clusters: list[dict] = []
    for coupling, frequency, residual in sorted(found, key=lambda item: item[2]):
        for group in clusters:
            if abs(group["K"] - coupling) < 2.0e-4 and abs(group["Omega"] - frequency) < 2.0e-3:
                group["count"] += 1
                if residual < group["residual"]:
                    group["K"] = coupling
                    group["Omega"] = frequency
                    group["residual"] = residual
                break
        else:
            clusters.append({"K": coupling, "Omega": frequency, "residual": residual, "count": 1})
    return clusters


def characteristic_grid(delta: float, eps: float) -> dict:
    edge = all_drifting_edge(delta)
    couplings = np.linspace(0.02, edge - 0.01, 36)
    frequencies = np.linspace(0.45, 1.55, 36)
    magnitude = np.empty((couplings.size, frequencies.size), dtype=float)
    for i, coupling in enumerate(couplings):
        for j, frequency in enumerate(frequencies):
            lam = eps + 1j * float(frequency)
            nodes, weights = quadrature_rule(delta, lam, float(coupling), 48, 96)
            chi = complex(np.sum(closed_form_observable(lam, nodes, float(coupling)) * weights))
            magnitude[i, j] = abs(1.0 - float(coupling) * chi)
    minima = []
    for i in range(1, couplings.size - 1):
        for j in range(1, frequencies.size - 1):
            patch = magnitude[i - 1 : i + 2, j - 1 : j + 2]
            if magnitude[i, j] <= patch.min() and magnitude[i, j] < 0.15:
                minima.append(
                    {
                        "K": float(couplings[i]),
                        "Omega": float(frequencies[j]),
                        "abs_D": float(magnitude[i, j]),
                    }
                )
    minima.sort(key=lambda item: item["abs_D"])
    return {
        "min_abs_D": float(magnitude.min()),
        "min_location": {
            "K": float(couplings[int(np.argmin(magnitude) // frequencies.size)]),
            "Omega": float(frequencies[int(np.argmin(magnitude) % frequencies.size)]),
        },
        "local_minima": minima[:5],
    }


def solve_galerkin_root(delta: float, eps: float, truncation: int, guesses: list[tuple[float, float]]) -> dict:
    edge = all_drifting_edge(delta)
    found: list[tuple[float, float, float]] = []
    attempts = []
    for guess_k, guess_omega in guesses:
        start = np.array(
            [
                float(np.clip(guess_k, 1.0e-3, edge - 1.0e-4)),
                float(np.clip(guess_omega, 0.30, 1.80)),
            ]
        )

        def residual(point: np.ndarray) -> np.ndarray:
            coupling = float(point[0])
            frequency = float(point[1])
            if not (1.0e-4 < coupling < edge - 1.0e-8 and 0.25 < frequency < 2.0):
                return np.array([10.0, 10.0])
            value = characteristic_galerkin(eps + 1j * frequency, coupling, delta, truncation)
            return np.array([value.real, value.imag], dtype=float)

        solved = least_squares(
            residual,
            start,
            bounds=([1.0e-3, 0.25], [edge - 1.0e-6, 2.0]),
            xtol=1.0e-12,
            ftol=1.0e-14,
            gtol=1.0e-12,
            diff_step=1.0e-5,
            max_nfev=50,
        )
        value = residual(solved.x)
        norm = float(np.linalg.norm(value))
        attempts.append(
            {
                "guess_K": float(start[0]),
                "guess_Omega": float(start[1]),
                "K": float(solved.x[0]),
                "Omega": float(solved.x[1]),
                "residual": norm,
                "success_flag": bool(solved.success),
            }
        )
        interior = 1.0e-3 < float(solved.x[0]) < edge - 1.0e-4
        if norm < 1.0e-7 and interior and float(solved.x[1]) > 0.2:
            found.append((float(solved.x[0]), float(solved.x[1]), norm))
    clusters = _cluster_roots(found)
    return {"attempts": attempts, "roots": clusters}


def extrapolate_axis(rows: list[dict], key: str) -> dict | None:
    usable = [row for row in rows if row.get("K") is not None and row["residual"] < 1.0e-7 and row["n_roots"] == 1]
    if len(usable) < 3:
        return None
    usable = sorted(usable, key=lambda row: row["eps"])
    eps = np.array([row["eps"] for row in usable], dtype=float)
    values = np.array([row[key] for row in usable], dtype=float)
    scaled = eps / eps[-1]
    linear_slope = (values[1] - values[0]) / (scaled[1] - scaled[0])
    linear_intercept = float(values[0] - scaled[0] * linear_slope)
    quadratic = np.polyfit(scaled, values, 2)
    quadratic_intercept = float(quadratic[-1])
    fitted = np.polyval(quadratic, scaled)
    return {
        "value": quadratic_intercept,
        "linear_value": linear_intercept,
        "uncertainty": abs(quadratic_intercept - linear_intercept),
        "fit_max_abs_deviation": float(np.max(np.abs(values - fitted))),
        "smallest_eps": float(eps[0]),
        "value_at_smallest_eps": float(values[0]),
        "n_points": len(usable),
    }


def response_continuation(delta: float, truncation: int) -> dict:
    edge = all_drifting_edge(delta)
    truncation_guess = series_eq7(delta)
    grid = characteristic_grid(delta, EPSILONS[0])
    base_guesses = [
        (truncation_guess, 1.0),
        (truncation_guess, 0.85),
        (0.75 * truncation_guess, 1.05),
    ]
    for item in grid["local_minima"][:3]:
        base_guesses.append((item["K"], item["Omega"]))
    rows = []
    previous: tuple[float, float] | None = None
    for eps in EPSILONS:
        if previous is None:
            guesses = list(base_guesses)
        else:
            guesses = [previous, (truncation_guess, previous[1])]
        solved = solve_galerkin_root(delta, eps, truncation, guesses)
        row = {
            "eps": float(eps),
            "M": truncation,
            "n_roots": len(solved["roots"]),
            "roots": solved["roots"],
            "K": None,
            "Omega": None,
            "residual": None,
        }
        if len(solved["roots"]) == 1:
            root = solved["roots"][0]
            row["K"] = root["K"]
            row["Omega"] = root["Omega"]
            row["residual"] = root["residual"]
            previous = (root["K"], root["Omega"])
            print(
                f"delta={delta:.3f} eps={eps:.1e} M={truncation} "
                f"K={root['K']:.8f} Omega={root['Omega']:.8f} |D|={root['residual']:.2e}",
                flush=True,
            )
        else:
            previous = None
            print(
                f"delta={delta:.3f} eps={eps:.1e} M={truncation} roots={len(solved['roots'])}",
                flush=True,
            )
        rows.append(row)
    return {"grid_at_largest_eps": grid, "rows": rows, "edge": edge}


def truncation_sweep(delta: float, eps: float, guess: tuple[float, float]) -> list[dict]:
    rows = []
    for truncation in M_SWEEP:
        solved = solve_galerkin_root(delta, eps, truncation, [guess])
        row = {"M": truncation, "eps": eps, "n_roots": len(solved["roots"]), "K": None, "Omega": None, "residual": None}
        if len(solved["roots"]) == 1:
            root = solved["roots"][0]
            row.update(K=root["K"], Omega=root["Omega"], residual=root["residual"])
            print(
                f"delta={delta:.3f} M-sweep M={truncation} K={root['K']:.8f} |D|={root['residual']:.2e}",
                flush=True,
            )
        else:
            print(f"delta={delta:.3f} M-sweep M={truncation} roots={len(solved['roots'])}", flush=True)
        rows.append(row)
    return rows


def assess_target(delta: float, paper: dict, continuation: dict, sweep: list[dict]) -> dict:
    edge = all_drifting_edge(delta)
    k_extra = extrapolate_axis(continuation["rows"], "K")
    omega_extra = extrapolate_axis(continuation["rows"], "Omega")
    produced = [row for row in sweep if row["K"] is not None]
    m_gap = None
    if len(produced) >= 2:
        by_m = {row["M"]: row["K"] for row in produced}
        if 32 in by_m and 24 in by_m:
            m_gap = abs(by_m[32] - by_m[24])
        elif 32 in by_m and 16 in by_m:
            m_gap = abs(by_m[32] - by_m[16])
    interior = bool(
        k_extra is not None
        and omega_extra is not None
        and 0.0 < k_extra["value"] < edge - 1.0e-4
        and k_extra["uncertainty"] < 1.0e-4
        and k_extra["fit_max_abs_deviation"] < 1.0e-4
        and m_gap is not None
        and m_gap < 1.0e-5
    )
    paper_k = paper.get("K")
    paper_inside = paper_k is not None and bool(paper.get("branch_ok", False)) and paper_k < edge
    if interior:
        validity = "valid_all_drifting"
        convergence = "converged"
    elif paper_inside:
        validity = "valid_all_drifting"
        convergence = "not_converged"
    elif paper_k is not None and paper_k >= edge:
        validity = "outside_all_drifting_branch"
        convergence = "not_applicable"
    else:
        validity = "no_hopf_root_inside_all_drifting_branch"
        convergence = "not_applicable"
    response_k = k_extra["value"] if convergence == "converged" else None
    response_omega = omega_extra["value"] if convergence == "converged" else None
    return {
        "delta": float(delta),
        "paper": paper,
        "continuation": continuation,
        "truncation_sweep": sweep,
        "extrapolation_K": k_extra,
        "extrapolation_Omega": omega_extra,
        "M_gap": m_gap,
        "response_K_H": response_k,
        "response_Omega": response_omega,
        "convergence_status": convergence,
        "validity_status": validity,
    }


def blank_number(value: float | None, digits: str = ".10f") -> str:
    if value is None or not np.isfinite(value):
        return ""
    return format(float(value), digits)


def comparison_row(result: dict) -> dict:
    paper_k = result["paper"].get("K")
    response_k = result["response_K_H"]
    eq7 = result["paper"].get("eq7_K_H")
    absolute = None if paper_k is None or response_k is None else response_k - paper_k
    relative = None if paper_k is None or response_k is None else abs(response_k - paper_k) / abs(paper_k)
    eq7_absolute = None if paper_k is None or eq7 is None else eq7 - paper_k
    eq7_relative = None if paper_k is None or eq7 is None else abs(eq7 - paper_k) / abs(paper_k)
    uncertainty = None if result["extrapolation_K"] is None else result["extrapolation_K"]["uncertainty"]
    if result["convergence_status"] != "converged":
        uncertainty = None
    return {
        "delta": f"{result['delta']:.2f}",
        "paper_K_H": blank_number(paper_k),
        "response_K_H": blank_number(response_k),
        "absolute_error": blank_number(absolute, ".6e"),
        "relative_error": blank_number(relative, ".6e"),
        "convergence_status": result["convergence_status"],
        "validity_status": result["validity_status"],
        "eq7_K_H": blank_number(eq7),
        "eq7_absolute_error": blank_number(eq7_absolute, ".6e"),
        "eq7_relative_error": blank_number(eq7_relative, ".6e"),
        "paper_Omega": blank_number(result["paper"].get("Omega")),
        "response_Omega": blank_number(result["response_Omega"]),
        "all_drifting_edge": blank_number(result["paper"]["all_drifting_edge"]),
        "extrapolation_uncertainty": blank_number(uncertainty, ".6e"),
        "paper_source": "Ariaratnam-Strogatz Eq. (6)" if paper_k is not None else "",
    }


def write_csv(rows: list[dict]) -> None:
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value):
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def draw_figure(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
    boundary = axes[0]
    errors = axes[1]
    widths = np.linspace(0.02, 0.30, 300)
    edge = 1.0 - widths
    boundary.fill_between(
        widths,
        edge,
        1.05,
        color="#d9d9d9",
        label="Outside all-drifting branch",
        zorder=0,
    )
    boundary.plot(widths, edge, color="#4d4d4d", lw=1.3, ls="--", label=r"All-drifting edge $K=1-\delta$")
    truncation_widths = np.linspace(0.02, 0.22, 200)
    boundary.plot(
        truncation_widths,
        [series_eq7(float(width)) for width in truncation_widths],
        color="#e66101",
        lw=1.8,
        label="Eq. (7) truncation",
    )
    truncation_deltas = [result["delta"] for result in results if result["delta"] <= 0.20]
    boundary.plot(
        truncation_deltas,
        [series_eq7(float(width)) for width in truncation_deltas],
        linestyle="none",
        marker="s",
        ms=5.5,
        mfc="none",
        mec="#e66101",
        mew=1.2,
        label="Eq. (7) at tested $\\delta$",
        zorder=3,
    )
    paper_d, paper_k, response_d, response_k = [], [], [], []
    for result in results:
        if result["paper"].get("K") is not None and result["validity_status"] == "valid_all_drifting":
            paper_d.append(result["delta"])
            paper_k.append(result["paper"]["K"])
        if result["response_K_H"] is not None:
            response_d.append(result["delta"])
            response_k.append(result["response_K_H"])
    boundary.plot(
        paper_d,
        paper_k,
        linestyle="none",
        marker="o",
        ms=8,
        mfc="none",
        mec="black",
        mew=1.3,
        label="Eq. (6) Hopf root",
        zorder=3,
    )
    boundary.plot(
        response_d,
        response_k,
        linestyle="none",
        marker="+",
        ms=12,
        mew=1.7,
        color="#2166ac",
        label="Linear response",
        zorder=4,
    )
    boundary.axvline(0.25, color="#b2182b", ls=":", lw=1.1)
    boundary.text(
        0.255,
        0.18,
        r"$\delta=0.25$" + "\nno Hopf root\ninside branch",
        color="#b2182b",
        fontsize=8,
        va="bottom",
    )
    boundary.set_xlim(0.0, 0.30)
    boundary.set_ylim(0.0, 1.02)
    boundary.set_xlabel(r"Frequency half-width $\delta$")
    boundary.set_ylabel(r"Coupling $K$")
    boundary.set_title("Hopf boundary")
    boundary.legend(fontsize=7.5, loc="upper left", framealpha=0.95)
    boundary.grid(True, color="0.88", lw=0.6)

    labels = []
    response_error = []
    truncation_error = []
    for result in results:
        if result["delta"] == 0.25:
            continue
        labels.append(f"{result['delta']:.2f}")
        paper_k = result["paper"].get("K")
        response_k = result["response_K_H"]
        eq7 = result["paper"].get("eq7_K_H")
        if paper_k is None or response_k is None:
            response_error.append(np.nan)
        else:
            response_error.append(abs(response_k - paper_k) / abs(paper_k))
        if paper_k is None or eq7 is None:
            truncation_error.append(np.nan)
        else:
            truncation_error.append(abs(eq7 - paper_k) / abs(paper_k))
    positions = np.arange(len(labels))
    width = 0.36
    response_plot = [1.0e-16 if np.isfinite(value) and value < 1.0e-16 else value for value in response_error]
    errors.bar(
        positions - width / 2,
        response_plot,
        width=width,
        color="#2166ac",
        label="Linear response vs Eq. (6)",
    )
    errors.bar(
        positions + width / 2,
        truncation_error,
        width=width,
        color="#e66101",
        label="Eq. (7) vs Eq. (6)",
    )
    errors.set_yscale("log")
    errors.set_xticks(positions)
    errors.set_xticklabels([rf"$\delta={label}$" for label in labels])
    errors.set_ylabel("Relative error")
    errors.set_title("Error against Eq. (6)")
    errors.grid(True, which="both", color="0.9", lw=0.6)
    errors.legend(fontsize=7.5, loc="upper left", framealpha=0.95)
    finite_errors = [value for value in response_error + truncation_error if np.isfinite(value) and value > 0.0]
    lower = 1.0e-16 if not finite_errors else min(1.0e-16, min(finite_errors) * 0.2)
    errors.set_ylim(lower, max(1.0, max(finite_errors) * 3.0 if finite_errors else 1.0))
    fig.suptitle(
        "p = 1 all-to-all Hopf boundary: linear response vs Ariaratnam & Strogatz (2001)",
        fontsize=11,
    )
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def baseline_decision(results: list[dict], identity: dict) -> dict:
    if not identity["passed"]:
        return {
            "passed": False,
            "reason": "identity_check_failed",
            "detail": "The Fourier recurrence did not reproduce the response identity, so no Hopf value was accepted.",
        }
    targets = [item for item in results if item["delta"] in TARGET_DELTAS]
    problems = []
    for item in targets:
        if item["validity_status"] != "valid_all_drifting":
            problems.append(f"delta={item['delta']:.2f} validity={item['validity_status']}")
        elif item["convergence_status"] != "converged" or item["response_K_H"] is None:
            problems.append(f"delta={item['delta']:.2f} convergence={item['convergence_status']}")
        else:
            relative = abs(item["response_K_H"] - item["paper"]["K"]) / abs(item["paper"]["K"])
            if relative >= 1.0e-3:
                problems.append(f"delta={item['delta']:.2f} relative_error={relative:.3e}")
    control = next(item for item in results if item["delta"] == 0.25)
    if control["response_K_H"] is not None:
        problems.append("control delta=0.25 was assigned a Hopf value inside a branch with no Eq. (6) root")
    if problems:
        return {"passed": False, "reason": "comparison_failed", "detail": "; ".join(problems)}
    return {
        "passed": True,
        "reason": "matches_eq6_inside_all_drifting_branch",
        "detail": (
            "At delta=0.05, 0.10, and 0.20 the extrapolated Galerkin root agrees with Eq. (6) "
            "to a relative error below 1e-3, stays below K=1-delta, and is stable under M and epsilon. "
            "Equation (7) is the small-delta truncation, not the pass target."
        ),
    }


def blank_failure_rows(identity: dict) -> list[dict]:
    rows = []
    for delta in TARGET_DELTAS + CONTROL_DELTAS:
        rows.append(
            {
                "delta": f"{delta:.2f}",
                "paper_K_H": "",
                "response_K_H": "",
                "absolute_error": "",
                "relative_error": "",
                "convergence_status": "self_check_failed",
                "validity_status": "not_checked",
                "eq7_K_H": blank_number(series_eq7(delta)),
                "eq7_absolute_error": "",
                "eq7_relative_error": "",
                "paper_Omega": "",
                "response_Omega": "",
                "all_drifting_edge": blank_number(all_drifting_edge(delta)),
                "extrapolation_uncertainty": "",
                "paper_source": "",
            }
        )
    payload = {"identity": identity, "baseline": baseline_decision([], identity)}
    JSON_PATH.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
    write_csv(rows)
    return rows


def run(check_only: bool) -> dict:
    started = time.perf_counter()
    identity = run_identity_checks()
    print("identity_passed", identity["passed"], flush=True)
    for check in identity["checks"]:
        print(check["name"], "passed" if check["passed"] else "FAILED", flush=True)
        if check["name"] == "finite_k_identity":
            for sample in check["samples"]:
                print(
                    "  sample",
                    f"K={sample['K']}",
                    f"galerkin={sample['galerkin_40_vs_closed_form']:.3e}",
                    f"fd={sample['richardson_vs_galerkin']:.3e}",
                    flush=True,
                )
        elif check["name"] != "finite_k_identity":
            print(" ", {key: check[key] for key in check if key != "name"}, flush=True)
    adjacency = adjacency_audit()
    if not identity["passed"]:
        blank_failure_rows(identity)
        raise SystemExit(2)
    if check_only:
        return {"identity": identity, "adjacency": adjacency, "elapsed_sec": time.perf_counter() - started}

    eq7_module = load_eq7_module()
    imported_series = float(eq7_module.series_eq7(0.10))
    if abs(imported_series - series_eq7(0.10)) > 1.0e-15:
        raise RuntimeError("local Eq. (7) does not match validate_eq7.series_eq7")

    results = []
    for delta in TARGET_DELTAS + CONTROL_DELTAS:
        print(f"paper Eq. (6) at delta={delta:.2f}", flush=True)
        paper = paper_hopf(delta, eq7_module)
        print(
            f"paper status={paper['status']} K={paper['K']} Omega={paper['Omega']} eq7={paper['eq7_K_H']:.6f}",
            flush=True,
        )
        continuation = response_continuation(delta, M_PRODUCTION)
        successful = [row for row in continuation["rows"] if row["K"] is not None]
        if successful:
            guess = (successful[-1]["K"], successful[-1]["Omega"])
            sweep = truncation_sweep(delta, 1.0e-3, guess)
        else:
            sweep = []
        results.append(assess_target(delta, paper, continuation, sweep))

    rows = [comparison_row(result) for result in results]
    decision = baseline_decision(results, identity)
    payload = {
        "identity": identity,
        "adjacency": adjacency,
        "results": results,
        "baseline": decision,
        "elapsed_sec": time.perf_counter() - started,
        "regularization": (
            "Roots solve D(eps + i*Omega)=0 for eps>0 and are extrapolated to eps=0. "
            "The imaginary axis itself is the continuous spectrum and is not used as a root finder."
        ),
    }
    JSON_PATH.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
    write_csv(rows)
    draw_figure(results)
    print("baseline", decision["passed"], decision["reason"], flush=True)
    print("wrote", CSV_PATH, flush=True)
    print("wrote", FIGURE_PATH, flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Run the derivation checks and stop before the root search.")
    args = parser.parse_args()
    run(check_only=args.check_only)


if __name__ == "__main__":
    main()
