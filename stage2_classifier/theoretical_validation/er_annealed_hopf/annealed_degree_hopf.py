"""Degree-based annealed Hopf boundary for the undirected ER Winfree model.

The single-frequency susceptibility is the p=1 function validated in
``linear_response_hopf.py``. The local stationary coefficient ``a_k = K k / N``
enters that function. The global coupling ``K`` enters only through ``a_k`` and
through the annealed feedback. Roots are continued from Re(lambda) > 0. A pure
imaginary matrix eigenvalue is not accepted as a Hopf root.

This script does not integrate the Winfree ODE and does not modify Classifier v1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-er-annealed-hopf")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
P1_DIR = HERE.parent / "p1_hopf_linear_response"
if str(P1_DIR) not in sys.path:
    sys.path.insert(0, str(P1_DIR))

import linear_response_hopf as p1  # noqa: E402

N_OSC = 1000
P_VALUES = (1.0, 0.8, 0.6, 0.4, 0.2)
DELTAS = (0.05, 0.10, 0.20)
TAIL_PRODUCTION = 1.0e-12
TAIL_LOOSE = 1.0e-8
PINNED_Q_LIMIT = 1.0e-8
QUAD_OUTER = 16
QUAD_INNER = 32
EPSILONS = p1.EPSILONS

CSV_PATH = HERE / "er_annealed_hopf.csv"
FIGURE_PATH = HERE / "er_annealed_hopf.png"
JSON_PATH = HERE / "er_annealed_hopf_diagnostics.json"

CSV_FIELDS = [
    "p",
    "delta",
    "K_H_scaling",
    "K_H_degree",
    "relative_deviation",
    "Omega_H",
    "validity_status",
    "convergence_status",
    "K_H_p1",
    "K_H_homogeneous",
    "heterogeneity_deviation",
    "pinned_q_mass",
    "discarded_probability",
    "continuation_uncertainty",
    "residual",
    "N",
]


def _composite_frequency_rule(lam: complex, delta: float, panel_factor: float = 2.0, points: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Shared real-frequency grid. Panel width tracks the distance to the Landau pole."""
    epsilon = max(abs(float(np.real(lam))), 1.0e-8)
    lower, upper = 1.0 - float(delta), 1.0 + float(delta)
    width = panel_factor * epsilon
    edges = [lower]
    while edges[-1] < upper - 1.0e-15:
        edges.append(min(upper, edges[-1] + width))
    raw, weights = np.polynomial.legendre.leggauss(points)
    nodes: list[np.ndarray] = []
    measure: list[np.ndarray] = []
    for left, right in zip(edges[:-1], edges[1:]):
        nodes.append(0.5 * (right - left) * raw + 0.5 * (left + right))
        measure.append(0.5 * (right - left) * weights / (2.0 * float(delta)))
    return np.concatenate(nodes), np.concatenate(measure)


def chi_many(lam: complex, coefficients: np.ndarray, delta: float) -> np.ndarray:
    """Susceptibility at many local coefficients on one frequency grid."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if np.any(coefficients >= 1.0 - float(delta)):
        raise ValueError("local coefficient is outside the all-drifting interval")
    omega, measure = _composite_frequency_rule(lam, delta)
    local = coefficients[None, :]
    frequency = omega[:, None]
    beta = np.sqrt(np.maximum(frequency * frequency - local * local, 0.0))
    denominator = lam * lam + frequency * frequency - local * local
    observable = lam * frequency / ((frequency + beta) * denominator)
    return measure @ observable


def chi_of_coefficient(lam: complex, coefficient: float, delta: float) -> complex:
    """Validated single-frequency susceptibility at local coefficient ``a``."""
    return complex(chi_many(lam, np.array([coefficient], dtype=np.float64), delta)[0])


def stable_observable(lam: complex, omega: np.ndarray, coefficient: float) -> np.ndarray:
    """Unit-response observable, algebraically identical to the p=1 closed form.

    ``omega - sqrt(omega^2 - a^2) = a^2 / (omega + sqrt(omega^2 - a^2))``, so

        lambda * omega * (omega - beta) / (a^2 * (lambda^2 + omega^2 - a^2))
        = lambda * omega / ((omega + beta) * (lambda^2 + omega^2 - a^2)).

    The second form stays accurate when the local coefficient ``a`` is small.
    ``a`` is the stationary transport coefficient, not the global coupling.
    """
    omega = np.asarray(omega, dtype=np.float64)
    beta = np.sqrt(np.maximum(omega * omega - coefficient * coefficient, 0.0))
    denominator = lam * lam + omega * omega - coefficient * coefficient
    return lam * omega / ((omega + beta) * denominator)


def excess_degree_distribution(n: int, probability: float, tail_tol: float) -> dict:
    """Neighbor distribution q(k) for an undirected ER graph on n nodes.

    P(k) = Binomial(n-1, p). For 0<p<1 the excess degree is Binomial(n-2, p)
    and the neighbor degree is excess + 1. At p=1 every degree is n-1.
    """
    if not 0.0 < probability <= 1.0:
        raise ValueError(f"p must lie in (0, 1], got {probability}")
    if probability >= 1.0 - 1.0e-15:
        return {
            "degree": np.array([n - 1], dtype=int),
            "q": np.array([1.0]),
            "discarded_probability": 0.0,
            "mean_degree_q": float(n - 1),
            "support_size": 1,
        }
    trials = n - 2
    mean = trials * probability
    std = float(np.sqrt(trials * probability * (1.0 - probability)))
    lower = 0
    upper = trials
    if std > 0.0:
        z = 3.0
        while z <= 40.0:
            lower = max(0, int(np.floor(mean - z * std)))
            upper = min(trials, int(np.ceil(mean + z * std)))
            lower_tail = float(binom.cdf(lower - 1, trials, probability)) if lower else 0.0
            upper_tail = float(binom.sf(upper, trials, probability))
            if lower_tail + upper_tail <= tail_tol or (lower == 0 and upper == trials):
                break
            z += 2.0
    else:
        lower = upper = int(round(mean))
    excess = np.arange(lower, upper + 1)
    log_weights = binom.logpmf(excess, trials, probability)
    weights = np.exp(log_weights - np.max(log_weights))
    q = weights / np.sum(weights)
    lower_tail = float(binom.cdf(lower - 1, trials, probability)) if lower else 0.0
    upper_tail = float(binom.sf(upper, trials, probability))
    degree = excess + 1
    return {
        "degree": degree.astype(int),
        "q": q,
        "discarded_probability": lower_tail + upper_tail,
        "mean_degree_q": float(np.sum(degree * q)),
        "support_size": int(degree.size),
    }


def exact_mean_neighbor_degree(n: int, probability: float) -> float:
    """E_q[k] = <k^2>/<k> = 1 + (n-2)p, and n-1 at p=1."""
    if probability >= 1.0 - 1.0e-15:
        return float(n - 1)
    return 1.0 + (n - 2) * probability


def pinned_neighbor_mass(n: int, probability: float, coupling: float, delta: float) -> float:
    """q-mass of degree classes with a_k = K k / N >= 1 - delta."""
    if coupling <= 0.0:
        return 0.0
    threshold = n * (1.0 - float(delta)) / float(coupling)
    k_pin = int(np.ceil(float(threshold) - 1.0e-12))
    if k_pin > n - 1:
        return 0.0
    if probability >= 1.0 - 1.0e-15:
        return 1.0 if (n - 1) >= k_pin else 0.0
    if k_pin <= 1:
        return 1.0
    excess_min = k_pin - 1
    return float(binom.sf(excess_min - 1, n - 2, probability))


def characteristic(lam: complex, coupling: float, delta: float, distribution: dict, n: int) -> complex:
    """D = 1 - sum_k q(k) a_k chi(lambda; a_k), with a_k = K k / N.

    This is the same statement as
    1 - (K/N) sum_k k q(k) chi(lambda; a_k).
    """
    degree = distribution["degree"]
    local = coupling * degree / float(n)
    edge = 1.0 - float(delta)
    drifting = local < edge
    # Classes with a_k >= 1-delta are outside the formula. They may be omitted
    # only when their neighbor weight is negligible.
    if not np.all(drifting):
        pinned_mass = float(np.sum(distribution["q"][~drifting]))
        if pinned_mass > PINNED_Q_LIMIT:
            raise ValueError("pinned neighbor mass is outside the all-drifting theory")
        degree = degree[drifting]
        local = local[drifting]
        weights = distribution["q"][drifting]
    else:
        weights = distribution["q"]
    if local.size == 0:
        raise ValueError("no drifting degree class remains")
    susceptibility = chi_many(lam, local, delta)
    feedback = weights * local * susceptibility
    return 1.0 - complex(np.sum(feedback))


def relative_gap(left: complex, right: complex) -> float:
    return abs(left - right) / max(abs(right), 1.0e-12)


def run_identity_checks() -> dict:
    """Confirm the local susceptibility is the validated p=1 function of a."""
    checks = []
    lam = 1.0e-3 + 0.96j
    delta = 0.10
    frequency = np.linspace(0.9, 1.1, 6)
    for coefficient in (0.05, 0.25, 0.55):
        stable = stable_observable(lam, frequency, coefficient)
        original = p1.closed_form_observable(lam, frequency, coefficient)
        gap = float(np.max(np.abs(stable - original) / np.maximum(np.abs(original), 1.0e-12)))
        checks.append({"name": f"stable_form_a_{coefficient:.2f}", "relative_error": gap, "passed": gap < 1.0e-10})

    quad_gaps = []
    for probe_eps, probe_delta, probe_a in (
        (1.0e-2, 0.20, 0.70),
        (1.0e-3, 0.10, 0.30),
        (1.0e-4, 0.20, 0.40),
    ):
        probe = probe_eps + 0.8j
        reference, _ = p1.chi_closed_quad(probe, probe_a, probe_delta)
        produced = chi_of_coefficient(probe, probe_a, probe_delta)
        quad_gaps.append(relative_gap(produced, reference))
    checks.append(
        {
            "name": "chi_matches_p1_adaptive_quad",
            "max_relative_error": float(max(quad_gaps)),
            "passed": bool(max(quad_gaps) < 1.0e-8),
        }
    )

    small = 1.0e-5
    small_chi = chi_of_coefficient(lam, small, delta)
    omega, weights = _composite_frequency_rule(lam, delta)
    limit = complex(np.sum((lam / (2.0 * (lam * lam + omega * omega))) * weights))
    limit_gap = relative_gap(small_chi, limit)
    checks.append({"name": "small_a_limit", "relative_error": float(limit_gap), "passed": bool(limit_gap < 1.0e-6)})

    point = excess_degree_distribution(N_OSC, 1.0, TAIL_PRODUCTION)
    network = characteristic(lam, 0.40, delta, point, N_OSC)
    local = 0.40 * (N_OSC - 1) / N_OSC
    scalar = 1.0 - local * chi_of_coefficient(lam, local, delta)
    p1_gap = relative_gap(network, scalar)
    checks.append({"name": "p1_reduces_to_local_coefficient", "relative_error": float(p1_gap), "passed": bool(p1_gap < 1.0e-10)})

    sample = excess_degree_distribution(N_OSC, 0.2, TAIL_PRODUCTION)
    exact_mean = exact_mean_neighbor_degree(N_OSC, 0.2)
    mean_gap = abs(sample["mean_degree_q"] - exact_mean) / exact_mean
    checks.append(
        {
            "name": "excess_degree_mean",
            "discarded_probability": sample["discarded_probability"],
            "support_size": sample["support_size"],
            "relative_mean_error": float(mean_gap),
            "passed": bool(sample["discarded_probability"] <= TAIL_PRODUCTION and mean_gap < 1.0e-8),
        }
    )
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def load_p1_boundaries() -> dict[float, dict]:
    module = p1.load_eq7_module()
    boundaries = {}
    for delta in DELTAS:
        solved = module.solve_eq6(float(delta))
        if solved["status"] != "ok" or len(solved["roots"]) != 1:
            raise RuntimeError(f"Eq. (6) solver did not return one root at delta={delta}")
        root = solved["roots"][0]
        boundaries[float(delta)] = {"K": float(root["K"]), "Omega": float(root["Omega"])}
        print(f"Eq. (6) delta={delta:.2f} K={root['K']:.8f} Omega={root['Omega']:.8f}", flush=True)
    return boundaries


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


def coupling_upper_bound(probability: float, delta: float, n: int) -> float:
    """Largest K whose pinned neighbor mass stays at the acceptance limit."""
    lower, upper = 1.0e-4, n * (1.0 - float(delta))
    for _ in range(50):
        middle = 0.5 * (lower + upper)
        if pinned_neighbor_mass(n, probability, middle, delta) > PINNED_Q_LIMIT:
            upper = middle
        else:
            lower = middle
    return lower


def solve_at_epsilon(
    delta: float,
    eps: float,
    distribution: dict,
    guesses: list[tuple[float, float]],
    upper: float,
) -> dict:
    found: list[tuple[float, float, float]] = []
    for guess_k, guess_omega in guesses:
        start = np.array(
            [
                float(np.clip(guess_k, 1.0e-3, upper * 0.999)),
                float(np.clip(guess_omega, 0.35, 1.70)),
            ]
        )

        def residual(point: np.ndarray) -> np.ndarray:
            coupling = float(point[0])
            frequency = float(point[1])
            if not (1.0e-3 < coupling < upper and 0.25 < frequency < 1.9):
                return np.array([10.0, 10.0])
            try:
                value = characteristic(eps + 1j * frequency, coupling, delta, distribution, N_OSC)
            except ValueError:
                return np.array([10.0, 10.0])
            return np.array([value.real, value.imag], dtype=float)

        solved = least_squares(
            residual,
            start,
            bounds=([1.0e-3, 0.25], [upper * 0.999, 1.9]),
            xtol=1.0e-12,
            ftol=1.0e-14,
            gtol=1.0e-12,
            diff_step=1.0e-5,
            max_nfev=40,
        )
        value = residual(solved.x)
        norm = float(np.linalg.norm(value))
        interior = 1.0e-3 < float(solved.x[0]) < upper * (1.0 - 1.0e-4)
        if norm < 1.0e-7 and interior and float(solved.x[1]) > 0.2:
            found.append((float(solved.x[0]), float(solved.x[1]), norm))
    clusters = _cluster_roots(found)
    return {"roots": clusters}


def continue_root(
    delta: float,
    probability: float,
    distribution: dict,
    homogeneous: float,
    paper_omega: float,
) -> dict:
    upper = coupling_upper_bound(probability, delta, N_OSC)
    guesses = [
        (homogeneous, paper_omega),
        (0.97 * homogeneous, paper_omega * 0.98),
        (min(1.03 * homogeneous, upper * 0.99), min(paper_omega * 1.02, 1.6)),
    ]
    rows = []
    previous: tuple[float, float] | None = None
    for eps in EPSILONS:
        if previous is None:
            start_guesses = guesses
        else:
            start_guesses = [previous, (min(homogeneous, upper * 0.99), previous[1])]
        solved = solve_at_epsilon(delta, float(eps), distribution, start_guesses, upper)
        row = {"eps": float(eps), "n_roots": len(solved["roots"]), "K": None, "Omega": None, "residual": None}
        if len(solved["roots"]) == 1:
            root = solved["roots"][0]
            row.update(K=root["K"], Omega=root["Omega"], residual=root["residual"])
            previous = (root["K"], root["Omega"])
            print(
                f"  eps={eps:.1e} K={root['K']:.8f} Omega={root['Omega']:.8f} |D|={root['residual']:.2e}",
                flush=True,
            )
        else:
            previous = None
            print(f"  eps={eps:.1e} roots={len(solved['roots'])}", flush=True)
        rows.append(row)
    return {"rows": rows, "upper": upper}


def truncation_shift(delta: float, probability: float, guess: tuple[float, float]) -> dict:
    """Compare the eps=1e-3 root when the discarded neighbor mass changes."""
    results = {}
    for tolerance in (1.0e-4, TAIL_LOOSE, TAIL_PRODUCTION):
        distribution = excess_degree_distribution(N_OSC, probability, tolerance)
        upper = coupling_upper_bound(probability, delta, N_OSC)
        solved = solve_at_epsilon(delta, 1.0e-3, distribution, [guess], upper)
        row = {
            "discarded_probability": distribution["discarded_probability"],
            "support_size": distribution["support_size"],
            "K": None,
        }
        if len(solved["roots"]) == 1:
            row["K"] = solved["roots"][0]["K"]
        results[f"{tolerance:.0e}"] = row
    loose = results[f"{TAIL_LOOSE:.0e}"]["K"]
    tight = results[f"{TAIL_PRODUCTION:.0e}"]["K"]
    coarse = results["1e-04"]["K"]
    shift = None if loose is None or tight is None else abs(loose - tight) / max(abs(tight), 1.0e-12)
    coarse_shift = None if coarse is None or tight is None else abs(coarse - tight) / max(abs(tight), 1.0e-12)
    return {
        "levels": results,
        "relative_shift": shift,
        "coarse_relative_shift": coarse_shift,
        "passed": bool(shift is not None and shift < 1.0e-5),
    }


def fourier_check(lam: complex, coefficient: float, delta: float) -> dict:
    """Galerkin truncation against the closed form on one shared frequency grid.

    The comparison uses the same nodes, so the difference is the Fourier tail
    rather than a second quadrature rule.
    """
    if coefficient <= 0.0 or coefficient >= 1.0 - float(delta):
        return {"passed": False, "reason": "coefficient_outside_all_drifting", "relative_errors": {}}
    omega, weights = p1.quadrature_rule(delta, lam, float(coefficient), 24, 48)
    closed = complex(np.sum(stable_observable(lam, omega, float(coefficient)) * weights))
    errors = {}
    for truncation in (16, 32, 48):
        galerkin = p1.galerkin_observable(lam, omega, float(coefficient), truncation)
        produced = complex(np.sum(galerkin * weights))
        errors[str(truncation)] = relative_gap(produced, closed)
    return {"relative_errors": errors, "passed": bool(errors["32"] < 1.0e-6 or errors["48"] < 1.0e-6)}


def assess_case(probability: float, delta: float, paper: dict) -> dict:
    print(f"p={probability:.2f} delta={delta:.2f}", flush=True)
    distribution = excess_degree_distribution(N_OSC, probability, TAIL_PRODUCTION)
    mean_neighbor = exact_mean_neighbor_degree(N_OSC, probability)
    paper_k = paper["K"]
    paper_omega = paper["Omega"]
    scaling = paper_k / probability
    homogeneous = paper_k * N_OSC / mean_neighbor
    record = {
        "p": probability,
        "delta": delta,
        "K_H_p1": paper_k,
        "K_H_scaling": scaling,
        "K_H_homogeneous": homogeneous,
        "paper_Omega": paper_omega,
        "discarded_probability": distribution["discarded_probability"],
        "support_size": distribution["support_size"],
        "mean_neighbor_degree": mean_neighbor,
        "distribution_mean": distribution["mean_degree_q"],
        "K_H_degree": None,
        "Omega_H": None,
        "relative_deviation": None,
        "heterogeneity_deviation": None,
        "pinned_q_mass": None,
        "continuation_uncertainty": None,
        "residual": None,
        "validity_status": "unresolved",
        "convergence_status": "not_converged",
    }
    continued = continue_root(delta, probability, distribution, homogeneous, paper_omega)
    record["continuation"] = continued
    k_extra = p1.extrapolate_axis(continued["rows"], "K")
    omega_extra = p1.extrapolate_axis(continued["rows"], "Omega")
    record["extrapolation_K"] = k_extra
    record["extrapolation_Omega"] = omega_extra
    if k_extra is None or omega_extra is None:
        record["validity_status"] = "unresolved"
        record["convergence_status"] = "not_converged"
        return record
    coupling = float(k_extra["value"])
    frequency = float(omega_extra["value"])
    pinned = pinned_neighbor_mass(N_OSC, probability, coupling, delta)
    k_max = int(np.max(distribution["degree"]))
    local_max = coupling * k_max / N_OSC
    record["pinned_q_mass"] = pinned
    record["a_max_retained"] = local_max
    record["all_drifting_edge"] = 1.0 - delta
    uncertainty = float(k_extra["uncertainty"])
    record["continuation_uncertainty"] = uncertainty
    smallest = min((row for row in continued["rows"] if row["residual"] is not None), key=lambda row: row["eps"])
    record["residual"] = smallest["residual"]
    relative_uncertainty = uncertainty / max(abs(coupling), 1.0e-12)
    fit_gap = float(k_extra["fit_max_abs_deviation"]) / max(abs(coupling), 1.0e-12)
    interior = local_max < (1.0 - delta) or pinned <= PINNED_Q_LIMIT
    if pinned > PINNED_Q_LIMIT or not interior:
        record["validity_status"] = "invalid_pinned_tail"
        record["convergence_status"] = "not_applicable"
        return record
    guess = (smallest["K"], smallest["Omega"])
    truncation = truncation_shift(delta, probability, guess)
    record["truncation_check"] = truncation
    probe = smallest["eps"] + 1j * smallest["Omega"]
    local_mean = smallest["K"] * mean_neighbor / N_OSC
    heavy = distribution["degree"][distribution["q"] > 1.0e-6]
    heavy_degree = int(np.max(heavy)) if heavy.size else int(np.max(distribution["degree"]))
    fourier_mean = fourier_check(probe, local_mean, delta)
    fourier_tail = fourier_check(probe, smallest["K"] * heavy_degree / N_OSC, delta)
    record["fourier_tail_degree"] = heavy_degree
    record["fourier_mean"] = fourier_mean
    record["fourier_tail"] = fourier_tail
    numerically_ok = bool(
        relative_uncertainty < 1.0e-4
        and fit_gap < 1.0e-4
        and smallest["residual"] < 1.0e-7
        and truncation["passed"]
        and fourier_mean["passed"]
        and fourier_tail["passed"]
    )
    if not numerically_ok:
        record["validity_status"] = "valid_all_drifting"
        record["convergence_status"] = "not_converged"
        return record
    record["K_H_degree"] = coupling
    record["Omega_H"] = frequency
    record["relative_deviation"] = (coupling - scaling) / scaling
    record["heterogeneity_deviation"] = (coupling - homogeneous) / homogeneous
    record["validity_status"] = "valid_all_drifting"
    record["convergence_status"] = "converged"
    print(
        f"  K_degree={coupling:.8f} scaling={scaling:.8f} "
        f"rel={record['relative_deviation']:.3e} het={record['heterogeneity_deviation']:.3e} "
        f"pinned_q={pinned:.3e}",
        flush=True,
    )
    return record


def blank(value: float | None, digits: str = ".10f") -> str:
    if value is None or not np.isfinite(value):
        return ""
    return format(float(value), digits)


def comparison_row(result: dict) -> dict:
    return {
        "p": f"{result['p']:.2f}",
        "delta": f"{result['delta']:.2f}",
        "K_H_scaling": blank(result["K_H_scaling"]),
        "K_H_degree": blank(result["K_H_degree"]),
        "relative_deviation": blank(result["relative_deviation"], ".6e"),
        "Omega_H": blank(result["Omega_H"]),
        "validity_status": result["validity_status"],
        "convergence_status": result["convergence_status"],
        "K_H_p1": blank(result["K_H_p1"]),
        "K_H_homogeneous": blank(result["K_H_homogeneous"]),
        "heterogeneity_deviation": blank(result["heterogeneity_deviation"], ".6e"),
        "pinned_q_mass": blank(result["pinned_q_mass"], ".6e"),
        "discarded_probability": blank(result["discarded_probability"], ".6e"),
        "continuation_uncertainty": blank(result["continuation_uncertainty"], ".6e"),
        "residual": blank(result["residual"], ".6e"),
        "N": str(N_OSC),
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
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def draw_figure(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    colors = {0.05: "#2166ac", 0.10: "#1b9e77", 0.20: "#e66101"}
    boundary, deviation = axes
    for delta in DELTAS:
        subset = [row for row in results if abs(row["delta"] - delta) < 1.0e-12]
        subset = sorted(subset, key=lambda row: row["p"])
        scaling_p = [row["p"] for row in subset]
        scaling_k = [row["K_H_scaling"] for row in subset]
        boundary.plot(
            scaling_p,
            scaling_k,
            color=colors[delta],
            lw=1.5,
            label=rf"$\delta={delta:.2f}$, $K_H(p=1)/p$",
        )
        valid_p = [row["p"] for row in subset if row["K_H_degree"] is not None]
        valid_k = [row["K_H_degree"] for row in subset if row["K_H_degree"] is not None]
        boundary.plot(
            valid_p,
            valid_k,
            linestyle="none",
            marker="o",
            ms=6.5,
            color=colors[delta],
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=rf"$\delta={delta:.2f}$, degree theory",
        )
        invalid = [row for row in subset if row["K_H_degree"] is None]
        if invalid:
            boundary.plot(
                [row["p"] for row in invalid],
                [row["K_H_scaling"] for row in invalid],
                linestyle="none",
                marker="x",
                ms=8,
                color=colors[delta],
                mew=1.6,
            )
    boundary.set_xlabel("ER probability $p$")
    boundary.set_ylabel("Hopf coupling $K_H$")
    boundary.set_title("Annealed degree theory and $pK$ scaling")
    boundary.set_xlim(0.12, 1.05)
    boundary.grid(True, color="0.88", lw=0.6)
    boundary.legend(fontsize=7.2, ncol=2, loc="upper right", framealpha=0.95)

    guide_p = np.linspace(0.15, 1.0, 200)
    guide = 100.0 * (2.0 * guide_p - 1.0) / (1.0 + (N_OSC - 2) * guide_p)
    deviation.plot(guide_p, guide, color="0.25", lw=1.1, ls="--", label="finite-$N$ mean degree, no spread")
    width = 0.018
    for index, delta in enumerate(DELTAS):
        subset = [row for row in results if abs(row["delta"] - delta) < 1.0e-12 and row["relative_deviation"] is not None]
        deviation.plot(
            [row["p"] + (index - 1) * width for row in subset],
            [100.0 * row["relative_deviation"] for row in subset],
            linestyle="none",
            marker="o",
            ms=6,
            color=colors[delta],
            label=rf"$\delta={delta:.2f}$, degree theory",
        )
    deviation.axhline(0.0, color="0.3", lw=0.8)
    deviation.set_xlim(0.12, 1.05)
    deviation.set_xlabel("ER probability $p$")
    deviation.set_ylabel(r"Relative deviation from $K_H(p{=}1)/p$ (%)")
    deviation.set_title("Deviation from naive $pK$ scaling")
    deviation.grid(True, color="0.88", lw=0.6)
    deviation.legend(fontsize=7.2, loc="best", framealpha=0.95)
    fig.suptitle(
        "Annealed degree Hopf boundary compared with naive $pK$ scaling",
        fontsize=11,
    )
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def p1_recovery_gate(results: list[dict]) -> dict:
    """The p=1 degree root must be the Eq. (6) root times N/(N-1)."""
    comparisons = []
    passed = True
    for delta in DELTAS:
        row = next(item for item in results if item["p"] == 1.0 and abs(item["delta"] - delta) < 1.0e-12)
        expected = row["K_H_p1"] * N_OSC / (N_OSC - 1)
        actual = row["K_H_degree"]
        if actual is None:
            passed = False
            comparisons.append({"delta": delta, "passed": False, "expected": expected, "actual": None})
            continue
        gap = abs(actual - expected) / expected
        ok = gap < 1.0e-4
        passed = passed and ok
        comparisons.append({"delta": delta, "passed": ok, "expected": expected, "actual": actual, "relative_gap": gap})
    return {"passed": passed, "comparisons": comparisons}


def run(check_only: bool) -> dict:
    started = time.perf_counter()
    identity = run_identity_checks()
    print("identity_passed", identity["passed"], flush=True)
    for check in identity["checks"]:
        print(" ", check["name"], "passed" if check["passed"] else "FAILED", check, flush=True)
    if not identity["passed"]:
        raise SystemExit(2)
    if check_only:
        return {"identity": identity, "elapsed_sec": time.perf_counter() - started}
    papers = load_p1_boundaries()
    results = []
    for probability in (1.0,) + tuple(value for value in P_VALUES if value != 1.0):
        for delta in DELTAS:
            results.append(assess_case(probability, delta, papers[delta]))
    recovery = p1_recovery_gate(results)
    print("p1_recovery", recovery["passed"], flush=True)
    if not recovery["passed"]:
        for result in results:
            result["K_H_degree"] = None
            result["Omega_H"] = None
            result["relative_deviation"] = None
            result["heterogeneity_deviation"] = None
            result["convergence_status"] = "not_converged"
            if result["p"] == 1.0:
                result["validity_status"] = "p1_recovery_failed"
    rows = [comparison_row(result) for result in results]
    payload = {
        "identity": identity,
        "p1_recovery": recovery,
        "pinned_q_limit": PINNED_Q_LIMIT,
        "tail_production": TAIL_PRODUCTION,
        "finite_n_factor": N_OSC / (N_OSC - 1),
        "results": results,
        "elapsed_sec": time.perf_counter() - started,
        "interpretation": (
            "K_H_scaling = K_H(Eq. 6)/p. K_H_homogeneous = K_H(Eq. 6)*N/E_q[k] "
            "uses the mean neighbor degree and no degree spread. "
            "(K_H_homogeneous-K_H_scaling)/K_H_scaling = (2p-1)/(1+(N-2)p), "
            "which equals 1/(N-1) only at p=1. "
            "heterogeneity_deviation = (K_H_degree-K_H_homogeneous)/K_H_homogeneous "
            "is the correction from the spread of q(k)."
        ),
    }
    JSON_PATH.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
    write_csv(rows)
    draw_figure(results)
    print("wrote", CSV_PATH, flush=True)
    print("wrote", FIGURE_PATH, flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(check_only=args.check_only)


if __name__ == "__main__":
    main()
