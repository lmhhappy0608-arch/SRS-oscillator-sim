#!/usr/bin/env python3
"""Calibrate the separated Method C penalties on synthetic profiles.

The original dynamic program is imported and not modified. Parameter choice
uses the synthetic calibration split only. Real Stage 1 profiles are scored
after that choice and do not enter the score.

Run from the repository root:

    python3 stage2_classifier/penalty_calibration/run_calibration.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-stage2c-penalty")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = OUT_DIR / "figures"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OUT_DIR))

from calibration_metrics import evaluate_profile
from method_c_separated import (
    equivalent_alpha_beta,
    load_method_c,
    original_penalties,
    segment_original,
    segment_separated,
    segmentation_equal,
)
from stage1.query import DEFAULT_OUTPUT_DIR, load_profiles
from synthetic_profiles import (
    CALIBRATION_SEEDS,
    CALIBRATION_STRUCTURES,
    N_OSCILLATORS,
    NOISE_LEVELS,
    OMEGA,
    T_STAT_DISTINGUISHABLE,
    VALIDATION_SEEDS,
    VALIDATION_STRUCTURES,
    build_profiles,
    truth_labels,
)

ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]
BETAS = [0.25, 0.5, 1.0, 2.0, 4.0]
ROBUST_SCORE_GAP = 0.02
ROBUST_FALSE_GAP = 0.02
VALIDATION_DROP_LIMIT = 0.15
EASY_MISSED_LIMIT = 0.25
EASY_F1_LIMIT = 0.80
FALSE_CONSTANT_LIMIT = 0.15
EASY_STRUCTURES = ("lcl_long", "lcl_medium", "clc", "two_plateaus", "three_plateaus", "pure_flat")
DIAGNOSTIC_OMEGA = (0.93, 1.04)
WINDOWS = {"W1": (8000.0, 10000.0), "W2": (10000.0, 12000.0), "W3": (12000.0, 14000.0)}
CASE_ROLES = {
    "C15": ("diagnostic", "Stage 2B candidate near-horizontal region"),
    "C23": ("diagnostic", "Stage 2B rising profile with local fluctuations"),
    "C01": ("examined", "Near-zero range at p=1, Delta=0.05"),
    "C02": ("examined", "Flat profile with range 0 and near-zero fraction 1"),
    "C05": ("examined", "Large range and partial near-zero fraction, p=1, Delta=0.8"),
    "C09": ("examined", "Mixed range at p=0.8, Delta=0.2"),
    "C13": ("examined", "Near-flat profile at p=0.4, Delta=0.05"),
    "C17": ("examined", "Partial near-zero fraction at p=0.4, Delta=0.8"),
    "C22": ("examined", "Slope-scale sensitive profile at p=0.2, Delta=0.2"),
    "C25": ("examined", "High window variation"),
    "C26": ("examined", "Neighbor jump with a flatter median local slope"),
    "C30": ("examined", "High rotation dispersion at p=0.2, Delta=0.8"),
    "C07": ("held_out", "Held out. Range 0.10 at p=0.8, Delta=0.05"),
    "C20": ("held_out", "Held out. Range 0 at p=0.2, Delta=0.05"),
    "C21": ("held_out", "Held out. K=0 profile whose range equals the omega span"),
    "C28": ("held_out", "Held out. Small near-zero fraction at p=0.2, Delta=0.8"),
}
FIGURE_CASES = ("C15", "C23", "C01", "C25", "C21", "C30")
COLOR_CONSTANT = "#0072B2"
COLOR_LINEAR = "#D55E00"


def cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "inf" if number > 0 else ""
        return f"{number:.8g}"
    return str(value)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: cell(row.get(key, "")) for key in fields})


def modes_of(n: int, segments) -> np.ndarray:
    modes = np.empty(n, dtype=int)
    for start, end, mode in segments:
        modes[start:end] = int(mode)
    return modes


def finite_mean(values) -> float:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not data:
        return math.nan
    return float(np.mean(data))


def time_one_fit(pilot, cost_audit, min_len: int) -> float:
    rng = np.random.default_rng(0)
    y = OMEGA + rng.normal(0.0, 0.01, size=OMEGA.size)
    sigma2 = float(pilot.estimate_sigma(y)) ** 2
    start = time.perf_counter()
    segment_separated(cost_audit, OMEGA, y, sigma2, 2.0, 1.0, min_len)
    return time.perf_counter() - start


def equivalence_case(pilot, cost_audit, omega, y, tau: float, min_len: int, label: str) -> dict:
    sigma = float(pilot.estimate_sigma(y))
    sigma2 = sigma * sigma
    old_segments, old_fitted, old_objective = segment_original(pilot, omega, y, sigma2, tau, min_len)
    alpha, beta = equivalent_alpha_beta(tau)
    new_segments, new_fitted, new_objective, pen0, pen1, _rss = segment_separated(
        cost_audit, omega, y, sigma2, alpha, beta, min_len
    )
    _unit, old_pen0, old_pen1 = original_penalties(len(y), sigma2, tau)
    penalty_gap = max(abs(pen0 - old_pen0), abs(pen1 - old_pen1))
    segments_match = segmentation_equal(old_segments, new_segments)
    fitted_gap = float(np.max(np.abs(old_fitted - new_fitted))) if len(y) else 0.0
    objective_gap = abs(old_objective - new_objective)
    passed = segments_match and fitted_gap <= 1e-6 and objective_gap <= 1e-6 * max(1.0, abs(old_objective))
    return {
        "label": label,
        "tau": tau,
        "alpha": alpha,
        "beta": beta,
        "sigma_hat": sigma,
        "segments_match": passed and segments_match,
        "segment_match": segments_match,
        "fitted_max_abs_gap": fitted_gap,
        "objective_abs_gap": objective_gap,
        "penalty_abs_gap": penalty_gap,
        "n_segments": len(old_segments),
        "passed": passed,
    }


def grid_distance(alpha: float, beta: float) -> float:
    return abs(math.log2(alpha) - 1.0) + abs(math.log2(beta))


def evaluate_fit(profile, segments, omega) -> dict:
    truth = truth_labels(profile)
    modes = modes_of(len(omega), segments)
    metrics = evaluate_profile(truth, modes, profile.segments, segments, omega, profile.model_mismatch)
    n_constant = sum(mode == 0 for _s, _e, mode in segments)
    coverage = sum(end - start for start, end, mode in segments if mode == 0) / len(omega)
    metrics.update(
        {
            "n_segments": len(segments),
            "n_constant_segments": n_constant,
            "n_linear_segments": len(segments) - n_constant,
            "constant_coverage": float(coverage),
        }
    )
    return metrics


def run_grid(cost_audit, pilot, profiles, alphas, betas, min_len: int) -> list[dict]:
    rows = []
    total = len(profiles) * len(alphas) * len(betas)
    done = 0
    clock = time.perf_counter()
    for profile in profiles:
        sigma = float(pilot.estimate_sigma(profile.rho))
        sigma2 = sigma * sigma
        for alpha in alphas:
            for beta in betas:
                segments, _fitted, objective, pen0, pen1, rss = segment_separated(
                    cost_audit, profile.omega, profile.rho, sigma2, alpha, beta, min_len
                )
                metrics = evaluate_fit(profile, segments, profile.omega)
                metrics.update(
                    {
                        "profile_id": profile.profile_id,
                        "split": profile.split,
                        "structure": profile.structure,
                        "noise_model": profile.noise_model,
                        "noise_sigma": profile.noise_sigma,
                        "seed": profile.seed,
                        "model_mismatch": int(profile.model_mismatch),
                        "score_included": int(profile.score_included),
                        "alpha": alpha,
                        "beta": beta,
                        "sigma_hat": sigma,
                        "penalty_constant": pen0,
                        "penalty_linear": pen1,
                        "total_rss": rss,
                        "objective": objective,
                    }
                )
                rows.append(metrics)
                done += 1
        if done % (len(alphas) * len(betas) * 12) == 0:
            rate = done / max(time.perf_counter() - clock, 1e-9)
            print(f"grid {done}/{total}  {rate:.1f} fits/s", flush=True)
    return rows


def aggregate(rows: list[dict], split: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] != split or not row["score_included"]:
            continue
        grouped[(row["alpha"], row["beta"])].append(row)
    summary = []
    for (alpha, beta), items in grouped.items():
        summary.append(
            {
                "alpha": alpha,
                "beta": beta,
                "n": len(items),
                "score": finite_mean(item["score"] for item in items),
                "constant_f1": finite_mean(item["constant_f1"] for item in items),
                "false_constant_coverage": finite_mean(item["false_constant_coverage"] for item in items),
                "missed_plateau_coverage": finite_mean(item["missed_plateau_coverage"] for item in items),
                "type_accuracy": finite_mean(item["type_accuracy"] for item in items),
                "abs_segment_count_error": finite_mean(abs(item["segment_count_error"]) for item in items),
                "boundary_mean_abs_index": finite_mean(item["boundary_mean_abs_index"] for item in items),
                "over_segmentation_rate": finite_mean(item["over_segmentation"] for item in items),
                "under_segmentation_rate": finite_mean(item["under_segmentation"] for item in items),
                "false_constant_rate": finite_mean(item["false_constant_flag"] for item in items),
                "missed_plateau_rate": finite_mean(item["missed_plateau_flag"] for item in items),
                "shallow_as_constant_rate": finite_mean(item["shallow_as_constant_flag"] for item in items),
                "plateau_as_linear_rate": finite_mean(item["plateau_as_linear_flag"] for item in items),
            }
        )
    return summary


def choose_cell(summary: list[dict]) -> dict:
    def key(item):
        return (
            item["score"],
            -item["false_constant_coverage"],
            -item["missed_plateau_coverage"],
            -grid_distance(item["alpha"], item["beta"]),
        )

    return max(summary, key=key)


def one_step_neighbors(alpha: float, beta: float, alphas, betas) -> set[tuple[float, float]]:
    a_index = alphas.index(alpha)
    b_index = betas.index(beta)
    cells = set()
    for i in range(max(0, a_index - 1), min(len(alphas), a_index + 2)):
        for j in range(max(0, b_index - 1), min(len(betas), b_index + 2)):
            cells.add((alphas[i], betas[j]))
    return cells


def robust_cells(summary: list[dict], chosen: dict) -> list[dict]:
    kept = [
        item
        for item in summary
        if item["score"] >= chosen["score"] - ROBUST_SCORE_GAP
        and item["false_constant_coverage"] <= chosen["false_constant_coverage"] + ROBUST_FALSE_GAP
    ]
    return sorted(kept, key=lambda item: (item["alpha"], item["beta"]))


def connected_components(cells: list[tuple[float, float]], alphas, betas) -> int:
    remaining = set(cells)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            alpha, beta = stack.pop()
            a_index = alphas.index(alpha)
            b_index = betas.index(beta)
            neighbors = []
            if a_index > 0:
                neighbors.append((alphas[a_index - 1], beta))
            if a_index + 1 < len(alphas):
                neighbors.append((alphas[a_index + 1], beta))
            if b_index > 0:
                neighbors.append((alpha, betas[b_index - 1]))
            if b_index + 1 < len(betas):
                neighbors.append((alpha, betas[b_index + 1]))
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def winner_by(summary: list[dict], metric: str, larger_is_better: bool) -> dict:
    if larger_is_better:
        return max(summary, key=lambda item: (item[metric], -grid_distance(item["alpha"], item["beta"])))
    return min(summary, key=lambda item: (item[metric], grid_distance(item["alpha"], item["beta"])))


def matrix_from(summary: list[dict], alphas, betas, metric: str) -> np.ndarray:
    lookup = {(item["alpha"], item["beta"]): item[metric] for item in summary}
    values = np.full((len(betas), len(alphas)), np.nan)
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            values[i, j] = lookup[(alpha, beta)]
    return values


def on_outer_boundary(alpha: float, beta: float, alphas, betas) -> bool:
    return alpha in (min(alphas), max(alphas)) or beta in (min(betas), max(betas))


def extend_grid(alphas, betas, chosen: dict) -> tuple[list[float], list[float], list[tuple[float, float]]]:
    new_alphas = list(alphas)
    new_betas = list(betas)
    if chosen["alpha"] == min(alphas):
        new_alphas = [min(alphas) / 2.0] + new_alphas
    elif chosen["alpha"] == max(alphas):
        new_alphas = new_alphas + [max(alphas) * 2.0]
    if chosen["beta"] == min(betas):
        new_betas = [min(betas) / 2.0] + new_betas
    elif chosen["beta"] == max(betas):
        new_betas = new_betas + [max(betas) * 2.0]
    old = {(alpha, beta) for alpha in alphas for beta in betas}
    added = [(alpha, beta) for alpha in new_alphas for beta in new_betas if (alpha, beta) not in old]
    return new_alphas, new_betas, added


def profile_summary_row(profile) -> dict:
    t_values = [segment["t_stat"] for segment in profile.segments if segment["segment_type"] == "linear"]
    return {
        "profile_id": profile.profile_id,
        "split": profile.split,
        "structure": profile.structure,
        "noise_model": profile.noise_model,
        "noise_sigma": profile.noise_sigma,
        "seed": profile.seed,
        "model_mismatch": int(profile.model_mismatch),
        "score_included": int(profile.score_included),
        "n_segments_truth": len(profile.segments),
        "segment_types": ";".join(segment["segment_type"] for segment in profile.segments),
        "segment_slopes": ";".join(f"{segment['slope']:.6g}" for segment in profile.segments),
        "segment_lengths": ";".join(str(segment["index_end"] - segment["index_start"]) for segment in profile.segments),
        "segment_omega_ranges": ";".join(
            f"{segment['omega_start']:.6f}:{segment['omega_end']:.6f}" for segment in profile.segments
        ),
        "detectability": ";".join(segment["detectability"] for segment in profile.segments),
        "min_linear_t_stat": min(t_values) if t_values else "",
        "weakly_identifiable": int(any(segment["detectability"] == "unresolved_slope" for segment in profile.segments)),
        "t_stat_threshold": T_STAT_DISTINGUISHABLE,
    }


def window_index(starts, ends, bounds) -> int:
    hits = [
        index
        for index, (start, end) in enumerate(zip(starts, ends))
        if abs(float(start) - bounds[0]) < 1e-6 and abs(float(end) - bounds[1]) < 1e-6
    ]
    if len(hits) != 1:
        raise RuntimeError(f"window {bounds} matched {hits}")
    return hits[0]


def load_real_cases() -> list[dict]:
    table = REPO / "stage1_analysis" / "pilot_exploration" / "representative_cases.csv"
    wanted = {row["case_id"]: row["run_id"] for row in csv.DictReader(table.open())}
    cases = []
    for case_id, (role, reason) in CASE_ROLES.items():
        record = load_profiles([wanted[case_id]], output_dir=DEFAULT_OUTPUT_DIR)[0]
        omega = np.asarray(record["omega"], dtype=float)
        rotation = np.asarray(record["rotation_numbers"], dtype=float)
        if np.any(np.diff(omega) < 0):
            order = np.argsort(omega, kind="mergesort")
            omega = omega[order]
            rotation = rotation[:, order]
        series = {}
        for name, bounds in WINDOWS.items():
            row = window_index(record["window_start"], record["window_end"], bounds)
            series[name] = rotation[row].copy()
        cases.append(
            {
                "case_id": case_id,
                "role": role,
                "reason": reason,
                "run_id": record["run_id"],
                "K": float(record["parameters"]["K"]),
                "p": float(record["parameters"]["p"]),
                "Delta": float(record["parameters"]["Delta"]),
                "omega": omega,
                "series": series,
            }
        )
    return cases


def constant_strings(omega, segments) -> dict:
    constants = [(start, end) for start, end, mode in segments if mode == 0]
    lengths = [end - start for start, end in constants]
    return {
        "constant_lengths": ";".join(str(length) for length in lengths),
        "constant_omega_ranges": ";".join(f"{omega[start]:.6f}:{omega[end - 1]:.6f}" for start, end in constants),
        "n_constant_length_lt_10": sum(length < 10 for length in lengths),
        "n_constant_length_lt_30": sum(length < 30 for length in lengths),
        "shortest_constant_length": min(lengths) if lengths else "",
        "median_constant_length": float(np.median(lengths)) if lengths else "",
    }


def diagnostic_fraction(omega, segments) -> float:
    mask = (omega >= DIAGNOSTIC_OMEGA[0]) & (omega <= DIAGNOSTIC_OMEGA[1])
    if not np.any(mask):
        return math.nan
    modes = modes_of(len(omega), segments)
    return float(np.mean(modes[mask] == 0))


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.sum(left | right)
    if union == 0:
        return 1.0
    return float(np.sum(left & right) / union)


def fit_real(cost_audit, pilot, case, window: str, alpha: float, beta: float, min_len: int) -> dict:
    omega = case["omega"]
    y = case["series"][window]
    sigma = float(pilot.estimate_sigma(y))
    segments, fitted, objective, pen0, pen1, rss = segment_separated(
        cost_audit, omega, y, sigma * sigma, alpha, beta, min_len
    )
    info = constant_strings(omega, segments)
    n_constant = sum(mode == 0 for _s, _e, mode in segments)
    coverage = sum(end - start for start, end, mode in segments if mode == 0) / len(y)
    row = {
        "case_id": case["case_id"],
        "role": case["role"],
        "reason": case["reason"],
        "run_id": case["run_id"],
        "K": case["K"],
        "p": case["p"],
        "Delta": case["Delta"],
        "window": window,
        "window_start": WINDOWS[window][0],
        "window_end": WINDOWS[window][1],
        "alpha": alpha,
        "beta": beta,
        "sigma_hat": sigma,
        "penalty_constant": pen0,
        "penalty_linear": pen1,
        "min_segment_length": min_len,
        "n_segments": len(segments),
        "n_constant_segments": n_constant,
        "n_linear_segments": len(segments) - n_constant,
        "constant_coverage": coverage,
        "total_rss": rss,
        "objective": objective,
        "diagnostic_omega_lo": DIAGNOSTIC_OMEGA[0] if case["case_id"] in {"C15", "C23"} else "",
        "diagnostic_omega_hi": DIAGNOSTIC_OMEGA[1] if case["case_id"] in {"C15", "C23"} else "",
        "diagnostic_constant_fraction": diagnostic_fraction(omega, segments) if case["case_id"] in {"C15", "C23"} else "",
    }
    row.update(info)
    return row, segments, fitted


def heatmap(ax, alphas, betas, values, title: str, robust: set[tuple[float, float]], chosen: tuple[float, float]):
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
    ax.set_yticks(range(len(betas)), [f"{beta:g}" for beta in betas])
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title(title, fontsize=10)
    finite = values[np.isfinite(values)]
    midpoint = float(np.median(finite)) if finite.size else 0.0
    for i in range(len(betas)):
        for j in range(len(alphas)):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if value >= midpoint else "white",
            )
            if (alphas[j], betas[i]) in robust:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", lw=1.4))
            if (alphas[j], betas[i]) == chosen:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#E69F00", lw=2.0))
    return image


def plot_fit(ax, omega, y, segments, fitted):
    ax.plot(omega, y, ",", color="0.45")
    for start, end, mode in segments:
        if mode == 0:
            ax.plot(omega[start:end], fitted[start:end], color=COLOR_CONSTANT, lw=1.8)
        else:
            ax.plot(omega[start:end], fitted[start:end], color=COLOR_LINEAR, lw=1.8, ls=(0, (5, 2)))


def main() -> None:
    pilot, cost_audit = load_method_c()
    min_len = int(pilot.MIN_SEGMENT)
    one_fit = time_one_fit(pilot, cost_audit, min_len)
    planned_profiles = (
        len(CALIBRATION_STRUCTURES) * (1 + (len(NOISE_LEVELS) - 1) * len(CALIBRATION_SEEDS))
        + len(VALIDATION_STRUCTURES) * (1 + (len(NOISE_LEVELS) - 1) * len(VALIDATION_SEEDS))
        + 2 * (len(NOISE_LEVELS) - 1) * len(VALIDATION_SEEDS)
        + (1 + (len(NOISE_LEVELS) - 1) * len(VALIDATION_SEEDS))
    )
    planned_fits = planned_profiles * len(ALPHAS) * len(BETAS)
    print(
        f"timing one_fit_s={one_fit:.4f} planned_profiles={planned_profiles} "
        f"planned_fits={planned_fits} estimated_s={planned_fits * one_fit:.1f}",
        flush=True,
    )
    calibration_seeds = CALIBRATION_SEEDS
    validation_seeds = VALIDATION_SEEDS
    if one_fit > 0.05:
        calibration_seeds = CALIBRATION_SEEDS[:2]
        validation_seeds = VALIDATION_SEEDS[:2]
        print("fit slower than 0.05 s; using two seeds per split", flush=True)

    profiles = build_profiles(calibration_seeds, validation_seeds)
    equivalence = []
    synthetic_probe = next(profile for profile in profiles if profile.structure == "lcl_long" and profile.noise_sigma == 0.0)
    noisy_probe = next(
        profile
        for profile in profiles
        if profile.structure == "slope_0p30" and profile.noise_sigma == 0.01 and profile.split == "calibration"
    )
    for tau in (0.5, 1.0, 2.0):
        equivalence.append(
            equivalence_case(pilot, cost_audit, synthetic_probe.omega, synthetic_probe.rho, tau, min_len, f"synthetic_noiseless_lcl_tau_{tau:g}")
        )
        equivalence.append(
            equivalence_case(pilot, cost_audit, noisy_probe.omega, noisy_probe.rho, tau, min_len, f"synthetic_slope_0p30_tau_{tau:g}")
        )
    cases = load_real_cases()
    for case in cases:
        if case["case_id"] not in {"C15", "C23", "C01", "C21", "C30"}:
            continue
        equivalence.append(
            equivalence_case(
                pilot, cost_audit, case["omega"], case["series"]["W3"], 1.0, min_len, f"{case['case_id']}_W3_tau_1"
            )
        )
    failed = [item for item in equivalence if not item["passed"]]
    if failed:
        print(json.dumps(failed, indent=2))
        raise SystemExit("equivalence test failed; calibration stopped")
    print(f"equivalence passed on {len(equivalence)} profiles", flush=True)

    rows = run_grid(cost_audit, pilot, profiles, ALPHAS, BETAS, min_len)
    alphas = list(ALPHAS)
    betas = list(BETAS)
    summary = aggregate(rows, "calibration")
    chosen = choose_cell(summary)
    initial_choice = {"alpha": chosen["alpha"], "beta": chosen["beta"], "score": chosen["score"]}
    extension_notes = []
    if on_outer_boundary(chosen["alpha"], chosen["beta"], alphas, betas):
        alphas, betas, added = extend_grid(alphas, betas, chosen)
        extension_notes.append({
            "reason": "selected cell was on the initial grid boundary",
            "added_pairs": [[float(alpha), float(beta)] for alpha, beta in added],
        })
        print(f"extending grid with {len(added)} cells", flush=True)
        rows.extend(run_pairs(cost_audit, pilot, profiles, added, min_len))
        summary = aggregate(rows, "calibration")
        chosen = choose_cell(summary)

    robust = robust_cells(summary, chosen)
    robust_pairs = [(item["alpha"], item["beta"]) for item in robust]
    components = connected_components(robust_pairs, alphas, betas)
    neighbors = one_step_neighbors(chosen["alpha"], chosen["beta"], alphas, betas)
    winners = {
        "constant_f1": winner_by(summary, "constant_f1", True),
        "false_constant_coverage": winner_by(summary, "false_constant_coverage", False),
        "missed_plateau_coverage": winner_by(summary, "missed_plateau_coverage", False),
        "abs_segment_count_error": winner_by(summary, "abs_segment_count_error", False),
    }
    far_winners = [
        name
        for name, item in winners.items()
        if (item["alpha"], item["beta"]) not in neighbors
    ]
    validation_summary = aggregate(rows, "validation")
    validation_chosen = next(
        item for item in validation_summary if item["alpha"] == chosen["alpha"] and item["beta"] == chosen["beta"]
    )
    easy = [
        row
        for row in rows
        if row["split"] == "calibration"
        and row["score_included"]
        and row["alpha"] == chosen["alpha"]
        and row["beta"] == chosen["beta"]
        and row["structure"] in EASY_STRUCTURES
        and row["noise_sigma"] <= 0.01
    ]
    easy_f1 = finite_mean(row["constant_f1"] for row in easy)
    easy_missed = finite_mean(row["missed_plateau_coverage"] for row in easy)
    checks = {
        "equivalence_passed": True,
        "easy_plateau_f1": easy_f1,
        "easy_plateau_missed": easy_missed,
        "easy_plateau_acceptable": bool(easy_f1 >= EASY_F1_LIMIT and easy_missed <= EASY_MISSED_LIMIT),
        "false_constant_coverage": chosen["false_constant_coverage"],
        "false_constant_acceptable": bool(chosen["false_constant_coverage"] <= FALSE_CONSTANT_LIMIT),
        "calibration_score": chosen["score"],
        "validation_score": validation_chosen["score"],
        "validation_drop": chosen["score"] - validation_chosen["score"],
        "validation_acceptable": bool(chosen["score"] - validation_chosen["score"] <= VALIDATION_DROP_LIMIT),
        "single_metric_winners_outside_neighborhood": far_winners,
        "winner_agreement_acceptable": len(far_winners) < 2,
        "selected_on_outer_boundary": on_outer_boundary(chosen["alpha"], chosen["beta"], alphas, betas),
    }
    freeze = all(
        [
            checks["easy_plateau_acceptable"],
            checks["false_constant_acceptable"],
            checks["validation_acceptable"],
            checks["winner_agreement_acceptable"],
            not checks["selected_on_outer_boundary"],
        ]
    )
    status = "candidate defaults available" if freeze else "Calibration inconclusive"
    print(status, flush=True)

    write_outputs(
        pilot,
        profiles,
        rows,
        summary,
        validation_summary,
        chosen,
        robust,
        winners,
        checks,
        status,
        initial_choice,
        extension_notes,
        alphas,
        betas,
        components,
        one_fit,
        equivalence,
    )
    plot_heatmaps(summary, alphas, betas, robust_pairs, (chosen["alpha"], chosen["beta"]))
    plot_examples(cost_audit, pilot, profiles, rows, chosen, min_len)
    real_rows, real_fits = run_real(cost_audit, pilot, cases, alphas, betas, min_len)
    write_real(real_rows)
    plot_real(cases, real_fits, chosen, summary)
    plot_robustness(summary, alphas, betas, robust_pairs, chosen, rows)
    print(f"wrote {OUT_DIR}", flush=True)


def run_pairs(cost_audit, pilot, profiles, pairs, min_len: int) -> list[dict]:
    rows = []
    for profile in profiles:
        sigma = float(pilot.estimate_sigma(profile.rho))
        sigma2 = sigma * sigma
        for alpha, beta in pairs:
            segments, _fitted, objective, pen0, pen1, rss = segment_separated(
                cost_audit, profile.omega, profile.rho, sigma2, alpha, beta, min_len
            )
            metrics = evaluate_fit(profile, segments, profile.omega)
            metrics.update(
                {
                    "profile_id": profile.profile_id,
                    "split": profile.split,
                    "structure": profile.structure,
                    "noise_model": profile.noise_model,
                    "noise_sigma": profile.noise_sigma,
                    "seed": profile.seed,
                    "model_mismatch": int(profile.model_mismatch),
                    "score_included": int(profile.score_included),
                    "alpha": alpha,
                    "beta": beta,
                    "sigma_hat": sigma,
                    "penalty_constant": pen0,
                    "penalty_linear": pen1,
                    "total_rss": rss,
                    "objective": objective,
                }
            )
            rows.append(metrics)
    return rows


GRID_FIELDS = [
    "profile_id", "split", "structure", "noise_model", "noise_sigma", "seed",
    "model_mismatch", "score_included", "alpha", "beta", "sigma_hat",
    "penalty_constant", "penalty_linear", "n_segments", "n_constant_segments",
    "n_linear_segments", "constant_coverage", "total_rss", "objective",
    "constant_precision", "constant_recall", "constant_f1",
    "false_constant_coverage", "missed_plateau_coverage", "type_accuracy",
    "segment_count_error", "boundary_mean_abs_index", "boundary_mean_abs_omega",
    "score", "over_segmentation", "under_segmentation", "false_constant_flag",
    "missed_plateau_flag", "shallow_as_constant_flag", "plateau_as_linear_flag",
]


def shallow_report(rows: list[dict], alpha: float, beta: float) -> list[dict]:
    names = {
        "slope_0p01", "slope_0p05", "slope_0p30", "slope_1",
        "slope_0p15", "slope_0p50", "slope_0p30_ar1", "pure_flat",
    }
    grouped = defaultdict(list)
    for row in rows:
        if row["alpha"] != alpha or row["beta"] != beta or row["structure"] not in names:
            continue
        if row["noise_model"] != "gaussian":
            continue
        grouped[(row["structure"], row["noise_sigma"], row["split"])].append(row)
    report = []
    for (structure, sigma, split), items in sorted(grouped.items()):
        report.append(
            {
                "structure": structure,
                "noise_sigma": sigma,
                "split": split,
                "mean_constant_coverage": finite_mean(item["constant_coverage"] for item in items),
                "mean_false_constant_coverage": finite_mean(item["false_constant_coverage"] for item in items),
                "mean_constant_f1": finite_mean(item["constant_f1"] for item in items),
                "fraction_score_included": finite_mean(item["score_included"] for item in items),
            }
        )
    return report


def write_outputs(
    pilot, profiles, rows, summary, validation_summary, chosen, robust, winners, checks, status,
    initial_choice, extension_notes, alphas, betas, components, one_fit, equivalence,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "synthetic_benchmark_summary.csv", [profile_summary_row(profile) for profile in profiles], [
        "profile_id", "split", "structure", "noise_model", "noise_sigma", "seed",
        "model_mismatch", "score_included", "n_segments_truth", "segment_types",
        "segment_slopes", "segment_lengths", "segment_omega_ranges", "detectability",
        "min_linear_t_stat", "weakly_identifiable", "t_stat_threshold",
    ])
    write_csv(OUT_DIR / "penalty_grid_results.csv", [row for row in rows if row["split"] == "calibration"], GRID_FIELDS)
    write_csv(OUT_DIR / "validation_results.csv", [row for row in rows if row["split"] == "validation"], GRID_FIELDS)
    method_hash = hashlib.sha256(Path("/Users/mohan/Desktop/SRS/data_examine/experiments/plateau_detection_pilot/pilot.py").read_bytes()).hexdigest()
    payload = {
        "status": status,
        "freeze_candidate_defaults": status == "candidate defaults available",
        "equivalence_passed": all(item["passed"] for item in equivalence),
        "equivalence": equivalence,
        "timing_one_fit_seconds": one_fit,
        "method_c_sha256": method_hash,
        "wrapper": "stage2_classifier/penalty_calibration/method_c_separated.py",
        "objective_new": "sum RSS_m + sigma_hat^2 * ln(N) * (alpha * M + beta * L)",
        "equivalence_map": "alpha = 2 * tau, beta = tau",
        "tie_break": "equal model costs keep constant; equal path costs keep the earliest start",
        "sigma_hat": "1.4826 * MAD(second difference) / sqrt(6), floor 1e-12",
        "min_segment_length": int(pilot.MIN_SEGMENT),
        "t_stat_threshold": T_STAT_DISTINGUISHABLE,
        "score_weights": {
            "constant_f1": 0.30,
            "one_minus_false_constant_coverage": 0.20,
            "one_minus_missed_plateau_coverage": 0.15,
            "type_accuracy": 0.15,
            "one_minus_count_penalty": 0.10,
            "one_minus_boundary_penalty": 0.10,
        },
        "score_notes": "Unresolved slopes are excluded from false-constant counts. Model-mismatch profiles are excluded from the score.",
        "initial_grid_choice": initial_choice,
        "grid_extension": extension_notes,
        "alphas": alphas,
        "betas": betas,
        "selected": {key: chosen[key] for key in chosen},
        "robust_region": robust,
        "robust_connected_components": components,
        "robust_definition": "score within 0.02 of the selected cell and false-constant coverage within 0.02 of that cell",
        "single_metric_winners": {
            name: {"alpha": item["alpha"], "beta": item["beta"], "score": item["score"], name: item[name]}
            for name, item in winners.items()
        },
        "checks": checks,
        "calibration_cells": summary,
        "validation_selected_cell": next(
            item for item in validation_summary if item["alpha"] == chosen["alpha"] and item["beta"] == chosen["beta"]
        ),
        "shallow_slope_at_selected": shallow_report(rows, chosen["alpha"], chosen["beta"]),
        "real_profiles_not_used_for_selection": list(CASE_ROLES),
    }
    (OUT_DIR / "candidate_parameters.json").write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    (OUT_DIR / "config.json").write_text(json.dumps({
        "method_c_file": "/Users/mohan/Desktop/SRS/data_examine/experiments/plateau_detection_pilot/pilot.py",
        "method_c_sha256": method_hash,
        "original_segment_profile_modified": False,
        "alphas_initial": ALPHAS,
        "betas_initial": BETAS,
        "alphas_used": alphas,
        "betas_used": betas,
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "noise_levels": list(NOISE_LEVELS),
        "master_seed": 20261101,
        "omega": [0.8, 1.2],
        "n": N_OSCILLATORS,
        "min_segment_length": int(pilot.MIN_SEGMENT),
        "case_roles": {case_id: {"role": role, "reason": reason} for case_id, (role, reason) in CASE_ROLES.items()},
    }, indent=2) + "\n", encoding="utf-8")


def plot_heatmaps(summary, alphas, betas, robust_pairs, chosen) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panels = [
        ("constant_f1", "Constant-region F1"),
        ("false_constant_coverage", "False constant coverage"),
        ("missed_plateau_coverage", "Missed plateau coverage"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        image = heatmap(ax, alphas, betas, matrix_from(summary, alphas, betas, metric), title, set(robust_pairs), chosen)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Figure 1. Calibration means. F1 and coverage are oscillator-level.\n"
        "White boxes are the robust set. The orange box is the selected cell. "
        "Unresolved slopes are not counted as false constants.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIGURE_DIR / "fig1_calibration_heatmaps.png", dpi=140)
    plt.close(fig)


def plot_examples(cost_audit, pilot, profiles, rows, chosen, min_len: int) -> None:
    by_id = {profile.profile_id: profile for profile in profiles}
    worst = max(aggregate(rows, "calibration"), key=lambda item: item["false_constant_coverage"])

    def find_profile(structure, sigma, split, seed):
        matches = [
            profile for profile in profiles
            if profile.structure == structure and profile.noise_sigma == sigma and profile.split == split and profile.seed == seed
        ]
        return matches[0]

    specs = [
        (find_profile("lcl_long", 0.01, "calibration", 0), chosen["alpha"], chosen["beta"], "selected, mixed profile"),
        (find_profile("slope_1", 0.01, "calibration", 0), worst["alpha"], worst["beta"], "highest false-constant cell"),
        (find_profile("lcl_short", 0.03, "calibration", 0), chosen["alpha"], max(betas_present(rows)), "short plateau, largest beta"),
        (find_profile("slope_0p01", 0.03, "calibration", 0), chosen["alpha"], chosen["beta"], "weak slope at selected cell"),
        (find_profile("lcl_long", 0.01, "calibration", 0), min(ALPHAS), min(BETAS), "weak penalty"),
        (find_profile("lcl_len180", 0.01, "validation", validation_seed(profiles)), chosen["alpha"], chosen["beta"], "validation length"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), sharex=True)
    for ax, (profile, alpha, beta, label) in zip(axes.ravel(), specs):
        sigma = float(pilot.estimate_sigma(profile.rho))
        segments, fitted, _objective, _pen0, _pen1, _rss = segment_separated(
            cost_audit, profile.omega, profile.rho, sigma * sigma, alpha, beta, min_len
        )
        plot_fit(ax, profile.omega, profile.rho, segments, fitted)
        ax.plot(profile.omega, profile.rho_clean, color="black", lw=1.0)
        ax.set_title(f"{label}\n{profile.structure}, sigma={profile.noise_sigma:g}, seed={profile.seed}\nalpha={alpha:g}, beta={beta:g}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, color="0.92", lw=0.4)
    axes[1, 0].set_xlabel("omega")
    axes[1, 1].set_xlabel("omega")
    axes[1, 2].set_xlabel("omega")
    axes[0, 0].set_ylabel("rotation number")
    axes[1, 0].set_ylabel("rotation number")
    fig.legend(
        handles=[
            Line2D([0], [0], color="0.2", lw=1.2, label="noiseless ground truth"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markersize=4, label="noisy profile"),
            Line2D([0], [0], color=COLOR_CONSTANT, lw=1.8, label="constant fit"),
            Line2D([0], [0], color=COLOR_LINEAR, lw=1.8, ls=(0, (5, 2)), label="linear fit"),
        ],
        loc="lower center", ncol=4, frameon=False, fontsize=8,
    )
    fig.suptitle("Figure 2. Synthetic examples at the indicated alpha and beta. Minimum segment length 4.", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(FIGURE_DIR / "fig2_synthetic_examples.png", dpi=140)
    plt.close(fig)
    del by_id


def betas_present(rows) -> list[float]:
    return sorted({row["beta"] for row in rows})


def validation_seed(profiles) -> int:
    seeds = sorted({profile.seed for profile in profiles if profile.split == "validation" and profile.noise_sigma > 0})
    return int(seeds[0])


def run_real(cost_audit, pilot, cases, alphas, betas, min_len: int):
    rows = []
    cache = {}
    for case in cases:
        for window in WINDOWS:
            for alpha in alphas:
                for beta in betas:
                    row, segments, fitted = fit_real(cost_audit, pilot, case, window, alpha, beta, min_len)
                    cache[(case["case_id"], window, alpha, beta)] = (segments, fitted)
                    rows.append(row)
    for row in rows:
        ref_segments, _fitted = cache[(row["case_id"], "W3", row["alpha"], row["beta"])]
        cur_segments, _fitted = cache[(row["case_id"], row["window"], row["alpha"], row["beta"])]
        n = N_OSCILLATORS
        ref_modes = modes_of(n, ref_segments)
        cur_modes = modes_of(n, cur_segments)
        row["vs_w3_type_mismatch_fraction"] = float(np.mean(ref_modes != cur_modes))
        row["vs_w3_constant_jaccard"] = jaccard(ref_modes == 0, cur_modes == 0)
        row["vs_w3_coverage_abs_difference"] = abs(float(np.mean(cur_modes == 0)) - float(np.mean(ref_modes == 0)))
    return rows, cache


def write_real(rows: list[dict]) -> None:
    fields = [
        "case_id", "role", "reason", "run_id", "K", "p", "Delta", "window",
        "window_start", "window_end", "alpha", "beta", "sigma_hat",
        "penalty_constant", "penalty_linear", "min_segment_length",
        "n_segments", "n_constant_segments", "n_linear_segments", "constant_coverage",
        "constant_lengths", "constant_omega_ranges", "n_constant_length_lt_10",
        "n_constant_length_lt_30", "shortest_constant_length", "median_constant_length",
        "total_rss", "objective", "diagnostic_omega_lo", "diagnostic_omega_hi",
        "diagnostic_constant_fraction", "vs_w3_type_mismatch_fraction",
        "vs_w3_constant_jaccard", "vs_w3_coverage_abs_difference",
    ]
    write_csv(OUT_DIR / "real_profile_diagnostics.csv", rows, fields)


def plot_real(cases, cache, chosen, summary) -> None:
    contrast = max(summary, key=lambda item: item["false_constant_coverage"])
    columns = [(2.0, 1.0, "original equivalent\nalpha=2, beta=1")]
    if (chosen["alpha"], chosen["beta"]) != (2.0, 1.0):
        columns.append((chosen["alpha"], chosen["beta"], f"selected\nalpha={chosen['alpha']:g}, beta={chosen['beta']:g}"))
    else:
        columns.append((chosen["alpha"], chosen["beta"], "selected equals\noriginal equivalent"))
    if (contrast["alpha"], contrast["beta"]) not in {(item[0], item[1]) for item in columns}:
        columns.append((contrast["alpha"], contrast["beta"], f"high false-constant cell\nalpha={contrast['alpha']:g}, beta={contrast['beta']:g}"))
    case_lookup = {case["case_id"]: case for case in cases}
    fig, axes = plt.subplots(len(FIGURE_CASES), len(columns), figsize=(4.2 * len(columns), 11.5), sharex=False, sharey=False)
    if len(columns) == 1:
        axes = np.array(axes)[:, None]
    for row_index, case_id in enumerate(FIGURE_CASES):
        case = case_lookup[case_id]
        for col_index, (alpha, beta, heading) in enumerate(columns):
            ax = axes[row_index, col_index]
            segments, fitted = cache[(case_id, "W3", alpha, beta)]
            plot_fit(ax, case["omega"], case["series"]["W3"], segments, fitted)
            if case_id == "C15":
                ax.axvspan(*DIAGNOSTIC_OMEGA, color="#f4e4b3", zorder=0)
            n_constant = sum(mode == 0 for _s, _e, mode in segments)
            ax.set_title(f"{case_id}  {heading}\n{n_constant} constant / {len(segments)} segments", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, color="0.92", lw=0.4)
            if col_index == 0:
                ax.set_ylabel(f"K={case['K']:.2f}, p={case['p']:.2f}\nDelta={case['Delta']:.2f}")
            if row_index == len(FIGURE_CASES) - 1:
                ax.set_xlabel("omega")
    fig.suptitle(
        "Figure 3. Real W3 profiles. The separated fit at alpha=2, beta=1 matches original Method C at tau=1.\n"
        "C15 shading is the diagnostic interval [0.93, 1.04]. No five-state label is assigned.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_DIR / "fig3_real_comparison.png", dpi=130)
    plt.close(fig)


def plot_robustness(summary, alphas, betas, robust_pairs, chosen, rows) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    image = heatmap(
        axes[0, 0], alphas, betas, matrix_from(summary, alphas, betas, "score"),
        "Calibration score", set(robust_pairs), (chosen["alpha"], chosen["beta"]),
    )
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)
    beta0 = chosen["beta"]
    along_alpha = [item for item in summary if item["beta"] == beta0]
    along_alpha.sort(key=lambda item: item["alpha"])
    for metric, label in (
        ("constant_f1", "F1"),
        ("false_constant_coverage", "false constant coverage"),
        ("missed_plateau_coverage", "missed plateau coverage"),
        ("score", "score"),
    ):
        axes[0, 1].plot([item["alpha"] for item in along_alpha], [item[metric] for item in along_alpha], marker="o", label=label)
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xlabel(f"alpha at beta={beta0:g}")
    axes[0, 1].legend(fontsize=7, frameon=False)
    axes[0, 1].set_title("Metrics along the selected beta")
    alpha0 = chosen["alpha"]
    along_beta = [item for item in summary if item["alpha"] == alpha0]
    along_beta.sort(key=lambda item: item["beta"])
    for metric, label in (
        ("constant_f1", "F1"),
        ("false_constant_coverage", "false constant coverage"),
        ("missed_plateau_coverage", "missed plateau coverage"),
        ("score", "score"),
    ):
        axes[1, 0].plot([item["beta"] for item in along_beta], [item[metric] for item in along_beta], marker="o", label=label)
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xlabel(f"beta at alpha={alpha0:g}")
    axes[1, 0].legend(fontsize=7, frameon=False)
    axes[1, 0].set_title("Metrics along the selected alpha")
    contrast = max(summary, key=lambda item: item["false_constant_coverage"])
    for (alpha, beta), label in (
        ((chosen["alpha"], chosen["beta"]), "selected"),
        ((2.0, 1.0), "alpha=2, beta=1"),
        ((contrast["alpha"], contrast["beta"]), "high false-constant cell"),
    ):
        xs, ys = [], []
        for sigma in NOISE_LEVELS:
            items = [
                row for row in rows
                if row["split"] == "calibration" and row["score_included"]
                and row["alpha"] == alpha and row["beta"] == beta and row["noise_sigma"] == sigma
            ]
            xs.append(sigma if sigma > 0 else 1e-4)
            ys.append(finite_mean(item["score"] for item in items))
        axes[1, 1].plot(xs, ys, marker="o", label=label)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("generating noise sigma (noiseless plotted at 1e-4)")
    axes[1, 1].set_ylabel("mean calibration score")
    axes[1, 1].legend(fontsize=7, frameon=False)
    axes[1, 1].set_title("Score by noise level")
    fig.suptitle(
        "Figure 4. White boxes mark the robust set around the orange selected cell.\n"
        "The score weights are fixed in config.json. This is not a probability of a dynamical state.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGURE_DIR / "fig4_parameter_robustness.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
