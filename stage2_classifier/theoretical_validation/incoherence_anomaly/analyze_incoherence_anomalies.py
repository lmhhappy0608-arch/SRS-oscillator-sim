"""Locate and diagnose operational Incoherence labels above the Eq. (7) boundary.

The screen and the diagnostics reuse the archived Classifier v1.0.0-pilot rules,
the printed Eq. (7) truncation, the Eq. (6) roots already reported for the
reliable widths, and the death-existence function from the death-boundary
validation. No ODE is integrated, no threshold is edited, and no archive is
rewritten.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-incoherence-anomaly")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "stage2_classifier" / "theoretical_validation" / "eq7_hopf"))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "theoretical_validation" / "death_boundary"))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "classifier_v1"))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "penalty_calibration"))
sys.path.insert(0, str(ROOT))

import validate_death_boundary as death  # noqa: E402
import validate_eq7 as hopf  # noqa: E402
from classify import (  # noqa: E402
    PHASE_COLORS,
    PHASE_STATES,
    classify_window,
    load_config,
    review_for_windows,
    window_row,
)
from method_c_separated import load_method_c  # noqa: E402
from stage1.model import make_adjacency, sha256_array  # noqa: E402
from stage1.query import query_runs  # noqa: E402

FIGURE_PATH = HERE / "incoherence_anomaly_diagnostics.png"
REPORT_PATH = HERE / "incoherence_anomaly_report.md"
WINDOWS = (("W1", 8000.0, 10000.0), ("W2", 10000.0, 12000.0), ("W3", 12000.0, 14000.0))
MAIN_DELTA_MAX = 0.20
# Eq. (6) roots printed in eq7_validation_report.md. Reliable only through
# delta=0.22; 0.225 and 0.25 had no valid root and are omitted. Five decimals
# is the printed precision, so a difference is uncertain by about 5e-6.
EQ6_ROOTS = {0.05: 0.12785, 0.10: 0.25920, 0.15: 0.39991, 0.20: 0.56905}
# Named controls. They are diagnosed even when they fail the strict screen,
# and they are never inserted into the strict set by name.
CONTROLS = (
    (0.15, 0.40, "near-boundary control"),
    (0.20, 0.60, "pre-locking control"),
    (0.20, 0.65, "global-locking control"),
    (0.20, 0.70, "non-monotonic transition control"),
    (0.20, 0.75, "non-monotonic transition control"),
)
CONTROL_MARKERS = {
    (0.15, 0.40): ("s", "#6a3d9a"),
    (0.20, 0.60): ("^", "#b15928"),
    (0.20, 0.65): ("*", "#1f78b4"),
    (0.20, 0.70): ("D", "#e31a1c"),
    (0.20, 0.75): ("P", "#ff7f00"),
}
HOPF_COLOR = "#e66101"
MISSING_COLOR = "#d0d0d0"
EXPECTED_W3 = {
    (0.15, 0.40): "Incoherence",
    (0.20, 0.60): "Incoherence",
    (0.20, 0.65): "Locking",
    (0.20, 0.70): "Incoherence",
    (0.20, 0.75): "Incoherence",
}


def eq6_root(delta: float) -> float | None:
    for key, value in EQ6_ROOTS.items():
        if abs(float(delta) - key) < 1e-8:
            return value
    return None


def match_named(delta: float, coupling: float) -> tuple | None:
    for item in CONTROLS:
        if abs(float(delta) - item[0]) < 1e-8 and abs(float(coupling) - item[1]) < 1e-8:
            return item
    return None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_small(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if value != 0.0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.6f}"


def require_config(config: dict) -> dict:
    """Stop if the archived rule card is not the pilot card used in the three validations."""
    if config["classifier_version"] != "v1.0.0-pilot":
        raise RuntimeError(config["classifier_version"])
    if config["state_priority"] != ["Death", "Locking", "Partial Death", "Partial Locking", "Incoherence"]:
        raise RuntimeError(config["state_priority"])
    alpha = float(config["alpha"]["value"])
    beta = float(config["beta"]["value"])
    min_len = int(config["method_c"]["minimum_segment_length"]["value"])
    near = float(config["rule_4_near_zero_tolerance"]["value"])
    death_level = float(config["rule_5_death"]["value"])
    partial_run = int(config["rule_6_partial_death"]["minimum_consecutive_near_zero_oscillators"])
    lock_range = float(config["rule_7_locking"]["maximum_profile_rho_range"])
    rule3 = config["rule_3_valid_plateau"]
    checks = {
        "alpha": abs(alpha - 2.0) < 1e-15,
        "beta": abs(beta - 1.0) < 1e-15,
        "min_len": min_len == 4,
        "near": abs(near - 0.001) < 1e-15,
        "death": abs(death_level - 1.0) < 1e-15,
        "partial_run": partial_run == 15,
        "lock": abs(lock_range - 0.01) < 1e-15,
        "rule3_n": int(rule3["minimum_oscillators"]) == 15,
        "rule3_rel": abs(float(rule3["minimum_relative_omega_span"]) - 0.02) < 1e-15,
        "rule3_span": abs(float(rule3["maximum_rho_span"]) - 0.02) < 1e-15,
        "rule3_level": abs(float(rule3["nonzero_level_for_partial_locking"]) - 0.001) < 1e-15,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"archived config does not match the pilot card: {failed}")
    anchor = death.existence_boundary(0.2)
    if anchor["status"] != "ok" or abs(float(anchor["K_exist"]) - 0.7902) > 5e-4:
        raise RuntimeError(f"death existence at delta=0.20 is not the validated value: {anchor}")
    series_15 = hopf.series_eq7(0.15)
    series_20 = hopf.series_eq7(0.20)
    if abs(series_15 - 0.39876) > 2e-4 or abs(series_20 - 0.55435) > 2e-4:
        raise RuntimeError(f"Eq. (7) series drifted: {series_15}, {series_20}")
    return {
        "alpha": alpha,
        "beta": beta,
        "min_len": min_len,
        "near": near,
        "death": death_level,
        "partial_run": partial_run,
        "lock_range": lock_range,
        "min_points": int(rule3["minimum_oscillators"]),
        "min_rel": float(rule3["minimum_relative_omega_span"]),
        "max_rho_span": float(rule3["maximum_rho_span"]),
        "nonzero_level": float(rule3["nonzero_level_for_partial_locking"]),
    }


def load_sorted_windows(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        omega = np.array(data["omega"], dtype=np.float64)
        rotation = np.array(data["rotation_numbers"], dtype=np.float64)
        starts = np.array(data["window_start"], dtype=np.float64)
        ends = np.array(data["window_end"], dtype=np.float64)
    if np.any(np.diff(omega) < 0):
        order = np.argsort(omega, kind="mergesort")
        omega = omega[order]
        rotation = rotation[:, order]
    rhos = {}
    for name, start, end in WINDOWS:
        rhos[name] = rotation[window_row(starts, ends, (start, end))]
    return omega, rhos


def fast_state(rho: np.ndarray, rules: dict) -> str | None:
    """Death, Locking, and Partial Death do not consult Method C.

    Returning None means the archived priority still has to separate Partial
    Locking from Incoherence, so Method C is required.
    """
    near = np.abs(rho) < rules["near"]
    if float(near.mean()) >= rules["death"]:
        return "Death"
    if float(rho.max() - rho.min()) <= rules["lock_range"]:
        return "Locking"
    best = current = 0
    for value in near:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    if best >= rules["partial_run"]:
        return "Partial Death"
    return None


def existence_cache(deltas: list[float]) -> dict[float, dict]:
    cached = {}
    for delta in deltas:
        key = round(float(delta), 5)
        if key in cached:
            continue
        item = death.existence_boundary(float(delta))
        if item["status"] != "ok":
            raise RuntimeError(f"death existence unresolved at delta={delta}: {item}")
        cached[key] = item
    return cached


def k_exist_of(cache: dict[float, dict], delta: float) -> float:
    return float(cache[round(float(delta), 5)]["K_exist"])


def classify_profile(cost_audit, pilot, omega, rho, config):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
        return classify_window(cost_audit, pilot, omega, rho, config)


def screen_record(row: dict, cache: dict[float, dict]) -> dict:
    delta = float(row["Delta"])
    coupling = float(row["K"])
    kh7 = hopf.series_eq7(delta) if delta > 0.0 else float("nan")
    kh6 = eq6_root(delta)
    exist = k_exist_of(cache, delta) if 0.0 < delta <= MAIN_DELTA_MAX + 1e-12 else float("nan")
    in_delta = 0.0 < delta <= MAIN_DELTA_MAX + 1e-12
    above = bool(in_delta and coupling > kh7)
    below_branch = bool(in_delta and coupling < 1.0 - delta)
    below_death = bool(in_delta and coupling < exist)
    strict = bool(in_delta and row["state"] == "Incoherence" and above and below_branch and below_death)
    excess7 = coupling - kh7 if in_delta else float("nan")
    excess6 = None if kh6 is None or not in_delta else coupling - kh6
    truncation = None if kh6 is None else kh6 - kh7
    if not strict:
        near = False
    elif kh6 is None:
        near = bool(excess7 <= 1e-3)
    else:
        # An excess that does not clear the known Eq. (6) truncation, or a
        # point within 0.001 of that root, is not a strong conflict.
        near = bool(excess7 <= max(0.0, truncation) + 1e-4 or (coupling - kh6) <= 1e-3)
    return {
        "kh7": kh7,
        "kh6": kh6,
        "k_exist": exist,
        "branch_edge": 1.0 - delta,
        "above_hopf": above,
        "below_branch": below_branch,
        "below_death": below_death,
        "strict": strict,
        "excess7": excess7,
        "excess6": excess6,
        "truncation": truncation,
        "near_boundary": near,
    }


def rule3_reason(segment: dict, rules: dict) -> str:
    if segment["mode"] != "constant":
        return "fitted as a linear segment, so it is not a constant candidate"
    reasons = []
    if segment["length"] < rules["min_points"]:
        reasons.append(f"length {segment['length']} < {rules['min_points']}")
    if segment["relative_omega_span"] < rules["min_rel"]:
        reasons.append(
            f"relative omega span {segment['relative_omega_span']:.4f} < {rules['min_rel']:.2f}"
        )
    if segment["rho_span"] > rules["max_rho_span"]:
        reasons.append(f"rho span {segment['rho_span']:.4f} > {rules['max_rho_span']:.2f}")
    if not reasons:
        if abs(segment["level"]) < rules["nonzero_level"]:
            return f"passes the span rules but |mean rho|={abs(segment['level']):.4f} < {rules['nonzero_level']}"
        return "passes Rule 3 and the nonzero level"
    return "; ".join(reasons)


def segment_at(segments: list[dict], index: int) -> dict | None:
    for segment in segments:
        if segment["start"] <= index < segment["end"]:
            return segment
    return None


def structure_metrics(omega: np.ndarray, rho: np.ndarray, rules: dict) -> dict:
    n = int(rho.size)
    lo = int(round(0.05 * n))
    hi = int(round(0.95 * n))
    central = rho[lo:hi]
    q05, q95 = np.quantile(rho, [0.05, 0.95])
    median = float(np.median(rho))
    deviate = np.abs(rho - median) > rules["lock_range"]
    indices = np.flatnonzero(deviate)
    if indices.size:
        deviate_omega_min = float(omega[indices].min())
        deviate_omega_max = float(omega[indices].max())
        deviate_index_min = int(indices.min())
    else:
        deviate_omega_min = float("nan")
        deviate_omega_max = float("nan")
        deviate_index_min = None
    if float(rho.std()) == 0.0 or float(omega.std()) == 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(omega, rho)[0, 1])
    slope = float(np.polyfit(omega, rho, 1)[0])
    return {
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "rho_range": float(rho.max() - rho.min()),
        "rho_mean": float(rho.mean()),
        "rho_median": median,
        "near_fraction": float(np.mean(np.abs(rho) < rules["near"])),
        "corr": corr,
        "slope": slope,
        "q05": float(q05),
        "q95": float(q95),
        "q_range": float(q95 - q05),
        "central90_range": float(central.max() - central.min()),
        "n_deviate": int(deviate.sum()),
        "n_within": int(n - deviate.sum()),
        "deviate_low": int(np.sum(indices < lo)) if indices.size else 0,
        "deviate_high": int(np.sum(indices >= hi)) if indices.size else 0,
        "deviate_mid": int(np.sum((indices >= lo) & (indices < hi))) if indices.size else 0,
        "deviate_omega_min": deviate_omega_min,
        "deviate_omega_max": deviate_omega_max,
        "deviate_index_min": deviate_index_min,
        "imin": int(np.argmin(rho)),
        "imax": int(np.argmax(rho)),
        "omega_at_min": float(omega[int(np.argmin(rho))]),
        "omega_at_max": float(omega[int(np.argmax(rho))]),
        "farthest": int(np.argmax(np.abs(rho - median))),
    }


def method_metrics(classified: dict, omega: np.ndarray, rho: np.ndarray, rules: dict, farthest: int) -> dict:
    segments = classified["segments"]
    constants = [row for row in segments if row["mode"] == "constant"]
    linears = [row for row in segments if row["mode"] == "linear"]
    constants_sorted = sorted(constants, key=lambda row: row["length"], reverse=True)
    flat_linear = [
        row
        for row in linears
        if row["length"] >= rules["min_points"] and row["rho_span"] <= rules["max_rho_span"]
    ]
    flat_linear.sort(key=lambda row: (row["rho_span"], -row["length"]))
    # A linear piece can look like a missed plateau only when it also meets the
    # archived frequency-span and length tests. A short piece of a slope does not.
    wide_flat = [
        row
        for row in flat_linear
        if row["relative_omega_span"] >= rules["min_rel"] and abs(row["level"]) >= rules["nonzero_level"]
    ]
    wide_flat.sort(key=lambda row: (-row["length"], row["rho_span"]))
    local_flat = [row for row in wide_flat if row["length"] < int(0.85 * rho.size)]
    carrier = segment_at(segments, farthest)
    carrier_short = bool(
        carrier is not None
        and carrier["length"] < rules["min_points"]
        and carrier["rho_span"] >= 0.5 * float(rho.max() - rho.min())
    )
    span_failures = []
    level_failures = []
    length_failures = []
    for row in constants:
        if row["length"] < rules["min_points"]:
            length_failures.append(row)
        elif row["relative_omega_span"] < rules["min_rel"] or row["rho_span"] > rules["max_rho_span"]:
            span_failures.append(row)
        elif abs(row["level"]) < rules["nonzero_level"]:
            level_failures.append(row)
    major = []
    for row in constants_sorted:
        if row["length"] >= rules["min_points"] or len(major) < 3:
            major.append(row)
        if len(major) >= 5:
            break
    if carrier is not None and carrier["mode"] == "constant" and all(item is not carrier for item in major):
        major.append(carrier)
    return {
        "n_segments": int(classified["n_segments"]),
        "n_constant": len(constants),
        "n_linear": len(linears),
        "n_validated": int(classified["validated_plateau_count"]),
        "plateau_coverage": float(classified["nonzero_plateau_mask"].mean()),
        "locked_fraction": float(classified["locked_fraction"]),
        "consecutive_near_zero": int(classified["consecutive_near_zero"]),
        "sigma_hat": float(classified["sigma_hat"]),
        "sigma_floor": int(classified["sigma_hat_at_floor"]),
        "largest": None if not constants_sorted else constants_sorted[0],
        "major": major,
        "n_short_constant": len(length_failures),
        "n_span_fail": len(span_failures),
        "n_level_fail": len(level_failures),
        "flat_linear": flat_linear,
        "wide_flat": wide_flat,
        "local_flat": local_flat,
        "carrier": carrier,
        "carrier_short": carrier_short,
        "longest_linear": None if not linears else max(linears, key=lambda row: row["length"]),
    }


def temporal_metrics(rhos: dict[str, np.ndarray], by_window: dict[str, dict]) -> dict:
    difference_early = np.abs(rhos["W2"] - rhos["W1"])
    difference_late = np.abs(rhos["W3"] - rhos["W2"])
    flag, reason = review_for_windows(by_window)
    states = [by_window[name]["state"] for name, *_bounds in WINDOWS]
    counts = [int(by_window[name]["validated_plateau_count"]) for name, *_bounds in WINDOWS]
    ranges = [float(by_window[name]["rho_range"]) for name, *_bounds in WINDOWS]
    coverages = [float(by_window[name]["nonzero_plateau_mask"].mean()) for name, *_bounds in WINDOWS]
    return {
        "median_D": float(np.median(difference_early)),
        "max_D": float(difference_early.max()),
        "median_F": float(np.median(difference_late)),
        "max_F": float(difference_late.max()),
        "frac_F_gt_lock": float(np.mean(difference_late > 0.01)),
        "states": states,
        "plateau_counts": counts,
        "ranges": ranges,
        "coverages": coverages,
        "review_flag": int(flag),
        "review_reason": reason,
        "label_change": len(set(states)) > 1,
        "count_change": len(set(counts)) > 1,
    }


def analysis_tags(diag: dict, rules: dict) -> list[str]:
    """Descriptive reading of one profile. These strings are not classifier states."""
    tags = []
    if diag["near_boundary"]:
        tags.append("Theoretically ambiguous near-boundary case")
    if diag["state"] == "Locking":
        tags.append("Global locking control")
        if diag["temporal"]["label_change"] or diag["temporal"]["count_change"]:
            tags.append("Temporally evolving profile")
        return tags
    near_lock = diag["state"] == "Incoherence" and diag["rho_range"] > rules["lock_range"] and (
        diag["central90_range"] <= rules["lock_range"] or diag["q_range"] <= rules["lock_range"]
    )
    # Descriptive only: 90% of oscillators inside the archived locking range of the median.
    majority_flat = diag["state"] == "Incoherence" and diag["n_within"] >= int(0.90 * diag["omega"].size)
    if near_lock:
        tags.append("Near-global-locking")
    elif majority_flat and diag["rho_range"] > rules["lock_range"]:
        tags.append("Flat majority with high-frequency tail")
    # A local shelf is a linear segment that would meet Rule 3 geometry, on a
    # profile that is not already a flat majority. The locking-style sigma-floor
    # partition of a flat bulk is not counted as a missed plateau.
    local_miss = bool(diag["method"]["wide_flat"]) and not near_lock and not majority_flat
    if local_miss and diag["method"]["n_validated"] == 0:
        tags.append("Possible missed local plateau")
    if diag["temporal"]["label_change"] or diag["temporal"]["count_change"]:
        tags.append("Temporally evolving profile")
    elif diag["temporal"]["max_F"] > rules["lock_range"] and diag["temporal"]["median_F"] > rules["near"]:
        tags.append("Temporally evolving profile")
    if diag["rho_range"] > 5.0 * rules["lock_range"] and abs(diag["corr"]) >= 0.9:
        tags.append("Broad drifting profile")
    elif (
        "Near-global-locking" not in tags
        and "Flat majority with high-frequency tail" not in tags
        and "Possible missed local plateau" not in tags
        and "Global locking control" not in tags
        and "Theoretically ambiguous near-boundary case" not in tags
    ):
        tags.append("Broad drifting profile" if diag["rho_range"] > rules["lock_range"] else "Tight unclassified profile")
    return tags


def situation_text(diag: dict) -> str:
    method = diag["method"]
    pieces = []
    if method["n_constant"] == 0 and not method["flat_linear"]:
        pieces.append("no constant candidate and no linear segment that is flat at the archived Rule 3 span")
    if method["wide_flat"]:
        lead = method["wide_flat"][0]
        pieces.append(
            f"{len(method['wide_flat'])} linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests "
            f"but are not constant candidates; the longest has length {lead['length']}, "
            f"relative omega span {lead['relative_omega_span']:.3f}, rho span {lead['rho_span']:.4f}, "
            f"mean {lead['level']:.4f}, omega [{lead['omega_start']:.4f}, {lead['omega_end']:.4f}]"
        )
    elif method["flat_linear"]:
        pieces.append(
            f"{len(method['flat_linear'])} linear segment(s) have rho span <= 0.02 and length >= 15, "
            "but none also reaches the archived relative omega span of 0.02"
        )
    if method["n_short_constant"]:
        pieces.append(
            f"{method['n_short_constant']} constant candidate(s) are shorter than 15 oscillators"
        )
    if method["n_span_fail"]:
        pieces.append(f"{method['n_span_fail']} constant candidate(s) fail the omega-span or rho-span rule")
    if method["n_level_fail"]:
        pieces.append(f"{method['n_level_fail']} Rule 3 segment(s) have a near-zero mean")
    if method["carrier_short"] and method["carrier"] is not None:
        carrier = method["carrier"]
        pieces.append(
            f"the oscillator farthest from the median sits in a {carrier['mode']} segment of length "
            f"{carrier['length']} whose rho span is {carrier['rho_span']:.4f}"
        )
    flat_majority = any(
        tag in diag.get("tags", [])
        for tag in ("Near-global-locking", "Flat majority with high-frequency tail", "Global locking control")
    )
    if flat_majority and method["wide_flat"]:
        pieces.append(
            f"sigma_hat={method['sigma_hat']:.3e}, so these linear pieces partition a flat majority; "
            "that is the noise-floor behavior already seen on Rule 7 Locking profiles, not a separate missed plateau"
        )
    if not pieces:
        pieces.append("no extra Method C failure mode was flagged")
    return "; ".join(pieces)


def load_grid(config: dict, rules: dict) -> tuple[list[dict], list[dict], dict]:
    started = time.perf_counter()
    all_runs = query_runs(status="completed", p=1.0)
    if not all_runs:
        raise RuntimeError("no completed p=1 runs")
    odd = [
        run
        for run in all_runs
        if int(run["N"]) != 1000 or int(run["trial_id"]) != 0
    ]
    if odd:
        raise RuntimeError(f"{len(odd)} completed p=1 runs are not N=1000, trial 0")
    band_runs = [run for run in all_runs if float(run["K"]) <= 1.0 + 1e-9]
    pilot, cost_audit = load_method_c()
    records = []
    method_calls = 0
    for index, run in enumerate(band_runs):
        omega, rhos = load_sorted_windows(run["output_npz"])
        rho = rhos["W3"]
        state = fast_state(rho, rules)
        if state is None:
            classified = classify_profile(cost_audit, pilot, omega, rho, config)
            state = classified["state"]
            method_calls += 1
        records.append(
            {
                "K": float(run["K"]),
                "Delta": float(run["Delta"]),
                "N": int(run["N"]),
                "trial_id": int(run["trial_id"]),
                "state": state,
                "rho_min": float(rho.min()),
                "rho_max": float(rho.max()),
                "rho_range": float(rho.max() - rho.min()),
                "rho_mean": float(rho.mean()),
                "near_fraction": float(np.mean(np.abs(rho) < rules["near"])),
                "output_npz": run["output_npz"],
                "run_id": run["run_id"],
            }
        )
        if index % 200 == 0:
            print(f"grid {index + 1}/{len(band_runs)} method_c={method_calls}", flush=True)
    keys = [(round(row["Delta"], 5), round(row["K"], 5)) for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate (delta, K) in the p=1, K<=1 slice")
    meta = {
        "all_runs": all_runs,
        "n_screened_index": len(all_runs),
        "n_loaded": len(records),
        "method_calls": method_calls,
        "seconds": time.perf_counter() - started,
        "pilot": pilot,
        "cost_audit": cost_audit,
    }
    print(f"grid done {len(records)} profiles, Method C calls {method_calls}, {meta['seconds']:.1f}s", flush=True)
    return records, all_runs, meta


def column_runs(records: list[dict], delta: float) -> str:
    rows = sorted(
        [row for row in records if abs(row["Delta"] - delta) < 1e-8 and row["K"] <= 1.0 + 1e-9],
        key=lambda row: row["K"],
    )
    compressed = []
    for row in rows:
        if not compressed or compressed[-1]["state"] != row["state"]:
            compressed.append({"state": row["state"], "k0": row["K"], "k1": row["K"]})
        else:
            compressed[-1]["k1"] = row["K"]
    return " | ".join(f"{item['state']} [{item['k0']:.2f}, {item['k1']:.2f}]" for item in compressed)


def diagnose_one(row: dict, screen: dict, config: dict, rules: dict, engines: dict, role: str) -> dict:
    omega, rhos = load_sorted_windows(row["output_npz"])
    by_window = {}
    for name, _start, _end in WINDOWS:
        by_window[name] = classify_profile(engines["cost_audit"], engines["pilot"], omega, rhos[name], config)
    if by_window["W3"]["state"] != row["state"]:
        raise RuntimeError(
            f"W3 state changed between passes at delta={row['Delta']}, K={row['K']}: "
            f"{row['state']} vs {by_window['W3']['state']}"
        )
    rho = rhos["W3"]
    structural = structure_metrics(omega, rho, rules)
    method = method_metrics(by_window["W3"], omega, rho, rules, structural["farthest"])
    temporal = temporal_metrics(rhos, by_window)
    validated_windows = {
        name: [row for row in classified["segments"] if row["validated_nonzero_plateau"]]
        for name, classified in by_window.items()
    }
    diag = {
        "delta": float(row["Delta"]),
        "K": float(row["K"]),
        "state": row["state"],
        "role": role,
        "strict": bool(screen["strict"]),
        "near_boundary": bool(screen["near_boundary"]),
        "kh7": float(screen["kh7"]),
        "kh6": screen["kh6"],
        "k_exist": float(screen["k_exist"]),
        "branch_edge": float(screen["branch_edge"]),
        "excess7": float(screen["excess7"]),
        "excess6": screen["excess6"],
        "truncation": screen["truncation"],
        "omega": omega,
        "rhos": rhos,
        "fitted": by_window["W3"]["fitted"],
        "segments": by_window["W3"]["segments"],
        "windows": by_window,
        "temporal": temporal,
        "validated_windows": validated_windows,
        "method": method,
        **structural,
    }
    diag["tags"] = analysis_tags(diag, rules)
    diag["situation"] = situation_text(diag)
    return diag


def choose_case_e(strict: list[dict]) -> list[dict]:
    """One or two strict candidates whose structure is not already Case A, C, or D."""
    reserved = {(0.15, 0.40), (0.20, 0.70), (0.20, 0.75)}
    pool = []
    for diag in strict:
        key = (round(diag["delta"], 2), round(diag["K"], 2))
        if key not in reserved:
            pool.append(diag)
    pool.sort(key=lambda diag: (-diag["excess7"], diag["delta"], diag["K"]))
    chosen = []
    seen = set()
    for diag in pool:
        label = tuple(tag for tag in diag["tags"] if tag != "Theoretically ambiguous near-boundary case")
        if not chosen or label not in seen:
            chosen.append(diag)
            seen.add(label)
        if len(chosen) == 2:
            break
    return chosen


def choose_temporal_panels(diags: list[dict]) -> list[dict]:
    strict = [diag for diag in diags if diag["strict"]]
    if not strict:
        return diags[:2]
    if max(diag["temporal"]["max_F"] for diag in strict) < 1e-3:
        ordered = sorted(strict, key=lambda diag: (not diag["near_boundary"], -diag["rho_range"]))
    else:
        ordered = sorted(strict, key=lambda diag: diag["temporal"]["max_F"], reverse=True)
    chosen = []
    for diag in ordered:
        if all(abs(diag["delta"] - item["delta"]) > 1e-8 or abs(diag["K"] - item["K"]) > 1e-8 for item in chosen):
            chosen.append(diag)
        if len(chosen) == 2:
            break
    return chosen


def choose_panel_d(diags: list[dict]) -> tuple[dict, str]:
    """Prefer a linear segment that meets Rule 3 geometry on a non-flat profile."""
    incoherence = [diag for diag in diags if diag["state"] == "Incoherence"]
    local = [diag for diag in incoherence if "Possible missed local plateau" in diag["tags"]]
    if local:
        local.sort(key=lambda diag: (-diag["method"]["wide_flat"][0]["length"], -diag["rho_range"]))
        return local[0], "linear segment meets Rule 3 geometry but is not constant"
    carriers = [diag for diag in incoherence if diag["method"]["carrier_short"]]
    if carriers:
        carriers.sort(key=lambda diag: diag["method"]["carrier"]["length"])
        return carriers[0], "no validated nonzero plateau; short segment carries the range"
    if not incoherence:
        return diags[0], "no Incoherence profile in the diagnosed set"
    incoherence.sort(key=lambda diag: -diag["rho_range"])
    return incoherence[0], "no validated nonzero plateau"


def continuity_text(strict: list[dict], k_step: float) -> str:
    if not strict:
        return "There are no strict candidates, so contiguity is undefined."
    by_delta = defaultdict(list)
    for diag in strict:
        by_delta[round(diag["delta"], 5)].append(diag["K"])
    parts = []
    isolated = 0
    for delta in sorted(by_delta):
        values = sorted(by_delta[delta])
        gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        broken = any(gap > k_step + 1e-8 for gap in gaps)
        if len(values) == 1:
            isolated += 1
            parts.append(f"delta={delta:.3f} has the single K={values[0]:.2f}")
        elif broken:
            rendered = ", ".join(f"{value:.2f}" for value in values)
            parts.append(f"delta={delta:.3f} has K={rendered}, with a gap larger than the {k_step:.2f} grid")
        else:
            parts.append(f"delta={delta:.3f} has a contiguous run from K={values[0]:.2f} to {values[-1]:.2f}")
    deltas = sorted(by_delta)
    neighbor_hits = 0
    for index, delta in enumerate(deltas):
        if index and abs(delta - deltas[index - 1] - 0.025) < 1e-8:
            neighbor_hits += 1
    joined = "; ".join(parts)
    return (
        f"{joined}. {isolated} delta value(s) contribute only one candidate. "
        f"{neighbor_hits} pair(s) of candidate deltas are adjacent on the 0.025 delta grid."
    )


def draw(records, diags, rules, temporal_points, panel_d_choice):
    """Draw the four-panel figure. Split from the draft so the annotation uses archived rules."""
    target, reason = panel_d_choice
    deltas = np.array(sorted({row["Delta"] for row in records}))
    ks = np.array(sorted({row["K"] for row in records if row["K"] <= 1.0 + 1e-9}))
    delta_edges = np.concatenate(
        ([deltas[0] - 0.0125], 0.5 * (deltas[:-1] + deltas[1:]), [deltas[-1] + 0.0125])
    )
    k_edges = np.concatenate(([ks[0] - 0.025], 0.5 * (ks[:-1] + ks[1:]), [ks[-1] + 0.025]))
    delta_index = {round(value, 5): index for index, value in enumerate(deltas)}
    k_index = {round(value, 5): index for index, value in enumerate(ks)}
    code = {state: index for index, state in enumerate(PHASE_STATES)}
    field = np.full((ks.size, deltas.size), np.nan)
    for row in records:
        if row["K"] > 1.0 + 1e-9:
            continue
        field[k_index[round(row["K"], 5)], delta_index[round(row["Delta"], 5)]] = code[row["state"]]
    cmap = ListedColormap(PHASE_COLORS)
    cmap.set_bad(MISSING_COLOR)
    norm = BoundaryNorm(np.arange(-0.5, len(PHASE_COLORS) + 0.5, 1), cmap.N)

    fig = plt.figure(figsize=(13.2, 11.4))
    outer = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28, bottom=0.16, top=0.90, left=0.07, right=0.98)
    ax_a = fig.add_subplot(outer[0, 0])
    b_grid = outer[0, 1].subgridspec(3, 1, hspace=0.16)
    axes_b = [fig.add_subplot(b_grid[0])]
    axes_b.append(fig.add_subplot(b_grid[1], sharex=axes_b[0], sharey=axes_b[0]))
    axes_b.append(fig.add_subplot(b_grid[2], sharex=axes_b[0], sharey=axes_b[0]))
    c_grid = outer[1, 0].subgridspec(len(temporal_points), 1, hspace=0.32)
    axes_c = [fig.add_subplot(c_grid[index]) for index in range(len(temporal_points))]
    ax_d = fig.add_subplot(outer[1, 1])

    ax_a.pcolormesh(delta_edges, k_edges, np.ma.masked_invalid(field), cmap=cmap, norm=norm, shading="flat", zorder=0)
    hopf_delta = np.linspace(0.001, MAIN_DELTA_MAX, 240)
    ax_a.plot(hopf_delta, [hopf.series_eq7(float(value)) for value in hopf_delta], color=HOPF_COLOR, lw=2.0, zorder=4)
    branch = np.linspace(0.0, 0.60, 200)
    ax_a.plot(branch, 1.0 - branch, color="black", lw=1.3, ls=(0, (1.2, 1.4)), zorder=4)
    strict = [diag for diag in diags if diag["strict"]]
    if strict:
        ax_a.scatter(
            [item["delta"] for item in strict],
            [item["K"] for item in strict],
            s=220,
            facecolors="none",
            edgecolors="black",
            linewidths=1.35,
            zorder=8,
        )
    for delta, coupling, _role in CONTROLS:
        marker, color = CONTROL_MARKERS[(delta, coupling)]
        ax_a.scatter(
            [delta],
            [coupling],
            s=78 if marker != "*" else 150,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=7,
        )
    ax_a.set_xlim(-0.02, 0.62)
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_xlabel("Frequency half-width δ")
    ax_a.set_ylabel("Coupling K")
    ax_a.set_title("A. Candidate locations")
    handles = [
        Patch(facecolor=color, edgecolor="0.35", linewidth=0.3, label=state)
        for state, color in zip(PHASE_STATES, PHASE_COLORS)
    ]
    handles.extend(
        [
            Line2D([0], [0], color=HOPF_COLOR, lw=2.0, label="Eq. (7), 0 < δ ≤ 0.20"),
            Line2D([0], [0], color="black", lw=1.3, ls=(0, (1.2, 1.4)), label="Eq. (5): K = 1 − δ"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor="black", markersize=7, label="Strict candidate"),
            Line2D([0], [0], marker="s", color="#6a3d9a", lw=0, label="Control δ=0.15, K=0.40"),
            Line2D([0], [0], marker="^", color="#b15928", lw=0, label="Control δ=0.20, K=0.60"),
            Line2D([0], [0], marker="*", color="#1f78b4", lw=0, markersize=9, label="Control δ=0.20, K=0.65"),
            Line2D([0], [0], marker="D", color="#e31a1c", lw=0, label="Control δ=0.20, K=0.70"),
            Line2D([0], [0], marker="P", color="#ff7f00", lw=0, label="Control δ=0.20, K=0.75"),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=6.4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    series_b = sorted(
        [item for item in diags if abs(item["delta"] - 0.20) < 1e-8 and round(item["K"], 2) in (0.65, 0.70, 0.75)],
        key=lambda item: item["K"],
    )
    colors_b = {0.65: "#0072B2", 0.70: "#E69F00", 0.75: "#D55E00"}
    for axis, diag in zip(axes_b, series_b):
        curve = diag["rhos"]["W3"]
        axis.plot(diag["omega"], curve, color=colors_b[round(diag["K"], 2)], lw=1.15)
        far = diag["farthest"]
        axis.scatter([diag["omega"][far]], [curve[far]], s=22, color="black", zorder=5)
        axis.text(
            0.02,
            0.80,
            f"K={diag['K']:.2f}, {diag['state']}, range={fmt_small(diag['rho_range'])}",
            transform=axis.transAxes,
            fontsize=8,
            va="top",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        axis.set_ylabel("ρ")
        axis.grid(True, alpha=0.25, lw=0.4)
    axes_b[0].set_title("B. δ = 0.20, shared scales", loc="left", fontsize=11)
    axes_b[-1].set_xlabel("Natural frequency ω")

    window_colors = {"W1": "#4C78A8", "W2": "#F58518", "W3": "#54A24B"}
    for axis, diag in zip(axes_c, temporal_points):
        for name, _start, _end in WINDOWS:
            axis.plot(diag["omega"], diag["rhos"][name], color=window_colors[name], lw=1.05, label=name)
        axis.set_ylabel("ρ")
        axis.set_title(
            f"δ={diag['delta']:.2f}, K={diag['K']:.2f}, max|W3−W2|={fmt_small(diag['temporal']['max_F'])}",
            fontsize=9,
            loc="left",
        )
        axis.grid(True, alpha=0.25, lw=0.4)
        axis.legend(fontsize=7, loc="best", frameon=False)
    axes_c[0].set_title("C. Temporal consistency", loc="left", fontsize=11)
    axes_c[-1].set_xlabel("Natural frequency ω")

    ax_d.plot(target["omega"], target["rhos"]["W3"], color="0.35", lw=0.9, label="W3 profile")
    ax_d.plot(target["omega"], target["fitted"], color="#D55E00", lw=1.2, label="Method C fit")
    highlight = None
    if target["method"]["wide_flat"] and "Possible missed local plateau" in target["tags"]:
        highlight = max(target["method"]["wide_flat"], key=lambda row: row["length"])
    elif target["method"]["carrier"] is not None:
        highlight = target["method"]["carrier"]
    elif target["segments"]:
        highlight = max(target["segments"], key=lambda row: row["rho_span"])
    if highlight is not None:
        ax_d.axvspan(
            target["omega"][highlight["start"]],
            target["omega"][highlight["end"] - 1],
            color="#0072B2",
            alpha=0.20,
            label="Highlighted segment",
        )
        note = (
            f"{highlight['mode']}, n={highlight['length']}, relative ω span={highlight['relative_omega_span']:.3f}, "
            f"ρ span={highlight['rho_span']:.4f}. {rule3_reason(highlight, rules)}"
        )
        ax_d.text(
            0.02,
            0.04,
            note,
            transform=ax_d.transAxes,
            fontsize=7.2,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
        )
    ax_d.set_xlabel("Natural frequency ω")
    ax_d.set_ylabel("ρ")
    ax_d.set_title(f"D. Method C at δ={target['delta']:.2f}, K={target['K']:.2f}\n{reason}", fontsize=10)
    ax_d.grid(True, alpha=0.25, lw=0.4)
    ax_d.legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle(
        "Operational Incoherence above Eq. (7): p=1, N=1000, trial 0, W3 primary, Classifier v1.0.0-pilot",
        fontsize=11.5,
    )
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def lookup(records: list[dict], delta: float, coupling: float) -> dict:
    hits = [
        row
        for row in records
        if abs(row["Delta"] - delta) < 1e-8 and abs(row["K"] - coupling) < 1e-8
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one cell at delta={delta}, K={coupling}, found {len(hits)}")
    return hits[0]


def provenance_paragraph(all_runs: list[dict]) -> str:
    database = Path(all_runs[0]["output_dir"]) / "index.sqlite"
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        totals = dict(connection.execute("SELECT COUNT(*) AS n, SUM(status='completed') AS completed FROM runs").fetchone())
        groups = list(
            connection.execute(
                """
                SELECT solver, dt, t_origin, t_end, master_seed, randomness_mode, seed_recipe,
                       network_seed, initial_phase_seed, frequency_seed, frequency_rule,
                       network_pairing_rule, phase_pairing_rule, model_version,
                       network_realization_id, initial_condition_id, n, trial_id,
                       COUNT(*) AS rows
                FROM runs
                WHERE status='completed' AND ABS(p-1.0)<1e-9
                GROUP BY solver, dt, t_origin, t_end, master_seed, randomness_mode, seed_recipe,
                         network_seed, initial_phase_seed, frequency_seed, frequency_rule,
                         network_pairing_rule, phase_pairing_rule, model_version,
                         network_realization_id, initial_condition_id, n, trial_id
                """
            ).fetchall()
        )
    if len(groups) != 1:
        raise RuntimeError(f"expected one p=1 provenance group, found {len(groups)}")
    group = dict(groups[0])
    sample = all_runs[0]
    adjacency = make_adjacency(int(sample["N"]), 1.0, int(group["network_seed"]))
    stored = __import__("json").loads(Path(sample["output_json"]).read_text())
    hash_ok = sha256_array(adjacency) == stored["adjacency_sha256"]
    diagonal = float(np.asarray(adjacency.diagonal()).sum())
    return (
        f"The Stage 1 index has {totals['n']} rows and {totals['completed']} completed runs. "
        f"The completed p=1 slice has {group['rows']} profiles, all with N={group['n']} and trial {group['trial_id']}. "
        f"One provenance group covers that slice: {group['solver']}, dt={group['dt']}, "
        f"t from {group['t_origin']:g} to {group['t_end']:g}, master_seed={group['master_seed']}, "
        f"seed_recipe={group['seed_recipe']}, {group['randomness_mode']} randomness, "
        f"network_seed={group['network_seed']}, initial_phase_seed={group['initial_phase_seed']}, "
        f"frequency rule {group['frequency_rule']}. Network realization {group['network_realization_id']} "
        f"and initial condition {group['initial_condition_id']} are shared. "
        f"The p=1 adjacency has diagonal sum {diagonal:.0f}. Regenerating it from the stored network seed "
        f"{'matches' if hash_ok else 'does not match'} the archived hash. "
        "W1=[8000, 10000], W2=[10000, 12000], and W3=[12000, 14000]. W3 assigns the primary state. "
        "The archived classification_results.csv contains the original 732-run pilot and does not cover this full p=1 rectangle. "
        "Labels used here were rebuilt in memory with Classifier v1.0.0-pilot and were not written back."
    )


def boundary_table(cache: dict[float, dict]) -> str:
    lines = [
        "| δ | K_H^(7) | K_H^(6) | K_H^(6) − K_H^(7) | 1 − δ | K_exist |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for delta in [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]:
        kh7 = hopf.series_eq7(delta)
        kh6 = eq6_root(delta)
        gap = "NA" if kh6 is None else f"{kh6 - kh7:+.5f}"
        root = "not solved" if kh6 is None else f"{kh6:.5f}"
        lines.append(
            f"| {delta:.3f} | {kh7:.5f} | {root} | {gap} | {1.0 - delta:.3f} | {k_exist_of(cache, delta):.4f} |"
        )
    return "\n".join(lines)


def candidate_table(strict: list[dict]) -> str:
    lines = [
        "| δ | K | K − K_H^(7) | K − K_H^(6) | K_exist | near-boundary | tags |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for diag in sorted(strict, key=lambda item: (item["delta"], item["K"])):
        excess6 = "NA" if diag["excess6"] is None else f"{diag['excess6']:+.5f}"
        lines.append(
            f"| {diag['delta']:.3f} | {diag['K']:.2f} | {diag['excess7']:+.5f} | {excess6} | "
            f"{diag['k_exist']:.4f} | {'yes' if diag['near_boundary'] else 'no'} | {'; '.join(diag['tags'])} |"
        )
    if len(lines) == 2:
        lines.append("| — | — | — | — | — | — | none |")
    return "\n".join(lines)


def control_table(diags: list[dict]) -> str:
    lines = [
        "| δ | K | W3 state | strict | near-boundary | assigned control type |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]
    by_key = {(round(diag["delta"], 2), round(diag["K"], 2)): diag for diag in diags}
    for delta, coupling, role in CONTROLS:
        diag = by_key[(round(delta, 2), round(coupling, 2))]
        lines.append(
            f"| {delta:.2f} | {coupling:.2f} | {diag['state']} | "
            f"{'yes' if diag['strict'] else 'no'} | {'yes' if diag['near_boundary'] else 'no'} | {role} |"
        )
    return "\n".join(lines)


def structure_table(diags: list[dict]) -> str:
    lines = [
        "| δ | K | state | range | mean | median | near-zero | corr(ρ, ω) | 5–95% width | central 90% range | outside median±0.01 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for diag in sorted(diags, key=lambda item: (item["delta"], item["K"])):
        lines.append(
            f"| {diag['delta']:.2f} | {diag['K']:.2f} | {diag['state']} | {fmt_small(diag['rho_range'])} | "
            f"{diag['rho_mean']:.4f} | {diag['rho_median']:.4f} | {diag['near_fraction']:.4f} | "
            f"{fmt(diag['corr'], 3)} | {fmt_small(diag['q_range'])} | {fmt_small(diag['central90_range'])} | "
            f"{diag['n_deviate']} |"
        )
    return "\n".join(lines)


def segment_brief(segment: dict | None, rules: dict) -> str:
    if segment is None:
        return "none"
    return (
        f"n={segment['length']}, ω span={segment['omega_span']:.4f}, "
        f"ρ span={segment['rho_span']:.4f}, mean={segment['level']:.4f}; {rule3_reason(segment, rules)}"
    )


def method_table(diags: list[dict], rules: dict) -> str:
    lines = [
        "| δ | K | constant segments | validated nonzero | largest constant candidate | short range carrier |",
        "| ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for diag in sorted(diags, key=lambda item: (item["delta"], item["K"])):
        method = diag["method"]
        carrier = "no"
        if method["carrier"] is not None:
            carrier = (
                f"{method['carrier']['mode']} n={method['carrier']['length']}, "
                f"ρ span={method['carrier']['rho_span']:.4f}"
            )
        lines.append(
            f"| {diag['delta']:.2f} | {diag['K']:.2f} | {method['n_constant']} | {method['n_validated']} | "
            f"{segment_brief(method['largest'], rules)} | {carrier} |"
        )
    return "\n".join(lines)


def temporal_table(diags: list[dict]) -> str:
    lines = [
        "| δ | K | med D | max D | med F | max F | W1, W2, W3 states | ranges | plateau counts | coverages | review flag |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for diag in sorted(diags, key=lambda item: (item["delta"], item["K"])):
        temporal = diag["temporal"]
        states = ", ".join(temporal["states"])
        ranges = ", ".join(fmt_small(value) for value in temporal["ranges"])
        counts = ", ".join(str(value) for value in temporal["plateau_counts"])
        covers = ", ".join(f"{value:.3f}" for value in temporal["coverages"])
        flag = "clear" if temporal["review_flag"] == 0 else temporal["review_reason"]
        lines.append(
            f"| {diag['delta']:.2f} | {diag['K']:.2f} | {fmt_small(temporal['median_D'])} | "
            f"{fmt_small(temporal['max_D'])} | {fmt_small(temporal['median_F'])} | {fmt_small(temporal['max_F'])} | "
            f"{states} | {ranges} | {counts} | {covers} | {flag} |"
        )
    return "\n".join(lines)


def why_incoherence(diag: dict) -> str:
    method = diag["method"]
    return (
        f"The W3 label is {diag['state']}. Near-zero fraction {diag['near_fraction']:.4f} "
        f"(Death requires 1), ρ range {fmt_small(diag['rho_range'])} "
        f"(Locking requires range ≤ 0.01), longest consecutive near-zero run "
        f"{method['consecutive_near_zero']} (Partial Death requires 15), and "
        f"{method['n_validated']} validated nonzero plateau(s) "
        f"(Partial Locking requires one). Sigma_hat is {diag['method']['sigma_hat']:.3e}"
        f"{', at its floor' if diag['method']['sigma_floor'] else ''}."
    )


def validated_sentence(diag: dict) -> str:
    parts = []
    for name, _start, _end in WINDOWS:
        rows = diag["validated_windows"][name]
        if not rows:
            continue
        bits = [
            (
                f"n={row['length']}, relative ω span={row['relative_omega_span']:.3f}, "
                f"ρ span={row['rho_span']:.4f}, mean={row['level']:.4f}, "
                f"ω=[{row['omega_start']:.4f}, {row['omega_end']:.4f}]"
            )
            for row in rows
        ]
        parts.append(f"{name}: " + "; ".join(bits))
    if not parts:
        return "No window contains a validated nonzero plateau."
    return "Validated nonzero plateaus by window — " + " | ".join(parts) + "."


def case_block(letter: str, title: str, diag: dict, column: str, rules: dict) -> str:
    excess6 = "not available" if diag["excess6"] is None else f"{diag['excess6']:+.5f}"
    truncation = "not available" if diag["truncation"] is None else f"{diag['truncation']:+.5f}"
    temporal = diag["temporal"]
    return "\n".join(
        [
            f"### Case {letter}: {title}",
            "",
            (
                f"Control type on the named list, where applicable: {diag['role']}. "
                f"Strict candidate: {'yes' if diag['strict'] else 'no'}. "
                f"Near-boundary flag: {'yes' if diag['near_boundary'] else 'no'}. "
                f"K_H^(7)={diag['kh7']:.5f}, K−K_H^(7)={diag['excess7']:+.5f}, "
                f"K_H^(6) difference={excess6}, truncation K_H^(6)−K_H^(7)={truncation}, "
                f"1−δ={diag['branch_edge']:.3f}, K_exist={diag['k_exist']:.4f}."
            ),
            "",
            (
                f"W3 ρ lies in [{diag['rho_min']:.4f}, {diag['rho_max']:.4f}], range {fmt_small(diag['rho_range'])}, "
                f"mean {diag['rho_mean']:.4f}, median {diag['rho_median']:.4f}, near-zero fraction {diag['near_fraction']:.4f}. "
                f"Pearson correlation of ρ with ω is {fmt(diag['corr'], 3)} and the least-squares slope is {diag['slope']:.4f} per unit frequency. "
                f"The 5–95% ρ width is {fmt_small(diag['q_range'])}. "
                f"The central 90% of oscillators in frequency order have ρ range {fmt_small(diag['central90_range'])}. "
                f"{diag['n_within']} oscillators lie within the archived locking range 0.01 of the median, and "
                f"{diag['n_deviate']} lie outside it ({diag['deviate_low']} in the low-frequency 5%, "
                f"{diag['deviate_mid']} in the interior, {diag['deviate_high']} in the high-frequency 5%). "
                + (
                    "The minimum is at ω={:.4f} and the maximum at ω={:.4f}.".format(
                        diag["omega_at_min"], diag["omega_at_max"]
                    )
                    if diag["n_deviate"] == 0
                    else (
                        "The lowest frequency among those outside points is ω={:.4f}. "
                        "The minimum is at ω={:.4f} and the maximum at ω={:.4f}."
                    ).format(diag["deviate_omega_min"], diag["omega_at_min"], diag["omega_at_max"])
                )
            ),
            "",
            why_incoherence(diag),
            "",
            (
                f"Method C: {diag['method']['n_segments']} segments, {diag['method']['n_constant']} constant and "
                f"{diag['method']['n_linear']} linear, {diag['method']['n_validated']} validated nonzero plateaus, "
                f"plateau coverage {diag['method']['plateau_coverage']:.3f}. "
                f"Largest constant candidate: {segment_brief(diag['method']['largest'], rules)}. "
                f"Reading of the segment record: {diag['situation']}. {validated_sentence(diag)}"
            ),
            "",
            (
                f"Window changes: median(D)={fmt_small(temporal['median_D'])}, max(D)={fmt_small(temporal['max_D'])}, "
                f"median(F)={fmt_small(temporal['median_F'])}, max(F)={fmt_small(temporal['max_F'])}. "
                f"States W1/W2/W3 = {' / '.join(temporal['states'])}. "
                f"Ranges = {', '.join(fmt_small(value) for value in temporal['ranges'])}. "
                f"Validated plateau counts = {', '.join(str(value) for value in temporal['plateau_counts'])}. "
                f"Plateau coverages = {', '.join(f'{value:.3f}' for value in temporal['coverages'])}. "
                f"Archived temporal review flag: {'set (' + temporal['review_reason'] + ')' if temporal['review_flag'] else 'clear'}. "
                "Agreement of the three window labels and a small pointwise change do not prove that the trajectory has converged."
            ),
            "",
            f"Diagnostic tags, which are not new states: {'; '.join(diag['tags'])}.",
            "",
            f"W3 column for this δ, restricted to K≤1: {column}.",
            "",
        ]
    )


def count_tag(diags: list[dict], tag: str) -> int:
    return sum(tag in diag["tags"] for diag in diags)


def recommendation(strict: list[dict]) -> str:
    if not strict:
        return (
            "No strict candidate was found. The useful next check is a finer K grid on the δ=0.20 column "
            "between the single Locking cell and the death onset, classified with the unchanged v1 rules, "
            "before any threshold is reconsidered."
        )
    keyed = {(round(diag["delta"], 2), round(diag["K"], 2)): diag for diag in strict}
    tail_keys = [(0.20, 0.60), (0.20, 0.70), (0.20, 0.75)]
    if all(key in keyed for key in tail_keys):
        tail_60 = keyed[(0.20, 0.60)]
        tail_70 = keyed[(0.20, 0.70)]
        tail_75 = keyed[(0.20, 0.75)]
        return (
            "The most useful next experiment is a finer K scan at δ=0.20, from just below the Eq. (6) root "
            f"(K_H^(6)={tail_60['kh6']:.5f}) through the death onset at K_exist={tail_60['k_exist']:.4f}, "
            "using the same N, windows, and unchanged Classifier v1. "
            f"On the stored 0.05 grid the high-frequency tail has {tail_60['n_deviate']} oscillators at K=0.60, "
            f"none at the single Locking cell K=0.65, {tail_70['n_deviate']} oscillator at K=0.70, and "
            f"{tail_75['n_deviate']} oscillators at K=0.75. "
            "The scan measures how wide the locking interval is and how the tail grows. It does not retune the 0.01 rule. "
            "The δ=0.15, K=0.40 cell is a separate check: it lies on the Eq. (6) root, and its only window disagreement "
            "is one validated plateau of about 2% coverage in W1. A second initial phase at that single point can test the shelf. "
            "A linear-stability calculation is still required before either label is identified with the Hopf eigenvalue."
        )
    near_lock = [
        diag
        for diag in strict
        if "Near-global-locking" in diag["tags"] and not diag["near_boundary"]
    ]
    missed = [diag for diag in strict if "Possible missed local plateau" in diag["tags"]]
    evolving = [diag for diag in strict if "Temporally evolving profile" in diag["tags"]]
    if evolving:
        listed = ", ".join(f"δ={diag['delta']:.2f}, K={diag['K']:.2f}" for diag in evolving)
        return (
            f"The first new experiment should extend the integration, or read the already saved order-parameter "
            f"time series, at {listed}. The three rotation-number windows disagree enough that a longer record "
            f"is required before those profiles can be treated as stationary. The classifier thresholds stay fixed."
        )
    if missed:
        listed = ", ".join(f"δ={diag['delta']:.2f}, K={diag['K']:.2f}" for diag in missed)
        return (
            f"The first new experiment should repeat {listed} from additional initial phases at the same "
            f"(K, δ), still classified with unchanged Method C. A single flat piece fit as linear on one "
            f"trajectory is not enough to decide that a plateau was missed. No threshold change is part of that run."
        )
    if near_lock:
        listed = ", ".join(f"δ={diag['delta']:.2f}, K={diag['K']:.2f}" for diag in near_lock)
        return (
            f"The most informative next experiment is a finer K scan at δ=0.20, from just below the Eq. (6) root "
            f"through the death onset, with the same N, trial protocol, windows, and unchanged Classifier v1. "
            f"The present 0.05 grid shows Locking at one cell and a return to Incoherence at the next cells, "
            f"and {listed} already have a bulk rotation number inside the archived locking range. "
            f"The scan measures how wide that locking interval is. It does not retune the 0.01 rule. "
            f"A separate linear-stability calculation would still be required before any of these labels is identified with the Hopf crossing."
        )
    boundary = [diag for diag in strict if diag["near_boundary"]]
    if boundary:
        listed = ", ".join(f"δ={diag['delta']:.2f}, K={diag['K']:.2f}" for diag in boundary)
        return (
            f"The most informative next experiment is a finer K grid around {listed}, spanning the Eq. (6) root "
            f"and one stored step on either side. The present excess over Eq. (7) is inside the known truncation, "
            f"so the stored cell cannot decide which side of the boundary the continuum threshold occupies. "
            f"Classify the new profiles with the unchanged v1 rules."
        )
    return (
        "The strict candidates are broad drifting profiles above both the series and the Eq. (6) root. "
        "The next experiment is a finite-N linear-stability calculation at one of those points and at the "
        "neighboring Locking cell. Rotation numbers alone do not identify the unstable mode."
    )


def needs_new_data(strict: list[dict], diags: list[dict]) -> str:
    lines = []
    for diag in sorted(diags, key=lambda item: (item["delta"], item["K"])):
        reasons = []
        if diag["near_boundary"]:
            reasons.append("the distance to the Hopf boundary is inside the Eq. (7) truncation, so this grid cannot place the crossing")
        if "Temporally evolving profile" in diag["tags"]:
            reasons.append("the windows disagree at the archived review-flag level or the pointwise change exceeds the locking range in the median as well as the maximum")
        if "Near-global-locking" in diag["tags"] or "Flat majority with high-frequency tail" in diag["tags"]:
            if diag["temporal"]["label_change"]:
                reasons.append("the flat majority is already visible, while the windows still disagree")
            else:
                reasons.append(
                    "the three windows already agree on a flat majority plus a high-frequency tail; "
                    "a longer copy of this trajectory is less useful than a finer K grid"
                )
        if "Possible missed local plateau" in diag["tags"]:
            reasons.append(
                "W3 contains a linear segment that meets Rule 3 geometry; one trajectory cannot show that the shelf is a stable plateau"
            )
        if not reasons and diag["state"] == "Locking" and not diag["temporal"]["label_change"]:
            reasons.append("the three windows already agree on global locking; new data are not required to describe this control")
        if reasons:
            lines.append(f"- δ={diag['delta']:.2f}, K={diag['K']:.2f}: " + "; ".join(reasons) + ".")
    if not lines:
        lines.append("- No diagnosed point asked for a further dynamical record beyond the checks already in hand.")
    return "\n".join(lines)


def write_report(context: dict) -> None:
    rules = context["rules"]
    strict = context["strict"]
    diags = context["diags"]
    cache = context["cache"]
    counts = context["counts"]
    near_lock = [diag for diag in strict if "Near-global-locking" in diag["tags"]]
    missed = [diag for diag in strict if "Possible missed local plateau" in diag["tags"]]
    evolving = [diag for diag in strict if "Temporally evolving profile" in diag["tags"]]
    outside = [diag for diag in strict if not diag["near_boundary"]]
    by_key = {(round(diag["delta"], 2), round(diag["K"], 2)): diag for diag in diags}
    case_a = by_key[(0.15, 0.40)]
    case_b = by_key[(0.20, 0.65)]
    case_c = by_key[(0.20, 0.70)]
    case_d = by_key[(0.20, 0.75)]
    excesses = [diag["excess7"] for diag in strict]
    excess_text = "NA" if not excesses else (
        f"minimum {min(excesses):+.5f}, median {float(np.median(excesses)):+.5f}, maximum {max(excesses):+.5f}"
    )
    regions = []
    by_delta = defaultdict(list)
    for diag in strict:
        by_delta[round(diag["delta"], 5)].append(diag["K"])
    for delta in sorted(by_delta):
        values = sorted(by_delta[delta])
        noun = "cell" if len(values) == 1 else "cells"
        regions.append(f"δ={delta:.3f}, K from {values[0]:.2f} to {values[-1]:.2f} ({len(values)} {noun})")
    region_text = "none" if not regions else "; ".join(regions)
    return_sentence = (
        f"Along δ=0.20 the W3 column on K≤1 is {context['column_020']}. "
        f"K=0.65 is Locking with ρ range {fmt_small(case_b['rho_range'])}. "
        f"K=0.60 has {by_key[(0.20, 0.60)]['n_deviate']} oscillators outside the median, all at high frequency, "
        f"with range {fmt_small(by_key[(0.20, 0.60)]['rho_range'])}. "
        f"K=0.70 has ρ range {fmt_small(case_c['rho_range'])}, central-90% range {fmt_small(case_c['central90_range'])}, "
        f"and {case_c['n_deviate']} oscillator outside the median, at ω={case_c['omega_at_max']:.4f}. "
        f"K=0.75 has ρ range {fmt_small(case_d['rho_range'])}, central-90% range {fmt_small(case_d['central90_range'])}, "
        f"and {case_d['n_deviate']} oscillators outside the median, the lowest of them at ω={case_d['deviate_omega_min']:.4f}. "
        "The return at K=0.70 is that one high-frequency oscillator. "
        "K=0.75 keeps a flat majority and replaces the single oscillator with a high-frequency tail. "
        "Neither profile is the frequency-tracking shape seen at δ=0.15, K=0.40."
    )
    stable_windows = sum(1 for diag in strict if not diag["temporal"]["label_change"] and not diag["temporal"]["count_change"])
    lines = [
        "# Incoherence anomaly diagnostic",
        "",
        "Operational Incoherence on the saved p=1 grid is compared with the printed Eq. (7) Hopf truncation. "
        "The linear instability of the incoherent state, the finite-time rotation-number profile, and the Classifier v1 label are kept separate. "
        "A green cell above the truncation is not treated as a misclassification by default.",
        "",
        "## 1. Data and classifier provenance",
        "",
        context["provenance"],
        "",
        "Classifier v1.0.0-pilot reads alpha=2, beta=1, and Method C minimum segment length 4 from the archived config. "
        "Near-zero means |ρ|<0.001. Death requires near-zero fraction 1. Locking requires the full-profile ρ range ≤ 0.01. "
        "Partial Death requires at least 15 consecutive near-zero oscillators after those two rules. "
        "Partial Locking requires at least one Method C constant segment that passes Rule 3 "
        "(at least 15 oscillators, relative ω span ≥ 0.02, ρ span ≤ 0.02) with |mean ρ|≥0.001. "
        "Incoherence is the remaining label. The priority is Death, Locking, Partial Death, Partial Locking, Incoherence. "
        "Cells with K>1 were screened from the index and not loaded: for every δ in (0, 0.20], both 1−δ and K_exist are below 1, "
        "so K≥1 cannot satisfy the strict inequalities. "
        f"Profiles with K≤1 were loaded ({context['n_loaded']} cells). "
        f"Method C ran on the {context['method_calls']} W3 profiles that were not already Death, Locking, or Partial Death. "
        "Diagnosed cells were then segmented on W1, W2, and W3.",
        "",
        f"State counts on the loaded K≤1 slice: {counts}.",
        "",
        "## 2. Theoretical candidate definition",
        "",
        "Eq. (7) is the printed product truncation, reused from the Eq. (7) validation and not refit:",
        "",
        "K_H^(7)(δ) = (8δ/π) [ 1 + 16δ²/π² + 16(π²+80)δ⁴/π⁴ ].",
        "",
        "Eq. (5) on the incoherent branch is K_c = 1−δ. "
        "K_exist(δ) is the death-equilibrium existence value from the same function used in the death-boundary validation. "
        "At δ=0.20 that function returns the previously checked value 0.7902 within 5×10⁻⁴. "
        "Eq. (6) roots are the values already printed for δ=0.05, 0.10, 0.15, and 0.20. "
        "Widths that were not solved, or that had no valid root, do not receive a K−K_H^(6) entry. "
        "No boundary was interpolated or refit.",
        "",
        boundary_table(cache),
        "",
        "A strict candidate meets all five conditions: W3 state Incoherence, 0<δ≤0.20, K>K_H^(7)(δ), K<1−δ, and K<K_exist(δ). "
        "Equality on Eq. (7) is not a candidate. "
        "A strict candidate is marked near-boundary when a reliable Eq. (6) root exists and the excess K−K_H^(7) does not exceed "
        "that truncation gap by more than 10⁻⁴, or when K is within 0.001 of the Eq. (6) root. "
        "If no Eq. (6) root is available, the near-boundary mark requires K−K_H^(7)≤0.001. "
        "The mark keeps the truncation visible. It does not remove the cell from the strict count and it does not create a new state.",
        "",
        "Named controls are δ=0.15, K=0.40; δ=0.20, K=0.60; δ=0.20, K=0.65; δ=0.20, K=0.70; and δ=0.20, K=0.75. "
        "They are diagnosed whether or not they meet the five conditions. "
        "A control enters the strict count only when the five conditions are true. The control list is not an inclusion rule.",
        "",
        "## 3. Strict candidate counts and locations",
        "",
        (
            f"The K≤1 slice has {context['funnel']['band_incoherence']} Incoherence cells with 0<δ≤0.20. "
            f"Of those, {context['funnel']['above']} have K>K_H^(7), "
            f"{context['funnel']['above_and_branch']} also have K<1−δ, and "
            f"{context['funnel']['strict']} also have K<K_exist. "
            f"That last number is the strict-candidate count: {len(strict)}. "
            f"They occupy {len(by_delta)} delta value(s): {region_text}. "
            f"Distances K−K_H^(7): {excess_text}. "
            f"{len(outside)} strict candidates lie outside the near-boundary mark, and "
            f"{sum(diag['near_boundary'] for diag in strict)} "
            f"{'is' if sum(diag['near_boundary'] for diag in strict) == 1 else 'are'} marked near-boundary."
        ),
        "",
        candidate_table(strict),
        "",
        continuity_text(strict, context["k_step"]),
        "",
        "Named controls, kept out of the count above unless the five conditions already include them:",
        "",
        control_table(diags),
        "",
        "## 4. Rotation-number structural diagnostics",
        "",
        "The 5–95% width and the central-90% range describe the bulk. "
        "An oscillator is counted as outside the main group when |ρ − median(ρ)| > 0.01, the archived locking range. "
        "A flat majority, used only as a descriptive tag, means at least 90% of oscillators lie inside that same band. "
        "Neither comparison replaces Rule 7, which still uses the full-profile range.",
        "",
        structure_table(diags),
        "",
        (
            f"Among the {len(strict)} strict candidates, {len(near_lock)} have a bulk range at or below 0.01 "
            f"while the full profile exceeds 0.01. "
            f"{count_tag(strict, 'Broad drifting profile')} "
            f"{'carries' if count_tag(strict, 'Broad drifting profile') == 1 else 'carry'} the broad-drifting tag. "
            f"{count_tag(strict, 'Theoretically ambiguous near-boundary case')} "
            f"{'carries' if count_tag(strict, 'Theoretically ambiguous near-boundary case') == 1 else 'carry'} the near-boundary tag."
        ),
        "",
        "## 5. Method C diagnostics",
        "",
        "Constant segments are the archived candidates. Linear segments are not searched again and are not merged. "
        "A linear segment is described as an approximate plateau only when it meets the archived Rule 3 geometry: "
        "length at least 15, relative ω span at least 0.02, and ρ span at most 0.02, with |mean ρ| at least 0.001. "
        "A short piece of a slope is not counted. These comparisons do not change the classifier.",
        "",
        method_table(diags, rules),
        "",
    ]
    for diag in sorted(diags, key=lambda item: (item["delta"], item["K"])):
        lines.append(
            f"- δ={diag['delta']:.2f}, K={diag['K']:.2f}: {diag['situation']} "
            f"Tags: {'; '.join(diag['tags'])}."
        )
    lines.extend(
        [
            "",
            (
                f"Validated nonzero plateaus on the diagnosed Incoherence cells: "
                f"{sum(diag['method']['n_validated'] for diag in diags if diag['state'] == 'Incoherence')}. "
                f"Strict candidates tagged as a possible missed local plateau: {len(missed)}. "
                "A short segment that carries the global range is a failed member-count check, not a missed long plateau."
            ),
            "",
            "## 6. W1/W2/W3 temporal consistency",
            "",
            "D is |ρ(W2)−ρ(W1)| and F is |ρ(W3)−ρ(W2)|, matched on the frequency-sorted oscillators. "
            "The review flag is the archived one: it is set when the three states differ or the validated nonzero-plateau counts differ. "
            "A small D and F, or a clear flag, does not prove convergence. The saved files also contain an order-parameter time series; it was not used here.",
            "",
            temporal_table(diags),
            "",
            (
                f"{stable_windows} of {len(strict)} strict candidates have the same archived state and the same validated-plateau count "
                f"in all three windows. {len(evolving)} strict "
                f"{'candidate carries' if len(evolving) == 1 else 'candidates carry'} the temporally evolving tag."
            ),
            "",
            "## 7. Representative case studies",
            "",
            case_block("A", "δ=0.15, K=0.40", case_a, context["column_015"], rules),
            (
                "The excess over Eq. (7) is inside the Eq. (6) truncation, and the cell sits on the printed Eq. (6) root "
                f"(K−K_H^(6)={case_a['excess6']:+.5f}). "
                f"The profile tracks frequency: correlation {case_a['corr']:.3f}, slope {case_a['slope']:.3f}, "
                f"range {fmt_small(case_a['rho_range'])}. "
                "W3 also has a linear segment that meets the Rule 3 length, relative-span, and ρ-span tests; "
                f"the longest has n={case_a['method']['wide_flat'][0]['length']}, "
                f"relative ω span {case_a['method']['wide_flat'][0]['relative_omega_span']:.3f}, "
                f"ρ span {case_a['method']['wide_flat'][0]['rho_span']:.4f}, "
                f"mean {case_a['method']['wide_flat'][0]['level']:.4f}. "
                "Method C keeps it linear, so it is not a validated plateau. "
                f"W1 validates {len(case_a['validated_windows']['W1'])} nonzero "
                f"{'plateau' if len(case_a['validated_windows']['W1']) == 1 else 'plateaus'}"
                + (
                    ""
                    if not case_a["validated_windows"]["W1"]
                    else (
                        f", the first with n={case_a['validated_windows']['W1'][0]['length']}, "
                        f"relative ω span {case_a['validated_windows']['W1'][0]['relative_omega_span']:.3f}, "
                        f"mean ρ {case_a['validated_windows']['W1'][0]['level']:.4f}"
                    )
                )
                + ", which is why the archived review flag is set. "
                "The shelf is visible on this trajectory and is close to the Rule 3 boundary. "
                "The cell remains a near-boundary strict candidate. It is not identified as a misclassification."
            ),
            "",
            case_block("B", "δ=0.20, K=0.65", case_b, context["column_020"], rules),
            (
                "This is the global-locking control on the non-monotonic column. "
                f"The full-profile range {fmt_small(case_b['rho_range'])} is far below 0.01, so Rule 7 assigns Locking. "
                "Method C is not the source of that label. The same pattern was already reported for other Locking cells: "
                "a nearly flat nonzero profile can be fit entirely by linear segments when the noise estimate is tiny."
            ),
            "",
            case_block("C", "δ=0.20, K=0.70", case_c, context["column_020"], rules),
            (
                f"{case_c['n_within']} of 1000 oscillators are within 0.01 of the median, and the central-90% range is "
                f"{fmt_small(case_c['central90_range'])}. "
                f"The full range is {fmt_small(case_c['rho_range'])}, which is what keeps Rule 7 from firing. "
                "The segment record attributes that range to a short piece rather than to a long plateau that failed a span test. "
                "Method C did not validate a nonzero plateau, and the short piece is below the archived membership minimum, "
                "so this is not evidence that a long plateau was overlooked."
            ),
            "",
            case_block("D", "δ=0.20, K=0.75", case_d, context["column_020"], rules),
            (
                f"Relative to K=0.70, the full range grows from {fmt_small(case_c['rho_range'])} to {fmt_small(case_d['rho_range'])} "
                f"and the central-90% range grows from {fmt_small(case_c['central90_range'])} to {fmt_small(case_d['central90_range'])}. "
                f"Oscillators outside the median by more than 0.01 increase from {case_c['n_deviate']} to {case_d['n_deviate']}. "
                f"{case_d['n_within']} oscillators remain within 0.01 of the median. "
                f"The outside points begin at ω={case_d['deviate_omega_min']:.4f} and include the entire high-frequency 5%. "
                "Method C still returns no constant segment: the flat majority is split into linear pieces at a sigma_hat near 10⁻⁹, "
                "the same partition seen on the Locking control. The new structure is the high-frequency tail, not a missed interior plateau. "
                f"Validated nonzero plateaus remain {case_d['method']['n_validated']}. "
                "The three windows keep the same state, so the tail is not a one-window fluctuation."
            ),
            "",
        ]
    )
    if context["case_e"]:
        for index, diag in enumerate(context["case_e"], start=1):
            column = column_runs(context["records"], diag["delta"])
            lines.append(
                case_block(
                    f"E{index}",
                    f"additional strict candidate δ={diag['delta']:.2f}, K={diag['K']:.2f}",
                    diag,
                    column,
                    rules,
                )
            )
        lines.append(
            "These additional cells were chosen from the strict set after Cases A, C, and D were reserved. "
            "The sort prefers a larger excess over Eq. (7) and then a different diagnostic tag. "
            "They are included because their excess or their segment record differs from the reserved cases, "
            "not because they confirm one explanation."
        )
    else:
        lines.append(
            "### Case E: no further strict candidate",
            "",
            "Every strict candidate is already one of Case A, the δ=0.20 pre-locking cell included with the controls, "
            "or Cases C and D. No extra cell is available to add.",
            "",
        )
    lines.extend(
        [
            "",
            "## 8. Interpretation and limitations",
            "",
            (
                f"There are {len(strict)} strict Incoherence candidates. They lie at {region_text}. "
                f"{len(near_lock)} of them have a bulk rotation number inside the archived locking range while the full range exceeds 0.01. "
                f"Possible missed local plateaus among the strict candidates: {len(missed)}. "
                f"The δ=0.20 Locking-to-Incoherence return is the structural change summarized above: "
                f"{return_sentence} "
                f"Window labels agree on {stable_windows} of {len(strict)} strict candidates; "
                f"the temporally evolving tag is present on {len(evolving)}. "
                "Agreement supports treating the W3 label as stable across these three windows. It does not prove convergence."
            ),
            "",
            "The operational Incoherence label is the residual class after Death, Locking, Partial Death, and Partial Locking. "
            "The theoretical incoherent state is the stationary uniform density, which Eq. (7) and Eq. (6) say is linearly unstable once K passes the Hopf value, "
            "provided that value still lies below 1−δ. Those are different statements. "
            "On this grid the operational label does not always mean that the theoretical incoherent state is the object being observed: "
            "cells with K>K_H^(7) are past the printed linear-instability threshold, and the near-global-locking cells are finite-time profiles "
            "whose bulk frequency is already common. "
            "That fact does not by itself convict the classifier. Rule 7 uses the full range, the short deviant piece fails the membership rule, "
            "and the near-boundary cell at δ=0.15 is inside the truncation gap of Eq. (7). "
            "No strict candidate is declared a misclassification in this report.",
            "",
            "Points that still need more than the three rotation-number windows:",
            "",
            needs_new_data(strict, diags),
            "",
            "Limitations that this run does not remove: the O(δ⁷) remainder, the 0.05 coupling grid, N=1000 against the continuum, "
            "one shared initial phase, the missing self-coupling term already noted in the Eq. (7) validation (a factor (N−1)/N), "
            "and the absence of a finite-N eigenvalue. The death boundary is an existence threshold, not a statement that every initial condition reaches it. "
            "Eq. (5) is the incoherent-branch edge, not a second Hopf curve.",
            "",
            "## 9. Recommended next experiment",
            "",
            recommendation(strict),
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_config()
    rules = require_config(config)
    records, all_runs, meta = load_grid(config, rules)
    for (delta, coupling), state in EXPECTED_W3.items():
        found = lookup(records, delta, coupling)
        if found["state"] != state:
            raise RuntimeError(f"archived expectation failed at delta={delta}, K={coupling}: {found['state']} != {state}")
    band_deltas = sorted({round(row["Delta"], 5) for row in records if 0.0 < row["Delta"] <= MAIN_DELTA_MAX + 1e-12})
    cache = existence_cache(band_deltas)
    screened = []
    for row in records:
        screened.append({**row, **screen_record(row, cache)})
    strict_rows = [row for row in screened if row["strict"]]
    engines = {"pilot": meta["pilot"], "cost_audit": meta["cost_audit"]}
    wanted = {(round(row["Delta"], 5), round(row["K"], 5)): row for row in strict_rows}
    for delta, coupling, _role in CONTROLS:
        row = lookup(records, delta, coupling)
        screen = screen_record(row, cache)
        wanted[(round(row["Delta"], 5), round(row["K"], 5))] = {**row, **screen}
    diags = []
    for row in wanted.values():
        named = match_named(row["Delta"], row["K"])
        if named is not None:
            role = named[2]
        elif row["strict"] and row["near_boundary"]:
            role = "strict candidate, near-boundary"
        elif row["strict"]:
            role = "strict candidate"
        else:
            role = "diagnostic control"
        print(f"diagnose delta={row['Delta']:.3f} K={row['K']:.2f} {row['state']}", flush=True)
        diags.append(diagnose_one(row, row, config, rules, engines, role))
    strict = [diag for diag in diags if diag["strict"]]
    # Controls that failed the screen must not be appended here. `strict` is already the five-condition set.
    counts: dict[str, int] = {}
    for row in records:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    k_values = np.array(sorted({row["K"] for row in records}))
    k_steps = np.unique(np.round(np.diff(k_values), 10))
    if k_steps.size != 1:
        raise RuntimeError(f"K grid is not uniform on the loaded slice: {k_steps}")
    band = [row for row in screened if 0.0 < row["Delta"] <= MAIN_DELTA_MAX + 1e-12 and row["state"] == "Incoherence"]
    funnel = {
        "band_incoherence": len(band),
        "above": sum(row["above_hopf"] for row in band),
        "above_and_branch": sum(row["above_hopf"] and row["below_branch"] for row in band),
        "strict": sum(row["strict"] for row in band),
    }
    if funnel["strict"] != len(strict):
        raise RuntimeError("strict count does not match the diagnosed strict set")
    context = {
        "rules": rules,
        "strict": strict,
        "diags": diags,
        "cache": cache,
        "counts": counts,
        "records": records,
        "provenance": provenance_paragraph(all_runs),
        "n_loaded": meta["n_loaded"],
        "method_calls": meta["method_calls"],
        "funnel": funnel,
        "k_step": float(k_steps[0]),
        "column_015": column_runs(records, 0.15),
        "column_020": column_runs(records, 0.20),
        "case_e": choose_case_e(strict),
    }
    temporal_points = choose_temporal_panels(diags)
    panel_d_choice = choose_panel_d(diags)
    draw(records, diags, rules, temporal_points, panel_d_choice)
    write_report(context)
    print("STRICT", [(round(diag["delta"], 3), round(diag["K"], 2), diag["near_boundary"], diag["tags"]) for diag in strict])
    print("FIGURE", FIGURE_PATH)
    print("REPORT", REPORT_PATH)


if __name__ == "__main__":
    main()
