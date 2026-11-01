#!/usr/bin/env python3
"""Penalty and minimum-length diagnostics for the existing Method C.

Method C is the constant-or-linear dynamic program in
``data_examine/experiments/plateau_detection_pilot/pilot.py``
(``segment_profile`` and ``estimate_sigma``). This script does not
reimplement that recursion, does not integrate the Winfree ODE, and does
not scan the Stage 1 grid.

The published Method C grid is tau in {0.5, 1, 2} with minimum segment
length 4. The sweep below keeps that objective and extends tau by powers
of two around the published baseline tau = 1.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-stage2b-methodc")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = OUT_DIR / "figures"
METHOD_C_DIR = Path(
    "/Users/mohan/Desktop/SRS/data_examine/experiments/plateau_detection_pilot"
)
METHOD_C_FILE = METHOD_C_DIR / "pilot.py"

# Task note spelling is tried first. The second C23 id is the completed
# Stage 1 run in stage1_analysis/pilot_exploration/representative_cases.csv.
RUN_CANDIDATES = {
    "C15": ["ba238fc07fe835f9a65ee9de85e4d601"],
    "C23": [
        "69ef3196fff7d8f950eb2eead86c537",
        "69ef3196fff7d8f950eb2eedad86c537",
    ],
}
EXPECTED = {
    "C15": {"K": 1.40, "p": 0.40, "Delta": 0.20},
    "C23": {"K": 2.75, "p": 0.20, "Delta": 0.20},
}
WINDOWS = {
    "W1": (8000.0, 10000.0),
    "W2": (10000.0, 12000.0),
    "W3": (12000.0, 14000.0),
}
DIAGNOSTIC_OMEGA = (0.93, 1.04)
# Display bins for short constant segments. They are not a locking-size rule.
SHORT_BINS = (10, 20, 30)
# Reporting label only: a window pair whose rotation numbers differ by at
# most this much, in the max-abs sense, is called profile-close.
PROFILE_CLOSE_MAX_ABS = 1e-3
SEGMENT_AGREE_MISMATCH = 0.02
PENALTY_MULTIPLIERS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
FIGURE1_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
WINDOW_MULTIPLIERS = (0.25, 1.0, 4.0)
EXTRA_MIN_LENGTHS = (5, 10, 20, 30)
COLOR_CONSTANT = "#0072B2"
COLOR_LINEAR = "#D55E00"
COLOR_SHORT = "#E69F00"
COLOR_VERY_SHORT = "#CC3311"


def load_method_c():
    """Import Method C without executing the rho-clustering classifier.

    ``pilot.py`` imports ``find_rho_peaks`` at module load. The constant-linear
    recursion does not call it. A stub satisfies that import.
    """
    if not METHOD_C_FILE.is_file():
        raise FileNotFoundError(f"Method C file is missing: {METHOD_C_FILE}")
    if "observe_plateaus" not in sys.modules:
        stub = types.ModuleType("observe_plateaus")

        def find_rho_peaks(*_args, **_kwargs):
            raise RuntimeError("this diagnostic does not call find_rho_peaks")

        stub.find_rho_peaks = find_rho_peaks
        sys.modules["observe_plateaus"] = stub
    sys.path.insert(0, str(METHOD_C_DIR))
    return importlib.import_module("pilot")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_cases(load_profiles) -> tuple[list[dict], list[dict]]:
    from stage1.query import DEFAULT_OUTPUT_DIR

    loaded = []
    notes = []
    for case_id, candidates in RUN_CANDIDATES.items():
        record = None
        tried = []
        for run_id in candidates:
            tried.append(run_id)
            try:
                record = load_profiles([run_id], output_dir=DEFAULT_OUTPUT_DIR)[0]
            except FileNotFoundError:
                continue
            break
        if record is None:
            raise FileNotFoundError(f"{case_id} run_id not found. Tried {tried}")
        params = record["parameters"]
        expected = EXPECTED[case_id]
        for key, value in expected.items():
            if abs(float(params[key]) - value) > 1e-9:
                raise RuntimeError(
                    f"{case_id} {record['run_id']} has {key}={params[key]}, expected {value}"
                )
        if record["run_id"] != candidates[0]:
            notes.append(
                {
                    "case_id": case_id,
                    "requested_run_id": candidates[0],
                    "resolved_run_id": record["run_id"],
                    "reason": "requested run_id is not in stage1_runs; used the completed representative-case run",
                }
            )
        record["case_id"] = case_id
        loaded.append(record)
    return loaded, notes


def window_row(starts: np.ndarray, ends: np.ndarray, bounds: tuple[float, float]) -> int:
    hits = [
        i
        for i, (start, end) in enumerate(zip(starts, ends))
        if abs(float(start) - bounds[0]) < 1e-6 and abs(float(end) - bounds[1]) < 1e-6
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one window {bounds}, found {hits}")
    return hits[0]


def prepare_case(record: dict) -> dict:
    omega = np.asarray(record["omega"], dtype=float)
    rotation = np.asarray(record["rotation_numbers"], dtype=float)
    order = np.arange(omega.size)
    if np.any(np.diff(omega) < 0):
        order = np.argsort(omega, kind="mergesort")
        omega = omega[order]
        rotation = rotation[:, order]
    if np.any(np.diff(omega) <= 0):
        raise RuntimeError(f"{record['case_id']} omega is not strictly increasing")
    series = {}
    for name, bounds in WINDOWS.items():
        row = window_row(record["window_start"], record["window_end"], bounds)
        series[name] = {
            "omega": omega,
            "rho": rotation[row].copy(),
            "window_start": bounds[0],
            "window_end": bounds[1],
        }
    return {
        "case_id": record["case_id"],
        "run_id": record["run_id"],
        "K": float(record["parameters"]["K"]),
        "p": float(record["parameters"]["p"]),
        "Delta": float(record["parameters"]["Delta"]),
        "N": int(record["parameters"]["N"]),
        "trial_id": int(record["realization"]["trial_id"]),
        "frequency_sorted": bool(np.any(order != np.arange(order.size))),
        "series": series,
    }


def penalties(n: int, sigma2: float, tau: float) -> tuple[float, float, float]:
    unit = float(sigma2 * math.log(n))
    return unit, tau * unit * 2.0, tau * unit * 3.0


def block_rss(method_c, omega: np.ndarray, y: np.ndarray, start: int, end: int) -> tuple[float, float]:
    pref = method_c._prefix_sums(omega, y)
    rss0, rss1 = method_c._block_rss(pref, np.array([start], dtype=int), end)
    return float(rss0[0]), float(rss1[0])


def ols_line(omega: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(omega, dtype=float)
    yy = np.asarray(y, dtype=float)
    x_center = x - x.mean()
    y_center = yy - yy.mean()
    ssx = float(np.dot(x_center, x_center))
    if ssx <= 1e-18:
        return float("nan"), float(yy.mean())
    slope = float(np.dot(x_center, y_center) / ssx)
    intercept = float(yy.mean() - slope * x.mean())
    return slope, intercept


def fit_profile(method_c, omega, y, tau: float, min_len: int, sigma: float) -> dict:
    sigma2 = float(sigma) ** 2
    t0 = time.perf_counter()
    segments, fitted, objective = method_c.segment_profile(omega, y, sigma2, tau, min_len)
    elapsed = time.perf_counter() - t0
    unit, pen0, pen1 = penalties(len(y), sigma2, tau)
    rss = 0.0
    penalty = 0.0
    rows = []
    for index, (start, end, mode) in enumerate(segments):
        rss0, rss1 = block_rss(method_c, omega, y, start, end)
        chosen = rss0 if mode == 0 else rss1
        rss += chosen
        penalty += pen0 if mode == 0 else pen1
        if mode == 0:
            level = float(np.mean(y[start:end]))
            slope = math.nan
            intercept = math.nan
        else:
            slope, intercept = ols_line(omega[start:end], y[start:end])
            level = math.nan
        rows.append(
            {
                "segment_index": index,
                "segment_type": "constant" if mode == 0 else "linear",
                "mode": int(mode),
                "index_start": int(start),
                "index_end": int(end),
                "omega_start": float(omega[start]),
                "omega_end": float(omega[end - 1]),
                "oscillator_count": int(end - start),
                "fitted_constant": level,
                "fitted_slope": slope,
                "fitted_intercept": intercept,
                "fit_residual_rss": chosen,
            }
        )
    draw_rss = float(np.sum((y - fitted) ** 2))
    if abs((rss + penalty) - objective) > 1e-6 * max(1.0, abs(objective)):
        raise RuntimeError("segment RSS plus penalty does not match segment_profile objective")
    if abs(draw_rss - rss) > 1e-5 * max(1.0, rss):
        raise RuntimeError(f"drawn fit RSS {draw_rss} differs from DP RSS {rss}")
    return {
        "tau": float(tau),
        "min_len": int(min_len),
        "sigma": float(sigma),
        "penalty_unit": unit,
        "penalty_constant": pen0,
        "penalty_linear": pen1,
        "segments": rows,
        "fitted": fitted,
        "objective": float(objective),
        "total_rss": float(rss),
        "rmse": float(math.sqrt(rss / len(y))),
        "seconds": elapsed,
        "omega": omega,
        "rho": y,
    }


def diagnostic_block(omega: np.ndarray, y: np.ndarray, fit: dict | None) -> dict:
    lo, hi = DIAGNOSTIC_OMEGA
    mask = (omega >= lo) & (omega <= hi)
    if not np.any(mask) or not np.all(np.diff(np.flatnonzero(mask)) == 1):
        raise RuntimeError("diagnostic interval is empty or not contiguous in frequency order")
    yy = y[mask]
    xx = omega[mask]
    slope, _intercept = ols_line(xx, yy)
    rss0 = float(np.sum((yy - yy.mean()) ** 2))
    if np.isfinite(slope):
        line = yy.mean() + slope * (xx - xx.mean())
        rss1 = float(np.sum((yy - line) ** 2))
    else:
        rss1 = rss0
    out = {
        "diagnostic_omega_lo": lo,
        "diagnostic_omega_hi": hi,
        "diagnostic_n": int(yy.size),
        "diagnostic_mean_rho": float(yy.mean()),
        "diagnostic_std_rho": float(yy.std(ddof=1)),
        "diagnostic_range_rho": float(yy.max() - yy.min()),
        "diagnostic_ols_slope": slope,
        "diagnostic_rss_constant": rss0,
        "diagnostic_rss_linear": rss1,
    }
    if fit is None:
        return out
    labels = np.empty(y.size, dtype=int)
    for segment in fit["segments"]:
        labels[segment["index_start"]:segment["index_end"]] = segment["mode"]
    overlapping = [
        segment
        for segment in fit["segments"]
        if segment["index_end"] > int(np.flatnonzero(mask)[0])
        and segment["index_start"] < int(np.flatnonzero(mask)[-1]) + 1
    ]
    out.update(
        {
            "diagnostic_constant_fraction": float(np.mean(labels[mask] == 0)),
            "diagnostic_n_overlapping_segments": len(overlapping),
            "diagnostic_overlapping_types": ";".join(segment["segment_type"] for segment in overlapping),
            "diagnostic_covering_omega_ranges": ";".join(
                f"{segment['omega_start']:.6f}:{segment['omega_end']:.6f}" for segment in overlapping
            ),
            "diagnostic_method_c_rss": float(np.sum((y[mask] - fit["fitted"][mask]) ** 2)),
        }
    )
    return out


def constant_segments(fit: dict) -> list[dict]:
    return [segment for segment in fit["segments"] if segment["mode"] == 0]


def linear_segments(fit: dict) -> list[dict]:
    return [segment for segment in fit["segments"] if segment["mode"] == 1]


def count_shorter_than(segments: list[dict], limit: int) -> int:
    return sum(segment["oscillator_count"] < limit for segment in segments)


def fmt(value, spec: str = ".8g") -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return format(number, spec)


def join_nums(values, spec: str = ".8g") -> str:
    return ";".join(fmt(value, spec) for value in values)


def join_ranges(segments: list[dict]) -> str:
    return ";".join(f"{segment['omega_start']:.6f}:{segment['omega_end']:.6f}" for segment in segments)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def type_labels(fit: dict) -> np.ndarray:
    labels = np.empty(fit["rho"].size, dtype=int)
    for segment in fit["segments"]:
        labels[segment["index_start"]:segment["index_end"]] = segment["mode"]
    return labels


def boundary_omega(fit: dict) -> list[float]:
    return [segment["omega_start"] for segment in fit["segments"][1:]]


def median_boundary_shift(left: list[float], right: list[float]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return math.nan
    forward = [min(abs(item - other) for other in right) for item in left]
    backward = [min(abs(item - other) for other in left) for item in right]
    return float(np.median(forward + backward))


def comparison_note(max_abs: float, mismatch: float) -> str:
    profile_close = max_abs <= PROFILE_CLOSE_MAX_ABS
    segments_agree = mismatch <= SEGMENT_AGREE_MISMATCH
    if profile_close and segments_agree:
        return "profile_close_and_segmentation_agrees"
    if profile_close and not segments_agree:
        return "profile_close_but_segmentation_differs"
    return "undecidable_profile_change_or_algorithm_sensitivity"


def base_identity(case: dict, window: str) -> dict:
    series = case["series"][window]
    return {
        "case_id": case["case_id"],
        "run_id": case["run_id"],
        "K": f"{case['K']:.2f}",
        "p": f"{case['p']:.2f}",
        "Delta": f"{case['Delta']:.2f}",
        "N": case["N"],
        "window": window,
        "window_start": f"{series['window_start']:.0f}",
        "window_end": f"{series['window_end']:.0f}",
    }


def summary_from_fit(case: dict, window: str, fit: dict, sigma_mode: str, multiplier: float) -> dict:
    constants = constant_segments(fit)
    linears = linear_segments(fit)
    row = base_identity(case, window)
    diagnostic = diagnostic_block(fit["omega"], fit["rho"], fit)
    for key, value in diagnostic.items():
        if isinstance(value, float):
            diagnostic[key] = fmt(value)
    row.update(diagnostic)
    row.update(
        {
            "tau": fmt(fit["tau"], ".8g"),
            "tau_multiplier": fmt(multiplier, ".8g"),
            "sigma_mode": sigma_mode,
            "sigma_hat": fmt(fit["sigma"], ".8g"),
            "penalty_unit": fmt(fit["penalty_unit"], ".8g"),
            "penalty_constant": fmt(fit["penalty_constant"], ".8g"),
            "penalty_linear": fmt(fit["penalty_linear"], ".8g"),
            "min_segment_length": fit["min_len"],
            "n_segments": len(fit["segments"]),
            "n_constant_segments": len(constants),
            "n_linear_segments": len(linears),
            "constant_coverage_fraction": fmt(
                sum(segment["oscillator_count"] for segment in constants) / case["N"]
            ),
            "constant_oscillator_count": sum(segment["oscillator_count"] for segment in constants),
            "total_rss": fmt(fit["total_rss"], ".8g"),
            "rmse": fmt(fit["rmse"], ".8g"),
            "objective": fmt(fit["objective"], ".8g"),
            "constant_lengths": join_nums(segment["oscillator_count"] for segment in constants),
            "constant_omega_ranges": join_ranges(constants),
            "constant_fitted_rho": join_nums(segment["fitted_constant"] for segment in constants),
            "linear_slopes": join_nums(segment["fitted_slope"] for segment in linears),
            "linear_omega_ranges": join_ranges(linears),
            "n_constant_length_lt_10": count_shorter_than(constants, 10),
            "n_constant_length_lt_20": count_shorter_than(constants, 20),
            "n_constant_length_lt_30": count_shorter_than(constants, 30),
            "shortest_constant_length": (
                min(segment["oscillator_count"] for segment in constants) if constants else ""
            ),
            "short_constant_omega_ranges": join_ranges(
                [segment for segment in constants if segment["oscillator_count"] < 30]
            ),
            "segment_type_sequence": ";".join(segment["segment_type"] for segment in fit["segments"]),
            "boundary_omega": join_nums(boundary_omega(fit), ".6f"),
            "seconds": f"{fit['seconds']:.4f}",
        }
    )
    return row


def segment_rows(case: dict, window: str, fit: dict, sigma_mode: str) -> list[dict]:
    rows = []
    for segment in fit["segments"]:
        row = base_identity(case, window)
        row.update(
            {
                "tau": fmt(fit["tau"], ".8g"),
                "sigma_mode": sigma_mode,
                "sigma_hat": fmt(fit["sigma"], ".8g"),
                "penalty_constant": fmt(fit["penalty_constant"], ".8g"),
                "penalty_linear": fmt(fit["penalty_linear"], ".8g"),
                "min_segment_length": fit["min_len"],
                "segment_index": segment["segment_index"],
                "segment_type": segment["segment_type"],
                "omega_start": f"{segment['omega_start']:.6f}",
                "omega_end": f"{segment['omega_end']:.6f}",
                "index_start": segment["index_start"],
                "index_end": segment["index_end"],
                "oscillator_count": segment["oscillator_count"],
                "fitted_slope": fmt(segment["fitted_slope"], ".8g"),
                "fitted_constant": fmt(segment["fitted_constant"], ".8g"),
                "fitted_intercept": fmt(segment["fitted_intercept"], ".8g"),
                "fit_residual_rss": fmt(segment["fit_residual_rss"], ".8g"),
            }
        )
        rows.append(row)
    return rows


def plot_segments(ax, fit: dict, view: tuple[float, float] | None = None) -> None:
    omega = fit["omega"]
    for segment in fit["segments"]:
        sl = slice(segment["index_start"], segment["index_end"])
        if view is not None and (omega[sl][-1] < view[0] or omega[sl][0] > view[1]):
            continue
        if segment["mode"] == 0:
            ax.plot(omega[sl], fit["fitted"][sl], color=COLOR_CONSTANT, lw=2.0, solid_capstyle="butt")
        else:
            ax.plot(
                omega[sl],
                fit["fitted"][sl],
                color=COLOR_LINEAR,
                lw=2.0,
                ls=(0, (6, 2)),
                solid_capstyle="butt",
            )


def style_ax(ax, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, color="0.9", lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_handles(include_interval: bool = False) -> list[Line2D]:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markersize=4, label="raw rotation number"),
        Line2D([0], [0], color=COLOR_CONSTANT, lw=1.8, label="constant fit"),
        Line2D([0], [0], color=COLOR_LINEAR, lw=2.0, ls=(0, (6, 2)), label="linear fit"),
    ]
    if include_interval:
        handles.extend(
            [
                Line2D([0], [0], color="black", lw=1.4, label="fixed-interval mean"),
                Line2D([0], [0], color="black", lw=1.2, ls=":", label="fixed-interval OLS"),
            ]
        )
    return handles


def figure_penalty(cases: list[dict], fits: dict) -> None:
    fig, axes = plt.subplots(2, len(FIGURE1_MULTIPLIERS), figsize=(14.5, 6.4), sharex="row", sharey="row")
    for row, case in enumerate(cases):
        for col, multiplier in enumerate(FIGURE1_MULTIPLIERS):
            ax = axes[row, col]
            fit = fits[(case["case_id"], "W3", multiplier, case["baseline_min_len"], "per_window")]
            ax.plot(fit["omega"], fit["rho"], ",", color="0.45", alpha=0.8)
            plot_segments(ax, fit)
            if case["case_id"] == "C15":
                ax.axvspan(*DIAGNOSTIC_OMEGA, color="#f4e4b3", zorder=0)
            constants = constant_segments(fit)
            style_ax(
                ax,
                f"tau={fit['tau']:g}\n{len(fit['segments'])} segments, {len(constants)} constant",
            )
            if row == 1:
                ax.set_xlabel("omega")
            if col == 0:
                ax.set_ylabel(
                    f"{case['case_id']}\nK={case['K']:.2f}, p={case['p']:.2f}, Delta={case['Delta']:.2f}\nrotation number",
                    fontsize=8,
                )
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(
        "Figure 1. Method C on W3 [12000, 14000], minimum segment length "
        f"{cases[0]['baseline_min_len']}. Shaded band on C15 is the diagnostic interval "
        "[0.93, 1.04], not a detected boundary.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(FIGURE_DIR / "fig1_penalty_comparison.png", dpi=150)
    plt.close(fig)


def figure_sensitivity(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True)
    metrics = [
        ("n_constant_segments", "Number of constant segments"),
        ("constant_coverage_fraction", "Constant-segment coverage"),
        ("n_segments", "Number of segments"),
        ("total_rss", "Residual sum of squares"),
    ]
    for ax, (key, label) in zip(axes.ravel(), metrics):
        for case_id, marker in (("C15", "o"), ("C23", "s")):
            chosen = [row for row in rows if row["case_id"] == case_id]
            xs = [float(row["tau"]) for row in chosen]
            ys = [float(row[key]) for row in chosen]
            ax.plot(xs, ys, marker=marker, label=case_id)
        ax.set_xscale("log", base=2)
        ax.set_ylabel(label)
        style_ax(ax, "")
        ax.set_title(label, fontsize=10)
    axes[1, 0].set_xlabel("tau")
    axes[1, 1].set_xlabel("tau")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].set_title("Residual sum of squares", fontsize=10)
    fig.suptitle("Figure 2. Penalty sweep on W3, minimum segment length 4.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "fig2_penalty_sensitivity.png", dpi=150)
    plt.close(fig)


def figure_windows(cases: list[dict], fits: dict, baseline_min_len: int) -> None:
    roles = [("weak", 0.25), ("baseline", 1.0), ("strong", 4.0)]
    fig, axes = plt.subplots(6, 3, figsize=(11.5, 14.5), sharex=True)
    row_index = 0
    for case in cases:
        for role, multiplier in roles:
            for col, window in enumerate(("W1", "W2", "W3")):
                ax = axes[row_index, col]
                fit = fits[(case["case_id"], window, multiplier, baseline_min_len, "per_window")]
                ax.plot(fit["omega"], fit["rho"], ",", color="0.45", alpha=0.85)
                plot_segments(ax, fit)
                if case["case_id"] == "C15":
                    ax.axvspan(*DIAGNOSTIC_OMEGA, color="#f4e4b3", zorder=0)
                constants = constant_segments(fit)
                style_ax(
                    ax,
                    f"{window}  {fit['series_bounds']}  tau={fit['tau']:g} ({role})\n"
                    f"sigma={fit['sigma']:.3g}  constant coverage={sum(s['oscillator_count'] for s in constants) / case['N']:.2f}",
                )
                if col == 0:
                    ax.set_ylabel(f"{case['case_id']}\nrotation number")
                if row_index == 5:
                    ax.set_xlabel("omega")
            row_index += 1
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(
        "Figure 3. Same tau and minimum segment length on W1, W2, and W3.\n"
        "sigma_hat is re-estimated on each window, so the numeric penalty can differ. "
        "C15 shading is the diagnostic interval [0.93, 1.04].",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGURE_DIR / "fig3_window_comparison.png", dpi=140)
    plt.close(fig)


def figure_c15(case: dict, fits: dict, baseline_min_len: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharey=True)
    view = (0.90, 1.07)
    for ax, window in zip(axes, ("W1", "W2", "W3")):
        series = case["series"][window]
        omega = series["omega"]
        rho = series["rho"]
        mask = (omega >= DIAGNOSTIC_OMEGA[0]) & (omega <= DIAGNOSTIC_OMEGA[1])
        view_mask = (omega >= view[0]) & (omega <= view[1])
        ax.axvspan(*DIAGNOSTIC_OMEGA, color="#f4e4b3", zorder=0)
        ax.plot(omega[view_mask], rho[view_mask], ",", color="0.35")
        slope, intercept = ols_line(omega[mask], rho[mask])
        ax.hlines(float(rho[mask].mean()), omega[mask][0], omega[mask][-1], color="black", lw=1.3)
        ax.plot(omega[mask], intercept + slope * omega[mask], color="black", lw=1.1, ls=":")
        for multiplier, alpha in ((0.25, 0.45), (1.0, 1.0), (4.0, 0.7)):
            fit = fits[(case["case_id"], window, multiplier, baseline_min_len, "per_window")]
            for segment in fit["segments"]:
                sl = slice(segment["index_start"], segment["index_end"])
                if omega[sl][-1] < view[0] or omega[sl][0] > view[1]:
                    continue
                color = COLOR_CONSTANT if segment["mode"] == 0 else COLOR_LINEAR
                ls = "-" if segment["mode"] == 0 else "--"
                ax.plot(omega[sl], fit["fitted"][sl], color=color, ls=ls, lw=1.3, alpha=alpha)
        block = diagnostic_block(omega, rho, None)
        baseline = fits[(case["case_id"], window, 1.0, baseline_min_len, "per_window")]
        midpoint = float(np.median(omega[mask]))
        covering = next(
            segment
            for segment in baseline["segments"]
            if segment["omega_start"] <= midpoint <= segment["omega_end"]
        )
        if covering["mode"] == 0:
            method_note = f"tau=1 model=constant, level={covering['fitted_constant']:.3f}"
        else:
            method_note = f"tau=1 model=linear, slope={covering['fitted_slope']:.3f}"
        style_ax(
            ax,
            f"{window} [{series['window_start']:.0f}, {series['window_end']:.0f}]\n"
            f"n={block['diagnostic_n']}  mean={block['diagnostic_mean_rho']:.4f}  "
            f"std={block['diagnostic_std_rho']:.4f}\n"
            f"fixed-interval OLS slope={block['diagnostic_ols_slope']:.3f}\n{method_note}",
        )
        ax.set_xlim(*view)
        ax.set_xlabel("omega")
    axes[0].set_ylabel("rotation number")
    fig.legend(handles=legend_handles(include_interval=True), loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(
        "Figure 4. C15, K=1.40, p=0.40, Delta=0.20. View spans omega 0.90 to 1.07.\n"
        "Black mean and dotted line use only oscillators inside [0.93, 1.04]. "
        "Colored curves are Method C fits of all 1000 oscillators at tau=0.25, 1, and 4.\n"
        "The shaded interval is a specified diagnostic region, not a boundary returned by Method C.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.84))
    fig.savefig(FIGURE_DIR / "fig4_c15_diagnostic_region.png", dpi=150)
    plt.close(fig)


def length_color(count: int) -> str:
    if count < 10:
        return COLOR_VERY_SHORT
    if count < 30:
        return COLOR_SHORT
    return COLOR_CONSTANT


def draw_constant_bars(ax, fits_in_order: list[tuple[str, dict]]) -> None:
    omega = fits_in_order[0][1]["omega"]
    for y, (_label, fit) in enumerate(fits_in_order):
        constants = constant_segments(fit)
        if not constants:
            ax.text(
                0.5,
                y,
                "no constant segment",
                va="center",
                ha="center",
                fontsize=7,
                color="0.35",
                transform=ax.get_yaxis_transform(),
            )
            continue
        for segment in constants:
            ax.barh(
                y,
                segment["omega_end"] - segment["omega_start"],
                left=segment["omega_start"],
                height=0.62,
                color=length_color(segment["oscillator_count"]),
                edgecolor="none",
            )
    pad = 0.02 * float(omega[-1] - omega[0])
    ax.set_xlim(float(omega[0]) - pad, float(omega[-1]) + pad)
    ax.set_ylim(-0.7, len(fits_in_order) - 0.3)
    ax.set_yticks(range(len(fits_in_order)))
    ax.set_yticklabels([label for label, _fit in fits_in_order], fontsize=8)
    ax.set_xlabel("omega")
    ax.grid(True, axis="x", color="0.9", lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_c23(case: dict, fits: dict, baseline_min_len: int, min_lengths: list[int]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 10.4))
    penalty_rows = []
    for multiplier in PENALTY_MULTIPLIERS:
        fit = fits[(case["case_id"], "W3", multiplier, baseline_min_len, "per_window")]
        penalty_rows.append((f"tau={fit['tau']:g}", fit))
    draw_constant_bars(axes[0], penalty_rows)
    axes[0].set_title(
        "W3 constant segments versus tau, minimum length 4. Empty rows have no constant segment.",
        fontsize=9,
    )
    window_rows = []
    for multiplier, role in ((0.25, "weak"), (1.0, "baseline"), (4.0, "strong")):
        for window in ("W1", "W2", "W3"):
            fit = fits[(case["case_id"], window, multiplier, baseline_min_len, "per_window")]
            window_rows.append((f"tau={fit['tau']:g} ({role})  {window}", fit))
    draw_constant_bars(axes[1], window_rows)
    axes[1].set_title("Constant segments on W1, W2, and W3 at the same tau. sigma_hat is per window.", fontsize=9)
    length_rows = []
    for min_len in min_lengths:
        fit = fits[(case["case_id"], "W3", 0.25, min_len, "per_window")]
        length_rows.append((f"tau=0.25, min length={min_len}", fit))
    draw_constant_bars(axes[2], length_rows)
    axes[2].set_title(
        "W3 at tau=0.25, where short constant segments are present. "
        "At baseline tau=1, C23 has no constant segment for these minimum lengths.",
        fontsize=9,
    )
    handles = [
        Line2D([0], [0], color=COLOR_VERY_SHORT, lw=6, label="< 10 oscillators"),
        Line2D([0], [0], color=COLOR_SHORT, lw=6, label="10 to 29 oscillators"),
        Line2D([0], [0], color=COLOR_CONSTANT, lw=6, label="30 or more oscillators"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(
        "Figure 5. C23 constant segments only. K=2.75, p=0.20, Delta=0.20.\n"
        "Bar color is segment length. These length bins are display groups, not a physical locking size. "
        "Linear segments are omitted.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(FIGURE_DIR / "fig5_c23_short_segments.png", dpi=150)
    plt.close(fig)


def attach_window_bounds(fits: dict) -> None:
    for fit in fits.values():
        fit["series_bounds"] = f"[{fit['window_start']:.0f}, {fit['window_end']:.0f}]"


def main() -> None:
    sys.path.insert(0, str(REPO))
    from stage1.query import load_profiles

    method_c = load_method_c()
    baseline_tau = float(method_c.PRIMARY_MULTIPLIER)
    baseline_min_len = int(method_c.MIN_SEGMENT)
    if baseline_min_len < 2:
        raise RuntimeError("Method C minimum segment length is below 2")
    records, id_notes = resolve_cases(load_profiles)
    cases = [prepare_case(record) for record in records]
    for case in cases:
        case["baseline_min_len"] = baseline_min_len
        if case["N"] != 1000:
            raise RuntimeError(f"{case['case_id']} has N={case['N']}, expected 1000")

    min_lengths = [baseline_min_len] + [item for item in EXTRA_MIN_LENGTHS if item != baseline_min_len]
    jobs = []
    for case in cases:
        for multiplier in PENALTY_MULTIPLIERS:
            jobs.append((case, "W3", multiplier, baseline_min_len, "per_window"))
        for min_len in min_lengths:
            if min_len == baseline_min_len:
                continue
            # Baseline tau is the Task C grid. tau=0.25 is the setting where
            # C23 still has short constant segments, so the length constraint
            # can change them.
            jobs.append((case, "W3", 1.0, min_len, "per_window"))
            jobs.append((case, "W3", 0.25, min_len, "per_window"))
        for multiplier in WINDOW_MULTIPLIERS:
            for window in ("W1", "W2"):
                jobs.append((case, window, multiplier, baseline_min_len, "per_window"))
                jobs.append((case, window, multiplier, baseline_min_len, "frozen_w3"))

    sigmas = {}
    for case in cases:
        for window, series in case["series"].items():
            sigmas[(case["case_id"], window)] = float(method_c.estimate_sigma(series["rho"]))

    fits = {}
    for case, window, multiplier, min_len, sigma_mode in jobs:
        tau = baseline_tau * multiplier
        series = case["series"][window]
        sigma_window = "W3" if sigma_mode == "frozen_w3" else window
        sigma = sigmas[(case["case_id"], sigma_window)]
        fit = fit_profile(method_c, series["omega"], series["rho"], tau, min_len, sigma)
        fit["window_start"] = series["window_start"]
        fit["window_end"] = series["window_end"]
        fits[(case["case_id"], window, multiplier, min_len, sigma_mode)] = fit

    # W3 window-comparison rows are the penalty-sweep fits. Alias them so
    # figure code can ask for every window at the three selected multipliers.
    for case in cases:
        for multiplier in WINDOW_MULTIPLIERS:
            key = (case["case_id"], "W3", multiplier, baseline_min_len, "per_window")
            if key not in fits:
                raise RuntimeError(f"missing baseline-grid fit {key}")

    attach_window_bounds(fits)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    penalty_rows = []
    min_rows = []
    segment_table = []
    for case in cases:
        for multiplier in PENALTY_MULTIPLIERS:
            fit = fits[(case["case_id"], "W3", multiplier, baseline_min_len, "per_window")]
            penalty_rows.append(summary_from_fit(case, "W3", fit, "per_window", multiplier))
            segment_table.extend(segment_rows(case, "W3", fit, "per_window"))
        for multiplier in (1.0, 0.25):
            for min_len in min_lengths:
                fit = fits[(case["case_id"], "W3", multiplier, min_len, "per_window")]
                min_rows.append(summary_from_fit(case, "W3", fit, "per_window", multiplier))
                if min_len != baseline_min_len:
                    segment_table.extend(segment_rows(case, "W3", fit, "per_window"))

    window_rows = []
    roles = {0.25: "weak", 1.0: "baseline", 4.0: "strong"}
    for case in cases:
        for multiplier in WINDOW_MULTIPLIERS:
            reference = fits[(case["case_id"], "W3", multiplier, baseline_min_len, "per_window")]
            ref_labels = type_labels(reference)
            ref_boundaries = boundary_omega(reference)
            for window in ("W1", "W2", "W3"):
                modes = ("per_window",) if window == "W3" else ("per_window", "frozen_w3")
                for sigma_mode in modes:
                    fit = fits[(case["case_id"], window, multiplier, baseline_min_len, sigma_mode)]
                    row = summary_from_fit(case, window, fit, sigma_mode, multiplier)
                    diff = np.abs(fit["rho"] - reference["rho"])
                    mismatch = float(np.mean(type_labels(fit) != ref_labels))
                    same_window = fits[(case["case_id"], window, multiplier, baseline_min_len, "per_window")]
                    row.update(
                        {
                            "tau_role": roles[multiplier],
                            "reference_window": "W3",
                            "vs_reference_max_abs_rho": fmt(float(diff.max()), ".8g"),
                            "vs_reference_median_abs_rho": fmt(float(np.median(diff)), ".8g"),
                            "vs_reference_type_mismatch_fraction": fmt(mismatch, ".8g"),
                            "vs_reference_boundary_median_abs_shift": fmt(
                                median_boundary_shift(boundary_omega(fit), ref_boundaries), ".8g"
                            ),
                            "comparison_note": (
                                "reference_window"
                                if window == "W3"
                                else comparison_note(float(diff.max()), mismatch)
                            ),
                            "vs_per_window_same_window_type_mismatch": fmt(
                                float(np.mean(type_labels(fit) != type_labels(same_window))), ".8g"
                            ),
                        }
                    )
                    window_rows.append(row)
                    if not (window == "W3" and sigma_mode == "per_window"):
                        segment_table.extend(segment_rows(case, window, fit, sigma_mode))

    penalty_fields = [
        "case_id", "run_id", "K", "p", "Delta", "N", "window", "window_start", "window_end",
        "tau", "tau_multiplier", "sigma_mode", "sigma_hat", "penalty_unit",
        "penalty_constant", "penalty_linear", "min_segment_length",
        "n_segments", "n_constant_segments", "n_linear_segments",
        "constant_oscillator_count", "constant_coverage_fraction",
        "total_rss", "rmse", "objective",
        "constant_lengths", "constant_omega_ranges", "constant_fitted_rho",
        "linear_slopes", "linear_omega_ranges",
        "diagnostic_omega_lo", "diagnostic_omega_hi", "diagnostic_n",
        "diagnostic_mean_rho", "diagnostic_std_rho", "diagnostic_range_rho",
        "diagnostic_ols_slope", "diagnostic_rss_constant", "diagnostic_rss_linear",
        "diagnostic_constant_fraction", "diagnostic_n_overlapping_segments",
        "diagnostic_overlapping_types", "diagnostic_covering_omega_ranges",
        "diagnostic_method_c_rss",
    ]
    min_fields = [
        "case_id", "run_id", "K", "p", "Delta", "N", "window", "window_start", "window_end",
        "tau", "sigma_hat", "penalty_constant", "penalty_linear", "min_segment_length",
        "n_segments", "n_constant_segments", "n_linear_segments",
        "constant_coverage_fraction", "constant_lengths", "constant_omega_ranges",
        "constant_fitted_rho", "n_constant_length_lt_10", "n_constant_length_lt_20",
        "n_constant_length_lt_30", "shortest_constant_length", "total_rss", "objective",
        "diagnostic_constant_fraction", "diagnostic_covering_omega_ranges",
    ]
    window_fields = [
        "case_id", "run_id", "K", "p", "Delta", "window", "window_start", "window_end",
        "tau", "tau_role", "sigma_mode", "sigma_hat", "penalty_constant", "penalty_linear",
        "min_segment_length", "n_segments", "n_constant_segments", "n_linear_segments",
        "constant_coverage_fraction", "constant_lengths", "constant_omega_ranges",
        "constant_fitted_rho", "linear_slopes", "linear_omega_ranges",
        "segment_type_sequence", "boundary_omega", "reference_window",
        "vs_reference_max_abs_rho", "vs_reference_median_abs_rho",
        "vs_reference_type_mismatch_fraction", "vs_reference_boundary_median_abs_shift",
        "comparison_note", "vs_per_window_same_window_type_mismatch",
        "diagnostic_mean_rho", "diagnostic_std_rho", "diagnostic_ols_slope",
        "diagnostic_constant_fraction", "diagnostic_covering_omega_ranges",
        "n_constant_length_lt_30", "short_constant_omega_ranges",
    ]
    segment_fields = [
        "case_id", "run_id", "K", "p", "Delta", "window", "window_start", "window_end",
        "tau", "sigma_mode", "sigma_hat", "penalty_constant", "penalty_linear",
        "min_segment_length", "segment_index", "segment_type",
        "omega_start", "omega_end", "index_start", "index_end", "oscillator_count",
        "fitted_slope", "fitted_constant", "fitted_intercept", "fit_residual_rss",
    ]
    write_csv(OUT_DIR / "penalty_sensitivity.csv", penalty_rows, penalty_fields)
    write_csv(OUT_DIR / "min_segment_sensitivity.csv", min_rows, min_fields)
    write_csv(OUT_DIR / "window_consistency.csv", window_rows, window_fields)
    write_csv(OUT_DIR / "segment_details.csv", segment_table, segment_fields)

    figure_penalty(cases, fits)
    figure_sensitivity(penalty_rows)
    figure_windows(cases, fits, baseline_min_len)
    figure_c15(cases[0], fits, baseline_min_len)
    c23 = next(case for case in cases if case["case_id"] == "C23")
    figure_c23(c23, fits, baseline_min_len, min_lengths)

    total_seconds = sum(fit["seconds"] for fit in fits.values())
    config = {
        "method_c_file": str(METHOD_C_FILE),
        "method_c_sha256": sha256_file(METHOD_C_FILE),
        "reused_functions": ["estimate_sigma", "segment_profile", "_prefix_sums", "_block_rss"],
        "objective": "sum over segments of RSS_m + tau * sigma_hat^2 * ln(n) * (1 + p_m)",
        "p_constant": 1,
        "p_linear": 2,
        "penalty_constant": "tau * sigma_hat^2 * ln(n) * 2",
        "penalty_linear": "tau * sigma_hat^2 * ln(n) * 3",
        "changepoint_and_parameter_penalties": "coupled through one tau; not separate inputs",
        "model_selection": "choose linear only when RSS_linear + penalty_linear < RSS_constant + penalty_constant; a tie keeps the constant",
        "sigma_hat": "1.4826 * MAD(second difference of rho) / sqrt(6), floored at 1e-12",
        "baseline_tau": baseline_tau,
        "baseline_min_segment_length": baseline_min_len,
        "published_tau_grid_in_method_c": [0.5, 1.0, 2.0],
        "tau_multipliers_relative_to_baseline": list(PENALTY_MULTIPLIERS),
        "figure1_tau_multipliers": list(FIGURE1_MULTIPLIERS),
        "window_tau_multipliers": list(WINDOW_MULTIPLIERS),
        "min_segment_lengths": min_lengths,
        "min_segment_length_tau_multipliers": [1.0, 0.25],
        "min_segment_length_note": "tau multiplier 1 is the published baseline. Multiplier 0.25 is included because that is where C23 still has short constant segments.",
        "windows": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "diagnostic_omega_interval": list(DIAGNOSTIC_OMEGA),
        "diagnostic_interval_status": "specified for C15 from a visual observation; not a Method C boundary",
        "short_length_bins_oscillators": list(SHORT_BINS),
        "profile_close_max_abs_rho": PROFILE_CLOSE_MAX_ABS,
        "segmentation_agree_mismatch_fraction": SEGMENT_AGREE_MISMATCH,
        "reporting_cutoffs_are_not_classifier_thresholds": True,
        "information_criterion_named_in_code": None,
        "smoothing_or_extra_threshold_inside_segment_profile": False,
        "size_matched_display_cutoff_applied": False,
        "penalty_split_variant_used": False,
        "ode_rerun": False,
        "cases": [
            {
                "case_id": case["case_id"],
                "run_id": case["run_id"],
                "K": case["K"],
                "p": case["p"],
                "Delta": case["Delta"],
                "N": case["N"],
                "trial_id": case["trial_id"],
                "frequency_sorted_for_segmentation": case["frequency_sorted"],
                "sigma_hat": {window: sigmas[(case["case_id"], window)] for window in WINDOWS},
            }
            for case in cases
        ],
        "run_id_resolution": id_notes,
        "n_segmentations": len(fits),
        "segmentation_cpu_seconds": total_seconds,
        "data_source": "stage1_runs via stage1.query.load_profiles",
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    facts = {
        "penalty_rows": [
            {
                "case_id": row["case_id"],
                "tau": row["tau"],
                "n_segments": row["n_segments"],
                "n_constant_segments": row["n_constant_segments"],
                "constant_coverage_fraction": row["constant_coverage_fraction"],
                "total_rss": row["total_rss"],
                "constant_lengths": row["constant_lengths"],
                "constant_omega_ranges": row["constant_omega_ranges"],
                "constant_fitted_rho": row["constant_fitted_rho"],
                "diagnostic_constant_fraction": row["diagnostic_constant_fraction"],
                "diagnostic_overlapping_types": row["diagnostic_overlapping_types"],
                "diagnostic_covering_omega_ranges": row["diagnostic_covering_omega_ranges"],
                "diagnostic_mean_rho": row["diagnostic_mean_rho"],
                "diagnostic_std_rho": row["diagnostic_std_rho"],
                "diagnostic_ols_slope": row["diagnostic_ols_slope"],
                "diagnostic_rss_constant": row["diagnostic_rss_constant"],
                "diagnostic_rss_linear": row["diagnostic_rss_linear"],
                "diagnostic_method_c_rss": row["diagnostic_method_c_rss"],
                "n_constant_length_lt_30": row["n_constant_length_lt_30"],
                "shortest_constant_length": row["shortest_constant_length"],
            }
            for row in penalty_rows
        ],
        "min_rows": [
            {
                "case_id": row["case_id"],
                "min_segment_length": row["min_segment_length"],
                "n_segments": row["n_segments"],
                "n_constant_segments": row["n_constant_segments"],
                "constant_coverage_fraction": row["constant_coverage_fraction"],
                "constant_omega_ranges": row["constant_omega_ranges"],
                "shortest_constant_length": row["shortest_constant_length"],
                "diagnostic_constant_fraction": row["diagnostic_constant_fraction"],
                "diagnostic_covering_omega_ranges": row["diagnostic_covering_omega_ranges"],
            }
            for row in min_rows
        ],
        "window_rows": [
            {
                "case_id": row["case_id"],
                "window": row["window"],
                "tau": row["tau"],
                "sigma_mode": row["sigma_mode"],
                "sigma_hat": row["sigma_hat"],
                "n_segments": row["n_segments"],
                "n_constant_segments": row["n_constant_segments"],
                "constant_coverage_fraction": row["constant_coverage_fraction"],
                "constant_omega_ranges": row["constant_omega_ranges"],
                "constant_fitted_rho": row["constant_fitted_rho"],
                "comparison_note": row["comparison_note"],
                "vs_reference_max_abs_rho": row["vs_reference_max_abs_rho"],
                "vs_reference_median_abs_rho": row["vs_reference_median_abs_rho"],
                "vs_reference_type_mismatch_fraction": row["vs_reference_type_mismatch_fraction"],
                "vs_reference_boundary_median_abs_shift": row["vs_reference_boundary_median_abs_shift"],
                "vs_per_window_same_window_type_mismatch": row["vs_per_window_same_window_type_mismatch"],
                "diagnostic_mean_rho": row["diagnostic_mean_rho"],
                "diagnostic_std_rho": row["diagnostic_std_rho"],
                "diagnostic_constant_fraction": row["diagnostic_constant_fraction"],
                "diagnostic_covering_omega_ranges": row["diagnostic_covering_omega_ranges"],
                "n_constant_length_lt_30": row["n_constant_length_lt_30"],
                "short_constant_omega_ranges": row["short_constant_omega_ranges"],
            }
            for row in window_rows
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print(f"segmentations={len(fits)} cpu_s={total_seconds:.3f}")
    print(f"run_id_notes={id_notes}")
    for case in cases:
        print(case["case_id"], case["run_id"], {w: sigmas[(case["case_id"], w)] for w in WINDOWS})


if __name__ == "__main__":
    main()
