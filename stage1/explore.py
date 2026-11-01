"""Descriptive exploration of saved Stage 1 rotation profiles.

Reads completed runs through the Stage 1 query API. Does not integrate the
ODE, does not modify the run index, and does not assign state labels.
Near-zero tolerances and local-slope cutoffs are exploratory parameters.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage1.config import load_config
from stage1.grid import canonical_label, quantize
from stage1.model import PROJECT_ROOT
from stage1.query import fetch_runs_by_id, load_profiles
from stage1.scan import build_run_specs
from stage1.stats import near_zero_fraction, sort_by_frequency

WINDOW_KEYS = ("w1", "w2", "w3")
WINDOW_BOUNDS = {
    "w1": (8000.0, 10000.0),
    "w2": (10000.0, 12000.0),
    "w3": (12000.0, 14000.0),
}
WINDOW_LABELS = {
    "w1": "8000-10000",
    "w2": "10000-12000",
    "w3": "12000-14000",
}
EPSILONS = (0.001, 0.005, 0.01, 0.02)
EPS_TAGS = {0.001: "0p001", 0.005: "0p005", 0.01: "0p01", 0.02: "0p02"}
SLOPE_WINDOWS = (11, 31)
SLOPE_CUTOFFS = (0.05, 0.1, 0.25)
CUTOFF_TAGS = {0.05: "0p05", 0.1: "0p1", 0.25: "0p25"}
DIFF_PAIRS = (("D12", "w1", "w2"), ("D23", "w2", "w3"), ("D13", "w1", "w3"))
DIFF_TOLERANCES = (0.001, 0.01, 0.05)
DIFF_TAGS = {0.001: "0p001", 0.01: "0p01", 0.05: "0p05"}
RHO_QUANTILES = ((0.05, "q05"), (0.25, "q25"), (0.5, "q50"), (0.75, "q75"), (0.95, "q95"))
DIVERSITY_FEATURES = (
    "zero_frac_eps0p001_w3",
    "zero_frac_eps0p01_w3",
    "zero_frac_eps_sensitivity_w3",
    "rho_range_w3",
    "rho_std_w3",
    "rho_minus_omega_mean_w3",
    "slope11_abs_q50_w3",
    "slope31_abs_q50_w3",
    "slope_scale_gap_w3",
    "slope31_std_w3",
    "D12_p95",
    "D13_median",
    "D13_p95",
)
REVIEW_METRICS = (
    ("high_window_variation", "D13_p95"),
    ("slope_scale_sensitive", "slope_scale_gap_w3"),
    ("neighbor_K_feature_jump", "neighbor_feature_distance"),
    ("high_rotation_dispersion", "rho_std_w3"),
    ("near_zero_tolerance_sensitive", "zero_frac_eps_sensitivity_w3"),
)
K_DISTANCE_WEIGHT = 0.5
CASES_PER_STRATUM = 2
N_SUPPLEMENTS = 6
REVIEW_TOP_N = 6
SUPPLEMENT_METRICS = (
    ("high_window_variation", "D13_p95"),
    ("early_window_variation", "D12_p95"),
    ("slope_scale_sensitive", "slope_scale_gap_w3"),
    ("neighbor_K_feature_jump", "neighbor_feature_distance"),
    ("high_rotation_dispersion", "rho_std_w3"),
    ("near_zero_tolerance_sensitive", "zero_frac_eps_sensitivity_w3"),
)
PROFILE_COLORS = {"w1": "#0072B2", "w2": "#E69F00", "w3": "#009E73"}
HEATMAP_FEATURES = (
    ("zero_frac_eps0p01_w3", "|rho| < 0.01 fraction"),
    ("rho_range_w3", "rotation-number range"),
    ("slope31_abs_q50_w3", "median |slope|, 31 oscillators"),
    ("D13_p95", "|w3 - w1| 95th percentile"),
)


class InventoryError(RuntimeError):
    """Raised when the saved pilot runs do not match the planned grid."""


def rolling_slopes(omega: np.ndarray, rotation_number: np.ndarray, window: int) -> np.ndarray:
    """Ordinary-least-squares slopes on sliding windows of a frequency-sorted profile.

    ``omega`` and ``rotation_number`` stay in their input order. Sorting is
    applied to a copy, with the same permutation on both arrays.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("slope window must be an odd integer of at least 3")
    ordered = sort_by_frequency(omega, rotation_number)
    x = np.asarray(ordered["omega"], dtype=np.float64)
    y = np.asarray(ordered["arrays"][0], dtype=np.float64)
    if x.size < window:
        return np.empty(0, dtype=np.float64)
    count = float(window)
    sum_x = _rolling_sum(x, window)
    sum_y = _rolling_sum(y, window)
    sum_xy = _rolling_sum(x * y, window)
    sum_x2 = _rolling_sum(x * x, window)
    numer = count * sum_xy - sum_x * sum_y
    denom = count * sum_x2 - sum_x * sum_x
    slopes = np.full(numer.shape, np.nan, dtype=np.float64)
    np.divide(numer, denom, out=slopes, where=denom > 0.0)
    return slopes


def slope_summaries(omega: np.ndarray, rotation_number: np.ndarray, window: int) -> dict:
    slopes = rolling_slopes(omega, rotation_number, window)
    finite = slopes[np.isfinite(slopes)] if slopes.size else slopes
    summary = {
        "count": int(finite.size),
        "std": np.nan,
        "abs_q10": np.nan,
        "abs_q50": np.nan,
        "abs_q90": np.nan,
    }
    for cutoff in SLOPE_CUTOFFS:
        summary[f"frac_abs_lt_{CUTOFF_TAGS[cutoff]}"] = np.nan
    if finite.size == 0:
        return summary
    absolute = np.abs(finite)
    summary["std"] = float(np.std(finite))
    summary["abs_q10"] = float(np.quantile(absolute, 0.10))
    summary["abs_q50"] = float(np.quantile(absolute, 0.50))
    summary["abs_q90"] = float(np.quantile(absolute, 0.90))
    for cutoff in SLOPE_CUTOFFS:
        summary[f"frac_abs_lt_{CUTOFF_TAGS[cutoff]}"] = float(np.mean(absolute < float(cutoff)))
    return summary


def assign_window_profiles(record: dict) -> dict[str, np.ndarray]:
    """Map rotation-number rows to the three pilot windows by their bounds."""
    starts = np.asarray(record["window_start"], dtype=np.float64)
    ends = np.asarray(record["window_end"], dtype=np.float64)
    rotation = np.asarray(record["rotation_numbers"], dtype=np.float64)
    if rotation.ndim != 2 or rotation.shape[0] != starts.size or starts.size != ends.size:
        raise InventoryError(f"{record['run_id']} has misaligned observation windows")
    assigned: dict[str, np.ndarray] = {}
    used: set[int] = set()
    for key, (start, end) in WINDOW_BOUNDS.items():
        matches = np.flatnonzero(np.isclose(starts, start) & np.isclose(ends, end))
        if matches.size != 1:
            raise InventoryError(f"{record['run_id']} does not have exactly one {key} window")
        index = int(matches[0])
        assigned[key] = np.asarray(rotation[index], dtype=np.float64)
        used.add(index)
    if len(used) != rotation.shape[0]:
        raise InventoryError(f"{record['run_id']} has an unexpected observation window")
    return assigned


def feature_row(record: dict) -> dict:
    """Descriptive features for one saved run. Undefined values are NaN."""
    parameters = record["parameters"]
    realization = record["realization"]
    omega = np.asarray(record["omega"], dtype=np.float64)
    n = int(parameters["N"])
    if omega.shape != (n,):
        raise InventoryError(f"{record['run_id']} omega length does not match N")
    if record["oscillator_index"].shape != (n,) or np.any(
        record["oscillator_index"] != np.arange(n, dtype=np.int32)
    ):
        raise InventoryError(f"{record['run_id']} oscillator_index is not 0..N-1")
    windows = assign_window_profiles(record)
    row: dict = {
        "run_id": record["run_id"],
        "K": float(parameters["K"]),
        "p": float(parameters["p"]),
        "Delta": float(parameters["Delta"]),
        "N": n,
        "trial_id": int(realization["trial_id"]),
        "network_seed": int(realization["network_seed"]),
        "initial_phase_seed": int(realization["initial_phase_seed"]),
        "frequency_seed": realization["frequency_seed"],
        "network_realization_id": int(realization["network_realization_id"]),
        "initial_condition_id": int(realization["initial_condition_id"]),
        "paired_group_id": realization["paired_group_id"],
    }
    for key, rho in windows.items():
        if rho.shape != (n,):
            raise InventoryError(f"{record['run_id']} {key} profile length does not match N")
        _add_profile_features(row, key, omega, rho)
    _add_window_differences(row, windows)
    gap_11 = row["slope11_abs_q50_w3"]
    gap_31 = row["slope31_abs_q50_w3"]
    row["slope_scale_gap_w3"] = (
        np.nan if not np.isfinite(gap_11) or not np.isfinite(gap_31) else float(abs(gap_11 - gap_31))
    )
    low = row["zero_frac_eps0p001_w3"]
    high = row["zero_frac_eps0p02_w3"]
    row["zero_frac_eps_sensitivity_w3"] = (
        np.nan if not np.isfinite(low) or not np.isfinite(high) else float(high - low)
    )
    return row


def verify_pilot_index(config_path: Path, output_dir: Path | None = None) -> list:
    """Return planned pilot specs whose index rows are completed and unique."""
    config = load_config(config_path)
    if output_dir is not None:
        from dataclasses import replace

        from stage1.config import validate_config

        config = validate_config(replace(config, output_dir=Path(output_dir)))
    specs = build_run_specs(config)
    expected = (
        len(config.resolved_k())
        * len(config.resolved_p())
        * len(config.resolved_delta())
        * config.n_trials
    )
    if len(specs) != expected:
        raise InventoryError(f"planned {len(specs)} runs, expected {expected}")
    ids = [spec.run_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise InventoryError("planned pilot run_id values are not unique")
    rows = fetch_runs_by_id(ids, output_dir=config.output_dir)
    missing = [spec.run_id for spec in specs if spec.run_id not in rows]
    if missing:
        raise InventoryError(f"{len(missing)} planned runs are absent from the index")
    bad_status = [row["run_id"] for row in rows.values() if row["status"] != "completed"]
    if bad_status:
        raise InventoryError(f"{len(bad_status)} planned runs are not completed")
    for spec in specs:
        row = rows[spec.run_id]
        if quantize(row["K"]) != spec.k_units or quantize(row["p"]) != spec.p_units:
            raise InventoryError(f"{spec.run_id} index parameters do not match the plan")
        if quantize(row["Delta"]) != spec.delta_units or int(row["N"]) != spec.N:
            raise InventoryError(f"{spec.run_id} index parameters do not match the plan")
        if int(row["trial_id"]) != spec.trial_id:
            raise InventoryError(f"{spec.run_id} trial_id does not match the plan")
    return specs


def collect_feature_rows(specs: list, output_dir: Path) -> list[dict]:
    rows = []
    ids = [spec.run_id for spec in specs]
    by_id = {spec.run_id: spec for spec in specs}
    for start in range(0, len(ids), 40):
        records = load_profiles(ids[start : start + 40], output_dir=output_dir)
        for record in records:
            spec = by_id[record["run_id"]]
            _check_record_matches_spec(record, spec)
            rows.append(feature_row(record))
        print(f"features {min(start + 40, len(ids))}/{len(ids)}", flush=True)
    if len(rows) != len(ids) or len({row["run_id"] for row in rows}) != len(rows):
        raise InventoryError("feature table does not contain each planned run once")
    _add_neighbor_distances(rows)
    return rows


def select_cases(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pick a deterministic diverse set and a separate review list.

    Stratified farthest-point sampling uses standardized descriptive features.
    A K-spacing term keeps the two picks in one (p, Delta) cell from collapsing
    onto adjacent grid points. Supplements add extreme values that the strata
    did not already include. No state labels are used.
    """
    matrix = np.column_stack([[float(row[name]) for row in rows] for name in DIVERSITY_FEATURES])
    if not np.all(np.isfinite(matrix)):
        raise InventoryError("diversity features contain undefined values")
    scale = np.std(matrix, axis=0)
    active = scale > 0.0
    if not np.any(active):
        raise InventoryError("diversity features have no variation")
    center = np.mean(matrix[:, active], axis=0)
    standardized = (matrix[:, active] - center) / scale[active]
    coupling = np.asarray([float(row["K"]) for row in rows], dtype=np.float64)
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(quantize(row["p"]), quantize(row["Delta"]))].append(index)

    chosen: list[int] = []
    reasons: dict[int, list[str]] = defaultdict(list)
    for key in sorted(strata, key=lambda item: (-item[0], item[1])):
        members = strata[key]
        first = _best_index(
            members,
            lambda index: (
                -float(np.linalg.norm(standardized[index])),
                quantize(rows[index]["K"]),
                rows[index]["run_id"],
            ),
        )
        chosen.append(first)
        reasons[first].append("stratum_diversity_extreme")
        if CASES_PER_STRATUM < 2:
            continue
        second = _best_index(
            [index for index in members if index != first],
            lambda index, anchor=first: (
                -_selection_distance(index, anchor, standardized, coupling),
                quantize(rows[index]["K"]),
                rows[index]["run_id"],
            ),
        )
        chosen.append(second)
        reasons[second].append("stratum_diversity_spread")

    already = set(chosen)
    for reason, metric in SUPPLEMENT_METRICS:
        if len(chosen) >= CASES_PER_STRATUM * len(strata) + N_SUPPLEMENTS:
            break
        ranked = sorted(range(len(rows)), key=lambda index: (-float(rows[index][metric]), rows[index]["run_id"]))
        for index in ranked:
            if index in already:
                continue
            chosen.append(index)
            reasons[index].append(reason)
            already.add(index)
            break

    cases = []
    for index in chosen:
        item = dict(rows[index])
        item["selection_reason"] = ";".join(reasons[index])
        item["selection_detail"] = _selection_detail(reasons[index])
        cases.append(item)
    cases.sort(key=lambda item: (-quantize(item["p"]), quantize(item["Delta"]), quantize(item["K"]), item["run_id"]))
    for number, item in enumerate(cases, start=1):
        item["case_id"] = f"C{number:02d}"

    reviews = _review_rows(rows, {item["run_id"]: item["case_id"] for item in cases})
    return cases, reviews


def write_feature_table(path: Path, rows: list[dict]) -> None:
    _write_csv(path, rows, _feature_fields(rows))


def write_case_tables(output_dir: Path, cases: list[dict], reviews: list[dict]) -> None:
    case_fields = [
        "case_id",
        "run_id",
        "K",
        "p",
        "Delta",
        "trial_id",
        "network_seed",
        "initial_phase_seed",
        "selection_reason",
        "selection_detail",
        "plot_path",
        *_RELEVANT_FEATURES,
    ]
    review_fields = [
        "run_id",
        "K",
        "p",
        "Delta",
        "trial_id",
        "network_seed",
        "initial_phase_seed",
        "review_reasons",
        "review_detail",
        "representative_case_id",
        *_RELEVANT_FEATURES,
    ]
    _write_csv(output_dir / "representative_cases.csv", cases, case_fields)
    _write_csv(output_dir / "review_candidates.csv", reviews, review_fields)


def write_figures(output_dir: Path, rows: list[dict], cases: list[dict], records: dict[str, dict]) -> None:
    _use_agg()
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    case_dir = output_dir / "cases"
    figure_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    for directory in (figure_dir, case_dir):
        for child in directory.glob("*.png"):
            child.unlink()
    _plot_lines_by_parameter(
        rows,
        figure_dir / "near_zero_fraction_vs_K.png",
        [
            ("zero_frac_eps0p001_w3", "0.001", "#0072B2"),
            ("zero_frac_eps0p005_w3", "0.005", "#56B4E9"),
            ("zero_frac_eps0p01_w3", "0.01", "#E69F00"),
            ("zero_frac_eps0p02_w3", "0.02", "#D55E00"),
        ],
        ylabel="fraction with |rho| < epsilon",
        title="Near-zero fraction versus K, window 12000-14000",
        ylim=(0.0, 1.0),
    )
    _plot_lines_by_parameter(
        rows,
        figure_dir / "rotation_range_vs_K.png",
        [
            ("rho_range_w1", "8000-10000", PROFILE_COLORS["w1"]),
            ("rho_range_w3", "12000-14000", PROFILE_COLORS["w3"]),
        ],
        ylabel="rotation-number range",
        title="Rotation-number range versus K",
        share_y_by_row=True,
    )
    _plot_lines_by_parameter(
        rows,
        figure_dir / "local_slope_vs_K.png",
        [
            ("slope11_abs_q50_w3", "11 oscillators", "#0072B2"),
            ("slope31_abs_q50_w3", "31 oscillators", "#D55E00"),
        ],
        ylabel="median |local slope|",
        title="Median absolute local slope versus K, window 12000-14000",
        share_y_by_row=True,
    )
    _plot_lines_by_parameter(
        rows,
        figure_dir / "window_variation_vs_K.png",
        [
            ("D12_p95", "|w2-w1| p95", PROFILE_COLORS["w1"]),
            ("D23_p95", "|w3-w2| p95", PROFILE_COLORS["w2"]),
            ("D13_p95", "|w3-w1| p95", PROFILE_COLORS["w3"]),
        ],
        ylabel="95th percentile of |Delta rho|",
        title="Window-to-window absolute difference versus K",
        share_y_by_row=True,
    )
    _plot_heatmaps(rows, figure_dir / "feature_heatmaps.png")
    for case in cases:
        destination = case_dir / f"{case['case_id']}.png"
        _plot_case(case, records[case["run_id"]], destination)
        case["plot_path"] = destination.relative_to(output_dir).as_posix()
    _plot_contact_sheet(cases, records, figure_dir / "representative_contact_sheet.png")
    plt.close("all")


def write_summary(path: Path, rows: list[dict], cases: list[dict], reviews: list[dict]) -> dict:
    quantile_names = [
        "zero_frac_eps0p001_w3",
        "zero_frac_eps0p01_w3",
        "zero_frac_eps0p02_w3",
        "zero_frac_eps_sensitivity_w3",
        "rho_range_w1",
        "rho_range_w3",
        "rho_std_w3",
        "rho_minus_omega_mean_w3",
        "slope11_abs_q50_w3",
        "slope31_abs_q50_w3",
        "slope_scale_gap_w3",
        "slope31_std_w3",
        "D12_median",
        "D12_p95",
        "D23_p95",
        "D13_median",
        "D13_p95",
        "D13_max",
        "delta_rho_mean_w3_minus_w1",
    ]
    summary = {
        "n_runs": len(rows),
        "n_unique_run_ids": len({row["run_id"] for row in rows}),
        "feature_quantiles": {name: _quantiles([row[name] for row in rows]) for name in quantile_names},
        "largest_adjacent_jumps": {
            name: _largest_jumps(rows, name, limit=8)
            for name in ("zero_frac_eps0p01_w3", "rho_range_w3", "slope31_abs_q50_w3", "D13_p95")
        },
        "representative_cases": [
            {
                "case_id": case["case_id"],
                "run_id": case["run_id"],
                "K": case["K"],
                "p": case["p"],
                "Delta": case["Delta"],
                "selection_reason": case["selection_reason"],
            }
            for case in cases
        ],
        "n_representative_cases": len(cases),
        "n_review_candidates": len(reviews),
        "review_reason_counts": _reason_counts(reviews),
        "notes": [
            "Quantiles and jump sizes are computed from saved rotation profiles.",
            "Adjacent jumps are descriptive changes between neighboring K values.",
            "They are not estimated phase boundaries.",
            "Near-zero tolerances and local-slope cutoffs are exploratory parameters.",
        ],
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def analysis_config_document(config_path: Path, output_dir: Path) -> dict:
    return {
        "pilot_config": str(config_path),
        "output_dir": str(output_dir),
        "windows": {key: list(bounds) for key, bounds in WINDOW_BOUNDS.items()},
        "near_zero_epsilons": list(EPSILONS),
        "slope_windows_oscillators": list(SLOPE_WINDOWS),
        "local_slope_absolute_cutoffs": list(SLOPE_CUTOFFS),
        "window_difference_tolerances": list(DIFF_TOLERANCES),
        "rho_quantiles": [probability for probability, _tag in RHO_QUANTILES],
        "diversity_features": list(DIVERSITY_FEATURES),
        "cases_per_stratum": CASES_PER_STRATUM,
        "n_supplements": N_SUPPLEMENTS,
        "review_top_n": REVIEW_TOP_N,
        "k_distance_weight": K_DISTANCE_WEIGHT,
        "selection_rng_seed": None,
        "selection_is_deterministic": True,
        "missing_csv_value": "empty field means undefined, not zero",
        "interpretation": (
            "Features describe saved profiles. Epsilons and slope cutoffs are not "
            "definitions of dynamical states."
        ),
    }


def run_analysis(config_path: Path, output_dir: Path, data_dir: Path | None = None) -> dict:
    config = load_config(config_path)
    directory = config.output_dir if data_dir is None else Path(data_dir)
    specs = verify_pilot_index(config_path, output_dir=directory)
    print(f"inventory {len(specs)} completed runs", flush=True)
    rows = collect_feature_rows(specs, directory)
    cases, reviews = select_cases(rows)
    selected = load_profiles([case["run_id"] for case in cases], output_dir=directory)
    records = {record["run_id"]: record for record in selected}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "cases").mkdir(exist_ok=True)
    write_figures(output_dir, rows, cases, records)
    write_feature_table(output_dir / "features.csv", rows)
    write_case_tables(output_dir, cases, reviews)
    document = analysis_config_document(config_path, output_dir)
    (output_dir / "analysis_config.json").write_text(
        json.dumps(document, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = write_summary(output_dir / "summary.json", rows, cases, reviews)
    _write_readme(output_dir)
    print(
        f"wrote {output_dir} features={len(rows)} cases={len(cases)} reviews={len(reviews)}",
        flush=True,
    )
    return summary


def _add_profile_features(row: dict, key: str, omega: np.ndarray, rho: np.ndarray) -> None:
    difference = rho - omega
    row[f"rho_min_{key}"] = float(np.min(rho))
    row[f"rho_max_{key}"] = float(np.max(rho))
    row[f"rho_mean_{key}"] = float(np.mean(rho))
    row[f"rho_std_{key}"] = float(np.std(rho))
    row[f"rho_range_{key}"] = float(np.max(rho) - np.min(rho))
    for probability, tag in RHO_QUANTILES:
        row[f"rho_{tag}_{key}"] = float(np.quantile(rho, probability))
    row[f"rho_minus_omega_mean_{key}"] = float(np.mean(difference))
    row[f"rho_minus_omega_std_{key}"] = float(np.std(difference))
    row[f"rho_minus_omega_min_{key}"] = float(np.min(difference))
    row[f"rho_minus_omega_max_{key}"] = float(np.max(difference))
    for epsilon in EPSILONS:
        row[f"zero_frac_eps{EPS_TAGS[epsilon]}_{key}"] = float(
            near_zero_fraction(rho, epsilon)["value"]
        )
    for window in SLOPE_WINDOWS:
        summary = slope_summaries(omega, rho, window)
        prefix = f"slope{window}"
        row[f"{prefix}_count_{key}"] = summary["count"]
        row[f"{prefix}_std_{key}"] = summary["std"]
        row[f"{prefix}_abs_q10_{key}"] = summary["abs_q10"]
        row[f"{prefix}_abs_q50_{key}"] = summary["abs_q50"]
        row[f"{prefix}_abs_q90_{key}"] = summary["abs_q90"]
        for cutoff in SLOPE_CUTOFFS:
            tag = CUTOFF_TAGS[cutoff]
            row[f"{prefix}_frac_abs_lt_{tag}_{key}"] = summary[f"frac_abs_lt_{tag}"]


def _add_window_differences(row: dict, windows: dict[str, np.ndarray]) -> None:
    for name, earlier, later in DIFF_PAIRS:
        signed = windows[later] - windows[earlier]
        absolute = np.abs(signed)
        row[f"{name}_mean"] = float(np.mean(absolute))
        row[f"{name}_median"] = float(np.median(absolute))
        row[f"{name}_p90"] = float(np.quantile(absolute, 0.90))
        row[f"{name}_p95"] = float(np.quantile(absolute, 0.95))
        row[f"{name}_max"] = float(np.max(absolute))
        row[f"delta_rho_mean_{later}_minus_{earlier}"] = float(np.mean(signed))
        row[f"delta_rho_median_{later}_minus_{earlier}"] = float(np.median(signed))
        for tolerance in DIFF_TOLERANCES:
            row[f"{name}_frac_gt_{DIFF_TAGS[tolerance]}"] = float(np.mean(absolute > float(tolerance)))


def _add_neighbor_distances(rows: list[dict]) -> None:
    matrix = np.column_stack([[float(row[name]) for row in rows] for name in DIVERSITY_FEATURES])
    scale = np.std(matrix, axis=0)
    active = scale > 0.0
    standardized = np.zeros_like(matrix[:, active])
    np.divide(matrix[:, active] - np.mean(matrix[:, active], axis=0), scale[active], out=standardized)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row["neighbor_feature_distance"] = np.nan
        row["neighbor_K"] = np.nan
        row["neighbor_run_id"] = ""
        groups[(quantize(row["p"]), quantize(row["Delta"]))].append(index)
    for members in groups.values():
        members.sort(key=lambda index: (quantize(rows[index]["K"]), rows[index]["run_id"]))
        for left, right in zip(members, members[1:]):
            distance = float(np.linalg.norm(standardized[left] - standardized[right]))
            for index, other in ((left, right), (right, left)):
                current = rows[index]["neighbor_feature_distance"]
                if not np.isfinite(current) or distance > float(current):
                    rows[index]["neighbor_feature_distance"] = distance
                    rows[index]["neighbor_K"] = float(rows[other]["K"])
                    rows[index]["neighbor_run_id"] = rows[other]["run_id"]


def _selection_distance(index: int, anchor: int, standardized: np.ndarray, coupling: np.ndarray) -> float:
    span = float(np.max(coupling) - np.min(coupling))
    spacing = 0.0 if span == 0.0 else abs(float(coupling[index] - coupling[anchor])) / span
    return float(np.linalg.norm(standardized[index] - standardized[anchor]) + K_DISTANCE_WEIGHT * spacing)


def _best_index(candidates: list[int], key) -> int:
    best = candidates[0]
    best_key = key(best)
    for index in candidates[1:]:
        item = key(index)
        if item < best_key:
            best = index
            best_key = item
    return best


def _selection_detail(reasons: list[str]) -> str:
    text = {
        "stratum_diversity_extreme": (
            "Largest standardized-feature distance from the global mean within this p and Delta."
        ),
        "stratum_diversity_spread": (
            "Farthest from the first selected run in this p and Delta, including a K-spacing term."
        ),
        "high_window_variation": "Large 95th percentile of |rho(12000-14000) - rho(8000-10000)|.",
        "early_window_variation": "Large 95th percentile of |rho(10000-12000) - rho(8000-10000)|.",
        "slope_scale_sensitive": (
            "Large absolute difference between median |local slope| at 11 and 31 oscillators."
        ),
        "neighbor_K_feature_jump": "Large standardized-feature distance from a neighboring K.",
        "high_rotation_dispersion": "Large standard deviation of rotation numbers in 12000-14000.",
        "near_zero_tolerance_sensitive": (
            "Large difference between fractions |rho| < 0.02 and |rho| < 0.001 in 12000-14000."
        ),
    }
    return " ".join(text[reason] for reason in reasons)


def _review_rows(rows: list[dict], case_ids: dict[str, str]) -> list[dict]:
    hits: dict[str, list[str]] = defaultdict(list)
    details: dict[str, list[str]] = defaultdict(list)
    n = len(rows)
    for reason, metric in REVIEW_METRICS:
        ranked = sorted(rows, key=lambda row: (-float(row[metric]), row["run_id"]))
        for rank, row in enumerate(ranked[:REVIEW_TOP_N], start=1):
            hits[row["run_id"]].append(reason)
            details[row["run_id"]].append(f"{reason}: {metric}={float(row[metric]):.6g} rank {rank}/{n}")
    selected = []
    for row in rows:
        if row["run_id"] not in hits:
            continue
        item = dict(row)
        item["review_reasons"] = ";".join(hits[row["run_id"]])
        item["review_detail"] = " | ".join(details[row["run_id"]])
        item["representative_case_id"] = case_ids.get(row["run_id"], "")
        selected.append(item)
    selected.sort(key=lambda item: (-quantize(item["p"]), quantize(item["Delta"]), quantize(item["K"]), item["run_id"]))
    return selected


def _check_record_matches_spec(record: dict, spec) -> None:
    parameters = record["parameters"]
    realization = record["realization"]
    if quantize(parameters["K"]) != spec.k_units or quantize(parameters["p"]) != spec.p_units:
        raise InventoryError(f"{spec.run_id} saved parameters do not match the plan")
    if quantize(parameters["Delta"]) != spec.delta_units or int(parameters["N"]) != spec.N:
        raise InventoryError(f"{spec.run_id} saved parameters do not match the plan")
    if int(realization["trial_id"]) != spec.trial_id:
        raise InventoryError(f"{spec.run_id} saved trial_id does not match the plan")
    if int(realization["network_seed"]) != int(spec.network_seed):
        raise InventoryError(f"{spec.run_id} network_seed does not match the plan")
    if int(realization["initial_phase_seed"]) != int(spec.initial_phase_seed):
        raise InventoryError(f"{spec.run_id} initial_phase_seed does not match the plan")


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(values)
    out = cumulative[window - 1 :].copy()
    if out.size > 1:
        out[1:] -= cumulative[:-window]
    return out


def _feature_fields(rows: list[dict]) -> list[str]:
    preferred = [
        "run_id",
        "K",
        "p",
        "Delta",
        "N",
        "trial_id",
        "network_seed",
        "initial_phase_seed",
        "frequency_seed",
        "network_realization_id",
        "initial_condition_id",
        "paired_group_id",
    ]
    rest = [key for key in rows[0] if key not in preferred]
    return preferred + rest


_RELEVANT_FEATURES = [
    "zero_frac_eps0p001_w3",
    "zero_frac_eps0p01_w3",
    "zero_frac_eps0p02_w3",
    "zero_frac_eps_sensitivity_w3",
    "rho_min_w3",
    "rho_max_w3",
    "rho_mean_w3",
    "rho_std_w3",
    "rho_range_w3",
    "rho_minus_omega_mean_w3",
    "slope11_abs_q50_w3",
    "slope31_abs_q50_w3",
    "slope_scale_gap_w3",
    "slope31_std_w3",
    "slope11_frac_abs_lt_0p1_w3",
    "slope31_frac_abs_lt_0p1_w3",
    "D12_median",
    "D12_p95",
    "D12_max",
    "D23_p95",
    "D13_median",
    "D13_p95",
    "D13_max",
    "delta_rho_mean_w3_minus_w1",
    "neighbor_feature_distance",
    "neighbor_K",
    "neighbor_run_id",
]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _csv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return ""
        return f"{number:.8g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"n_finite": 0, "min": None, "q05": None, "q50": None, "q95": None, "max": None}
    return {
        "n_finite": int(finite.size),
        "min": float(np.min(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q50": float(np.quantile(finite, 0.50)),
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _largest_jumps(rows: list[dict], feature: str, limit: int) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(quantize(row["p"]), quantize(row["Delta"]))].append(row)
    jumps = []
    for group in groups.values():
        group.sort(key=lambda row: quantize(row["K"]))
        for left, right in zip(group, group[1:]):
            difference = float(right[feature] - left[feature])
            jumps.append(
                {
                    "p": left["p"],
                    "Delta": left["Delta"],
                    "K_left": left["K"],
                    "K_right": right["K"],
                    "run_id_left": left["run_id"],
                    "run_id_right": right["run_id"],
                    "difference_right_minus_left": difference,
                    "abs_difference": abs(difference),
                }
            )
    jumps.sort(key=lambda item: (-item["abs_difference"], item["run_id_left"]))
    return jumps[:limit]


def _reason_counts(reviews: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in reviews:
        for reason in str(row["review_reasons"]).split(";"):
            if reason:
                counts[reason] += 1
    return dict(counts)


def _parameter_grid(rows: list[dict]) -> tuple[list[float], list[float], list[float]]:
    coupling = sorted({quantize(row["K"]) for row in rows})
    delta = sorted({quantize(row["Delta"]) for row in rows})
    p_values = sorted({quantize(row["p"]) for row in rows}, reverse=True)
    return (
        [next(row["K"] for row in rows if quantize(row["K"]) == key) for key in coupling],
        [next(row["p"] for row in rows if quantize(row["p"]) == key) for key in p_values],
        [next(row["Delta"] for row in rows if quantize(row["Delta"]) == key) for key in delta],
    )


def _plot_lines_by_parameter(rows, path, series, ylabel, title, ylim=None, share_y_by_row=False) -> None:
    import matplotlib.pyplot as plt

    coupling, p_values, delta_values = _parameter_grid(rows)
    figure, axes = plt.subplots(
        len(delta_values),
        len(p_values),
        figsize=(3.1 * len(p_values), 2.5 * len(delta_values)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, delta in enumerate(delta_values):
        panel_rows = []
        for column_index, p_value in enumerate(p_values):
            axis = axes[row_index, column_index]
            group = [
                item
                for item in rows
                if quantize(item["p"]) == quantize(p_value) and quantize(item["Delta"]) == quantize(delta)
            ]
            group.sort(key=lambda item: quantize(item["K"]))
            panel_rows.append(group)
            k_values = [item["K"] for item in group]
            for feature, label, color in series:
                axis.plot(
                    k_values,
                    [item[feature] for item in group],
                    color=color,
                    linewidth=1.2,
                    label=label,
                )
            axis.set_title(f"p = {canonical_label(p_value)}, Delta = {canonical_label(delta)}")
            axis.grid(True, linewidth=0.3, alpha=0.5)
            if column_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == len(delta_values) - 1:
                axis.set_xlabel("K")
        if ylim is not None:
            for axis in axes[row_index]:
                axis.set_ylim(*ylim)
        elif share_y_by_row:
            values = [
                item[feature]
                for group in panel_rows
                for item in group
                for feature, _, _ in series
                if np.isfinite(item[feature])
            ]
            if values:
                low, high = min(values), max(values)
                pad = 0.05 * (high - low if high > low else 1.0)
                for axis in axes[row_index]:
                    axis.set_ylim(low - pad, high + pad)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    figure.suptitle(title)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_heatmaps(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    coupling, p_values, delta_values = _parameter_grid(rows)
    figure, axes = plt.subplots(
        len(HEATMAP_FEATURES),
        len(p_values),
        figsize=(3.3 * len(p_values), 2.3 * len(HEATMAP_FEATURES)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    extent = [
        coupling[0] - 0.025,
        coupling[-1] + 0.025,
        -0.5,
        len(delta_values) - 0.5,
    ]
    for row_index, (feature, label) in enumerate(HEATMAP_FEATURES):
        matrices = [_feature_matrix(rows, p_value, coupling, delta_values, feature) for p_value in p_values]
        stacked = np.concatenate([matrix.ravel() for matrix in matrices])
        finite = stacked[np.isfinite(stacked)]
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
        image = None
        for column_index, matrix in enumerate(matrices):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                interpolation="nearest",
            )
            axis.set_yticks(range(len(delta_values)), [canonical_label(value) for value in delta_values])
            if column_index == 0:
                axis.set_ylabel("Delta")
            if row_index == len(HEATMAP_FEATURES) - 1:
                axis.set_xlabel("K")
            axis.set_title(f"p = {canonical_label(p_values[column_index])}")
        figure.colorbar(image, ax=list(axes[row_index, :]), fraction=0.03, pad=0.02, label=label)
    figure.suptitle("Window 12000-14000, except |w3-w1|. Shared color scale within each row.")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _feature_matrix(rows, p_value, coupling, delta_values, feature) -> np.ndarray:
    matrix = np.full((len(delta_values), len(coupling)), np.nan, dtype=np.float64)
    k_index = {quantize(value): index for index, value in enumerate(coupling)}
    d_index = {quantize(value): index for index, value in enumerate(delta_values)}
    for row in rows:
        if quantize(row["p"]) != quantize(p_value):
            continue
        matrix[d_index[quantize(row["Delta"])], k_index[quantize(row["K"])]] = float(row[feature])
    return matrix


def _plot_case(case: dict, record: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    windows = assign_window_profiles(record)
    omega = np.asarray(record["omega"], dtype=np.float64)
    ordered = sort_by_frequency(omega, *[windows[key] for key in WINDOW_KEYS])
    omega_sorted = ordered["omega"]
    profiles = ordered["arrays"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    low, high = _profile_limits(omega_sorted, profiles)
    for key, profile in zip(WINDOW_KEYS, profiles):
        axes[0].plot(omega_sorted, profile, color=PROFILE_COLORS[key], linewidth=1.15, label=WINDOW_LABELS[key])
    axes[0].plot(omega_sorted, omega_sorted, color="#666666", linewidth=0.8, linestyle="--", label="rho = omega")
    axes[0].set_ylim(low, high)
    axes[0].set_xlabel("omega")
    axes[0].set_ylabel("rho")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, linewidth=0.3, alpha=0.5)
    difference_colors = {"D12": PROFILE_COLORS["w1"], "D23": PROFILE_COLORS["w2"], "D13": PROFILE_COLORS["w3"]}
    difference_labels = {
        "D12": "|rho(10000-12000) - rho(8000-10000)|",
        "D23": "|rho(12000-14000) - rho(10000-12000)|",
        "D13": "|rho(12000-14000) - rho(8000-10000)|",
    }
    for name, earlier, later in DIFF_PAIRS:
        absolute = np.abs(windows[later] - windows[earlier])
        sorted_difference = sort_by_frequency(omega, absolute)["arrays"][0]
        axes[1].plot(
            omega_sorted,
            sorted_difference,
            color=difference_colors[name],
            linewidth=1.0,
            label=difference_labels[name],
        )
    axes[1].set_xlabel("omega")
    axes[1].set_ylabel("|Delta rho|")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(True, linewidth=0.3, alpha=0.5)
    figure.suptitle(
        f"{case['case_id']}  {case['run_id']}\n"
        f"K={canonical_label(case['K'])}  p={canonical_label(case['p'])}  "
        f"Delta={canonical_label(case['Delta'])}  trial={case['trial_id']}\n"
        f"|rho|<0.01 fraction (12000-14000)={case['zero_frac_eps0p01_w3']:.3f}   "
        f"rho range={case['rho_range_w3']:.3f}   "
        f"median |slope| 11/31={case['slope11_abs_q50_w3']:.3f}/{case['slope31_abs_q50_w3']:.3f}   "
        f"D13 p95={case['D13_p95']:.3g}"
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_contact_sheet(cases: list[dict], records: dict[str, dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    p_values = sorted({quantize(case["p"]) for case in cases}, reverse=True)
    columns = []
    for p_key in p_values:
        column = [case for case in cases if quantize(case["p"]) == p_key]
        column.sort(key=lambda case: (quantize(case["Delta"]), quantize(case["K"])))
        columns.append(column)
    n_rows = max(len(column) for column in columns)
    figure, axes = plt.subplots(
        n_rows,
        len(columns),
        figsize=(3.3 * len(columns), 2.15 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for column_index, column in enumerate(columns):
        axes[0, column_index].set_title(f"p = {canonical_label(columns[column_index][0]['p'])}")
        for row_index in range(n_rows):
            axis = axes[row_index, column_index]
            if row_index >= len(column):
                axis.axis("off")
                continue
            case = column[row_index]
            windows = assign_window_profiles(records[case["run_id"]])
            omega = np.asarray(records[case["run_id"]]["omega"], dtype=np.float64)
            ordered = sort_by_frequency(omega, *[windows[key] for key in WINDOW_KEYS])
            low, high = _profile_limits(ordered["omega"], ordered["arrays"])
            for key, profile in zip(WINDOW_KEYS, ordered["arrays"]):
                axis.plot(ordered["omega"], profile, color=PROFILE_COLORS[key], linewidth=0.9)
            axis.plot(ordered["omega"], ordered["omega"], color="#666666", linewidth=0.6, linestyle="--")
            axis.set_ylim(low, high)
            axis.set_title(
                f"{case['case_id']}  Delta={canonical_label(case['Delta'])}  K={canonical_label(case['K'])}",
                fontsize=8,
            )
            axis.tick_params(labelsize=7)
            if row_index == n_rows - 1:
                axis.set_xlabel("omega")
            if column_index == 0:
                axis.set_ylabel("rho")
    figure.suptitle("Representative rotation profiles. Colors are the three observation windows; dashed line is rho = omega.")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _profile_limits(omega: np.ndarray, profiles: tuple[np.ndarray, ...]) -> tuple[float, float]:
    stacked = np.concatenate([np.asarray(profile, dtype=np.float64).ravel() for profile in profiles])
    low = min(float(np.min(omega)), float(np.min(stacked)), 0.0)
    high = max(float(np.max(omega)), float(np.max(stacked)))
    if high <= low:
        high = low + 1.0
    pad = 0.05 * (high - low)
    return low - pad, high + pad


def _use_agg() -> None:
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass


def _write_readme(path: Path) -> None:
    text = """# Pilot exploratory analysis

Descriptive summaries of the saved Stage 1 pilot runs. The script reads
`stage1_runs` through `load_profiles` and does not integrate the ODE or modify
the SQLite index.

Near-zero tolerances and local-slope cutoffs are exploratory parameters stored
in `analysis_config.json`. They are not definitions of dynamical states.
Empty CSV fields are undefined values, not zeros. `frequency_seed` is empty
because the pilot frequencies are a deterministic grid.

Window keys are `w1` = 8000-10000, `w2` = 10000-12000, and `w3` = 12000-14000.
`D12`, `D23`, and `D13` are absolute differences between those windows.
Local slopes are least-squares slopes of rho against omega after one shared
frequency sort, on windows of 11 and 31 oscillators.

```bash
python3 -m stage1.explore
```

Outputs:

- `features.csv`: one row per run
- `representative_cases.csv`: diversity sample for visual review
- `review_candidates.csv`: high window variation, slope-scale sensitivity, neighboring-K changes, dispersion, and near-zero tolerance sensitivity
- `figures/`: parameter-space maps and one contact sheet
- `cases/`: one two-panel figure per representative case
- `summary.json`: numeric quantiles and the largest adjacent-K feature changes
"""
    (path / "README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="stage1.explore")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "stage1" / "configs" / "stage1_pilot.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "stage1_analysis" / "pilot_exploration")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    data_dir = None if args.data_dir is None else (args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir)
    run_analysis(args.config, output, data_dir=data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
