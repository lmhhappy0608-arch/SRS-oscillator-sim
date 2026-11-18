"""Recalibrate the Hybrid A trend cap from incremental errors.

The archived Rule 3 geometry, Method C cuts, and state priority stay fixed.
tau_S is the only free parameter. Constant-segment false positives are
reported and are not used to reject a trend cap. The script does not rerun
the Winfree ODE and does not write Classifier v1 or v2.

Run from this directory:

    python3 tau_s_recalibration.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-hybrid-method-c")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

import hybrid_a_validation as stage1
import hybrid_method_c as hybrid
from calibrate_plateau_detector import catalog

HERE = Path(__file__).resolve().parent
REPORT_PATH = HERE / "tau_s_recalibration_report.md"
FIGURE_PATH = HERE / "tau_s_recalibration.png"
CONFIG_PATH = HERE / "tau_s_recalibration_config.json"
MASTER_SEED = 20261109
BENCHMARK_VERSION = "hybrid-a-tau-calibration-v1"
# A zero-FP cap is called meaningful only when it recovers at least this
# fraction of the baseline misses that some geometry-valid linear segment
# could match. The fraction is a reporting rule, not a tuned score.
MEANINGFUL_RECOVERY = 0.5
FOCUS = (
    (1.0, 0.20, 0.60),
    (1.0, 0.20, 0.70),
    (1.0, 0.20, 0.75),
    (1.0, 0.15, 0.40),
    (0.4, 0.20, 1.45),
    (0.4, 0.15, 1.00),
    (0.4, 0.175, 1.45),
    (0.4, 0.175, 1.50),
    (0.4, 0.175, 1.65),
)
SPLIT_SPEC = {
    "dev": {
        "delta": 0.20,
        "seeds": (0, 1),
        "slow": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
        "drift": (5e-4, 1e-3, 2e-3, 5e-3),
        "quad": (2e-4, 5e-4, 1e-3, 2e-3),
        "sine": (2e-4, 1e-3, 3e-3),
        "block": (1e-5, 5e-5, 1e-4, 3e-4),
        "tail_body": (0.0, 1e-5, 1e-4),
        "tail_rise": (5e-4, 2e-3, 0.05),
        "noise": (1e-5, 1e-4, 1e-3, 0.01),
    },
    "val": {
        "delta": 0.15,
        "seeds": (2, 3),
        "slow": (2e-5, 6e-5, 2e-4, 6e-4, 2e-3),
        "drift": (8e-4, 1.5e-3, 4e-3),
        "quad": (3e-4, 8e-4, 1.5e-3),
        "sine": (4e-4, 1.5e-3),
        "block": (2e-5, 8e-5, 2e-4),
        "tail_body": (0.0, 2e-5, 8e-5),
        "tail_rise": (8e-4, 3e-3, 0.08),
        "noise": (3e-5, 3e-4, 3e-3),
    },
    "test": {
        "delta": 0.25,
        "seeds": (4, 5),
        "slow": (1.5e-5, 4e-5, 1.5e-4, 4e-4, 1.5e-3, 4e-3),
        "drift": (7e-4, 1.2e-3, 3e-3),
        "quad": (2.5e-4, 7e-4, 1.2e-3),
        "sine": (3e-4, 2e-3),
        "block": (1.5e-5, 4e-5, 1.5e-4, 4e-4),
        "tail_body": (0.0, 1.5e-5, 4e-5),
        "tail_rise": (4e-4, 1.5e-3, 0.04),
        "noise": (2e-5, 2e-4, 5e-3),
    },
}
NEGATIVE_FAMILIES = ("slow_slope", "fragmented_drift", "nonlinear")
POSITIVE_FAMILIES = ("fragmented_plateau", "plateau_tail", "noisy_plateau")


def search_grid() -> np.ndarray:
    knots = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    values = list(np.geomspace(1e-6, 1e-2, 33)) + knots
    folded = np.round(np.log10(np.asarray(values, dtype=float)), 6)
    unique = np.unique(folded)
    return np.power(10.0, unique)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denom = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def pack_profile(cost_audit, pilot, omega: np.ndarray, rho: np.ndarray, truth: np.ndarray, meta: dict) -> dict:
    segments = stage1.segment_fast(cost_audit, pilot, omega, rho)
    index = hybrid.ProfileIndex(omega, rho, segments)
    truth = np.asarray(truth, dtype=bool)
    truth_runs = hybrid.true_runs(truth)
    constants = []
    linears = []
    fail_geometry = 0
    for atom, mode in zip(index.atomic, index.modes):
        if mode == 1 and not hybrid.geometry_ok(atom):
            fail_geometry += 1
            continue
        if not hybrid.geometry_ok(atom) or not hybrid.nonzero_ok(atom):
            continue
        record = {
            "start": int(atom["start"]),
            "end": int(atom["end"]),
            "n": int(atom["n"]),
            "relative_omega_span": float(atom["relative_omega_span"]),
            "R": float(atom["R"]),
            "S": float(atom["S"]),
            "mean": float(atom["mean"]),
        }
        if truth_runs:
            ious = [hybrid.interval_iou((record["start"], record["end"]), run) for run in truth_runs]
            best = int(np.argmax(ious))
            record["best_iou"] = float(ious[best])
            record["best_truth"] = best
        else:
            record["best_iou"] = 0.0
            record["best_truth"] = -1
        if mode == 1:
            linears.append(record)
        else:
            constants.append(record)
    recoverable = []
    for run_index, run in enumerate(truth_runs):
        candidates = [item["S"] for item in linears if item["best_truth"] == run_index and item["best_iou"] >= 0.5]
        recoverable.append(min(candidates) if candidates else None)
    return {
        **meta,
        "truth_runs": truth_runs,
        "truth_lengths": [end - start for start, end in truth_runs],
        "constants": constants,
        "linears": linears,
        "fail_geometry": fail_geometry,
        "n_segments": len(segments),
        "recoverable_S": recoverable,
        "sigma_hat": float(pilot.estimate_sigma(rho)),
    }


def _matched(items: list[dict], run_index: int, tau: float | None) -> dict | None:
    pool = [
        item
        for item in items
        if item["best_truth"] == run_index and item["best_iou"] >= 0.5 and (tau is None or item["S"] <= tau)
    ]
    if not pool:
        return None
    return max(pool, key=lambda item: item["best_iou"])


def evaluate(packs: list[dict], tau: float) -> dict:
    inc_fp = 0
    inc_fp_denom = 0
    extra_profiles = 0
    extra_segments = 0
    baseline_fp = 0
    negatives = 0
    inc_tp = 0
    baseline_misses = 0
    recoverable_misses = 0
    recovered_recoverable = 0
    boundary = []
    precisions = []
    false_segments = 0
    osc_tp = osc_fp = osc_fn = 0
    base_osc_tp = base_osc_fp = base_osc_fn = 0
    by_family_fp = {}
    first_fp_family = []
    for pack in packs:
        accepted_linear = [item for item in pack["linears"] if item["S"] <= tau]
        positive = bool(pack["truth_runs"])
        if not positive:
            negatives += 1
            flagged = bool(pack["constants"])
            if flagged:
                baseline_fp += 1
                if accepted_linear:
                    extra_profiles += 1
                    extra_segments += len(accepted_linear)
            else:
                inc_fp_denom += 1
                if accepted_linear:
                    inc_fp += 1
                    by_family_fp[pack["family"]] = by_family_fp.get(pack["family"], 0) + 1
                    first_fp_family.append(pack["family"])
            continue
        truth_total = int(sum(pack["truth_lengths"]))
        base_cover = _coverage(pack["constants"], pack["truth_runs"], truth_total)
        hybrid_items = pack["constants"] + accepted_linear
        hybrid_cover = _coverage(hybrid_items, pack["truth_runs"], truth_total)
        for item in accepted_linear:
            overlap = sum(_overlap((item["start"], item["end"]), run) for run in pack["truth_runs"])
            if item["n"] == 0 or overlap / item["n"] < 0.5:
                false_segments += 1
        base_osc_tp += base_cover[0]
        base_osc_fp += base_cover[1]
        base_osc_fn += base_cover[2]
        osc_tp += hybrid_cover[0]
        osc_fp += hybrid_cover[1]
        osc_fn += hybrid_cover[2]
        for run_index, run in enumerate(pack["truth_runs"]):
            base_hit = _matched(pack["constants"], run_index, None) is not None
            linear_hit = _matched(accepted_linear, run_index, tau)
            recoverable = pack["recoverable_S"][run_index] is not None
            if base_hit:
                continue
            baseline_misses += 1
            if recoverable:
                recoverable_misses += 1
            if linear_hit is None:
                continue
            inc_tp += 1
            if recoverable:
                recovered_recoverable += 1
            overlap = _overlap((linear_hit["start"], linear_hit["end"]), run)
            precisions.append(overlap / linear_hit["n"] if linear_hit["n"] else 0.0)
            boundary.append(0.5 * (abs(linear_hit["start"] - run[0]) + abs(linear_hit["end"] - run[1])))
    fp_low, fp_high = wilson(inc_fp, inc_fp_denom)
    rec_low, rec_high = wilson(recovered_recoverable, recoverable_misses)
    return {
        "tau": float(tau),
        "negatives": negatives,
        "baseline_fp": baseline_fp,
        "inc_fp": inc_fp,
        "inc_fp_denom": inc_fp_denom,
        "inc_fpr": (inc_fp / inc_fp_denom) if inc_fp_denom else None,
        "inc_fpr_low": fp_low,
        "inc_fpr_high": fp_high,
        "extra_profiles": extra_profiles,
        "extra_segments": extra_segments,
        "false_segments": false_segments,
        "inc_fp_by_family": by_family_fp,
        "baseline_misses": baseline_misses,
        "inc_tp": inc_tp,
        "recoverable_misses": recoverable_misses,
        "recovered_recoverable": recovered_recoverable,
        "recovery": (recovered_recoverable / recoverable_misses) if recoverable_misses else None,
        "recovery_low": rec_low,
        "recovery_high": rec_high,
        "precision_median": float(np.median(precisions)) if precisions else None,
        "boundary_median": float(np.median(boundary)) if boundary else None,
        "osc_precision": osc_tp / (osc_tp + osc_fp) if (osc_tp + osc_fp) else None,
        "osc_recall": osc_tp / (osc_tp + osc_fn) if (osc_tp + osc_fn) else None,
        "osc_f1": _f1(osc_tp, osc_fp, osc_fn),
        "baseline_osc_f1": _f1(base_osc_tp, base_osc_fp, base_osc_fn),
    }


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _coverage(items: list[dict], truth_runs: list[tuple[int, int]], truth_total: int) -> tuple[int, int, int]:
    true_positive = 0
    predicted = 0
    for item in items:
        predicted += item["n"]
        true_positive += sum(_overlap((item["start"], item["end"]), run) for run in truth_runs)
    false_positive = predicted - true_positive
    false_negative = truth_total - true_positive
    return true_positive, false_positive, false_negative


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    if true_positive + false_positive + false_negative == 0:
        return None
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def old_style_stage(packs: list[dict], tau: float) -> dict:
    """Reproduce the previous profile-level false-positive rule on one split."""
    groups = {"drift": [], "slow": [], "nonlinear": []}
    long_hits = []
    for pack in packs:
        accepted = bool(pack["constants"] or any(item["S"] <= tau for item in pack["linears"]))
        if pack["family"] in groups and not pack["truth_runs"]:
            groups[pack["family"]].append(float(accepted))
        for run_index, length in enumerate(pack["truth_lengths"]):
            easy = pack.get("sigma", 1.0) <= 1e-3 + 1e-15 and length >= 80
            if not easy:
                continue
            hit = _matched(pack["constants"], run_index, None) is not None or _matched(
                [item for item in pack["linears"] if item["S"] <= tau], run_index, tau
            ) is not None
            long_hits.append(float(hit))
    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")
    return {
        "fp_drift": mean(groups["drift"]),
        "fp_slow": mean(groups["slow"]),
        "fp_nonlinear": mean(groups["nonlinear"]),
        "recall_long": mean(long_hits),
        "n_drift": len(groups["drift"]),
        "n_slow": len(groups["slow"]),
        "n_nonlinear": len(groups["nonlinear"]),
    }


def audit_previous(original_dev: list[dict]) -> dict:
    grid = tuple(float(value) for value in np.geomspace(1e-4, hybrid.MAX_R, 8))
    stages = []
    for tau in grid:
        metrics = old_style_stage(original_dev, tau)
        fp_ok = metrics["fp_drift"] <= 0.05 and metrics["fp_slow"] <= 0.05 and metrics["fp_nonlinear"] <= 0.05
        fp_loose = metrics["fp_drift"] <= 0.10 and metrics["fp_slow"] <= 0.10 and metrics["fp_nonlinear"] <= 0.10
        stage = -1
        if fp_ok and metrics["recall_long"] >= 0.80:
            stage = 0
        elif fp_ok and metrics["recall_long"] >= 0.50:
            stage = 2
        elif fp_loose and metrics["recall_long"] >= 0.50:
            stage = 3
        stages.append({"tau": tau, "stage_reached": stage, **metrics})
    quad = [
        pack
        for pack in original_dev
        if pack["family"] == "nonlinear" and pack.get("kind") == "quad" and pack.get("sigma", 0.0) >= 0.009
    ]
    quad_constants = [
        {
            "seed": pack["seed"],
            "n": item["n"],
            "R": item["R"],
            "S": item["S"],
        }
        for pack in quad
        for item in pack["constants"]
    ]
    return {
        "grid": list(grid),
        "grid_matches_report": [round(value, 6) for value in grid],
        "lower_edge": grid[0],
        "quad_constant_false_positives": quad_constants,
        "grid_stages": stages,
    }


SPLIT_CODE = {"dev": 1, "val": 2, "test": 3}
FAMILY_CODE = {"slow": 1, "drift": 2, "quad": 3, "sine": 4, "block": 5, "tail": 6, "noise": 7}


def _rng(split: str, family: str, seed: int, scale: float) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [MASTER_SEED, SPLIT_CODE[split], FAMILY_CODE[family], int(seed), int(round(float(scale) * 1e12))]
        )
    )


def _omega(delta: float) -> np.ndarray:
    return np.linspace(1.0 - delta, 1.0 + delta, 1000)


def _noise(rng: np.random.Generator, scale: float = 1e-6) -> np.ndarray:
    return rng.normal(0.0, scale, size=1000)


def build_calibration(split: str) -> list[dict]:
    spec = SPLIT_SPEC[split]
    delta = float(spec["delta"])
    omega = _omega(delta)
    span = float(omega[-1] - omega[0])
    built = []
    serial = 0

    def add(family: str, kind: str, seed: int, rho: np.ndarray, truth: np.ndarray, sigma: float, note: str) -> None:
        nonlocal serial
        built.append(
            {
                "omega": omega,
                "rho": rho,
                "truth": truth,
                "split": split,
                "benchmark": "calibration_v1",
                "family": family,
                "kind": kind,
                "sigma": sigma,
                "seed": seed,
                "delta": delta,
                "note": note,
                "serial": serial,
            }
        )
        serial += 1

    for seed in spec["seeds"]:
        for target in spec["slow"]:
            rng = _rng(split, "slow", seed, target)
            rho = 0.45 + (target / span) * (omega - omega[0]) + _noise(rng)
            add("slow_slope", "line", seed, rho, np.zeros(1000, dtype=bool), 1e-6, f"designed_S={target}")
        for target in spec["drift"]:
            rng = _rng(split, "drift", seed, target)
            cycles = 6
            # Eighty percent of the slope that would reverse the curve, so the
            # path stays monotone while the bend is large enough to be cut.
            amplitude = 0.8 * target / (2.0 * math.pi * cycles)
            t = (omega - omega[0]) / span
            rho = 0.35 + target * t + amplitude * np.sin(2.0 * math.pi * cycles * t) + _noise(rng)
            add("fragmented_drift", "wobble", seed, rho, np.zeros(1000, dtype=bool), 1e-6, f"designed_S={target}")
        for height in spec["quad"]:
            rng = _rng(split, "quad", seed, height)
            t = (omega - omega[0]) / span
            rho = 0.30 + height * (t - 0.5) ** 2 + _noise(rng)
            add("nonlinear", "quad", seed, rho, np.zeros(1000, dtype=bool), 1e-6, f"height={height}")
        for height in spec["sine"]:
            rng = _rng(split, "sine", seed, height)
            t = (omega - omega[0]) / span
            rho = 0.40 + height * t + 0.25 * height * np.sin(2.0 * math.pi * t) + _noise(rng)
            add("nonlinear", "sine", seed, rho, np.zeros(1000, dtype=bool), 1e-6, f"height={height}")
        for piece_s in spec["block"]:
            rho = np.zeros(1000)
            truth = np.zeros(1000, dtype=bool)
            cursor = 0
            level = 0.50
            for block in range(5):
                end = cursor + 140
                local = float(omega[end - 1] - omega[cursor])
                rho[cursor:end] = level + (piece_s / local) * (omega[cursor:end] - omega[cursor])
                truth[cursor:end] = True
                level = float(rho[end - 1])
                cursor = end
                if block < 4:
                    gap = 40
                    rho[cursor : cursor + gap] = np.linspace(level, level + 0.01, gap)
                    cursor += gap
                    level = float(rho[cursor - 1])
            if cursor < 1000:
                rho[cursor:] = np.linspace(level, level + 0.01, 1000 - cursor)
            rng = _rng(split, "block", seed, piece_s)
            add("fragmented_plateau", "blocks", seed, rho + _noise(rng), truth, 1e-6, f"piece_S={piece_s}")
        for body_s in spec["tail_body"]:
            for rise in spec["tail_rise"]:
                body = 800
                rho = np.empty(1000)
                local = float(omega[body - 1] - omega[0])
                slope = 0.0 if local == 0.0 else body_s / local
                rho[:body] = 0.60 + slope * (omega[:body] - omega[0])
                rho[body:] = rho[body - 1] + rise * np.linspace(0.0, 1.0, 1000 - body)
                truth = np.zeros(1000, dtype=bool)
                truth[:body] = True
                rng = _rng(split, "tail", seed, body_s * 1000.0 + rise)
                add("plateau_tail", "tail", seed, rho + _noise(rng), truth, 1e-6, f"body_S={body_s};rise={rise}")
        for sigma in spec["noise"]:
            rng = _rng(split, "noise", seed, sigma)
            rho = 0.62 + rng.normal(0.0, sigma, size=1000)
            add("noisy_plateau", "flat", seed, rho, np.ones(1000, dtype=bool), sigma, f"sigma={sigma}")
    return built


def choose_from_development(rows: list[dict]) -> dict:
    """Development-only rule. Validation and test are not read here.

    A threshold is clean when it adds no false profile on a baseline-negative
    curve and no linear segment whose oscillators are mostly outside the
    generated plateau. Extra linear segments on a curve the baseline already
    flagged are recorded elsewhere and do not enter this rule.
    """
    clean = [row for row in rows if row["inc_fp"] == 0 and row["false_segments"] == 0]
    if not clean:
        return {
            "status": "no_clean_frontier",
            "admissible": [],
            "provisional_tau": None,
            "meaningful": False,
            "reason": (
                "No development threshold simultaneously has zero incremental false "
                "profiles and zero mostly-off-plateau linear segments."
            ),
        }
    best = max(row["recovered_recoverable"] for row in clean)
    frontier = [row for row in clean if row["recovered_recoverable"] == best]
    boundaries = [row["boundary_median"] for row in frontier if row["boundary_median"] is not None]
    if boundaries and max(boundaries) - min(boundaries) > 5.0:
        best_boundary = min(boundaries)
        frontier = [
            row
            for row in frontier
            if row["boundary_median"] is not None and row["boundary_median"] <= best_boundary + 5.0
        ]
    taus = [row["tau"] for row in frontier]
    spanned = [row for row in rows if taus[0] - 1e-15 <= row["tau"] <= taus[-1] + 1e-15]
    contiguous = len(spanned) == len(frontier)
    recovery = frontier[-1]["recovery"]
    meaningful = recovery is not None and recovery >= MEANINGFUL_RECOVERY
    if len(taus) == 1 and meaningful:
        status = "unique"
        reason = (
            "One clean development threshold recovers strictly the most "
            "geometry-recoverable baseline misses."
        )
    elif meaningful:
        status = "equivalent_interval"
        reason = (
            "Several clean development thresholds recover the same number of "
            "geometry-recoverable baseline misses. They are not distinguished."
        )
    else:
        status = "clean_but_weak_recovery"
        reason = (
            "The clean thresholds recover less than half of the baseline misses "
            "that a geometry-valid linear segment could match. Further recall "
            "requires an incremental false profile or a mostly-off-plateau segment."
        )
    return {
        "status": status,
        "admissible": [taus[0], taus[-1]] if contiguous else taus,
        "contiguous": contiguous,
        "frontier_taus": taus,
        "recovered": best,
        "recovery": recovery,
        "meaningful": meaningful,
        "provisional_tau": None if status != "unique" else taus[0],
        "reason": reason,
    }


def real_scan(cost_audit, pilot, grid: np.ndarray) -> dict:
    _audit, rows = stage1.audit_rows()
    records = []
    focus = []
    linear_counts = {float(tau): 0 for tau in grid}
    flip_counts = {float(tau): 0 for tau in grid}
    short_counts = {float(tau): 0 for tau in grid}
    by_delta = {float(tau): {} for tau in grid}
    started = __import__("time").perf_counter()
    for index, row in enumerate(rows, start=1):
        loaded = stage1.load_run(row["path"])
        if loaded is None:
            continue
        omega, matrix = loaded
        segments = stage1.segment_fast(cost_audit, pilot, omega, matrix[2])
        profile = hybrid.ProfileIndex(omega, matrix[2], segments)
        constants = []
        linears = []
        for atom, mode in zip(profile.atomic, profile.modes):
            if not hybrid.geometry_ok(atom) or not hybrid.nonzero_ok(atom):
                continue
            item = (int(atom["n"]), float(atom["relative_omega_span"]), float(atom["R"]), float(atom["S"]))
            if mode == 1:
                linears.append(item)
            else:
                constants.append(item)
        const_mask = np.zeros(omega.size, dtype=bool)
        for atom, mode in zip(profile.atomic, profile.modes):
            if mode == 0 and hybrid.geometry_ok(atom) and hybrid.nonzero_ok(atom):
                const_mask[atom["start"] : atom["end"]] = True
        base_state = hybrid.assign_state(matrix[2], const_mask)
        min_s = min((item[3] for item in linears), default=None)
        key = (round(row["p"], 4), round(row["delta"], 4), round(row["K"], 4))
        if key in FOCUS:
            windows = [hybrid.ProfileIndex(omega, matrix[window], [(0, omega.size, 1)]) for window in (0, 1)]
            details = []
            for atom, mode in zip(profile.atomic, profile.modes):
                if mode != 1 or not hybrid.geometry_ok(atom) or not hybrid.nonzero_ok(atom):
                    continue
                w1_ok, w1 = stage1.passes_on(windows[0], atom["start"], atom["end"])
                w2_ok, w2 = stage1.passes_on(windows[1], atom["start"], atom["end"])
                gap = np.abs(matrix[2, atom["start"] : atom["end"]] - matrix[0, atom["start"] : atom["end"]])
                details.append(
                    {
                        "start": int(atom["start"]),
                        "end": int(atom["end"]),
                        "n": int(atom["n"]),
                        "relative_omega_span": float(atom["relative_omega_span"]),
                        "R": float(atom["R"]),
                        "S": float(atom["S"]),
                        "w1_S": float(w1["S"]),
                        "w2_S": float(w2["S"]),
                        "w1_pass_at_1e-4": bool(w1_ok),
                        "w2_pass_at_1e-4": bool(w2_ok),
                        "mean_abs_w1": float(gap.mean()),
                    }
                )
            focus.append(
                {
                    "p": row["p"],
                    "delta": row["delta"],
                    "K": row["K"],
                    "base_state": base_state,
                    "n_segments": len(segments),
                    "min_S": min_s,
                    "rho_range": float(matrix[2].max() - matrix[2].min()),
                    "linears": details,
                }
            )
        for tau in grid:
            accepted = [item for item in linears if item[3] <= tau]
            linear_counts[float(tau)] += len(accepted)
            short_counts[float(tau)] += sum(item[0] < 30 for item in accepted)
            if base_state == "Incoherence" and accepted:
                flip_counts[float(tau)] += 1
                delta_key = f"{round(row['delta'], 4):.4f}"
                by_delta[float(tau)][delta_key] = by_delta[float(tau)].get(delta_key, 0) + 1
        records.append({"base_state": base_state, "min_S": min_s, "p": row["p"], "delta": row["delta"], "K": row["K"]})
        if index % 1000 == 0 or index == len(rows):
            print(f"real scan {index}/{len(rows)} in {__import__('time').perf_counter() - started:.1f}s", flush=True)
    quality = []
    # Recompute quality from the stored focus is incomplete. Keep run-level min_S
    # and the aggregate counts produced above.
    ordered = []
    for tau in grid:
        quality.append(
            {
                "tau": float(tau),
                "linear_plateaus": linear_counts[float(tau)],
                "short_n_lt_30": short_counts[float(tau)],
                "incoherence_to_partial": flip_counts[float(tau)],
                "label_changes_by_delta": by_delta[float(tau)],
            }
        )
    return {"runs": len(records), "focus": focus, "curve": quality}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "na"
    if isinstance(value, float):
        return f"{value:.6g}" if abs(value) < 0.001 or abs(value) >= 1000 else f"{value:.{digits}g}"
    return str(value)


def write_figure(dev: list[dict], val: list[dict], real: dict, decision: dict) -> None:
    figure = plt.figure(figsize=(11.2, 7.4))
    grid = figure.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    taus = [row["tau"] for row in dev]
    axis = figure.add_subplot(grid[0, 0])
    axis.plot(taus, [row["recovery"] or 0.0 for row in dev], color="#4C78A8", label="development recovery")
    axis.plot(taus, [row["inc_fpr"] or 0.0 for row in dev], color="#E45756", label="development incremental FPR")
    axis.plot(taus, [row["recovery"] or 0.0 for row in val], color="#4C78A8", ls="--", label="validation recovery")
    axis.plot(taus, [row["inc_fpr"] or 0.0 for row in val], color="#E45756", ls="--", label="validation incremental FPR")
    if decision["admissible"] and decision.get("contiguous", True):
        axis.axvspan(decision["admissible"][0], decision["admissible"][1], color="#F2C14E", alpha=0.35, label="zero error, no recovery")
    axis.set_xscale("log")
    axis.set_xlabel("tau_S")
    axis.set_ylim(-0.02, 1.05)
    axis.set_title("Calibration benchmark", loc="left", fontsize=10)
    axis.legend(frameon=False, fontsize=7)

    axis = figure.add_subplot(grid[0, 1])
    axis.plot([row["inc_fpr"] or 0.0 for row in dev], [row["recovery"] or 0.0 for row in dev], "o-", color="#4C78A8", ms=3, label="development")
    axis.plot([row["inc_fpr"] or 0.0 for row in val], [row["recovery"] or 0.0 for row in val], "s--", color="#72B7B2", ms=3, label="validation")
    axis.set_xlabel("incremental FPR")
    axis.set_ylabel("recoverable-shelf recall")
    axis.set_title("Incremental trade-off", loc="left", fontsize=10)
    axis.legend(frameon=False, fontsize=7)

    axis = figure.add_subplot(grid[1, 0])
    real_tau = [row["tau"] for row in real["curve"]]
    axis.plot(real_tau, [row["linear_plateaus"] for row in real["curve"]], color="#4C78A8")
    if decision["admissible"] and decision.get("contiguous", True):
        axis.axvspan(decision["admissible"][0], decision["admissible"][1], color="#F2C14E", alpha=0.35)
    axis.set_xscale("log")
    axis.set_xlabel("tau_S")
    axis.set_ylabel("accepted linear plateaus")
    axis.set_title("Saved Winfree profiles", loc="left", fontsize=10)

    axis = figure.add_subplot(grid[1, 1])
    axis.plot(real_tau, [row["incoherence_to_partial"] for row in real["curve"]], color="#E45756")
    if decision["admissible"] and decision.get("contiguous", True):
        axis.axvspan(decision["admissible"][0], decision["admissible"][1], color="#F2C14E", alpha=0.35)
    axis.axvline(1e-4, color="#333333", lw=0.8, label="previous candidate")
    axis.set_xscale("log")
    axis.set_xlabel("tau_S")
    axis.set_ylabel("Incoherence to Partial Locking")
    axis.set_title("Label changes, not an objective", loc="left", fontsize=10)
    axis.legend(frameon=False, fontsize=7)
    figure.savefig(FIGURE_PATH, dpi=140)
    plt.close(figure)


def _row_line(row: dict) -> str:
    return (
        f"| {row['tau']:.6g} | {row['recovered_recoverable']}/{row['recoverable_misses']} | "
        f"{fmt(row['recovery'])} | {row['inc_fp']}/{row['inc_fp_denom']} | {fmt(row['inc_fpr'])} | "
        f"{row['false_segments']} | {fmt(row['precision_median'])} | {fmt(row['boundary_median'])} | {fmt(row['osc_f1'])} |"
    )


def write_report(audit: dict, original: dict, calibration: dict, decision: dict, val_note: dict, test_rows: list[dict], real: dict) -> None:
    quad = audit["quad_constant_false_positives"]
    quad_text = ", ".join(f"seed {item['seed']}: n={item['n']}, R={item['R']:.4g}, S={item['S']:.4g}" for item in quad) or "none"
    stage_lines = [
        f"| {row['tau']:.6g} | {row['fp_drift']:.3f} | {row['fp_slow']:.3f} | {row['fp_nonlinear']:.3f} | {row['recall_long']:.3f} | {row['stage_reached']} |"
        for row in audit["grid_stages"]
    ]
    shown = []
    for row in calibration["dev"]:
        log_value = math.log10(row["tau"])
        named = abs(log_value - round(log_value)) < 0.05 or any(
            abs(row["tau"] - knot) / knot < 0.02 for knot in (5e-5, 2e-4, 5e-4)
        )
        on_frontier = bool(decision["admissible"]) and decision["admissible"][0] - 1e-15 <= row["tau"] <= decision["admissible"][1] + 1e-15
        endpoint = bool(decision["admissible"]) and any(abs(row["tau"] - edge) <= 1e-15 * max(1.0, edge) for edge in decision["admissible"])
        if named or endpoint or (on_frontier and abs(log_value * 2 - round(log_value * 2)) < 0.04):
            shown.append(row)
    # Deduplicate while preserving order. Frontier endpoints are always kept.
    seen = set()
    ordered = []
    for row in shown:
        key = round(math.log10(row["tau"]), 6)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(row)
    shown = ordered
    dev_table = "\n".join(_row_line(row) for row in shown)
    val_table = "\n".join(_row_line(row) for row in calibration["val"] if any(abs(math.log10(row["tau"]) - math.log10(item["tau"])) < 1e-9 for item in shown))
    test_table = "\n".join(_row_line(row) for row in test_rows if any(abs(math.log10(row["tau"]) - math.log10(item["tau"])) < 1e-9 for item in shown))
    focus_lines = []
    for item in sorted(real["focus"], key=lambda row: (row["p"], row["delta"], row["K"])):
        accepting = [seg for seg in item["linears"] if item["min_S"] is not None and abs(seg["S"] - item["min_S"]) <= 1e-15 * max(1.0, item["min_S"])]
        lead = min(item["linears"], key=lambda seg: seg["S"]) if item["linears"] else None
        if lead is None:
            detail = "no geometry-valid linear segment"
        else:
            detail = (
                f"n={lead['n']}, rel={lead['relative_omega_span']:.3f}, R={lead['R']:.3g}, "
                f"W1 S={lead['w1_S']:.3g}, W2 S={lead['w2_S']:.3g}, mean |W3-W1|={lead['mean_abs_w1']:.3g}"
            )
        focus_lines.append(
            f"| {item['p']:.3g} | {item['delta']:.3g} | {item['K']:.3g} | {item['base_state']} | {item['n_segments']} | "
            f"{fmt(item['min_S'])} | {item['rho_range']:.4g} | {detail} |"
        )
    real_lines = []
    for row in real["curve"]:
        if any(abs(math.log10(row["tau"]) - math.log10(item["tau"])) < 1e-9 for item in shown):
            real_lines.append(
                f"| {row['tau']:.6g} | {row['linear_plateaus']} | {row['short_n_lt_30']} | {row['incoherence_to_partial']} |"
            )
    admissible = decision["admissible"]
    if not admissible:
        interval = "none"
    elif decision.get("contiguous", True):
        interval = f"[{admissible[0]:.6g}, {admissible[1]:.6g}]"
    else:
        interval = ", ".join(f"{value:.6g}" for value in admissible)
    first_fp = next((row for row in calibration["dev"] if row["inc_fp"] > 0), None)
    first_off = next((row for row in calibration["dev"] if row["false_segments"] > 0), None)
    family_start = {}
    for row in calibration["dev"]:
        for family in row["inc_fp_by_family"]:
            family_start.setdefault(family, row["tau"])
    family_text = ", ".join(f"{name} at {value:.6g}" for name, value in family_start.items()) or "none inside the grid"
    text = f"""# Hybrid A tau_S recalibration

Hybrid A is otherwise fixed: archived Method C cuts, archived Rule 3, no segment merging, and the existing state priority. Only `tau_S` is recalibrated. The previous choice `tau_S = 1e-4` is not reused as an answer. Real Winfree labels are not ground truth, and the number of new Partial Locking labels is not an objective. Synthetic curves are generated profiles, not Winfree trajectories.

## Audit of the previous selection

The old search grid in `hybrid_method_c.py` is `geomspace(1e-4, 0.02, 8)`:

| tau_S | FP drift | FP slow | FP nonlinear | Long recall | First stage reached |
|---|---:|---:|---:|---:|---:|
{chr(10).join(stage_lines)}

Stage codes follow the old script: 0 is the strict joint rule, 2 is false-positive rate at most 0.05 with long recall at least 0.50, 3 is false-positive rate at most 0.10 with long recall at least 0.50, and -1 means even stage 3 failed. The old selector took the first stage that any grid point passed, then the smallest stable `tau_S` inside it. Recomputing that grid on the original development catalog agrees with the recorded facts:

- `1e-4` is the lower edge. No smaller cap was on the grid.
- The nonlinear false-positive rate stays at 0.083 because one noisy quadratic has a Constant segment that passes Rule 3. Recorded Constant hits: {quad_text}. A linear trend cap cannot remove a Constant segment.
- That single Constant hit keeps the false-positive rate above 0.05, so the first three stages never open. The selector therefore lands in the loosest stage for every tau that has not yet started accepting slow slopes.
- Inside that stage the smallest stable value is the grid edge, `1e-4`.

This is an evaluation-design problem, not a segmentation bug. The old false-positive rate mixes baseline Constant errors with new Linear errors, and the smallest-stable rule then returns the untested boundary of the grid. The piece guard is not identified at that edge; it only belongs to the rejected merging rule.

## What the original benchmark can separate

On the original development catalog, geometry-valid linear segments of true plateaus have `S_I` at numerical zero or they do not exist. Noise of 0.01 makes the plateau fail the archived range cap, so `tau_S` cannot recover it. Noiseless slow slopes remain one segment whose `S_I` is the full rotation-number change, about `0.002` and above.

| tau_S | Recovered / recoverable | Recall | Incremental FP | FPR | Off-plateau segments | Precision | Boundary | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(_row_line(row) for row in original["dev_selected"])}

From `1e-6` through `1e-3` the incremental counts do not move. The first incremental false profiles appear near `0.002`, where a full slow slope finally satisfies the cap. The original benchmark therefore cannot choose among `5e-5`, `1e-4`, `2e-4`, and `5e-4`. It mostly scores Constant segments and full-span slopes. The validation and test splits of that catalog were already inspected in earlier rounds, so they are reported only as history and are not used to place the new cap.

## Calibration benchmark

Because the original catalog is flat across the relevant decade, a separate calibration catalog was generated. It is not a modification of the old test set. Development, validation, and test use disjoint seeds and shifted numerical values. The master seed is `{MASTER_SEED}`. Development uses `delta = 0.20`, validation `0.15`, and test `0.25`.

- Fragmented plateau: five length-140 blocks with a designed segment trend, separated by short steep connectors. The blocks are the truth.
- Fragmented drift: a monotone curve plus a sine wobble at 80 percent of the slope that would reverse it, so Method C can cut locally flatter pieces. There is no truth plateau.
- Slow slope: one designed trend from about `1e-5` to a few times `1e-3`, with `1e-6` noise so the noise scale does not collapse to the numerical floor.
- Nonlinear: shallow quadratics and sine bends, with no truth plateau.
- Plateau with tail: a body of designed trend and a separate tail whose trend is `5e-4`, a few times `1e-3`, or several times `1e-2`. Only the body is truth. A body trend of 0 is the Constant-segment control.
- Noisy plateau: a flat truth at several noise scales, including values that violate the Rule 3 range cap.

The selection rule was fixed before validation or test scores were consulted:

1. On development, a threshold is clean when it has zero incremental false profiles and zero accepted linear segments whose oscillators lie mostly outside the generated plateau.
2. Among clean thresholds, keep those that recover the most geometry-recoverable baseline misses.
3. If their median boundary errors differ by more than five oscillators, drop the worse ones.
4. Call the remaining set meaningful only if it recovers at least half of those recoverable misses.
5. Emit one value only when one threshold remains. Otherwise report the equivalent interval.
6. Validation may shrink that interval when it adds an incremental false profile or a mostly-off-plateau segment. The held-out test is scored after the decision and does not move it.
7. Linear segments added to a curve the baseline already flagged are reported separately. They do not enter the clean-threshold rule, because a Constant-segment error cannot be removed by `tau_S`.

## Development and validation

Development, calibration catalog:

| tau_S | Recovered / recoverable | Recall | Incremental FP | FPR | Off-plateau segments | Precision | Boundary | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{dev_table}

Validation uses the same columns. It can shrink the frontier. It does not invent the rule.

| tau_S | Recovered / recoverable | Recall | Incremental FP | FPR | Off-plateau segments | Precision | Boundary | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{val_table}

Development decision: **{decision["status"]}**. Clean frontier: {interval}. Recovered shelves at that frontier: {decision.get("recovered")}. {decision["reason"]}

The first development threshold with an incremental false profile is {fmt(None if first_fp is None else first_fp["tau"])} ({0 if first_fp is None else first_fp["inc_fp"]} profiles). The first with a mostly-off-plateau segment is {fmt(None if first_off is None else first_off["tau"])}. Incremental false profiles begin by family at: {family_text}.

Validation stability: {val_note["text"]}

## Held-out calibration test, scored once

| tau_S | Recovered / recoverable | Recall | Incremental FP | FPR | Off-plateau segments | Precision | Boundary | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{test_table}

Wilson intervals for the small counts are in `tau_s_recalibration_config.json`. A percentage from a denominator of a few dozen profiles is not a precise risk.

## Saved Winfree profiles

All {real["runs"]} saved profiles were segmented once. Higher-priority states do not change with `tau_S`. The only possible label change is Incoherence to Partial Locking, and only when a geometry-valid linear segment has `S_I` at or below the cap.

| tau_S | Linear plateaus | n < 30 | Incoherence to Partial Locking |
|---|---:|---:|---:|
{chr(10).join(real_lines)}

At `1e-4` this reproduces the previous screen: 1781 linear plateaus and 6 label changes. The six changes, and the five shelves that appear between `1e-4` and `5e-4`, were read from the curves in the previous gallery. They are diagnostics.

| p | delta | K | Baseline state | Segments | Minimum linear S | Profile range | Flattest linear segment |
|---|---:|---:|---|---:|---:|---:|---|
{chr(10).join(focus_lines)}

`p=1, delta=0.15, K=0.40` has a visible flat step whose flattest geometry-valid segment has `S_I` near `8e-4`. Accepting it requires a cap in that neighborhood. The same cap also accepts slower members of the synthetic drift families. Window gaps on the new pieces remain finite-window disagreements, not long-time locking.

## Recommendation

Status: **{decision["status"]}**.

{decision["recommendation"]}

The result is not written into Classifier v2. A shelf segment and a drift segment with the same `S_I` are the same object under this rule, so a cap cannot separate them once their trends overlap.

## Limitations

- The calibration curves are constructed segments, not integrated Winfree trajectories.
- Rule 3 still rejects short pieces and any piece whose rotation-number range exceeds 0.02, whatever `tau_S` is. Noise at 0.01 on a flat plateau is mostly in that group.
- The original validation and test catalogs were already inspected and are not a fresh final test.
- The new test split was scored once, but it is small and uses the same six generators with shifted numbers.
- Every saved run has `N=1000`.
- Three finite windows are not a long-time frequency lock.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    hybrid.load_archived_constants()
    pilot, cost_audit = hybrid.load_method_c()
    grid = search_grid()
    print("segment original catalogs", flush=True)
    original_packs = {"dev": [], "val": [], "test": []}
    for split in ("dev", "val", "test"):
        for profile in catalog(split):
            rho = profile.rho if profile.rho.ndim == 1 else profile.rho[2]
            truth = profile.truth if profile.truth.ndim == 1 else profile.truth[2]
            original_packs[split].append(
                pack_profile(
                    cost_audit,
                    pilot,
                    profile.omega,
                    rho,
                    truth,
                    {
                        "split": split,
                        "benchmark": "original",
                        "family": profile.family,
                        "kind": profile.kind,
                        "sigma": float(profile.sigma),
                        "seed": int(profile.seed),
                    },
                )
            )
        print(split, len(original_packs[split]), flush=True)
    audit = audit_previous(original_packs["dev"])
    original_eval = {
        split: [evaluate(original_packs[split], float(tau)) for tau in grid]
        for split in ("dev", "val", "test")
    }
    original_eval["dev_selected"] = [
        row
        for row in original_eval["dev"]
        if any(abs(math.log10(row["tau"]) - math.log10(knot)) < 0.02 for knot in (1e-6, 1e-5, 1e-4, 1e-3, 2e-3, 1e-2))
    ]
    print("segment calibration catalogs", flush=True)
    calibration_packs = {}
    for split in ("dev", "val", "test"):
        raw = build_calibration(split)
        calibration_packs[split] = [
            pack_profile(cost_audit, pilot, item["omega"], item["rho"], item["truth"], {key: item[key] for key in item if key not in {"omega", "rho", "truth"}})
            for item in raw
        ]
        print(split, len(calibration_packs[split]), "median segments", float(np.median([pack["n_segments"] for pack in calibration_packs[split]])), flush=True)
    dev_rows = [evaluate(calibration_packs["dev"], float(tau)) for tau in grid]
    val_rows = [evaluate(calibration_packs["val"], float(tau)) for tau in grid]
    decision = choose_from_development(dev_rows)
    decision["development_frontier"] = list(decision.get("frontier_taus") or [])
    # Validation can only remove thresholds that gain an incremental false profile.
    if decision["admissible"]:
        frontier_keys = {round(math.log10(value), 6) for value in decision["frontier_taus"]}
        stable = [
            row["tau"]
            for row in val_rows
            if round(math.log10(row["tau"]), 6) in frontier_keys and row["inc_fp"] == 0 and row["false_segments"] == 0
        ]
        if not stable:
            decision["status"] = "unstable_on_validation"
            decision["provisional_tau"] = None
            decision["admissible"] = []
            decision["recommendation"] = (
                "Development has a clean frontier, but validation adds an incremental "
                "false profile or a mostly-off-plateau segment throughout that frontier. "
                "No value is recommended."
            )
            val_note = {"text": "Validation errors cover the whole development frontier."}
        elif [round(math.log10(value), 6) for value in stable] != sorted(frontier_keys):
            inside = [row["tau"] for row in val_rows if stable[0] - 1e-15 <= row["tau"] <= stable[-1] + 1e-15]
            decision["contiguous"] = len(inside) == len(stable)
            decision["admissible"] = [stable[0], stable[-1]] if decision["contiguous"] else stable
            decision["frontier_taus"] = stable
            one = len(stable) == 1
            if one and decision.get("meaningful"):
                decision["status"] = "unique"
                decision["provisional_tau"] = stable[0]
                decision["recommendation"] = (
                    f"Validation leaves one clean threshold, {stable[0]:.6g}. " + decision["reason"]
                )
            else:
                decision["status"] = "shrunk_by_validation"
                decision["provisional_tau"] = None
                span = f"{stable[0]:.6g}" if one else f"[{stable[0]:.6g}, {stable[-1]:.6g}]"
                decision["recommendation"] = (
                    f"Validation removes part of the development frontier. The overlap is {span}. " + decision["reason"]
                )
            val_note = {"text": f"Validation stays clean on {decision['admissible']} and rejects the rest of the development frontier."}
        else:
            decision["recommendation"] = decision["reason"] + " Validation stays clean inside this frontier."
            val_note = {"text": "Validation also stays clean on the development frontier."}
    else:
        decision["recommendation"] = decision["reason"]
        val_note = {"text": "No development frontier was available to check."}
    test_rows = [evaluate(calibration_packs["test"], float(tau)) for tau in grid]
    print("real scan", flush=True)
    real = real_scan(cost_audit, pilot, grid)
    at_candidate = next(row for row in real["curve"] if abs(row["tau"] - 1e-4) / 1e-4 < 1e-6)
    if at_candidate["linear_plateaus"] != 1781 or at_candidate["incoherence_to_partial"] != 6:
        raise RuntimeError(
            f"real scan disagrees with the previous screen: {at_candidate['linear_plateaus']} plateaus, "
            f"{at_candidate['incoherence_to_partial']} label changes"
        )
    write_figure(dev_rows, val_rows, real, decision)
    write_report(audit, original_eval, {"dev": dev_rows, "val": val_rows}, decision, val_note, test_rows, real)
    config = {
        "benchmark_version": BENCHMARK_VERSION,
        "master_seed": MASTER_SEED,
        "status": decision["status"],
        "provisional_tau_s": decision.get("provisional_tau"),
        "admissible_interval": decision.get("admissible"),
        "meaningful_recovery_rule": MEANINGFUL_RECOVERY,
        "selection_rule": decision["reason"],
        "recommendation": decision.get("recommendation"),
        "split_spec": SPLIT_SPEC,
        "search_grid": [float(value) for value in grid],
        "previous_grid_audit": {
            "lower_edge": audit["lower_edge"],
            "quad_constant_false_positives": audit["quad_constant_false_positives"],
            "stages": audit["grid_stages"],
        },
        "development": dev_rows,
        "validation": val_rows,
        "test_scored_once": test_rows,
        "real_curve": real["curve"],
        "focus": [
            {
                "p": item["p"],
                "delta": item["delta"],
                "K": item["K"],
                "base_state": item["base_state"],
                "min_S": item["min_S"],
                "n_segments": item["n_segments"],
                "rho_range": item["rho_range"],
            }
            for item in real["focus"]
        ],
    }
    CONFIG_PATH.write_text(json.dumps(_jsonable(config), indent=1), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "admissible": decision.get("admissible"), "provisional": decision.get("provisional_tau")}, indent=2), flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
