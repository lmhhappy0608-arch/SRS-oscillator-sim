"""Oscillator-level and boundary-level scores for piecewise profiles.

Constant precision, recall, and F1 compare the predicted constant mask with
true plateaus. A distinguishable linear oscillator predicted as constant is a
false constant. An unresolved slope predicted as constant is neither a true
positive nor a false positive: the slope was pre-tagged as practically
unresolved when its OLS t-statistic is below 3.

Weights below were fixed before reading the grid. They are not fit to C15 or C23.
"""

from __future__ import annotations

import math

import numpy as np

# Primary score weights. They sum to 1.
WEIGHT_F1 = 0.30
WEIGHT_FALSE_CONSTANT = 0.20
WEIGHT_MISSED_PLATEAU = 0.15
WEIGHT_TYPE = 0.15
WEIGHT_COUNT = 0.10
WEIGHT_BOUNDARY = 0.10

# A boundary error of this many oscillators saturates the boundary term.
BOUNDARY_SCALE_OSCILLATORS = 20.0
# Segment-count term saturates at this multiple of the true segment count.
COUNT_SCALE = 1.0

FALSE_CONSTANT_FLAG = 0.05
MISSED_PLATEAU_FLAG = 0.50
MIN_PLATEAU_FOR_FLAG = 20
OVER_SEGMENT_EXTRA = 2


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def boundary_indexes(segments: list[tuple[int, int, int]] | list[dict], key: str = "index") -> np.ndarray:
    if not segments:
        return np.zeros(0, dtype=float)
    if isinstance(segments[0], dict):
        cuts = [segment["index_start"] for segment in segments[1:]]
    else:
        cuts = [start for start, _end, _mode in segments[1:]]
    return np.asarray(cuts, dtype=float)


def symmetric_boundary_distance(true_cuts: np.ndarray, pred_cuts: np.ndarray, span: float) -> float:
    """Mean nearest-neighbor distance. Empty versus nonempty uses ``span``."""
    if true_cuts.size == 0 and pred_cuts.size == 0:
        return 0.0
    if true_cuts.size == 0 or pred_cuts.size == 0:
        return float(span)

    def one_way(source: np.ndarray, target: np.ndarray) -> float:
        distances = [float(np.min(np.abs(target - value))) for value in source]
        return float(np.mean(distances))

    return 0.5 * (one_way(true_cuts, pred_cuts) + one_way(pred_cuts, true_cuts))


def evaluate_profile(
    truth: np.ndarray,
    predicted_modes: np.ndarray,
    true_segments: list[dict],
    pred_segments: list[tuple[int, int, int]],
    omega: np.ndarray,
    model_mismatch: bool,
) -> dict:
    """``truth`` uses 0 plateau, 1 distinguishable slope, 2 unresolved, -1 unlabeled."""
    predicted_constant = predicted_modes == 0
    plateau = truth == 0
    distinguishable = truth == 1
    n = int(truth.size)
    tp = int(np.sum(predicted_constant & plateau))
    fp = int(np.sum(predicted_constant & distinguishable))
    fn = int(np.sum((~predicted_constant) & plateau))
    n_plateau = int(np.sum(plateau))
    n_distinguishable = int(np.sum(distinguishable))

    if tp + fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)
    if n_plateau == 0:
        recall = 1.0
    else:
        recall = tp / n_plateau
    f1 = _safe_f1(precision, recall)
    false_constant_coverage = fp / n
    missed_plateau_coverage = 0.0 if n_plateau == 0 else fn / n_plateau

    labeled = truth >= 0
    comparable = labeled & (truth != 2)
    if int(np.sum(comparable)) == 0:
        type_accuracy = math.nan
    else:
        expected_constant = truth == 0
        type_match = predicted_constant == expected_constant
        type_accuracy = float(np.mean(type_match[comparable]))

    true_n = 0 if model_mismatch else len(true_segments)
    pred_n = len(pred_segments)
    count_error = math.nan if model_mismatch else float(pred_n - true_n)

    if model_mismatch:
        boundary_index = math.nan
        boundary_omega = math.nan
    else:
        true_cuts = boundary_indexes(true_segments)
        pred_cuts = boundary_indexes(pred_segments)
        boundary_index = symmetric_boundary_distance(true_cuts, pred_cuts, float(n))
        true_omega = omega[true_cuts.astype(int)] if true_cuts.size else np.zeros(0)
        pred_omega = omega[pred_cuts.astype(int)] if pred_cuts.size else np.zeros(0)
        boundary_omega = symmetric_boundary_distance(
            true_omega, pred_omega, float(omega[-1] - omega[0])
        )

    if model_mismatch or not math.isfinite(type_accuracy):
        score = math.nan
    else:
        count_penalty = min(1.0, abs(count_error) / max(COUNT_SCALE * true_n, 1.0))
        boundary_penalty = min(1.0, boundary_index / BOUNDARY_SCALE_OSCILLATORS)
        score = (
            WEIGHT_F1 * f1
            + WEIGHT_FALSE_CONSTANT * (1.0 - false_constant_coverage)
            + WEIGHT_MISSED_PLATEAU * (1.0 - missed_plateau_coverage)
            + WEIGHT_TYPE * type_accuracy
            + WEIGHT_COUNT * (1.0 - count_penalty)
            + WEIGHT_BOUNDARY * (1.0 - boundary_penalty)
        )

    single_slope = (
        (not model_mismatch)
        and n_plateau == 0
        and n_distinguishable == n
        and len(true_segments) == 1
    )
    return {
        "constant_precision": precision,
        "constant_recall": recall,
        "constant_f1": f1,
        "false_constant_coverage": false_constant_coverage,
        "missed_plateau_coverage": missed_plateau_coverage,
        "type_accuracy": type_accuracy,
        "segment_count_error": count_error,
        "boundary_mean_abs_index": boundary_index,
        "boundary_mean_abs_omega": boundary_omega,
        "score": score,
        "n_plateau": n_plateau,
        "n_distinguishable": n_distinguishable,
        "over_segmentation": int((not model_mismatch) and pred_n >= true_n + OVER_SEGMENT_EXTRA),
        "under_segmentation": int((not model_mismatch) and pred_n <= true_n - 1 and true_n > 0),
        "false_constant_flag": int(false_constant_coverage >= FALSE_CONSTANT_FLAG),
        "missed_plateau_flag": int(n_plateau >= MIN_PLATEAU_FOR_FLAG and missed_plateau_coverage >= MISSED_PLATEAU_FLAG),
        "shallow_as_constant_flag": int(single_slope and float(np.mean(predicted_constant)) >= 0.5),
        "plateau_as_linear_flag": int(n_plateau >= MIN_PLATEAU_FOR_FLAG and missed_plateau_coverage >= MISSED_PLATEAU_FLAG),
    }
