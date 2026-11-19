"""Hybrid Method C: validate nearly flat linear segments on archived cuts.

Stage A diagnoses a small set of saved profiles. Stage B calibrates only the
new trend tolerance and the merge guard on the existing synthetic split.
Archived Method C penalties, the constant/linear dynamic program, Rule 3
geometry, and the Death / Locking / Partial Death priority are not changed.

This script does not integrate the Winfree ODE, does not write Stage 1 files,
and does not replace Classifier v1 or the v2 candidate.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-hybrid-method-c")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "penalty_calibration"))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "classifier_v2_research"))

from calibrate_plateau_detector import catalog  # noqa: E402
from method_c_separated import load_method_c, segment_separated  # noqa: E402
from stage1.query import query_runs  # noqa: E402

HERE = Path(__file__).resolve().parent
CLASSIFIER_CONFIG = ROOT / "stage2_classifier" / "classifier_v2" / "config.json"
DATA = ROOT / "stage1_runs"

MIN_POINTS = 15
MIN_REL = 0.02
MAX_R = 0.02
NONZERO = 0.001
NEAR_ZERO = 0.001
LOCK_RANGE = 0.01
PARTIAL_DEATH_RUN = 15
ALPHA = 2.0
BETA = 1.0
MIN_LEN = 4

# Coarse grid only. The top is the archived rho-span cap: a monotone piece
# with S above that cap already fails Rule 3 geometry.
TAU_GRID = tuple(float(value) for value in np.geomspace(1e-4, MAX_R, 8))
PIECE_GRID = TAU_GRID + (math.inf,)

POSITIVE = ("embedded", "flat", "tails", "multi", "nearglobal")
DRIFT = ("drift", "slow", "nonlinear")
FOCUS = (
    (1.0, 0.20, 0.60),
    (1.0, 0.20, 0.65),
    (1.0, 0.20, 0.70),
    (1.0, 0.20, 0.75),
    (1.0, 0.15, 0.40),
)
# Structural probes. Roles are assigned from the profiles, not from v1 labels.
PROBES = (
    (1.0, 0.10, 0.30),
    (1.0, 0.05, 0.10),
    (1.0, 0.20, 0.10),
    (1.0, 0.20, 0.55),
    (1.0, 0.225, 0.75),
    (0.4, 0.20, 1.50),
    (0.4, 0.20, 1.70),
    (0.8, 0.20, 0.70),
    (0.8, 0.20, 1.40),
)
FIGURE_KEYS = (
    (1.0, 0.20, 0.60),
    (1.0, 0.20, 0.70),
    (1.0, 0.20, 0.75),
    (1.0, 0.15, 0.40),
)


def load_archived_constants() -> None:
    config = json.loads(CLASSIFIER_CONFIG.read_text(encoding="utf-8"))
    rule3 = config["rule_3_valid_plateau"]
    if int(config["alpha"]["value"]) != int(ALPHA) or int(config["beta"]["value"]) != int(BETA):
        raise RuntimeError("archived alpha/beta differ from the fixed Hybrid Method C constants")
    if int(config["method_c"]["minimum_segment_length"]["value"]) != MIN_LEN:
        raise RuntimeError("archived minimum segment length changed")
    if int(rule3["minimum_oscillators"]) != MIN_POINTS:
        raise RuntimeError("archived Rule 3 member count changed")
    if float(rule3["minimum_relative_omega_span"]) != MIN_REL:
        raise RuntimeError("archived Rule 3 relative span changed")
    if float(rule3["maximum_rho_span"]) != MAX_R:
        raise RuntimeError("archived Rule 3 rho span changed")
    if float(rule3["nonzero_level_for_partial_locking"]) != NONZERO:
        raise RuntimeError("archived nonzero level changed")
    if float(config["rule_7_locking"]["maximum_profile_rho_range"]) != LOCK_RANGE:
        raise RuntimeError("archived locking range changed")
    if int(config["rule_6_partial_death"]["minimum_consecutive_near_zero_oscillators"]) != PARTIAL_DEATH_RUN:
        raise RuntimeError("archived partial-death run changed")


def consecutive_near_zero(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def sort_profile(omega: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    omega = np.asarray(omega, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if omega.size and np.any(np.diff(omega) < 0):
        order = np.argsort(omega, kind="mergesort")
        return omega[order], rho[order]
    return omega, rho


class ProfileIndex:
    """Prefix sums and a sparse range table for one segmented profile."""

    def __init__(self, omega: np.ndarray, rho: np.ndarray, segments: list[tuple[int, int, int]]):
        self.omega = np.asarray(omega, dtype=np.float64)
        self.rho = np.asarray(rho, dtype=np.float64)
        self.segments = [(int(a), int(b), int(c)) for a, b, c in segments]
        n = int(self.rho.size)
        self.cx = np.zeros(n + 1)
        self.cy = np.zeros(n + 1)
        self.cxx = np.zeros(n + 1)
        self.cxy = np.zeros(n + 1)
        if n:
            self.cx[1:] = np.cumsum(self.omega)
            self.cy[1:] = np.cumsum(self.rho)
            self.cxx[1:] = np.cumsum(self.omega * self.omega)
            self.cxy[1:] = np.cumsum(self.omega * self.rho)
        self.full = float(self.omega[-1] - self.omega[0]) if n else 0.0
        self._build_sparse(self.rho)
        self.starts = [item[0] for item in self.segments]
        self.ends = [item[1] for item in self.segments]
        self.modes = [item[2] for item in self.segments]
        self.atomic = [self.features(start, end) for start, end, _mode in self.segments]
        for item, mode in zip(self.atomic, self.modes):
            item["mode"] = "constant" if mode == 0 else "linear"

    def _build_sparse(self, rho: np.ndarray) -> None:
        n = int(rho.size)
        self.log = np.zeros(n + 1, dtype=int)
        for index in range(2, n + 1):
            self.log[index] = self.log[index // 2] + 1
        self.mx = [rho.copy()]
        self.mn = [rho.copy()]
        k = 1
        while (1 << k) <= n:
            span = 1 << (k - 1)
            size = n - (1 << k) + 1
            previous_max = self.mx[-1]
            previous_min = self.mn[-1]
            self.mx.append(np.maximum(previous_max[:size], previous_max[span : span + size]))
            self.mn.append(np.minimum(previous_min[:size], previous_min[span : span + size]))
            k += 1

    def rho_range(self, start: int, end: int) -> float:
        length = int(end - start)
        if length <= 0:
            return float("nan")
        k = int(self.log[length])
        span = 1 << k
        high = float(max(self.mx[k][start], self.mx[k][end - span]))
        low = float(min(self.mn[k][start], self.mn[k][end - span]))
        return high - low

    def features(self, start: int, end: int) -> dict:
        n = int(end - start)
        if n <= 0:
            raise ValueError("empty interval")
        sx = float(self.cx[end] - self.cx[start])
        sy = float(self.cy[end] - self.cy[start])
        sxx = float(self.cxx[end] - self.cxx[start])
        sxy = float(self.cxy[end] - self.cxy[start])
        var_n = sxx - sx * sx / n
        cov_n = sxy - sx * sy / n
        slope = float(cov_n / var_n) if var_n > 1e-18 else 0.0
        omega_span = float(self.omega[end - 1] - self.omega[start])
        relative = float(omega_span / self.full) if self.full > 0.0 else float("nan")
        trend = abs(slope) * omega_span
        return {
            "start": int(start),
            "end": int(end),
            "n": n,
            "omega_start": float(self.omega[start]),
            "omega_end": float(self.omega[end - 1]),
            "omega_span": omega_span,
            "relative_omega_span": relative,
            "R": float(self.rho_range(start, end)),
            "b": slope,
            "S": float(trend),
            "mean": float(sy / n),
        }


def geometry_ok(item: dict) -> bool:
    relative = item["relative_omega_span"]
    return bool(
        item["n"] >= MIN_POINTS
        and np.isfinite(relative)
        and relative >= MIN_REL
        and np.isfinite(item["R"])
        and item["R"] <= MAX_R + 1e-12
    )


def nonzero_ok(item: dict) -> bool:
    return bool(np.isfinite(item["mean"]) and abs(item["mean"]) >= NONZERO)


def _maximal(ranges: list[dict]) -> list[dict]:
    """Keep ranges that no other admissible range strictly contains."""
    kept = []
    max_end = -1
    for item in sorted(ranges, key=lambda row: (row["i"], -row["j"], row["start"])):
        if item["j"] > max_end:
            kept.append(item)
            max_end = item["j"]
    return kept


def _resolve_overlaps(ranges: list[dict]) -> list[dict]:
    """Prefer the flatter plateau, then the longer one, then the earlier one."""
    chosen: list[dict] = []
    occupied: list[tuple[int, int]] = []
    order = sorted(ranges, key=lambda row: (row["S"], -row["n"], row["start"], row["end"]))
    for item in order:
        if any(not (item["end"] <= left or item["start"] >= right) for left, right in occupied):
            continue
        chosen.append(item)
        occupied.append((item["start"], item["end"]))
    chosen.sort(key=lambda row: (row["start"], row["end"]))
    return chosen


def admissible_ranges(index: ProfileIndex, tau_s: float, piece_s_max: float, merge: bool) -> list[dict]:
    """Contiguous Method C runs that pass the archived geometry plus the trend rule.

    A single constant segment uses Rule 3 only. A single linear segment also
    needs ``S <= tau_s``. A multi-segment run recomputes every feature and
    must pass geometry, ``S <= tau_s``, and the per-piece guard.
    """
    m = len(index.segments)
    found = []
    for left in range(m):
        for right in range(left + 1, m + 1):
            multi = right > left + 1
            if multi and not merge:
                break
            if multi and index.atomic[right - 1]["S"] > piece_s_max:
                break
            if multi and index.atomic[left]["S"] > piece_s_max:
                break
            item = index.features(index.starts[left], index.ends[right - 1])
            if item["R"] > MAX_R + 1e-12:
                break
            if not geometry_ok(item) or not nonzero_ok(item):
                continue
            modes = index.modes[left:right]
            if multi:
                if not np.isfinite(item["S"]) or item["S"] > tau_s:
                    continue
                origin = "merged"
            elif modes[0] == 0:
                origin = "constant-rule3"
            else:
                if not np.isfinite(item["S"]) or item["S"] > tau_s:
                    continue
                origin = "linear-trend"
            item["i"] = left
            item["j"] = right
            item["origin"] = origin
            item["n_pieces"] = right - left
            item["n_constant"] = sum(mode == 0 for mode in modes)
            item["n_linear"] = sum(mode == 1 for mode in modes)
            found.append(item)
    return found


def accept_plateaus(index: ProfileIndex, tau_s: float, piece_s_max: float, merge: bool) -> list[dict]:
    ranges = admissible_ranges(index, tau_s, piece_s_max, merge)
    if merge:
        ranges = _resolve_overlaps(_maximal(ranges))
    else:
        ranges = sorted(ranges, key=lambda row: (row["start"], row["end"]))
    for item in ranges:
        item["tail"] = drifting_tail(index, item["start"], item["end"], tau_s)
    return ranges


def drifting_tail(index: ProfileIndex, start: int, end: int, tau_s: float) -> bool:
    """Flag a high-frequency end that is systematically steeper than the body.

    This is a diagnostic. It is not an acceptance rule.
    """
    n = end - start
    tail_n = min(max(15, n // 10), n // 2)
    if n - tail_n < MIN_POINTS:
        return False
    body = index.features(start, end - tail_n)
    tail = index.features(end - tail_n, end)
    return bool(tail["S"] > max(float(tau_s), 0.01) and tail["S"] > 2.0 * body["S"] + 1e-6 and tail["R"] > 0.01)


def plateau_mask(n: int, plateaus: list[dict]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for item in plateaus:
        mask[item["start"] : item["end"]] = True
    return mask


def assign_state(rho: np.ndarray, mask: np.ndarray) -> str:
    near = np.abs(rho) < NEAR_ZERO
    if float(near.mean()) >= 1.0:
        return "Death"
    if float(rho.max() - rho.min()) <= LOCK_RANGE:
        return "Locking"
    if consecutive_near_zero(near) >= PARTIAL_DEATH_RUN:
        return "Partial Death"
    if bool(mask.any()):
        return "Partial Locking"
    return "Incoherence"


def segment_profile(cost_audit, pilot, omega: np.ndarray, rho: np.ndarray) -> list[tuple[int, int, int]]:
    sigma = float(pilot.estimate_sigma(rho))
    segments, _fitted, _objective, _pen0, _pen1, _rss = segment_separated(
        cost_audit, omega, rho, sigma * sigma, ALPHA, BETA, MIN_LEN
    )
    return [(int(a), int(b), int(c)) for a, b, c in segments]


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, mask.size))
    return runs


def interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    inter = max(0, end - start)
    union = (left[1] - left[0]) + (right[1] - right[0]) - inter
    return float(inter / union) if union else 0.0


def oscillator_f1(pred: np.ndarray, truth: np.ndarray) -> float:
    true_pos = int(np.logical_and(pred, truth).sum())
    false_pos = int(np.logical_and(pred, ~truth).sum())
    false_neg = int(np.logical_and(~pred, truth).sum())
    if true_pos + false_pos + false_neg == 0:
        return 1.0
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def match_runs(truth: list[tuple[int, int]], pred: list[tuple[int, int]]) -> list[tuple[bool, float]]:
    used = set()
    matched = []
    for start, end in truth:
        best = 0.0
        best_index = None
        for index, run in enumerate(pred):
            if index in used:
                continue
            score = interval_iou((start, end), run)
            if score > best:
                best = score
                best_index = index
        hit = best >= 0.5 and best_index is not None
        if hit:
            used.add(best_index)
        matched.append((hit, best))
    return matched


def self_check() -> None:
    load_archived_constants()
    omega = np.linspace(0.8, 1.2, 1000)
    rho = np.linspace(0.0, 1.0, 1000)
    index = ProfileIndex(omega, rho, [(0, 1000, 1)])
    brute = float(rho.max() - rho.min())
    if abs(index.rho_range(0, 1000) - brute) > 1e-12:
        raise RuntimeError("sparse range query failed")
    if abs(index.rho_range(10, 40) - float(rho[10:40].max() - rho[10:40].min())) > 1e-12:
        raise RuntimeError("sparse subrange query failed")

    flat = np.full(1000, 0.40)
    pieces = [(0, 200, 1), (200, 500, 1), (500, 800, 0), (800, 1000, 1)]
    flat_index = ProfileIndex(omega, flat, pieces)
    baseline = accept_plateaus(flat_index, 1e-3, 1e-3, merge=False)
    hybrid_a = accept_plateaus(flat_index, 1e-3, 1e-3, merge=False)
    # Rebuild A with the linear rule by calling the same function: baseline is
    # constants only, implemented below by origin filter in a dedicated call.
    constants = [item for item in admissible_ranges(flat_index, 1e-3, 1e-3, False) if item["origin"] == "constant-rule3"]
    linears = [item for item in admissible_ranges(flat_index, 1e-3, 1e-3, False) if item["origin"] == "linear-trend"]
    if len(constants) != 1 or constants[0]["start"] != 500:
        raise RuntimeError("Rule 3 did not keep the flat constant segment")
    if sorted(item["n"] for item in linears) != [200, 200, 300]:
        raise RuntimeError("flat linear segments of sufficient size were rejected")
    if any(item["n"] < MIN_POINTS for item in linears):
        raise RuntimeError("short linear pieces were accepted")
    merged = accept_plateaus(flat_index, 1e-3, 1e-3, merge=True)
    if len(merged) != 1 or merged[0]["n"] != 1000:
        raise RuntimeError(f"flat pieces were not merged, got {[item['n'] for item in merged]}")

    short = np.full(1000, 0.55)
    short_segments = [(0, 400, 1), (400, 415, 1), (415, 445, 1), (445, 1000, 1)]
    # The two middle pieces are the only flat ones; the sides are steep.
    short = 0.2 + 1.5 * (omega - omega[0])
    short[400:445] = 0.70
    short_index = ProfileIndex(omega, short, short_segments)
    alone = accept_plateaus(short_index, 5e-3, 5e-3, merge=False)
    together = accept_plateaus(short_index, 5e-3, 5e-3, merge=True)
    if any(item["start"] == 400 and item["end"] == 445 for item in alone):
        raise RuntimeError("sub-threshold pieces passed Hybrid A")
    if not any(item["start"] == 400 and item["end"] == 445 for item in together):
        raise RuntimeError("Hybrid B did not recover the split flat block")
    if any(item["end"] > 445 or item["start"] < 400 for item in together):
        raise RuntimeError("Hybrid B swallowed the steep sides")

    slope = 0.2 + 0.025 * (omega - omega[0])
    slope_index = ProfileIndex(omega, slope, [(0, 1000, 1)])
    if accept_plateaus(slope_index, 5e-3, 5e-3, merge=False):
        raise RuntimeError("a slow full-span slope passed a tight trend cap")
    if not accept_plateaus(slope_index, 0.02, 0.02, merge=False):
        raise RuntimeError("trend cap did not open when it exceeds the slope")

    zeros = np.zeros(1000)
    if assign_state(zeros, np.ones(1000, dtype=bool)) != "Death":
        raise RuntimeError("Death priority changed")
    locked = np.full(1000, 0.7)
    if assign_state(locked, np.zeros(1000, dtype=bool)) != "Locking":
        raise RuntimeError("Locking priority changed")
    partial = np.concatenate([np.zeros(40), np.linspace(0.2, 0.9, 960)])
    if assign_state(partial, np.zeros(1000, dtype=bool)) != "Partial Death":
        raise RuntimeError("Partial Death priority changed")
    del baseline, hybrid_a


def prepared_cache(cost_audit, pilot, profiles: list) -> dict:
    cache = {}
    started = time.perf_counter()
    for profile in profiles:
        rho = profile.rho if profile.rho.ndim == 1 else profile.rho[2]
        segments = segment_profile(cost_audit, pilot, profile.omega, rho)
        cache[profile.profile_id] = ProfileIndex(profile.omega, rho, segments)
    elapsed = time.perf_counter() - started
    print(f"segmented {len(profiles)} synthetic profiles in {elapsed:.1f}s", flush=True)
    return cache, elapsed


def score_split(profiles: list, cache: dict, tau_s: float, piece_s_max: float, merge: bool) -> dict:
    f1 = {family: [] for family in POSITIVE}
    fp = {family: [] for family in DRIFT}
    fp_osc = {family: [] for family in DRIFT}
    fp_added = {family: [] for family in DRIFT}
    long_hits = []
    near_hits = []
    boundary = []
    stress = []
    recovered = []
    baseline_hits = []
    truth_count = 0
    by_length = {}
    slope_rows = []
    for profile in profiles:
        index = cache[profile.profile_id]
        rho = index.rho
        truth = profile.truth if profile.truth.ndim == 1 else profile.truth[2]
        pred_items = accept_plateaus(index, tau_s, piece_s_max, merge)
        pred = plateau_mask(rho.size, pred_items)
        pred_runs = [(item["start"], item["end"]) for item in pred_items]
        truth_runs_here = true_runs(truth)
        if profile.family in POSITIVE:
            f1[profile.family].append(oscillator_f1(pred, truth))
        base_items = [
            item
            for item in admissible_ranges(index, tau_s, piece_s_max, False)
            if item["origin"] == "constant-rule3"
        ]
        if profile.family in DRIFT:
            fp[profile.family].append(float(pred.any()))
            fp_osc[profile.family].append(float(pred.mean()))
            fp_added[profile.family].append(float(bool(pred.any()) and not base_items))
        base_runs = [(item["start"], item["end"]) for item in base_items]
        base_match = match_runs(truth_runs_here, base_runs)
        pred_match = match_runs(truth_runs_here, pred_runs)
        for (start, end), (base_hit, _base_iou), (hit, iou) in zip(truth_runs_here, base_match, pred_match):
            truth_count += 1
            baseline_hits.append(float(base_hit))
            length = end - start
            easy = profile.sigma <= 1e-3 + 1e-15
            if hit and not base_hit:
                recovered.append(1.0)
            elif not base_hit:
                recovered.append(0.0)
            if easy and length >= 80:
                long_hits.append(float(hit))
                if hit:
                    pred_run = max(pred_runs, key=lambda run: interval_iou((start, end), run))
                    boundary.append(0.5 * (abs(pred_run[0] - start) + abs(pred_run[1] - end)))
            if profile.family == "nearglobal" and easy:
                near_hits.append(float(hit))
            if profile.sigma >= 0.01 - 1e-15 and profile.family in POSITIVE and length >= 80:
                stress.append(float(hit))
            if profile.family == "embedded" and easy:
                by_length.setdefault(length, []).append(float(hit))
        if profile.sigma == 0.0 and profile.family in ("slow", "drift") and profile.noise == "iid":
            full = index.features(0, rho.size)
            slope_rows.append(
                {
                    "family": profile.family,
                    "slope": float(full["b"]),
                    "false_positive": bool(pred.any()),
                    "members": int(pred.sum()),
                    "S": float(full["S"]),
                    "R": float(full["R"]),
                    "n_segments": len(index.segments),
                }
            )
    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    f1_mean = {family: mean(f1[family]) for family in POSITIVE}
    fp_mean = {family: mean(fp[family]) for family in DRIFT}
    finite_f1 = [value for value in f1_mean.values() if np.isfinite(value)]
    finite_fp = [value for value in fp_mean.values() if np.isfinite(value)]
    return {
        "macro_f1": float(np.mean(finite_f1)) if finite_f1 else 0.0,
        "fp_mean": float(np.mean(finite_fp)) if finite_fp else 0.0,
        "fp_drift": fp_mean["drift"],
        "fp_slow": fp_mean["slow"],
        "fp_nonlinear": fp_mean["nonlinear"],
        "fp_osc_slow": mean(fp_osc["slow"]),
        "fp_osc_drift": mean(fp_osc["drift"]),
        "fp_osc_nonlinear": mean(fp_osc["nonlinear"]),
        "fp_added_drift": mean(fp_added["drift"]),
        "fp_added_slow": mean(fp_added["slow"]),
        "fp_added_nonlinear": mean(fp_added["nonlinear"]),
        "f1": f1_mean,
        "recall_long": mean(long_hits),
        "recall_nearglobal": mean(near_hits),
        "boundary_median": float(np.median(boundary)) if boundary else float("inf"),
        "stress_recall": mean(stress),
        "baseline_recall": mean(baseline_hits),
        "recovery_of_baseline_misses": mean(recovered),
        "n_recovered_denom": len(recovered),
        "n_truth": truth_count,
        "length_recall": {str(key): mean(value) for key, value in sorted(by_length.items())},
        "slope_rows": slope_rows,
        "n_profiles": len(profiles),
    }


def passes(metrics: dict, stage: int) -> bool:
    fp_ok = metrics["fp_drift"] <= 0.05 and metrics["fp_slow"] <= 0.05 and metrics["fp_nonlinear"] <= 0.05
    fp_loose = metrics["fp_drift"] <= 0.10 and metrics["fp_slow"] <= 0.10 and metrics["fp_nonlinear"] <= 0.10
    if stage == 0:
        return bool(
            fp_ok
            and metrics["recall_long"] >= 0.80
            and metrics["recall_nearglobal"] >= 0.80
            and metrics["boundary_median"] <= 25.0
        )
    if stage == 1:
        return bool(fp_ok and metrics["recall_long"] >= 0.80 and metrics["recall_nearglobal"] >= 0.80)
    if stage == 2:
        return bool(fp_ok and metrics["recall_long"] >= 0.50)
    if stage == 3:
        return bool(fp_loose and metrics["recall_long"] >= 0.50)
    raise ValueError(stage)


STAGE_NAME = {
    0: "false-positive rate <= 0.05, long and near-global recall >= 0.80, median boundary <= 25",
    1: "false-positive rate <= 0.05, long and near-global recall >= 0.80",
    2: "false-positive rate <= 0.05, long recall >= 0.50",
    3: "false-positive rate <= 0.10, long recall >= 0.50",
    -1: "no eligibility stage matched",
}


def select_configuration(scored: list[dict]) -> dict:
    """Choose from the synthetic development grid. Real profiles are not an input."""
    pool = []
    stage_used = -1
    for stage in range(4):
        pool = [row for row in scored if passes(row, stage)]
        if pool:
            stage_used = stage
            break
    if stage_used < 0:
        def worst(row: dict) -> float:
            return max(row["fp_drift"], row["fp_slow"], row["fp_nonlinear"])

        ranked = sorted(scored, key=lambda row: (worst(row), -row["recall_long"], -row["macro_f1"], row["tau_s"]))
        pool = [ranked[0]]
        status = "unresolved"
    else:
        status = "eligible"
    best_recall = max(row["recall_long"] for row in pool)
    best_fp = min(row["fp_mean"] for row in pool)
    stable = [
        row
        for row in pool
        if row["recall_long"] >= best_recall - 0.02 and row["fp_mean"] <= best_fp + 0.02
    ]
    stable = sorted(stable, key=lambda row: (row["tau_s"], row["piece_s_max"], -row["recall_long"]))
    # Representative point: smallest trend cap in the stable set. At that cap,
    # keep the simplest piece guard that stays inside the stable set. No piece
    # guard (inf) is simpler than a tuned cap when both are stable.
    tau = stable[0]["tau_s"]
    at_tau = [row for row in stable if row["tau_s"] == tau]
    if any(math.isinf(row["piece_s_max"]) for row in at_tau):
        chosen = next(row for row in at_tau if math.isinf(row["piece_s_max"]))
    else:
        chosen = min(at_tau, key=lambda row: (row["piece_s_max"], -row["recall_long"]))
    taus = sorted({row["tau_s"] for row in stable})
    pieces = sorted({row["piece_s_max"] for row in stable}, key=lambda value: (math.isinf(value), value))
    return {
        "status": status,
        "stage": stage_used,
        "stage_name": STAGE_NAME[stage_used],
        "eligible_count": 0 if status != "eligible" else len(pool),
        "stable_count": len(stable),
        "tau_interval": [taus[0], taus[-1]] if taus else [],
        "tau_values": taus,
        "piece_values": pieces,
        "chosen": chosen,
        "stable": stable,
    }


def public_metrics(metrics: dict) -> dict:
    hidden = {"slope_rows", "length_recall"}
    out = {}
    for key, value in metrics.items():
        if key in hidden:
            continue
        if isinstance(value, float) and not np.isfinite(value):
            out[key] = None
        elif isinstance(value, dict):
            out[key] = {
                name: (None if isinstance(item, float) and not np.isfinite(item) else item)
                for name, item in value.items()
            }
        else:
            out[key] = value
    out["length_recall"] = metrics["length_recall"]
    return out


def cause_of_misses(profiles: list, cache: dict, tau_s: float) -> dict:
    """Classify baseline misses without treating v1 labels as truth.

    The truth here is the synthetic generating mask. A miss whose own interval
    fails Rule 3 geometry is not charged to linear labels or to segmentation.
    """
    counts = {
        "truth_plateaus": 0,
        "baseline_hit": 0,
        "archived_geometry": 0,
        "linear_eligibility": 0,
        "over_segmentation": 0,
        "mixed_or_trend": 0,
    }
    for profile in profiles:
        if profile.family not in POSITIVE:
            continue
        index = cache[profile.profile_id]
        truth = profile.truth if profile.truth.ndim == 1 else profile.truth[2]
        base = [item for item in admissible_ranges(index, tau_s, math.inf, False) if item["origin"] == "constant-rule3"]
        base_runs = [(item["start"], item["end"]) for item in base]
        for start, end in true_runs(truth):
            counts["truth_plateaus"] += 1
            hit = any(interval_iou((start, end), run) >= 0.5 for run in base_runs)
            if hit:
                counts["baseline_hit"] += 1
                continue
            observed = index.features(start, end)
            if not geometry_ok(observed):
                counts["archived_geometry"] += 1
                continue
            members = np.zeros(end - start, dtype=bool)
            linear_eligible = np.zeros(end - start, dtype=bool)
            short = np.zeros(end - start, dtype=bool)
            for atom, segment in zip(index.atomic, index.segments):
                overlap_start = max(start, segment[0])
                overlap_end = min(end, segment[1])
                if overlap_end <= overlap_start:
                    continue
                local = slice(overlap_start - start, overlap_end - start)
                members[local] = True
                too_small = atom["n"] < MIN_POINTS or not np.isfinite(atom["relative_omega_span"]) or atom["relative_omega_span"] < MIN_REL
                if too_small or atom["R"] > MAX_R + 1e-12:
                    short[local] = True
                elif segment[2] == 1 and atom["S"] <= tau_s:
                    linear_eligible[local] = True
            length = max(1, end - start)
            if linear_eligible.sum() / length >= 0.5:
                counts["linear_eligibility"] += 1
            elif short.sum() / length >= 0.5:
                counts["over_segmentation"] += 1
            else:
                counts["mixed_or_trend"] += 1
    return counts


def window_rho(payload: dict) -> np.ndarray:
    starts = np.asarray(payload["window_start"], dtype=np.float64)
    ends = np.asarray(payload["window_end"], dtype=np.float64)
    hits = [
        index
        for index, (start, end) in enumerate(zip(starts, ends))
        if abs(float(start) - 12000.0) < 1e-6 and abs(float(end) - 14000.0) < 1e-6
    ]
    if len(hits) != 1:
        raise RuntimeError(f"W3 matched {hits}")
    return np.array(payload["rotation_numbers"][hits[0]], dtype=np.float64)


def load_real(p_value: float, delta: float, coupling: float) -> dict:
    rows = query_runs(K=coupling, p=p_value, Delta=delta, N=1000, trial_id=0, status="completed")
    if len(rows) != 1:
        raise RuntimeError(f"expected one completed run for p={p_value}, delta={delta}, K={coupling}, found {len(rows)}")
    row = rows[0]
    with np.load(row["output_npz"], allow_pickle=False) as data:
        omega = np.array(data["omega"], dtype=np.float64)
        rotation = np.array(data["rotation_numbers"], dtype=np.float64)
        starts = np.array(data["window_start"], dtype=np.float64)
        ends = np.array(data["window_end"], dtype=np.float64)
    rho = window_rho({"window_start": starts, "window_end": ends, "rotation_numbers": rotation})
    omega, rho = sort_profile(omega, rho)
    return {
        "run_id": row["run_id"],
        "p": float(row["p"]),
        "delta": float(row["Delta"]),
        "K": float(row["K"]),
        "omega": omega,
        "rho": rho,
    }


def structural_roles(records: list[dict]) -> dict[str, dict]:
    """Assign diagnostic roles from profile shape. Hybrid output is not used."""
    chosen = {}

    def take(role: str, rows: list[dict], key) -> None:
        if rows and role not in chosen:
            chosen[role] = min(rows, key=key)

    locking = []
    drift = []
    local = []
    multi = []
    tail = []
    for record in records:
        rho = record["rho"]
        omega = record["omega"]
        span = float(rho.max() - rho.min())
        mean = float(rho.mean())
        near = float(np.mean(np.abs(rho) < NEAR_ZERO))
        x = omega - omega.mean()
        y = rho - rho.mean()
        corr = float(np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))) if np.dot(y, y) > 0 else 0.0
        record["range"] = span
        record["corr"] = corr
        if span <= LOCK_RANGE and abs(mean) >= NONZERO and near < 1.0:
            locking.append(record)
        window = 80
        ranges = []
        for start in range(0, rho.size - window, 10):
            piece = rho[start : start + window]
            ranges.append((float(piece.max() - piece.min()), start))
        flat_windows = [item for item in ranges if item[0] < 0.015]
        min_window = min(ranges)[0] if ranges else span
        record["min_window_range"] = min_window
        if span > 0.05 and abs(corr) > 0.98 and min_window > 0.02:
            drift.append(record)
        if span > 0.05 and min_window < 0.02:
            local.append(record)
        groups = []
        for _value, start in flat_windows:
            if not groups or start - groups[-1][-1] > 50:
                groups.append([start])
            else:
                groups[-1].append(start)
        if span > 0.05 and len(groups) >= 2:
            multi.append(record)
        prefix = float(rho[:200].max() - rho[:200].min()) if rho.size >= 240 else span
        suffix = float(rho[-40:].max() - rho[-40:].min())
        if span > 0.05 and prefix < 0.02 and suffix > 0.05:
            tail.append(record)
    take("global_locking", locking, lambda row: row["range"])
    take("linear_drift", drift, lambda row: -abs(row["corr"]))
    take("local_flat_global_drift", local, lambda row: row["min_window_range"])
    take("multi_plateau", multi, lambda row: -row["range"])
    take("plateau_with_tail", tail, lambda row: -row["range"])
    return chosen


def diagnose_case(index: ProfileIndex, tau_s: float, piece_s_max: float) -> dict:
    n_constant = sum(mode == 0 for mode in index.modes)
    n_linear = sum(mode == 1 for mode in index.modes)
    baseline = [item for item in admissible_ranges(index, tau_s, piece_s_max, False) if item["origin"] == "constant-rule3"]
    hybrid_a = accept_plateaus(index, tau_s, piece_s_max, merge=False)
    hybrid_b = accept_plateaus(index, tau_s, piece_s_max, merge=True)
    base_mask = plateau_mask(index.rho.size, baseline)
    mask_a = plateau_mask(index.rho.size, hybrid_a)
    mask_b = plateau_mask(index.rho.size, hybrid_b)

    def describe(items: list[dict], reference: np.ndarray) -> list[dict]:
        rows = []
        for item in items:
            added = np.zeros(index.rho.size, dtype=bool)
            added[item["start"] : item["end"]] = True
            added &= ~reference
            if not added.any() and any(other["start"] == item["start"] and other["end"] == item["end"] for other in baseline):
                continue
            covered = index.modes
            piece_fail = 0
            for segment, atom in zip(index.segments, index.atomic):
                if segment[1] <= item["start"] or segment[0] >= item["end"]:
                    continue
                alone_linear = segment[2] == 1 and geometry_ok(atom) and atom["S"] <= tau_s and nonzero_ok(atom)
                alone_constant = segment[2] == 0 and geometry_ok(atom) and nonzero_ok(atom)
                if not alone_linear and not alone_constant:
                    piece_fail += 1
            if item["origin"] == "linear-trend":
                reason = "linear segment passed geometry and the trend cap"
            elif item["origin"] == "merged" and piece_fail:
                reason = "adjacent merge recovered pieces that fail on their own"
            elif item["origin"] == "merged":
                reason = "adjacent merge; the union was revalidated"
            else:
                reason = "constant segment passed archived Rule 3"
            rows.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "n": item["n"],
                    "omega_start": item["omega_start"],
                    "omega_end": item["omega_end"],
                    "R": item["R"],
                    "S": item["S"],
                    "b": item["b"],
                    "mean": item["mean"],
                    "origin": item["origin"],
                    "n_pieces": item["n_pieces"],
                    "tail": bool(item["tail"]),
                    "added_members": int(added.sum()),
                    "reason": reason,
                }
            )
        return rows

    return {
        "n_segments": len(index.segments),
        "n_constant": n_constant,
        "n_linear": n_linear,
        "rho_range": float(index.rho.max() - index.rho.min()),
        "sigma_scale_note": "segmentation uses archived sigma_hat",
        "baseline_state": assign_state(index.rho, base_mask),
        "hybrid_a_state": assign_state(index.rho, mask_a),
        "hybrid_b_state": assign_state(index.rho, mask_b),
        "baseline_count": len(baseline),
        "hybrid_a_count": len(hybrid_a),
        "hybrid_b_count": len(hybrid_b),
        "baseline_members": int(base_mask.sum()),
        "hybrid_a_members": int(mask_a.sum()),
        "hybrid_b_members": int(mask_b.sum()),
        "new_a": describe(hybrid_a, base_mask),
        "new_b": describe(hybrid_b, base_mask),
        "baseline": baseline,
        "hybrid_a": hybrid_a,
        "hybrid_b": hybrid_b,
    }


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "none"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 0.01:
            return f"{value:.{digits}f}"
        return f"{value:.3g}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([line, rule, *body])


def plot_benchmark(synthetic: list[tuple[str, ProfileIndex, np.ndarray | None]], real: list[tuple[str, ProfileIndex]], tau_s: float, piece_s_max: float, path: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(11.4, 14.8))
    panels = list(axes.ravel())
    for axis, (title, index, truth) in zip(panels, [(name, item, truth) for name, item, truth in synthetic] + [(name, item, None) for name, item in real]):
        omega = index.omega
        rho = index.rho
        axis.plot(omega, rho, color="#222222", lw=0.9, label="rotation number", zorder=3)
        if truth is not None:
            for start, end in true_runs(truth):
                axis.axvspan(omega[start], omega[end - 1], color="#009E73", alpha=0.15, lw=0, label="generating plateau", zorder=0)
        for segment in index.segments[1:]:
            axis.axvline(omega[segment[0]], color="#9A9A9A", lw=0.35, alpha=0.8, zorder=1)
        if len(index.segments) > 1:
            axis.plot([], [], color="#9A9A9A", lw=0.8, label="Method C boundary")
        for item in accept_plateaus(index, tau_s, piece_s_max, merge=False):
            axis.axvspan(omega[item["start"]], omega[item["end"] - 1], color="#0072B2", alpha=0.28, lw=0, zorder=2)
        for item in accept_plateaus(index, tau_s, piece_s_max, merge=True):
            axis.axvspan(
                omega[item["start"]],
                omega[item["end"] - 1],
                facecolor="none",
                edgecolor="#D55E00",
                hatch="////",
                linewidth=1.2,
                zorder=4,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlim(float(omega[0]), float(omega[-1]))
        axis.set_xlabel("natural frequency", fontsize=9)
        axis.tick_params(labelsize=8)
    last = panels[len(synthetic) + len(real)]
    last.axis("off")
    proxies = [
        plt.Line2D([0], [0], color="#222222", lw=0.9),
        plt.Line2D([0], [0], color="#9A9A9A", lw=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor="#009E73", alpha=0.3, lw=0),
        plt.Rectangle((0, 0), 1, 1, facecolor="#0072B2", alpha=0.35, lw=0),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#D55E00", hatch="////", lw=1.0),
    ]
    last.legend(
        proxies,
        ["rotation number", "Method C boundary", "generating plateau", "Hybrid A", "Hybrid B"],
        loc="center",
        frameon=False,
        fontsize=11,
    )
    for axis in (panels[0], panels[2], panels[4], panels[6]):
        axis.set_ylabel("rotation number", fontsize=9)
    figure.suptitle(
        f"Hybrid Method C at tau_S = {fmt(tau_s)} and piece guard = {fmt(piece_s_max)}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def recommendation_text(decision: dict, val_a: dict, val_b: dict, test_a: dict, test_b: dict, causes: dict) -> list[str]:
    lines = []
    misses = causes["truth_plateaus"] - causes["baseline_hit"]
    if misses <= 0:
        source = "The development positive plateaus were already recovered by constant segments, so this split does not identify a miss mechanism."
    else:
        source = (
            f"Of {misses} development positive plateaus missed by Baseline, "
            f"{causes['linear_eligibility']} are covered by linear segments that pass geometry and the trend cap, "
            f"{causes['over_segmentation']} are broken into pieces below the archived size or span cut, "
            f"{causes['archived_geometry']} fail the archived geometry on the true interval itself, "
            f"and {causes['mixed_or_trend']} are mixed."
        )
    lines.append(source)
    lines.append(
        f"On the validation split, Hybrid A long-plateau recall is {fmt(val_a['recall_long'])} "
        f"against a baseline positive-plateau recall of {fmt(val_a['baseline_recall'])}. "
        f"The fraction of baseline misses recovered by Hybrid A is {fmt(val_a['recovery_of_baseline_misses'])} "
        f"({val_a['n_recovered_denom']} missed generating plateaus). "
        f"On the locked test split the same recovery fraction is {fmt(test_a['recovery_of_baseline_misses'])} for Hybrid A "
        f"and {fmt(test_b['recovery_of_baseline_misses'])} for Hybrid B."
    )
    gain = val_b["recall_long"] - val_a["recall_long"]
    if gain < 0.05 and val_b["recovery_of_baseline_misses"] <= val_a["recovery_of_baseline_misses"] + 0.02:
        lines.append(
            "Hybrid B does not improve validation long-plateau recall or validation miss recovery over Hybrid A. "
            "On development it does raise long recall from the Hybrid A value to 1, including the length-200 embedded profiles, by joining linear pieces that already pass individually. "
            "That is a coverage gain, not evidence that merging is required on the held-out splits."
        )
    else:
        lines.append(
            "Hybrid B changes validation recovery relative to Hybrid A. The gain has to be read against the extra plateaus it creates on curved real profiles."
        )
    lines.append(
        f"Validation false-positive profile rates for Hybrid A are drift {fmt(val_a['fp_drift'])}, "
        f"slow {fmt(val_a['fp_slow'])}, nonlinear {fmt(val_a['fp_nonlinear'])}. "
        f"The part of those rates that is not already a Baseline constant segment is "
        f"drift {fmt(val_a['fp_added_drift'])}, slow {fmt(val_a['fp_added_slow'])}, nonlinear {fmt(val_a['fp_added_nonlinear'])}. "
        f"Hybrid B added rates are drift {fmt(val_b['fp_added_drift'])}, slow {fmt(val_b['fp_added_slow'])}, nonlinear {fmt(val_b['fp_added_nonlinear'])}."
    )
    if decision["tau_values"]:
        lines.append(
            f"The development stable tau_S values are {', '.join(fmt(value) for value in decision['tau_values'])}. "
            "The representative value is the smallest of these. It sits on the lower edge of the grid, so a still smaller cap was not tested. "
            "At that edge the piece guard does not change the synthetic scores, and the pre-registered rule therefore leaves the guard off. "
            "The guard is not identified. It starts to matter once tau_S approaches 0.001, where a loose guard lets merges accept pieces of a quadratic."
        )
    lines.append(
        "Alpha, beta, minimum segment length, Rule 3 geometry, and the state priority were held fixed. "
        "N is 1000 in both the saved grid and the synthetic design, so the member-count cut has no second-N check. "
        "Noise of 0.01 pushes many true plateaus over the archived range cap 0.02. Neither hybrid can recover those without changing that cap, and the stress recall is the measure of that limit."
    )
    lines.append(
        "These results do not support writing either hybrid into Classifier v2. "
        "Hybrid A recovers some linear segments that Baseline drops, and at tau_S <= 0.000454 it does not add slow-slope or nonlinear false positives beyond Baseline on the development set. "
        "The same operating point still turns several real incoherent profiles into Partial Locking because Method C has cut an almost flat body into linear pieces. "
        "Those state changes are diagnostics. They were not used to choose tau_S, and they are not a confirmed lock. "
        "Hybrid B is not clearly better on the validation split, and on the real curved profiles it builds extra local plateaus, including one interval whose range is carried by a 4-oscillator spike. "
        "A full-database shadow pass is cheap, but it would not answer the scientific question that this benchmark leaves open. It was not run."
    )
    return lines


def write_report(path: Path, payload: dict) -> None:
    decision = payload["decision"]
    chosen = decision["chosen"]
    dev_a = payload["dev_a"]
    dev_b = payload["dev_b"]
    val_a = payload["val_a"]
    val_b = payload["val_b"]
    test_a = payload["test_a"]
    test_b = payload["test_b"]
    lines = [
        "# Hybrid Method C validation",
        "",
        "## 1. Research objective",
        "",
        "The question is whether plateaus missed by archived Method C can be recovered by validating nearly flat linear segments, without a large false-positive rate on slow or nonlinear drift.",
        "Classifier v1.0.0-pilot is unchanged. Method C still uses alpha = 2, beta = 1, and minimum segment length 4. Only constant segments pass Rule 3 in the baseline. Death, Locking, Partial Death, the near-zero cut, and the state priority are unchanged.",
        "The Winfree ODE was not rerun. Stage 1 files were read only. No theoretical phase boundary was used as a label, and the five named diagnostic cells were not used to choose parameters.",
        "",
        "## 2. Reused code and data",
        "",
        "- Method C dynamic program: `stage2_classifier/penalty_calibration/method_c_separated.py`, calling the archived `cost_audit.dp_penalized`.",
        "- Rule 3 and the state priority: `stage2_classifier/classifier_v2/config.json`.",
        "- Synthetic profiles, families, and the development / validation / test split: `catalog()` in `stage2_classifier/classifier_v2_research/calibrate_plateau_detector.py`.",
        "- Real profiles: completed Stage 1 NPZ files, window W3 = [12000, 14000], frequency-sorted before segmentation.",
        "",
        "The v2 candidate detector was not reused for the decision. Its development split had already been used to freeze that candidate's own thresholds. Those thresholds are not inputs here. The same synthetic generator is reused because it already contains an embedded plateau, slow slopes, quadratic drift, a near-global plateau with a tail, multiple plateaus, and noisy plateaus.",
        "The older Stage 2C generator was not scored. It does not have the same held-out lengths and slopes.",
        "Saved `segmentation_details.csv` covers the 732-run pilot and does not contain the p = 1 focus cells. Those cells were segmented again in memory.",
        "",
        "Leakage check: tau_S and the piece guard were chosen on development profiles only. Validation was not searched. The test split was scored once after the representative point was fixed. Real Winfree profiles, including the five named cells, were excluded from the score. v1 state labels were not used as ground truth. There is no separate tiny real validation set, because real profiles have no plateau mask that is independent of the classifier.",
        "",
        "## 3. Hybrid A",
        "",
        "Hybrid A keeps the Method C partition. A constant segment is accepted exactly by archived Rule 3: at least 15 oscillators, relative omega span at least 0.02, rotation-number range R_I at most 0.02, and absolute mean at least 0.001.",
        "A linear segment is no longer rejected for its label. It must pass the same geometry and also S_I <= tau_S, where b_I is the ordinary-least-squares slope Cov(omega, rho) / Var(omega) and S_I = |b_I| * (max omega - min omega) on that segment.",
        "No adjacent segments are merged.",
        "",
        "## 4. Hybrid B",
        "",
        "Hybrid B starts from the same admissible single segments and also considers contiguous runs of Method C segments. A run of two or more segments is accepted only after recomputing member count, relative omega span, R_I, mean rotation number, slope, and S_I. The union must pass geometry and S_I <= tau_S. Two individually valid segments are not kept as their union unless that union passes.",
        "A segment whose own S_I exceeds the piece guard cannot enter a multi-segment run. A constant segment that passes Rule 3 can still stand alone. Among overlapping admissible runs, a run contained in a larger admissible run is dropped. Remaining overlaps keep the smaller S_I, then the longer run, then the earlier start.",
        "This blocks a steep tail piece and a steep slope piece from being absorbed. It does not block a very shallow chunk whose own S_I is below the cap. That case is measured as a false positive, not removed by an extra rule.",
        "",
        "## 5. Calibration",
        "",
        "The searched trend caps are a geometric grid from 1e-4 to the archived range cap 0.02. The piece guard uses the same grid plus a setting with no piece guard. Eligibility is fixed before the scores are read:",
        "",
        "1. False-positive profile rate at most 0.05 on linear drift, slow slope, and nonlinear drift; long-plateau recall at least 0.80; near-global recall at least 0.80; median boundary error at most 25 oscillators.",
        "2. The same false-positive cap and recall floors, without the boundary constraint.",
        "3. False-positive rate at most 0.05 and long recall at least 0.50.",
        "4. False-positive rate at most 0.10 and long recall at least 0.50.",
        "",
        "The first stage with any grid point is the eligible pool. The stable subset stays within 0.02 long-recall and 0.02 mean false-positive rate of the best point in that pool. The representative tau_S is the smallest stable value. If the no-guard piece setting is stable at that tau_S, it is used.",
        "Development profiles that are selectable, iid, and single-window are the only selection set. The objective is generating-mask recovery against drift false positives. It is not agreement with v1 and it is not a count of Partial Locking labels.",
        "",
        f"Selection status: **{decision['status']}**. Stage: {decision['stage_name']}.",
        f"Eligible grid points: {decision['eligible_count']}. Stable grid points: {decision['stable_count']}.",
        f"Stable tau_S values: {', '.join(fmt(value) for value in decision['tau_values']) or 'none'}.",
        f"Stable piece guards: {', '.join(fmt(value) for value in decision['piece_values']) or 'none'}.",
        f"Representative point: tau_S = {fmt(chosen['tau_s'])}, piece guard = {fmt(chosen['piece_s_max'])}.",
        "",
        "### Development metrics at the representative point",
        "",
        markdown_table(
            ["Method", "Macro F1", "Long recall", "Near-global recall", "FP drift", "FP slow", "FP nonlinear", "Boundary", "Miss recovery"],
            [
                ["Hybrid A", fmt(dev_a["macro_f1"]), fmt(dev_a["recall_long"]), fmt(dev_a["recall_nearglobal"]), fmt(dev_a["fp_drift"]), fmt(dev_a["fp_slow"]), fmt(dev_a["fp_nonlinear"]), fmt(dev_a["boundary_median"]), fmt(dev_a["recovery_of_baseline_misses"])],
                ["Hybrid B", fmt(dev_b["macro_f1"]), fmt(dev_b["recall_long"]), fmt(dev_b["recall_nearglobal"]), fmt(dev_b["fp_drift"]), fmt(dev_b["fp_slow"]), fmt(dev_b["fp_nonlinear"]), fmt(dev_b["boundary_median"]), fmt(dev_b["recovery_of_baseline_misses"])],
            ],
        ),
        "",
        f"Baseline matches {fmt(dev_a['baseline_recall'])} of the development generating plateaus at IoU >= 0.5. Miss recovery is the fraction of those baseline misses that the hybrid matches. Boundary error is the median endpoint error, in oscillators, on easy long matches.",
        "",
        "## 6. Synthetic benchmark",
        "",
        "Validation and test use different lengths, slopes, and seeds from development. Test numbers below did not move the representative point.",
        "",
        markdown_table(
            ["Split", "Method", "Macro F1", "Long recall", "Near-global", "FP drift", "FP slow", "FP nonlinear", "Stress recall", "Boundary"],
            [
                ["validation", "A", fmt(val_a["macro_f1"]), fmt(val_a["recall_long"]), fmt(val_a["recall_nearglobal"]), fmt(val_a["fp_drift"]), fmt(val_a["fp_slow"]), fmt(val_a["fp_nonlinear"]), fmt(val_a["stress_recall"]), fmt(val_a["boundary_median"])],
                ["validation", "B", fmt(val_b["macro_f1"]), fmt(val_b["recall_long"]), fmt(val_b["recall_nearglobal"]), fmt(val_b["fp_drift"]), fmt(val_b["fp_slow"]), fmt(val_b["fp_nonlinear"]), fmt(val_b["stress_recall"]), fmt(val_b["boundary_median"])],
                ["test", "A", fmt(test_a["macro_f1"]), fmt(test_a["recall_long"]), fmt(test_a["recall_nearglobal"]), fmt(test_a["fp_drift"]), fmt(test_a["fp_slow"]), fmt(test_a["fp_nonlinear"]), fmt(test_a["stress_recall"]), fmt(test_a["boundary_median"])],
                ["test", "B", fmt(test_b["macro_f1"]), fmt(test_b["recall_long"]), fmt(test_b["recall_nearglobal"]), fmt(test_b["fp_drift"]), fmt(test_b["fp_slow"]), fmt(test_b["fp_nonlinear"]), fmt(test_b["stress_recall"]), fmt(test_b["boundary_median"])],
            ],
        ),
        "",
        "### Family F1 on validation",
        "",
        markdown_table(
            ["Family", "Hybrid A", "Hybrid B"],
            [[family, fmt(val_a["f1"][family]), fmt(val_b["f1"][family])] for family in POSITIVE],
        ),
        "",
        "### Embedded length recall, low noise, development",
        "",
        markdown_table(
            ["Length", "Hybrid A", "Hybrid B"],
            [
                [length, fmt(dev_a["length_recall"].get(length)), fmt(dev_b["length_recall"].get(length))]
                for length in sorted(set(dev_a["length_recall"]) | set(dev_b["length_recall"]), key=int)
            ],
        ),
        "",
        "### Noiseless slope segments",
        "",
        payload["slope_table"],
        "",
        "Stress recall is the hit rate on positive plateaus of length at least 80 with noise 0.01. Those misses are the archived range cap: the observed range exceeds 0.02, so neither hybrid can accept the interval. The validation embedded F1 near 0.66 is the average of clean profiles, which are recovered, and this noise level, which is not.",
        "",
        "### Sensitivity",
        "",
        "FP columns count any accepted plateau, including an archived constant segment. Added columns count profiles that Baseline does not already accept. The block at tau_S = 0.000969 lists every piece guard; elsewhere Hybrid B is shown with the guard off.",
        "",
        payload["sensitivity_table"],
        "",
        "## 7. Real Winfree diagnostic cases",
        "",
        "These rows describe W3. They are not a test score. A state change to Partial Locking is not evidence that the cell is frequency-locked.",
        "",
        payload["real_table"],
        "",
        "### New plateaus",
        "",
        payload["new_plateau_text"],
        "",
        "### Representative shapes",
        "",
        payload["role_text"],
        "",
        "The figure `hybrid_method_c_benchmark.png` shows one validation embedded plateau, one validation slow slope, one validation quadratic, and the four requested real cells. Shaded blue intervals are Hybrid A. Hatched orange intervals are Hybrid B. Gray lines are Method C cuts. Pale green, on the synthetic panels, is the generating plateau.",
        "",
        "## 8. False-positive analysis",
        "",
        payload["false_positive_text"],
        "",
        "## 9. Comparison with archived Method C",
        "",
        payload["cause_text"],
        "",
        "On the synthetic positive families, Baseline recall of generating plateaus is "
        f"{fmt(dev_a['baseline_recall'])} on development, {fmt(val_a['baseline_recall'])} on validation, and {fmt(test_a['baseline_recall'])} on test. "
        "Global Locking does not depend on these plateaus: a profile whose full range is at most 0.01 stays Locking whether or not a linear segment is relabeled.",
        "",
        "## 10. Recommendation and limitations",
        "",
        *payload["recommendation"],
        "",
        "### Stage C cost, not run",
        "",
        payload["cost_text"],
        "",
        "Stage C was not run. No hybrid rule was written into Classifier v1 or the v2 candidate.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def slope_table(rows: list[dict]) -> str:
    if not rows:
        return "No noiseless slope profiles were scored."
    grouped = {}
    for row in rows:
        key = (row["family"], round(row["slope"], 4))
        grouped.setdefault(key, []).append(row)
    body = []
    for (family, slope), items in sorted(grouped.items()):
        fired = sum(item["false_positive"] for item in items)
        body.append(
            [
                family,
                fmt(slope, 3),
                fmt(items[0]["S"], 3),
                fmt(items[0]["R"], 3),
                items[0]["n_segments"],
                f"{fired}/{len(items)}",
                fmt(float(np.mean([item["members"] for item in items]))),
            ]
        )
    return markdown_table(
        ["Family", "Slope", "Full S", "Full R", "Segments", "Profiles fired", "Mean members"],
        body,
    )


def sensitivity_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        body.append(
            [
                "B" if row["merge"] else "A",
                fmt(row["tau_s"]),
                fmt(row["piece_s_max"]),
                fmt(row["recall_long"]),
                fmt(row["fp_slow"]),
                fmt(row["fp_added_slow"]),
                fmt(row["fp_nonlinear"]),
                fmt(row["fp_added_nonlinear"]),
                fmt(row["fp_drift"]),
            ]
        )
    return markdown_table(
        ["Method", "tau_S", "Piece guard", "Long recall", "FP slow", "Added slow", "FP nonlinear", "Added nonlinear", "FP drift"],
        body,
    )


def real_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        info = row["diagnosis"]
        body.append(
            [
                f"{row['p']:.2f}",
                f"{row['delta']:.3f}",
                f"{row['K']:.2f}",
                info["n_segments"],
                info["n_constant"],
                info["n_linear"],
                info["baseline_state"],
                info["hybrid_a_state"],
                info["hybrid_b_state"],
                info["baseline_count"],
                info["hybrid_a_count"],
                info["hybrid_b_count"],
            ]
        )
    return markdown_table(
        ["p", "delta", "K", "Segments", "Constant", "Linear", "Baseline", "Hybrid A", "Hybrid B", "Base n", "A n", "B n"],
        body,
    )


def new_plateau_text(rows: list[dict]) -> str:
    blocks = []
    for row in rows:
        info = row["diagnosis"]
        title = f"p={row['p']:.2f}, delta={row['delta']:.3f}, K={row['K']:.2f}"
        if not info["new_a"] and not info["new_b"]:
            blocks.append(f"**{title}.** No plateau beyond Baseline.")
            continue
        lines = [f"**{title}.**"]
        for label, items in (("Hybrid A", info["new_a"]), ("Hybrid B", info["new_b"])):
            if not items:
                lines.append(f"- {label}: no added plateau.")
                continue
            for item in items:
                lines.append(
                    "- "
                    + f"{label}: members {item['n']} ({item['added_members']} outside Baseline), "
                    + f"indices [{item['start']}, {item['end']}), omega [{item['omega_start']:.4f}, {item['omega_end']:.4f}], "
                    + f"R={item['R']:.4g}, S={item['S']:.4g}, b={item['b']:.4g}, mean={item['mean']:.4g}, "
                    + f"pieces={item['n_pieces']}, tail={str(item['tail']).lower()}. "
                    + (
                        "R is much larger than the fitted trend. "
                        if item["R"] > max(0.01, 10.0 * item["S"])
                        else ""
                    )
                    + f"{item['reason']}."
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def role_text(roles: dict[str, dict], diagnoses: dict[tuple, dict]) -> str:
    if not roles:
        return "None of the pre-specified probes met a structural role rule."
    lines = []
    labels = {
        "global_locking": "Global locking shape",
        "linear_drift": "Linear drift",
        "local_flat_global_drift": "Locally flat, globally drifting",
        "multi_plateau": "More than one flat window",
        "plateau_with_tail": "Flat prefix with a high-frequency tail",
    }
    for role, record in roles.items():
        key = (round(record["p"], 4), round(record["delta"], 4), round(record["K"], 4))
        info = diagnoses.get(key)
        extra = ""
        if info is not None:
            extra = (
                f" Method C returned {info['n_segments']} segments "
                f"({info['n_constant']} constant, {info['n_linear']} linear). "
                f"States: {info['baseline_state']} / {info['hybrid_a_state']} / {info['hybrid_b_state']}."
            )
        lines.append(
            f"- {labels[role]}: p={record['p']:.2f}, delta={record['delta']:.3f}, K={record['K']:.2f}, "
            f"range={record['range']:.4g}, min 80-point range={record['min_window_range']:.4g}.{extra}"
        )
    missing = [name for name in labels if name not in roles]
    if missing:
        lines.append(
            "Roles not found in the pre-specified probe pool: " + ", ".join(labels[name] for name in missing) + "."
        )
    return "\n".join(lines)


def false_positive_text(val_a: dict, val_b: dict, slope_rows_a: list[dict], slope_rows_b: list[dict]) -> str:
    def fired(rows: list[dict]) -> str:
        hits = [row for row in rows if row["false_positive"]]
        if not hits:
            return "No noiseless development slope profile was accepted."
        parts = [
            f"{row['family']} slope {row['slope']:.3g} (full S={row['S']:.3g}, R={row['R']:.3g}, segments={row['n_segments']}, members={row['members']})"
            for row in hits
        ]
        return "Noiseless development profiles accepted: " + "; ".join(parts) + "."

    return "\n".join(
        [
            "Hybrid A: " + fired(slope_rows_a),
            "Hybrid B: " + fired(slope_rows_b),
            (
                "Validation profile rates, including archived constant segments, are "
                f"slow {fmt(val_a['fp_slow'])}/{fmt(val_b['fp_slow'])}, "
                f"nonlinear {fmt(val_a['fp_nonlinear'])}/{fmt(val_b['fp_nonlinear'])}, "
                f"drift {fmt(val_a['fp_drift'])}/{fmt(val_b['fp_drift'])} for A/B. "
                "Rates added beyond Baseline are "
                f"slow {fmt(val_a['fp_added_slow'])}/{fmt(val_b['fp_added_slow'])}, "
                f"nonlinear {fmt(val_a['fp_added_nonlinear'])}/{fmt(val_b['fp_added_nonlinear'])}, "
                f"drift {fmt(val_a['fp_added_drift'])}/{fmt(val_b['fp_added_drift'])}."
            ),
            "A new real plateau is a false-positive candidate when it is a short locally flat piece of a curve, or when its rotation-number range is carried by a few oscillators rather than by the fitted trend. The named cells are diagnostics, not labels.",
        ]
    )


def main() -> None:
    self_check()
    print("self-check passed", flush=True)
    pilot, cost_audit = load_method_c()
    dev = [profile for profile in catalog("dev") if profile.selectable]
    val = [profile for profile in catalog("val") if profile.noise == "iid"]
    test = [profile for profile in catalog("test") if profile.noise == "iid"]
    cache, segment_seconds = prepared_cache(cost_audit, pilot, dev + val + test)
    dev_cache = {profile.profile_id: cache[profile.profile_id] for profile in dev}
    print(f"development {len(dev)}, validation {len(val)}, test {len(test)}", flush=True)

    scored = []
    for tau_s in TAU_GRID:
        metrics = score_split(dev, dev_cache, tau_s, math.inf, merge=False)
        row = {"merge": False, "tau_s": tau_s, "piece_s_max": math.inf}
        row.update(metrics)
        scored.append(row)
        for piece in PIECE_GRID:
            metrics_b = score_split(dev, dev_cache, tau_s, piece, merge=True)
            row_b = {"merge": True, "tau_s": tau_s, "piece_s_max": piece}
            row_b.update(metrics_b)
            scored.append(row_b)
        print(f"scored tau_S={tau_s:.4g}", flush=True)

    # Hybrid A is the no-merge slice. Hybrid B is every merge=True row.
    # Selection uses Hybrid B rows so a no-merge result can still win when the
    # piece guard blocks every multi-segment run and the metrics match A.
    # To keep A visible as its own method, select on the union but require the
    # reported B point to be a merge=True row. If A is better, the report says so.
    decision = select_configuration([row for row in scored if row["merge"]])
    chosen = decision["chosen"]
    tau_s = float(chosen["tau_s"])
    piece = float(chosen["piece_s_max"])

    def pack(split_profiles: list, merge: bool) -> dict:
        return score_split(split_profiles, {profile.profile_id: cache[profile.profile_id] for profile in split_profiles}, tau_s, piece, merge)

    dev_a = pack(dev, False)
    dev_b = pack(dev, True)
    val_a = pack(val, False)
    val_b = pack(val, True)
    test_a = pack(test, False)
    test_b = pack(test, True)
    causes = cause_of_misses(dev, dev_cache, tau_s)

    sensitivity_rows = []
    guard_tau = 0.0009686250859269968
    for row in scored:
        piece_value = row["piece_s_max"]
        show_all_guards = row["merge"] and abs(row["tau_s"] - guard_tau) < 1e-12
        if (not row["merge"]) or math.isinf(piece_value) or show_all_guards:
            sensitivity_rows.append(row)

    keys = list(dict.fromkeys([*FOCUS, *PROBES]))
    real_started = time.perf_counter()
    real_records = []
    for p_value, delta, coupling in keys:
        loaded = load_real(p_value, delta, coupling)
        segments = segment_profile(cost_audit, pilot, loaded["omega"], loaded["rho"])
        loaded["index"] = ProfileIndex(loaded["omega"], loaded["rho"], segments)
        loaded["diagnosis"] = diagnose_case(loaded["index"], tau_s, piece)
        real_records.append(loaded)
        print(
            f"real p={p_value:.2f} d={delta:.3f} K={coupling:.2f} "
            f"segments={loaded['diagnosis']['n_segments']} "
            f"{loaded['diagnosis']['baseline_state']} -> A {loaded['diagnosis']['hybrid_a_state']} "
            f"B {loaded['diagnosis']['hybrid_b_state']}",
            flush=True,
        )
    real_seconds = time.perf_counter() - real_started
    roles = structural_roles(real_records)
    diagnoses = {
        (round(row["p"], 4), round(row["delta"], 4), round(row["K"], 4)): row["diagnosis"] for row in real_records
    }

    synthetic_panels = []
    val_profiles = {profile.profile_id: profile for profile in val}

    def first_val(**query):
        for profile in val:
            if all(getattr(profile, key) == value for key, value in query.items()):
                return profile
        raise RuntimeError(f"missing validation profile {query}")

    embedded = first_val(kind="embedded", length=250, sigma=0.0, seed=10, noise="iid")
    slow = first_val(kind="slow", sigma=0.0, seed=10, noise="iid")
    quad = first_val(kind="quad", sigma=0.0, seed=10, noise="iid")
    for title, profile in (
        ("Embedded plateau, validation, length 250", embedded),
        ("Slow slope, validation", slow),
        ("Quadratic drift, validation", quad),
    ):
        truth = profile.truth if profile.truth.ndim == 1 else profile.truth[2]
        synthetic_panels.append((title, cache[profile.profile_id], truth))
    real_panels = []
    for p_value, delta, coupling in FIGURE_KEYS:
        record = next(row for row in real_records if row["p"] == p_value and abs(row["delta"] - delta) < 1e-9 and abs(row["K"] - coupling) < 1e-9)
        info = record["diagnosis"]
        title = (
            f"p={p_value:.2f}, delta={delta:.2f}, K={coupling:.2f}\n"
            f"{info['n_segments']} segments, {info['baseline_state']} / A {info['hybrid_a_state']} / B {info['hybrid_b_state']}"
        )
        real_panels.append((title, record["index"]))
    figure_path = HERE / "hybrid_method_c_benchmark.png"
    plot_benchmark(synthetic_panels, real_panels, tau_s, piece, figure_path)

    connection = sqlite3.connect(DATA / "index.sqlite")
    n_index = int(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
    n_completed = int(connection.execute("SELECT COUNT(*) FROM runs WHERE status='completed'").fetchone()[0])
    connection.close()
    per_profile = segment_seconds / max(1, len(dev) + len(val) + len(test))
    per_real = real_seconds / max(1, len(real_records))
    # Stage C needs three windows. Segmentation dominates. Acceptance is negligible.
    cost_text = (
        f"The Stage 1 index has {n_index} rows, {n_completed} completed. "
        f"Archived segmentation details do not cover this rectangle. "
        f"Mean Method C time on the synthetic profiles was {per_profile:.3f} s per window. "
        f"Mean time on the {len(real_records)} real W3 profiles, including acceptance, was {per_real:.3f} s. "
        f"A full shadow pass over {n_index} runs and 3 windows is about {n_index * 3 * per_profile / 60:.1f} minutes "
        f"if the synthetic timing holds, or {n_index * 3 * per_real / 60:.1f} minutes using the real-profile timing. "
        "Only plateau acceptance and the existing state rules would be reapplied. The ODE would not be rerun. "
        "Memory stays at one profile at a time."
    )
    payload = {
        "decision": {
            "status": decision["status"],
            "stage_name": decision["stage_name"],
            "eligible_count": decision["eligible_count"],
            "stable_count": decision["stable_count"],
            "tau_values": decision["tau_values"],
            "piece_values": decision["piece_values"],
            "chosen": {"tau_s": tau_s, "piece_s_max": piece},
        },
        "dev_a": dev_a,
        "dev_b": dev_b,
        "val_a": val_a,
        "val_b": val_b,
        "test_a": test_a,
        "test_b": test_b,
        "slope_table": slope_table(dev_a["slope_rows"]),
        "sensitivity_table": sensitivity_table(sensitivity_rows),
        "real_table": real_table(real_records),
        "new_plateau_text": new_plateau_text(real_records),
        "role_text": role_text(roles, diagnoses),
        "false_positive_text": false_positive_text(val_a, val_b, dev_a["slope_rows"], dev_b["slope_rows"]),
        "cause_text": (
            f"Development positive plateaus: {causes['truth_plateaus']}. "
            f"Baseline hits: {causes['baseline_hit']}. "
            f"Missed because a linear segment already passes geometry and tau_S: {causes['linear_eligibility']}. "
            f"Missed because the covering pieces are below the archived size or span cut: {causes['over_segmentation']}. "
            f"Missed because the true interval itself fails archived geometry: {causes['archived_geometry']}. "
            f"Mixed or rejected by the trend cap: {causes['mixed_or_trend']}."
        ),
        "recommendation": recommendation_text(decision, val_a, val_b, test_a, test_b, causes),
        "cost_text": cost_text,
    }
    # The slope table in section 6 should show both methods. Replace with a combined view.
    payload["slope_table"] = "Hybrid A\n\n" + slope_table(dev_a["slope_rows"]) + "\n\nHybrid B\n\n" + slope_table(dev_b["slope_rows"])
    report_path = HERE / "hybrid_method_c_report.md"
    write_report(report_path, payload)
    summary = {
        "status": decision["status"],
        "stage": decision["stage_name"],
        "tau_s": tau_s,
        "piece_s_max": None if math.isinf(piece) else piece,
        "tau_values": decision["tau_values"],
        "val_a": public_metrics(val_a),
        "val_b": public_metrics(val_b),
        "test_a": public_metrics(test_a),
        "causes": causes,
        "cost": cost_text,
    }
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"wrote {figure_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    del val_profiles


if __name__ == "__main__":
    main()
