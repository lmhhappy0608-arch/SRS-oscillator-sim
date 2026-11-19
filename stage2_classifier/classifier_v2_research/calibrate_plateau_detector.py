"""Calibrate and shadow-evaluate the v2 plateau detector.

The selection rule in ``choose_configuration`` is fixed in this file before
the grid is scored. It does not maximize agreement with Classifier v1, and it
does not use the quarantined diagnostic stripes or the p=1 exam slice.

Synthetic development profiles choose the absolute-threshold configuration.
Validation and held-out synthetic profiles are scored after that choice.
Real Stage 1 profiles are used for a stability check and for the shadow
comparison. They are not a ground-truth accuracy set.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-classifier-v2")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "classifier_v2_research"))

from plateau_detector_candidate import (  # noqa: E402
    DetectorConfig,
    accept_plateaus,
    classify_profile,
    config_from_mapping,
    discover_candidates,
    merge_accepted,
    plateau_mask,
    review_windows,
)

HERE = Path(__file__).resolve().parent
DATA = ROOT / "stage1_runs"
MASTER_SEED = 20261108
N = 1000
DELTA_MAIN = 0.35
QUARANTINE_UNITS = {5, 6, 7, 8, 9}
DEV_P = {0.4, 0.8}
STATES = ("Death", "Locking", "Partial Death", "Partial Locking", "Incoherence")
FOCUS = (
    (0.20, 0.60),
    (0.20, 0.65),
    (0.20, 0.70),
    (0.20, 0.75),
    (0.15, 0.40),
)
POSITIVE_FAMILIES = ("embedded", "flat", "tails", "multi", "nearglobal")
DRIFT_FAMILIES = ("drift", "slow", "nonlinear")


def delta_unit(value: float) -> int:
    return int(round(float(value) / 0.025))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    inter = max(0, end - start)
    union = (left[1] - left[0]) + (right[1] - right[0]) - inter
    return float(inter / union) if union else 0.0


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


def assemble(delta: float, pieces: list[tuple[int, float, bool]], level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a continuous piecewise-linear profile on a uniform frequency grid."""
    omega = np.linspace(1.0 - delta, 1.0 + delta, N)
    rho = np.empty(N, dtype=np.float64)
    truth = np.zeros(N, dtype=bool)
    if sum(length for length, _slope, _flag in pieces) != N:
        raise ValueError("piece lengths must sum to N")
    cursor = 0
    height = float(level)
    for length, slope, is_plateau in pieces:
        block = omega[cursor : cursor + length]
        rho[cursor : cursor + length] = height + float(slope) * (block - block[0])
        height = float(rho[cursor + length - 1])
        if is_plateau:
            truth[cursor : cursor + length] = True
        cursor += length
    return omega, rho, truth


def add_noise(rho: np.ndarray, sigma: float, rng: np.random.Generator, kind: str) -> np.ndarray:
    if sigma <= 0.0:
        return rho.copy()
    if kind == "iid":
        return rho + rng.normal(0.0, sigma, size=rho.shape)
    if kind != "ar1":
        raise ValueError(kind)
    phi = 0.9
    innov = sigma * np.sqrt(1.0 - phi * phi)
    shock = np.zeros(rho.size, dtype=np.float64)
    shock[0] = float(rng.normal(0.0, sigma))
    draw = rng.normal(0.0, innov, size=rho.size - 1)
    for index, value in enumerate(draw, start=1):
        shock[index] = phi * shock[index - 1] + float(value)
    return rho + shock


class Profile:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _embedded(length: int) -> list[tuple[int, float, bool]]:
    side = (N - length) // 2
    return [(side, 1.0, False), (length, 0.0, True), (N - length - side, 0.8, False)]


def _tails(body: int) -> list[tuple[int, float, bool]]:
    side = (N - body) // 2
    return [(side, 1.1, False), (body, 0.0, True), (N - body - side, 0.7, False)]


def _multi() -> list[tuple[int, float, bool]]:
    return [(180, 0.0, True), (220, 1.2, False), (220, 0.0, True), (180, 0.9, False), (200, 0.0, True)]


def _steps(lengths: tuple[int, int, int]) -> list[tuple[int, float, bool]]:
    first, second, third = lengths
    gap = N - first - second - third
    return [(first, 0.0, True), (gap // 2, 1.0, False), (second, 0.0, True), (gap - gap // 2, 0.6, False), (third, 0.0, True)]


def make_clean(kind: str, delta: float, **item) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == "embedded":
        return assemble(delta, _embedded(int(item["length"])), 0.35)
    if kind == "flat":
        return assemble(delta, [(N, 0.0, True)], 0.62)
    if kind == "drift":
        return assemble(delta, [(N, float(item["slope"]), False)], 0.40)
    if kind == "slow":
        return assemble(delta, [(N, float(item["slope"]), False)], 0.55)
    if kind == "quad":
        omega = np.linspace(1.0 - delta, 1.0 + delta, N)
        span = float(omega[-1] - omega[0])
        t = (omega - omega[0]) / span
        return omega, 0.25 + 1.4 * t * t, np.zeros(N, dtype=bool)
    if kind == "sine":
        omega = np.linspace(1.0 - delta, 1.0 + delta, N)
        span = float(omega[-1] - omega[0])
        t = (omega - omega[0]) / span
        return omega, 0.30 + 0.9 * t + 0.03 * np.sin(4.0 * np.pi * t), np.zeros(N, dtype=bool)
    if kind == "tails":
        return assemble(delta, _tails(int(item["body"])), 0.20)
    if kind == "multi":
        return assemble(delta, _multi(), 0.30)
    if kind == "steps":
        return assemble(delta, _steps(tuple(item["lengths"])), 0.25)
    if kind == "nearglobal":
        omega, rho, truth = assemble(delta, [(N, 0.0, True)], 0.70)
        tail = int(item["tail"])
        rho = rho.copy()
        truth = truth.copy()
        rho[-tail:] = rho[-tail] + float(item.get("jump", 0.15)) * np.linspace(0.0, 1.0, tail)
        truth[-tail:] = False
        return omega, rho, truth
    raise ValueError(kind)


def materialize(spec: dict) -> Profile:
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, spec["seed"], spec["serial"]]))
    details = dict(spec)
    kind = details.pop("kind")
    delta = details.pop("delta")
    omega, clean, truth = make_clean(kind, delta, **details)
    if spec["windows"] == "stable":
        panes = np.vstack([add_noise(clean, spec["sigma"], rng, spec["noise"]) for _ in range(3)])
        truths = np.vstack([truth, truth, truth])
    elif spec["windows"] == "shift":
        panes = []
        truths = []
        for shift in (-40, 0, 40):
            runs = true_runs(truth)
            mask = np.zeros(N, dtype=bool)
            slid = clean.copy()
            for start, end in runs:
                new_start = min(max(0, start + shift), N - 2)
                new_end = min(max(new_start + 1, end + shift), N)
                mask[new_start:new_end] = True
                slid[new_start:new_end] = float(np.median(clean[start:end]))
            panes.append(add_noise(slid, spec["sigma"], rng, spec["noise"]))
            truths.append(mask)
        panes = np.vstack(panes)
        truths = np.vstack(truths)
    else:
        panes = add_noise(clean, spec["sigma"], rng, spec["noise"])
        truths = truth
    return Profile(
        profile_id=spec["profile_id"],
        split=spec["split"],
        family=spec["family"],
        kind=spec["kind"],
        sigma=float(spec["sigma"]),
        delta=float(spec["delta"]),
        seed=int(spec["seed"]),
        noise=spec["noise"],
        selectable=bool(spec["selectable"]),
        omega=omega,
        rho=panes,
        truth=truths,
        length=int(spec.get("length", spec.get("body", 0))),
    )


def catalog(split: str) -> list[Profile]:
    seeds = {"dev": (0, 1, 2), "val": (10, 11), "test": (20, 21)}[split]
    specs = []

    def add(**kwargs):
        kwargs.setdefault("delta", DELTA_MAIN)
        kwargs.setdefault("noise", "iid")
        kwargs.setdefault("windows", "none")
        kwargs.setdefault("selectable", split == "dev" and kwargs["windows"] == "none" and kwargs["noise"] == "iid")
        kwargs.setdefault("sigma", 0.0)
        specs.append(kwargs)

    if split == "dev":
        for length in (40, 100, 200, 500, 900):
            for sigma in (0.0, 1e-3, 0.01):
                add(family="embedded", kind="embedded", length=length, sigma=sigma)
        for sigma in (0.0, 1e-3, 0.01):
            add(family="flat", kind="flat", length=N, sigma=sigma)
        for slope in (0.30, 1.0):
            for sigma in (0.0, 1e-3, 0.01):
                add(family="drift", kind="drift", slope=slope, sigma=sigma)
        for slope in (0.02, 0.05, 0.10):
            for sigma in (0.0, 0.01):
                add(family="slow", kind="slow", slope=slope, sigma=sigma)
        for kind in ("quad", "sine"):
            for sigma in (0.0, 0.01):
                add(family="nonlinear", kind=kind, sigma=sigma)
        for body in (200, 500):
            for sigma in (0.0, 1e-3, 0.01):
                add(family="tails", kind="tails", body=body, length=body, sigma=sigma)
        for sigma in (0.0, 0.01):
            add(family="multi", kind="multi", sigma=sigma)
        for tail in (5, 20, 40):
            for sigma in (0.0, 1e-3):
                add(family="nearglobal", kind="nearglobal", tail=tail, sigma=sigma)
    elif split == "val":
        for length in (60, 250):
            for sigma in (0.0, 1e-3, 0.01):
                add(family="embedded", kind="embedded", length=length, sigma=sigma)
        for slope in (0.08, 0.40, 1.20):
            for sigma in (0.0, 1e-3):
                add(family="drift", kind="drift", slope=slope, sigma=sigma)
        for sigma in (0.0, 0.01):
            add(family="slow", kind="slow", slope=0.03, sigma=sigma)
            add(family="nonlinear", kind="quad", sigma=sigma)
        for sigma in (0.0, 1e-3):
            add(family="tails", kind="tails", body=350, length=350, sigma=sigma)
            add(family="nearglobal", kind="nearglobal", tail=8, sigma=sigma)
            add(family="nearglobal", kind="nearglobal", tail=30, sigma=sigma)
        add(family="multi", kind="steps", lengths=(120, 160, 140), sigma=0.0)
        add(family="multi", kind="steps", lengths=(120, 160, 140), sigma=1e-3)
        add(family="drift", kind="drift", slope=0.30, sigma=0.01, noise="ar1", selectable=False)
        add(family="embedded", kind="embedded", length=200, sigma=0.01, noise="ar1", selectable=False)
    else:
        for length in (80, 700):
            for sigma in (0.0, 1e-3, 0.01):
                add(family="embedded", kind="embedded", length=length, sigma=sigma, selectable=False)
        for slope in (0.15, 0.70):
            for sigma in (0.0, 0.01):
                add(family="drift", kind="drift", slope=slope, sigma=sigma, selectable=False)
        add(family="slow", kind="slow", slope=0.04, sigma=0.0, selectable=False)
        add(family="slow", kind="slow", slope=0.04, sigma=0.01, selectable=False)
        add(family="nonlinear", kind="sine", sigma=0.0, selectable=False)
        add(family="nonlinear", kind="sine", sigma=0.01, selectable=False)
        add(family="tails", kind="tails", body=450, length=450, sigma=0.0, selectable=False)
        add(family="tails", kind="tails", body=450, length=450, sigma=1e-3, selectable=False)
        add(family="nearglobal", kind="nearglobal", tail=12, sigma=0.0, selectable=False)
        add(family="nearglobal", kind="nearglobal", tail=12, sigma=1e-3, selectable=False)
        add(family="flat", kind="flat", length=N, sigma=1e-3, selectable=False)
    built = []
    serial = 0
    for seed in seeds:
        for spec in specs:
            item = dict(spec)
            item.update(split=split, seed=seed, serial=serial)
            item["profile_id"] = f"{split}-{item['family']}-{serial}"
            built.append(materialize(item))
            serial += 1
    return built


def window_arrays(profile: Profile, window: int) -> tuple[np.ndarray, np.ndarray]:
    if profile.rho.ndim == 1:
        return profile.rho, profile.truth
    return profile.rho[window], profile.truth[window]


def discovery_settings(reference_gain: float) -> list[dict]:
    """Cover the noiseless break scale before any score is computed."""
    gain = max(float(reference_gain), 1e-8)
    rss_values = [float(value) for value in np.geomspace(gain / 100.0, gain * 100.0, 5)]
    settings = []
    for min_segment in (8, 16):
        for split_rss in rss_values:
            for peel_max in (0, 8):
                settings.append(
                    {
                        "min_segment": min_segment,
                        "split_rss": split_rss,
                        "peel_max": peel_max,
                        "peel_drop": 0.01,
                    }
                )
    return settings


def acceptance_grid() -> list[dict]:
    grid = []
    for n_min in (15, 30, 60, 120):
        for l_min in (0.01, 0.02, 0.05):
            for r_max in (0.005, 0.01, 0.02, 0.05):
                for s_max in (0.005, 0.01, 0.02, 0.05, 0.10):
                    grid.append({"n_min": n_min, "l_min": l_min, "r_max": r_max, "s_max": s_max})
    return grid


def config_from_parts(discovery: dict, acceptance: dict) -> DetectorConfig:
    payload = dict(discovery)
    payload.update(acceptance)
    payload.update(
        nonzero_level=0.001,
        near_zero=0.001,
        lock_range=0.01,
        partial_death_run=15,
    )
    return config_from_mapping(payload)


def reference_break_gain() -> float:
    omega, rho, _truth = make_clean("tails", DELTA_MAIN, body=500)
    segments = discover_candidates(
        omega,
        rho,
        DetectorConfig(min_segment=8, split_rss=1e30, peel_max=0, peel_drop=0.01),
    )
    # One unsplit segment: measure the best available linear-to-two-line gain directly.
    from plateau_detector_candidate import _best_split, _prefixes

    gain, _cut = _best_split(_prefixes(omega, rho), 0, rho.size, 8)
    if segments and segments[0]["n"] != rho.size:
        raise RuntimeError("a huge split threshold still split the reference profile")
    return float(gain)


def build_discovery_cache(profiles: list[Profile], settings: list[dict]) -> dict:
    cache = {}
    started = time.perf_counter()
    for profile in profiles:
        rho, _truth = window_arrays(profile, 2)
        for setting in settings:
            config = config_from_parts(setting, {"n_min": 15, "l_min": 0.0, "r_max": 1e9, "s_max": 1e9})
            found = discover_candidates(profile.omega, rho, config)
            key = (profile.profile_id, setting["min_segment"], setting["split_rss"], setting["peel_max"])
            cache[key] = found
    elapsed = time.perf_counter() - started
    print(f"discovery cache: {len(profiles)} profiles x {len(settings)} settings in {elapsed:.1f}s", flush=True)
    return cache


def predicted_intervals(profile: Profile, discovered: list[dict], config: DetectorConfig, rho: np.ndarray) -> list[tuple[int, int]]:
    kept = []
    for item in discovered:
        relative = item["relative_omega_span"]
        trend = item["trend"]
        if (
            item["n"] >= config.n_min
            and np.isfinite(relative)
            and relative >= config.l_min
            and item["rho_range"] <= config.r_max
            and np.isfinite(trend)
            and trend <= config.s_max
            and abs(item["rho_mean"]) >= config.nonzero_level
        ):
            kept.append(dict(item))
    merged = merge_accepted(profile.omega, rho, kept, config)
    return [(int(item["start"]), int(item["end"])) for item in merged if abs(item["rho_mean"]) >= config.nonzero_level]


def score_profiles(profiles: list[Profile], config: DetectorConfig, cache: dict | None) -> dict:
    per_family_f1 = defaultdict(list)
    per_family_fp = defaultdict(list)
    long_hits = []
    near_hits = []
    boundary = []
    length_hits = defaultdict(list)
    stress_hits = []
    for profile in profiles:
        if cache is None:
            rho, truth = window_arrays(profile, 2)
            discovered = discover_candidates(profile.omega, rho, config)
        else:
            rho, truth = window_arrays(profile, 2)
            key = (profile.profile_id, config.min_segment, config.split_rss, config.peel_max)
            discovered = cache[key]
        intervals = predicted_intervals(profile, discovered, config, rho)
        pred = np.zeros(N, dtype=bool)
        for start, end in intervals:
            pred[start:end] = True
        if profile.family in POSITIVE_FAMILIES:
            per_family_f1[profile.family].append(oscillator_f1(pred, truth))
        if profile.family in DRIFT_FAMILIES:
            per_family_fp[profile.family].append(float(pred.any()))
        truth_runs = true_runs(truth)
        used = set()
        for start, end in truth_runs:
            best = 0.0
            best_index = None
            for index, pred_run in enumerate(intervals):
                if index in used:
                    continue
                score = interval_iou((start, end), pred_run)
                if score > best:
                    best = score
                    best_index = index
            matched = best >= 0.5 and best_index is not None
            if matched:
                used.add(best_index)
                pred_run = intervals[best_index]
                boundary.append(0.5 * (abs(pred_run[0] - start) + abs(pred_run[1] - end)))
            length = end - start
            easy = profile.sigma <= 1e-3 + 1e-15
            if easy and length >= 80:
                long_hits.append(float(matched))
            if profile.family == "nearglobal" and easy:
                near_hits.append(float(matched))
            if easy and profile.family == "embedded":
                length_hits[length].append(float(matched))
            if profile.sigma >= 0.01 - 1e-15 and profile.family in POSITIVE_FAMILIES and length >= 80:
                stress_hits.append(float(matched))
    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    fp = {family: mean(per_family_fp[family]) for family in DRIFT_FAMILIES}
    f1 = {family: mean(per_family_f1[family]) for family in POSITIVE_FAMILIES}
    finite_f1 = [value for value in f1.values() if np.isfinite(value)]
    finite_fp = [value for value in fp.values() if np.isfinite(value)]
    return {
        "macro_f1": float(np.mean(finite_f1)) if finite_f1 else 0.0,
        "fp_mean": float(np.mean(finite_fp)) if finite_fp else 0.0,
        "fp_drift": fp["drift"],
        "fp_slow": fp["slow"],
        "fp_nonlinear": fp["nonlinear"],
        "f1_by_family": f1,
        "recall_long": mean(long_hits),
        "recall_nearglobal": mean(near_hits),
        "boundary_median": float(np.median(boundary)) if boundary else float("inf"),
        "stress_recall": mean(stress_hits),
        "n_long": len(long_hits),
        "n_nearglobal": len(near_hits),
        "n_boundary": len(boundary),
        "length_recall": {str(key): mean(value) for key, value in sorted(length_hits.items())},
        "n_profiles": len(profiles),
    }


def passes(metrics: dict, stage: int) -> bool:
    """Pre-registered eligibility. Stage 0 is the target; later stages relax it."""
    fp05 = metrics["fp_drift"] <= 0.05 and metrics["fp_slow"] <= 0.05 and metrics["fp_nonlinear"] <= 0.05
    fp10 = metrics["fp_drift"] <= 0.10 and metrics["fp_slow"] <= 0.10 and metrics["fp_nonlinear"] <= 0.10
    if stage == 0:
        return bool(
            fp05
            and metrics["recall_long"] >= 0.80
            and metrics["recall_nearglobal"] >= 0.80
            and metrics["boundary_median"] <= 20.0
        )
    if stage == 1:
        return bool(fp05 and metrics["recall_long"] >= 0.80 and metrics["recall_nearglobal"] >= 0.80)
    if stage == 2:
        return bool(fp05 and metrics["recall_long"] >= 0.60 and metrics["recall_nearglobal"] >= 0.60)
    if stage == 3:
        return bool(fp10 and metrics["recall_long"] >= 0.60 and metrics["recall_nearglobal"] >= 0.60)
    raise ValueError(stage)


STAGE_NAME = {
    0: "full",
    1: "boundary constraint dropped",
    2: "recall floor lowered to 0.60",
    3: "false-positive floor raised to 0.10 and recall floor 0.60",
}


def snap(value: float, grid: list[float]) -> float:
    array = np.asarray(grid, dtype=np.float64)
    return float(array[int(np.argmin(np.abs(np.log10(array) - np.log10(value))))] if min(grid) > 0 else array[int(np.argmin(np.abs(array - value)))])


def component_median(rows: list[dict]) -> DetectorConfig:
    """Snap each parameter to a value that actually occurs in the near-optimal rows."""

    def median_snap(key: str, logarithmic: bool) -> float:
        present = np.asarray(sorted({float(row[key]) for row in rows}), dtype=np.float64)
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        center = float(np.exp(np.median(np.log(values)))) if logarithmic else float(np.median(values))
        if logarithmic:
            index = int(np.argmin(np.abs(np.log(present) - np.log(center))))
        else:
            index = int(np.argmin(np.abs(present - center)))
        return float(present[index])

    discovery = {
        "min_segment": int(median_snap("min_segment", False)),
        "split_rss": median_snap("split_rss", True),
        "peel_max": int(median_snap("peel_max", False)),
        "peel_drop": 0.01,
    }
    acceptance = {
        "n_min": int(median_snap("n_min", False)),
        "l_min": median_snap("l_min", True),
        "r_max": median_snap("r_max", True),
        "s_max": median_snap("s_max", True),
    }
    return config_from_parts(discovery, acceptance)


def choose_configuration(scored: list[dict], settings: list[dict], acceptances: list[dict]) -> dict:
    """Select one grid point. This rule is not refit after the scores exist."""
    stage_used = None
    pool = []
    for stage in range(4):
        pool = [row for row in scored if passes(row, stage)]
        if pool:
            stage_used = stage
            break
    status = "frozen"
    if stage_used is None:
        status = "unresolved"
        viable = [row for row in scored if row["recall_long"] >= 0.50]
        if not viable:
            viable = scored
        def worst_fp(row: dict) -> float:
            return max(row["fp_drift"], row["fp_slow"], row["fp_nonlinear"])
        viable.sort(key=lambda row: (worst_fp(row), -row["recall_long"], -row["macro_f1"]))
        pool = [viable[0]]
        stage_used = -1
    best_f1 = max(row["macro_f1"] for row in pool)
    best_fp = min(row["fp_mean"] for row in pool)
    near = [
        row
        for row in pool
        if row["macro_f1"] >= best_f1 - 0.02 and row["fp_mean"] <= best_fp + 0.01
    ]
    return {
        "status": status,
        "stage": stage_used,
        "stage_name": "unresolved fallback" if stage_used < 0 else STAGE_NAME[stage_used],
        "eligible_count": len(pool),
        "near_count": len(near),
        "near": near,
        "best_f1": best_f1,
        "best_fp": best_fp,
    }


def self_check() -> None:
    config = DetectorConfig(min_segment=8, split_rss=1e-4, peel_max=8, peel_drop=0.01, n_min=15, l_min=0.02, r_max=0.02, s_max=0.02)
    omega = np.linspace(0.8, 1.2, N)
    death = classify_profile(omega, np.zeros(N), config)
    if death["state"] != "Death" or death["plateau_count"] != 0:
        raise RuntimeError(f"death check returned {death['state']}")
    locked = classify_profile(omega, np.full(N, 0.4), config)
    if locked["state"] != "Locking":
        raise RuntimeError(f"locking check returned {locked['state']}")
    partial = np.concatenate([np.zeros(40), np.linspace(0.2, 1.0, N - 40)])
    partial_state = classify_profile(omega, partial, config)
    if partial_state["state"] != "Partial Death":
        raise RuntimeError(f"partial death check returned {partial_state['state']}")
    degenerate = classify_profile(np.ones(N), np.linspace(0.2, 0.8, N), config, delta=0.0)
    if degenerate["plateau_count"] != 0 or degenerate["state"] != "Incoherence":
        raise RuntimeError(f"delta 0 check returned {degenerate['state']} plateaus={degenerate['plateau_count']}")
    _omega, clean, truth = make_clean("tails", DELTA_MAIN, body=500)
    found = accept_plateaus(_omega, clean, config)
    if len(found) != 1 or interval_iou((found[0]["start"], found[0]["end"]), true_runs(truth)[0]) < 0.98:
        raise RuntimeError("noiseless plateau-with-tails was not recovered as one interior plateau")
    slow_omega, slow, _slow_truth = make_clean("slow", DELTA_MAIN, slope=0.05)
    slow_found = accept_plateaus(slow_omega, slow, config)
    if slow_found:
        raise RuntimeError("a noiseless slow slope was accepted as a plateau")
    print("self-check passed", flush=True)


def load_index() -> list[dict]:
    connection = sqlite3.connect(DATA / "index.sqlite")
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute("SELECT run_id, status, n, K, p, delta, trial_id, output_npz FROM runs")]
    connection.close()
    units = {delta_unit(row["delta"]) for row in rows}
    split_of = {}
    usable = sorted(unit for unit in units if unit != 0 and unit not in QUARANTINE_UNITS)
    for index, unit in enumerate(usable):
        split_of[unit] = ("dev", "val", "heldout")[index % 3]
    for row in rows:
        unit = delta_unit(row["delta"])
        p_value = round(float(row["p"]), 4)
        row["delta_unit"] = unit
        row["p_rounded"] = p_value
        if unit == 0:
            row["role"] = "degenerate"
        elif unit in QUARANTINE_UNITS:
            row["role"] = "quarantine"
        elif p_value == 1.0:
            row["role"] = "exam"
        elif p_value == 0.2:
            row["role"] = "transfer"
        else:
            row["role"] = split_of[unit]
        row["delta_split"] = split_of.get(unit, "excluded")
    return rows, split_of


def load_arrays(row: dict) -> tuple[np.ndarray, np.ndarray]:
    path = DATA / row["output_npz"]
    with np.load(path, allow_pickle=False) as data:
        omega = np.array(data["omega"], dtype=np.float64)
        rotation = np.array(data["rotation_numbers"], dtype=np.float64)
    if np.any(np.diff(omega) < 0):
        order = np.argsort(omega, kind="mergesort")
        omega = omega[order]
        rotation = rotation[:, order]
    return omega, rotation


def stability_score(rows: list[dict], config: DetectorConfig) -> dict:
    grouped = defaultdict(list)
    window_scores = []
    loaded = []
    for row in rows:
        omega, rotation = load_arrays(row)
        masks = []
        for window in range(3):
            found = accept_plateaus(omega, rotation[window], config)
            masks.append(plateau_mask(omega.size, found))
        window_scores.append(jaccard(masks[2], masks[1]))
        window_scores.append(jaccard(masks[2], masks[0]))
        loaded.append((round(float(row["p"]), 4), row["delta_unit"], float(row["K"]), masks[2]))
        grouped[(round(float(row["p"]), 4), row["delta_unit"])].append((float(row["K"]), masks[2]))
    neighbor = []
    for bundle in grouped.values():
        bundle.sort()
        for (_k0, left), (_k1, right) in zip(bundle, bundle[1:]):
            neighbor.append(jaccard(left, right))
    return {
        "window_jaccard_median": float(np.median(window_scores)) if window_scores else float("nan"),
        "neighbor_jaccard_median": float(np.median(neighbor)) if neighbor else float("nan"),
        "n_runs": len(rows),
        "n_neighbor_pairs": len(neighbor),
    }


def public_metrics(metrics: dict) -> dict:
    return {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in metrics.items()
        if key != "length_recall"
    } | {"length_recall": metrics["length_recall"]}


def calibration() -> dict:
    self_check()
    gain = reference_break_gain()
    settings = discovery_settings(gain)
    acceptances = acceptance_grid()
    dev = [profile for profile in catalog("dev") if profile.selectable]
    print(f"reference split gain {gain:.6g}; dev profiles {len(dev)}; discovery {len(settings)}; acceptance {len(acceptances)}", flush=True)
    cache = build_discovery_cache(dev, settings)
    scored = []
    started = time.perf_counter()
    for discovery_index, discovery in enumerate(settings, start=1):
        for acceptance in acceptances:
            config = config_from_parts(discovery, acceptance)
            metrics = score_profiles(dev, config, cache)
            row = dict(discovery)
            row.update(acceptance)
            row.update(metrics)
            scored.append(row)
        print(f"discovery {discovery_index}/{len(settings)} scored", flush=True)
    print(f"scored {len(scored)} configurations in {time.perf_counter() - started:.1f}s", flush=True)
    decision = choose_configuration(scored, settings, acceptances)
    near = decision["near"]
    index_rows, split_of = load_index()
    dev_rows = [row for row in index_rows if row["role"] == "dev"]
    if len(near) > 1:
        ranked = sorted(near, key=lambda row: (-row["macro_f1"], row["fp_mean"]))[:8]
        print(f"stability tie-break on {len(ranked)} of {len(near)} near-optimal configurations", flush=True)
        decorated = []
        for row in ranked:
            config = config_from_parts(row, row)
            summary = stability_score(dev_rows, config)
            row["real_window_jaccard"] = summary["window_jaccard_median"]
            row["real_neighbor_jaccard"] = summary["neighbor_jaccard_median"]
            decorated.append(row)
        best_neighbor = max(row["real_neighbor_jaccard"] for row in decorated)
        stable = [row for row in decorated if row["real_neighbor_jaccard"] >= best_neighbor - 0.05]
        if any(row["real_neighbor_jaccard"] >= 0.50 for row in decorated):
            stable = [row for row in stable if row["real_neighbor_jaccard"] >= 0.50] or stable
        chosen = component_median(stable)
        stability_pool = stable
    else:
        chosen = component_median(near)
        stability_pool = near
    chosen_metrics = score_profiles(dev, chosen, cache)
    dev_stability = stability_score(dev_rows, chosen)
    val = catalog("val")
    test = catalog("test")
    val_metrics = score_profiles([profile for profile in val if profile.noise == "iid"], chosen, None)
    test_metrics = score_profiles(test, chosen, None)
    ar_metrics = score_profiles([profile for profile in val if profile.noise == "ar1"], chosen, None)
    ablation = {}
    for name, override in (
        ("drop_s", {"s_max": 1e9}),
        ("drop_r", {"r_max": 1e9}),
        ("drop_l", {"l_min": 0.0}),
        ("drop_n", {"n_min": 2}),
    ):
        payload = {field: getattr(chosen, field) for field in DetectorConfig.__dataclass_fields__}
        payload.update(override)
        ablation[name] = public_metrics(score_profiles(dev, config_from_mapping(payload), cache))
    temporal = temporal_check(chosen)
    delta_sensitivity = {
        str(delta): public_metrics(score_profiles(sensitivity_profiles(delta), chosen, None))
        for delta in (0.05, 0.80)
    }
    noise_aware = noise_aware_check(val, chosen, val_metrics)
    final = chosen if not noise_aware["adopted"] else noise_aware["config"]
    if noise_aware["adopted"]:
        test_metrics = score_profiles(test, final, None)
        dev_stability = stability_score(dev_rows, final)
    transfer_rows = [row for row in index_rows if row["role"] == "transfer" and row["delta_split"] == "val"]
    transfer = stability_score(transfer_rows, final)
    result = {
        "reference_split_gain": gain,
        "n_dev_profiles": len(dev),
        "n_configurations": len(scored),
        "selection_status": decision["status"],
        "selection_stage": decision["stage_name"],
        "eligible_count": decision["eligible_count"],
        "near_count": decision["near_count"],
        "near_parameter_ranges": parameter_ranges(decision["near"]),
        "config": {field: getattr(final, field) for field in DetectorConfig.__dataclass_fields__},
        "dev_metrics": public_metrics(chosen_metrics),
        "validation_metrics": public_metrics(val_metrics),
        "test_metrics": public_metrics(test_metrics),
        "ar1_sensitivity": public_metrics(ar_metrics),
        "ablation_dev": ablation,
        "temporal_check": temporal,
        "delta_sensitivity": delta_sensitivity,
        "noise_aware": {key: value for key, value in noise_aware.items() if key != "config"},
        "real_dev_stability": dev_stability,
        "real_transfer_stability": transfer,
        "delta_splits": {str(unit * 0.025): name for unit, name in split_of.items()},
        "roles": dict(Counter(row["role"] for row in index_rows)),
        "stability_pool_size": len(stability_pool),
    }
    write_json(HERE / "candidate_parameters.json", result)
    update_frozen_block(result["config"])
    print(json.dumps({"status": result["selection_status"], "stage": result["selection_stage"], "config": result["config"]}, indent=2), flush=True)
    return result


def parameter_ranges(rows: list[dict]) -> dict:
    keys = ("min_segment", "split_rss", "peel_max", "n_min", "l_min", "r_max", "s_max")
    ranges = {}
    for key in keys:
        values = sorted({row[key] for row in rows})
        ranges[key] = values
    return ranges


def update_frozen_block(config: dict) -> None:
    path = HERE / "plateau_detector_candidate.py"
    text = path.read_text(encoding="utf-8")
    start = text.index("FROZEN = {")
    end = text.index("}", start) + 1
    lines = ["FROZEN = {"]
    for key, value in config.items():
        if isinstance(value, float):
            lines.append(f"    {key!r}: {value!r},")
        else:
            lines.append(f"    {key!r}: {value!r},")
    lines.append("}")
    path.write_text(text[:start] + "\n".join(lines) + text[end:], encoding="utf-8")


def temporal_check(config: DetectorConfig) -> dict:
    profiles = []
    for mode in ("stable", "shift"):
        for sigma in (1e-3, 0.01):
            for seed in (0, 1, 2):
                spec = {
                    "family": "embedded",
                    "kind": "embedded",
                    "length": 400,
                    "sigma": sigma,
                    "delta": DELTA_MAIN,
                    "noise": "iid",
                    "windows": mode,
                    "selectable": False,
                    "seed": seed,
                    "serial": seed + (0 if mode == "stable" else 100) + int(sigma * 1000),
                    "split": "temporal",
                }
                spec["profile_id"] = f"temporal-{mode}-{spec['serial']}"
                profiles.append(materialize(spec))
    stable_fail = []
    shift_jaccard = []
    for profile in profiles:
        masks = []
        for window in range(3):
            rho, _truth = window_arrays(profile, window)
            found = accept_plateaus(profile.omega, rho, config)
            masks.append(plateau_mask(N, found))
        score = jaccard(masks[2], masks[1])
        if profile.profile_id.startswith("temporal-stable"):
            stable_fail.append(float(score < 0.5))
        else:
            shift_jaccard.append(score)
    failure_rate = float(np.mean(stable_fail)) if stable_fail else float("nan")
    return {
        "stable_low_jaccard_rate": failure_rate,
        "shift_w3_w2_jaccard_median": float(np.median(shift_jaccard)) if shift_jaccard else float("nan"),
        "gate_recommended": bool(failure_rate > 0.20),
        "gate_enabled": False,
    }


def sensitivity_profiles(delta: float) -> list[Profile]:
    specs = []
    for family, kind, extra, sigma in (
        ("embedded", "embedded", {"length": 200}, 0.0),
        ("embedded", "embedded", {"length": 200}, 1e-3),
        ("slow", "slow", {"slope": 0.05}, 0.0),
        ("drift", "drift", {"slope": 1.0}, 0.0),
        ("nearglobal", "nearglobal", {"tail": 20}, 0.0),
        ("nonlinear", "quad", {}, 0.0),
    ):
        item = {
            "family": family,
            "kind": kind,
            "sigma": sigma,
            "delta": delta,
            "noise": "iid",
            "windows": "none",
            "selectable": False,
            "seed": 0,
            "serial": len(specs),
            "split": f"delta-{delta}",
            **extra,
        }
        item["profile_id"] = f"delta-{delta}-{len(specs)}"
        specs.append(materialize(item))
    return specs


def noise_aware_check(val_profiles: list[Profile], chosen: DetectorConfig, baseline: dict) -> dict:
    """Compare one residual-scale variant on validation. Do not use sigma_hat."""
    iid = [profile for profile in val_profiles if profile.noise == "iid"]
    best = {"adopted": False, "reason": "absolute thresholds retained", "trials": []}
    for scale in (3.0, 5.0, 10.0):
        adopted_metrics = score_scaled(iid, chosen, scale)
        best["trials"].append({"scale": scale, **public_metrics(adopted_metrics)})
        gain = adopted_metrics["macro_f1"] - baseline["macro_f1"]
        fp_ok = (
            adopted_metrics["fp_drift"] <= 0.05
            and adopted_metrics["fp_slow"] <= 0.05
            and adopted_metrics["fp_nonlinear"] <= 0.05
        )
        if gain >= 0.05 and fp_ok and adopted_metrics["recall_long"] >= baseline["recall_long"] - 1e-9:
            payload = {field: getattr(chosen, field) for field in DetectorConfig.__dataclass_fields__}
            payload["r_max"] = float(chosen.r_max)
            payload["s_max"] = float(chosen.s_max)
            best = {
                "adopted": False,
                "reason": "scale improved validation F1 but the shipped rule stays absolute; a scale would change the acceptance definition",
                "would_adopt_under_numeric_rule": True,
                "scale": scale,
                "trials": best["trials"],
            }
            # The confirmed detector uses absolute limits. A numeric gain is recorded
            # and does not replace the absolute rule.
            break
    best["baseline_macro_f1"] = baseline["macro_f1"]
    best["config"] = chosen
    return best


def score_scaled(profiles: list[Profile], config: DetectorConfig, scale: float) -> dict:
    """Validation diagnostic: widen R and S by a residual MAD factor, per segment."""
    per_family_f1 = defaultdict(list)
    per_family_fp = defaultdict(list)
    long_hits = []
    near_hits = []
    for profile in profiles:
        rho, truth = window_arrays(profile, 2)
        discovered = discover_candidates(profile.omega, rho, config)
        pred = np.zeros(N, dtype=bool)
        for item in discovered:
            piece = rho[item["start"] : item["end"]]
            center = piece - np.linspace(piece[0], piece[-1], piece.size)
            mad = float(np.median(np.abs(center - np.median(center)))) if piece.size else 0.0
            allowance = max(config.r_max, scale * 1.4826 * mad)
            trend_allowance = max(config.s_max, allowance)
            relative = item["relative_omega_span"]
            if (
                item["n"] >= config.n_min
                and np.isfinite(relative)
                and relative >= config.l_min
                and item["rho_range"] <= allowance
                and np.isfinite(item["trend"])
                and item["trend"] <= trend_allowance
                and abs(item["rho_mean"]) >= config.nonzero_level
            ):
                pred[item["start"] : item["end"]] = True
        if profile.family in POSITIVE_FAMILIES:
            per_family_f1[profile.family].append(oscillator_f1(pred, truth))
        if profile.family in DRIFT_FAMILIES:
            per_family_fp[profile.family].append(float(pred.any()))
        for start, end in true_runs(truth):
            overlap = pred[start:end].mean() if end > start else 0.0
            matched = overlap >= 0.5
            if profile.sigma <= 1e-3 + 1e-15 and end - start >= 80:
                long_hits.append(float(matched))
            if profile.family == "nearglobal" and profile.sigma <= 1e-3 + 1e-15:
                near_hits.append(float(matched))
    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")
    f1 = {family: mean(per_family_f1[family]) for family in POSITIVE_FAMILIES}
    fp = {family: mean(per_family_fp[family]) for family in DRIFT_FAMILIES}
    finite_f1 = [value for value in f1.values() if np.isfinite(value)]
    return {
        "macro_f1": float(np.mean(finite_f1)) if finite_f1 else 0.0,
        "fp_drift": fp["drift"],
        "fp_slow": fp["slow"],
        "fp_nonlinear": fp["nonlinear"],
        "recall_long": mean(long_hits),
        "recall_nearglobal": mean(near_hits),
        "boundary_median": float("nan"),
        "fp_mean": float(np.nanmean(list(fp.values()))),
        "f1_by_family": f1,
        "stress_recall": float("nan"),
        "n_long": len(long_hits),
        "n_nearglobal": len(near_hits),
        "n_boundary": 0,
        "length_recall": {},
        "n_profiles": len(profiles),
    }


def flat_linear_count(segments: list[dict]) -> int:
    count = 0
    for segment in segments:
        if (
            segment["mode"] == "linear"
            and segment["length"] >= 15
            and segment["relative_omega_span"] >= 0.02
            and segment["rho_span"] <= 0.02
        ):
            count += 1
    return count


def collect_focus_cases(config: DetectorConfig, pilot, cost_audit, v1_config) -> list[dict]:
    from stage2_classifier.classifier_v2.classify import classify_window

    rows, _split = load_index()
    details = []
    for delta, coupling in FOCUS:
        match = [
            row for row in rows
            if round(float(row["p"]), 4) == 1.0
            and abs(float(row["delta"]) - delta) < 1e-9
            and abs(float(row["K"]) - coupling) < 1e-9
        ]
        row = match[0]
        omega, rotation = load_arrays(row)
        v1 = classify_window(cost_audit, pilot, omega, rotation[2], v1_config)
        v2 = classify_profile(omega, rotation[2], config, delta=float(row["delta"]))
        details.append(focus_record(row, omega, rotation[2], v1, v2))
    return details


def shadow(config: DetectorConfig) -> dict:
    from stage2_classifier.classifier_v2.classify import WINDOWS, classify_window, load_config
    from stage2_classifier.penalty_calibration.method_c_separated import load_method_c

    pilot, cost_audit = load_method_c()
    v1_config = load_config()
    rows, _split_of = load_index()
    output = HERE / "shadow_comparison.csv"
    fields = [
        "run_id", "index_status", "p", "delta", "K", "N", "trial_id", "role",
        "v1_state", "v2_state",
        "v1_w1", "v1_w2", "v1_w3", "v2_w1", "v2_w2", "v2_w3",
        "v1_plateau_count", "v2_plateau_count",
        "v1_flat_linear_count", "v2_coverage", "v2_intervals",
        "v2_w3_w2_jaccard", "v2_w3_w1_jaccard",
        "v1_review", "v2_review", "v1_seconds", "v2_seconds",
    ]
    done = set()
    if output.is_file():
        with output.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                done.add(row["run_id"])
    mode = "a" if done else "w"
    started = time.perf_counter()
    transitions = Counter()
    other_changes = 0
    focus_details = []
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not done:
            writer.writeheader()
        for index, row in enumerate(rows):
            if row["run_id"] in done:
                continue
            omega, rotation = load_arrays(row)
            t0 = time.perf_counter()
            v1 = {}
            for name, start, end in WINDOWS:
                slot = int(np.argmin(np.abs(np.array([8000.0, 10000.0, 12000.0]) - start)))
                v1[name] = classify_window(cost_audit, pilot, omega, rotation[slot], v1_config)
            v1_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            v2 = {}
            masks = []
            for window, (name, _start, _end) in enumerate(WINDOWS):
                v2[name] = classify_profile(omega, rotation[window], config, delta=float(row["delta"]))
                masks.append(plateau_mask(omega.size, v2[name]["plateaus"]))
            v2_seconds = time.perf_counter() - t1
            intervals = [
                f"{item['start']}:{item['end']}"
                for item in v2["W3"]["plateaus"]
                if item["nonzero_plateau"]
            ]
            record = {
                "run_id": row["run_id"],
                "index_status": row["status"],
                "p": row["p"],
                "delta": row["delta"],
                "K": row["K"],
                "N": row["n"],
                "trial_id": row["trial_id"],
                "role": row["role"],
                "v1_state": v1["W3"]["state"],
                "v2_state": v2["W3"]["state"],
                "v1_w1": v1["W1"]["state"],
                "v1_w2": v1["W2"]["state"],
                "v1_w3": v1["W3"]["state"],
                "v2_w1": v2["W1"]["state"],
                "v2_w2": v2["W2"]["state"],
                "v2_w3": v2["W3"]["state"],
                "v1_plateau_count": v1["W3"]["validated_plateau_count"],
                "v2_plateau_count": v2["W3"]["plateau_count"],
                "v1_flat_linear_count": flat_linear_count(v1["W3"]["segments"]),
                "v2_coverage": f"{v2['W3']['locked_fraction']:.6f}",
                "v2_intervals": ";".join(intervals),
                "v2_w3_w2_jaccard": f"{jaccard(masks[2], masks[1]):.6f}",
                "v2_w3_w1_jaccard": f"{jaccard(masks[2], masks[0]):.6f}",
                "v1_review": int(len({v1[name]['state'] for name in ('W1', 'W2', 'W3')}) > 1 or len({v1[name]['validated_plateau_count'] for name in ('W1', 'W2', 'W3')}) > 1),
                "v2_review": review_windows(v2)[0],
                "v1_seconds": f"{v1_seconds:.6f}",
                "v2_seconds": f"{v2_seconds:.6f}",
            }
            writer.writerow(record)
            transitions[(record["v1_state"], record["v2_state"])] += 1
            if record["v1_state"] in {"Death", "Locking", "Partial Death"} and record["v1_state"] != record["v2_state"]:
                other_changes += 1
            if (index + 1) % 250 == 0:
                handle.flush()
                print(f"shadow {index + 1}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    table = read_shadow(output)
    focus_details = collect_focus_cases(config, pilot, cost_audit, v1_config)
    summary = summarize_shadow(table, elapsed, other_changes, focus_details)
    write_json(HERE / "shadow_summary.json", summary)
    print(
        f"shadow rows {summary['n_rows']} transitions incoherence_to_partial {summary['incoherence_to_partial_locking']} partial_to_incoherence {summary['partial_locking_to_incoherence']}",
        flush=True,
    )
    return summary


def focus_record(row: dict, omega: np.ndarray, rho: np.ndarray, v1: dict, v2: dict) -> dict:
    return {
        "run_id": row["run_id"],
        "p": float(row["p"]),
        "delta": float(row["delta"]),
        "K": float(row["K"]),
        "v1_state": v1["state"],
        "v2_state": v2["state"],
        "rho_range": float(rho.max() - rho.min()),
        "rho_median": float(np.median(rho)),
        "outside_median_0p01": int(np.sum(np.abs(rho - np.median(rho)) > 0.01)),
        "v1_segments": len(v1["segments"]),
        "v1_constant_segments": sum(item["mode"] == "constant" for item in v1["segments"]),
        "v1_flat_linear": flat_linear_count(v1["segments"]),
        "v1_validated": int(v1["validated_plateau_count"]),
        "v2_plateaus": [
            {
                "start": item["start"],
                "end": item["end"],
                "n": item["n"],
                "relative_omega_span": item["relative_omega_span"],
                "rho_range": item["rho_range"],
                "trend": item["trend"],
                "slope": item["slope"],
                "rho_mean": item["rho_mean"],
            }
            for item in v2["plateaus"]
            if item["nonzero_plateau"]
        ],
    }


def read_shadow(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_shadow(table: list[dict], elapsed: float, _other_from_run: int, focus_details: list[dict]) -> dict:
    transitions = Counter((row["v1_state"], row["v2_state"]) for row in table)
    matrix = {
        left: {right: int(transitions[(left, right)]) for right in STATES}
        for left in STATES
    }
    other = 0
    for row in table:
        if row["v1_state"] in {"Death", "Locking", "Partial Death"} and row["v1_state"] != row["v2_state"]:
            other += 1
    incoherence_to_partial = int(transitions[("Incoherence", "Partial Locking")])
    partial_to_incoherence = int(transitions[("Partial Locking", "Incoherence")])

    def band_delta(value: float) -> str:
        if abs(value) <= 1e-15:
            return "0"
        if value <= 0.10:
            return "(0,0.10]"
        if value <= 0.30:
            return "(0.10,0.30]"
        if value <= 0.60:
            return "(0.30,0.60]"
        return "(0.60,1]"

    def band_k(value: float) -> str:
        if value < 0.50:
            return "[0,0.50)"
        if value < 1.0:
            return "[0.50,1)"
        if value < 2.0:
            return "[1,2)"
        return "[2,3]"

    def stratum(rows: list[dict]) -> dict:
        change_ip = sum(row["v1_state"] == "Incoherence" and row["v2_state"] == "Partial Locking" for row in rows)
        change_pi = sum(row["v1_state"] == "Partial Locking" and row["v2_state"] == "Incoherence" for row in rows)
        return {"n": len(rows), "incoherence_to_partial_locking": int(change_ip), "partial_locking_to_incoherence": int(change_pi)}

    by_p = {}
    by_delta = {}
    by_k = {}
    grouped_p = defaultdict(list)
    grouped_d = defaultdict(list)
    grouped_k = defaultdict(list)
    for row in table:
        grouped_p[f"{float(row['p']):.1f}"].append(row)
        grouped_d[band_delta(float(row["delta"]))].append(row)
        grouped_k[band_k(float(row["K"]))].append(row)
    for key, rows in grouped_p.items():
        by_p[key] = stratum(rows)
    for key, rows in grouped_d.items():
        by_delta[key] = stratum(rows)
    for key, rows in grouped_k.items():
        by_k[key] = stratum(rows)
    stripes = defaultdict(list)
    for row in table:
        stripes[(row["p"], row["delta"])].append(row)
    isolated = 0
    mass = []
    for key, rows in stripes.items():
        rows.sort(key=lambda item: float(item["K"]))
        labels = [item["v2_state"] for item in rows]
        for index, label in enumerate(labels):
            if label != "Partial Locking":
                continue
            left = labels[index - 1] if index > 0 else None
            right = labels[index + 1] if index + 1 < len(labels) else None
            if left not in (None, "Partial Locking") and right not in (None, "Partial Locking") and left != "Partial Locking" and right != "Partial Locking":
                isolated += 1
        incoherence = [item for item in rows if item["v1_state"] == "Incoherence"]
        if len(incoherence) >= 8:
            flipped = sum(item["v2_state"] == "Partial Locking" for item in incoherence)
            fraction = flipped / len(incoherence)
            if fraction >= 0.50:
                mass.append({"p": float(key[0]), "delta": float(key[1]), "fraction": fraction, "n_incoherence": len(incoherence)})
    flat_linear_rows = [row for row in table if int(row["v1_flat_linear_count"]) > 0]
    flat_to_v2 = Counter(row["v2_state"] for row in flat_linear_rows)
    v1_time = float(np.mean([float(row["v1_seconds"]) for row in table]))
    v2_time = float(np.mean([float(row["v2_seconds"]) for row in table]))
    jaccard_values = [float(row["v2_w3_w2_jaccard"]) for row in table]
    return {
        "n_rows": len(table),
        "elapsed_seconds": elapsed,
        "mean_v1_seconds": v1_time,
        "mean_v2_seconds": v2_time,
        "transition_matrix": matrix,
        "incoherence_to_partial_locking": incoherence_to_partial,
        "partial_locking_to_incoherence": partial_to_incoherence,
        "protected_state_changes": other,
        "by_p": by_p,
        "by_delta_band": by_delta,
        "by_K_band": by_k,
        "isolated_partial_locking_cells": isolated,
        "mass_expansion_stripes": mass,
        "v1_flat_linear_runs": len(flat_linear_rows),
        "v1_flat_linear_v2_states": dict(flat_to_v2),
        "median_w3_w2_jaccard": float(np.median(jaccard_values)),
        "focus_cases": focus_details,
        "index_status_counts": dict(Counter(row["index_status"] for row in table)),
    }


def draw_profile(ax, omega: np.ndarray, rho: np.ndarray, intervals: list[tuple[int, int]], title: str) -> None:
    ax.plot(omega, rho, color="0.45", lw=0.8)
    for start, end in intervals:
        ax.axvspan(omega[start], omega[end - 1], color="#E69F00", alpha=0.35, lw=0)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("omega")
    ax.set_ylabel("rotation number")


def make_figure(config: DetectorConfig) -> None:
    examples = []
    specs = [
        ("embedded plateau", {"family": "embedded", "kind": "embedded", "length": 400, "sigma": 1e-3}),
        ("slow slope", {"family": "slow", "kind": "slow", "slope": 0.05, "sigma": 0.0}),
        ("quadratic drift", {"family": "nonlinear", "kind": "quad", "sigma": 0.0}),
        ("near-global tail", {"family": "nearglobal", "kind": "nearglobal", "tail": 30, "sigma": 0.0}),
    ]
    for index, (title, spec) in enumerate(specs):
        item = {
            **spec,
            "delta": DELTA_MAIN,
            "noise": "iid",
            "windows": "none",
            "selectable": False,
            "seed": 7,
            "serial": index,
            "split": "figure",
            "profile_id": f"figure-{index}",
        }
        profile = materialize(item)
        found = accept_plateaus(profile.omega, profile.rho, config)
        intervals = [(item["start"], item["end"]) for item in found if item["nonzero_plateau"]]
        examples.append((title, profile.omega, profile.rho, intervals))
    rows, _split = load_index()
    focus_rows = []
    for delta, coupling in FOCUS:
        match = [
            row for row in rows
            if round(float(row["p"]), 4) == 1.0
            and abs(float(row["delta"]) - delta) < 1e-9
            and abs(float(row["K"]) - coupling) < 1e-9
        ]
        focus_rows.append(match[0])
    from stage2_classifier.classifier_v2.classify import classify_window, load_config
    from stage2_classifier.penalty_calibration.method_c_separated import load_method_c

    pilot, cost_audit = load_method_c()
    v1_config = load_config()
    real_panels = []
    for row in focus_rows:
        omega, rotation = load_arrays(row)
        rho = rotation[2]
        v1 = classify_window(cost_audit, pilot, omega, rho, v1_config)
        v2 = classify_profile(omega, rho, config, delta=float(row["delta"]))
        intervals = [(item["start"], item["end"]) for item in v2["plateaus"] if item["nonzero_plateau"]]
        title = f"p=1  d={float(row['delta']):.2f}  K={float(row['K']):.2f}\n{v1['state']} -> {v2['state']}"
        real_panels.append((title, omega, rho, intervals))
    figure, axes = plt.subplots(3, 3, figsize=(12.4, 9.2))
    panels = examples + real_panels
    for ax, panel in zip(axes.ravel(), panels):
        draw_profile(ax, panel[1], panel[2], panel[3], panel[0])
    figure.suptitle("v2 candidate plateaus on synthetic profiles and p=1 diagnostic cases", fontsize=12)
    figure.tight_layout()
    figure.savefig(HERE / "plateau_detection_benchmark.png", dpi=140)
    plt.close(figure)


def main() -> None:
    result = calibration()
    config = config_from_mapping(result["config"])
    shadow(config)
    make_figure(config)


if __name__ == "__main__":
    main()
