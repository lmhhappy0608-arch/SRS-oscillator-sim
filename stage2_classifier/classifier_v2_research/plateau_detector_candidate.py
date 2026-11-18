"""Candidate plateau detector for Classifier v2.

Discovery and acceptance are separate. Discovery partitions a frequency-sorted
rotation-number profile into maximal ordinary-least-squares slope-homogeneous
segments. Acceptance then keeps a segment only when its size, relative
frequency span, rotation-number range, and absolute trend are all inside
fixed limits. Method C constant/linear labels are not used, and the
acceptance limits are not scaled by the Method C sigma_hat floor.

The numbers in ``FROZEN`` are the calibration result. The selection rule that
produced them is in ``calibrate_plateau_detector.py`` and was fixed before the
grid was scored. Death, Locking, Partial Death, the near-zero cut, and the
W3 priority order are the archived v1 rules and are not retuned here.

Indices on every returned segment are positions in the frequency-sorted
profile, not the saved oscillator order.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

VERSION = "v2.0.0-candidate"
NEAR_ZERO = 0.001
LOCK_RANGE = 0.01
PARTIAL_DEATH_RUN = 15
NONZERO_LEVEL = 0.001

# Filled after calibration. Do not treat these as inherited optima from v1.
FROZEN = {
    'min_segment': 8,
    'split_rss': 0.006853319119303251,
    'peel_max': 0,
    'peel_drop': 0.01,
    'n_min': 15,
    'l_min': 0.02,
    'r_max': 0.05,
    's_max': 0.005,
    'nonzero_level': 0.001,
    'near_zero': 0.001,
    'lock_range': 0.01,
    'partial_death_run': 15,
}


@dataclass(frozen=True)
class DetectorConfig:
    """Discovery parameters and acceptance limits.

    ``min_segment``, ``split_rss``, ``peel_max`` and ``peel_drop`` define
    candidate intervals. ``n_min``, ``l_min``, ``r_max`` and ``s_max`` decide
    which of those intervals are plateaus. The remaining fields reproduce
    archived state rules and are not calibration targets.
    """

    min_segment: int = 12
    split_rss: float = 1e-3
    peel_max: int = 8
    peel_drop: float = 0.01
    n_min: int = 40
    l_min: float = 0.02
    r_max: float = 0.02
    s_max: float = 0.02
    nonzero_level: float = NONZERO_LEVEL
    near_zero: float = NEAR_ZERO
    lock_range: float = LOCK_RANGE
    partial_death_run: int = PARTIAL_DEATH_RUN

    @classmethod
    def frozen(cls) -> "DetectorConfig":
        path = Path(__file__).with_name("candidate_parameters.json")
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return config_from_mapping(payload["config"])
        return cls(**FROZEN)


def config_from_mapping(mapping: dict) -> DetectorConfig:
    fields = DetectorConfig.__dataclass_fields__
    return DetectorConfig(**{key: mapping[key] for key in fields})


def sort_profile(omega: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a nondecreasing frequency order, stable under ties."""
    omega = np.asarray(omega, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if omega.shape != rho.shape or omega.ndim != 1:
        raise ValueError("omega and rho must be 1-D arrays of the same length")
    if omega.size and np.any(np.diff(omega) < 0):
        order = np.argsort(omega, kind="mergesort")
        return omega[order], rho[order]
    return omega, rho


def _prefixes(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    n = int(y.size)
    cx = np.empty(n + 1, dtype=np.float64)
    cy = np.empty(n + 1, dtype=np.float64)
    cxx = np.empty(n + 1, dtype=np.float64)
    cxy = np.empty(n + 1, dtype=np.float64)
    cyy = np.empty(n + 1, dtype=np.float64)
    cx[0] = cy[0] = cxx[0] = cxy[0] = cyy[0] = 0.0
    cx[1:] = np.cumsum(x)
    cy[1:] = np.cumsum(y)
    cxx[1:] = np.cumsum(x * x)
    cxy[1:] = np.cumsum(x * y)
    cyy[1:] = np.cumsum(y * y)
    return cx, cy, cxx, cxy, cyy


def _rss_from_sums(
    n: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    varx = sxx - sx * sx / n
    vary = syy - sy * sy / n
    cov = sxy - sx * sy / n
    # A repeated frequency has no slope. RSS is then the constant-model RSS.
    rss = np.where(varx > 1e-18, vary - (cov * cov) / np.maximum(varx, 1e-30), vary)
    return np.maximum(rss, 0.0)


def _interval_sums(pref: tuple[np.ndarray, ...], start: int, end: int) -> tuple[float, ...]:
    cx, cy, cxx, cxy, cyy = pref
    n = end - start
    return (
        float(n),
        float(cx[end] - cx[start]),
        float(cy[end] - cy[start]),
        float(cxx[end] - cxx[start]),
        float(cxy[end] - cxy[start]),
        float(cyy[end] - cyy[start]),
    )


def _best_split(
    pref: tuple[np.ndarray, ...],
    start: int,
    end: int,
    min_segment: int,
) -> tuple[float, int | None]:
    """Return the largest RSS reduction from one linear fit to two, and its cut.

    The cut index is the first point of the right piece. Equal gains keep the
    earliest cut.
    """
    if end - start < 2 * min_segment:
        return 0.0, None
    cuts = np.arange(start + min_segment, end - min_segment + 1, dtype=int)
    cx, cy, cxx, cxy, cyy = pref
    n_all, sx, sy, sxx, sxy, syy = _interval_sums(pref, start, end)
    rss_all = _rss_from_sums(
        np.array([n_all]),
        np.array([sx]),
        np.array([sy]),
        np.array([sxx]),
        np.array([sxy]),
        np.array([syy]),
    )[0]
    n_left = cuts - start
    n_right = end - cuts
    rss_left = _rss_from_sums(
        n_left,
        cx[cuts] - cx[start],
        cy[cuts] - cy[start],
        cxx[cuts] - cxx[start],
        cxy[cuts] - cxy[start],
        cyy[cuts] - cyy[start],
    )
    rss_right = _rss_from_sums(
        n_right,
        cx[end] - cx[cuts],
        cy[end] - cy[cuts],
        cxx[end] - cxx[cuts],
        cxy[end] - cxy[cuts],
        cyy[end] - cyy[cuts],
    )
    gain = rss_all - rss_left - rss_right
    index = int(np.argmax(gain))
    return float(gain[index]), int(cuts[index])


def slope_homogeneous_segments(
    omega: np.ndarray,
    rho: np.ndarray,
    min_segment: int,
    split_rss: float,
) -> list[tuple[int, int]]:
    """Partition ``[0, n)`` into maximal segments.

    A split is kept only when it reduces linear RSS by more than ``split_rss``.
    The threshold is an absolute residual sum of squares, not a multiple of
    sigma_hat. A pure line, and a numerically flat profile, therefore stay one
    segment.
    """
    n = int(rho.size)
    if n == 0:
        return []
    if min_segment < 2:
        raise ValueError("min_segment must be at least 2")
    if n < 2 * min_segment:
        return [(0, n)]
    pref = _prefixes(omega, rho)
    leaves: list[tuple[int, int]] = []
    stack = [(0, n)]
    while stack:
        start, end = stack.pop()
        gain, cut = _best_split(pref, start, end, min_segment)
        if cut is None or gain <= float(split_rss):
            leaves.append((start, end))
            continue
        stack.append((cut, end))
        stack.append((start, cut))
    leaves.sort()
    return _merge_homogeneous(omega, rho, leaves, float(split_rss))


def _rss_scalar(pref: tuple[np.ndarray, ...], start: int, end: int) -> float:
    sums = _interval_sums(pref, start, end)
    return float(_rss_from_sums(np.array([sums[0]]), np.array([sums[1]]), np.array([sums[2]]), np.array([sums[3]]), np.array([sums[4]]), np.array([sums[5]]))[0])


def _merge_homogeneous(
    omega: np.ndarray,
    rho: np.ndarray,
    leaves: list[tuple[int, int]],
    split_rss: float,
) -> list[tuple[int, int]]:
    """Join adjacent pieces when the cut between them does not meet ``split_rss``.

    Binary segmentation can place its first cut inside a flat region because
    that cut is the best single split of a larger mixed interval. A following
    pass keeps a cut only when that cut, by itself, reduces linear RSS by
    more than the discovery threshold.
    """
    if len(leaves) < 2:
        return leaves
    pref = _prefixes(omega, rho)
    merged: list[tuple[int, int]] = [leaves[0]]
    for start, end in leaves[1:]:
        previous_start, previous_end = merged[-1]
        if previous_end != start:
            merged.append((start, end))
            continue
        gain = _rss_scalar(pref, previous_start, end) - _rss_scalar(pref, previous_start, start) - _rss_scalar(pref, start, end)
        if gain <= split_rss:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _geometry(omega: np.ndarray, rho: np.ndarray, start: int, end: int) -> dict:
    xs = omega[start:end]
    ys = rho[start:end]
    n = int(end - start)
    full = float(omega[-1] - omega[0]) if omega.size else 0.0
    length = float(xs[-1] - xs[0]) if n else 0.0
    relative = float(length / full) if full > 0.0 else float("nan")
    center_x = float(xs.mean()) if n else float("nan")
    center_y = float(ys.mean()) if n else float("nan")
    varx = float(np.dot(xs - center_x, xs - center_x)) if n else 0.0
    if n < 2 or varx <= 0.0:
        slope = float("nan")
        trend = float("nan")
        fitted = np.full(n, center_y, dtype=np.float64)
    else:
        slope = float(np.dot(xs - center_x, ys - center_y) / varx)
        trend = float(abs(slope) * length)
        fitted = center_y + slope * (xs - center_x)
    constant_resid = ys - center_y
    linear_resid = ys - fitted
    diffs = np.diff(ys) if n >= 2 else np.empty(0, dtype=np.float64)
    if diffs.size == 0 or not np.isfinite(slope) or abs(slope) <= 1e-15:
        monotonicity = float("nan")
    else:
        signed = np.sign(diffs)
        monotonicity = float(np.mean(signed == np.sign(slope)))
    return {
        "start": int(start),
        "end": int(end),
        "n": n,
        "omega_start": float(xs[0]) if n else float("nan"),
        "omega_end": float(xs[-1]) if n else float("nan"),
        "omega_span": length,
        "relative_omega_span": relative,
        "rho_mean": center_y,
        "rho_median": float(np.median(ys)) if n else float("nan"),
        "rho_range": float(ys.max() - ys.min()) if n else float("nan"),
        "slope": slope,
        "trend": trend,
        "rss_constant": float(np.dot(constant_resid, constant_resid)) if n else 0.0,
        "rss_linear": float(np.dot(linear_resid, linear_resid)) if n else 0.0,
        "monotonicity": monotonicity,
    }


def _edge_contrast(rho: np.ndarray, start: int, end: int, width: int = 8) -> dict:
    """Compare the inside edge of a candidate with the adjacent oscillators."""
    width = max(2, min(width, end - start))
    out = {"left_jump": float("nan"), "right_jump": float("nan")}
    if start >= width:
        inside = float(np.median(rho[start : start + width]))
        outside = float(np.median(rho[start - width : start]))
        out["left_jump"] = abs(inside - outside)
    if end + width <= rho.size:
        inside = float(np.median(rho[end - width : end]))
        outside = float(np.median(rho[end : end + width]))
        out["right_jump"] = abs(inside - outside)
    return out


def _neighbor_slopes(segments: list[dict], index: int) -> dict:
    def slope_of(item: dict | None) -> float:
        if item is None or not np.isfinite(item["slope"]):
            return float("nan")
        return float(item["slope"])

    left = segments[index - 1] if index > 0 else None
    right = segments[index + 1] if index + 1 < len(segments) else None
    return {"left_neighbor_slope": slope_of(left), "right_neighbor_slope": slope_of(right)}


def peel_endpoints(
    rho: np.ndarray,
    start: int,
    end: int,
    peel_max: int,
    peel_drop: float,
) -> tuple[int, int, int]:
    """Drop endpoints that each account for at least ``peel_drop`` of the range.

    At most ``peel_max`` points are removed, from either end. This only moves
    a boundary. It does not search interior subintervals.
    """
    removed = 0
    start = int(start)
    end = int(end)
    while removed < int(peel_max) and end - start >= 3:
        piece = rho[start:end]
        full = float(piece.max() - piece.min())
        drop_left = full - float(piece[1:].max() - piece[1:].min())
        drop_right = full - float(piece[:-1].max() - piece[:-1].min())
        if drop_left >= drop_right and drop_left >= float(peel_drop):
            start += 1
            removed += 1
            continue
        if drop_right > drop_left and drop_right >= float(peel_drop):
            end -= 1
            removed += 1
            continue
        break
    return start, end, removed


def discover_candidates(omega: np.ndarray, rho: np.ndarray, config: DetectorConfig) -> list[dict]:
    """Return slope-homogeneous candidates after endpoint peeling.

    An empty list means the frequency span is zero, so a slope is undefined
    and no frequency-locking plateau is proposed.
    """
    if omega.size == 0:
        return []
    full = float(omega[-1] - omega[0])
    if not np.isfinite(full) or full <= 0.0:
        return []
    raw = slope_homogeneous_segments(omega, rho, int(config.min_segment), float(config.split_rss))
    candidates = []
    for start, end in raw:
        peeled_start, peeled_end, removed = peel_endpoints(
            rho, start, end, int(config.peel_max), float(config.peel_drop)
        )
        if peeled_end - peeled_start < 2:
            continue
        item = _geometry(omega, rho, peeled_start, peeled_end)
        item["raw_start"] = int(start)
        item["raw_end"] = int(end)
        item["peeled_points"] = int(removed)
        item.update(_edge_contrast(rho, peeled_start, peeled_end))
        candidates.append(item)
    for index, item in enumerate(candidates):
        item.update(_neighbor_slopes(candidates, index))
    return candidates


def _passes_geometry(item: dict, config: DetectorConfig) -> bool:
    relative = item["relative_omega_span"]
    trend = item["trend"]
    return bool(
        item["n"] >= int(config.n_min)
        and np.isfinite(relative)
        and relative >= float(config.l_min)
        and np.isfinite(item["rho_range"])
        and item["rho_range"] <= float(config.r_max)
        and np.isfinite(trend)
        and trend <= float(config.s_max)
    )


def merge_accepted(omega: np.ndarray, rho: np.ndarray, accepted: list[dict], config: DetectorConfig) -> list[dict]:
    """Merge adjacent accepted plateaus when the contiguous union still passes.

    The union includes the oscillators between the two intervals. A gap that
    inflates the range or the trend blocks the merge, so the two plateaus
    stay separate instead of hiding that gap.
    """
    if len(accepted) < 2:
        return accepted
    ordered = sorted(accepted, key=lambda item: (item["start"], item["end"]))
    merged: list[dict] = [ordered[0]]
    for item in ordered[1:]:
        previous = merged[-1]
        if item["start"] < previous["end"]:
            merged.append(item)
            continue
        union = _geometry(omega, rho, previous["start"], item["end"])
        if _passes_geometry(union, config):
            union["raw_start"] = int(previous.get("raw_start", previous["start"]))
            union["raw_end"] = int(item.get("raw_end", item["end"]))
            union["peeled_points"] = int(previous.get("peeled_points", 0) + item.get("peeled_points", 0))
            union.update(_edge_contrast(rho, union["start"], union["end"]))
            merged[-1] = union
        else:
            merged.append(item)
    return merged


def accept_plateaus(omega: np.ndarray, rho: np.ndarray, config: DetectorConfig) -> list[dict]:
    """Discover candidates, then apply the acceptance limits.

    ``plateau`` is the geometry decision. ``nonzero_plateau`` additionally
    requires the archived nonzero level. Near-zero plateaus are retained in
    the list and do not by themselves assign Partial Locking.
    """
    omega, rho = sort_profile(omega, rho)
    found = []
    for item in discover_candidates(omega, rho, config):
        item["plateau"] = _passes_geometry(item, config)
        item["nonzero_plateau"] = bool(
            item["plateau"] and abs(item["rho_mean"]) >= float(config.nonzero_level)
        )
        if item["plateau"]:
            found.append(item)
    accepted = merge_accepted(omega, rho, found, config)
    for index, item in enumerate(accepted):
        item["plateau"] = True
        item["nonzero_plateau"] = bool(abs(item["rho_mean"]) >= float(config.nonzero_level))
        item["residual_ratio"] = float(
            item["rss_linear"] / item["rss_constant"] if item["rss_constant"] > 0.0 else 0.0
        )
        item.update(_neighbor_slopes(accepted, index))
    return accepted


def plateau_mask(n: int, plateaus: list[dict], *, nonzero_only: bool = True) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for item in plateaus:
        if nonzero_only and not item["nonzero_plateau"]:
            continue
        mask[int(item["start"]) : int(item["end"])] = True
    return mask


def consecutive_near_zero(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def classify_profile(
    omega: np.ndarray,
    rho: np.ndarray,
    config: DetectorConfig | None = None,
    *,
    delta: float | None = None,
) -> dict:
    """Assign one window state.

    Priority is Death, Locking, Partial Death, Partial Locking, Incoherence.
    A zero frequency span, including every delta = 0 profile, produces no
    plateau candidates. Locking still uses the full-profile range, so a large
    flat body with a few outliers is not promoted to Locking.
    """
    config = DetectorConfig.frozen() if config is None else config
    omega, rho = sort_profile(omega, rho)
    near = np.abs(rho) < float(config.near_zero)
    near_fraction = float(near.mean()) if rho.size else 0.0
    run = consecutive_near_zero(near)
    profile_range = float(rho.max() - rho.min()) if rho.size else 0.0
    frequency_span = float(omega[-1] - omega[0]) if omega.size else 0.0
    degenerate = frequency_span <= 0.0 or (delta is not None and abs(float(delta)) <= 1e-15)
    plateaus = [] if degenerate else accept_plateaus(omega, rho, config)
    nonzero = [item for item in plateaus if item["nonzero_plateau"]]
    if near_fraction >= 1.0:
        state = "Death"
    elif profile_range <= float(config.lock_range):
        state = "Locking"
    elif run >= int(config.partial_death_run):
        state = "Partial Death"
    elif nonzero:
        state = "Partial Locking"
    else:
        state = "Incoherence"
    if state == "Locking":
        locked_fraction = 1.0
    elif rho.size == 0:
        locked_fraction = 0.0
    else:
        locked_fraction = float(plateau_mask(rho.size, plateaus).mean())
    return {
        "state": state,
        "near_zero_fraction": near_fraction,
        "consecutive_near_zero": run,
        "rho_range": profile_range,
        "frequency_span": frequency_span,
        "degenerate_frequency": bool(degenerate),
        "locked_fraction": locked_fraction,
        "plateau_count": len(nonzero),
        "near_zero_plateau_count": sum(not item["nonzero_plateau"] for item in plateaus),
        "plateaus": plateaus,
    }


def review_windows(window_results: dict[str, dict]) -> tuple[int, str]:
    """Match the archived review flag: disagreement is recorded, not a veto."""
    names = ("W1", "W2", "W3")
    states = [window_results[name]["state"] for name in names]
    counts = [window_results[name]["plateau_count"] for name in names]
    reasons = []
    if len(set(states)) > 1:
        reasons.append("temporal_state_disagreement")
    if len(set(counts)) > 1:
        reasons.append("temporal_plateau_count_disagreement")
    return int(bool(reasons)), ";".join(reasons)


def frozen_payload() -> dict:
    return {
        "version": VERSION,
        "config": asdict(DetectorConfig.frozen()),
        "acceptance": [
            "n >= n_min",
            "relative omega span >= l_min",
            "rho range <= r_max",
            "abs(slope) * omega span <= s_max",
            "nonzero plateau also requires abs(mean rho) >= nonzero_level",
        ],
        "not_used": [
            "Method C constant/linear label",
            "sigma_hat scaling of acceptance limits",
            "W1/W2 veto of the W3 state",
        ],
    }
