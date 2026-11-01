"""Known piecewise constant/linear rotation-number profiles.

Omega matches a Stage 1 curve with Delta = 0.2 and N = 1000:
``linspace(0.8, 1.2, 1000)``. Pieces meet at the shared endpoint, so the
noiseless signal is continuous. A slope of zero is a constant piece.

Detectability uses the ordinary-least-squares slope t-statistic under the
generating marginal noise, with a threshold fixed before the grid search.
Correlation in AR(1) noise is ignored by that t-statistic; those profiles are
tagged. Model-mismatch profiles do not have a unique piecewise-linear truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

N_OSCILLATORS = 1000
OMEGA = np.linspace(0.8, 1.2, N_OSCILLATORS)
LEVEL0 = 0.75
T_STAT_DISTINGUISHABLE = 3.0
# Stage 2B sigma_hat was about 0.013 on C15 and 0.020 on C23.
# These scales set the synthetic generator. They are not a noise model for the ODEs.
NOISE_LEVELS = (0.0, 0.002, 0.01, 0.03)
CALIBRATION_SEEDS = (0, 1, 2, 3)
VALIDATION_SEEDS = (10, 11, 12)
AR_PHI = 0.9
MASTER_SEED = 20261101

# (length, slope). Lengths in each tuple sum to 1000.
CALIBRATION_STRUCTURES: dict[str, tuple[tuple[int, float], ...]] = {
    "lcl_long": ((300, 1.2), (400, 0.0), (300, 0.8)),
    "lcl_medium": ((440, 1.0), (120, 0.0), (440, 1.1)),
    "lcl_short": ((480, 1.0), (40, 0.0), (480, 0.9)),
    "clc": ((250, 0.0), (500, 1.0), (250, 0.0)),
    "two_plateaus": ((200, 0.0), (200, 1.5), (250, 0.0), (350, 0.6)),
    "three_plateaus": ((150, 0.0), (150, 1.2), (200, 0.0), (150, 0.7), (200, 0.0), (150, 1.0)),
    "pure_flat": ((1000, 0.0),),
    "slope_0p01": ((1000, 0.01),),
    "slope_0p05": ((1000, 0.05),),
    "slope_0p30": ((1000, 0.30),),
    "slope_1": ((1000, 1.0),),
    "short_shallow": ((460, 1.2), (80, 0.30), (460, 0.8)),
}

VALIDATION_STRUCTURES: dict[str, tuple[tuple[int, float], ...]] = {
    "lcl_len180": ((410, 1.0), (180, 0.0), (410, 0.85)),
    "lcl_right": ((550, 1.1), (250, 0.0), (200, 0.7)),
    "slope_0p15": ((1000, 0.15),),
    "slope_0p50": ((1000, 0.50),),
    "lcl_len25": ((488, 1.0), (25, 0.0), (487, 0.9)),
}


@dataclass
class SyntheticProfile:
    profile_id: str
    split: str
    structure: str
    noise_model: str
    noise_sigma: float
    seed: int
    model_mismatch: bool
    omega: np.ndarray
    rho_clean: np.ndarray
    rho: np.ndarray
    segments: list[dict]
    score_included: bool


def _render_pieces(pieces: tuple[tuple[int, float], ...], level0: float) -> tuple[np.ndarray, list[dict]]:
    if sum(length for length, _slope in pieces) != N_OSCILLATORS:
        raise ValueError("piece lengths must sum to 1000")
    signal = np.empty(N_OSCILLATORS, dtype=float)
    segments = []
    cursor = 0
    level = float(level0)
    for length, slope in pieces:
        start = cursor
        end = cursor + int(length)
        signal[start:end] = level + float(slope) * (OMEGA[start:end] - OMEGA[start])
        segments.append(
            {
                "index_start": start,
                "index_end": end,
                "slope": float(slope),
                "segment_type": "constant" if float(slope) == 0.0 else "linear",
                "omega_start": float(OMEGA[start]),
                "omega_end": float(OMEGA[end - 1]),
                "level_start": float(signal[start]),
            }
        )
        level = float(signal[end - 1])
        cursor = end
    return signal, segments


def _quadratic_with_plateau() -> tuple[np.ndarray, list[dict]]:
    """Smooth quadratic background with one inserted constant block.

    The background is not a piecewise-linear truth, so boundary metrics are
    not defined. The inserted block is still a constant region.
    """
    centered = OMEGA - 1.0
    signal = 0.70 + 0.35 * centered + 6.0 * centered**2
    start, end = 420, 620
    signal[start:end] = float(np.mean(signal[start:end]))
    segments = [
        {
            "index_start": start,
            "index_end": end,
            "slope": 0.0,
            "segment_type": "constant",
            "omega_start": float(OMEGA[start]),
            "omega_end": float(OMEGA[end - 1]),
            "level_start": float(signal[start]),
            "note": "embedded plateau; surrounding quadratic has no unique linear partition",
        }
    ]
    return signal, segments


def _noise_vector(sigma: float, seed: int, noise_model: str, structure_index: int, noise_index: int) -> np.ndarray:
    if sigma == 0.0:
        return np.zeros(N_OSCILLATORS, dtype=float)
    rng = np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, structure_index, noise_index, int(seed), 1 if noise_model == "ar1" else 0])
    )
    if noise_model == "gaussian":
        return sigma * rng.normal(size=N_OSCILLATORS)
    if noise_model == "ar1":
        phi = AR_PHI
        innovation_scale = sigma * math.sqrt(1.0 - phi * phi)
        eps = innovation_scale * rng.normal(size=N_OSCILLATORS)
        out = np.empty(N_OSCILLATORS, dtype=float)
        out[0] = sigma * rng.normal()
        for index in range(1, N_OSCILLATORS):
            out[index] = phi * out[index - 1] + eps[index]
        return out
    raise ValueError(noise_model)


def _annotate_detectability(segments: list[dict], sigma: float) -> None:
    for segment in segments:
        start = segment["index_start"]
        end = segment["index_end"]
        slope = float(segment["slope"])
        width = float(OMEGA[end - 1] - OMEGA[start])
        count = end - start
        x = OMEGA[start:end]
        ssx = float(np.sum((x - x.mean()) ** 2))
        delta_rho = abs(slope) * width
        if slope == 0.0:
            role = "plateau"
            t_stat = math.inf
        elif sigma == 0.0 or ssx <= 0.0:
            role = "distinguishable_slope"
            t_stat = math.inf
        else:
            t_stat = abs(slope) / (sigma / math.sqrt(ssx))
            role = "distinguishable_slope" if t_stat >= T_STAT_DISTINGUISHABLE else "unresolved_slope"
        segment["delta_rho"] = delta_rho
        segment["t_stat"] = t_stat
        segment["detectability"] = role
        segment["oscillator_count"] = count


def _score_included(segments: list[dict], model_mismatch: bool) -> bool:
    if model_mismatch:
        return False
    n_plateau = sum(
        segment["index_end"] - segment["index_start"]
        for segment in segments
        if segment["detectability"] == "plateau"
    )
    n_distinguishable = sum(
        segment["index_end"] - segment["index_start"]
        for segment in segments
        if segment["detectability"] == "distinguishable_slope"
    )
    return n_plateau > 0 or n_distinguishable > 0


def _make(
    split: str,
    structure: str,
    pieces: tuple[tuple[int, float], ...] | None,
    sigma: float,
    seed: int,
    noise_model: str,
    structure_index: int,
    noise_index: int,
    model_mismatch: bool,
) -> SyntheticProfile:
    if model_mismatch:
        clean, segments = _quadratic_with_plateau()
    else:
        clean, segments = _render_pieces(pieces, LEVEL0)
    _annotate_detectability(segments, sigma)
    rho = clean + _noise_vector(sigma, seed, noise_model, structure_index, noise_index)
    seed_label = 0 if sigma == 0.0 else int(seed)
    profile_id = f"{structure}__{noise_model}__sigma_{sigma:.3f}__seed_{seed_label}"
    return SyntheticProfile(
        profile_id=profile_id,
        split=split,
        structure=structure,
        noise_model=noise_model,
        noise_sigma=float(sigma),
        seed=seed_label,
        model_mismatch=model_mismatch,
        omega=OMEGA,
        rho_clean=clean,
        rho=rho,
        segments=segments,
        score_included=_score_included(segments, model_mismatch),
    )


def build_profiles(calibration_seeds=CALIBRATION_SEEDS, validation_seeds=VALIDATION_SEEDS) -> list[SyntheticProfile]:
    profiles: list[SyntheticProfile] = []
    for structure_index, (name, pieces) in enumerate(CALIBRATION_STRUCTURES.items()):
        for noise_index, sigma in enumerate(NOISE_LEVELS):
            seeds = (0,) if sigma == 0.0 else calibration_seeds
            for seed in seeds:
                profiles.append(
                    _make("calibration", name, pieces, sigma, seed, "gaussian", structure_index, noise_index, False)
                )
    offset = len(CALIBRATION_STRUCTURES)
    for structure_index, (name, pieces) in enumerate(VALIDATION_STRUCTURES.items(), start=offset):
        for noise_index, sigma in enumerate(NOISE_LEVELS):
            seeds = (0,) if sigma == 0.0 else validation_seeds
            for seed in seeds:
                profiles.append(
                    _make("validation", name, pieces, sigma, seed, "gaussian", structure_index, noise_index, False)
                )
    ar_structures = ("lcl_long", "slope_0p30")
    for extra, name in enumerate(ar_structures):
        pieces = CALIBRATION_STRUCTURES[name]
        for noise_index, sigma in enumerate(NOISE_LEVELS):
            if sigma == 0.0:
                continue
            for seed in validation_seeds:
                profiles.append(
                    _make(
                        "validation",
                        f"{name}_ar1",
                        pieces,
                        sigma,
                        seed,
                        "ar1",
                        offset + len(VALIDATION_STRUCTURES) + extra,
                        noise_index,
                        False,
                    )
                )
    mismatch_index = offset + len(VALIDATION_STRUCTURES) + len(ar_structures)
    for noise_index, sigma in enumerate(NOISE_LEVELS):
        seeds = (0,) if sigma == 0.0 else validation_seeds
        for seed in seeds:
            profiles.append(
                _make(
                    "validation",
                    "mismatch_quad_plateau",
                    None,
                    sigma,
                    seed,
                    "gaussian",
                    mismatch_index,
                    noise_index,
                    True,
                )
            )
    ids = [profile.profile_id for profile in profiles]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate synthetic profile id")
    return profiles


def truth_labels(profile: SyntheticProfile) -> np.ndarray:
    """0 plateau, 1 distinguishable slope, 2 unresolved slope, -1 no unique truth."""
    labels = np.full(N_OSCILLATORS, -1, dtype=int)
    if profile.model_mismatch:
        for segment in profile.segments:
            if segment["segment_type"] == "constant":
                labels[segment["index_start"]:segment["index_end"]] = 0
        return labels
    code = {"plateau": 0, "distinguishable_slope": 1, "unresolved_slope": 2}
    for segment in profile.segments:
        labels[segment["index_start"]:segment["index_end"]] = code[segment["detectability"]]
    if np.any(labels < 0):
        raise RuntimeError(f"{profile.profile_id} has unlabeled oscillators")
    return labels
