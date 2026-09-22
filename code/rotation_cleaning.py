"""
Rotation-number profile cleaning (Professor K's suggestions).

Applied at plot/classify time only. Raw npz files are never modified unless
you explicitly pass --save-cleaned-npz, which writes copies to a sibling
*_cleaned/ directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CleanParams:
    """How to preprocess rho(omega) before classification/plotting."""

    method: str = "none"  # none | monotone | monotone_smooth
    smooth_window: int = 5  # odd integer; used only for monotone_smooth


def cleaned_data_dir(source_raw_dir: Path) -> Path:
    """Sibling directory for optional cleaned npz copies."""
    return source_raw_dir.parent / f"{source_raw_dir.name}_cleaned"


def sort_by_omega(
    omega: np.ndarray, rotation_number: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = np.asarray(omega, dtype=float)
    rho = np.asarray(rotation_number, dtype=float)
    order = np.argsort(omega)
    return omega[order], rho[order], order


def restore_original_order(values_sorted: np.ndarray, order: np.ndarray) -> np.ndarray:
    restored = np.empty_like(values_sorted)
    restored[order] = values_sorted
    return restored


def monotone_carry_forward_sorted(rho_sorted: np.ndarray) -> np.ndarray:
    """
    K's rule: after sorting by omega, if rho drops, raise it to the previous value.

    Walk left-to-right along increasing omega. Whenever rho[i] < rho[i-1], set
    rho[i] = rho[i-1]. This removes downward spikes/noise while preserving the
    overall upward trend expected in incoherence / partial-locking profiles.
    """
    rho = np.asarray(rho_sorted, dtype=float).copy()
    for i in range(1, len(rho)):
        if rho[i] < rho[i - 1]:
            rho[i] = rho[i - 1]
    return rho


def moving_average_sorted(rho_sorted: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average on an already omega-sorted profile."""
    rho = np.asarray(rho_sorted, dtype=float)
    n = len(rho)
    if n == 0:
        return rho.copy()
    if window < 1 or window % 2 == 0:
        raise ValueError(f"smooth_window must be a positive odd integer, got {window}")
    if window == 1 or n == 1:
        return rho.copy()

    half = window // 2
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(rho, (half, half), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(float, copy=False)


def clean_rotation_profile(
    omega: np.ndarray,
    rotation_number: np.ndarray,
    params: CleanParams | None = None,
) -> np.ndarray:
    """
    Return cleaned rotation numbers in the original oscillator index order.

    Steps (when enabled):
      1. sort by omega
      2. monotone carry-forward (optional)
      3. moving-average smooth on sorted profile (optional)
      4. restore original index order
    """
    if params is None:
        params = CleanParams()

    method = params.method
    if method in ("none", "", "off"):
        return np.asarray(rotation_number, dtype=float).copy()

    _, rho_sorted, order = sort_by_omega(omega, rotation_number)

    if method in ("monotone", "monotone_smooth"):
        rho_sorted = monotone_carry_forward_sorted(rho_sorted)
    else:
        raise ValueError(
            f"Unknown clean method {method!r}. "
            "Use none, monotone, or monotone_smooth."
        )

    if method == "monotone_smooth":
        rho_sorted = moving_average_sorted(rho_sorted, params.smooth_window)
        rho_sorted = monotone_carry_forward_sorted(rho_sorted)

    return restore_original_order(rho_sorted, order)


def cleaning_label(params: CleanParams) -> str:
    if params.method in ("none", "", "off"):
        return "raw"
    if params.method == "monotone":
        return "monotone"
    if params.method == "monotone_smooth":
        return f"monotone+ma{params.smooth_window}"
    return params.method


def default_cleaned_figure_dir(base_figure_dir: Path, params: CleanParams) -> Path:
    if params.method in ("none", "", "off"):
        return base_figure_dir
    return base_figure_dir.parent / f"{base_figure_dir.name}_{cleaning_label(params)}"


def save_cleaned_trial_copy(
    source_path: Path,
    output_dir: Path,
    omega: np.ndarray,
    rotation_number_clean: np.ndarray,
    clean_params: CleanParams,
) -> Path:
    """
    Write a copy of the trial npz with cleaned rotation_number.

    Original files are never touched. Metadata records the cleaning recipe.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / source_path.name
    with np.load(source_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
        rotation_number_raw = np.asarray(data["rotation_number"], dtype=float)
    payload["rotation_number"] = rotation_number_clean
    payload["rotation_number_raw"] = rotation_number_raw
    payload["cleaning_method"] = np.array(cleaning_label(clean_params))
    np.savez_compressed(out_path, **payload)
    return out_path
