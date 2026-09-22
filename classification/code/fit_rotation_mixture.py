#!/usr/bin/env python3
"""
Fit a three-line mixture to each trial's (omega, rho) cloud.

Each oscillator is a sample. The three geometric components are
    death:  rho ~ N(0, sigma_d^2)
    lock:   rho ~ N(c, sigma_l^2)          (horizontal plateau)
    drift:  rho ~ N(omega, sigma_i^2)      (incoherent identity)

Trial-level weights (w_death, w_lock, w_drift) are the mixture
responsibilities. No |rho|<0.01 or plateau-length cut is used to
assign the global state.

Example:
  python3 code/fit_rotation_mixture.py
  python3 code/fit_rotation_mixture.py --p 0.2 0.4 0.6 0.8 1.0
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import project_paths  # noqa: F401  # adds Cursor/code to sys.path
from project_paths import (
    CLASSIFICATION_ROOT,
    CURSOR_ROOT,
    DEFAULT_MIXTURE_FIGURE_DIR,
    DEFAULT_P_VALUES,
    DEFAULT_RAW_DIR,
    DEFAULT_TABLE_DIR,
    resolve_user_path,
)
from plot_rotation_vs_frequency import (
    STATE_COLORS,
    ClassifyParams,
    classify_state,
)
from rotation_cleaning import CleanParams, clean_rotation_profile, cleaning_label

FILENAME_PATTERN = re.compile(
    r"K_([\d.]+)_delta_([\d.]+)_p_([\d.]+)_trial_(\d+)\.npz$"
)
COMPONENT_NAMES = ("death", "lock", "drift")
COMPONENT_COLORS = {
    "death": STATE_COLORS["Death"],
    "lock": STATE_COLORS["Locking"],
    "drift": STATE_COLORS["Incoherence"],
}
LOG_2PI = np.log(2.0 * np.pi)
VAR_FLOOR = 1e-12


@dataclass(frozen=True)
class MixtureFit:
    weights: np.ndarray  # (3,) death, lock, drift
    lock_center: float
    variances: np.ndarray  # (3,)
    loglik: float
    n_iter: int
    gamma: np.ndarray  # (n, 3)


def _log_normal(residual: np.ndarray, var: float) -> np.ndarray:
    return -0.5 * (LOG_2PI + np.log(var) + residual * residual / var)


def _lock_inits(omega: np.ndarray, rho: np.ndarray) -> list[float]:
    inits = [float(np.median(rho))]
    abs_rho = np.abs(rho)
    high = abs_rho >= np.median(abs_rho)
    if np.any(high):
        inits.append(float(np.median(rho[high])))
    inits.append(float(np.mean(omega)))
    # unique while preserving order
    unique: list[float] = []
    for value in inits:
        if not any(np.isclose(value, seen, atol=1e-12) for seen in unique):
            unique.append(value)
    return unique


def fit_three_line_mixture(
    omega: np.ndarray,
    rho: np.ndarray,
    n_iter: int = 80,
    tol: float = 1e-8,
) -> MixtureFit:
    """EM for rho | omega ~ mix of N(0), N(c), N(omega)."""
    omega = np.asarray(omega, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if omega.shape != rho.shape:
        raise ValueError("omega and rho must have the same shape")
    if omega.size == 0:
        raise ValueError("empty oscillator sample")

    best: MixtureFit | None = None
    for c0 in _lock_inits(omega, rho):
        fit = _em_once(omega, rho, c0, n_iter=n_iter, tol=tol)
        if best is None or fit.loglik > best.loglik:
            best = fit
    assert best is not None
    return _merge_lock_into_death_if_at_zero(best)


def _em_once(
    omega: np.ndarray,
    rho: np.ndarray,
    c0: float,
    n_iter: int,
    tol: float,
) -> MixtureFit:
    n = rho.size
    pi = np.full(3, 1.0 / 3.0)
    c = float(c0)
    var_d = max(float(np.mean(rho * rho)), VAR_FLOOR)
    var_l = max(float(np.mean((rho - c) ** 2)), VAR_FLOOR)
    var_i = max(float(np.mean((rho - omega) ** 2)), VAR_FLOOR)

    loglik = -np.inf
    gamma = np.full((n, 3), 1.0 / 3.0)
    n_used = 0
    for n_used in range(1, n_iter + 1):
        logp = np.column_stack(
            (
                np.log(pi[0] + 1e-300) + _log_normal(rho, var_d),
                np.log(pi[1] + 1e-300) + _log_normal(rho - c, var_l),
                np.log(pi[2] + 1e-300) + _log_normal(rho - omega, var_i),
            )
        )
        shift = np.max(logp, axis=1, keepdims=True)
        unnorm = np.exp(logp - shift)
        denom = unnorm.sum(axis=1, keepdims=True)
        gamma = unnorm / denom
        loglik_new = float(np.sum(shift + np.log(denom)))

        counts = gamma.sum(axis=0) + 1e-12
        pi = counts / counts.sum()
        c = float(np.dot(gamma[:, 1], rho) / counts[1])
        var_d = max(float(np.dot(gamma[:, 0], rho * rho) / counts[0]), VAR_FLOOR)
        var_l = max(float(np.dot(gamma[:, 1], (rho - c) ** 2) / counts[1]), VAR_FLOOR)
        var_i = max(
            float(np.dot(gamma[:, 2], (rho - omega) ** 2) / counts[2]), VAR_FLOOR
        )

        if abs(loglik_new - loglik) < tol:
            loglik = loglik_new
            break
        loglik = loglik_new

    return MixtureFit(
        weights=pi.copy(),
        lock_center=c,
        variances=np.array([var_d, var_l, var_i], dtype=float),
        loglik=loglik,
        n_iter=n_used,
        gamma=gamma,
    )


def _merge_lock_into_death_if_at_zero(fit: MixtureFit) -> MixtureFit:
    """If the fitted plateau sits on rho=0 within its own noise, it is Death."""
    sigma = float(np.sqrt(max(fit.variances[0], fit.variances[1])))
    if abs(fit.lock_center) > 3.0 * max(sigma, 1e-8):
        return fit
    weights = fit.weights.copy()
    weights[0] += weights[1]
    weights[1] = 0.0
    return MixtureFit(
        weights=weights,
        lock_center=fit.lock_center,
        variances=fit.variances,
        loglik=fit.loglik,
        n_iter=fit.n_iter,
        gamma=fit.gamma,
    )


def discover_raw_cells(
    raw_dir: Path,
) -> dict[tuple[float, float, float], list[tuple[int, Path]]]:
    """Map (K, delta, p) -> sorted (trial, path) pairs.

    Paths are taken from the filenames on disk so both delta_0.000
    (rk4) and delta_0.03 (legacy raw_trials) resolve.
    """
    cells: dict[tuple[float, float, float], list[tuple[int, Path]]] = {}
    if not raw_dir.exists():
        return cells
    for path in raw_dir.glob("*.npz"):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        key = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        cells.setdefault(key, []).append((int(match.group(4)), path))
    for key in cells:
        cells[key] = sorted(cells[key], key=lambda item: item[0])
    return cells


def trial_path(raw_dir: Path, K: float, delta: float, p: float, trial: int) -> Path:
    """Resolve a trial file, trying 3-decimal then 2-decimal delta names."""
    candidates = (
        raw_dir
        / f"K_{K:.2f}_delta_{delta:.3f}_p_{p:.2f}_trial_{trial:02d}.npz",
        raw_dir
        / f"K_{K:.2f}_delta_{delta:.2f}_p_{p:.2f}_trial_{trial:02d}.npz",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_trial_arrays(
    path: Path, clean_params: CleanParams
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    with np.load(path, allow_pickle=True) as data:
        omega = np.asarray(data["omega"], dtype=float)
        rho = np.asarray(data["rotation_number"], dtype=float)
        meta = {
            "K": float(data["K"]),
            "delta": float(data["delta"]),
            "p": float(data["p"]),
            "trial": float(data["trial"]),
        }
    rho = clean_rotation_profile(omega, rho, clean_params)
    return omega, rho, meta


def _axes_from_cells(
    cells: dict[tuple[float, float, float], list[tuple[int, Path]]], p: float
) -> tuple[np.ndarray, np.ndarray]:
    Ks = sorted({K for (K, _, pv) in cells if abs(pv - p) < 1e-12})
    deltas = sorted({d for (_, d, pv) in cells if abs(pv - p) < 1e-12})
    return np.asarray(Ks, dtype=float), np.asarray(deltas, dtype=float)


def _extent(k_axis: np.ndarray, delta_axis: np.ndarray) -> list[float]:
    return [
        float(delta_axis[0]),
        float(delta_axis[-1]),
        float(k_axis[0]),
        float(k_axis[-1]),
    ]


def fit_grid(
    raw_dir: Path,
    p: float,
    clean_params: CleanParams,
    classify_params: ClassifyParams,
) -> dict[str, np.ndarray | float | str]:
    cells = discover_raw_cells(raw_dir)
    k_axis, delta_axis = _axes_from_cells(cells, p)
    if k_axis.size == 0 or delta_axis.size == 0:
        raise FileNotFoundError(f"No npz files for p={p:.2f} in {raw_dir}")

    shape = (delta_axis.size, k_axis.size)
    w_death = np.full(shape, np.nan)
    w_lock = np.full(shape, np.nan)
    w_drift = np.full(shape, np.nan)
    lock_center = np.full(shape, np.nan)
    loglik = np.full(shape, np.nan)
    n_trials = np.zeros(shape, dtype=int)
    old_code = np.full(shape, -1, dtype=int)
    old_name_to_code = {
        "Incoherence": 0,
        "Locking": 1,
        "Partial Locking": 2,
        "Partial Death": 3,
        "Death": 4,
    }

    rows: list[dict[str, float | str | int]] = []
    total = sum(
        1 for (K, delta, pv) in cells if abs(pv - p) < 1e-12
    )
    done = 0
    for i_delta, delta in enumerate(delta_axis):
        for i_K, K in enumerate(k_axis):
            trials = cells.get((float(K), float(delta), float(p)), [])
            if not trials:
                continue
            weights = []
            centers = []
            likes = []
            old_states = []
            for trial, path in trials:
                if not path.exists():
                    continue
                omega, rho, _meta = load_trial_arrays(path, clean_params)
                fit = fit_three_line_mixture(omega, rho)
                old_state = classify_state(omega, rho, classify_params)
                weights.append(fit.weights)
                centers.append(fit.lock_center)
                likes.append(fit.loglik)
                old_states.append(old_state)
                rows.append(
                    {
                        "K": float(K),
                        "delta": float(delta),
                        "p": float(p),
                        "trial": int(trial),
                        "w_death": float(fit.weights[0]),
                        "w_lock": float(fit.weights[1]),
                        "w_drift": float(fit.weights[2]),
                        "lock_center": float(fit.lock_center),
                        "loglik": float(fit.loglik),
                        "n_iter": int(fit.n_iter),
                        "old_state": old_state,
                    }
                )
            if not weights:
                continue
            mean_w = np.mean(np.stack(weights, axis=0), axis=0)
            w_death[i_delta, i_K] = mean_w[0]
            w_lock[i_delta, i_K] = mean_w[1]
            w_drift[i_delta, i_K] = mean_w[2]
            lock_center[i_delta, i_K] = float(np.mean(centers))
            loglik[i_delta, i_K] = float(np.mean(likes))
            n_trials[i_delta, i_K] = len(weights)
            majority = max(set(old_states), key=old_states.count)
            old_code[i_delta, i_K] = old_name_to_code[majority]
            done += 1
            if done % 50 == 0 or done == total:
                print(f"fitted {done}/{total} cells", flush=True)

    return {
        "K_axis": k_axis,
        "delta_axis": delta_axis,
        "p": float(p),
        "w_death": w_death,
        "w_lock": w_lock,
        "w_drift": w_drift,
        "lock_center": lock_center,
        "loglik": loglik,
        "n_trials": n_trials,
        "old_code": old_code,
        "rows": rows,
        "clean": cleaning_label(clean_params),
    }


def _imshow_weight(ax, field, k_axis, delta_axis, title, cmap):
    image = ax.imshow(
        field.T,
        origin="lower",
        aspect="auto",
        extent=_extent(k_axis, delta_axis),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Frequency width δ")
    ax.set_ylabel("Coupling strength K")
    ax.set_title(title)
    ax.set_xticks(delta_axis[:: max(len(delta_axis) // 10, 1)])
    ax.set_yticks(k_axis[:: max(len(k_axis) // 10, 1)])
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
    return image


def _rgb_composite(w_death, w_lock, w_drift) -> np.ndarray:
    death = np.array(matplotlib.colors.to_rgb(COMPONENT_COLORS["death"]))
    lock = np.array(matplotlib.colors.to_rgb(COMPONENT_COLORS["lock"]))
    drift = np.array(matplotlib.colors.to_rgb(COMPONENT_COLORS["drift"]))
    rgb = (
        w_death[..., None] * death
        + w_lock[..., None] * lock
        + w_drift[..., None] * drift
    )
    return np.clip(rgb, 0.0, 1.0)


def plot_weight_maps(payload: dict, figure_dir: Path, dpi: int) -> Path:
    k_axis = payload["K_axis"]
    delta_axis = payload["delta_axis"]
    p = payload["p"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), layout="constrained")
    panels = [
        (axes[0, 0], payload["w_death"], r"$w_{\mathrm{death}}$"),
        (axes[0, 1], payload["w_lock"], r"$w_{\mathrm{lock}}$"),
        (axes[1, 0], payload["w_drift"], r"$w_{\mathrm{drift}}$"),
    ]
    for ax, field, title in panels:
        image = _imshow_weight(ax, field, k_axis, delta_axis, title, "viridis")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="weight")

    rgb = _rgb_composite(
        payload["w_death"], payload["w_lock"], payload["w_drift"]
    )
    ax = axes[1, 1]
    ax.imshow(
        np.transpose(rgb, (1, 0, 2)),
        origin="lower",
        aspect="auto",
        extent=_extent(k_axis, delta_axis),
    )
    ax.set_xlabel("Frequency width δ")
    ax.set_ylabel("Coupling strength K")
    ax.set_title("RGB composite (death / lock / drift)")
    ax.set_xticks(delta_axis[:: max(len(delta_axis) // 10, 1)])
    ax.set_yticks(k_axis[:: max(len(k_axis) // 10, 1)])
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(
        handles=[
            Patch(color=COMPONENT_COLORS["death"], label="death"),
            Patch(color=COMPONENT_COLORS["lock"], label="lock"),
            Patch(color=COMPONENT_COLORS["drift"], label="drift"),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    fig.suptitle(
        "Three-line mixture on "
        r"$\rho(\omega)$"
        f"\nN=1000, p={p:.2f}, cleaning={payload['clean']}"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"mixture_weights_p_{p:.2f}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_old_vs_mixture(payload: dict, figure_dir: Path, dpi: int) -> Path:
    k_axis = payload["K_axis"]
    delta_axis = payload["delta_axis"]
    p = payload["p"]
    old_colors = [
        STATE_COLORS["Incoherence"],
        STATE_COLORS["Locking"],
        STATE_COLORS["Partial Locking"],
        STATE_COLORS["Partial Death"],
        STATE_COLORS["Death"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    ax = axes[0]
    ax.imshow(
        payload["old_code"].T,
        origin="lower",
        aspect="auto",
        extent=_extent(k_axis, delta_axis),
        cmap=ListedColormap(old_colors),
        vmin=-0.5,
        vmax=4.5,
    )
    ax.set_title("Existing rotation-number rules")
    ax.set_xlabel("Frequency width δ")
    ax.set_ylabel("Coupling strength K")
    ax.legend(
        handles=[
            Patch(color=STATE_COLORS["Incoherence"], label="Incoherence"),
            Patch(color=STATE_COLORS["Locking"], label="Locking"),
            Patch(color=STATE_COLORS["Partial Locking"], label="Partial Locking"),
            Patch(color=STATE_COLORS["Partial Death"], label="Partial Death"),
            Patch(color=STATE_COLORS["Death"], label="Death"),
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.9,
    )

    rgb = _rgb_composite(
        payload["w_death"], payload["w_lock"], payload["w_drift"]
    )
    ax = axes[1]
    ax.imshow(
        np.transpose(rgb, (1, 0, 2)),
        origin="lower",
        aspect="auto",
        extent=_extent(k_axis, delta_axis),
    )
    ax.set_title("Mixture weights (no state threshold)")
    ax.set_xlabel("Frequency width δ")
    ax.set_ylabel("Coupling strength K")
    ax.legend(
        handles=[
            Patch(color=COMPONENT_COLORS["death"], label="death"),
            Patch(color=COMPONENT_COLORS["lock"], label="lock"),
            Patch(color=COMPONENT_COLORS["drift"], label="drift"),
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.9,
    )
    for ax in axes:
        ax.set_xticks(delta_axis[:: max(len(delta_axis) // 10, 1)])
        ax.set_yticks(k_axis[:: max(len(k_axis) // 10, 1)])
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(f"Rule-based labels vs mixture weights, p={p:.2f}")
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"mixture_vs_rules_p_{p:.2f}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _entropy(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.nan_to_num(weights, nan=0.0), 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=0)
    entropy[np.isnan(weights[0])] = np.nan
    return entropy


def plot_example_profiles(
    raw_dir: Path,
    payload: dict,
    clean_params: CleanParams,
    figure_dir: Path,
    dpi: int,
) -> Path | None:
    k_axis = payload["K_axis"]
    delta_axis = payload["delta_axis"]
    p = payload["p"]
    w = np.stack(
        (payload["w_death"], payload["w_lock"], payload["w_drift"]), axis=0
    )
    if np.all(np.isnan(w)):
        return None

    picks: list[tuple[int, int, str]] = []

    def add_pick(i_delta: int, i_K: int, label: str) -> None:
        key = (i_delta, i_K)
        if key in {(a, b) for a, b, _ in picks}:
            return
        if np.isnan(w[0, i_delta, i_K]):
            return
        picks.append((i_delta, i_K, label))

    add_pick(*np.unravel_index(np.nanargmax(payload["w_death"]), w[0].shape), "high death")
    add_pick(*np.unravel_index(np.nanargmax(payload["w_lock"]), w[0].shape), "high lock")
    add_pick(*np.unravel_index(np.nanargmax(payload["w_drift"]), w[0].shape), "high drift")
    mixed = _entropy(w)
    mixed[np.isnan(w[0])] = -np.inf
    # delta=0 makes the lock line rho=c and the drift line rho=omega coincide.
    if float(delta_axis[0]) == 0.0:
        mixed[0, :] = -np.inf
    for _ in range(6):
        idx = int(np.argmax(mixed))
        i_delta, i_K = np.unravel_index(idx, mixed.shape)
        if not np.isfinite(mixed[i_delta, i_K]):
            break
        add_pick(i_delta, i_K, "mixed")
        mixed[i_delta, i_K] = -np.inf
        if len(picks) >= 6:
            break

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, (i_delta, i_K, label) in zip(axes, picks):
        K = float(k_axis[i_K])
        delta = float(delta_axis[i_delta])
        path = trial_path(raw_dir, K, delta, float(p), 0)
        if not path.exists():
            ax.set_visible(False)
            continue
        omega, rho, _ = load_trial_arrays(path, clean_params)
        fit = fit_three_line_mixture(omega, rho)
        hard = np.argmax(fit.gamma, axis=1)
        order = np.argsort(omega)
        omega_s = omega[order]
        rho_s = rho[order]
        hard_s = hard[order]
        for comp, name in enumerate(COMPONENT_NAMES):
            mask = hard_s == comp
            if not np.any(mask):
                continue
            ax.scatter(
                omega_s[mask],
                rho_s[mask],
                s=8,
                alpha=0.7,
                color=COMPONENT_COLORS[name],
                label=name,
            )
        ax.plot(omega_s, np.zeros_like(omega_s), color="0.6", linewidth=0.8)
        ax.plot(omega_s, omega_s, color="0.6", linewidth=0.8, linestyle="--")
        if fit.weights[1] > 0.05:
            ax.axhline(fit.lock_center, color=COMPONENT_COLORS["lock"], linewidth=0.9)
        ax.set_title(
            f"{label}\nK={K:.2f}, δ={delta:.3f}\n"
            f"d={fit.weights[0]:.2f}  l={fit.weights[1]:.2f}  "
            f"r={fit.weights[2]:.2f}"
        )
        ax.set_xlabel(r"$\omega_i$")
        ax.set_ylabel(r"$\rho_i$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Example (ω, ρ) clouds colored by mixture assignment")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    out = figure_dir / f"mixture_examples_p_{p:.2f}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def save_tables(payload: dict, table_dir: Path, p: float) -> tuple[Path, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    npz_path = table_dir / f"mixture_weights_p_{p:.2f}.npz"
    np.savez_compressed(
        npz_path,
        K_axis=payload["K_axis"],
        delta_axis=payload["delta_axis"],
        p=payload["p"],
        w_death=payload["w_death"],
        w_lock=payload["w_lock"],
        w_drift=payload["w_drift"],
        lock_center=payload["lock_center"],
        loglik=payload["loglik"],
        n_trials=payload["n_trials"],
        old_code=payload["old_code"],
        clean=np.array(payload["clean"]),
    )
    csv_path = table_dir / f"mixture_weights_p_{p:.2f}.csv"
    fieldnames = [
        "K",
        "delta",
        "p",
        "trial",
        "w_death",
        "w_lock",
        "w_drift",
        "lock_center",
        "loglik",
        "n_iter",
        "old_state",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])
    return npz_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit death / lock / drift mixture weights on raw (omega, rho) trials."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory of trial npz files (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=list(DEFAULT_P_VALUES),
        help=(
            "ER connection probabilities to scan "
            f"(default: {' '.join(f'{p:.1f}' for p in DEFAULT_P_VALUES)})"
        ),
    )
    parser.add_argument(
        "--clean",
        choices=("none", "monotone", "monotone_smooth"),
        default="none",
        help="Optional rho(omega) cleaning before the fit (default: none)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Odd window for monotone_smooth (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MIXTURE_FIGURE_DIR,
        help=f"Figure directory (default: {DEFAULT_MIXTURE_FIGURE_DIR})",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help=f"CSV/NPZ directory (default: {DEFAULT_TABLE_DIR})",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def run_one_p(
    raw_dir: Path,
    p: float,
    clean_params: CleanParams,
    output_dir: Path,
    table_dir: Path,
    dpi: int,
) -> None:
    print(
        f"Fitting three-line mixture in {raw_dir} "
        f"(p={p:.2f}, clean={cleaning_label(clean_params)})"
    )
    try:
        payload = fit_grid(raw_dir, p, clean_params, ClassifyParams())
    except FileNotFoundError as exc:
        print(f"skip p={p:.2f}: {exc}")
        return
    npz_path, csv_path = save_tables(payload, table_dir, p)
    weight_fig = plot_weight_maps(payload, output_dir, dpi)
    compare_fig = plot_old_vs_mixture(payload, output_dir, dpi)
    example_fig = plot_example_profiles(
        raw_dir, payload, clean_params, output_dir, dpi
    )
    w = np.stack(
        (payload["w_death"], payload["w_lock"], payload["w_drift"]), axis=0
    )
    finite = np.isfinite(w[0])
    print(
        "mean weights over grid: "
        f"death={np.nanmean(w[0]):.3f}  "
        f"lock={np.nanmean(w[1]):.3f}  "
        f"drift={np.nanmean(w[2]):.3f}"
    )
    print(
        "cells with a dominant component (>0.8): "
        f"death={int(np.sum((w[0] > 0.8) & finite))}  "
        f"lock={int(np.sum((w[1] > 0.8) & finite))}  "
        f"drift={int(np.sum((w[2] > 0.8) & finite))}  "
        f"mixed={int(np.sum((np.max(w, axis=0) <= 0.8) & finite))}"
    )
    print(f"Saved {npz_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {weight_fig}")
    print(f"Saved {compare_fig}")
    if example_fig is not None:
        print(f"Saved {example_fig}")


def main() -> None:
    args = parse_args()
    raw_dir = resolve_user_path(args.raw_dir, CLASSIFICATION_ROOT, CURSOR_ROOT)
    output_dir = resolve_user_path(
        args.output_dir, CLASSIFICATION_ROOT, CURSOR_ROOT
    )
    table_dir = resolve_user_path(
        args.table_dir, CLASSIFICATION_ROOT, CURSOR_ROOT
    )
    clean_params = CleanParams(method=args.clean, smooth_window=args.smooth_window)
    p_values = [float(p) for p in args.p]
    print(
        f"Scanning p = {', '.join(f'{p:.1f}' for p in p_values)} "
        f"in {raw_dir}"
    )
    for p in p_values:
        run_one_p(
            raw_dir,
            p,
            clean_params,
            output_dir,
            table_dir,
            args.dpi,
        )


if __name__ == "__main__":
    main()
