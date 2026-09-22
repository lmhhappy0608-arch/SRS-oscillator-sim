#!/usr/bin/env python3
"""
Step 1 of the mixture stats plan: residual diagnostics per (K, delta) cell.

For each trial, refit the three-line mixture and record
    RMSE, max |residual|, and residual-vs-omega shape summaries
after sorting oscillators by natural frequency.

Example (run from classification_method/):
  python3 code/analyze_mixture_residuals.py
  python3 code/analyze_mixture_residuals.py --p 0.2 0.4 0.6 0.8 1.0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import project_paths  # noqa: F401  # adds Cursor/code to sys.path
from fit_rotation_mixture import (
    COMPONENT_COLORS,
    COMPONENT_NAMES,
    _axes_from_cells,
    _extent,
    discover_raw_cells,
    fit_three_line_mixture,
    load_trial_arrays,
)
from project_paths import (
    CLASSIFICATION_ROOT,
    CURSOR_ROOT,
    DEFAULT_P_VALUES,
    DEFAULT_RAW_DIR,
    DEFAULT_RESIDUAL_FIGURE_DIR,
    DEFAULT_TABLE_DIR,
    resolve_user_path,
)
from rotation_cleaning import CleanParams, cleaning_label

N_PROFILE_BINS = 25


def mixture_prediction(
    omega: np.ndarray, fit
) -> tuple[np.ndarray, np.ndarray]:
    """Hard assignment: rho_hat is 0, lock_center, or omega."""
    hard = np.argmax(fit.gamma, axis=1)
    pred = np.zeros(omega.shape, dtype=float)
    pred[hard == 1] = fit.lock_center
    pred[hard == 2] = omega[hard == 2]
    return pred, hard


def lag1_autocorr(values: np.ndarray) -> float:
    if values.size < 3:
        return float("nan")
    x = values[:-1] - np.mean(values[:-1])
    y = values[1:] - np.mean(values[1:])
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(x, y) / denom)


def quadratic_shape(omega: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    """Tilt and curvature of residual versus centered omega."""
    if omega.size < 4 or float(np.std(omega)) < 1e-12:
        return float("nan"), float("nan")
    x = omega - np.mean(omega)
    X = np.column_stack((np.ones_like(x), x, x * x))
    try:
        coef, *_ = np.linalg.lstsq(X, residual, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    return float(coef[1]), float(coef[2])


def tercile_rmse(residual: np.ndarray) -> tuple[float, float, float]:
    n = residual.size
    if n < 9:
        return float("nan"), float("nan"), float("nan")
    cuts = (n // 3, 2 * n // 3)
    chunks = (residual[: cuts[0]], residual[cuts[0] : cuts[1]], residual[cuts[1] :])
    return tuple(float(np.sqrt(np.mean(chunk * chunk))) for chunk in chunks)


def binned_residual_profile(
    omega: np.ndarray, residual: np.ndarray, n_bins: int = N_PROFILE_BINS
) -> np.ndarray:
    profile = np.full(n_bins, np.nan)
    if omega.size == 0:
        return profile
    order = np.argsort(omega)
    ranks = np.linspace(0.0, 1.0, omega.size, endpoint=False)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    res = residual[order]
    for i in range(n_bins):
        mask = (ranks >= edges[i]) & (ranks < edges[i + 1])
        if not np.any(mask):
            continue
        profile[i] = float(np.mean(res[mask]))
    return profile


def residual_metrics(omega: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    sq = residual * residual
    rmse_low, rmse_mid, rmse_high = tercile_rmse(residual)
    tilt, curvature = quadratic_shape(omega, residual)
    return {
        "rmse": float(np.sqrt(np.mean(sq))),
        "max_abs": float(np.max(np.abs(residual))),
        "mean_residual": float(np.mean(residual)),
        "lag1_acf": lag1_autocorr(residual),
        "rmse_low_omega": rmse_low,
        "rmse_mid_omega": rmse_mid,
        "rmse_high_omega": rmse_high,
        "tilt": tilt,
        "curvature": curvature,
    }


def diagnose_cell(omega: np.ndarray, rho: np.ndarray) -> dict:
    fit = fit_three_line_mixture(omega, rho)
    pred, hard = mixture_prediction(omega, fit)
    residual = rho - pred
    order = np.argsort(omega)
    metrics = residual_metrics(omega[order], residual[order])
    metrics.update(
        {
            "w_death": float(fit.weights[0]),
            "w_lock": float(fit.weights[1]),
            "w_drift": float(fit.weights[2]),
            "lock_center": float(fit.lock_center),
            "loglik": float(fit.loglik),
        }
    )
    return {
        "metrics": metrics,
        "fit": fit,
        "pred": pred,
        "hard": hard,
        "residual": residual,
        "profile": binned_residual_profile(omega, residual),
    }


def analyze_grid(
    raw_dir: Path, p: float, clean_params: CleanParams
) -> dict:
    cells = discover_raw_cells(raw_dir)
    k_axis, delta_axis = _axes_from_cells(cells, p)
    if k_axis.size == 0 or delta_axis.size == 0:
        raise FileNotFoundError(f"No npz files for p={p:.2f} in {raw_dir}")

    shape = (delta_axis.size, k_axis.size)
    fields = {
        name: np.full(shape, np.nan)
        for name in (
            "rmse",
            "max_abs",
            "mean_residual",
            "lag1_acf",
            "rmse_low_omega",
            "rmse_mid_omega",
            "rmse_high_omega",
            "tilt",
            "curvature",
            "w_death",
            "w_lock",
            "w_drift",
        )
    }
    profiles = np.full(shape + (N_PROFILE_BINS,), np.nan)
    rows: list[dict[str, float | int]] = []
    examples: list[dict] = []

    total = sum(1 for (_, _, pv) in cells if abs(pv - p) < 1e-12)
    done = 0
    for i_delta, delta in enumerate(delta_axis):
        for i_K, K in enumerate(k_axis):
            trials = cells.get((float(K), float(delta), float(p)), [])
            if not trials:
                continue
            cell_metrics = []
            cell_profiles = []
            last_diag = None
            last_omega = None
            last_rho = None
            for trial, path in trials:
                if not path.exists():
                    continue
                omega, rho, _meta = load_trial_arrays(path, clean_params)
                diag = diagnose_cell(omega, rho)
                cell_metrics.append(diag["metrics"])
                cell_profiles.append(diag["profile"])
                last_diag = diag
                last_omega = omega
                last_rho = rho
                row = {
                    "K": float(K),
                    "delta": float(delta),
                    "p": float(p),
                    "trial": int(trial),
                }
                row.update(diag["metrics"])
                rows.append(row)
            if not cell_metrics:
                continue
            mean_metrics = {
                key: float(np.nanmean([m[key] for m in cell_metrics]))
                for key in cell_metrics[0]
            }
            for key, grid in fields.items():
                grid[i_delta, i_K] = mean_metrics[key]
            profiles[i_delta, i_K] = np.nanmean(
                np.stack(cell_profiles, axis=0), axis=0
            )
            if last_diag is not None:
                examples.append(
                    {
                        "K": float(K),
                        "delta": float(delta),
                        "rmse": mean_metrics["rmse"],
                        "max_abs": mean_metrics["max_abs"],
                        "omega": last_omega,
                        "rho": last_rho,
                        "residual": last_diag["residual"],
                        "hard": last_diag["hard"],
                        "lock_center": last_diag["fit"].lock_center,
                    }
                )
            done += 1
            if done % 50 == 0 or done == total:
                print(f"p={p:.2f}: residuals {done}/{total} cells", flush=True)

    return {
        "K_axis": k_axis,
        "delta_axis": delta_axis,
        "p": float(p),
        "clean": cleaning_label(clean_params),
        "profiles": profiles,
        "rows": rows,
        "examples": examples,
        **fields,
    }


def _imshow(ax, field, k_axis, delta_axis, title, cmap, vlim=None):
    kwargs = {
        "origin": "lower",
        "aspect": "auto",
        "extent": _extent(k_axis, delta_axis),
        "cmap": cmap,
    }
    if vlim is not None:
        kwargs["vmin"], kwargs["vmax"] = vlim
    image = ax.imshow(field.T, **kwargs)
    ax.set_xlabel("Frequency width δ")
    ax.set_ylabel("Coupling strength K")
    ax.set_title(title)
    ax.set_xticks(delta_axis[:: max(len(delta_axis) // 10, 1)])
    ax.set_yticks(k_axis[:: max(len(k_axis) // 10, 1)])
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
    return image


def plot_residual_maps(payload: dict, figure_dir: Path, dpi: int) -> Path:
    k_axis = payload["K_axis"]
    delta_axis = payload["delta_axis"]
    p = payload["p"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), layout="constrained")
    panels = [
        (axes[0, 0], payload["rmse"], "RMSE", "magma", None),
        (axes[0, 1], payload["max_abs"], r"max $|\mathrm{residual}|$", "magma", None),
        (axes[1, 0], payload["lag1_acf"], "residual lag-1 ACF vs ω", "coolwarm", (-1.0, 1.0)),
        (
            axes[1, 1],
            payload["rmse_mid_omega"]
            - 0.5 * (payload["rmse_low_omega"] + payload["rmse_high_omega"]),
            "RMSE(mid ω) − RMSE(ends)",
            "coolwarm",
            None,
        ),
    ]
    for ax, field, title, cmap, vlim in panels:
        image = _imshow(ax, field, k_axis, delta_axis, title, cmap, vlim)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        r"Three-line residual diagnostics on $\rho(\omega)$"
        f"\np={p:.2f}, cleaning={payload['clean']}"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"mixture_residual_maps_p_{p:.2f}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_high_rmse_profiles(
    payload: dict, figure_dir: Path, dpi: int
) -> Path | None:
    examples = [
        item
        for item in payload["examples"]
        if np.isfinite(item["rmse"]) and item["delta"] > 0.0
    ]
    if not examples:
        return None
    examples.sort(key=lambda item: item["rmse"], reverse=True)
    picks = examples[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8), layout="constrained")
    for ax, item in zip(axes.ravel(), picks):
        order = np.argsort(item["omega"])
        omega = item["omega"][order]
        residual = item["residual"][order]
        hard = item["hard"][order]
        for comp, name in enumerate(COMPONENT_NAMES):
            mask = hard == comp
            if not np.any(mask):
                continue
            ax.scatter(
                omega[mask],
                residual[mask],
                s=8,
                alpha=0.7,
                color=COMPONENT_COLORS[name],
                label=name,
            )
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_title(
            f"K={item['K']:.2f}, δ={item['delta']:.3f}\n"
            f"RMSE={item['rmse']:.3g}, max={item['max_abs']:.3g}"
        )
        ax.set_xlabel(r"$\omega_i$")
        ax.set_ylabel(r"$\rho_i-\hat\rho_i$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"Highest-RMSE residual vs ω, p={payload['p']:.2f}"
        "\n(points colored by mixture assignment)"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"mixture_residual_examples_p_{payload['p']:.2f}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_tables(payload: dict, table_dir: Path, p: float) -> tuple[Path, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    npz_path = table_dir / f"mixture_residuals_p_{p:.2f}.npz"
    np.savez_compressed(
        npz_path,
        K_axis=payload["K_axis"],
        delta_axis=payload["delta_axis"],
        p=payload["p"],
        rmse=payload["rmse"],
        max_abs=payload["max_abs"],
        mean_residual=payload["mean_residual"],
        lag1_acf=payload["lag1_acf"],
        rmse_low_omega=payload["rmse_low_omega"],
        rmse_mid_omega=payload["rmse_mid_omega"],
        rmse_high_omega=payload["rmse_high_omega"],
        tilt=payload["tilt"],
        curvature=payload["curvature"],
        w_death=payload["w_death"],
        w_lock=payload["w_lock"],
        w_drift=payload["w_drift"],
        profiles=payload["profiles"],
        clean=np.array(payload["clean"]),
    )
    csv_path = table_dir / f"mixture_residuals_p_{p:.2f}.csv"
    fieldnames = [
        "K",
        "delta",
        "p",
        "trial",
        "rmse",
        "max_abs",
        "mean_residual",
        "lag1_acf",
        "rmse_low_omega",
        "rmse_mid_omega",
        "rmse_high_omega",
        "tilt",
        "curvature",
        "w_death",
        "w_lock",
        "w_drift",
        "lock_center",
        "loglik",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])
    return npz_path, csv_path


def run_one_p(
    raw_dir: Path,
    p: float,
    clean_params: CleanParams,
    output_dir: Path,
    table_dir: Path,
    dpi: int,
) -> None:
    print(
        f"Residual diagnostics in {raw_dir} "
        f"(p={p:.2f}, clean={cleaning_label(clean_params)})"
    )
    try:
        payload = analyze_grid(raw_dir, p, clean_params)
    except FileNotFoundError as exc:
        print(f"skip p={p:.2f}: {exc}")
        return
    npz_path, csv_path = save_tables(payload, table_dir, p)
    map_fig = plot_residual_maps(payload, output_dir, dpi)
    example_fig = plot_high_rmse_profiles(payload, output_dir, dpi)
    finite = np.isfinite(payload["rmse"])
    print(
        "RMSE over grid: "
        f"mean={np.nanmean(payload['rmse']):.4g}  "
        f"max={np.nanmax(payload['rmse']):.4g}  "
        f"cells={int(np.sum(finite))}"
    )
    print(f"Saved {npz_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {map_fig}")
    if example_fig is not None:
        print(f"Saved {example_fig}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute RMSE, max residual, and residual-vs-omega shape "
            "for the three-line mixture on each (K, delta) cell."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Trial npz directory (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=list(DEFAULT_P_VALUES),
        help=(
            "ER connection probabilities "
            f"(default: {' '.join(f'{p:.1f}' for p in DEFAULT_P_VALUES)})"
        ),
    )
    parser.add_argument(
        "--clean",
        choices=("none", "monotone", "monotone_smooth"),
        default="none",
    )
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESIDUAL_FIGURE_DIR,
        help=f"Figure directory (default: {DEFAULT_RESIDUAL_FIGURE_DIR})",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help=f"CSV/NPZ directory (default: {DEFAULT_TABLE_DIR})",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


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
        run_one_p(raw_dir, p, clean_params, output_dir, table_dir, args.dpi)


if __name__ == "__main__":
    main()
