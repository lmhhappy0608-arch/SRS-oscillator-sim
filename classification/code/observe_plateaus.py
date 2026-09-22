#!/usr/bin/env python3
"""
Step 1 of plateau observation: look at raw rho(omega) and the rho histogram.

This does not classify states and does not require consecutive oscillators
in omega. For each trial it reports histogram peaks in rho: center c, width,
and the fraction of oscillators inside the half-maximum band. Consecutive
runs in omega are recorded afterwards as a check, not as a gate.

Example (from classification_method/):
  python3 code/observe_plateaus.py
  python3 code/observe_plateaus.py --k 0.4 --delta 0.08 --p 0.8 --trial 0
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_rotation_mixture import trial_path
from project_paths import (
    CLASSIFICATION_ROOT,
    DEFAULT_PLATEAU_FIGURE_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_TABLE_DIR,
    resolve_user_path,
)

# Menu of interior-looking trials at p=0.8, trial 0. Labels are only for
# choosing what to inspect; stats are computed from the raw histogram.
DEFAULT_GALLERY: tuple[tuple[str, float, float, float, int], ...] = (
    ("death", 1.60, 0.05, 0.8, 0),
    ("death", 1.10, 0.30, 0.8, 0),
    ("lock", 0.40, 0.08, 0.8, 0),
    ("lock", 0.70, 0.08, 0.8, 0),
    ("partial_death", 0.80, 0.50, 0.8, 0),
    ("partial_death", 0.75, 0.65, 0.8, 0),
    ("partial_lock", 0.70, 0.20, 0.8, 0),
    ("partial_lock", 0.40, 0.12, 0.8, 0),
    ("incoherence", 0.20, 0.23, 0.8, 0),
    ("incoherence", 0.00, 0.25, 0.8, 0),
)


@dataclass(frozen=True)
class RhoPeak:
    c: float
    sigma: float
    mad: float
    fwhm: float
    fraction: float
    n_in: int
    longest_omega_run: int
    zero_like: bool
    sharp: bool
    rho_lo: float
    rho_hi: float


@dataclass(frozen=True)
class TrialObservation:
    menu_label: str
    K: float
    delta: float
    p: float
    trial: int
    path: Path
    omega: np.ndarray
    rho: np.ndarray
    peaks: tuple[RhoPeak, ...]
    rho_median: float
    rho_mad: float
    ident_mad: float


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.median(np.abs(values - np.median(values))))


def _longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return int(best)


def _histogram_bins(rho: np.ndarray) -> np.ndarray:
    """Freedman–Diaconis bins, capped so a delta-like sample stays one peak."""
    n = rho.size
    q1, q3 = np.percentile(rho, [25.0, 75.0])
    iqr = float(q3 - q1)
    span = float(rho.max() - rho.min())
    if span <= 1e-15 or iqr <= 1e-15:
        center = float(np.median(rho))
        pad = max(1e-8, 1e-3 * max(abs(center), 1.0))
        return np.linspace(center - pad, center + pad, 8)
    width = 2.0 * iqr / n ** (1.0 / 3.0)
    n_bins = int(np.ceil(span / max(width, 1e-12)))
    n_bins = max(24, min(80, n_bins))
    pad = 0.02 * span
    return np.linspace(rho.min() - pad, rho.max() + pad, n_bins + 1)


def find_rho_peaks(
    omega: np.ndarray,
    rho: np.ndarray,
    max_peaks: int = 3,
    min_fraction: float = 0.05,
) -> list[RhoPeak]:
    """Tight clusters in rho, split where consecutive sorted gaps exceed ~0.5 Δω.

    Membership is a band in rho. Consecutive runs in omega are stored afterwards
    and are not used to accept or reject a cluster. min_fraction is only a
    display cutoff so histogram speckle is not listed as a plateau.
    """
    omega = np.asarray(omega, dtype=float)
    rho = np.asarray(rho, dtype=float)
    n = rho.size
    order = np.argsort(omega)
    omega_span = float(omega.max() - omega.min()) if n else 0.0
    drift_gap = omega_span / max(n - 1, 1) if omega_span > 0 else 1.0
    eps = 0.5 * max(drift_gap, 1e-12)
    min_size = max(10, int(np.ceil(min_fraction * n)))

    idx = np.argsort(rho)
    rho_sorted = rho[idx]
    if n == 1 or float(rho_sorted[-1] - rho_sorted[0]) <= 1e-14:
        c = float(rho_sorted[0]) if n else 0.0
        return [
            RhoPeak(
                c=c,
                sigma=0.0,
                mad=0.0,
                fwhm=0.0,
                fraction=1.0,
                n_in=n,
                longest_omega_run=n,
                zero_like=abs(c) <= 1e-8,
                sharp=True,
                rho_lo=c,
                rho_hi=c,
            )
        ]

    gaps = np.diff(rho_sorted)
    breaks = np.where(gaps > eps)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))

    peaks: list[RhoPeak] = []
    for start, end in zip(starts, ends):
        size = int(end - start)
        if size < min_size:
            continue
        values = rho_sorted[start:end]
        member_mask = np.zeros(n, dtype=bool)
        member_mask[idx[start:end]] = True
        c = float(np.median(values))
        sigma = float(np.std(values))
        mad = _mad(values)
        rho_lo = float(values.min())
        rho_hi = float(values.max())
        width = rho_hi - rho_lo
        scale = max(sigma, mad, 1e-12)
        fraction = size / n
        peaks.append(
            RhoPeak(
                c=c,
                sigma=sigma,
                mad=mad,
                fwhm=width,
                fraction=fraction,
                n_in=size,
                longest_omega_run=_longest_true_run(member_mask[order]),
                zero_like=abs(c) <= max(3.0 * scale, 1e-3),
                sharp=width <= 0.25 * max(omega_span, 1e-12),
                rho_lo=rho_lo,
                rho_hi=rho_hi,
            )
        )

    peaks.sort(key=lambda peak: peak.fraction, reverse=True)
    return peaks[:max_peaks]


def load_raw_trial(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    with np.load(path, allow_pickle=True) as data:
        omega = np.asarray(data["omega"], dtype=float)
        rho = np.asarray(data["rotation_number"], dtype=float)
        meta = {
            "K": float(data["K"]),
            "delta": float(data["delta"]),
            "p": float(data["p"]),
            "trial": float(data["trial"]),
        }
    return omega, rho, meta


def observe_trial(
    path: Path,
    menu_label: str,
    K: float,
    delta: float,
    p: float,
    trial: int,
) -> TrialObservation:
    omega, rho, _meta = load_raw_trial(path)
    peaks = tuple(find_rho_peaks(omega, rho))
    return TrialObservation(
        menu_label=menu_label,
        K=K,
        delta=delta,
        p=p,
        trial=trial,
        path=path,
        omega=omega,
        rho=rho,
        peaks=peaks,
        rho_median=float(np.median(rho)),
        rho_mad=_mad(rho),
        ident_mad=_mad(rho - omega),
    )


def _peak_kind(peak: RhoPeak) -> str:
    if not peak.sharp:
        return "broad"
    if peak.zero_like:
        return "zero"
    return "offset"


def _draw_trial(ax_scatter, ax_hist, obs: TrialObservation) -> None:
    omega = obs.omega
    rho = obs.rho
    order = np.argsort(omega)
    ax_scatter.plot(omega[order], omega[order], color="0.75", lw=1.0, label=r"$\rho=\omega$")
    ax_scatter.axhline(0.0, color="0.85", lw=0.8)
    ax_scatter.plot(omega[order], rho[order], "o", ms=2.2, color="#2166ac", alpha=0.75, rasterized=True)
    y_lo = min(0.0, float(rho.min()), float(omega.min())) - 0.05
    y_hi = max(float(rho.max()), float(omega.max())) + 0.05
    ax_scatter.set_ylim(y_lo, y_hi)
    colors = ("#d73027", "#fdae61", "#1a9850")
    for i, peak in enumerate(obs.peaks):
        color = colors[i % len(colors)]
        ax_scatter.axhline(peak.c, color=color, lw=1.1, ls="--")
        ax_scatter.axhspan(peak.rho_lo, peak.rho_hi, color=color, alpha=0.12)
    ax_scatter.set_xlabel(r"natural frequency $\omega$")
    ax_scatter.set_ylabel(r"rotation number $\rho$")
    ax_scatter.set_title(
        f"{obs.menu_label}  K={obs.K:.2f}  δ={obs.delta:.2f}  p={obs.p:.2f}  trial {obs.trial:02d}"
    )
    ax_scatter.grid(True, alpha=0.25)

    edges = _histogram_bins(rho)
    ax_hist.hist(rho, bins=edges, color="#2166ac", alpha=0.75, orientation="horizontal")
    for i, peak in enumerate(obs.peaks):
        color = colors[i % len(colors)]
        ax_hist.axhline(peak.c, color=color, lw=1.1, ls="--")
        ax_hist.axhspan(peak.rho_lo, peak.rho_hi, color=color, alpha=0.12)
        ax_hist.text(
            0.98,
            0.06 + 0.12 * i,
            f"{_peak_kind(peak)}  c={peak.c:.4f}  σ={peak.sigma:.2e}  f={peak.fraction:.2f}",
            transform=ax_hist.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=color,
        )
    ax_hist.set_xlabel("count")
    ax_hist.set_ylabel(r"$\rho$")
    ax_hist.set_title("histogram of ρ (raw)")
    ax_hist.grid(True, alpha=0.25)


def save_trial_figure(obs: TrialObservation, figure_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    _draw_trial(axes[0], axes[1], obs)
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / (
        f"rho_hist_K_{obs.K:.2f}_delta_{obs.delta:.2f}_p_{obs.p:.2f}_trial_{obs.trial:02d}.png"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_gallery_figure(observations: list[TrialObservation], figure_dir: Path) -> Path:
    n = len(observations)
    fig, axes = plt.subplots(n, 2, figsize=(9.4, 3.15 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    for row, obs in enumerate(observations):
        _draw_trial(axes[row, 0], axes[row, 1], obs)
    figure_dir.mkdir(parents=True, exist_ok=True)
    p_tag = "_".join(f"{obs.p:.2f}" for obs in observations[:1])
    path = figure_dir / f"plateau_gallery_p_{p_tag}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_table(observations: list[TrialObservation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "menu_label",
        "K",
        "delta",
        "p",
        "trial",
        "peak_rank",
        "kind",
        "c",
        "sigma",
        "mad",
        "fwhm",
        "fraction",
        "n_in",
        "longest_omega_run",
        "zero_like",
        "sharp",
        "rho_median",
        "rho_mad_all",
        "ident_mad",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for obs in observations:
            if not obs.peaks:
                writer.writerow(
                    {
                        "menu_label": obs.menu_label,
                        "K": f"{obs.K:.4f}",
                        "delta": f"{obs.delta:.4f}",
                        "p": f"{obs.p:.4f}",
                        "trial": obs.trial,
                        "peak_rank": "",
                        "kind": "none",
                        "c": "",
                        "sigma": "",
                        "mad": "",
                        "fwhm": "",
                        "fraction": "",
                        "n_in": "",
                        "longest_omega_run": "",
                        "zero_like": "",
                        "sharp": "",
                        "rho_median": f"{obs.rho_median:.8g}",
                        "rho_mad_all": f"{obs.rho_mad:.8g}",
                        "ident_mad": f"{obs.ident_mad:.8g}",
                        "path": str(obs.path),
                    }
                )
                continue
            for rank, peak in enumerate(obs.peaks, start=1):
                writer.writerow(
                    {
                        "menu_label": obs.menu_label,
                        "K": f"{obs.K:.4f}",
                        "delta": f"{obs.delta:.4f}",
                        "p": f"{obs.p:.4f}",
                        "trial": obs.trial,
                        "peak_rank": rank,
                        "kind": _peak_kind(peak),
                        "c": f"{peak.c:.8g}",
                        "sigma": f"{peak.sigma:.8g}",
                        "mad": f"{peak.mad:.8g}",
                        "fwhm": f"{peak.fwhm:.8g}",
                        "fraction": f"{peak.fraction:.6f}",
                        "n_in": peak.n_in,
                        "longest_omega_run": peak.longest_omega_run,
                        "zero_like": int(peak.zero_like),
                        "sharp": int(peak.sharp),
                        "rho_median": f"{obs.rho_median:.8g}",
                        "rho_mad_all": f"{obs.rho_mad:.8g}",
                        "ident_mad": f"{obs.ident_mad:.8g}",
                        "path": str(obs.path),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observe raw rho histograms and estimate plateau centers."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_PLATEAU_FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--k", type=float, nargs="+", default=None)
    parser.add_argument("--delta", type=float, nargs="+", default=None)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument(
        "--gallery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the default representative cells (ignored if --k/--delta given).",
    )
    return parser.parse_args()


def _jobs_from_args(args: argparse.Namespace) -> list[tuple[str, float, float, float, int]]:
    if args.k is not None or args.delta is not None:
        if args.k is None or args.delta is None:
            raise SystemExit("Pass both --k and --delta, or neither for the default gallery.")
        jobs: list[tuple[str, float, float, float, int]] = []
        for K in args.k:
            for delta in args.delta:
                jobs.append(("custom", float(K), float(delta), float(args.p), int(args.trial)))
        return jobs
    if args.gallery:
        return list(DEFAULT_GALLERY)
    raise SystemExit("Nothing to observe: pass --k/--delta or keep the default gallery.")


def main() -> None:
    args = parse_args()
    raw_dir = resolve_user_path(args.raw_dir)
    figure_dir = resolve_user_path(args.figure_dir)
    table_dir = resolve_user_path(args.table_dir)
    jobs = _jobs_from_args(args)

    observations: list[TrialObservation] = []
    missing: list[str] = []
    for menu_label, K, delta, p, trial in jobs:
        path = trial_path(raw_dir, K, delta, p, trial)
        if not path.exists():
            missing.append(str(path))
            continue
        observations.append(observe_trial(path, menu_label, K, delta, p, trial))

    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  {path}")
    if not observations:
        raise SystemExit("No trials loaded.")

    table_path = table_dir / "plateau_observe_gallery.csv"
    write_table(observations, table_path)
    gallery_path = save_gallery_figure(observations, figure_dir)
    trial_paths = [save_trial_figure(obs, figure_dir) for obs in observations]

    print(f"Wrote {table_path.relative_to(CLASSIFICATION_ROOT)}")
    print(f"Wrote {gallery_path.relative_to(CLASSIFICATION_ROOT)}")
    print()
    print(
        f"{'label':<15} {'K':>5} {'δ':>5} {'kind':<7} {'c':>10} {'σ':>10} "
        f"{'f':>6} {'n':>5} {'run':>5}"
    )
    for obs in observations:
        if not obs.peaks:
            print(
                f"{obs.menu_label:<15} {obs.K:5.2f} {obs.delta:5.2f} {'none':<7} "
                f"{'':>10} {'':>10} {'':>6} {'':>5} {'':>5}"
            )
            continue
        for rank, peak in enumerate(obs.peaks):
            label = obs.menu_label if rank == 0 else ""
            print(
                f"{label:<15} {obs.K:5.2f} {obs.delta:5.2f} {_peak_kind(peak):<7} "
                f"{peak.c:10.4f} {peak.sigma:10.2e} {peak.fraction:6.3f} "
                f"{peak.n_in:5d} {peak.longest_omega_run:5d}"
            )
    print()
    print("Per-trial figures:")
    for path in trial_paths:
        print(f"  {path.relative_to(CLASSIFICATION_ROOT)}")


if __name__ == "__main__":
    main()
