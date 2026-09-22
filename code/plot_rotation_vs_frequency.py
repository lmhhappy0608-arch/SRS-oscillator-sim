

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import PROJECT_ROOT
from rotation_cleaning import (
    CleanParams,
    cleaned_data_dir,
    clean_rotation_profile,
    cleaning_label,
    default_cleaned_figure_dir,
    save_cleaned_trial_copy,
)

RAW_DATA_DIR = PROJECT_ROOT / "raw_trials_rk4"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures" / "rotation_vs_frequency"

# Edit these defaults, or pass --k / --delta on the command line.
K_VALUE = 0.4
DELTA_VALUE = 0.6
P_VALUE = 0.8

# Cleaning: only applied when CLEAN_METHOD is not "none".
# none | monotone | monotone_smooth
CLEAN_METHOD = "monotone"
COMPARE_RAW = False
SAVE_CLEANED_NPZ = False
SMOOTH_WINDOW = 5
STATE_COLORS = {
    "Locking": "#2166ac",
    "Partial Locking": "#fdae61",
    "Incoherence": "#1a9850",
    "Partial Death": "#d73027",
    "Death": "#762a83",
}


@dataclass(frozen=True)
class ClassifyParams:
    tol_death: float = 1e-2
    min_fraction: float = 0.05
    lock_threshold: float = 0.95
    tol_lock_rho: float = 0.01
    tol_plateau_rho: float = 0.02
    min_omega_span: float = 0.03
    min_plateau_points: int = 15
    tol_identity: float = 0.01
    identity_threshold: float = 0.90
    min_zero_run: int = 4


@dataclass(frozen=True)
class TrialRecord:
    K: float
    delta: float
    p: float
    trial: int
    omega: np.ndarray
    rotation_number: np.ndarray
    state: str
    path: Path
    rotation_number_raw: np.ndarray | None = None
    state_raw: str | None = None


def find_locking_plateaus(
    omega: np.ndarray,
    rotation_number: np.ndarray,
    tol_plateau_rho: float = 0.02,
    min_omega_span: float = 0.03,
    min_points: int = 15,
) -> list[dict[str, float | int]]:
    """
    Detect horizontal plateaus in omega-rho space.

    Scan consecutive oscillators sorted by natural frequency. A plateau is a
    window where rho stays nearly flat while omega spans a meaningful range.
    """
    omega = np.asarray(omega, dtype=float)
    rho = np.asarray(rotation_number, dtype=float)
    n = len(rho)
    if n == 0:
        return []

    idx = np.argsort(omega)
    omega_sorted = omega[idx]
    rho_sorted = rho[idx]

    plateaus: list[dict[str, float | int]] = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n:
            rho_window = rho_sorted[start : end + 2]
            if rho_window.max() - rho_window.min() <= tol_plateau_rho:
                end += 1
            else:
                break

        size = end - start + 1
        if size >= min_points:
            omega_span = float(omega_sorted[end] - omega_sorted[start])
            if omega_span >= min_omega_span:
                rho_window = rho_sorted[start : end + 1]
                plateaus.append(
                    {
                        "rho_center": float(np.mean(rho_window)),
                        "size": int(size),
                        "fraction": float(size / n),
                        "omega_span": omega_span,
                        "rho_span": float(rho_window.max() - rho_window.min()),
                    }
                )

        start = end + 1

    return plateaus


def has_consecutive_zero_rotation(
    omega: np.ndarray,
    rotation_number: np.ndarray,
    tol_death: float = 1e-2,
    min_points: int = 4,
) -> bool:
    """True if at least min_points consecutive oscillators have rho ≈ 0 (sorted by omega)."""
    rho_sorted = np.asarray(rotation_number, dtype=float)[np.argsort(omega)]
    run = 0
    for rho in rho_sorted:
        if abs(rho) < tol_death:
            run += 1
            if run >= min_points:
                return True
        else:
            run = 0
    return False


def classify_state(
    omega: np.ndarray,
    rotation_number: np.ndarray,
    params: ClassifyParams | None = None,
) -> str:
    if params is None:
        params = ClassifyParams()

    omega = np.asarray(omega, dtype=float)
    rho = np.asarray(rotation_number, dtype=float)
    n = len(rho)

    zero_fraction = np.mean(np.abs(rho) < params.tol_death)
    if zero_fraction >= params.lock_threshold:
        return "Death"

    if has_consecutive_zero_rotation(
        omega,
        rho,
        tol_death=params.tol_death,
        min_points=params.min_zero_run,
    ):
        return "Partial Death"

    plateaus = find_locking_plateaus(
        omega,
        rho,
        tol_plateau_rho=params.tol_plateau_rho,
        min_omega_span=params.min_omega_span,
        min_points=params.min_plateau_points,
    )
    nonzero_plateaus = [p for p in plateaus if abs(p["rho_center"]) >= params.tol_death]

    if rho.max() - rho.min() <= params.tol_lock_rho:
        return "Locking"

    largest_nonzero_fraction = (
        max(p["fraction"] for p in nonzero_plateaus) if nonzero_plateaus else 0.0
    )
    if largest_nonzero_fraction >= params.lock_threshold:
        return "Locking"

    if nonzero_plateaus:
        return "Partial Locking"

    identity_fraction = np.mean(np.abs(rho - omega) < params.tol_identity)
    if identity_fraction >= params.identity_threshold:
        return "Incoherence"

    return "Incoherence"


def _format_delta_filename(delta: float) -> str:
    return f"{delta:.3f}"


def trial_data_path(K: float, delta: float, p: float, trial: int) -> Path:
    return RAW_DATA_DIR / (
        f"K_{K:.2f}_delta_{_format_delta_filename(delta)}_p_{p:.2f}_trial_{trial:02d}.npz"
    )


def legacy_trial_data_path(K: float, delta: float, p: float, trial: int) -> Path:
    return RAW_DATA_DIR / (
        f"K_{K:.2f}_delta_{delta:.2f}_p_{p:.2f}_trial_{trial:02d}.npz"
    )


def resolve_trial_data_path(K: float, delta: float, p: float, trial: int) -> Path:
    path = trial_data_path(K, delta, p, trial)
    if path.exists():
        return path
    legacy = legacy_trial_data_path(K, delta, p, trial)
    if legacy.exists():
        return legacy
    return path


def load_trial(
    K: float,
    delta: float,
    p: float,
    trial: int,
    classify_params: ClassifyParams,
    clean_params: CleanParams | None = None,
    save_cleaned_npz: bool = False,
) -> TrialRecord:
    path = resolve_trial_data_path(K, delta, p, trial)
    if not path.exists():
        raise FileNotFoundError(path)

    if clean_params is None:
        clean_params = CleanParams()

    data = np.load(path, allow_pickle=True)
    omega = np.asarray(data["omega"], dtype=float)
    rotation_number_raw = np.asarray(data["rotation_number"], dtype=float)
    state_raw = classify_state(omega, rotation_number_raw, classify_params)

    using_clean = clean_params.method not in ("none", "", "off")
    if using_clean:
        rotation_number = clean_rotation_profile(
            omega, rotation_number_raw, clean_params
        )
        if save_cleaned_npz:
            save_cleaned_trial_copy(
                path,
                cleaned_data_dir(RAW_DATA_DIR),
                omega,
                rotation_number,
                clean_params,
            )
    else:
        rotation_number = rotation_number_raw

    state = classify_state(omega, rotation_number, classify_params)

    return TrialRecord(
        K=float(data["K"]),
        delta=float(data["delta"]),
        p=float(data["p"]),
        trial=int(data["trial"]),
        omega=omega,
        rotation_number=rotation_number,
        state=state,
        path=path,
        rotation_number_raw=rotation_number_raw if using_clean else None,
        state_raw=state_raw if using_clean else None,
    )


def discover_raw_cells() -> dict[tuple[float, float, float], list[int]]:
    """Map (K, delta, p) -> sorted trial indices present on disk."""
    pattern = re.compile(
        r"K_([\d.]+)_delta_([\d.]+)_p_([\d.]+)_trial_(\d+)\.npz$"
    )
    cells: dict[tuple[float, float, float], list[int]] = {}
    if not RAW_DATA_DIR.exists():
        return cells

    for path in RAW_DATA_DIR.glob("*.npz"):
        match = pattern.match(path.name)
        if not match:
            continue
        key = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        trial = int(match.group(4))
        cells.setdefault(key, []).append(trial)

    for key in cells:
        cells[key] = sorted(cells[key])
    return cells


def print_raw_inventory(
    raw_cells: dict[tuple[float, float, float], list[int]],
    p_filter: float | None = None,
) -> None:
    """Print all (K, delta, p) combinations present on disk."""
    items = sorted(
        (key, trials)
        for key, trials in raw_cells.items()
        if p_filter is None or abs(key[2] - p_filter) < 1e-6
    )
    if not items:
        scope = f"p={p_filter:.2f}" if p_filter is not None else "all p"
        print(f"No raw trial data on disk ({scope}).")
        return

    deltas = sorted({d for (_, d, _) in (k for k, _ in items)})
    Ks = sorted({K for (K, _, _) in (k for k, _ in items)})
    total_files = sum(len(t) for _, t in items)

    print(f"\n=== Raw trial inventory ({len(items)} cells, {total_files} files) ===")
    print(f"K values:     {[f'{K:.2f}' for K in Ks]}")
    print(f"delta values: {[f'{d:.3f}' for d in deltas]}")
    print(f"p values:     {[f'{p:.2f}' for p in sorted({k[2] for k, _ in items})]}")
    print()
    print(f"{'K':>6} {'delta':>6} {'p':>5}  trials")
    print("-" * 40)
    for (K, delta, p), trials in items:
        print(f"{K:6.2f} {delta:6.3f} {p:5.2f}  {len(trials):2d}  {trials}")


def resolve_cells(
    k_list: list[float] | None,
    delta_list: list[float] | None,
    p: float,
    default_k: float,
    default_delta: float,
) -> list[tuple[float, float, float]]:
    if k_list is None and delta_list is None:
        return [(float(default_k), float(default_delta), float(p))]
    if k_list is None or delta_list is None:
        raise SystemExit("Provide both --k and --delta, or neither (use script defaults).")
    if len(k_list) != len(delta_list):
        raise SystemExit(
            f"--k and --delta must have the same length "
            f"(got {len(k_list)} and {len(delta_list)})"
        )
    return [(float(k), float(d), float(p)) for k, d in zip(k_list, delta_list)]


def trials_for_cell(
    K: float,
    delta: float,
    p: float,
    trial_args: list[int] | None,
    raw_cells: dict[tuple[float, float, float], list[int]],
) -> list[int]:
    available = raw_cells.get((K, delta, p), [])
    if trial_args is None:
        return available
    return [t for t in trial_args if t in available]


def plot_single_cell(
    K: float,
    delta: float,
    p: float,
    records: list[TrialRecord],
    figure_dir: Path,
    mode: str,
    dpi: int,
    show: bool,
    compare_raw: bool = False,
    clean_params: CleanParams | None = None,
) -> Path | None:
    if not records:
        return None

    fig, ax = plt.subplots(figsize=(8, 5.5))

    if mode == "average":
        omega = records[0].omega
        rho_stack = np.stack([r.rotation_number for r in records], axis=0)
        rho_mean = np.mean(rho_stack, axis=0)
        rho_std = np.std(rho_stack, axis=0)
        order = np.argsort(omega)
        omega_s = omega[order]
        ax.plot(omega_s, rho_mean[order], color="#2166ac", linewidth=1.8, label="mean")
        ax.fill_between(
            omega_s,
            (rho_mean - rho_std)[order],
            (rho_mean + rho_std)[order],
            color="#2166ac",
            alpha=0.2,
            label="±1 std",
        )
        title_mode = f"mean over {len(records)} trials"
    else:
        cmap = plt.cm.viridis
        for i, rec in enumerate(records):
            order = np.argsort(rec.omega)
            color = STATE_COLORS.get(rec.state, cmap(i / max(len(records) - 1, 1)))
            if compare_raw and rec.rotation_number_raw is not None:
                ax.scatter(
                    rec.omega[order],
                    rec.rotation_number_raw[order],
                    s=6,
                    alpha=0.25,
                    color="#999999",
                    marker="x",
                )
                label = (
                    f"trial {rec.trial:02d} cleaned={rec.state}"
                    + (f" raw={rec.state_raw}" if rec.state_raw else "")
                )
            else:
                label = f"trial {rec.trial:02d} ({rec.state})"
            ax.scatter(
                rec.omega[order],
                rec.rotation_number[order],
                s=8,
                alpha=0.55,
                color=color,
                label=label,
            )
        title_mode = f"{len(records)} trial(s) overlaid"

    majority_state = records[0].state
    if len({r.state for r in records}) > 1:
        states = ", ".join(sorted({r.state for r in records}))
        majority_state = f"mixed [{states}]"

    clean_note = ""
    if clean_params is not None and clean_params.method not in ("none", "", "off"):
        clean_note = f"\ncleaning: {cleaning_label(clean_params)}"
        if compare_raw:
            clean_note += " (gray x = raw)"
    else:
        clean_note = "\ncleaning: none (raw rotation_number)"

    ax.set_xlabel(r"Natural frequency $\omega_i$")
    ax.set_ylabel(r"Rotation number $\rho_i$")
    ax.set_title(
        r"Rotation number vs. natural frequency"
        "\n"
        f"$K={K:.2f}$, $\\delta={delta:.3f}$, $p={p:.2f}$ — {title_mode}"
        "\n"
        f"classification: {majority_state}{clean_note}"
    )
    ax.grid(True, alpha=0.3)
    if len(records) <= 8:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    figure_dir.mkdir(parents=True, exist_ok=True)
    tag = f"K_{K:.2f}_delta_{_format_delta_filename(delta)}_p_{p:.2f}_{mode}"
    figure_path = figure_dir / f"rotation_vs_omega_{tag}.png"
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot rotation number vs. natural frequency from raw_trials/ "
            "and classify phase state at plot time."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python code/plot_rotation_vs_frequency.py\n"
            "  python code/plot_rotation_vs_frequency.py --k 0.0 0.0 --delta 0.10 0.15\n"
            "  python code/plot_rotation_vs_frequency.py --k 0.4 --delta 0.3 --trial 0\n"
            "  python code/plot_rotation_vs_frequency.py --lock-threshold 0.90\n"
            "  python code/plot_rotation_vs_frequency.py --clean monotone --compare-raw\n"
            "  python code/plot_rotation_vs_frequency.py --clean monotone_smooth "
            "--smooth-window 5 --save-cleaned-npz\n"
        ),
    )
    parser.add_argument(
        "--clean",
        choices=("none", "monotone", "monotone_smooth"),
        default=CLEAN_METHOD,
        help=(
            "Preprocess rho(omega) before classify/plot. "
            "'monotone' applies K's carry-forward rule; "
            "'monotone_smooth' adds moving-average smoothing."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=SMOOTH_WINDOW,
        help="Odd window for monotone_smooth (default: 5)",
    )
    parser.add_argument(
        "--compare-raw",
        action=argparse.BooleanOptionalAction,
        default=COMPARE_RAW,
        help="Overlay raw rho (gray x) with cleaned rho when --clean is enabled",
    )
    parser.add_argument(
        "--save-cleaned-npz",
        action=argparse.BooleanOptionalAction,
        default=SAVE_CLEANED_NPZ,
        help=(
            "Write cleaned copies to a sibling *_cleaned/ directory; "
            "never overwrites raw npz files"
        ),
    )
    parser.add_argument(
        "--k",
        type=float,
        nargs="+",
        default=None,
        help=f"Coupling K (default: script constant {K_VALUE})",
    )
    parser.add_argument(
        "--delta",
        type=float,
        nargs="+",
        default=None,
        help=f"Frequency width delta (default: script constant {DELTA_VALUE})",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=P_VALUE,
        help=f"ER connection probability (default: {P_VALUE})",
    )
    parser.add_argument(
        "--trial",
        type=int,
        nargs="+",
        default=None,
        help="Trial index(es) to plot (default: all available for each cell)",
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="Print raw_trials/ inventory and exit (no plots)",
    )
    parser.add_argument(
        "--mode",
        choices=("overlay", "average"),
        default="overlay",
        help="Plot trials overlaid or averaged (default: overlay)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Directory for saved figures (default: {DEFAULT_FIGURE_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution (default: 150)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively",
    )
    # Classification thresholds (edit ClassifyParams defaults or pass here)
    parser.add_argument(
        "--tol-death",
        type=float,
        default=ClassifyParams.tol_death,
        help="|rho| below this counts as death (default: 1e-2)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=ClassifyParams.min_fraction,
        help="Minimum zero-oscillator fraction for Partial Death (default: 0.05)",
    )
    parser.add_argument(
        "--lock-threshold",
        type=float,
        default=ClassifyParams.lock_threshold,
        help="Plateau coverage fraction for Locking label (default: 0.95)",
    )
    parser.add_argument(
        "--tol-lock-rho",
        type=float,
        default=ClassifyParams.tol_lock_rho,
        help="Global rho span below this counts as Locking (default: 0.01)",
    )
    parser.add_argument(
        "--tol-plateau-rho",
        type=float,
        default=ClassifyParams.tol_plateau_rho,
        help="Max rho span within a plateau (default: 0.02)",
    )
    parser.add_argument(
        "--min-omega-span",
        type=float,
        default=ClassifyParams.min_omega_span,
        help="Min omega span for a valid plateau (default: 0.03)",
    )
    parser.add_argument(
        "--min-plateau-points",
        type=int,
        default=ClassifyParams.min_plateau_points,
        help="Min oscillators in a plateau (default: 15)",
    )
    parser.add_argument(
        "--min-zero-run",
        type=int,
        default=ClassifyParams.min_zero_run,
        help="Min consecutive rho≈0 points for Partial Death (default: 4)",
    )
    parser.add_argument(
        "--tol-identity",
        type=float,
        default=ClassifyParams.tol_identity,
        help="|rho - omega| tolerance for Incoherence (default: 0.01)",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=ClassifyParams.identity_threshold,
        help="Fraction near rho=omega required for Incoherence (default: 0.90)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        print("error: --smooth-window must be a positive odd integer", file=sys.stderr)
        return 1

    clean_params = CleanParams(
        method=args.clean,
        smooth_window=args.smooth_window,
    )
    classify_params = ClassifyParams(
        tol_death=args.tol_death,
        min_fraction=args.min_fraction,
        lock_threshold=args.lock_threshold,
        tol_lock_rho=args.tol_lock_rho,
        tol_plateau_rho=args.tol_plateau_rho,
        min_omega_span=args.min_omega_span,
        min_plateau_points=args.min_plateau_points,
        tol_identity=args.tol_identity,
        identity_threshold=args.identity_threshold,
        min_zero_run=args.min_zero_run,
    )
    raw_cells = discover_raw_cells()

    if args.list_available:
        print_raw_inventory(raw_cells, args.p)
        return 0

    cells = resolve_cells(args.k, args.delta, args.p, K_VALUE, DELTA_VALUE)
    figure_dir = (
        default_cleaned_figure_dir(args.output_dir, clean_params)
        if clean_params.method not in ("none", "", "off")
        else args.output_dir
    )
    print(
        f"Plotting {len(cells)} cell(s), p={args.p:.2f} "
        f"(classify at plot time, lock_threshold={classify_params.lock_threshold})"
    )
    if clean_params.method not in ("none", "", "off"):
        print(f"Cleaning: {cleaning_label(clean_params)}")
        print(f"Figure output: {figure_dir}/")
        if args.save_cleaned_npz:
            print(f"Cleaned npz copies: {cleaned_data_dir(RAW_DATA_DIR)}/")
    else:
        print("Cleaning: none (using raw rotation_number from npz)")
        print(f"Figure output: {figure_dir}/")
    if args.compare_raw and clean_params.method in ("none", "", "off"):
        print("warning: --compare-raw has no effect without --clean", file=sys.stderr)

    saved: list[Path] = []
    skipped: list[tuple[float, float, float]] = []

    for K, delta, p in cells:
        trial_ids = trials_for_cell(K, delta, p, args.trial, raw_cells)
        if not trial_ids:
            skipped.append((K, delta, p))
            print(f"Skip K={K:.2f}, delta={delta:.2f}, p={p:.2f}: no raw trials on disk")
            print(f"  expected: {trial_data_path(K, delta, p, 0).name}")
            continue

        records = [
            load_trial(
                K,
                delta,
                p,
                trial,
                classify_params,
                clean_params=clean_params,
                save_cleaned_npz=args.save_cleaned_npz,
            )
            for trial in trial_ids
        ]
        for rec in records:
            if rec.state_raw is not None and rec.state_raw != rec.state:
                print(
                    f"  K={rec.K:.2f}, delta={rec.delta:.2f}, trial={rec.trial:02d} "
                    f"-> cleaned={rec.state} (raw={rec.state_raw})"
                )
            else:
                print(
                    f"  K={rec.K:.2f}, delta={rec.delta:.2f}, trial={rec.trial:02d} "
                    f"-> {rec.state}"
                )

        path = plot_single_cell(
            K,
            delta,
            p,
            records,
            figure_dir,
            args.mode,
            args.dpi,
            args.show,
            compare_raw=args.compare_raw,
            clean_params=clean_params,
        )
        if path is not None:
            saved.append(path)
            print(f"Saved {path}")

    print(f"\nDone: {len(saved)} figure(s) saved to {figure_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} cell(s) without raw data.")
        print("Run with --list-available to see available (K, delta).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
