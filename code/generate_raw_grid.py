#!/usr/bin/env python3
"""
Generate raw Winfree/ER trial data on a (K, delta) grid.

Saves ODE numerical results only (omega, rotation_number). Phase-state
classification is done later in plot_rotation_vs_frequency.py so you can
change classify rules without re-running simulations.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from project_paths import PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "raw_trials_rk4"

# Default grid: 11×11 (K step 0.2, δ step 0.1).
DEFAULT_K_STEP = 0.1
DEFAULT_DELTA_STEP = 0.05
DEFAULT_P_VALUES = (0.8,)

# Physics (match Ariaratnam2001.py meeting version)
N = 1000
T = 10000.0
T0 = 8000.0

# Per-process omega cache (same omega for all K at fixed delta)
_OMEGA_CACHE: dict[tuple[float, int], np.ndarray] = {}


def build_axis(vmin: float, vmax: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    n = int(round((vmax - vmin) / step)) + 1
    values = vmin + step * np.arange(n, dtype=float)
    if values[-1] > vmax + 1e-9:
        values = values[values <= vmax + 1e-9]
    return values


def natural_frequencies(delta: float, n: int = N) -> np.ndarray:
    key = (float(delta), n)
    cached = _OMEGA_CACHE.get(key)
    if cached is None:
        if n == 1:
            cached = np.array([1.0])
        else:
            cached = np.linspace(1.0 - delta, 1.0 + delta, n)
        _OMEGA_CACHE[key] = cached
    return cached


def make_er_adjacency(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Undirected ER graph via upper triangle only (half the random draws)."""
    idx = np.triu_indices(n, k=1)
    edges = rng.random(idx[0].size) < p
    adjacency = np.zeros((n, n), dtype=float)
    adjacency[idx] = edges
    adjacency += adjacency.T
    return adjacency


def _make_ode_buffers(n: int = N) -> dict[str, np.ndarray]:
    return {
        "one_plus_cos": np.empty(n),
        "interaction": np.empty(n),
        "sin_theta": np.empty(n),
        "dtheta": np.empty(n),
        "k1": np.empty(n),
        "k2": np.empty(n),
        "k3": np.empty(n),
        "k4": np.empty(n),
        "ystage": np.empty(n),
        "y_next": np.empty(n),
        "theta_t0": np.empty(n),
        "theta_t": np.empty(n),
    }


def _max_omega_magnitude(omega: np.ndarray) -> float:
    return float(np.max(np.abs(omega)))


def default_rk4_dt(omega: np.ndarray) -> float:
    """Fixed RK4 step: ~20 samples per fastest natural period 2π/ω."""
    wmax = max(_max_omega_magnitude(omega), 1.0)
    return min(0.1, (2.0 * np.pi) / (20.0 * wmax))


def winfree_rhs(
    theta: np.ndarray,
    omega: np.ndarray,
    adjacency: np.ndarray,
    K: float,
    n: int,
    buffers: dict[str, np.ndarray],
    out: np.ndarray,
) -> None:
    one_plus_cos = buffers["one_plus_cos"]
    interaction = buffers["interaction"]
    sin_theta = buffers["sin_theta"]
    scale = K / n

    np.cos(theta, out=one_plus_cos)
    np.add(one_plus_cos, 1.0, out=one_plus_cos)
    np.matmul(adjacency, one_plus_cos, out=interaction)
    np.sin(theta, out=sin_theta)
    np.multiply(sin_theta, interaction, out=out)
    np.multiply(out, scale, out=out)
    np.subtract(omega, out, out=out)


def rk4_step(
    theta: np.ndarray,
    h: float,
    omega: np.ndarray,
    adjacency: np.ndarray,
    K: float,
    n: int,
    buffers: dict[str, np.ndarray],
) -> np.ndarray:
    k1 = buffers["k1"]
    k2 = buffers["k2"]
    k3 = buffers["k3"]
    k4 = buffers["k4"]
    ystage = buffers["ystage"]
    y_next = buffers["y_next"]
    h2 = 0.5 * h
    h6 = h / 6.0

    winfree_rhs(theta, omega, adjacency, K, n, buffers, k1)

    np.multiply(k1, h2, out=ystage)
    np.add(theta, ystage, out=ystage)
    winfree_rhs(ystage, omega, adjacency, K, n, buffers, k2)

    np.multiply(k2, h2, out=ystage)
    np.add(theta, ystage, out=ystage)
    winfree_rhs(ystage, omega, adjacency, K, n, buffers, k3)

    np.multiply(k3, h, out=ystage)
    np.add(theta, ystage, out=ystage)
    winfree_rhs(ystage, omega, adjacency, K, n, buffers, k4)

    np.multiply(k2, 2.0, out=ystage)
    np.add(k1, ystage, out=ystage)
    np.multiply(k3, 2.0, out=k3)
    np.add(ystage, k3, out=ystage)
    np.add(ystage, k4, out=ystage)
    np.multiply(ystage, h6, out=ystage)
    np.add(theta, ystage, out=y_next)
    return y_next


def _capture_crossing(
    t: float,
    t_next: float,
    target: float,
    got: bool,
    y_before: np.ndarray,
    y_after: np.ndarray,
    out: np.ndarray,
) -> bool:
    if not got and t < target <= t_next:
        alpha = (target - t) / (t_next - t)
        np.subtract(y_after, y_before, out=out)
        np.multiply(out, alpha, out=out)
        np.add(y_before, out, out=out)
        return True
    return got


def integrate_rk4_to_times(
    theta0: np.ndarray,
    omega: np.ndarray,
    adjacency: np.ndarray,
    K: float,
    n: int,
    buffers: dict[str, np.ndarray],
    t0: float = T0,
    t_end: float = T,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    step = default_rk4_dt(omega) if dt is None else dt
    theta = theta0.copy()
    theta_t0 = buffers["theta_t0"]
    theta_t = buffers["theta_t"]
    y_before = buffers["dtheta"]
    t = 0.0
    got_t0 = False
    got_t = False

    while t < t_end and not (got_t0 and got_t):
        h = min(step, t_end - t)
        np.copyto(y_before, theta)
        y_after = rk4_step(theta, h, omega, adjacency, K, n, buffers)
        t_next = t + h

        got_t0 = _capture_crossing(t, t_next, t0, got_t0, y_before, y_after, theta_t0)
        got_t = _capture_crossing(t, t_next, t_end, got_t, y_before, y_after, theta_t)

        np.copyto(theta, y_after)
        t = t_next

    if not got_t0 or not got_t:
        raise RuntimeError("RK4 integration did not reach T0 and T")

    return theta_t0, theta_t


def simulate_winfree(
    K: float,
    delta: float,
    p: float,
    rng: np.random.Generator,
    buffers: dict[str, np.ndarray],
    n: int = N,
) -> tuple[np.ndarray, np.ndarray]:
    omega = natural_frequencies(delta, n)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, n)
    adjacency = make_er_adjacency(n, p, rng)

    theta_t0, theta_t = integrate_rk4_to_times(
        theta0, omega, adjacency, K, n, buffers
    )
    rotation_number = (theta_t - theta_t0) / (T - T0)
    return omega, rotation_number


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


def save_trial_raw_data(
    K: float,
    delta: float,
    p: float,
    trial: int,
    omega: np.ndarray,
    rotation_number: np.ndarray,
) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = trial_data_path(K, delta, p, trial)
    np.savez_compressed(
        path,
        K=np.float64(K),
        delta=np.float64(delta),
        p=np.float64(p),
        trial=np.int32(trial),
        N=np.int32(N),
        T=np.float64(T),
        T0=np.float64(T0),
        integrator=np.array("RK4"),
        omega=omega,
        rotation_number=rotation_number,
    )
    return path


def _trial_seed(K: float, delta: float, p: float, trial: int) -> int:
    return hash((K, delta, p, trial)) & 0xFFFFFFFF


def run_cell(task: tuple[float, float, float, int, bool]) -> dict[str, object]:
    """Worker entry: all trials for one (K, delta, p) cell; saves ODE data only."""
    K, delta, p, n_trials, skip_existing = task
    buffers = _make_ode_buffers(N)
    computed = 0
    skipped = 0

    for trial in range(n_trials):
        path = resolve_trial_data_path(K, delta, p, trial)
        if skip_existing and path.exists():
            skipped += 1
            continue

        rng = np.random.default_rng(_trial_seed(K, delta, p, trial))
        omega, rotation_number = simulate_winfree(
            K, delta, p, rng=rng, buffers=buffers, n=N
        )
        save_trial_raw_data(K, delta, p, trial, omega, rotation_number)
        computed += 1

    return {
        "K": K,
        "delta": delta,
        "p": p,
        "computed": computed,
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate Winfree dynamics on ER networks for every (K, delta) grid "
            "point and save ODE results (omega, rotation_number) to raw_trials_rk4/."
        )
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_P_VALUES),
        metavar="P",
        help=(
            "ER connection probabilities "
            f"(default: {' '.join(f'{p:.1f}' for p in DEFAULT_P_VALUES)})"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Trials per (K, delta) cell (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel processes over grid cells (default: CPU count - 1)",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip npz files that already exist in raw_trials_rk4/ (default: on)",
    )
    parser.add_argument(
        "--force",
        dest="skip_existing",
        action="store_false",
        help="Recompute all cells even if npz files already exist in raw_trials_rk4/",
    )
    parser.add_argument(
        "--k-min", type=float, default=0.0, help="Minimum K (default: 0.0)"
    )
    parser.add_argument(
        "--k-max", type=float, default=2.0, help="Maximum K (default: 2.0)"
    )
    parser.add_argument(
        "--k-step",
        type=float,
        default=DEFAULT_K_STEP,
        help=f"K grid step (default: {DEFAULT_K_STEP})",
    )
    parser.add_argument(
        "--delta-min",
        type=float,
        default=0.0,
        help="Minimum delta (default: 0.0)",
    )
    parser.add_argument(
        "--delta-max",
        type=float,
        default=1.0,
        help="Maximum delta (default: 1.0)",
    )
    parser.add_argument(
        "--delta-step",
        type=float,
        default=DEFAULT_DELTA_STEP,
        help=f"delta grid step (default: {DEFAULT_DELTA_STEP})",
    )
    return parser.parse_args()


def count_trial_files(
    k_values: np.ndarray,
    delta_values: np.ndarray,
    p_values: list[float],
    n_trials: int,
) -> tuple[int, int]:
    """Return (existing, missing) trial npz counts for the requested grid."""
    existing = 0
    missing = 0
    for p in p_values:
        for delta in delta_values:
            for K in k_values:
                for trial in range(n_trials):
                    if resolve_trial_data_path(float(K), float(delta), p, trial).exists():
                        existing += 1
                    else:
                        missing += 1
    return existing, missing


def main() -> int:
    args = parse_args()

    if args.trials < 1:
        print("error: --trials must be >= 1", file=sys.stderr)
        return 1
    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 1
    p_values = [float(p) for p in args.p_values]
    if not p_values:
        print("error: --p-values must list at least one value", file=sys.stderr)
        return 1
    if any(not (0.0 <= p <= 1.0) for p in p_values):
        print("error: every p in --p-values must be in [0, 1]", file=sys.stderr)
        return 1

    k_values = build_axis(args.k_min, args.k_max, args.k_step)
    delta_values = build_axis(args.delta_min, args.delta_max, args.delta_step)
    n_cells_per_p = len(k_values) * len(delta_values)
    n_cells = n_cells_per_p * len(p_values)
    n_files = n_cells * args.trials

    print("=== generate_raw_grid ===")
    print(f"Output: {RAW_DATA_DIR}/")
    print(f"N={N}, T={T}, T0={T0}, method=RK4 (fixed step, auto dt from omega)")
    print(
        f"p values: {', '.join(f'{p:.2f}' for p in p_values)}, "
        f"trials/cell={args.trials}, workers={args.workers}"
    )
    print(
        f"K: [{args.k_min}, {args.k_max}] step {args.k_step} -> {len(k_values)} values"
    )
    print(
        f"delta: [{args.delta_min}, {args.delta_max}] step {args.delta_step} "
        f"-> {len(delta_values)} values"
    )
    print(f"Grid per p: {len(delta_values)} x {len(k_values)} = {n_cells_per_p} cells")
    print(f"Total cells ({len(p_values)} p): {n_cells}")
    print(f"Expected npz files: {n_files}")
    print(f"skip-existing: {args.skip_existing}")

    on_disk, to_compute = count_trial_files(
        k_values, delta_values, p_values, args.trials
    )
    print(f"On disk: {on_disk} trial files already in {RAW_DATA_DIR.name}/")
    if args.skip_existing:
        print(f"Will compute: {to_compute} new trial files (reuse {on_disk} existing)")
    else:
        print(f"Will compute: {n_files} trial files (--force, ignoring existing)")
    print("Saved fields: omega, rotation_number (+ metadata). No state label.")
    print()

    tasks = [
        (float(K), float(delta), float(p), args.trials, args.skip_existing)
        for p in p_values
        for delta in delta_values
        for K in k_values
    ]

    t_start = time.perf_counter()
    done = 0
    total_computed = 0
    total_skipped = 0

    if args.workers <= 1:
        for task in tasks:
            result = run_cell(task)
            done += 1
            total_computed += int(result["computed"])
            total_skipped += int(result["skipped"])
            print(
                f"[{done}/{n_cells}] p={result['p']:.2f}, "
                f"K={result['K']:.2f}, delta={result['delta']:.2f}: "
                f"computed={result['computed']}, skipped={result['skipped']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_cell, task): task for task in tasks}
            for future in as_completed(future_map):
                result = future.result()
                done += 1
                total_computed += int(result["computed"])
                total_skipped += int(result["skipped"])
                print(
                    f"[{done}/{n_cells}] p={result['p']:.2f}, "
                    f"K={result['K']:.2f}, delta={result['delta']:.2f}: "
                    f"computed={result['computed']}, skipped={result['skipped']}"
                )

    elapsed = time.perf_counter() - t_start
    print()
    print(
        f"Done in {elapsed:.1f}s — computed {total_computed} new files, "
        f"skipped {total_skipped} existing"
    )
    if total_computed:
        print(f"Average {elapsed / total_computed:.2f}s per newly computed trial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
