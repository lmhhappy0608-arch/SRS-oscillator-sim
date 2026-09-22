"""Winfree phase diagrams from C++ simulation data in cpp/raw_trials_c/."""

import argparse
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from project_paths import PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "cpp" / "raw_trials_c_rk4"

# Parameters
N = 1000
T = 10000
T0 = 8000

t_eval = np.array([T0, T], dtype=float)

# C++ grid in cpp/raw_trials_c/: K 0→2 step 0.1, δ 0→1 step 0.05 → 21×21 = 441
# cells (currently p=0.8 only).
REFINED_K_STEP = 0.2
REFINED_DELTA_STEP = 0.1
K_values = np.arange(0.00, 2.00 + 1e-9, REFINED_K_STEP)
delta_values = np.arange(0.00, 1.00 + 1e-9, REFINED_DELTA_STEP)
DEFAULT_P_VALUES = (0.8,)

num_trials = 1

_omega_cache: dict[float, np.ndarray] = {}


def natural_frequencies(delta, n=N):
    key = (delta, n)
    if key not in _omega_cache:
        if n == 1:
            _omega_cache[key] = np.array([1.0])
        else:
            _omega_cache[key] = np.linspace(1.0 - delta, 1.0 + delta, n)
    return _omega_cache[key]


def make_er_adjacency(n, p, rng=None):
    """Sample undirected ER edges via upper-triangle only (half the random draws)."""
    rng = np.random.default_rng() if rng is None else rng
    idx = np.triu_indices(n, k=1)
    edges = rng.random(idx[0].size) < p
    A = np.zeros((n, n), dtype=float)
    A[idx] = edges
    A += A.T
    return A


def _make_ode_buffers(n=N):
    return {
        "one_plus_cos": np.empty(n),
        "interaction": np.empty(n),
        "sin_theta": np.empty(n),
        "dtheta": np.empty(n),
    }


def simulate_winfree(K, delta, p, rng=None, buffers=None, n=N):
    rng = np.random.default_rng() if rng is None else rng
    omega = natural_frequencies(delta, n)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, n)
    A = make_er_adjacency(n, p, rng=rng)

    if buffers is None:
        buffers = _make_ode_buffers(n)

    one_plus_cos = buffers["one_plus_cos"]
    interaction = buffers["interaction"]
    sin_theta = buffers["sin_theta"]
    dtheta = buffers["dtheta"]
    scale = K / n

    def winfree(_t, theta):
        np.cos(theta, out=one_plus_cos)
        np.add(one_plus_cos, 1.0, out=one_plus_cos)
        np.matmul(A, one_plus_cos, out=interaction)
        np.sin(theta, out=sin_theta)
        np.multiply(sin_theta, interaction, out=dtheta)
        np.multiply(dtheta, scale, out=dtheta)
        np.subtract(omega, dtheta, out=dtheta)
        return dtheta

    sol = solve_ivp(
        winfree,
        [0.0, T],
        theta0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    theta_t0 = sol.y[:, 0]
    theta_t = sol.y[:, 1]
    rotation_number = (theta_t - theta_t0) / (T - T0)

    z = np.mean(np.exp(1j * theta_t))
    R = np.abs(z)

    return omega, rotation_number, R


def trial_data_path(K, delta, p, trial):
    return RAW_DATA_DIR / (
        f"K_{K:.2f}_delta_{delta:.2f}_p_{p:.2f}_trial_{trial:02d}.npz"
    )


def load_trial_state(path):
    """Classify phase state from omega and rotation_number only (ignore npz state)."""
    data = np.load(path, allow_pickle=True)
    return classify_state(data["omega"], data["rotation_number"])


def discover_saved_p_values(raw_cells=None):
    """All p with trial data in cpp/raw_trials_c/."""
    raw_cells = discover_raw_cells() if raw_cells is None else raw_cells
    p_values = set()
    for (_, _, p_val), trials in raw_cells.items():
        if trials:
            p_values.add(float(p_val))
    return sorted(p_values)


def count_saved_cells(raw_cells=None):
    """Number of (K, δ, p) groups with at least one trial file."""
    raw_cells = discover_raw_cells() if raw_cells is None else raw_cells
    return sum(1 for trials in raw_cells.values() if trials)


def raw_data_exists_for_p(p, raw_cells=None):
    raw_cells = discover_raw_cells() if raw_cells is None else raw_cells
    return any(
        trials and abs(p_val - p) < 1e-6
        for (_, _, p_val), trials in raw_cells.items()
    )


def describe_saved_data():
    """Summarize the cpp/raw_trials_c/ dataset (for --list-saved)."""
    lines = []
    raw_cells = discover_raw_cells()
    total_cells = count_saved_cells(raw_cells)
    p_raw = discover_saved_p_values(raw_cells)
    for p in p_raw:
        items = [
            (K, d)
            for (K, d, p_val), tr in raw_cells.items()
            if tr and abs(p_val - p) < 1e-6
        ]
        k_grid = sorted({K for K, _ in items})
        d_grid = sorted({d for _, d in items})
        n_trials = max(
            len(tr)
            for (_, _, p_val), tr in raw_cells.items()
            if tr and abs(p_val - p) < 1e-6
        )
        k_step = k_grid[1] - k_grid[0] if len(k_grid) > 1 else float("nan")
        d_step = d_grid[1] - d_grid[0] if len(d_grid) > 1 else float("nan")
        lines.append(
            f"  p={p:.2f}: {len(d_grid)}×{len(k_grid)} (δ×K), "
            f"K step {k_step:.3g}, δ step {d_step:.3g}, "
            f"up to {n_trials} trial(s)/cell"
        )
    if not lines:
        lines.append("  (empty)")
    header = (
        f"cpp/raw_trials_c/: {total_cells} (K,δ,p) groups, "
        f"{sum(len(tr) for tr in raw_cells.values())} npz files"
    )
    return [header, *lines], p_raw


def discover_raw_cells():
    """Map (K, delta, p) -> sorted trial indices present in cpp/raw_trials_c/."""
    pattern = re.compile(
        r"K_([\d.]+)_delta_([\d.]+)_p_([\d.]+)_trial_(\d+)\.npz$"
    )
    cells = {}
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


def majority_state_for_cell(K, delta, p, raw_cells):
    trials = raw_cells.get((K, delta, p), [])
    if not trials:
        return None
    if len(trials) == 1:
        return load_trial_state(trial_data_path(K, delta, p, trials[0]))
    states = [load_trial_state(trial_data_path(K, delta, p, trial)) for trial in trials]
    return Counter(states).most_common(1)[0][0]


def build_phase_map_from_raw(raw_cells, p):
    """Build phase-code grid for one p from the cpp/raw_trials_c/ dataset."""
    items = [
        (K, delta)
        for (K, delta, p_val), trials in raw_cells.items()
        if trials and abs(p_val - p) < 1e-6
    ]
    if not items:
        raise FileNotFoundError(f"No raw trial data for p={p:.2f} in {RAW_DATA_DIR}/")

    k_grid = np.array(sorted({K for K, _ in items}), dtype=float)
    delta_grid = np.array(sorted({delta for _, delta in items}), dtype=float)
    phase_codes = np.full((len(delta_grid), len(k_grid)), -1, dtype=int)

    for i_delta, delta in enumerate(delta_grid):
        for i_K, K in enumerate(k_grid):
            trials = raw_cells.get((K, delta, p), [])
            if not trials:
                continue
            if len(trials) == 1:
                code = load_trial_phase_code(trial_data_path(K, delta, p, trials[0]))
            else:
                state = majority_state_for_cell(K, delta, p, raw_cells)
                code = state_to_code.get(state, -1) if state else -1
            phase_codes[i_delta, i_K] = code

    return phase_codes, k_grid, delta_grid


def load_saved_phase_codes(p, raw_cells=None):
    """Build phase-code grid from cpp/raw_trials_c/ for the given p."""
    raw_cells = discover_raw_cells() if raw_cells is None else raw_cells
    if not raw_data_exists_for_p(p, raw_cells):
        raise FileNotFoundError(
            f"No raw trial data for p={p:.2f} in {RAW_DATA_DIR}/"
        )
    return build_phase_map_from_raw(raw_cells, p)


def saved_trials_count(p, raw_cells=None):
    raw_cells = discover_raw_cells() if raw_cells is None else raw_cells
    counts = [
        len(trials)
        for (_, _, p_val), trials in raw_cells.items()
        if trials and abs(p_val - p) < 1e-6
    ]
    return int(max(counts)) if counts else None


def _resolve_plot_axes(k_values=None, delta_values=None):
    k_axis = K_values if k_values is None else np.asarray(k_values, dtype=float)
    delta_axis = (
        delta_values if delta_values is None else np.asarray(delta_values, dtype=float)
    )
    return k_axis, delta_axis


def _phase_codes_for_display(phase_codes):
    """Stored as [delta, K]; display with x=delta, y=K."""
    return phase_codes.T


def _phase_extent(k_values, delta_values):
    return [
        float(delta_values[0]),
        float(delta_values[-1]),
        float(k_values[0]),
        float(k_values[-1]),
    ]


def save_trial_raw_data(K, delta, p, trial, omega, rotation_number):
    """Save ODE results only; phase classification is done at plot/load time."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = trial_data_path(K, delta, p, trial)
    np.savez(
        path,
        K=K,
        delta=delta,
        p=p,
        trial=trial,
        N=N,
        T=T,
        T0=T0,
        omega=omega,
        rotation_number=rotation_number,
    )
    return path


def find_locking_plateaus(
    omega,
    rotation_number,
    tol_plateau_rho=0.02,
    min_omega_span=0.03,
    min_points=15,
):
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

    plateaus = []
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
    omega,
    rotation_number,
    tol_death=1e-2,
    min_points=4,
):
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
    omega,
    rotation_number,
    tol_death=1e-2,
    min_fraction=0.05,
    lock_threshold=0.95,
    tol_lock_rho=0.01,
    tol_plateau_rho=0.02,
    min_omega_span=0.03,
    min_plateau_points=15,
    tol_identity=0.01,
    identity_threshold=0.90,
    min_zero_run=4,
):
    """
    Classify from rotation-number vs natural-frequency structure.

    Priority: Death -> Partial Death -> Locking -> Partial Locking -> Incoherence
    """
    omega = np.asarray(omega, dtype=float)
    rho = np.asarray(rotation_number, dtype=float)
    n = len(rho)

    zero_fraction = np.mean(np.abs(rho) < tol_death)
    if zero_fraction >= lock_threshold:
        return "Death"

    if has_consecutive_zero_rotation(
        omega, rho, tol_death=tol_death, min_points=min_zero_run
    ):
        return "Partial Death"

    plateaus = find_locking_plateaus(
        omega,
        rho,
        tol_plateau_rho=tol_plateau_rho,
        min_omega_span=min_omega_span,
        min_points=min_plateau_points,
    )
    nonzero_plateaus = [p for p in plateaus if abs(p["rho_center"]) >= tol_death]

    if rho.max() - rho.min() <= tol_lock_rho:
        return "Locking"

    largest_nonzero_fraction = (
        max(p["fraction"] for p in nonzero_plateaus) if nonzero_plateaus else 0.0
    )
    if largest_nonzero_fraction >= lock_threshold:
        return "Locking"

    if nonzero_plateaus:
        return "Partial Locking"

    identity_fraction = np.mean(np.abs(rho - omega) < tol_identity)
    if identity_fraction >= identity_threshold:
        return "Incoherence"

    return "Incoherence"


state_to_code = {
    "Locking": 0,
    "Partial Locking": 1,
    "Incoherence": 2,
    "Partial Death": 3,
    "Death": 4,
}

code_to_state = {value: key for key, value in state_to_code.items()}


def load_trial_phase_code(path):
    state = load_trial_state(path)
    return state_to_code.get(state, -1)


state_colors = [
    "#2166ac",
    "#fdae61",
    "#1a9850",
    "#d73027",
    "#762a83",
]


def _run_trial_task(task):
    K, delta, p, trial, seed, save_raw, resume = task
    path = trial_data_path(K, delta, p, trial)
    if resume and path.exists():
        return load_trial_state(path)

    rng = np.random.default_rng(seed)
    buffers = _make_ode_buffers(N)
    omega, rho, _R = simulate_winfree(K, delta, p, rng=rng, buffers=buffers, n=N)
    state = classify_state(omega, rho)
    if save_raw:
        save_trial_raw_data(K, delta, p, trial, omega, rho)
    return state


def _run_trials(K, delta, p, n_trials, save_raw, workers, resume=True):
    tasks = [
        (K, delta, p, trial, hash((K, delta, p, trial)) & 0xFFFFFFFF, save_raw, resume)
        for trial in range(n_trials)
    ]

    if workers <= 1 or n_trials <= 1:
        return [_run_trial_task(task) for task in tasks]

    trial_states = [None] * n_trials
    with ProcessPoolExecutor(max_workers=min(workers, n_trials)) as executor:
        future_map = {
            executor.submit(_run_trial_task, task): task[3] for task in tasks
        }
        for future in as_completed(future_map):
            trial_idx = future_map[future]
            trial_states[trial_idx] = future.result()
    return trial_states


def run_parameter_sweep(p_values, plot=True, save_raw=False, workers=1, resume=True):
    """
    Run simulations incrementally over p values.

    For each (p, delta, K), runs num_trials simulations and records
    majority-vote phase classification. Raw trial data is saved only when
    save_raw=True.
    """
    all_phase_codes = {}

    for p in p_values:
        phase_codes = np.zeros((len(delta_values), len(K_values)), dtype=int)
        phase_results = []
        total = len(delta_values) * len(K_values)
        count = 0
        t_start = time.perf_counter()

        print(f"\n=== Running p = {p:.2f} ===")
        print(f"Grid: {len(delta_values)} x {len(K_values)}, trials={num_trials}, workers={workers}")

        for i_delta, delta in enumerate(delta_values):
            for i_K, K in enumerate(K_values):
                count += 1
                trial_states = _run_trials(
                    K, delta, p, num_trials, save_raw, workers, resume=resume
                )
                majority_state = Counter(trial_states).most_common(1)[0][0]
                phase_codes[i_delta, i_K] = state_to_code[majority_state]
                phase_results.append((K, delta, majority_state, trial_states))

                print(
                    f"[{count}/{total}] p={p:.2f}, K={K:.2f}, delta={delta:.2f}: "
                    f"majority={majority_state}, trials={trial_states}"
                )

        elapsed = time.perf_counter() - t_start
        print(f"Finished p={p:.2f} in {elapsed:.1f}s")

        all_phase_codes[p] = phase_codes

        if plot:
            plot_phase_diagram(phase_codes, p)

    return all_phase_codes


def plot_phase_diagram(
    phase_codes, p, k_values=None, delta_values=None, trials_count=None
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    k_axis, delta_axis = _resolve_plot_axes(k_values, delta_values)
    t_note = trials_count if trials_count is not None else num_trials
    source_note = " (from saved data)" if trials_count is not None else ""

    plt.figure(figsize=(9, 7))

    cmap = ListedColormap(state_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(state_colors) + 0.5, 1), cmap.N)

    plt.imshow(
        _phase_codes_for_display(phase_codes),
        origin="lower",
        aspect="auto",
        extent=_phase_extent(k_axis, delta_axis),
        cmap=cmap,
        norm=norm,
    )

    ax = plt.gca()
    ax.set_xticks(delta_axis)
    ax.set_yticks(k_axis)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6)

    plt.xlabel("Frequency width δ")
    plt.ylabel("Coupling strength K")
    plt.title(
        f"Phase diagram for the Winfree model (Ariaratnam & Strogatz 2001)\n"
        f"N={N}, p={p}, T={T}, T₀={T0}, trials={t_note}{source_note}"
    )

    legend_handles = [
        Patch(color=state_colors[code], label=code_to_state[code])
        for code in range(len(state_colors))
    ]
    plt.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.0),
    )

    plt.tight_layout()
    figure_path = PROJECT_ROOT / f"winfree_phase_diagram_c_p_{p:.2f}.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"Saved {figure_path}")
    plt.close()


def plot_from_saved(p_values, plot_individual=True):
    """Plot 2D phase diagrams from the cpp/raw_trials_c/ dataset."""
    raw_cells = discover_raw_cells()
    total = count_saved_cells(raw_cells)
    print(
        f"=== cpp/raw_trials_c/: {total} (K,δ,p) groups, "
        f"{sum(len(tr) for tr in raw_cells.values())} npz files ==="
    )

    plotted = []
    for p in p_values:
        try:
            phase_codes, k_grid, delta_grid = load_saved_phase_codes(p, raw_cells)
        except FileNotFoundError as exc:
            print(f"Skip p={p:.2f}: {exc}")
            continue

        n_filled = int(np.sum(phase_codes >= 0))
        print(
            f"Loaded p={p:.2f}: {len(delta_grid)}×{len(k_grid)} (δ×K), "
            f"{n_filled} cells with data"
        )
        if plot_individual:
            plot_phase_diagram(
                phase_codes,
                p,
                k_grid,
                delta_grid,
                trials_count=saved_trials_count(p, raw_cells),
            )
        plotted.append(p)

    if not plotted:
        raise FileNotFoundError("No saved data found for any requested p value")
    return plotted


def _majority_state(trial_states):
    return Counter(trial_states).most_common(1)[0][0]


def _run_trials_at_n(K, delta, p, n_trials, n_nodes, workers=1):
    """Run trials with a temporary node count (for N sensitivity checks)."""
    global N
    saved_n = N
    N = n_nodes
    try:
        return _run_trials(K, delta, p, n_trials, save_raw=False, workers=workers, resume=False)
    finally:
        N = saved_n


def compare_n_sensitivity(
    p_values,
    n_small=1000,
    n_large=10000,
    n_trials=5,
    workers=1,
    cells=None,
):
    """
    Compare majority-vote phase labels at two network sizes.

    When p < 1, ER graph realizations add noise; this quantifies how often
    N=1000 vs N=10000 disagree on the majority phase label.
    """
    if cells is None:
        cells = [
            (float(K), float(delta))
            for delta in delta_values
            for K in K_values
        ]

    results = []
    total = len(p_values) * len(cells)
    count = 0

    print(
        f"\n=== N sensitivity: N={n_small} vs N={n_large}, "
        f"trials={n_trials}, cells={len(cells)}, p values={p_values} ==="
    )

    for p in p_values:
        disagree = 0
        small_mixed = 0
        large_mixed = 0
        for K, delta in cells:
            count += 1
            small_trials = _run_trials_at_n(K, delta, p, n_trials, n_small, workers)
            large_trials = _run_trials_at_n(K, delta, p, n_trials, n_large, workers)
            small_majority = _majority_state(small_trials)
            large_majority = _majority_state(large_trials)
            if small_majority != large_majority:
                disagree += 1
            if len(set(small_trials)) > 1:
                small_mixed += 1
            if len(set(large_trials)) > 1:
                large_mixed += 1
            results.append(
                {
                    "p": p,
                    "K": K,
                    "delta": delta,
                    "small_majority": small_majority,
                    "large_majority": large_majority,
                    "small_trials": small_trials,
                    "large_trials": large_trials,
                }
            )
            print(
                f"[{count}/{total}] p={p:.2f}, K={K:.2f}, δ={delta:.2f}: "
                f"N={n_small}->{small_majority} | N={n_large}->{large_majority} "
                f"| agree={small_majority == large_majority}"
            )

        n_cells = len(cells)
        print(
            f"p={p:.2f}: disagree={disagree}/{n_cells} "
            f"({100 * disagree / n_cells:.1f}%), "
            f"within-N spread N={n_small}: {small_mixed}/{n_cells}, "
            f"N={n_large}: {large_mixed}/{n_cells}"
        )

    n_total = len(results)
    n_disagree = sum(r["small_majority"] != r["large_majority"] for r in results)
    summary = {
        "n_small": n_small,
        "n_large": n_large,
        "n_trials": n_trials,
        "total_cells": n_total,
        "disagree_cells": n_disagree,
        "disagree_fraction": n_disagree / n_total if n_total else 0.0,
        "results": results,
    }

    print(
        f"\nOverall: {n_disagree}/{n_total} cells disagree "
        f"({100 * summary['disagree_fraction']:.1f}%)"
    )
    return summary


# Borderline cells from p=0.8 run (mixed trials at N=1000).
BORDERLINE_CELLS = [
    (0.4, 0.3),
    (0.4, 0.5),
    (0.4, 0.6),
    (1.2, 0.7),
    (0.6, 0.3),
    (0.8, 0.3),
    (0.4, 0.4),
    (0.6, 0.5),
    (1.0, 0.5),
    (0.8, 0.6),
]


def _apply_benchmark_overrides():
    """Shrink grid and trial count for a quick smoke test."""
    global K_values, delta_values, num_trials
    K_values = np.array([0.0, 1.0, 2.0])
    delta_values = np.array([0.0, 0.5, 1.0])
    num_trials = 1
    print("Benchmark mode: 3x3 grid, 1 trial per cell")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Winfree phase diagram (C++ data). Default: load cpp/raw_trials_c/ "
            "(21×21 grid, p=0.8) and plot 2D diagrams. Generate data with "
            "cpp/build/generate_raw_grid; use --run only for Python smoke tests."
        )
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=None,
        help="ER connection probabilities to plot/run (default: all saved p on disk)",
    )
    parser.add_argument(
        "--list-saved",
        action="store_true",
        help="Print inventory of saved data and exit (no plots)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip phase diagram plotting",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run ODE simulations (default: load saved data and re-plot only)",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="With --run: save per-trial raw data to cpp/raw_trials_c/",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel worker processes for trials (default: 1)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Quick test: 3x3 K/delta grid, 1 trial per cell",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run trials even if raw data files already exist in cpp/raw_trials_c/",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="With --run: override num_trials (default: 5)",
    )
    parser.add_argument(
        "--compare-n",
        action="store_true",
        help="Compare phase labels at N=1000 vs N=10000 (see --compare-cells)",
    )
    parser.add_argument(
        "--n-small",
        type=int,
        default=1000,
        help="Smaller N for --compare-n (default: 1000)",
    )
    parser.add_argument(
        "--n-large",
        type=int,
        default=10000,
        help="Larger N for --compare-n (default: 10000)",
    )
    parser.add_argument(
        "--compare-cells",
        choices=("borderline", "full"),
        default="borderline",
        help="Cell set for --compare-n: borderline probes or full K–δ grid",
    )
    parser.add_argument(
        "--compare-trials",
        type=int,
        default=5,
        help="Trials per cell for --compare-n (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.benchmark:
        _apply_benchmark_overrides()

    if args.trials is not None:
        if args.trials < 1:
            raise SystemExit("--trials must be >= 1")
        globals()["num_trials"] = args.trials

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    p_values = (
        args.p_values
        if args.p_values is not None
        else discover_saved_p_values()
    )

    if args.list_saved:
        lines, all_p = describe_saved_data()
        print("=== Saved data inventory ===")
        for line in lines:
            print(line)
        print(f"All p on disk: {', '.join(f'{p:.2f}' for p in all_p) or '(none)'}")
        raise SystemExit(0)

    if args.compare_n:
        cells = BORDERLINE_CELLS if args.compare_cells == "borderline" else None
        summary = compare_n_sensitivity(
            p_values,
            n_small=args.n_small,
            n_large=args.n_large,
            n_trials=args.compare_trials,
            workers=args.workers,
            cells=cells,
        )
        out = PROJECT_ROOT / "n_sensitivity_summary_c.npz"
        np.savez(
            out,
            p_values=np.array(p_values),
            n_small=args.n_small,
            n_large=args.n_large,
            disagree_fraction=summary["disagree_fraction"],
            results=np.array(summary["results"], dtype=object),
        )
        print(f"Saved {out}")
        raise SystemExit(0)

    if args.run:
        run_parameter_sweep(
            p_values,
            plot=not args.no_plot,
            save_raw=args.save_raw,
            workers=args.workers,
            resume=not args.no_resume,
        )
        raise SystemExit(0)

    # Default: re-plot ALL saved p values — no ODE integration.
    if not p_values:
        raise SystemExit(
            "No saved data in cpp/raw_trials_c/.\n"
            "Generate data first: cd cpp && ./build/generate_raw_grid\n"
            "Or: python code/Ariaratnam2001_c.py --run --save-raw"
        )

    try:
        plot_from_saved(
            p_values,
            plot_individual=not args.no_plot,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc