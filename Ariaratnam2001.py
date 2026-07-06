import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import Counter
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

OUTPUT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = OUTPUT_DIR / "raw_trials"

# Parameters
N = 10000
T = 10000
T0 = 8000
num_trials = 3

# Only store T0 and T (enough for rotation numbers; much faster than full t_eval)
t_eval = np.array([T0, T], dtype=float)

K_values = np.arange(0.00, 2.00 + 0.001, 0.5)
delta_values = np.arange(0.00, 1.00 + 0.001, 0.5)

# Incremental runs: start with P=0.8, then extend P_VALUES when ready.
# Full range example: np.arange(0.0, 1.0 + 1e-9, 0.1)
P_VALUES = [0.8]


def natural_frequencies(delta, n=N):
    if n == 1:
        return np.array([1.0])
    return np.linspace(1.0 - delta, 1.0 + delta, n)


def make_er_adjacency(n, p, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    A = (rng.random((n, n)) < p).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def simulate_winfree(K, delta, p, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    omega = natural_frequencies(delta, N)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, N)
    A = make_er_adjacency(N, p, rng=rng)

    def winfree(_t, theta):
        interaction = A @ (1.0 + np.cos(theta))
        return omega - (K / N) * np.sin(theta) * interaction

    sol = solve_ivp(
        winfree,
        [0.0, T],
        theta0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-6,
        atol=1e-8,
    )

    theta_t0 = sol.y[:, 0]
    theta_t = sol.y[:, 1]
    rotation_number = (theta_t - theta_t0) / (T - T0)

    # Order parameter at final time
    z = np.mean(np.exp(1j * theta_t))
    R = np.abs(z)

    return omega, rotation_number, R


def save_trial_raw_data(K, delta, p, trial, omega, rotation_number, state):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / (
        f"K_{K:.2f}_delta_{delta:.2f}_p_{p:.2f}_trial_{trial:02d}.npz"
    )
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
        state=state,
    )
    return path


def find_rotation_clusters(rotation_number, tol=1e-2, min_fraction=0.05):
    rho = np.asarray(rotation_number, dtype=float)
    rho_sorted = np.sort(rho)
    n = len(rho_sorted)
    min_size = max(5, int(min_fraction * n))

    clusters = []
    start = 0

    while start < n:
        end = start
        while end + 1 < n and rho_sorted[end + 1] - rho_sorted[start] <= tol:
            end += 1

        size = end - start + 1
        if size >= min_size:
            values = rho_sorted[start : end + 1]
            clusters.append(
                {
                    "center": float(np.mean(values)),
                    "size": int(size),
                    "fraction": float(size / n),
                    "min": float(values[0]),
                    "max": float(values[-1]),
                }
            )

        start = end + 1

    return clusters


def classify_state(
    rotation_number,
    tol_cluster=1e-2,
    tol_death=1e-2,
    min_fraction=0.05,
    lock_threshold=0.95,
):
    rho = np.asarray(rotation_number, dtype=float)
    n = len(rho)

    zero_mask = np.abs(rho) < tol_death
    zero_fraction = np.sum(zero_mask) / n

    # Death: essentially all oscillators have rho ≈ 0
    if zero_fraction >= lock_threshold:
        return "Death"

    clusters = find_rotation_clusters(
        rho,
        tol=tol_cluster,
        min_fraction=min_fraction,
    )
    zero_clusters = [c for c in clusters if abs(c["center"]) < tol_death]
    nonzero_clusters = [c for c in clusters if abs(c["center"]) >= tol_death]

    # Partial Death: a significant zero-frequency cluster coexists with movers
    if zero_clusters and zero_fraction >= min_fraction:
        return "Partial Death"

    largest_nonzero_fraction = (
        max(c["fraction"] for c in nonzero_clusters) if nonzero_clusters else 0.0
    )

    # Locking: almost all oscillators share one nonzero rotation number
    if nonzero_clusters and largest_nonzero_fraction >= lock_threshold:
        return "Locking"

    # Partial Locking: a significant frequency-locked plateau, but not global
    if nonzero_clusters:
        return "Partial Locking"

    # Incoherence: no clear plateaus in the rotation-number distribution
    return "Incoherence"


# Map state names to integer codes for the phase diagram
state_to_code = {
    "Locking": 0,
    "Partial Locking": 1,
    "Incoherence": 2,
    "Partial Death": 3,
    "Death": 4,
}

code_to_state = {value: key for key, value in state_to_code.items()}

state_colors = [
    "#2166ac",  # Locking
    "#fdae61",  # Partial Locking
    "#1a9850",  # Incoherence
    "#d73027",  # Partial Death
    "#762a83",  # Death
]


def run_parameter_sweep(p_values, plot=True):
    """
    Run simulations incrementally over p values.

    For each (p, delta, K), runs num_trials simulations, saves raw trial data,
    and records majority-vote phase classification.
    """
    all_phase_codes = {}

    for p in p_values:
        phase_codes = np.zeros((len(delta_values), len(K_values)), dtype=int)
        phase_results = []
        total = len(delta_values) * len(K_values)
        count = 0

        print(f"\n=== Running p = {p:.2f} ===")

        for i_delta, delta in enumerate(delta_values):
            for i_K, K in enumerate(K_values):
                count += 1
                trial_states = []

                for trial in range(num_trials):
                    rng = np.random.default_rng()
                    omega, rho, _R = simulate_winfree(K, delta, p, rng=rng)
                    state = classify_state(rho)
                    trial_states.append(state)
                    save_trial_raw_data(K, delta, p, trial, omega, rho, state)

                majority_state = Counter(trial_states).most_common(1)[0][0]
                phase_codes[i_delta, i_K] = state_to_code[majority_state]
                phase_results.append((K, delta, majority_state, trial_states))

                print(
                    f"[{count}/{total}] p={p:.2f}, K={K:.2f}, delta={delta:.2f}: "
                    f"majority={majority_state}, trials={trial_states}"
                )

        codes_path = OUTPUT_DIR / f"winfree_phase_codes_p_{p:.2f}.npy"
        np.save(codes_path, phase_codes)
        print(f"Saved {codes_path}")
        all_phase_codes[p] = phase_codes

        if plot:
            plot_phase_diagram(phase_codes, p)

    return all_phase_codes


def plot_phase_diagram(phase_codes, p):
    plt.figure(figsize=(9, 7))

    cmap = ListedColormap(state_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(state_colors) + 0.5, 1), cmap.N)

    plt.imshow(
        phase_codes,
        origin="lower",
        aspect="auto",
        extent=[
            K_values[0],
            K_values[-1],
            delta_values[0],
            delta_values[-1],
        ],
        cmap=cmap,
        norm=norm,
    )

    plt.xlabel("Coupling strength K")
    plt.ylabel("Frequency width δ")
    plt.title(
        f"Phase diagram for the Winfree model (Ariaratnam & Strogatz 2001)\n"
        f"N={N}, p={p}, T={T}, T₀={T0}, trials={num_trials}"
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
    figure_path = OUTPUT_DIR / f"winfree_phase_diagram_p_{p:.2f}.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"Saved {figure_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ariaratnam & Strogatz 2001 Winfree model phase diagram"
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=None,
        help="ER connection probabilities to run (default: P_VALUES in script)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip phase diagram plotting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    p_values = args.p_values if args.p_values is not None else P_VALUES
    run_parameter_sweep(p_values, plot=not args.no_plot)
