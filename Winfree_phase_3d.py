"""
3D phase diagram for the Winfree model on Erdos-Renyi (ER) networks.

Axes:
  x = K        (coupling strength)
  y = delta    (frequency spread)
  z = p        (ER connection probability, 0 -> 1)

State classification uses rotation-number clusters (same logic as the 2D paper case).
Each (K, delta, p) point is simulated num_trials times; a new ER graph and random
initial phases are drawn for every trial.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import Counter
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 1000
T = 10000
T0 = 8000
num_trials = 10

# Only save phases at T0 and T (enough for rotation numbers; much faster)
t_eval = np.array([T0, T], dtype=float)

K_values = np.arange(0.0, 2.0 + 1e-9, 0.2)
delta_values = np.arange(0.0, 1.0 + 1e-9, 0.1)
p_values = np.arange(0.0, 1.0 + 1e-9, 0.2)

# ---------------------------------------------------------------------------
# State maps
# ---------------------------------------------------------------------------
STATE_TO_CODE = {
    "Locking": 0,
    "Partial Locking": 1,
    "Incoherence": 2,
    "Partial Death": 3,
    "Death": 4,
}
CODE_TO_STATE = {v: k for k, v in STATE_TO_CODE.items()}

STATE_COLORS = [
    "#2166ac",  # Locking
    "#fdae61",  # Partial Locking
    "#1a9850",  # Incoherence
    "#d73027",  # Partial Death
    "#762a83",  # Death
]


def make_er_adjacency(n, p, rng=None):
    """ER network: A_ij = 1 with probability p, no self-loops."""
    rng = np.random.default_rng() if rng is None else rng
    A = (rng.random((n, n)) < p).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


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
    """
    Classify from rotation-number structure (Ariaratnam & Strogatz 2001 logic).
    Priority: Death -> Partial Death -> Locking -> Partial Locking -> Incoherence
    """
    rho = np.asarray(rotation_number, dtype=float)
    n = len(rho)

    zero_fraction = np.mean(np.abs(rho) < tol_death)
    if zero_fraction >= lock_threshold:
        return "Death"

    clusters = find_rotation_clusters(rho, tol=tol_cluster, min_fraction=min_fraction)
    zero_clusters = [c for c in clusters if abs(c["center"]) < tol_death]
    nonzero_clusters = [c for c in clusters if abs(c["center"]) >= tol_death]

    if zero_clusters and zero_fraction >= min_fraction:
        return "Partial Death"

    largest_nonzero = (
        max(c["fraction"] for c in nonzero_clusters) if nonzero_clusters else 0.0
    )
    if nonzero_clusters and largest_nonzero >= lock_threshold:
        return "Locking"
    if nonzero_clusters:
        return "Partial Locking"
    return "Incoherence"


def simulate_winfree(K, delta, p, rng=None):
    """
    One Winfree simulation on an ER network.

    Returns rotation numbers rho_i. Only T0 and T are stored.
    """
    rng = np.random.default_rng() if rng is None else rng
    omega = np.linspace(1.0 - delta, 1.0 + delta, N)
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
    return rotation_number


def run_parameter_sweep():
    """
    Scan (K, delta, p) and store majority-vote state codes.

    phase_codes[i_p, i_delta, i_K] -> integer state code
    """
    phase_codes = np.zeros(
        (len(p_values), len(delta_values), len(K_values)), dtype=int
    )

    total = len(p_values) * len(delta_values) * len(K_values)
    count = 0

    for i_p, p in enumerate(p_values):
        for i_delta, delta in enumerate(delta_values):
            for i_K, K in enumerate(K_values):
                count += 1
                trial_states = []
                for _ in range(num_trials):
                    rho = simulate_winfree(K, delta, p)
                    trial_states.append(classify_state(rho))

                majority = Counter(trial_states).most_common(1)[0][0]
                phase_codes[i_p, i_delta, i_K] = STATE_TO_CODE[majority]

                print(
                    f"[{count}/{total}] p={p:.2f}, delta={delta:.2f}, K={K:.2f} "
                    f"-> {majority} | trials={trial_states}"
                )

    return phase_codes


def plot_2d_slices(phase_codes):
    """One imshow panel per p (easy to read)."""
    n_p = len(p_values)
    n_cols = min(3, n_p)
    n_rows = int(np.ceil(n_p / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(STATE_COLORS) + 0.5, 1), cmap.N)

    for i_p, p in enumerate(p_values):
        ax = axes[i_p]
        ax.imshow(
            phase_codes[i_p],
            origin="lower",
            aspect="auto",
            extent=[K_values[0], K_values[-1], delta_values[0], delta_values[-1]],
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(f"p = {p:.2f}")
        ax.set_xlabel("K")
        ax.set_ylabel("δ")

    for j in range(i_p + 1, len(axes)):
        axes[j].axis("off")

    legend_handles = [
        Patch(color=STATE_COLORS[c], label=CODE_TO_STATE[c])
        for c in range(len(STATE_COLORS))
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Winfree ER phase slices (N={N}, T={T}, T₀={T0}, trials={num_trials})",
        y=1.06,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "winfree_phase_3d_slices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    return fig


def plot_3d_scatter(phase_codes):
    """
    3D scatter: each grid point (K, delta, p) is one colored marker.
    Good overview of how states vary with connection probability.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    kk, dd, pp = np.meshgrid(K_values, delta_values, p_values, indexing="ij")
    # phase_codes is [p, delta, K], reorder to match meshgrid (K, delta, p)
    codes = np.transpose(phase_codes, (2, 1, 0))

    x = kk.ravel()
    y = dd.ravel()
    z = pp.ravel()
    c = codes.ravel()

    scatter = ax.scatter(
        x,
        y,
        z,
        c=c,
        cmap=ListedColormap(STATE_COLORS),
        norm=BoundaryNorm(np.arange(-0.5, len(STATE_COLORS) + 0.5, 1), len(STATE_COLORS)),
        s=35,
        depthshade=True,
        edgecolors="none",
    )

    ax.set_xlabel("K")
    ax.set_ylabel("δ")
    ax.set_zlabel("p")
    ax.set_title("3D phase diagram on ER networks")

    legend_handles = [
        Patch(color=STATE_COLORS[c], label=CODE_TO_STATE[c])
        for c in range(len(STATE_COLORS))
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out = OUTPUT_DIR / "winfree_phase_3d_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    return fig


def main():
    phase_codes = run_parameter_sweep()

    np.save(OUTPUT_DIR / "winfree_phase_codes_3d.npy", phase_codes)
    np.savez(
        OUTPUT_DIR / "winfree_phase_axes_3d.npz",
        K_values=K_values,
        delta_values=delta_values,
        p_values=p_values,
        state_to_code=STATE_TO_CODE,
    )
    print(f"Saved {OUTPUT_DIR / 'winfree_phase_codes_3d.npy'}")

    plot_2d_slices(phase_codes)
    plot_3d_scatter(phase_codes)
    plt.show()


if __name__ == "__main__":
    main()
