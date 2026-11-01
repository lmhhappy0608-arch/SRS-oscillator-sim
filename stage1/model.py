"""Winfree–ER model conventions reused from the validated solvers.

Production ODE, shared by ``code/Ariaratnam2001.py``, ``code/generate_raw_grid.py``,
and ``cpp/main.cpp``:

    dθ/dt = ω - (K/N) * sin(θ) ⊙ (A @ (1 + cos(θ)))

``A`` is an undirected ER adjacency matrix with a zero diagonal. Natural
frequencies are the deterministic grid ``linspace(1 - Δ, 1 + Δ, N)``. At
``p = 1`` this is the complete graph without self-loops, still scaled by
``K/N``. It is not rescaled by ``N - 1`` and it does not add a self-pulse.

Known differences that this package does not reconcile:

* ``code/Winfree_phase_3d.py`` draws an asymmetric adjacency for ``p < 1``.
  At ``p = 1`` that matrix coincides with the undirected complete graph.
* The C++ driver consumes its RNG as network-then-phases and hashes seeds
  differently from the historical Python ``hash(...)`` helper. The ODE is
  the same. Stage 1 uses separate NumPy streams and the SHA-256 recipe in
  ``stage1.seeds`` so a paired scan can share one stream across parameters.
* Random natural frequencies are not part of the validated model. This
  package records ``frequency_seed = null`` and refuses a random-frequency
  request instead of inventing a sampler.

Stage 1 calls ``generate_raw_grid.winfree_rhs`` and ``rk4_step`` for the
default solver. It does not modify those functions.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from stage1.grid import quantize
from stage1.seeds import FREQUENCY_PAIRING_RULE, FREQUENCY_RULE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import generate_raw_grid as legacy_rk4  # noqa: E402

MODEL_VERSION = "winfree_er_v1"
COUPLING_NORMALIZATION = "K_over_N"
PULSE_FUNCTION = "1_plus_cos"
PHASE_RESPONSE = "minus_sin"
ADJACENCY_CONVENTION = "undirected_er_no_self_loops"
RK4_SOLVER = "rk4"
RK45_SOLVER = "rk45"
RK4_SOURCE = "code/generate_raw_grid.py:rk4_step"
RK45_SOURCE = "code/Ariaratnam2001.py:integrate_winfree_system"
RK45_RTOL = 1e-6
RK45_ATOL = 1e-8

P1_DEFINITION = (
    "p=1 is the undirected complete graph with a zero diagonal. "
    "The coupling factor remains K/N. There is no self-pulse and no K/(N-1) rescaling."
)


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def natural_frequencies(delta: float, n: int) -> np.ndarray:
    """Copy of the legacy deterministic frequency grid.

    Oscillator ``i`` owns ``omega[i]``. This function does not sort.
    """
    if n < 1:
        raise ValueError(f"N must be positive, got {n}")
    omega = np.array(legacy_rk4.natural_frequencies(float(delta), int(n)), dtype=np.float64, copy=True)
    if omega.shape != (n,):
        raise RuntimeError("natural frequency array has an unexpected shape")
    return omega


def frequency_quantiles(n: int) -> np.ndarray:
    """Shared quantile coordinate ``q_i`` with ``omega_i = 1 + Δ q_i``."""
    if n == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(-1.0, 1.0, int(n))


def make_adjacency(n: int, p: float, network_seed: int) -> np.ndarray:
    """Undirected ER graph from one dedicated random stream.

    The stream is restarted from ``network_seed`` on every call, then the
    legacy upper-triangle sampler thresholds it at ``p``. Equal seeds and a
    larger ``p`` therefore keep every edge of the sparser graph.
    """
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}")
    rng = np.random.default_rng(int(network_seed))
    adjacency = np.array(
        legacy_rk4.make_er_adjacency(int(n), float(p), rng),
        dtype=np.float64,
        copy=True,
    )
    if adjacency.shape != (n, n):
        raise RuntimeError("adjacency has an unexpected shape")
    return adjacency


def make_initial_phases(n: int, phase_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(phase_seed))
    return np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=int(n)), dtype=np.float64)


def resolve_dt(omega: np.ndarray, dt: float | None) -> float:
    if dt is None:
        resolved = float(legacy_rk4.default_rk4_dt(np.asarray(omega, dtype=np.float64)))
    else:
        resolved = float(dt)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    return resolved


def make_buffers(n: int) -> dict[str, np.ndarray]:
    return legacy_rk4._make_ode_buffers(int(n))


def rk4_step_copy(
    theta: np.ndarray,
    h: float,
    omega: np.ndarray,
    adjacency: np.ndarray,
    K: float,
    buffers: dict[str, np.ndarray],
) -> np.ndarray:
    """One legacy RK4 step, copied out of the reused buffer."""
    y_next = legacy_rk4.rk4_step(
        theta,
        float(h),
        omega,
        adjacency,
        float(K),
        int(omega.shape[0]),
        buffers,
    )
    return np.array(y_next, dtype=np.float64, copy=True)


def parameter_units(K: float, p: float, delta: float) -> tuple[int, int, int, float, float, float]:
    k_units = quantize(K)
    p_units = quantize(p)
    delta_units = quantize(delta)
    from stage1.grid import from_units

    return (
        k_units,
        p_units,
        delta_units,
        from_units(k_units),
        from_units(p_units),
        from_units(delta_units),
    )


def assert_supported_frequency_mode(mode: str) -> None:
    if mode != "deterministic_linspace":
        raise ValueError(
            "Random natural frequencies are not implemented. "
            "The validated Winfree–ER model uses a deterministic linspace "
            f"({FREQUENCY_RULE}). Refusing to invent a sampler."
        )


__all__ = [
    "ADJACENCY_CONVENTION",
    "COUPLING_NORMALIZATION",
    "FREQUENCY_PAIRING_RULE",
    "FREQUENCY_RULE",
    "MODEL_VERSION",
    "P1_DEFINITION",
    "PHASE_RESPONSE",
    "PULSE_FUNCTION",
    "RK4_SOLVER",
    "RK45_ATOL",
    "RK45_RTOL",
    "RK45_SOLVER",
    "RK45_SOURCE",
    "RK4_SOURCE",
]
