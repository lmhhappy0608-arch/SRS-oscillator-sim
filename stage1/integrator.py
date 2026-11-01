"""Fixed-step RK4 integration with checkpointed continuation.

The right-hand side and the RK4 weights come from
``code/generate_raw_grid.py``. Continuation resumes at the last completed
full step. A shortened step is used only to land on the requested end time
of the current run, and that shortened state is not the resume point.

Samples that fall inside a step are linearly interpolated, which is the same
capture used by the legacy rotation-number driver. Extending the end time
recomputes samples that lie after the last full step, so those samples match
an uninterrupted integration to the new end time.

RK45 can evaluate a fresh trajectory through
``Ariaratnam2001.integrate_winfree_system``. Its step-size controller is not
checkpointed, so midpoint continuation raises ``ContinuationError``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from stage1.model import (
    RK45_SOLVER,
    frequency_quantiles,
    make_buffers,
    rk4_step_copy,
    sha256_array,
)
from stage1.spec import RunSpec

TIME_TOL = 1e-8


class ContinuationError(RuntimeError):
    """Raised when a trajectory cannot be restored from its checkpoint."""


@dataclass
class TrajectoryState:
    theta_grid: np.ndarray
    t_grid: float
    step_index: int
    theta_final: np.ndarray
    t_final: float
    snapshots: dict[str, np.ndarray] = field(default_factory=dict)
    order_parameter: dict[str, float] = field(default_factory=dict)
    omega: np.ndarray | None = None
    adjacency_sha256: str = ""
    omega_sha256: str = ""
    runtime_seconds: float = 0.0
    resumed_from_time: float = 0.0
    resumed_from_step: int = 0

    def snapshot_at(self, time: float) -> np.ndarray:
        key = float(time).hex()
        if key not in self.snapshots:
            available = sorted(float.fromhex(item) for item in self.snapshots)
            raise ContinuationError(
                f"No saved phase snapshot at t={time}. "
                "Refusing to draw new initial conditions. "
                f"Available snapshot times: {available}"
            )
        return self.snapshots[key]


def grid_time(t_origin: float, step_index: int, dt: float) -> float:
    return float(t_origin) + int(step_index) * float(dt)


def full_step_count(t_origin: float, t_end: float, dt: float) -> int:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    span = float(t_end) - float(t_origin)
    if span < -TIME_TOL:
        raise ValueError("t_end precedes t_origin")
    if span <= TIME_TOL:
        return 0
    ratio = span / float(dt)
    nearest = round(ratio)
    if abs(ratio - nearest) <= 1e-6:
        return int(nearest)
    return int(math.floor(ratio + 1e-12))


def _interp(y0: np.ndarray, y1: np.ndarray, t0: float, t1: float, time: float) -> np.ndarray:
    scale = max(1.0, abs(t0), abs(t1))
    if abs(time - t0) <= TIME_TOL * scale:
        return np.array(y0, dtype=np.float64, copy=True)
    if abs(time - t1) <= TIME_TOL * scale:
        return np.array(y1, dtype=np.float64, copy=True)
    alpha = (float(time) - float(t0)) / (float(t1) - float(t0))
    return np.asarray(y0 + alpha * (y1 - y0), dtype=np.float64)


def _order_parameter(theta: np.ndarray) -> float:
    return float(np.abs(np.mean(np.exp(1j * np.asarray(theta, dtype=np.float64)))))


def _capture(
    y0: np.ndarray,
    y1: np.ndarray,
    t0: float,
    t1: float,
    theta_times: list[float],
    theta_index: int,
    r_times: list[float],
    r_index: int,
    snapshots: dict[str, np.ndarray],
    order_parameter: dict[str, float],
) -> tuple[int, int]:
    while theta_index < len(theta_times) and theta_times[theta_index] <= t1 + TIME_TOL:
        time = theta_times[theta_index]
        if time >= t0 - TIME_TOL:
            snapshots[float(time).hex()] = _interp(y0, y1, t0, t1, time)
        theta_index += 1
    while r_index < len(r_times) and r_times[r_index] <= t1 + TIME_TOL:
        time = r_times[r_index]
        if time >= t0 - TIME_TOL:
            phase = _interp(y0, y1, t0, t1, time)
            order_parameter[float(time).hex()] = _order_parameter(phase)
        r_index += 1
    return theta_index, r_index


def theta_sample_times(spec: RunSpec) -> list[float]:
    times = {float(spec.t_origin), float(spec.t_end)}
    for start, end in spec.observation_windows:
        times.add(float(start))
        times.add(float(end))
    interval = float(spec.checkpoint_interval)
    if interval > 0.0:
        mark = float(spec.t_origin) + interval
        while mark <= float(spec.t_end) + TIME_TOL:
            times.add(float(mark))
            mark += interval
    return sorted(times)


def order_parameter_times(spec: RunSpec) -> list[float]:
    if not spec.sample_order_parameter:
        return []
    if spec.order_parameter_dt is None or spec.order_parameter_dt <= 0.0:
        raise ValueError("order_parameter_dt must be positive when sampling R(t)")
    times = [float(spec.t_origin)]
    mark = float(spec.t_origin) + float(spec.order_parameter_dt)
    while mark < float(spec.t_end) - TIME_TOL:
        times.append(float(mark))
        mark += float(spec.order_parameter_dt)
    times.append(float(spec.t_end))
    return times


def rotation_numbers(state: TrajectoryState, windows: tuple[tuple[float, float], ...]) -> np.ndarray:
    rows = []
    for start, end in windows:
        if float(end) <= float(start):
            raise ValueError("observation window must have positive length")
        theta_start = state.snapshot_at(start)
        theta_end = state.snapshot_at(end)
        rows.append((theta_end - theta_start) / (float(end) - float(start)))
    if not rows:
        raise ValueError("at least one observation window is required")
    stacked = np.vstack(rows)
    return np.asarray(stacked, dtype=np.float64)


def advance_rk4(
    spec: RunSpec,
    theta_grid: np.ndarray,
    step_index: int,
    omega: np.ndarray,
    adjacency: np.ndarray,
    snapshots: dict[str, np.ndarray],
    order_parameter: dict[str, float],
    buffers: dict[str, np.ndarray] | None = None,
    checkpoint_callback=None,
    runtime_already: float = 0.0,
) -> TrajectoryState:
    """Integrate from a saved full-step state through ``spec.t_end``."""
    if spec.solver != "rk4":
        raise ContinuationError(
            f"Solver {spec.solver!r} has no checkpointed continuation. "
            "Refusing to restart the trajectory from new or old initial conditions."
        )
    dt = float(spec.dt)
    t_origin = float(spec.t_origin)
    t_end = float(spec.t_end)
    start_step = int(step_index)
    t_grid = grid_time(t_origin, start_step, dt)
    theta = np.array(theta_grid, dtype=np.float64, copy=True)
    if theta.shape != omega.shape:
        raise ContinuationError(
            f"checkpoint phase length {theta.shape} does not match omega length {omega.shape}"
        )
    snapshots = dict(snapshots)
    order_parameter = dict(order_parameter)
    snapshots.setdefault(float(t_grid).hex(), theta.copy())

    theta_times = theta_sample_times(spec)
    r_times = order_parameter_times(spec)
    missing = [
        time
        for time in theta_times
        if time <= t_grid + TIME_TOL and float(time).hex() not in snapshots
    ]
    if missing:
        raise ContinuationError(
            "Cannot resume: phase snapshots are missing at "
            f"{missing}, which are at or before the checkpoint time {t_grid}. "
            "Refusing to regenerate initial conditions."
        )
    missing_order = [
        time
        for time in r_times
        if time <= t_grid + TIME_TOL and float(time).hex() not in order_parameter
    ]
    if missing_order:
        raise ContinuationError(
            "Cannot resume: order-parameter samples are missing at "
            f"{missing_order}. Refusing to regenerate initial conditions."
        )

    n_full = full_step_count(t_origin, t_end, dt)
    if start_step > n_full:
        raise ContinuationError(
            f"checkpoint step {start_step} is past the {n_full} full steps "
            f"required to reach t_end={t_end}"
        )
    if buffers is None:
        buffers = make_buffers(int(spec.N))

    theta_index = 0
    while theta_index < len(theta_times) and theta_times[theta_index] <= t_grid + TIME_TOL:
        theta_index += 1
    r_index = 0
    while r_index < len(r_times) and r_times[r_index] <= t_grid + TIME_TOL:
        key = float(r_times[r_index]).hex()
        if key not in order_parameter and abs(r_times[r_index] - t_grid) <= TIME_TOL:
            order_parameter[key] = _order_parameter(theta)
        r_index += 1

    next_checkpoint = t_grid + float(spec.checkpoint_interval)
    resumed_from_time = t_grid
    resumed_from_step = start_step

    for step in range(start_step, n_full):
        t0 = grid_time(t_origin, step, dt)
        t1 = grid_time(t_origin, step + 1, dt)
        y1 = rk4_step_copy(theta, dt, omega, adjacency, spec.K, buffers)
        theta_index, r_index = _capture(
            theta,
            y1,
            t0,
            t1,
            theta_times,
            theta_index,
            r_times,
            r_index,
            snapshots,
            order_parameter,
        )
        theta = y1
        if (
            checkpoint_callback is not None
            and spec.checkpoint_interval > 0.0
            and t1 + TIME_TOL >= next_checkpoint
        ):
            checkpoint_callback(
                theta=theta.copy(),
                t_grid=t1,
                step_index=step + 1,
                snapshots=dict(snapshots),
                order_parameter=dict(order_parameter),
                omega=omega,
                partial=False,
            )
            next_checkpoint = t1 + float(spec.checkpoint_interval)

    t_after = grid_time(t_origin, n_full, dt)
    theta_final = theta.copy()
    t_final = t_after
    remainder = float(t_end) - t_after
    if remainder > TIME_TOL:
        y1 = rk4_step_copy(theta, remainder, omega, adjacency, spec.K, buffers)
        theta_index, r_index = _capture(
            theta,
            y1,
            t_after,
            float(t_end),
            theta_times,
            theta_index,
            r_times,
            r_index,
            snapshots,
            order_parameter,
        )
        theta_final = y1
        t_final = float(t_end)
    else:
        snapshots[float(t_end).hex()] = theta_final.copy()

    uncovered = [
        time for time in theta_times if float(time).hex() not in snapshots and time <= t_end + TIME_TOL
    ]
    if uncovered:
        raise ContinuationError(f"integration finished without snapshots at {uncovered}")

    return TrajectoryState(
        theta_grid=theta.copy(),
        t_grid=t_after,
        step_index=n_full,
        theta_final=np.array(theta_final, dtype=np.float64, copy=True),
        t_final=float(t_final),
        snapshots=snapshots,
        order_parameter=order_parameter,
        omega=np.array(omega, dtype=np.float64, copy=True),
        adjacency_sha256=sha256_array(adjacency),
        omega_sha256=sha256_array(omega),
        runtime_seconds=float(runtime_already),
        resumed_from_time=float(resumed_from_time),
        resumed_from_step=int(resumed_from_step),
    )


def integrate_rk45(spec: RunSpec, theta0: np.ndarray, omega: np.ndarray, adjacency: np.ndarray) -> TrajectoryState:
    """Fresh RK45 evaluation. Midpoint continuation is intentionally absent."""
    if spec.solver != RK45_SOLVER:
        raise ValueError(f"integrate_rk45 received solver {spec.solver!r}")
    from Ariaratnam2001 import integrate_winfree_system

    sample_times = np.asarray(theta_sample_times(spec), dtype=float)
    if sample_times.size == 0 or float(sample_times[0]) < float(spec.t_origin) - TIME_TOL:
        raise ContinuationError("RK45 sample times must start at or after t_origin")
    phases = integrate_winfree_system(
        float(spec.K),
        np.asarray(omega, dtype=float),
        np.asarray(adjacency, dtype=float),
        np.asarray(theta0, dtype=float),
        sample_times,
        start_time=float(spec.t_origin),
    )
    snapshots = {
        float(sample_times[index]).hex(): np.array(phases[:, index], dtype=np.float64, copy=True)
        for index in range(sample_times.size)
    }
    order_parameter = {}
    if spec.sample_order_parameter:
        for time in order_parameter_times(spec):
            phase = snapshots.get(float(time).hex())
            if phase is None:
                raise ContinuationError(
                    "RK45 order-parameter time is not a stored sample. "
                    "Refusing to interpolate an adaptive step without its controller state."
                )
            order_parameter[float(time).hex()] = _order_parameter(phase)
    theta_final = snapshots[float(spec.t_end).hex()]
    return TrajectoryState(
        theta_grid=theta_final.copy(),
        t_grid=float(spec.t_end),
        step_index=-1,
        theta_final=theta_final.copy(),
        t_final=float(spec.t_end),
        snapshots=snapshots,
        order_parameter=order_parameter,
        omega=np.array(omega, dtype=np.float64, copy=True),
        adjacency_sha256=sha256_array(adjacency),
        omega_sha256=sha256_array(omega),
        resumed_from_time=float(spec.t_origin),
        resumed_from_step=0,
    )


def quantile_check(omega: np.ndarray, delta: float) -> None:
    """Guard against saving a frequency array that was sorted on its own."""
    expected = 1.0 + float(delta) * frequency_quantiles(omega.size)
    if not np.array_equal(omega, expected):
        # linspace and the algebraic form can differ by a rounding ulp.
        if not np.allclose(omega, expected, rtol=0.0, atol=1e-12):
            raise RuntimeError("omega does not match the deterministic quantile rule")
