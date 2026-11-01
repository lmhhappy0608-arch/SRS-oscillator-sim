"""Scan configuration loaded from JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from stage1.grid import canonicalize_values, inclusive_range
from stage1.model import (
    PROJECT_ROOT,
    RK45_ATOL,
    RK45_RTOL,
    RK45_SOLVER,
    RK4_SOLVER,
)

DEFAULT_WINDOWS = (
    (8000.0, 10000.0),
    (10000.0, 12000.0),
    (12000.0, 14000.0),
)
RANDOMNESS_MODES = ("paired", "network", "initial_condition", "independent")


@dataclass
class ScanConfig:
    N: int = 1000
    k_min: float = 0.0
    k_max: float = 3.0
    k_step: float = 0.01
    k_values: tuple[float, ...] | None = None
    p_values: tuple[float, ...] = (
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.3,
        0.2,
        0.1,
        0.05,
    )
    delta_min: float = 0.05
    delta_max: float = 1.0
    delta_step: float = 0.05
    delta_values: tuple[float, ...] | None = None
    n_trials: int = 3
    randomness_mode: str = "paired"
    master_seed: int = 0
    fixed_network_realization_id: int = 0
    fixed_initial_condition_id: int = 0
    solver: str = RK4_SOLVER
    dt: float | None = None
    rtol: float | None = None
    atol: float | None = None
    t_start: float = 0.0
    t_end: float = 14000.0
    observation_windows: tuple[tuple[float, float], ...] = DEFAULT_WINDOWS
    sample_order_parameter: bool = False
    order_parameter_dt: float | None = None
    checkpoint_interval: float = 2000.0
    output_dir: Path = PROJECT_ROOT / "stage1_runs"
    workers: int = 1
    stale_seconds: float = 3600.0
    allow_incomplete_restart: bool = False
    frequency_mode: str = "deterministic_linspace"

    def resolved_k(self) -> tuple[float, ...]:
        if self.k_values is not None:
            return canonicalize_values(self.k_values)
        return inclusive_range(self.k_min, self.k_max, self.k_step)

    def resolved_p(self) -> tuple[float, ...]:
        return canonicalize_values(self.p_values)

    def resolved_delta(self) -> tuple[float, ...]:
        if self.delta_values is not None:
            return canonicalize_values(self.delta_values)
        return inclusive_range(self.delta_min, self.delta_max, self.delta_step)


def _windows_from_json(raw: list) -> tuple[tuple[float, float], ...]:
    windows = []
    for item in raw:
        if len(item) != 2:
            raise ValueError(f"observation window must be [start, end], got {item}")
        windows.append((float(item[0]), float(item[1])))
    return tuple(windows)


def _optional_float_tuple(raw) -> tuple[float, ...] | None:
    if raw is None:
        return None
    return tuple(float(value) for value in raw)


def load_config(path: Path) -> ScanConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a JSON object")
    frequency = payload.get("frequency", {})
    frequency_mode = frequency.get("mode", "deterministic_linspace") if isinstance(frequency, dict) else frequency
    k_block = payload.get("K", {})
    delta_block = payload.get("Delta", {})
    config = ScanConfig(
        N=int(payload.get("N", 1000)),
        k_min=float(k_block.get("min", 0.0)),
        k_max=float(k_block.get("max", 3.0)),
        k_step=float(k_block.get("step", 0.01)),
        k_values=_optional_float_tuple(k_block.get("values")),
        p_values=tuple(float(value) for value in payload.get("p", ScanConfig.p_values)),
        delta_min=float(delta_block.get("min", 0.05)),
        delta_max=float(delta_block.get("max", 1.0)),
        delta_step=float(delta_block.get("step", 0.05)),
        delta_values=_optional_float_tuple(delta_block.get("values")),
        n_trials=int(payload.get("n_trials", 3)),
        randomness_mode=str(payload.get("randomness_mode", "paired")),
        master_seed=int(payload.get("master_seed", 0)),
        fixed_network_realization_id=int(payload.get("fixed_network_realization_id", 0)),
        fixed_initial_condition_id=int(payload.get("fixed_initial_condition_id", 0)),
        solver=str(payload.get("solver", RK4_SOLVER)),
        dt=None if payload.get("dt", None) is None else float(payload["dt"]),
        t_start=float(payload.get("t_start", 0.0)),
        t_end=float(payload.get("t_end", 14000.0)),
        observation_windows=_windows_from_json(payload.get("observation_windows", DEFAULT_WINDOWS)),
        sample_order_parameter=bool(payload.get("sample_order_parameter", False)),
        order_parameter_dt=(
            None
            if payload.get("order_parameter_dt", None) is None
            else float(payload["order_parameter_dt"])
        ),
        checkpoint_interval=float(payload.get("checkpoint_interval", 2000.0)),
        output_dir=Path(payload.get("output_dir", "stage1_runs")),
        workers=int(payload.get("workers", 1)),
        stale_seconds=float(payload.get("stale_seconds", 3600.0)),
        allow_incomplete_restart=bool(payload.get("allow_incomplete_restart", False)),
        frequency_mode=str(frequency_mode),
    )
    if not config.output_dir.is_absolute():
        config = replace(config, output_dir=PROJECT_ROOT / config.output_dir)
    return validate_config(config)


def validate_config(config: ScanConfig) -> ScanConfig:
    from stage1.model import assert_supported_frequency_mode

    assert_supported_frequency_mode(config.frequency_mode)
    if config.N < 1:
        raise ValueError("N must be positive")
    if config.n_trials < 1:
        raise ValueError("n_trials must be positive")
    if config.workers < 1:
        raise ValueError("workers must be positive")
    if config.randomness_mode not in RANDOMNESS_MODES:
        raise ValueError(f"randomness_mode must be one of {RANDOMNESS_MODES}")
    if config.solver not in {RK4_SOLVER, RK45_SOLVER}:
        raise ValueError("solver must be 'rk4' or 'rk45'")
    if config.t_start != 0.0:
        raise ValueError(
            "A fresh scan must start at t=0. Extend an existing trajectory with "
            "'python -m stage1 continue' instead of changing t_start."
        )
    if config.t_end <= config.t_start:
        raise ValueError("t_end must be greater than t_start")
    if config.checkpoint_interval < 0.0:
        raise ValueError("checkpoint_interval must be non-negative")
    if config.dt is not None and config.dt <= 0.0:
        raise ValueError("dt must be positive")
    if config.sample_order_parameter and (
        config.order_parameter_dt is None or config.order_parameter_dt <= 0.0
    ):
        raise ValueError("order_parameter_dt must be positive when sample_order_parameter is true")
    windows = list(config.observation_windows)
    if not windows:
        raise ValueError("at least one observation window is required")
    normalized = []
    seen = set()
    for start, end in windows:
        if end <= start:
            raise ValueError(f"observation window {(start, end)} must have positive length")
        if start < config.t_start or end > config.t_end + 1e-12:
            raise ValueError(
                f"observation window {(start, end)} is outside [{config.t_start}, {config.t_end}]"
            )
        key = (float(start).hex(), float(end).hex())
        if key in seen:
            raise ValueError(f"duplicate observation window {(start, end)}")
        seen.add(key)
        normalized.append((float(start), float(end)))
    normalized.sort(key=lambda item: (item[0], item[1]))
    k_values = config.resolved_k()
    p_values = config.resolved_p()
    delta_values = config.resolved_delta()
    for value in p_values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p must lie in [0, 1], got {value}")
    for value in delta_values:
        if value < 0.0:
            raise ValueError(f"Delta must be non-negative, got {value}")
    if config.solver == RK45_SOLVER:
        rtol = RK45_RTOL if config.rtol is None else config.rtol
        atol = RK45_ATOL if config.atol is None else config.atol
    else:
        rtol = config.rtol
        atol = config.atol
    if not k_values or not p_values or not delta_values:
        raise ValueError("parameter grid is empty")
    return replace(
        config,
        observation_windows=tuple(normalized),
        p_values=p_values,
        k_values=k_values if config.k_values is not None else None,
        delta_values=delta_values if config.delta_values is not None else None,
        rtol=rtol,
        atol=atol,
    )
