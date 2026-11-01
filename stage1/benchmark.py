"""Runtime benchmark separate from the production parameter scan.

Measured timings come only from ``measure_representative_points``. Grid totals
computed from those timings are labeled extrapolated. With no measurement file,
runtime and compressed disk totals are reported as measurement_required.
A structural uncompressed byte count is a layout estimate, not a measurement.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from stage1.config import DEFAULT_WINDOWS, ScanConfig, validate_config
from stage1.grid import inclusive_range
from stage1.integrator import advance_rk4, theta_sample_times
from stage1.model import PROJECT_ROOT, make_adjacency, make_buffers, make_initial_phases, natural_frequencies, sha256_array
from stage1.scan import build_run_specs
from stage1.storage import write_completed_run
from stage1.integrator import rotation_numbers

PRESET_NAMES = ("moderate", "dense", "dense_extra_trials")

REPRESENTATIVE_POINTS = (
    {"K": 0.5, "p": 0.05, "Delta": 0.2, "label": "sparse_network"},
    {"K": 0.5, "p": 0.5, "Delta": 0.5, "label": "intermediate_network"},
    {"K": 0.5, "p": 1.0, "Delta": 1.0, "label": "complete_graph"},
    {"K": 0.1, "p": 1.0, "Delta": 0.2, "label": "weak_coupling_complete"},
    {"K": 2.0, "p": 0.05, "Delta": 0.8, "label": "strong_coupling_sparse"},
    {"K": 1.5, "p": 0.8, "Delta": 0.5, "label": "intermediate_coupling"},
)


def preset_config(name: str) -> ScanConfig:
    """Return one of the three planning grids. This does not integrate."""
    if name == "moderate":
        config = ScanConfig(
            k_min=0.0,
            k_max=3.0,
            k_step=0.1,
            p_values=(1.0, 0.8, 0.5, 0.2, 0.05),
            delta_values=(0.05, 0.25, 0.5, 0.75, 1.0),
            n_trials=1,
            t_end=14000.0,
            observation_windows=DEFAULT_WINDOWS,
            checkpoint_interval=2000.0,
            workers=1,
        )
    elif name == "dense":
        config = ScanConfig(
            k_min=0.0,
            k_max=3.0,
            k_step=0.01,
            delta_min=0.05,
            delta_max=1.0,
            delta_step=0.05,
            n_trials=1,
            t_end=14000.0,
            observation_windows=DEFAULT_WINDOWS,
            checkpoint_interval=2000.0,
            workers=1,
        )
    elif name == "dense_extra_trials":
        config = preset_config("dense")
        config.n_trials = 3
    else:
        raise ValueError(f"unknown preset {name!r}")
    return validate_config(config)


def axis_sizes(name: str) -> dict:
    config = preset_config(name)
    k_values = config.resolved_k()
    p_values = config.resolved_p()
    delta_values = config.resolved_delta()
    run_count = len(k_values) * len(p_values) * len(delta_values) * config.n_trials
    return {
        "preset": name,
        "K_count": len(k_values),
        "K_first": k_values[0],
        "K_last": k_values[-1],
        "p_count": len(p_values),
        "Delta_count": len(delta_values),
        "Delta_first": delta_values[0],
        "Delta_last": delta_values[-1],
        "n_trials": config.n_trials,
        "N": config.N,
        "t_end": config.t_end,
        "window_count": len(config.observation_windows),
        "run_count": run_count,
        "label": "count",
    }


def structural_uncompressed_bytes(config: ScanConfig) -> dict:
    """Byte count implied by the saved arrays, before compression.

    This is not a measured file size.
    """
    sample = validate_config(
        ScanConfig(
            N=config.N,
            k_values=(config.resolved_k()[0],),
            p_values=(config.resolved_p()[0],),
            delta_values=(config.resolved_delta()[0],),
            n_trials=1,
            t_end=config.t_end,
            observation_windows=config.observation_windows,
            checkpoint_interval=config.checkpoint_interval,
            sample_order_parameter=config.sample_order_parameter,
            order_parameter_dt=config.order_parameter_dt,
            dt=config.dt,
            solver=config.solver,
            randomness_mode=config.randomness_mode,
            master_seed=config.master_seed,
        )
    )
    representative = build_run_specs(sample)[0]
    snapshot_count = len(theta_sample_times(representative))
    r_count = 0
    if config.sample_order_parameter and config.order_parameter_dt:
        r_count = int(round((config.t_end - config.t_start) / config.order_parameter_dt)) + 1
    n = int(config.N)
    windows = len(config.observation_windows)
    float_values = n + windows * n + n + n + n + snapshot_count * n + r_count + r_count
    byte_count = float_values * 8 + n * 4
    return {
        "label": "structural_estimate_not_measured",
        "uncompressed_bytes_per_run": byte_count,
        "snapshot_count": snapshot_count,
        "window_count": windows,
        "order_parameter_count": r_count,
        "N": n,
    }


def preset_report() -> dict:
    presets = {name: axis_sizes(name) for name in PRESET_NAMES}
    for name, info in presets.items():
        info["disk_structural"] = structural_uncompressed_bytes(preset_config(name))
        info["runtime"] = {
            "status": "measurement_required",
            "label": "not_measured",
            "cpu_seconds": None,
        }
    # The dense axes are the candidate scan. Keep the arithmetic explicit.
    dense_k = len(inclusive_range(0.0, 3.0, 0.01))
    dense_delta = len(inclusive_range(0.05, 1.0, 0.05))
    presets["dense"]["expected_run_count_check"] = dense_k * 11 * dense_delta
    presets["dense_extra_trials"]["expected_run_count_check"] = dense_k * 11 * dense_delta * 3
    return {"kind": "count_only", "presets": presets}


def _fit_runtime(short_time: float, short_seconds: float, long_time: float, long_seconds: float) -> dict:
    if long_time <= short_time:
        return {"valid": False, "reason": "horizons are not increasing"}
    rate = (long_seconds - short_seconds) / (long_time - short_time)
    overhead = short_seconds - rate * short_time
    valid = bool(rate > 0.0 and overhead >= -0.05 * max(short_seconds, long_seconds, 1e-9))
    return {
        "valid": valid,
        "seconds_per_unit_time": float(rate),
        "overhead_seconds": float(overhead),
        "method": "affine_fit" if valid else "rejected",
    }


def measure_representative_points(
    *,
    horizons: tuple[float, float] = (20.0, 40.0),
    N: int = 1000,
    dt: float | None = None,
    output_path: Path | None = None,
    points: tuple[dict, ...] = REPRESENTATIVE_POINTS,
) -> dict:
    """Time a few parameter points. This is not the production grid.

    The reused RHS is a dense matrix-vector product, so ``p`` may change the
    runtime only weakly. Both horizons are still recorded.
    """
    if len(horizons) != 2 or horizons[1] <= horizons[0] or horizons[0] <= 0.0:
        raise ValueError("horizons must be two increasing positive times")
    records = []
    for point in points:
        records.append(
            _measure_one(
                K=float(point["K"]),
                p=float(point["p"]),
                delta=float(point["Delta"]),
                label=str(point["label"]),
                horizons=horizons,
                n=int(N),
                dt=dt,
            )
        )
    document = {
        "kind": "measured",
        "solver": "rk4",
        "solver_source": "code/generate_raw_grid.py:rk4_step",
        "platform": sys.platform,
        "N": int(N),
        "horizons": [float(horizons[0]), float(horizons[1])],
        "points": records,
        "note": (
            "These timings were measured for the listed points and horizons only. "
            "The dense RHS does not skip absent ER edges, so network density may "
            "have little effect on runtime."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _measure_one(*, K: float, p: float, delta: float, label: str, horizons, n: int, dt: float | None) -> dict:
    import resource

    windows = ((horizons[0] * 0.5, horizons[0]),)
    # The longer horizon is measured on its own fresh trajectory so the affine
    # fit sees two complete integrations rather than a continuation.
    short = _time_horizon(K, p, delta, n, dt, horizons[0], windows)
    long_windows = (
        (horizons[0] * 0.5, horizons[0]),
        (horizons[0], horizons[1]),
    )
    long_windows = tuple((start, end) for start, end in long_windows if end <= horizons[1] + 1e-12 and start < end)
    if not long_windows:
        long_windows = ((0.0, horizons[1]),)
    long = _time_horizon(K, p, delta, n, dt, horizons[1], long_windows)
    fit = _fit_runtime(horizons[0], short["integration_seconds"], horizons[1], long["integration_seconds"])
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024
    return {
        "label": label,
        "K": K,
        "p": p,
        "Delta": delta,
        "N": n,
        "dt": long["dt"],
        "horizons": [short, long],
        "fit": fit,
        "peak_rss_bytes": rss,
        "peak_rss_note": "process ru_maxrss, not an isolated allocation of one run",
        "disk_bytes": long["disk_bytes"],
        "disk_note": "measured size of the NPZ plus JSON at the longer horizon",
    }


def _time_horizon(K, p, delta, n, dt, t_end, windows) -> dict:
    config = validate_config(
        ScanConfig(
            N=n,
            k_values=(K,),
            p_values=(p,),
            delta_values=(delta,),
            n_trials=1,
            t_end=float(t_end),
            observation_windows=tuple(windows),
            dt=dt,
            checkpoint_interval=float(t_end),
            output_dir=PROJECT_ROOT / "stage1_benchmark",
            workers=1,
            t_start=0.0,
        )
    )
    spec = build_run_specs(config)[0]
    omega = natural_frequencies(spec.Delta, spec.N)
    theta0 = make_initial_phases(spec.N, spec.initial_phase_seed)
    adjacency = make_adjacency(spec.N, spec.p, spec.network_seed)
    snapshots = {float(spec.t_origin).hex(): theta0.copy()}
    started = time.perf_counter()
    state = advance_rk4(
        spec,
        theta0,
        0,
        omega,
        adjacency,
        snapshots,
        {},
        buffers=make_buffers(spec.N),
    )
    integration_seconds = time.perf_counter() - started
    state.runtime_seconds = integration_seconds
    state.omega = omega
    state.adjacency_sha256 = sha256_array(adjacency)
    state.omega_sha256 = sha256_array(omega)
    rotation = rotation_numbers(state, spec.observation_windows)
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="stage1-benchmark-") as temporary:
        directory = Path(temporary)
        write_completed_run(directory, spec, state, rotation)
        from stage1.storage import run_paths

        paths = run_paths(directory, spec.run_id)
        disk_bytes = paths["npz"].stat().st_size + paths["json"].stat().st_size
    return {
        "t_end": float(t_end),
        "integration_seconds": float(integration_seconds),
        "seconds_per_unit_time": float(integration_seconds / float(t_end)),
        "disk_bytes": int(disk_bytes),
        "dt": float(spec.dt),
        "label": "measured",
    }


def extrapolate(
    preset: str,
    measurements: dict | None,
    *,
    workers: int = 1,
    efficiency: float | None = None,
) -> dict:
    """Turn measured point timings into a labeled grid estimate.

    ``efficiency`` is never inferred. Pass a value only when a worker-scaling
    study measured it. ``workers`` then scales that supplied efficiency; the
    result stays labeled extrapolated.
    """
    counts = axis_sizes(preset)
    structural = structural_uncompressed_bytes(preset_config(preset))
    report = {
        "kind": "extrapolated" if measurements else "count_only",
        "preset": preset,
        "run_count": counts["run_count"],
        "t_end": counts["t_end"],
        "axes": counts,
        "disk_structural": structural,
        "workers": int(workers),
    }
    if not measurements:
        report["runtime"] = {
            "status": "measurement_required",
            "label": "not_measured",
            "cpu_seconds": None,
        }
        report["disk"] = {
            "status": "measurement_required",
            "label": "not_measured",
            "total_bytes": None,
            "structural_uncompressed_bytes_per_run": structural["uncompressed_bytes_per_run"],
        }
        report["wall_time"] = _wall_time(None, workers, efficiency)
        return report
    if measurements.get("kind") != "measured":
        raise ValueError("measurements must be a document produced by measure_representative_points")
    rates = []
    overheads = []
    proportional_rates = []
    disk_sizes = []
    fit_valid = True
    for point in measurements["points"]:
        fit = point["fit"]
        long = point["horizons"][-1]
        proportional_rates.append(float(long["seconds_per_unit_time"]))
        disk_sizes.append(int(point["disk_bytes"]))
        if fit.get("valid"):
            rates.append(float(fit["seconds_per_unit_time"]))
            overheads.append(float(fit["overhead_seconds"]))
        else:
            fit_valid = False
    t_end = float(counts["t_end"])
    run_count = int(counts["run_count"])
    if fit_valid and rates:
        order = np.argsort(rates)
        median_rate = float(rates[int(order[len(order) // 2])])
        median_overhead = float(overheads[int(order[len(order) // 2])])
        max_rate = float(max(rates))
        max_overhead = float(overheads[int(np.argmax(rates))])
        method = "affine_fit"
        cpu_median = run_count * (median_overhead + median_rate * t_end)
        cpu_conservative = run_count * (max_overhead + max_rate * t_end)
    else:
        median_rate = float(np.median(proportional_rates))
        max_rate = float(np.max(proportional_rates))
        method = "proportional_to_longer_horizon"
        cpu_median = run_count * median_rate * t_end
        cpu_conservative = run_count * max_rate * t_end
        median_overhead = None
    disk_median = int(np.median(disk_sizes))
    report["runtime"] = {
        "status": "extrapolated",
        "label": "extrapolated",
        "method": method,
        "cpu_seconds_median": float(cpu_median),
        "cpu_seconds_conservative_max_point": float(cpu_conservative),
        "seconds_per_unit_time_median": median_rate,
        "overhead_seconds_median": median_overhead,
        "assumption": (
            "Each unmeasured run is estimated from the representative-point fit. "
            "This is not a measurement of the full preset. Fixed costs that the "
            "two horizons did not identify remain inside a rejected fit's "
            "proportional rate, which overestimates long runs."
        ),
    }
    report["disk"] = {
        "status": "extrapolated",
        "label": "extrapolated",
        "measured_bytes_per_run_median": disk_median,
        "total_bytes": int(disk_median) * run_count,
        "note": (
            "Median measured file size from the benchmark horizons, multiplied "
            "by the preset run count. File size depends on snapshot count and "
            "compression, so a short benchmark file is only a proxy."
        ),
        "structural_uncompressed_bytes_per_run": structural["uncompressed_bytes_per_run"],
        "structural_label": "structural_estimate_not_measured",
    }
    report["wall_time"] = _wall_time(float(cpu_median), workers, efficiency)
    return report


def _wall_time(cpu_seconds: float | None, workers: int, efficiency: float | None) -> dict:
    if cpu_seconds is None:
        return {
            "status": "measurement_required",
            "label": "not_measured",
            "workers": int(workers),
            "efficiency": None,
            "efficiency_source": "not_measured",
            "wall_seconds": None,
            "note": "No measured CPU time is available, so wall time is not estimated.",
        }
    if efficiency is None:
        return {
            "status": "efficiency_not_measured",
            "label": "not_measured",
            "workers": int(workers),
            "efficiency": None,
            "efficiency_source": "not_measured",
            "wall_seconds": None,
            "single_worker_cpu_seconds": float(cpu_seconds),
            "note": (
                "Refusing to guess parallel efficiency. "
                "Supply a measured efficiency to extrapolate wall time."
            ),
        }
    if efficiency <= 0.0 or efficiency > 1.0:
        raise ValueError("efficiency must lie in (0, 1]")
    if workers < 1:
        raise ValueError("workers must be positive")
    return {
        "status": "extrapolated",
        "label": "extrapolated",
        "workers": int(workers),
        "efficiency": float(efficiency),
        "efficiency_source": "user_supplied",
        "wall_seconds": float(cpu_seconds) / (int(workers) * float(efficiency)),
        "note": "Wall time uses the supplied efficiency. It was not measured here.",
    }


def load_measurements(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
