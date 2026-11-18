"""Plan, run, resume, and extend Stage 1 simulations."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from stage1.config import ScanConfig
from stage1.grid import canonical_label, quantize
from stage1.integrator import (
    TIME_TOL,
    ContinuationError,
    TrajectoryState,
    advance_rk4,
    grid_time,
    integrate_rk45,
    quantile_check,
    rotation_numbers,
    theta_sample_times,
)
from stage1.model import (
    PROJECT_ROOT,
    ADJACENCY_CONVENTION,
    COUPLING_NORMALIZATION,
    FREQUENCY_PAIRING_RULE,
    FREQUENCY_RULE,
    MODEL_VERSION,
    PHASE_RESPONSE,
    PULSE_FUNCTION,
    RK45_SOLVER,
    RK45_SOURCE,
    RK4_SOLVER,
    RK4_SOURCE,
    make_adjacency,
    make_buffers,
    make_initial_phases,
    natural_frequencies,
    parameter_units,
    resolve_dt,
    sha256_array,
)
from stage1.seeds import (
    SEED_RECIPE,
    assign_realizations,
    network_and_phase_seeds,
    paired_group_id,
)
from stage1.spec import RunSpec
from stage1.storage import (
    checkpoint_arrays,
    claim_run,
    connect,
    count_status_for_ids,
    insert_planned,
    load_checkpoint,
    mark_status,
    output_is_complete,
    resolve_stored_path,
    run_paths,
    touch_heartbeat,
    write_checkpoint,
    write_completed_run,
)


def _rectangles(config: ScanConfig) -> tuple[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]], ...]:
    """Base grid first, then each extension. Overlaps are removed later by run_id."""
    blocks = [(config.resolved_k(), config.resolved_p(), config.resolved_delta())]
    for region in config.extensions:
        blocks.append((region.resolved_k(), region.resolved_p(), region.resolved_delta()))
    return tuple(blocks)


def build_run_specs(config: ScanConfig) -> list[RunSpec]:
    """Expand a validated configuration into stable run specifications.

    An extension that repeats a base point keeps the first specification.
    A repeated run_id with different parameters is an error.
    """
    rectangles = _rectangles(config)
    dt_by_delta: dict[int, float] = {}
    for _couplings, _probabilities, deltas in rectangles:
        for delta in deltas:
            key = quantize(delta)
            if key in dt_by_delta:
                continue
            omega = natural_frequencies(delta, config.N)
            quantile_check(omega, delta)
            dt_by_delta[key] = resolve_dt(omega, config.dt if config.solver == RK4_SOLVER else config.dt)
    solver_source = RK4_SOURCE if config.solver == RK4_SOLVER else RK45_SOURCE
    specs: list[RunSpec] = []
    seen: dict[str, RunSpec] = {}
    for k_values, p_values, delta_values in rectangles:
        for p in p_values:
            for delta in delta_values:
                for coupling in k_values:
                    for trial_id in range(config.n_trials):
                        spec = _make_spec(
                            config,
                            K=coupling,
                            p=p,
                            delta=delta,
                            trial_id=trial_id,
                            dt=dt_by_delta[quantize(delta)],
                            solver_source=solver_source,
                        )
                        previous = seen.get(spec.run_id)
                        if previous is not None:
                            same = (
                                previous.k_units == spec.k_units
                                and previous.p_units == spec.p_units
                                and previous.delta_units == spec.delta_units
                                and previous.trial_id == spec.trial_id
                            )
                            if not same:
                                raise RuntimeError(f"duplicate run_id {spec.run_id}")
                            continue
                        seen[spec.run_id] = spec
                        specs.append(spec)
    return specs


def _make_spec(
    config: ScanConfig,
    *,
    K: float,
    p: float,
    delta: float,
    trial_id: int,
    dt: float,
    solver_source: str,
    t_end: float | None = None,
    windows: tuple[tuple[float, float], ...] | None = None,
    parent_run_id: str | None = None,
    t_origin: float = 0.0,
) -> RunSpec:
    k_units, p_units, delta_units, k_value, p_value, delta_value = parameter_units(K, p, delta)
    network_id, phase_id = assign_realizations(
        config.randomness_mode,
        trial_id,
        config.fixed_network_realization_id,
        config.fixed_initial_condition_id,
    )
    k_label = canonical_label(k_value)
    p_label = canonical_label(p_value)
    delta_label = canonical_label(delta_value)
    network_seed, phase_seed, network_rule, phase_rule = network_and_phase_seeds(
        config.randomness_mode,
        config.master_seed,
        trial_id,
        network_id,
        phase_id,
        k_label,
        p_label,
        delta_label,
    )
    group = paired_group_id(
        config.randomness_mode,
        network_id,
        phase_id,
        trial_id,
        k_label,
        p_label,
        delta_label,
    )
    return RunSpec(
        model_version=MODEL_VERSION,
        coupling_normalization=COUPLING_NORMALIZATION,
        pulse_function=PULSE_FUNCTION,
        phase_response=PHASE_RESPONSE,
        adjacency_convention=ADJACENCY_CONVENTION,
        frequency_rule=FREQUENCY_RULE,
        frequency_pairing_rule=FREQUENCY_PAIRING_RULE,
        K=k_value,
        p=p_value,
        Delta=delta_value,
        k_units=k_units,
        p_units=p_units,
        delta_units=delta_units,
        N=int(config.N),
        frequency_seed=None,
        network_seed=network_seed,
        initial_phase_seed=phase_seed,
        network_realization_id=network_id,
        initial_condition_id=phase_id,
        trial_id=int(trial_id),
        randomness_mode=config.randomness_mode,
        paired_group_id=group,
        network_pairing_rule=network_rule,
        phase_pairing_rule=phase_rule,
        seed_recipe=SEED_RECIPE,
        solver=config.solver,
        solver_source=solver_source,
        dt=float(dt),
        rtol=config.rtol,
        atol=config.atol,
        t_origin=float(t_origin),
        t_end=float(config.t_end if t_end is None else t_end),
        observation_windows=config.observation_windows if windows is None else windows,
        sample_order_parameter=bool(config.sample_order_parameter),
        order_parameter_dt=config.order_parameter_dt,
        checkpoint_interval=float(config.checkpoint_interval),
        parent_run_id=parent_run_id,
        master_seed=int(config.master_seed),
    )


def execute_run(
    spec: RunSpec,
    output_dir: Path,
    *,
    stale_seconds: float,
    allow_incomplete_restart: bool,
) -> dict:
    output_dir = Path(output_dir)
    connection = connect(output_dir)
    run_id = spec.run_id
    try:
        action = claim_run(
            connection,
            run_id,
            output_dir=output_dir,
            stale_seconds=stale_seconds,
            allow_incomplete_restart=allow_incomplete_restart,
        )
        paths = run_paths(output_dir, run_id)
        if action == "skip_completed":
            return _result(spec, "skipped", "already completed", None)
        if action == "skip_busy":
            return _result(spec, "busy", "another worker holds this run", None)
        if action == "blocked":
            row = connection.execute("SELECT error_message FROM runs WHERE run_id=?", (run_id,)).fetchone()
            message = "" if row is None or row["error_message"] is None else str(row["error_message"])
            return _result(spec, "blocked", message, None)
        if output_is_complete(paths["npz"], spec.t_end):
            mark_status(connection, run_id, "completed", error_message=None)
            return _result(spec, "skipped", "existing output", None)
        attempt_started = time.perf_counter()
        if action == "resume" or paths["checkpoint"].exists():
            state = _integrate_from_checkpoint(
                spec,
                paths["checkpoint"],
                output_dir,
                connection,
                allow_incomplete_restart=allow_incomplete_restart,
                attempt_started=attempt_started,
            )
        else:
            state = _integrate_from_seeds(
                spec,
                paths["checkpoint"],
                output_dir,
                connection,
                attempt_started=attempt_started,
            )
        state.runtime_seconds = float(state.runtime_seconds)
        rotation = rotation_numbers(state, spec.observation_windows)
        if state.omega is None or rotation.shape[-1] != state.omega.shape[0]:
            raise RuntimeError("rotation numbers and omega do not index the same oscillators")
        write_completed_run(output_dir, spec, state, rotation)
        _write_grid_checkpoint(spec, state, paths["checkpoint"])
        mark_status(
            connection,
            run_id,
            "completed",
            error_message=None,
            runtime_seconds=state.runtime_seconds,
        )
        return _result(spec, "completed", "", state.runtime_seconds)
    except KeyboardInterrupt:
        mark_status(connection, run_id, "checkpoint", error_message="interrupted by user")
        raise
    except Exception as exc:
        message = str(exc) if isinstance(exc, ContinuationError) else repr(exc)
        try:
            mark_status(connection, run_id, "failed", error_message=message)
        except sqlite_error():
            pass
        return _result(spec, "failed", message, None)
    finally:
        connection.close()


def sqlite_error():
    import sqlite3

    return sqlite3.Error


def _result(spec: RunSpec, status: str, message: str, runtime_seconds: float | None) -> dict:
    return {
        "run_id": spec.run_id,
        "status": status,
        "message": message,
        "K": spec.K,
        "p": spec.p,
        "Delta": spec.Delta,
        "trial_id": spec.trial_id,
        "runtime_seconds": runtime_seconds,
    }


def _integrate_from_seeds(spec, checkpoint_path, output_dir, connection, attempt_started: float) -> TrajectoryState:
    omega = natural_frequencies(spec.Delta, spec.N)
    quantile_check(omega, spec.Delta)
    theta0 = make_initial_phases(spec.N, spec.initial_phase_seed)
    adjacency = make_adjacency(spec.N, spec.p, spec.network_seed)
    snapshots = {float(spec.t_origin).hex(): theta0.copy()}
    order_parameter: dict[str, float] = {}
    if spec.sample_order_parameter:
        order_parameter[float(spec.t_origin).hex()] = float(np.abs(np.mean(np.exp(1j * theta0))))
    _persist_checkpoint(
        spec,
        checkpoint_path,
        connection,
        theta_grid=theta0,
        t_grid=float(spec.t_origin),
        step_index=0 if spec.solver == RK4_SOLVER else -1,
        omega=omega,
        snapshots=snapshots,
        order_parameter=order_parameter,
        adjacency_sha256=sha256_array(adjacency),
        omega_sha256=sha256_array(omega),
        runtime_seconds=0.0,
    )
    if spec.solver == RK45_SOLVER:
        state = integrate_rk45(spec, theta0, omega, adjacency)
    else:
        state = advance_rk4(
            spec,
            theta0,
            0,
            omega,
            adjacency,
            snapshots,
            order_parameter,
            buffers=make_buffers(spec.N),
            checkpoint_callback=_callback(spec, checkpoint_path, connection, omega, adjacency, attempt_started, 0.0),
            runtime_already=0.0,
        )
    state.runtime_seconds = time.perf_counter() - attempt_started
    state.adjacency_sha256 = sha256_array(adjacency)
    state.omega_sha256 = sha256_array(omega)
    state.omega = omega
    return state


def _integrate_from_checkpoint(
    spec: RunSpec,
    checkpoint_path: Path,
    output_dir: Path,
    connection,
    *,
    allow_incomplete_restart: bool,
    attempt_started: float,
) -> TrajectoryState:
    del output_dir
    checkpoint = load_checkpoint(checkpoint_path)
    _validate_checkpoint(spec, checkpoint)
    omega = checkpoint["omega"]
    adjacency = make_adjacency(spec.N, spec.p, spec.network_seed)
    rebuilt_hash = sha256_array(adjacency)
    if rebuilt_hash != checkpoint["adjacency_sha256"]:
        raise ContinuationError(
            "Rebuilt ER adjacency does not match the checkpoint fingerprint. "
            "Refusing to continue on a different network."
        )
    regenerated = natural_frequencies(spec.Delta, spec.N)
    if sha256_array(regenerated) != checkpoint["omega_sha256"] or not np.allclose(
        regenerated, omega, rtol=0.0, atol=0.0
    ):
        if not np.array_equal(regenerated, omega):
            raise ContinuationError(
                "Regenerated natural frequencies do not match the checkpoint. "
                "Refusing to continue with a different frequency vector."
            )
    previous_runtime = float(checkpoint["runtime_seconds"])
    if spec.solver == RK45_SOLVER:
        if not allow_incomplete_restart:
            raise ContinuationError(
                "RK45 has no saved step-size controller, so a midpoint resume is impossible. "
                "Refusing to restart the adaptive solver silently. "
                "Pass --allow-incomplete-restart to recompute from the saved initial condition."
            )
        if int(checkpoint["step_index"]) != -1 and int(checkpoint["step_index"]) != 0:
            raise ContinuationError(
                "RK45 checkpoint is not the initial condition. "
                "Refusing to continue an adaptive trajectory."
            )
        theta0 = np.array(checkpoint["theta_grid"], dtype=np.float64, copy=True)
        state = integrate_rk45(spec, theta0, omega, adjacency)
        state.runtime_seconds = previous_runtime + (time.perf_counter() - attempt_started)
        state.resumed_from_time = float(spec.t_origin)
        state.resumed_from_step = 0
        return state
    if int(checkpoint["continuable"]) != 1 or int(checkpoint["step_index"]) < 0:
        raise ContinuationError(
            "Checkpoint is not an RK4 full-step state. Refusing to restart from initial conditions."
        )
    state = advance_rk4(
        spec,
        checkpoint["theta_grid"],
        int(checkpoint["step_index"]),
        omega,
        adjacency,
        checkpoint["snapshots"],
        checkpoint["order_parameter"],
        buffers=make_buffers(spec.N),
        checkpoint_callback=_callback(
            spec,
            checkpoint_path,
            connection,
            omega,
            adjacency,
            attempt_started,
            previous_runtime,
        ),
        runtime_already=previous_runtime,
    )
    state.runtime_seconds = previous_runtime + (time.perf_counter() - attempt_started)
    state.omega = omega
    state.adjacency_sha256 = rebuilt_hash
    state.omega_sha256 = sha256_array(omega)
    return state


def _validate_checkpoint(spec: RunSpec, checkpoint: dict) -> None:
    if checkpoint["run_id"] != spec.run_id:
        raise ContinuationError(
            f"Checkpoint run_id {checkpoint['run_id']} does not match {spec.run_id}. "
            "Refusing to load another trajectory."
        )
    if checkpoint["solver"] != spec.solver:
        raise ContinuationError(
            f"Checkpoint solver {checkpoint['solver']} does not match requested {spec.solver}."
        )
    if int(checkpoint["network_seed"]) != int(spec.network_seed) or int(checkpoint["initial_phase_seed"]) != int(
        spec.initial_phase_seed
    ):
        raise ContinuationError("Checkpoint seeds do not match this run. Refusing to continue.")
    if float(checkpoint["dt"]).hex() != float(spec.dt).hex():
        raise ContinuationError(
            f"Checkpoint dt {checkpoint['dt']} does not match this run's dt {spec.dt}."
        )
    if abs(float(checkpoint["t_origin"]) - float(spec.t_origin)) > TIME_TOL:
        raise ContinuationError("Checkpoint t_origin does not match this run.")
    if checkpoint["omega"].shape != (spec.N,):
        raise ContinuationError("Checkpoint omega length does not match N.")
    if checkpoint["theta_grid"].shape != (spec.N,):
        raise ContinuationError("Checkpoint phase length does not match N.")
    index = np.asarray(checkpoint["oscillator_index"])
    if index.shape != (spec.N,) or np.any(index != np.arange(spec.N)):
        raise ContinuationError(
            "Checkpoint oscillator_index is not the original identity 0..N-1. "
            "Refusing to load a reordered profile."
        )
    if spec.solver == RK4_SOLVER and int(checkpoint["step_index"]) >= 0:
        expected = grid_time(spec.t_origin, int(checkpoint["step_index"]), spec.dt)
        if abs(expected - float(checkpoint["t_grid"])) > 1e-6 * max(1.0, abs(expected)):
            raise ContinuationError(
                "Checkpoint time is not on the fixed RK4 grid "
                f"(stored t_grid={checkpoint['t_grid']}, expected {expected}). "
                "Refusing to continue from an interpolated state."
            )


def _callback(spec, checkpoint_path, connection, omega, adjacency, attempt_started, previous_runtime):
    adjacency_sha = sha256_array(adjacency)
    omega_sha = sha256_array(omega)

    def _save(*, theta, t_grid, step_index, snapshots, order_parameter, omega, partial):
        del partial
        elapsed = previous_runtime + (time.perf_counter() - attempt_started)
        _persist_checkpoint(
            spec,
            checkpoint_path,
            connection,
            theta_grid=theta,
            t_grid=t_grid,
            step_index=step_index,
            omega=omega,
            snapshots=snapshots,
            order_parameter=order_parameter,
            adjacency_sha256=adjacency_sha,
            omega_sha256=omega_sha,
            runtime_seconds=elapsed,
        )

    return _save


def _persist_checkpoint(
    spec,
    checkpoint_path,
    connection,
    *,
    theta_grid,
    t_grid,
    step_index,
    omega,
    snapshots,
    order_parameter,
    adjacency_sha256,
    omega_sha256,
    runtime_seconds,
) -> None:
    arrays = checkpoint_arrays(
        spec,
        theta_grid=theta_grid,
        t_grid=t_grid,
        step_index=step_index,
        omega=omega,
        snapshots=snapshots,
        order_parameter=order_parameter,
        adjacency_sha256=adjacency_sha256,
        omega_sha256=omega_sha256,
        runtime_seconds=runtime_seconds,
    )
    write_checkpoint(checkpoint_path, arrays)
    touch_heartbeat(connection, spec.run_id)


def _write_grid_checkpoint(spec: RunSpec, state: TrajectoryState, checkpoint_path: Path) -> None:
    if spec.solver != RK4_SOLVER or state.step_index < 0 or state.omega is None:
        return
    arrays = checkpoint_arrays(
        spec,
        theta_grid=state.theta_grid,
        t_grid=state.t_grid,
        step_index=state.step_index,
        omega=state.omega,
        snapshots=state.snapshots,
        order_parameter=state.order_parameter,
        adjacency_sha256=state.adjacency_sha256,
        omega_sha256=state.omega_sha256,
        runtime_seconds=state.runtime_seconds,
    )
    write_checkpoint(checkpoint_path, arrays)


def execute_task(payload: dict) -> dict:
    spec = payload["spec"]
    try:
        return execute_run(
            spec,
            Path(payload["output_dir"]),
            stale_seconds=float(payload["stale_seconds"]),
            allow_incomplete_restart=bool(payload["allow_incomplete_restart"]),
        )
    except Exception as exc:  # pragma: no cover - worker safety net
        return _result(spec, "failed", repr(exc), None)


def run_scan(config: ScanConfig) -> dict:
    """Insert the planned index and execute pending or resumable runs."""
    specs = build_run_specs(config)
    print(
        "stage1 planning "
        f"{len(specs)} runs "
        f"({len(config.resolved_p())} p, {len(config.extensions)} extensions, "
        f"{config.n_trials} trials); completed run_ids are skipped"
    )
    output_dir = Path(config.output_dir)
    connection = connect(output_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for spec in specs:
                insert_planned(connection, spec, output_dir)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        before = count_status_for_ids(connection, [spec.run_id for spec in specs])
    finally:
        connection.close()

    print(
        "stage1 scan: "
        f"planned_now={len(specs)} "
        f"completed={before['completed']} pending={before['pending']} "
        f"checkpoint={before['checkpoint']} running={before['running']} "
        f"failed={before['failed']} absent={before['absent']} output={output_dir}"
    )
    print(
        f"model={MODEL_VERSION} coupling={COUPLING_NORMALIZATION} "
        f"adjacency={ADJACENCY_CONVENTION} frequency={FREQUENCY_RULE}"
    )
    print(
        f"solver={config.solver} mode={config.randomness_mode} "
        f"N={config.N} trials={config.n_trials} workers={config.workers} "
        f"t_end={config.t_end}"
    )

    payloads = [
        {
            "spec": spec,
            "output_dir": str(output_dir),
            "stale_seconds": config.stale_seconds,
            "allow_incomplete_restart": config.allow_incomplete_restart,
        }
        for spec in specs
    ]
    started = time.perf_counter()
    results = _execute_payloads(payloads, config.workers)
    wall = time.perf_counter() - started
    summary = _summarize(results, wall, len(specs))
    _print_summary(summary)
    return summary


def _execute_payloads(payloads: list[dict], workers: int) -> list[dict]:
    root = str(PROJECT_ROOT)
    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = root if not previous else root + os.pathsep + previous
    if workers <= 1 or len(payloads) <= 1:
        results = []
        for index, payload in enumerate(payloads, start=1):
            result = execute_task(payload)
            results.append(result)
            _print_progress(index, len(payloads), result, results)
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
    except (PermissionError, OSError) as exc:
        print(
            "stage1: process pool is unavailable "
            f"({exc.__class__.__name__}: {exc}); continuing with one worker"
        )
        return _execute_payloads(payloads, 1)
    with executor:
        futures = [executor.submit(execute_task, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            _print_progress(index, len(payloads), result, results)
    return results


def _print_progress(index: int, total: int, result: dict, results: list[dict]) -> None:
    computed = [item for item in results if item["status"] == "completed" and item["runtime_seconds"]]
    eta_note = "eta=waiting_for_measured_runtime"
    if computed:
        mean_runtime = sum(float(item["runtime_seconds"]) for item in computed) / len(computed)
        remaining = total - index
        eta_note = (
            f"measured_mean_runtime_s={mean_runtime:.3f} "
            f"extrapolated_session_eta_s={mean_runtime * remaining:.1f}"
        )
    message = f" message={result['message']}" if result["message"] else ""
    print(
        f"[{index}/{total}] status={result['status']} "
        f"K={result['K']} p={result['p']} Delta={result['Delta']} "
        f"trial={result['trial_id']} run_id={result['run_id']} {eta_note}{message}",
        flush=True,
    )


def _summarize(results: list[dict], wall_seconds: float, planned: int) -> dict:
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    computed = [item for item in results if item["status"] == "completed" and item["runtime_seconds"]]
    mean_runtime = None
    if computed:
        mean_runtime = sum(float(item["runtime_seconds"]) for item in computed) / len(computed)
    return {
        "planned": planned,
        "wall_seconds": wall_seconds,
        "counts": counts,
        "measured_mean_runtime_seconds": mean_runtime,
        "measured_label": "measured_this_invocation",
    }


def _print_summary(summary: dict) -> None:
    mean = summary["measured_mean_runtime_seconds"]
    mean_text = "n/a" if mean is None else f"{mean:.3f}"
    print(
        "stage1 scan finished: "
        f"counts={summary['counts']} wall_s={summary['wall_seconds']:.3f} "
        f"measured_mean_runtime_s={mean_text} "
        "(mean is measured for runs computed in this invocation; "
        "it is not an estimate of an unmeasured grid)"
    )


def continue_trajectory(
    output_dir: Path,
    run_id: str,
    *,
    t_end: float | None = None,
    extra_windows: list[tuple[float, float]] | None = None,
    allow_incomplete_restart: bool = False,
    stale_seconds: float = 3600.0,
) -> dict:
    """Extend a completed RK4 trajectory or append windows that are already stored.

    Samples after the last full step are recomputed when ``t_end`` increases,
    so the extended run matches an uninterrupted integration. Missing interior
    snapshots raise ``ContinuationError`` instead of starting a new random trial.
    """
    from stage1.query import load_run

    output_dir = Path(output_dir)
    loaded = load_run(run_id, output_dir=output_dir)
    meta = loaded["metadata"]
    if meta.get("solver") != RK4_SOLVER or not meta.get("continuable", False):
        raise ContinuationError(
            "This run cannot be extended. "
            "RK45 adaptive state is not checkpointed, and a shortened or interpolated "
            "phase is not a valid RK4 resume point. "
            f"solver={meta.get('solver')} continuable={meta.get('continuable')}."
        )
    windows = _merge_windows(meta["windows_hex"], extra_windows or [])
    parent_end = float.fromhex(meta["t_end_hex"])
    requested_end = parent_end if t_end is None else float(t_end)
    if windows and requested_end + TIME_TOL < max(end for _, end in windows):
        raise ValueError("t_end must cover every observation window")
    if requested_end + TIME_TOL < parent_end:
        raise ContinuationError(
            f"Refusing to shorten a trajectory from t_end={parent_end} to {requested_end}."
        )
    spec = _spec_from_metadata(meta, t_end=requested_end, windows=windows, parent_run_id=run_id)
    for start, end in spec.observation_windows:
        if start < spec.t_origin - TIME_TOL or end > spec.t_end + TIME_TOL:
            raise ValueError(f"window {(start, end)} lies outside the continued trajectory")
    if spec.run_id == run_id:
        return _result(spec, "unchanged", "windows and t_end already match this run", None)
    _seed_child_checkpoint(output_dir, loaded, spec, extending=requested_end > parent_end + TIME_TOL)
    connection = connect(output_dir)
    try:
        insert_planned(connection, spec, output_dir)
    finally:
        connection.close()
    result = execute_run(
        spec,
        output_dir,
        stale_seconds=stale_seconds,
        allow_incomplete_restart=allow_incomplete_restart,
    )
    if result["status"] in {"failed", "blocked"}:
        raise ContinuationError(result["message"] or result["status"])
    return result


def _merge_windows(
    stored_hex: list[list[str]],
    extra: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[tuple[float, float]] = []
    seen: set[tuple[str, str]] = set()
    for pair in stored_hex:
        start = float.fromhex(pair[0])
        end = float.fromhex(pair[1])
        key = (start.hex(), end.hex())
        if key not in seen:
            seen.add(key)
            merged.append((start, end))
    for start, end in extra:
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            raise ValueError(f"window {(start, end)} must have positive length")
        key = (start_f.hex(), end_f.hex())
        if key not in seen:
            seen.add(key)
            merged.append((start_f, end_f))
    merged.sort(key=lambda item: (item[0], item[1]))
    if not merged:
        raise ValueError("at least one observation window is required")
    return tuple(merged)


def _spec_from_metadata(
    meta: dict,
    *,
    t_end: float,
    windows: tuple[tuple[float, float], ...],
    parent_run_id: str,
) -> RunSpec:
    from stage1.grid import from_units

    return RunSpec(
        model_version=meta["model_version"],
        coupling_normalization=meta["coupling_normalization"],
        pulse_function=meta["pulse_function"],
        phase_response=meta["phase_response"],
        adjacency_convention=meta["adjacency_convention"],
        frequency_rule=meta["frequency_rule"],
        frequency_pairing_rule=meta["frequency_pairing_rule"],
        K=from_units(int(meta["k_units"])),
        p=from_units(int(meta["p_units"])),
        Delta=from_units(int(meta["delta_units"])),
        k_units=int(meta["k_units"]),
        p_units=int(meta["p_units"]),
        delta_units=int(meta["delta_units"]),
        N=int(meta["N"]),
        frequency_seed=meta["frequency_seed"],
        network_seed=int(meta["network_seed"]),
        initial_phase_seed=int(meta["initial_phase_seed"]),
        network_realization_id=int(meta["network_realization_id"]),
        initial_condition_id=int(meta["initial_condition_id"]),
        trial_id=int(meta["trial_id"]),
        randomness_mode=meta["randomness_mode"],
        paired_group_id=meta["paired_group_id"],
        network_pairing_rule=meta["network_pairing_rule"],
        phase_pairing_rule=meta["phase_pairing_rule"],
        seed_recipe=meta["seed_recipe"],
        solver=meta["solver"],
        solver_source=meta["solver_source"],
        dt=float.fromhex(meta["dt_hex"]),
        rtol=meta["rtol"],
        atol=meta["atol"],
        t_origin=float.fromhex(meta["t_origin_hex"]),
        t_end=float(t_end),
        observation_windows=windows,
        sample_order_parameter=bool(meta["sample_order_parameter"]),
        order_parameter_dt=(
            None if meta["order_parameter_dt"] is None else float(meta["order_parameter_dt"])
        ),
        checkpoint_interval=float(meta["checkpoint_interval"]),
        parent_run_id=parent_run_id,
        master_seed=int(meta["master_seed"]),
    )


def _seed_child_checkpoint(output_dir: Path, parent: dict, spec: RunSpec, *, extending: bool) -> None:
    child = run_paths(output_dir, spec.run_id)["checkpoint"]
    if child.exists():
        return
    meta = parent["metadata"]
    t_grid = float(parent["t_grid"])
    snapshots: dict[str, np.ndarray] = {}
    times = np.asarray(parent["snapshot_times"], dtype=np.float64)
    phases = np.asarray(parent["snapshot_theta"], dtype=np.float64)
    for index, time in enumerate(times):
        if extending and float(time) > t_grid + TIME_TOL:
            continue
        snapshots[float(time).hex()] = np.array(phases[index], dtype=np.float64, copy=True)
    order_parameter: dict[str, float] = {}
    r_times = np.asarray(parent["r_times"], dtype=np.float64)
    r_values = np.asarray(parent["r_values"], dtype=np.float64)
    for index, time in enumerate(r_times):
        if extending and float(time) > t_grid + TIME_TOL:
            continue
        order_parameter[float(time).hex()] = float(r_values[index])
    required = [time for time in theta_sample_times(spec) if time <= t_grid + TIME_TOL]
    missing = [time for time in required if float(time).hex() not in snapshots]
    if missing:
        available = sorted(float.fromhex(key) for key in snapshots)
        raise ContinuationError(
            "Cannot add observation times "
            f"{missing}: they are at or before checkpoint time {t_grid} "
            "and were not saved. Refusing to regenerate initial conditions. "
            f"Available snapshot times: {available}"
        )
    omega = np.asarray(parent["omega"], dtype=np.float64)
    arrays = checkpoint_arrays(
        spec,
        theta_grid=np.asarray(parent["theta_grid"], dtype=np.float64),
        t_grid=t_grid,
        step_index=int(parent["step_index"]),
        omega=omega,
        snapshots=snapshots,
        order_parameter=order_parameter,
        adjacency_sha256=str(meta["adjacency_sha256"]),
        omega_sha256=str(meta["omega_sha256"]),
        runtime_seconds=float(meta.get("runtime_seconds") or 0.0),
    )
    write_checkpoint(child, arrays)


def adopt_completed_file(output_dir: Path, run_id: str) -> None:
    """Mark a finished file completed without rewriting it."""
    connection = connect(output_dir)
    try:
        row = connection.execute("SELECT output_npz, t_end FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        path = resolve_stored_path(output_dir, str(row["output_npz"]))
        if output_is_complete(path, float(row["t_end"])):
            mark_status(connection, run_id, "completed", error_message=None)
    finally:
        connection.close()
