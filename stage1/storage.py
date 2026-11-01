"""SQLite index and atomic NPZ/JSON run files.

The index is the lookup path. Startup does not read every NPZ. Writes go to
a temporary file in the destination directory and then replace the target.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from stage1.integrator import TrajectoryState
from stage1.spec import RunSpec

SCHEMA_VERSION = 1

RUN_COLUMNS = (
    "run_id",
    "lineage_id",
    "status",
    "k_units",
    "p_units",
    "delta_units",
    "n",
    "K",
    "p",
    "delta",
    "network_seed",
    "initial_phase_seed",
    "frequency_seed",
    "frequency_rule",
    "network_realization_id",
    "initial_condition_id",
    "trial_id",
    "randomness_mode",
    "paired_group_id",
    "network_pairing_rule",
    "phase_pairing_rule",
    "solver",
    "dt",
    "t_origin",
    "t_end",
    "model_version",
    "seed_recipe",
    "master_seed",
    "output_npz",
    "output_json",
    "checkpoint_path",
    "parent_run_id",
    "runtime_seconds",
    "error_message",
    "worker_pid",
    "heartbeat_unix",
    "continuable",
    "created_unix",
    "updated_unix",
)


def database_path(output_dir: Path) -> Path:
    return Path(output_dir) / "index.sqlite"


def run_paths(output_dir: Path, run_id: str) -> dict[str, Path]:
    shard = Path(output_dir) / "runs" / run_id[:2]
    return {
        "directory": shard,
        "npz": shard / f"{run_id}.npz",
        "json": shard / f"{run_id}.json",
        "checkpoint": shard / f"{run_id}.checkpoint.npz",
    }


def relative_path(path: Path, output_dir: Path) -> str:
    return Path(path).resolve().relative_to(Path(output_dir).resolve()).as_posix()


def connect(output_dir: Path) -> sqlite3.Connection:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path(output_dir), timeout=60.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if user_version not in (0, SCHEMA_VERSION):
        raise RuntimeError(
            f"unsupported stage1 index schema version {user_version}; expected {SCHEMA_VERSION}"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            lineage_id TEXT NOT NULL,
            status TEXT NOT NULL,
            k_units INTEGER NOT NULL,
            p_units INTEGER NOT NULL,
            delta_units INTEGER NOT NULL,
            n INTEGER NOT NULL,
            K REAL NOT NULL,
            p REAL NOT NULL,
            delta REAL NOT NULL,
            network_seed INTEGER NOT NULL,
            initial_phase_seed INTEGER NOT NULL,
            frequency_seed INTEGER,
            frequency_rule TEXT NOT NULL,
            network_realization_id INTEGER NOT NULL,
            initial_condition_id INTEGER NOT NULL,
            trial_id INTEGER NOT NULL,
            randomness_mode TEXT NOT NULL,
            paired_group_id TEXT NOT NULL,
            network_pairing_rule TEXT NOT NULL,
            phase_pairing_rule TEXT NOT NULL,
            solver TEXT NOT NULL,
            dt REAL NOT NULL,
            t_origin REAL NOT NULL,
            t_end REAL NOT NULL,
            model_version TEXT NOT NULL,
            seed_recipe TEXT NOT NULL,
            master_seed INTEGER NOT NULL,
            output_npz TEXT NOT NULL,
            output_json TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            parent_run_id TEXT,
            runtime_seconds REAL,
            error_message TEXT,
            worker_pid INTEGER,
            heartbeat_unix REAL,
            continuable INTEGER NOT NULL,
            created_unix REAL NOT NULL,
            updated_unix REAL NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_params ON runs(k_units, p_units, delta_units, n)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_seeds ON runs(network_seed, initial_phase_seed)"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_lineage ON runs(lineage_id)")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_trial ON runs(trial_id, network_realization_id, initial_condition_id)"
    )
    if user_version == 0:
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def planned_row(spec: RunSpec, output_dir: Path, now: float | None = None) -> dict:
    paths = run_paths(output_dir, spec.run_id)
    timestamp = time.time() if now is None else float(now)
    return {
        "run_id": spec.run_id,
        "lineage_id": spec.lineage_id,
        "status": "pending",
        "k_units": spec.k_units,
        "p_units": spec.p_units,
        "delta_units": spec.delta_units,
        "n": spec.N,
        "K": spec.K,
        "p": spec.p,
        "delta": spec.Delta,
        "network_seed": spec.network_seed,
        "initial_phase_seed": spec.initial_phase_seed,
        "frequency_seed": spec.frequency_seed,
        "frequency_rule": spec.frequency_rule,
        "network_realization_id": spec.network_realization_id,
        "initial_condition_id": spec.initial_condition_id,
        "trial_id": spec.trial_id,
        "randomness_mode": spec.randomness_mode,
        "paired_group_id": spec.paired_group_id,
        "network_pairing_rule": spec.network_pairing_rule,
        "phase_pairing_rule": spec.phase_pairing_rule,
        "solver": spec.solver,
        "dt": spec.dt,
        "t_origin": spec.t_origin,
        "t_end": spec.t_end,
        "model_version": spec.model_version,
        "seed_recipe": spec.seed_recipe,
        "master_seed": spec.master_seed,
        "output_npz": relative_path(paths["npz"], output_dir),
        "output_json": relative_path(paths["json"], output_dir),
        "checkpoint_path": relative_path(paths["checkpoint"], output_dir),
        "parent_run_id": spec.parent_run_id,
        "runtime_seconds": None,
        "error_message": None,
        "worker_pid": None,
        "heartbeat_unix": None,
        "continuable": 1 if spec.solver == "rk4" else 0,
        "created_unix": timestamp,
        "updated_unix": timestamp,
    }


def insert_planned(connection: sqlite3.Connection, spec: RunSpec, output_dir: Path) -> None:
    row = planned_row(spec, output_dir)
    columns = RUN_COLUMNS
    placeholders = ",".join("?" for _ in columns)
    connection.execute(
        f"INSERT OR IGNORE INTO runs ({','.join(columns)}) VALUES ({placeholders})",
        [row[column] for column in columns],
    )


def claim_run(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    output_dir: Path,
    stale_seconds: float,
    allow_incomplete_restart: bool,
    now: float | None = None,
) -> str:
    """Claim one run.

    Returns ``skip_completed``, ``skip_busy``, ``blocked``, ``start``,
    ``resume``, or ``restart``.
    """
    timestamp = time.time() if now is None else float(now)
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            connection.execute("ROLLBACK")
            raise KeyError(run_id)
        status = str(row["status"])
        heartbeat = float(row["heartbeat_unix"] or 0.0)
        fresh = status == "running" and (timestamp - heartbeat) < float(stale_seconds)
        if status == "completed":
            connection.execute("ROLLBACK")
            return "skip_completed"
        if fresh:
            connection.execute("ROLLBACK")
            return "skip_busy"
        checkpoint = resolve_stored_path(output_dir, str(row["checkpoint_path"]))
        checkpoint_exists = checkpoint.exists()
        if status == "pending" and not checkpoint_exists:
            action = "start"
        elif checkpoint_exists and status in {"pending", "checkpoint", "running", "failed"}:
            action = "resume"
        elif allow_incomplete_restart and status in {"checkpoint", "running", "failed"}:
            action = "restart"
        else:
            message = (
                "Interrupted run has no restorable checkpoint. "
                "Refusing to draw new initial conditions and start over. "
                "Pass --allow-incomplete-restart to recompute this run from t_origin "
                "using the recorded seeds."
            )
            if status != "failed" or row["error_message"] != message:
                connection.execute(
                    """
                    UPDATE runs
                    SET status='failed', error_message=?, updated_unix=?, worker_pid=NULL
                    WHERE run_id=?
                    """,
                    (message, timestamp, run_id),
                )
            connection.execute("COMMIT")
            return "blocked"
        connection.execute(
            """
            UPDATE runs
            SET status='running', worker_pid=?, heartbeat_unix=?, updated_unix=?, error_message=NULL
            WHERE run_id=?
            """,
            (os.getpid(), timestamp, timestamp, run_id),
        )
        connection.execute("COMMIT")
        return action
    except Exception:
        connection.execute("ROLLBACK")
        raise


def resolve_stored_path(output_dir: Path, stored: str) -> Path:
    path = Path(stored)
    if path.is_absolute():
        return path
    return Path(output_dir) / stored


def mark_status(
    connection: sqlite3.Connection,
    run_id: str,
    status: str,
    *,
    error_message: str | None = None,
    runtime_seconds: float | None = None,
) -> None:
    timestamp = time.time()
    connection.execute(
        """
        UPDATE runs
        SET status=?, error_message=?, runtime_seconds=COALESCE(?, runtime_seconds),
            updated_unix=?, heartbeat_unix=?, worker_pid=?
        WHERE run_id=?
        """,
        (
            status,
            error_message,
            runtime_seconds,
            timestamp,
            timestamp,
            os.getpid(),
            run_id,
        ),
    )


def touch_heartbeat(connection: sqlite3.Connection, run_id: str) -> None:
    timestamp = time.time()
    connection.execute(
        "UPDATE runs SET heartbeat_unix=?, updated_unix=?, worker_pid=? WHERE run_id=?",
        (timestamp, timestamp, os.getpid(), run_id),
    )


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".json", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _snapshot_arrays(snapshots: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    if not snapshots:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, n), dtype=np.float64),
        )
    items = sorted(snapshots.items(), key=lambda item: float.fromhex(item[0]))
    times = np.asarray([float.fromhex(key) for key, _ in items], dtype=np.float64)
    phases = np.vstack([np.asarray(phase, dtype=np.float64) for _, phase in items])
    return times, phases


def _order_arrays(order_parameter: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    if not order_parameter:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    items = sorted(order_parameter.items(), key=lambda item: float.fromhex(item[0]))
    times = np.asarray([float.fromhex(key) for key, _ in items], dtype=np.float64)
    values = np.asarray([value for _, value in items], dtype=np.float64)
    return times, values


def checkpoint_arrays(
    spec: RunSpec,
    *,
    theta_grid: np.ndarray,
    t_grid: float,
    step_index: int,
    omega: np.ndarray,
    snapshots: dict[str, np.ndarray],
    order_parameter: dict[str, float],
    adjacency_sha256: str,
    omega_sha256: str,
    runtime_seconds: float,
) -> dict[str, np.ndarray]:
    n = int(spec.N)
    snapshot_times, snapshot_theta = _snapshot_arrays(snapshots, n)
    r_times, r_values = _order_arrays(order_parameter)
    return {
        "run_id": np.array(spec.run_id),
        "solver": np.array(spec.solver),
        "continuable": np.int32(1 if spec.solver == "rk4" and int(step_index) >= 0 else 0),
        "theta_grid": np.asarray(theta_grid, dtype=np.float64),
        "t_grid": np.float64(t_grid),
        "t_origin": np.float64(spec.t_origin),
        "dt": np.float64(spec.dt),
        "step_index": np.int64(step_index),
        "omega": np.asarray(omega, dtype=np.float64),
        "oscillator_index": np.arange(n, dtype=np.int32),
        "snapshot_times": snapshot_times,
        "snapshot_theta": snapshot_theta,
        "r_times": r_times,
        "r_values": r_values,
        "adjacency_sha256": np.array(adjacency_sha256),
        "omega_sha256": np.array(omega_sha256),
        "network_seed": np.int64(spec.network_seed),
        "initial_phase_seed": np.int64(spec.initial_phase_seed),
        "runtime_seconds": np.float64(runtime_seconds),
    }


def write_checkpoint(path: Path, arrays: dict[str, np.ndarray]) -> None:
    atomic_savez(path, **arrays)


def load_checkpoint(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        snapshot_times = np.asarray(data["snapshot_times"], dtype=np.float64)
        snapshot_theta = np.asarray(data["snapshot_theta"], dtype=np.float64)
        snapshots: dict[str, np.ndarray] = {}
        for index, time in enumerate(snapshot_times):
            snapshots[float(time).hex()] = np.array(snapshot_theta[index], dtype=np.float64, copy=True)
        r_times = np.asarray(data["r_times"], dtype=np.float64)
        r_values = np.asarray(data["r_values"], dtype=np.float64)
        order_parameter = {
            float(r_times[index]).hex(): float(r_values[index]) for index in range(r_times.size)
        }
        return {
            "run_id": str(np.asarray(data["run_id"]).item()),
            "solver": str(np.asarray(data["solver"]).item()),
            "continuable": int(np.asarray(data["continuable"]).item()),
            "theta_grid": np.array(data["theta_grid"], dtype=np.float64, copy=True),
            "t_grid": float(np.asarray(data["t_grid"]).item()),
            "t_origin": float(np.asarray(data["t_origin"]).item()),
            "dt": float(np.asarray(data["dt"]).item()),
            "step_index": int(np.asarray(data["step_index"]).item()),
            "omega": np.array(data["omega"], dtype=np.float64, copy=True),
            "oscillator_index": np.array(data["oscillator_index"], dtype=np.int32, copy=True),
            "snapshots": snapshots,
            "order_parameter": order_parameter,
            "adjacency_sha256": str(np.asarray(data["adjacency_sha256"]).item()),
            "omega_sha256": str(np.asarray(data["omega_sha256"]).item()),
            "network_seed": int(np.asarray(data["network_seed"]).item()),
            "initial_phase_seed": int(np.asarray(data["initial_phase_seed"]).item()),
            "runtime_seconds": float(np.asarray(data["runtime_seconds"]).item()),
        }


def metadata_document(
    spec: RunSpec,
    state: TrajectoryState,
    *,
    output_dir: Path,
    status: str,
    error_message: str | None,
) -> dict:
    paths = run_paths(output_dir, spec.run_id)
    windows = [[float(start), float(end)] for start, end in spec.observation_windows]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_id": spec.run_id,
        "lineage_id": spec.lineage_id,
        "parent_run_id": spec.parent_run_id,
        "model_version": spec.model_version,
        "coupling_normalization": spec.coupling_normalization,
        "pulse_function": spec.pulse_function,
        "phase_response": spec.phase_response,
        "adjacency_convention": spec.adjacency_convention,
        "K": spec.K,
        "p": spec.p,
        "Delta": spec.Delta,
        "k_units": spec.k_units,
        "p_units": spec.p_units,
        "delta_units": spec.delta_units,
        "N": spec.N,
        "frequency_rule": spec.frequency_rule,
        "frequency_pairing_rule": spec.frequency_pairing_rule,
        "frequency_seed": spec.frequency_seed,
        "network_seed": spec.network_seed,
        "initial_phase_seed": spec.initial_phase_seed,
        "network_realization_id": spec.network_realization_id,
        "initial_condition_id": spec.initial_condition_id,
        "trial_id": spec.trial_id,
        "randomness_mode": spec.randomness_mode,
        "paired_group_id": spec.paired_group_id,
        "network_pairing_rule": spec.network_pairing_rule,
        "phase_pairing_rule": spec.phase_pairing_rule,
        "seed_recipe": spec.seed_recipe,
        "master_seed": spec.master_seed,
        "solver": spec.solver,
        "solver_source": spec.solver_source,
        "dt": spec.dt,
        "dt_hex": float(spec.dt).hex(),
        "rtol": spec.rtol,
        "atol": spec.atol,
        "t_origin": spec.t_origin,
        "t_origin_hex": float(spec.t_origin).hex(),
        "t_end": spec.t_end,
        "t_end_hex": float(spec.t_end).hex(),
        "t_grid": state.t_grid,
        "t_final": state.t_final,
        "step_index": state.step_index,
        "resumed_from_time": state.resumed_from_time,
        "resumed_from_step": state.resumed_from_step,
        "observation_windows": windows,
        "windows_hex": [
            [float(start).hex(), float(end).hex()] for start, end in spec.observation_windows
        ],
        "sample_order_parameter": spec.sample_order_parameter,
        "order_parameter_dt": spec.order_parameter_dt,
        "checkpoint_interval": spec.checkpoint_interval,
        "continuable": spec.solver == "rk4" and state.step_index >= 0,
        "continuation_note": (
            "RK4 continuation resumes from theta_grid at the last completed full step. "
            "The shortened step that lands on t_end is stored as theta_final and is not "
            "the resume state. RK45 midpoint continuation is not supported."
        ),
        "adjacency_sha256": state.adjacency_sha256,
        "omega_sha256": state.omega_sha256,
        "output_npz": relative_path(paths["npz"], output_dir),
        "output_json": relative_path(paths["json"], output_dir),
        "checkpoint_path": relative_path(paths["checkpoint"], output_dir),
        "runtime_seconds": state.runtime_seconds,
        "error_message": error_message,
        "oscillator_order": "original_index_not_sorted_by_frequency",
    }


def result_arrays(spec: RunSpec, state: TrajectoryState, rotation: np.ndarray) -> dict[str, np.ndarray]:
    n = int(spec.N)
    snapshot_times, snapshot_theta = _snapshot_arrays(state.snapshots, n)
    r_times, r_values = _order_arrays(state.order_parameter)
    window_start = np.asarray([start for start, _ in spec.observation_windows], dtype=np.float64)
    window_end = np.asarray([end for _, end in spec.observation_windows], dtype=np.float64)
    if rotation.shape != (window_start.size, n):
        raise RuntimeError(
            f"rotation_numbers shape {rotation.shape} does not match {(window_start.size, n)}"
        )
    if state.omega is None or state.omega.shape != (n,):
        raise RuntimeError("omega is missing from the trajectory state")
    if rotation.shape[-1] != state.omega.shape[0]:
        raise RuntimeError("rotation numbers and omega index different oscillators")
    return {
        "omega": np.asarray(state.omega, dtype=np.float64),
        "oscillator_index": np.arange(n, dtype=np.int32),
        "rotation_numbers": np.asarray(rotation, dtype=np.float64),
        "rotation_number": np.asarray(rotation[-1], dtype=np.float64),
        "window_start": window_start,
        "window_end": window_end,
        "theta_final": np.asarray(state.theta_final, dtype=np.float64),
        "theta_grid": np.asarray(state.theta_grid, dtype=np.float64),
        "t_final": np.float64(state.t_final),
        "t_grid": np.float64(state.t_grid),
        "t_origin": np.float64(spec.t_origin),
        "dt": np.float64(spec.dt),
        "step_index": np.int64(state.step_index),
        "snapshot_times": snapshot_times,
        "snapshot_theta": snapshot_theta,
        "r_times": r_times,
        "r_values": r_values,
        "K": np.float64(spec.K),
        "p": np.float64(spec.p),
        "Delta": np.float64(spec.Delta),
        "N": np.int32(spec.N),
        "network_seed": np.int64(spec.network_seed),
        "initial_phase_seed": np.int64(spec.initial_phase_seed),
        "trial_id": np.int32(spec.trial_id),
    }


def output_is_complete(path: Path, t_end: float) -> bool:
    if not Path(path).exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if "rotation_numbers" not in data or "omega" not in data or "t_final" not in data:
                return False
            omega = np.asarray(data["omega"])
            rotation = np.asarray(data["rotation_numbers"])
            if rotation.ndim != 2 or rotation.shape[1] != omega.shape[0]:
                return False
            return float(np.asarray(data["t_final"]).item()) >= float(t_end) - 1e-8
    except (OSError, ValueError, KeyError):
        return False


def write_completed_run(
    output_dir: Path,
    spec: RunSpec,
    state: TrajectoryState,
    rotation: np.ndarray,
) -> dict:
    paths = run_paths(output_dir, spec.run_id)
    arrays = result_arrays(spec, state, rotation)
    document = metadata_document(
        spec,
        state,
        output_dir=output_dir,
        status="completed",
        error_message=None,
    )
    atomic_savez(paths["npz"], **arrays)
    atomic_json(paths["json"], document)
    return document


def count_by_status(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute("SELECT status, COUNT(*) AS n FROM runs GROUP BY status").fetchall()
    counts = {str(row["status"]): int(row["n"]) for row in rows}
    for status in ("pending", "running", "checkpoint", "completed", "failed"):
        counts.setdefault(status, 0)
    return counts


def count_status_for_ids(connection: sqlite3.Connection, run_ids: list[str]) -> dict[str, int]:
    """Count statuses for one planned set of run ids.

    Rows outside this set are ignored, so a later refinement does not change
    the progress line of an earlier config. Ids with no row are ``absent``.
    """
    counts = {
        "pending": 0,
        "running": 0,
        "checkpoint": 0,
        "completed": 0,
        "failed": 0,
        "absent": 0,
    }
    found = 0
    for start in range(0, len(run_ids), 400):
        chunk = run_ids[start : start + 400]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"SELECT status, COUNT(*) AS n FROM runs WHERE run_id IN ({placeholders}) GROUP BY status",
            chunk,
        ).fetchall()
        for row in rows:
            status = str(row["status"])
            counts[status] = counts.get(status, 0) + int(row["n"])
            found += int(row["n"])
    counts["absent"] = len(run_ids) - found
    return counts
