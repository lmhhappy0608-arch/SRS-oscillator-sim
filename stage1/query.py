"""Query the Stage 1 index and load one run without scanning NPZ files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage1.grid import quantize
from stage1.model import PROJECT_ROOT
from stage1.storage import connect, resolve_stored_path

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "stage1_runs"


def resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return DEFAULT_OUTPUT_DIR
    path = Path(output_dir)
    if not path.is_absolute():
        return PROJECT_ROOT / path
    return path


def query_runs(
    *,
    K: float | None = None,
    p: float | None = None,
    Delta: float | None = None,
    N: int | None = None,
    trial_id: int | None = None,
    network_seed: int | None = None,
    initial_phase_seed: int | None = None,
    frequency_seed: int | None = None,
    network_realization_id: int | None = None,
    initial_condition_id: int | None = None,
    randomness_mode: str | None = None,
    status: str | None = None,
    paired_group_id: str | None = None,
    lineage_id: str | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Return index rows for the requested parameter and seed filters.

    Parameters use the same 1e-9 quantization as the scan grid, so ``K=0.7``
    and ``K=0.70`` match the same stored point. Array payloads stay in the
    NPZ files; call :func:`load_run` for those.
    """
    directory = resolve_output_dir(output_dir)
    database = directory / "index.sqlite"
    if not database.exists():
        return []
    clauses: list[str] = []
    values: list[object] = []
    if K is not None:
        clauses.append("k_units=?")
        values.append(quantize(K))
    if p is not None:
        clauses.append("p_units=?")
        values.append(quantize(p))
    if Delta is not None:
        clauses.append("delta_units=?")
        values.append(quantize(Delta))
    if N is not None:
        clauses.append("n=?")
        values.append(int(N))
    optional = {
        "trial_id": trial_id,
        "network_seed": network_seed,
        "initial_phase_seed": initial_phase_seed,
        "frequency_seed": frequency_seed,
        "network_realization_id": network_realization_id,
        "initial_condition_id": initial_condition_id,
        "randomness_mode": randomness_mode,
        "status": status,
        "paired_group_id": paired_group_id,
        "lineage_id": lineage_id,
        "run_id": run_id,
    }
    for column, value in optional.items():
        if value is not None:
            clauses.append(f"{column}=?")
            values.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    connection = connect(directory)
    try:
        rows = connection.execute(
            f"""
            SELECT *
            FROM runs
            {where}
            ORDER BY delta_units, p_units, k_units, trial_id, run_id
            """,
            values,
        ).fetchall()
    finally:
        connection.close()
    return [_public_row(directory, row) for row in rows]


def fetch_runs_by_id(
    run_ids: list[str] | tuple[str, ...],
    output_dir: str | Path | None = None,
) -> dict[str, dict]:
    """Return index rows for the given ids.

    Missing ids are omitted. A missing database returns an empty dict and
    does not create ``index.sqlite``.
    """
    directory = resolve_output_dir(output_dir)
    database = directory / "index.sqlite"
    if not database.exists() or not run_ids:
        return {}
    connection = connect(directory)
    found: dict[str, dict] = {}
    try:
        for start in range(0, len(run_ids), 400):
            chunk = list(run_ids[start : start + 400])
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT * FROM runs WHERE run_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                public = _public_row(directory, row)
                found[str(public["run_id"])] = public
    finally:
        connection.close()
    return found


def load_profiles(
    run_ids: list[str] | tuple[str, ...],
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Load saved profiles for an external classifier.

    Each record keeps ``omega[i]`` aligned with ``rotation_numbers[:, i]``.
    The oscillator order is the saved identity ``0 .. N-1``. This function
    reads completed runs and does not integrate or assign state labels.
    The result follows the order of ``run_ids``.
    """
    if not run_ids:
        raise ValueError("run_ids is empty")
    records = []
    for run_id in run_ids:
        loaded = load_run(str(run_id), output_dir=output_dir)
        metadata = loaded["metadata"]
        records.append(
            {
                "run_id": loaded["run_id"],
                "omega": loaded["omega"],
                "oscillator_index": loaded["oscillator_index"],
                "rotation_numbers": loaded["rotation_numbers"],
                "window_start": loaded["window_start"],
                "window_end": loaded["window_end"],
                "parameters": {
                    "K": float(metadata["K"]),
                    "p": float(metadata["p"]),
                    "Delta": float(metadata["Delta"]),
                    "N": int(metadata["N"]),
                },
                "realization": {
                    "trial_id": int(metadata["trial_id"]),
                    "network_seed": int(metadata["network_seed"]),
                    "initial_phase_seed": int(metadata["initial_phase_seed"]),
                    "frequency_seed": metadata["frequency_seed"],
                    "network_realization_id": int(metadata["network_realization_id"]),
                    "initial_condition_id": int(metadata["initial_condition_id"]),
                    "paired_group_id": metadata["paired_group_id"],
                    "randomness_mode": metadata["randomness_mode"],
                    "master_seed": int(metadata["master_seed"]),
                    "seed_recipe": metadata["seed_recipe"],
                },
                "metadata": metadata,
            }
        )
    return records


def load_run(run_id: str, output_dir: str | Path | None = None) -> dict:
    """Load natural frequencies, windowed rotation numbers, and metadata.

    ``omega[i]`` and every rotation profile at index ``i`` belong to the same
    oscillator. Files are stored in that original order. ``oscillator_index``
    is ``0 .. N-1`` and is checked on load.
    """
    directory = resolve_output_dir(output_dir)
    rows = query_runs(run_id=run_id, output_dir=directory)
    if not rows:
        raise FileNotFoundError(f"run_id {run_id} is not in {directory / 'index.sqlite'}")
    row = rows[0]
    npz_path = Path(row["output_npz"])
    json_path = Path(row["output_json"])
    if not npz_path.exists():
        raise FileNotFoundError(f"missing run file {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"missing metadata {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("run_id") != run_id:
        raise RuntimeError(f"metadata run_id does not match {run_id}")
    with np.load(npz_path, allow_pickle=False) as data:
        omega = np.array(data["omega"], dtype=np.float64, copy=True)
        oscillator_index = np.array(data["oscillator_index"], dtype=np.int32, copy=True)
        rotation_numbers = np.array(data["rotation_numbers"], dtype=np.float64, copy=True)
        rotation_number = np.array(data["rotation_number"], dtype=np.float64, copy=True)
        snapshot_times = np.array(data["snapshot_times"], dtype=np.float64, copy=True)
        snapshot_theta = np.array(data["snapshot_theta"], dtype=np.float64, copy=True)
        r_times = np.array(data["r_times"], dtype=np.float64, copy=True)
        r_values = np.array(data["r_values"], dtype=np.float64, copy=True)
        theta_final = np.array(data["theta_final"], dtype=np.float64, copy=True)
        theta_grid = np.array(data["theta_grid"], dtype=np.float64, copy=True)
        t_grid = float(np.asarray(data["t_grid"]).item())
        t_final = float(np.asarray(data["t_final"]).item())
        step_index = int(np.asarray(data["step_index"]).item())
    _check_alignment(omega, oscillator_index, rotation_numbers, rotation_number, theta_final, theta_grid)
    return {
        "run_id": run_id,
        "omega": omega,
        "oscillator_index": oscillator_index,
        "rotation_numbers": rotation_numbers,
        "rotation_number": rotation_number,
        "window_start": np.asarray([item[0] for item in metadata["observation_windows"]], dtype=np.float64),
        "window_end": np.asarray([item[1] for item in metadata["observation_windows"]], dtype=np.float64),
        "order_parameter_times": r_times,
        "order_parameter": r_values,
        "snapshot_times": snapshot_times,
        "snapshot_theta": snapshot_theta,
        "theta_final": theta_final,
        "theta_grid": theta_grid,
        "t_grid": t_grid,
        "t_final": t_final,
        "step_index": step_index,
        "r_times": r_times,
        "r_values": r_values,
        "metadata": metadata,
        "output_npz": str(npz_path),
        "output_json": str(json_path),
    }


def _check_alignment(
    omega: np.ndarray,
    oscillator_index: np.ndarray,
    rotation_numbers: np.ndarray,
    rotation_number: np.ndarray,
    theta_final: np.ndarray,
    theta_grid: np.ndarray,
) -> None:
    n = int(omega.shape[0])
    if oscillator_index.shape != (n,) or np.any(oscillator_index != np.arange(n, dtype=np.int32)):
        raise RuntimeError(
            "oscillator_index is not the saved identity 0..N-1. "
            "Refusing to treat a frequency-sorted file as the original order."
        )
    if rotation_numbers.ndim != 2 or rotation_numbers.shape[1] != n:
        raise RuntimeError("rotation_numbers must have shape (windows, N) aligned with omega")
    if rotation_number.shape != (n,) or theta_final.shape != (n,) or theta_grid.shape != (n,):
        raise RuntimeError("phase or rotation profile length does not match omega")


def _public_row(output_dir: Path, row) -> dict:
    npz_path = resolve_stored_path(output_dir, str(row["output_npz"]))
    json_path = resolve_stored_path(output_dir, str(row["output_json"]))
    checkpoint = resolve_stored_path(output_dir, str(row["checkpoint_path"]))
    return {
        "run_id": row["run_id"],
        "lineage_id": row["lineage_id"],
        "status": row["status"],
        "K": float(row["K"]),
        "p": float(row["p"]),
        "Delta": float(row["delta"]),
        "N": int(row["n"]),
        "trial_id": int(row["trial_id"]),
        "network_seed": int(row["network_seed"]),
        "initial_phase_seed": int(row["initial_phase_seed"]),
        "frequency_seed": None if row["frequency_seed"] is None else int(row["frequency_seed"]),
        "frequency_rule": row["frequency_rule"],
        "network_realization_id": int(row["network_realization_id"]),
        "initial_condition_id": int(row["initial_condition_id"]),
        "randomness_mode": row["randomness_mode"],
        "paired_group_id": row["paired_group_id"],
        "network_pairing_rule": row["network_pairing_rule"],
        "phase_pairing_rule": row["phase_pairing_rule"],
        "solver": row["solver"],
        "dt": float(row["dt"]),
        "t_origin": float(row["t_origin"]),
        "t_end": float(row["t_end"]),
        "model_version": row["model_version"],
        "parent_run_id": row["parent_run_id"],
        "runtime_seconds": None if row["runtime_seconds"] is None else float(row["runtime_seconds"]),
        "error_message": row["error_message"],
        "continuable": bool(row["continuable"]),
        "output_npz": str(npz_path),
        "output_json": str(json_path),
        "checkpoint_path": str(checkpoint),
        "output_dir": str(output_dir),
    }
