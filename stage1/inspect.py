"""Read-only progress, file checks, and rotation-profile plots.

Nothing in this module integrates the ODE or assigns five-state labels.
A frequency sort is applied only to a copy used for plotting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stage1.config import ScanConfig
from stage1.grid import quantize
from stage1.query import fetch_runs_by_id, load_profiles, query_runs
from stage1.scan import build_run_specs
from stage1.spec import RunSpec
from stage1.stats import sort_by_frequency, window_difference

_PRINT_LIMIT = 10


def audit_plan(config: ScanConfig, *, check_files: bool = False) -> dict:
    """Compare one config's planned run ids with the existing index.

    This does not insert rows and does not create ``index.sqlite``.
    """
    specs = build_run_specs(config)
    rows = fetch_runs_by_id([spec.run_id for spec in specs], output_dir=config.output_dir)
    counts = {
        "pending": 0,
        "running": 0,
        "checkpoint": 0,
        "completed": 0,
        "failed": 0,
        "absent": 0,
    }
    failures: list[dict] = []
    file_problems: list[dict] = []
    files_checked = 0
    files_ok = 0
    for spec in specs:
        row = rows.get(spec.run_id)
        if row is None:
            counts["absent"] += 1
            continue
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
        if status == "failed":
            failures.append(
                {
                    "run_id": spec.run_id,
                    "K": row["K"],
                    "p": row["p"],
                    "Delta": row["Delta"],
                    "trial_id": row["trial_id"],
                    "error_message": row["error_message"],
                }
            )
        if check_files and status == "completed":
            files_checked += 1
            problems = _completed_file_problems(spec, row)
            if problems:
                file_problems.append({"run_id": spec.run_id, "problems": problems})
            else:
                files_ok += 1
    unfinished = len(specs) - counts.get("completed", 0)
    files_complete = (not check_files) or files_checked == counts.get("completed", 0)
    return {
        "planned": len(specs),
        "output_dir": str(config.output_dir),
        "counts": counts,
        "failures": failures,
        "files_checked": files_checked,
        "files_ok": files_ok,
        "files_bad": len(file_problems),
        "file_problems": file_problems,
        "check_files": bool(check_files),
        "complete": unfinished == 0 and not file_problems and files_complete,
    }


def format_audit(report: dict) -> str:
    counts = report["counts"]
    lines = [
        f"stage1 inspect: planned={report['planned']} output={report['output_dir']}",
        (
            f"completed={counts.get('completed', 0)} pending={counts.get('pending', 0)} "
            f"running={counts.get('running', 0)} checkpoint={counts.get('checkpoint', 0)} "
            f"failed={counts.get('failed', 0)} absent={counts.get('absent', 0)}"
        ),
        "completed run_ids are skipped when this config is scanned again",
    ]
    extra = sorted(key for key in counts if key not in {
        "pending",
        "running",
        "checkpoint",
        "completed",
        "failed",
        "absent",
    })
    for key in extra:
        lines.append(f"{key}={counts[key]}")
    if report["check_files"]:
        lines.append(
            f"files_checked={report['files_checked']} files_ok={report['files_ok']} "
            f"files_bad={report['files_bad']}"
        )
    else:
        lines.append("file_check=index_only")
    for item in report["failures"][:_PRINT_LIMIT]:
        message = "" if item["error_message"] is None else str(item["error_message"])
        lines.append(
            f"failed K={item['K']} p={item['p']} Delta={item['Delta']} "
            f"trial={item['trial_id']} run_id={item['run_id']} error={message[:200]}"
        )
    hidden_failures = len(report["failures"]) - _PRINT_LIMIT
    if hidden_failures > 0:
        lines.append(f"failed_not_shown={hidden_failures}")
    for item in report["file_problems"][:_PRINT_LIMIT]:
        lines.append(f"file_problem run_id={item['run_id']} {'; '.join(item['problems'])}")
    hidden_files = len(report["file_problems"]) - _PRINT_LIMIT
    if hidden_files > 0:
        lines.append(f"file_problems_not_shown={hidden_files}")
    return "\n".join(lines)


def selected_rows(
    config: ScanConfig,
    *,
    K: float | None = None,
    p: float | None = None,
    Delta: float | None = None,
    trial_id: int | None = None,
    run_id: str | None = None,
) -> list[dict]:
    if run_id:
        return query_runs(run_id=run_id, output_dir=config.output_dir)
    return query_runs(
        K=K,
        p=p,
        Delta=Delta,
        N=config.N,
        trial_id=trial_id,
        output_dir=config.output_dir,
    )


def compare_observation_windows(record: dict) -> list[dict]:
    """Pairwise absolute differences between saved windows.

    The result is descriptive. It does not assign a dynamical state.
    """
    rotation = np.asarray(record["rotation_numbers"], dtype=np.float64)
    starts = np.asarray(record["window_start"], dtype=np.float64)
    ends = np.asarray(record["window_end"], dtype=np.float64)
    if rotation.ndim != 2 or rotation.shape[0] != starts.size or starts.size != ends.size:
        raise ValueError("rotation_numbers rows must match the stored windows")
    comparisons = []
    for earlier in range(rotation.shape[0]):
        for later in range(earlier + 1, rotation.shape[0]):
            difference = window_difference(rotation[earlier], rotation[later])
            delta = difference["difference"]
            comparisons.append(
                {
                    "window_a": [float(starts[earlier]), float(ends[earlier])],
                    "window_b": [float(starts[later]), float(ends[later])],
                    "max_abs": difference["max_abs"],
                    "mean_abs": float(np.mean(np.abs(delta))),
                    "interpretation": "descriptive",
                }
            )
    return comparisons


def format_profile_report(record: dict) -> str:
    parameters = record["parameters"]
    realization = record["realization"]
    lines = [
        (
            f"run_id={record['run_id']} K={parameters['K']} p={parameters['p']} "
            f"Delta={parameters['Delta']} N={parameters['N']} trial={realization['trial_id']}"
        ),
        (
            f"network_seed={realization['network_seed']} "
            f"initial_phase_seed={realization['initial_phase_seed']} "
            f"frequency_seed={realization['frequency_seed']} "
            f"network_realization_id={realization['network_realization_id']} "
            f"initial_condition_id={realization['initial_condition_id']} "
            f"paired_group_id={realization['paired_group_id']} "
            f"mode={realization['randomness_mode']}"
        ),
        (
            f"omega={tuple(record['omega'].shape)} "
            f"rotation_numbers={tuple(record['rotation_numbers'].shape)}"
        ),
    ]
    for item in compare_observation_windows(record):
        lines.append(
            "windows "
            f"{_window_label(item['window_a'])} vs {_window_label(item['window_b'])} "
            f"max_abs={item['max_abs']:.6g} mean_abs={item['mean_abs']:.6g}"
        )
    return "\n".join(lines)


def plot_rotation_profile(record: dict, path: str | Path) -> Path:
    """Plot rho against omega for every stored window.

    Sorting is applied to a copy so the saved oscillator order stays unchanged.
    """
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sort_by_frequency(record["omega"], record["rotation_numbers"])
    omega = ordered["omega"]
    rotation = ordered["arrays"][0]
    starts = np.asarray(record["window_start"], dtype=np.float64)
    ends = np.asarray(record["window_end"], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for index in range(rotation.shape[0]):
        label = _window_label([float(starts[index]), float(ends[index])])
        axis.plot(omega, rotation[index], linewidth=1.0, label=label)
    parameters = record["parameters"]
    realization = record["realization"]
    axis.set_xlabel(r"$\omega$")
    axis.set_ylabel(r"$\rho$")
    axis.set_title(
        f"K={parameters['K']:g}  p={parameters['p']:g}  "
        f"Delta={parameters['Delta']:g}  trial={realization['trial_id']}"
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def _completed_file_problems(spec: RunSpec, row: dict) -> list[str]:
    problems: list[str] = []
    npz_path = Path(row["output_npz"])
    json_path = Path(row["output_json"])
    if not npz_path.is_file():
        problems.append(f"missing file {npz_path}")
    if not json_path.is_file():
        problems.append(f"missing file {json_path}")
    if problems:
        return problems
    try:
        loaded = load_profiles([spec.run_id], output_dir=row["output_dir"])[0]
    except Exception as exc:
        return [f"{exc.__class__.__name__}: {exc}"]
    metadata = loaded["metadata"]
    parameters = loaded["parameters"]
    realization = loaded["realization"]
    if metadata.get("run_id") != spec.run_id:
        problems.append("metadata run_id mismatch")
    if parameters["N"] != spec.N:
        problems.append("N mismatch")
    if quantize(parameters["K"]) != spec.k_units:
        problems.append("K mismatch")
    if quantize(parameters["p"]) != spec.p_units:
        problems.append("p mismatch")
    if quantize(parameters["Delta"]) != spec.delta_units:
        problems.append("Delta mismatch")
    if realization["trial_id"] != spec.trial_id:
        problems.append("trial_id mismatch")
    if realization["network_seed"] != spec.network_seed:
        problems.append("network_seed mismatch")
    if realization["initial_phase_seed"] != spec.initial_phase_seed:
        problems.append("initial_phase_seed mismatch")
    expected_windows = [
        [float(start).hex(), float(end).hex()] for start, end in spec.observation_windows
    ]
    if metadata.get("windows_hex") != expected_windows:
        problems.append("observation windows differ from the planned run")
    rotation = loaded["rotation_numbers"]
    if rotation.shape != (len(spec.observation_windows), spec.N):
        problems.append(
            f"rotation_numbers shape {tuple(rotation.shape)} "
            f"does not match {(len(spec.observation_windows), spec.N)}"
        )
    if loaded["omega"].shape != (spec.N,):
        problems.append("omega length does not match N")
    if float(metadata["t_final"]) < float(spec.t_end) - 1e-8:
        problems.append("t_final is before t_end")
    return problems


def _window_label(window: list[float]) -> str:
    return f"[{window[0]:g}, {window[1]:g}]"
