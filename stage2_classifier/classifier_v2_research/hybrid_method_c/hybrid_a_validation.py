"""Stage 1 screen of archived profiles with Hybrid A only.

Hybrid A is imported from ``hybrid_method_c``. Constant segments use archived
Rule 3. Linear segments must also have ``S_I <= 0.0001``. Segments are not
merged. The Winfree ODE is not rerun, and the Stage 1 index is opened
read-only.

The segmentation itself is ``cost_audit.dp_penalized`` with the same penalties
as ``segment_separated``. A startup check compares that call with
``segment_profile`` before the scan.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np

import hybrid_method_c as hybrid
from method_c_separated import separated_penalties
from stage1.storage import resolve_stored_path

TAU = 1e-4
TAU_GRID = (5e-5, 1e-4, 2e-4, 5e-4)
WINDOWS = ((8000.0, 10000.0), (10000.0, 12000.0), (12000.0, 14000.0))
HERE = Path(__file__).resolve().parent
DATA = hybrid.DATA
SUMMARY_PATH = HERE / "hybrid_a_stage1_summary.json"


def segment_fast(cost_audit, pilot, omega: np.ndarray, rho: np.ndarray):
    sigma = float(pilot.estimate_sigma(rho))
    _unit, pen0, pen1 = separated_penalties(rho.size, sigma * sigma, hybrid.ALPHA, hybrid.BETA)
    segments, _objective = cost_audit.dp_penalized(omega, rho, pen0, pen1, hybrid.MIN_LEN)
    return [(int(a), int(b), int(c)) for a, b, c in segments]


def window_matrix(starts: np.ndarray, ends: np.ndarray, rotation: np.ndarray) -> np.ndarray | None:
    order = []
    for left, right in WINDOWS:
        hits = [
            index
            for index, (start, end) in enumerate(zip(starts, ends))
            if abs(float(start) - left) < 1e-6 and abs(float(end) - right) < 1e-6
        ]
        if len(hits) != 1:
            return None
        order.append(hits[0])
    return np.asarray(rotation[order], dtype=np.float64)


def load_run(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    with np.load(path, allow_pickle=False) as data:
        if "omega" not in data or "rotation_numbers" not in data:
            return None
        omega = np.array(data["omega"], dtype=np.float64)
        rotation = np.array(data["rotation_numbers"], dtype=np.float64)
        starts = np.array(data["window_start"], dtype=np.float64)
        ends = np.array(data["window_end"], dtype=np.float64)
    matrix = window_matrix(starts, ends, rotation)
    if matrix is None or omega.ndim != 1 or matrix.shape[1] != omega.size:
        return None
    if not np.isfinite(omega).all() or not np.isfinite(matrix).all():
        return None
    if omega.size and np.any(np.diff(omega) < 0):
        order = np.argsort(omega, kind="mergesort")
        omega = omega[order]
        matrix = matrix[:, order]
    return omega, matrix


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    keys = (0, 10, 50, 90, 100)
    return {
        "n": int(array.size),
        **{f"p{key}": float(np.percentile(array, key)) for key in keys},
        "mean": float(array.mean()),
    }


def classify_window(index: hybrid.ProfileIndex) -> dict:
    accepted = hybrid.accept_plateaus(index, TAU, float("inf"), merge=False)
    constants = [item for item in accepted if item["origin"] == "constant-rule3"]
    linears = [item for item in accepted if item["origin"] == "linear-trend"]
    other = [item for item in accepted if item["origin"] not in {"constant-rule3", "linear-trend"}]
    if other:
        raise RuntimeError(f"Hybrid A returned an unexpected origin: {other[0]['origin']}")
    fail_geometry = 0
    fail_trend = 0
    candidate_s = []
    candidates = []
    for atom, mode in zip(index.atomic, index.modes):
        if mode != 1:
            continue
        if not hybrid.geometry_ok(atom):
            fail_geometry += 1
            continue
        if not hybrid.nonzero_ok(atom):
            continue
        candidate_s.append(float(atom["S"]))
        candidates.append((int(atom["n"]), float(atom["S"])))
        if not np.isfinite(atom["S"]) or atom["S"] > TAU:
            fail_trend += 1
    if len(linears) != sum(value <= TAU for value in candidate_s):
        raise RuntimeError("accepted linear count does not match the geometry-and-trend rule")
    mask = hybrid.plateau_mask(index.rho.size, accepted)
    base_mask = hybrid.plateau_mask(index.rho.size, constants)
    if np.any(base_mask & ~mask):
        raise RuntimeError("Hybrid A dropped a constant plateau")
    return {
        "constants": constants,
        "linears": linears,
        "fail_geometry": fail_geometry,
        "fail_trend": fail_trend,
        "candidate_s": candidate_s,
        "candidates": candidates,
        "state_base": hybrid.assign_state(index.rho, base_mask),
        "state_hybrid": hybrid.assign_state(index.rho, mask),
        "n_segments": len(index.segments),
        "n_constant_segments": sum(mode == 0 for mode in index.modes),
        "n_linear_segments": sum(mode == 1 for mode in index.modes),
    }


def passes_on(index: hybrid.ProfileIndex, start: int, end: int) -> tuple[bool, dict]:
    item = index.features(start, end)
    accepted = bool(
        hybrid.geometry_ok(item) and hybrid.nonzero_ok(item) and np.isfinite(item["S"]) and item["S"] <= TAU
    )
    return accepted, item


def screen_record(cost_audit, pilot, omega: np.ndarray, matrix: np.ndarray) -> dict:
    segments = segment_fast(cost_audit, pilot, omega, matrix[2])
    primary = hybrid.ProfileIndex(omega, matrix[2], segments)
    decision = classify_window(primary)
    x = omega - float(omega.mean())
    y = matrix[2] - float(matrix[2].mean())
    denom = float(np.dot(x, x) * np.dot(y, y))
    corr = float(np.dot(x, y) / np.sqrt(denom)) if denom > 0.0 else 0.0
    plateaus = []
    if decision["linears"]:
        others = [hybrid.ProfileIndex(omega, matrix[window], [(0, matrix.shape[1], 1)]) for window in (0, 1)]
        for item in decision["linears"]:
            w1_ok, w1 = passes_on(others[0], item["start"], item["end"])
            w2_ok, w2 = passes_on(others[1], item["start"], item["end"])
            members = slice(item["start"], item["end"])
            plateaus.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "n": item["n"],
                    "relative_omega_span": item["relative_omega_span"],
                    "R": item["R"],
                    "b": item["b"],
                    "S": item["S"],
                    "mean": item["mean"],
                    "w1_pass": w1_ok,
                    "w2_pass": w2_ok,
                    "w1_mean": w1["mean"],
                    "w2_mean": w2["mean"],
                    "w1_R": w1["R"],
                    "w2_R": w2["R"],
                    "w1_S": w1["S"],
                    "w2_S": w2["S"],
                    "mean_abs_w1": float(np.mean(np.abs(matrix[2][members] - matrix[0][members]))),
                    "mean_abs_w2": float(np.mean(np.abs(matrix[2][members] - matrix[1][members]))),
                }
            )
    return {
        "state_base": decision["state_base"],
        "state_hybrid": decision["state_hybrid"],
        "n_segments": decision["n_segments"],
        "n_constant_segments": decision["n_constant_segments"],
        "n_linear_segments": decision["n_linear_segments"],
        "n_constant_plateaus": len(decision["constants"]),
        "n_linear_plateaus": len(decision["linears"]),
        "n_linear_fail_geometry": decision["fail_geometry"],
        "n_linear_fail_trend": decision["fail_trend"],
        "candidate_s": decision["candidate_s"],
        "candidates": decision["candidates"],
        "rho_range": float(matrix[2].max() - matrix[2].min()),
        "abs_corr": abs(corr),
        "plateaus": plateaus,
    }


def verify_segmentation(cost_audit, pilot, rows: list[dict]) -> None:
    checked = 0
    for row in rows:
        if row["status"] != "completed":
            continue
        loaded = load_run(row["path"])
        if loaded is None:
            continue
        omega, matrix = loaded
        fast = segment_fast(cost_audit, pilot, omega, matrix[2])
        archived = hybrid.segment_profile(cost_audit, pilot, omega, matrix[2])
        if fast != archived:
            raise RuntimeError(f"fast segmentation differs from segment_profile on {row['run_id']}")
        checked += 1
        if checked == 3:
            break
    if checked < 3:
        raise RuntimeError("could not verify segmentation on three completed profiles")


def audit_rows() -> tuple[dict, list[dict]]:
    connection = sqlite3.connect(f"file:{DATA / 'index.sqlite'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    raw = [dict(row) for row in connection.execute("SELECT run_id, status, n, K, p, delta, trial_id, output_npz FROM runs")]
    connection.close()
    status_counts = Counter(row["status"] for row in raw)
    usable = []
    missing = 0
    for row in raw:
        path = resolve_stored_path(DATA, str(row["output_npz"]))
        if not path.is_file():
            missing += 1
            continue
        usable.append(
            {
                "run_id": row["run_id"],
                "status": row["status"],
                "N": int(row["n"]),
                "K": float(row["K"]),
                "p": float(row["p"]),
                "delta": float(row["delta"]),
                "trial": int(row["trial_id"]),
                "path": path,
            }
        )
    report = {
        "index_rows": len(raw),
        "status_counts": dict(status_counts),
        "missing_files": missing,
        "files_present": len(usable),
        "N_values": sorted({row["N"] for row in usable}),
        "p_values": sorted({round(row["p"], 4) for row in usable}),
    }
    return report, usable


def bucket(rows: list[dict], key: str) -> list[dict]:
    groups = {}
    for row in rows:
        value = row[key]
        label = f"{value:.4g}" if isinstance(value, float) else str(value)
        item = groups.setdefault(
            label,
            {
                "value": value,
                "runs": 0,
                "constant_plateaus": 0,
                "linear_plateaus": 0,
                "incoherence_to_partial": 0,
                "partial_to_incoherence": 0,
                "stay_incoherence": 0,
                "stay_partial": 0,
            },
        )
        item["runs"] += 1
        item["constant_plateaus"] += row["n_constant_plateaus"]
        item["linear_plateaus"] += row["n_linear_plateaus"]
        pair = (row["state_base"], row["state_hybrid"])
        if pair == ("Incoherence", "Partial Locking"):
            item["incoherence_to_partial"] += 1
        elif pair == ("Partial Locking", "Incoherence"):
            item["partial_to_incoherence"] += 1
        elif pair == ("Incoherence", "Incoherence"):
            item["stay_incoherence"] += 1
        elif pair == ("Partial Locking", "Partial Locking"):
            item["stay_partial"] += 1
    return [groups[label] for label in sorted(groups, key=lambda name: groups[name]["value"])]


def main() -> None:
    hybrid.load_archived_constants()
    pilot, cost_audit = hybrid.load_method_c()
    audit, rows = audit_rows()
    print(json.dumps(audit, indent=2), flush=True)
    verify_segmentation(cost_audit, pilot, rows)
    print("segmentation check passed", flush=True)

    transitions = Counter()
    tau_plateaus = {tau: 0 for tau in TAU_GRID}
    tau_short = {tau: 0 for tau in TAU_GRID}
    protected_changed = []
    scanned = []
    accepted = []
    started = time.perf_counter()
    invalid = 0
    for index, row in enumerate(rows, start=1):
        loaded = load_run(row["path"])
        if loaded is None:
            invalid += 1
            continue
        omega, matrix = loaded
        if matrix.shape[1] != row["N"]:
            invalid += 1
            continue
        result = screen_record(cost_audit, pilot, omega, matrix)
        pair = (result["state_base"], result["state_hybrid"])
        transitions[pair] += 1
        if pair[0] in {"Death", "Locking", "Partial Death"} and pair[0] != pair[1]:
            protected_changed.append(row["run_id"])
        if pair == ("Partial Locking", "Incoherence"):
            protected_changed.append(row["run_id"])
        record = {
            "run_id": row["run_id"],
            "status": row["status"],
            "N": row["N"],
            "p": row["p"],
            "delta": row["delta"],
            "K": row["K"],
            "trial": row["trial"],
            "state_base": result["state_base"],
            "state_hybrid": result["state_hybrid"],
            "n_segments": result["n_segments"],
            "n_constant_segments": result["n_constant_segments"],
            "n_linear_segments": result["n_linear_segments"],
            "n_constant_plateaus": result["n_constant_plateaus"],
            "n_linear_plateaus": result["n_linear_plateaus"],
            "n_linear_fail_geometry": result["n_linear_fail_geometry"],
            "n_linear_fail_trend": result["n_linear_fail_trend"],
            "rho_range": result["rho_range"],
            "abs_corr": result["abs_corr"],
            "min_candidate_s": min(result["candidate_s"]) if result["candidate_s"] else None,
        }
        scanned.append(record)
        for plateau in result["plateaus"]:
            accepted.append({"run_id": row["run_id"], "N": row["N"], "p": row["p"], "delta": row["delta"], "K": row["K"], "trial": row["trial"], **plateau})
        for length, slope in result["candidates"]:
            for tau in TAU_GRID:
                if slope <= tau:
                    tau_plateaus[tau] += 1
                    if length < 30:
                        tau_short[tau] += 1
        if index % 500 == 0 or index == len(rows):
            elapsed = time.perf_counter() - started
            print(f"screened {index}/{len(rows)} in {elapsed:.1f}s", flush=True)

    if protected_changed:
        raise RuntimeError(f"unexpected state changes: {protected_changed[:8]}")

    new_pl = [row for row in scanned if row["state_base"] == "Incoherence" and row["state_hybrid"] == "Partial Locking"]
    stay_i = [row for row in scanned if row["state_base"] == "Incoherence" and row["state_hybrid"] == "Incoherence"]
    old_pl = [row for row in scanned if row["state_base"] == "Partial Locking"]
    s_values = [item["S"] for item in accepted]
    n_values = [item["n"] for item in accepted]
    sensitivity = []
    for tau in TAU_GRID:
        gained = [
            row
            for row in scanned
            if row["state_base"] == "Incoherence" and row["min_candidate_s"] is not None and row["min_candidate_s"] <= tau
        ]
        sensitivity.append(
            {
                "tau_s": tau,
                "linear_plateaus": tau_plateaus[tau],
                "short_linear_plateaus_n_lt_30": tau_short[tau],
                "incoherence_to_partial": len(gained),
            }
        )
    below = [row["min_candidate_s"] for row in scanned if row["min_candidate_s"] is not None and row["min_candidate_s"] <= 5e-5]
    summary = {
        "tau_s": TAU,
        "merge": False,
        "audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
        "invalid_profiles": invalid,
        "transitions": {f"{left} -> {right}": count for (left, right), count in sorted(transitions.items())},
        "constant_plateaus": sum(row["n_constant_plateaus"] for row in scanned),
        "linear_plateaus": len(accepted),
        "runs_with_linear_plateau": sum(row["n_linear_plateaus"] > 0 for row in scanned),
        "linear_fail_geometry": sum(row["n_linear_fail_geometry"] for row in scanned),
        "linear_fail_trend": sum(row["n_linear_fail_trend"] for row in scanned),
        "incoherence_to_partial": len(new_pl),
        "stay_incoherence": len(stay_i),
        "previous_partial": len(old_pl),
        "plateau_n": quantiles(n_values),
        "plateau_S": quantiles(s_values),
        "plateau_R": quantiles([item["R"] for item in accepted]),
        "plateau_relative_span": quantiles([item["relative_omega_span"] for item in accepted]),
        "window_both_pass": sum(item["w1_pass"] and item["w2_pass"] for item in accepted),
        "window_w3_only": sum(not item["w1_pass"] and not item["w2_pass"] for item in accepted),
        "new_pl_median_segments": float(np.median([row["n_segments"] for row in new_pl])) if new_pl else None,
        "stay_i_median_segments": float(np.median([row["n_segments"] for row in stay_i])) if stay_i else None,
        "new_pl_median_abs_corr": float(np.median([row["abs_corr"] for row in new_pl])) if new_pl else None,
        "new_pl_median_range": float(np.median([row["rho_range"] for row in new_pl])) if new_pl else None,
        "runs_with_candidate_s_at_most_5e-5": len(below),
        "by_p": bucket(scanned, "p"),
        "by_delta": bucket(scanned, "delta"),
        "by_K": bucket(scanned, "K"),
        "by_N": bucket(scanned, "N"),
        "sensitivity_state": sensitivity,
        "focus": [
            {
                "p": row["p"],
                "delta": row["delta"],
                "K": row["K"],
                "state_base": row["state_base"],
                "state_hybrid": row["state_hybrid"],
                "n_segments": row["n_segments"],
                "n_linear_plateaus": row["n_linear_plateaus"],
                "n_constant_plateaus": row["n_constant_plateaus"],
            }
            for row in scanned
            if round(row["p"], 4) == 1.0
            and (
                (round(row["delta"], 4) == 0.2 and round(row["K"], 4) in {0.6, 0.65, 0.7, 0.75})
                or (round(row["delta"], 4) == 0.15 and round(row["K"], 4) == 0.4)
            )
        ],
    }
    # Sampling frame for a later Stage 2. Plateau geometry stays in memory only
    # long enough to write this one summary.
    summary["new_partial_ids"] = [row["run_id"] for row in new_pl]
    summary["previous_partial_ids"] = [row["run_id"] for row in old_pl]
    summary["near_tau_ids"] = [
        row["run_id"]
        for row in scanned
        if row["min_candidate_s"] is not None and 5e-5 <= row["min_candidate_s"] <= 2e-4
    ]
    summary["drift_control_ids"] = [
        row["run_id"]
        for row in stay_i
        if row["abs_corr"] >= 0.98 and row["rho_range"] >= 0.05 and row["n_linear_plateaus"] == 0
    ]
    summary["plateau_like_incoherence_ids"] = [
        row["run_id"]
        for row in stay_i
        if row["min_candidate_s"] is not None and row["min_candidate_s"] <= 5e-4
    ]
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    printable = {
        key: value
        for key, value in summary.items()
        if key
        not in {
            "new_partial_ids",
            "previous_partial_ids",
            "near_tau_ids",
            "drift_control_ids",
            "plateau_like_incoherence_ids",
        }
    }
    printable["category_counts"] = {
        "new_partial": len(summary["new_partial_ids"]),
        "previous_partial": len(summary["previous_partial_ids"]),
        "near_tau": len(summary["near_tau_ids"]),
        "drift_controls": len(summary["drift_control_ids"]),
        "plateau_like_incoherence": len(summary["plateau_like_incoherence_ids"]),
    }
    print(json.dumps(printable, indent=2), flush=True)
    print(f"wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    import sys

    if "--stage2" in sys.argv:
        from hybrid_a_stage2 import main as stage2_main

        stage2_main()
    else:
        main()
