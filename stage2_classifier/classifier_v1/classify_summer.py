"""Label the Stage 1 pilot with the unmodified summer classifier.

The assigned state is the return value of ``classify_state`` in
``code/Ariaratnam2001.py``, using that function's default arguments.
Profiles come from the completed Stage 1 runs. W3 is the label written to
``assigned_state``. This script does not integrate the ODE and does not
rewrite the v1 result tables.
"""

from __future__ import annotations

import csv
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from Ariaratnam2001 import (  # noqa: E402
    classify_state,
    find_locking_plateaus,
    has_consecutive_zero_rotation,
)
from classify import (  # noqa: E402
    WINDOWS,
    plot_distribution,
    plot_phase_comparison,
    plot_phase_diagrams,
    window_row,
    write_csv,
)
from stage1.query import query_runs  # noqa: E402

SUMMER_LABEL = "Summer classifier"
SUMMER_SOURCE = "code/Ariaratnam2001.py:classify_state"
V1_RESULTS = HERE / "classification_results.csv"
STATES = ("Death", "Locking", "Partial Death", "Partial Locking", "Incoherence")


def summer_defaults() -> dict:
    defaults = {}
    for name, param in inspect.signature(classify_state).parameters.items():
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default
    return defaults


def profile_features(omega: np.ndarray, rho: np.ndarray, defaults: dict) -> dict:
    plateaus = find_locking_plateaus(omega, rho)
    nonzero = [item for item in plateaus if abs(item["rho_center"]) >= defaults["tol_death"]]
    largest = max((item["fraction"] for item in nonzero), default=0.0)
    return {
        "zero_fraction": float(np.mean(np.abs(rho) < defaults["tol_death"])),
        "has_consecutive_zero": int(has_consecutive_zero_rotation(omega, rho)),
        "rho_range": float(rho.max() - rho.min()),
        "plateau_count": len(plateaus),
        "nonzero_plateau_count": len(nonzero),
        "largest_nonzero_fraction": float(largest),
        "identity_fraction": float(np.mean(np.abs(rho - omega) < defaults["tol_identity"])),
    }


def classify_pilot(defaults: dict) -> list[dict]:
    rows = []
    for run in query_runs(status="completed"):
        record = {
            "run_id": run["run_id"],
            "classifier": SUMMER_LABEL,
            "classifier_source": SUMMER_SOURCE,
            "primary_window": "W3",
            "K": run["K"],
            "p": run["p"],
            "Delta": run["Delta"],
            "N": run["N"],
            "execution_status": "ok",
            "error_message": "",
        }
        try:
            with np.load(run["output_npz"], allow_pickle=False) as data:
                omega = np.array(data["omega"], dtype=np.float64)
                rotation = np.array(data["rotation_numbers"], dtype=np.float64)
                starts = np.array(data["window_start"], dtype=np.float64)
                ends = np.array(data["window_end"], dtype=np.float64)
            if np.any(np.diff(omega) < 0):
                order = np.argsort(omega, kind="mergesort")
                omega = omega[order]
                rotation = rotation[:, order]
            states = {}
            features = {}
            for name, start, end in WINDOWS:
                rho = rotation[window_row(starts, ends, (start, end))]
                states[name] = classify_state(omega, rho)
                features[name] = profile_features(omega, rho, defaults)
            primary = features["W3"]
            record.update(
                {
                    "assigned_state": states["W3"],
                    "state_w1": states["W1"],
                    "state_w2": states["W2"],
                    "state_w3": states["W3"],
                    **primary,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "assigned_state": "",
                    "state_w1": "",
                    "state_w2": "",
                    "state_w3": "",
                    "zero_fraction": "",
                    "has_consecutive_zero": "",
                    "rho_range": "",
                    "plateau_count": "",
                    "nonzero_plateau_count": "",
                    "largest_nonzero_fraction": "",
                    "identity_fraction": "",
                    "execution_status": "failed",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(record)
    return rows


def summary_rows(results: list[dict]) -> list[dict]:
    ok = [row for row in results if row["execution_status"] == "ok"]
    total = len(ok)
    counts = Counter(row["assigned_state"] for row in ok)
    output = []
    for state in STATES:
        count = counts[state]
        output.append(
            {
                "classifier": SUMMER_LABEL,
                "scope": "all",
                "p": "",
                "Delta": "",
                "assigned_state": state,
                "count": count,
                "fraction": count / total if total else 0.0,
            }
        )
    grouped = Counter((row["p"], row["Delta"], row["assigned_state"]) for row in ok)
    sizes = Counter((row["p"], row["Delta"]) for row in ok)
    for p_value, delta, state in sorted(grouped):
        count = grouped[(p_value, delta, state)]
        output.append(
            {
                "classifier": SUMMER_LABEL,
                "scope": "p_delta",
                "p": p_value,
                "Delta": delta,
                "assigned_state": state,
                "count": count,
                "fraction": count / sizes[(p_value, delta)],
            }
        )
    return output


def comparison_rows(results: list[dict]) -> tuple[list[dict], list[dict]]:
    v1 = {}
    if V1_RESULTS.is_file():
        with V1_RESULTS.open() as handle:
            for row in csv.DictReader(handle):
                v1[row["run_id"]] = row.get("assigned_state", "")
    compared = []
    for row in results:
        v1_state = v1.get(row["run_id"], "")
        summer_state = row["assigned_state"]
        changed = int(bool(v1_state) and bool(summer_state) and v1_state != summer_state)
        compared.append(
            {
                "run_id": row["run_id"],
                "K": row["K"],
                "p": row["p"],
                "Delta": row["Delta"],
                "N": row["N"],
                "v1_assigned_state": v1_state,
                "summer_assigned_state": summer_state,
                "changed": changed,
                "summer_state_w1": row["state_w1"],
                "summer_state_w2": row["state_w2"],
                "summer_state_w3": row["state_w3"],
                "execution_status": row["execution_status"],
            }
        )
    transitions = Counter(
        (row["v1_assigned_state"], row["summer_assigned_state"])
        for row in compared
        if row["execution_status"] == "ok" and row["v1_assigned_state"] and row["summer_assigned_state"]
    )
    change_summary = [
        {
            "classifier_pair": "v1.0.0-pilot vs Summer classifier",
            "v1_assigned_state": v1_state,
            "summer_assigned_state": summer_state,
            "count": count,
        }
        for (v1_state, summer_state), count in sorted(transitions.items())
    ]
    return compared, change_summary


def main() -> None:
    defaults = summer_defaults()
    (HERE / "summer_classifier_parameters.json").write_text(
        json.dumps(
            {
                "classifier": SUMMER_LABEL,
                "source": SUMMER_SOURCE,
                "priority": "Death -> Partial Death -> Locking -> Partial Locking -> Incoherence",
                "primary_window": "W3",
                "defaults": defaults,
            },
            indent=2,
        )
        + "\n"
    )
    figure_dir = HERE / "figures"
    figure_dir.mkdir(exist_ok=True)
    results = classify_pilot(defaults)
    write_csv(HERE / "classification_results_summer.csv", results)
    write_csv(HERE / "state_summary_summer.csv", summary_rows(results))
    compared, change_summary = comparison_rows(results)
    write_csv(HERE / "classification_comparison_v1_summer.csv", compared)
    write_csv(HERE / "classifier_change_summary.csv", change_summary)
    plot_distribution(
        results,
        figure_dir / "state_distribution_summer.png",
        title="Summer classifier\nW3 state counts on the Stage 1 pilot",
    )
    plot_phase_diagrams(
        results,
        figure_dir,
        label=SUMMER_LABEL,
        filename_prefix="phase_diagram_summer",
    )
    if V1_RESULTS.is_file():
        with V1_RESULTS.open() as handle:
            v1_results = list(csv.DictReader(handle))
        plot_phase_comparison(v1_results, results, figure_dir, summer_label=SUMMER_LABEL)
    ok = sum(row["execution_status"] == "ok" for row in results)
    failed = sum(row["execution_status"] == "failed" for row in results)
    changed = sum(row["changed"] for row in compared)
    print(
        json.dumps(
            {
                "classifier": SUMMER_LABEL,
                "classified": ok,
                "failed": failed,
                "changed_from_v1": changed,
            }
        )
    )


if __name__ == "__main__":
    main()
