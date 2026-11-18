"""Label the new Stage 1 cells with Classifier v2.0.0 and redraw the overlays.

The archived classifier tables and figures are not modified. Existing labels
are reused. Only cells absent from that table are classified.
"""

from __future__ import annotations

import csv
import os
import sqlite3
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-extend-phase")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLASSIFIER = ROOT / "stage2_classifier" / "classifier_v2"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLASSIFIER))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "penalty_calibration"))

from classify import WINDOWS, classify_window, load_config, window_row  # noqa: E402
from full_grid_boundaries import draw_phase, read_theory_csv  # noqa: E402
from method_c_separated import load_method_c  # noqa: E402

ARCHIVE = CLASSIFIER / "classification_results.csv"
INDEX = ROOT / "stage1_runs" / "index.sqlite"
OUT_CSV = HERE / "extended_classification.csv"
TARGET_P = (0.2, 0.4, 0.6)


def key_of(probability: float, delta: float, coupling: float) -> tuple[int, int, int]:
    return (round(probability, 10), round(delta, 10), round(coupling, 10))


def load_extended() -> list[dict]:
    if not OUT_CSV.is_file():
        return []
    rows = []
    with OUT_CSV.open() as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "p": float(raw["p"]),
                    "delta": float(raw["Delta"]),
                    "K": float(raw["K"]),
                    "state": raw["assigned_state"],
                    "run_id": raw["run_id"],
                }
            )
    return rows


def load_archive() -> list[dict]:
    rows = []
    with ARCHIVE.open() as handle:
        for raw in csv.DictReader(handle):
            if raw["execution_status"] != "ok" or raw["trial_id"] != "0":
                continue
            if int(float(raw["N"])) != 1000:
                continue
            rows.append(
                {
                    "p": float(raw["p"]),
                    "delta": float(raw["Delta"]),
                    "K": float(raw["K"]),
                    "state": raw["assigned_state"],
                    "run_id": raw["run_id"],
                }
            )
    return rows


def classify_new(archive_keys: set[tuple[int, int, int]]) -> list[dict]:
    warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
    config = load_config()
    pilot, cost_audit = load_method_c()
    connection = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True)
    runs = connection.execute(
        """
        SELECT run_id, p, delta, K, output_npz
        FROM runs
        WHERE status='completed' AND trial_id=0 AND N=1000
          AND p IN (0.2, 0.4, 0.6)
        ORDER BY p, delta, K
        """
    ).fetchall()
    pending = [row for row in runs if key_of(row[1], row[2], row[3]) not in archive_keys]
    print(f"new cells {len(pending)}", flush=True)
    labeled = []
    for index, (run_id, probability, delta, coupling, relative) in enumerate(pending, start=1):
        path = ROOT / "stage1_runs" / relative
        with np.load(path, allow_pickle=False) as data:
            omega = np.array(data["omega"], dtype=np.float64)
            rotation = np.array(data["rotation_numbers"], dtype=np.float64)
            starts = np.array(data["window_start"], dtype=np.float64)
            ends = np.array(data["window_end"], dtype=np.float64)
        if np.any(np.diff(omega) < 0):
            order = np.argsort(omega, kind="mergesort")
            omega = omega[order]
            rotation = rotation[:, order]
        states = {}
        for name, start, end in WINDOWS:
            rho = rotation[window_row(starts, ends, (start, end))]
            states[name] = classify_window(cost_audit, pilot, omega, rho, config)["state"]
        labeled.append(
            {
                "run_id": run_id,
                "p": float(probability),
                "Delta": float(delta),
                "K": float(coupling),
                "N": 1000,
                "trial_id": 0,
                "assigned_state": states["W3"],
                "state_w1": states["W1"],
                "state_w2": states["W2"],
                "state_w3": states["W3"],
                "execution_status": "ok",
                "classifier_version": "v2.0.0",
                "alpha": 2,
                "beta": 1,
            }
        )
        if index % 400 == 0 or index == len(pending):
            print(f"classified {index}/{len(pending)}", flush=True)
    return labeled


def write_labels(rows: list[dict], *, append: bool) -> None:
    fields = [
        "run_id",
        "p",
        "Delta",
        "K",
        "N",
        "trial_id",
        "assigned_state",
        "state_w1",
        "state_w2",
        "state_w3",
        "execution_status",
        "classifier_version",
        "alpha",
        "beta",
    ]
    mode = "a" if append and OUT_CSV.is_file() else "w"
    with OUT_CSV.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    archive = load_archive()
    extended = load_extended()
    known = {key_of(row["p"], row["delta"], row["K"]) for row in archive}
    known.update(key_of(row["p"], row["delta"], row["K"]) for row in extended)
    fresh = classify_new(known)
    if fresh:
        write_labels(fresh, append=True)
    combined = list(archive)
    combined.extend(extended)
    combined.extend(
        {"p": row["p"], "delta": row["Delta"], "K": row["K"], "state": row["assigned_state"], "run_id": row["run_id"]}
        for row in fresh
    )
    theory = read_theory_csv()
    deltas = np.array(sorted({row["delta"] for row in theory if abs(row["p"] - 0.2) < 1e-12}))
    draw_phase(combined, theory, 0.2, deltas, show_states=True)
    print("drew p=0.2", flush=True)


if __name__ == "__main__":
    main()
