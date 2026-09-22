#!/usr/bin/env python3
"""
Compare phase-state classification before and after rotation-number cleaning.

Scans every (K, delta, p) cell in a raw_trials directory and reports:
  - how many groups change state after cleaning
  - which (K, delta, p, trial) groups changed, and raw -> cleaned states
  - optionally writes changed groups to a CSV table under tables/

Default directory is raw_trials_rk4/ (21×21 grid, p=0.8 → 441 groups).
Uses the same classify_state() and clean_rotation_profile() rules as
plot_rotation_vs_frequency.py.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_rotation_vs_frequency import ClassifyParams, classify_state
from project_paths import PROJECT_ROOT
from rotation_cleaning import CleanParams, clean_rotation_profile, cleaning_label

DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "raw_trials_rk4"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "tables"

STATE_CHANGE_TABLE_COLUMNS = (
    "K",
    "delta",
    "p",
    "trial",
    "state_raw",
    "state_cleaned",
    "npz_file",
)


@dataclass(frozen=True)
class StateChange:
    K: float
    delta: float
    p: float
    trial: int
    state_raw: str
    state_cleaned: str
    path: Path


def _format_delta_filename(delta: float) -> str:
    return f"{delta:.3f}"


def trial_data_path(
    raw_dir: Path, K: float, delta: float, p: float, trial: int
) -> Path:
    return raw_dir / (
        f"K_{K:.2f}_delta_{_format_delta_filename(delta)}_p_{p:.2f}_trial_{trial:02d}.npz"
    )


def legacy_trial_data_path(
    raw_dir: Path, K: float, delta: float, p: float, trial: int
) -> Path:
    return raw_dir / (
        f"K_{K:.2f}_delta_{delta:.2f}_p_{p:.2f}_trial_{trial:02d}.npz"
    )


def resolve_trial_data_path(
    raw_dir: Path, K: float, delta: float, p: float, trial: int
) -> Path:
    path = trial_data_path(raw_dir, K, delta, p, trial)
    if path.exists():
        return path
    legacy = legacy_trial_data_path(raw_dir, K, delta, p, trial)
    if legacy.exists():
        return legacy
    return path


def discover_raw_cells(
    raw_dir: Path,
) -> dict[tuple[float, float, float], list[int]]:
    """Map (K, delta, p) -> sorted trial indices present on disk."""
    pattern = re.compile(
        r"K_([\d.]+)_delta_([\d.]+)_p_([\d.]+)_trial_(\d+)\.npz$"
    )
    cells: dict[tuple[float, float, float], list[int]] = {}
    if not raw_dir.exists():
        return cells

    for path in raw_dir.glob("*.npz"):
        match = pattern.match(path.name)
        if not match:
            continue
        key = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        trial = int(match.group(4))
        cells.setdefault(key, []).append(trial)

    for key in cells:
        cells[key] = sorted(cells[key])
    return cells


def compare_trial_state(
    path: Path,
    classify_params: ClassifyParams,
    clean_params: CleanParams,
) -> tuple[str, str]:
    """Return (state_raw, state_cleaned) for one trial npz."""
    data = np.load(path, allow_pickle=True)
    omega = np.asarray(data["omega"], dtype=float)
    rotation_number_raw = np.asarray(data["rotation_number"], dtype=float)

    state_raw = classify_state(omega, rotation_number_raw, classify_params)
    rotation_number_clean = clean_rotation_profile(
        omega, rotation_number_raw, clean_params
    )
    state_cleaned = classify_state(omega, rotation_number_clean, classify_params)
    return state_raw, state_cleaned


def scan_state_changes(
    raw_dir: Path,
    classify_params: ClassifyParams,
    clean_params: CleanParams,
    p_filter: float | None = None,
) -> tuple[list[StateChange], list[tuple[float, float, float, int, str]]]:
    """
    Scan all cells and return (changed_groups, unchanged_groups).

    unchanged_groups entries are (K, delta, p, trial, state).
    """
    raw_cells = discover_raw_cells(raw_dir)
    changed: list[StateChange] = []
    unchanged: list[tuple[float, float, float, int, str]] = []

    for (K, delta, p), trials in sorted(raw_cells.items()):
        if p_filter is not None and abs(p - p_filter) > 1e-6:
            continue
        for trial in trials:
            path = resolve_trial_data_path(raw_dir, K, delta, p, trial)
            if not path.exists():
                continue
            state_raw, state_cleaned = compare_trial_state(
                path, classify_params, clean_params
            )
            if state_raw != state_cleaned:
                changed.append(
                    StateChange(
                        K=K,
                        delta=delta,
                        p=p,
                        trial=trial,
                        state_raw=state_raw,
                        state_cleaned=state_cleaned,
                        path=path,
                    )
                )
            else:
                unchanged.append((K, delta, p, trial, state_raw))

    return changed, unchanged


def print_report(
    raw_dir: Path,
    clean_params: CleanParams,
    changed: list[StateChange],
    unchanged: list[tuple[float, float, float, int, str]],
) -> None:
    total = len(changed) + len(unchanged)
    print(f"Raw data directory: {raw_dir}/")
    print(f"Cleaning method:    {cleaning_label(clean_params)}")
    print(f"Total groups:       {total}")
    print(f"State changed:      {len(changed)}")
    print(f"State unchanged:    {len(unchanged)}")
    print()

    if changed:
        print("=== Groups where cleaning changes state ===")
        print(f"{'K':>6} {'delta':>7} {'p':>5} {'trial':>5}  {'raw':<16} -> {'cleaned'}")
        print("-" * 62)
        for rec in changed:
            print(
                f"{rec.K:6.2f} {rec.delta:7.3f} {rec.p:5.2f} {rec.trial:5d}  "
                f"{rec.state_raw:<16} -> {rec.state_cleaned}"
            )
        print()

        transitions: dict[tuple[str, str], int] = {}
        for rec in changed:
            key = (rec.state_raw, rec.state_cleaned)
            transitions[key] = transitions.get(key, 0) + 1
        print("=== Transition summary ===")
        for (raw, cleaned), count in sorted(
            transitions.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"  {raw} -> {cleaned}: {count}")
    else:
        print("No groups changed state after cleaning.")


def default_table_path(raw_dir: Path, clean_params: CleanParams) -> Path:
    """Default CSV path for changed groups under tables/."""
    raw_tag = raw_dir.name.replace("/", "_")
    return DEFAULT_TABLE_DIR / f"clean_state_changes_{raw_tag}_{cleaning_label(clean_params)}.csv"


def save_state_changes_table(
    changed: list[StateChange],
    output_path: Path,
) -> Path:
    """Write changed (K, delta, p, trial) groups and raw/cleaned states to CSV."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATE_CHANGE_TABLE_COLUMNS)
        writer.writeheader()
        for rec in changed:
            writer.writerow(
                {
                    "K": f"{rec.K:.2f}",
                    "delta": f"{rec.delta:.3f}",
                    "p": f"{rec.p:.2f}",
                    "trial": rec.trial,
                    "state_raw": rec.state_raw,
                    "state_cleaned": rec.state_cleaned,
                    "npz_file": rec.path.name,
                }
            )

    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare phase state before/after rotation-number cleaning "
            "for all (K, delta, p) groups in a raw_trials directory."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help=f"Directory with trial npz files (default: {DEFAULT_RAW_DATA_DIR.name}/)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="Only scan cells with this p value (default: all p on disk)",
    )
    parser.add_argument(
        "--clean",
        choices=("monotone", "monotone_smooth"),
        default="monotone",
        help="Cleaning method applied before re-classification (default: monotone)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Odd window for monotone_smooth (default: 5)",
    )
    parser.add_argument(
        "--lock-threshold",
        type=float,
        default=ClassifyParams.lock_threshold,
        help=f"Locking fraction threshold (default: {ClassifyParams.lock_threshold})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Save changed groups to this CSV table "
            f"(default: {DEFAULT_TABLE_DIR.name}/clean_state_changes_<raw>_<clean>.csv)"
        ),
    )
    parser.add_argument(
        "--no-save-table",
        action="store_true",
        help="Skip writing the CSV table of changed groups",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.resolve()

    if not raw_dir.exists():
        print(f"error: raw data directory not found: {raw_dir}", file=sys.stderr)
        return 1

    clean_params = CleanParams(method=args.clean, smooth_window=args.smooth_window)
    classify_params = ClassifyParams(lock_threshold=args.lock_threshold)

    changed, unchanged = scan_state_changes(
        raw_dir, classify_params, clean_params, p_filter=args.p
    )
    print_report(raw_dir, clean_params, changed, unchanged)

    if not args.no_save_table:
        table_path = (
            args.output.resolve()
            if args.output is not None
            else default_table_path(raw_dir, clean_params)
        )
        saved_path = save_state_changes_table(changed, table_path)
        print(f"Saved {len(changed)} changed group(s) to {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
