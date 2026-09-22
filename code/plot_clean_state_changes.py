#!/usr/bin/env python3
"""
Plot raw vs. cleaned rotation profiles for groups listed in tables/*.csv.

Reads the four clean_state_changes_*.csv files under tables/, loads the
corresponding trial npz files, and saves comparison figures to
figures/clean_state_changes/<dataset>/.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_rotation_vs_frequency import STATE_COLORS
from project_paths import PROJECT_ROOT
from rotation_cleaning import CleanParams, clean_rotation_profile, cleaning_label

DEFAULT_TABLE_DIR = PROJECT_ROOT / "tables"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures" / "clean_state_changes"

DEFAULT_CSV_FILES = (
    "clean_state_changes_raw_trials_c_rk45_monotone.csv",
    "clean_state_changes_raw_trials_c_monotone.csv",
    "clean_state_changes_raw_trials_rk4_monotone.csv",
    "clean_state_changes_raw_trials_monotone.csv",
)

CSV_RAW_DIR_RULES: tuple[tuple[str, Path], ...] = (
    ("raw_trials_c_rk45", PROJECT_ROOT / "cpp" / "raw_trials_c_rk45"),
    ("raw_trials_c", PROJECT_ROOT / "cpp" / "raw_trials_c"),
    ("raw_trials_rk4", PROJECT_ROOT / "raw_trials_rk4"),
    ("raw_trials", PROJECT_ROOT / "raw_trials"),
)


@dataclass(frozen=True)
class ChangedGroup:
    K: float
    delta: float
    p: float
    trial: int
    state_raw: str
    state_cleaned: str
    npz_file: str
    csv_path: Path
    dataset_tag: str
    raw_dir: Path


def dataset_tag_from_csv(csv_path: Path) -> str:
    """e.g. clean_state_changes_raw_trials_rk4_monotone.csv -> raw_trials_rk4"""
    match = re.match(r"clean_state_changes_(.+)_monotone\.csv$", csv_path.name)
    if match:
        return match.group(1)
    return csv_path.stem.removeprefix("clean_state_changes_")


def raw_dir_for_dataset_tag(tag: str) -> Path:
    for key, raw_dir in CSV_RAW_DIR_RULES:
        if tag == key:
            return raw_dir
    raise ValueError(f"Unknown dataset tag {tag!r} from CSV filename")


def clean_params_for_csv(csv_path: Path) -> CleanParams:
    name = csv_path.name
    if "monotone_smooth" in name:
        match = re.search(r"monotone_smooth_ma(\d+)", name)
        window = int(match.group(1)) if match else 5
        return CleanParams(method="monotone_smooth", smooth_window=window)
    return CleanParams(method="monotone")


def resolve_npz_path(raw_dir: Path, npz_file: str) -> Path:
    path = raw_dir / npz_file
    if path.exists():
        return path
    legacy = re.sub(
        r"delta_(\d+\.\d{3})_",
        lambda m: f"delta_{float(m.group(1)):.2f}_",
        npz_file,
    )
    legacy_path = raw_dir / legacy
    if legacy_path.exists():
        return legacy_path
    return path


def load_changed_groups(csv_path: Path) -> list[ChangedGroup]:
    dataset_tag = dataset_tag_from_csv(csv_path)
    raw_dir = raw_dir_for_dataset_tag(dataset_tag)
    groups: list[ChangedGroup] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            groups.append(
                ChangedGroup(
                    K=float(row["K"]),
                    delta=float(row["delta"]),
                    p=float(row["p"]),
                    trial=int(row["trial"]),
                    state_raw=row["state_raw"],
                    state_cleaned=row["state_cleaned"],
                    npz_file=row["npz_file"],
                    csv_path=csv_path,
                    dataset_tag=dataset_tag,
                    raw_dir=raw_dir,
                )
            )
    return groups


def plot_raw_vs_cleaned(
    group: ChangedGroup,
    clean_params: CleanParams,
    figure_dir: Path,
    dpi: int,
) -> Path | None:
    npz_path = resolve_npz_path(group.raw_dir, group.npz_file)
    if not npz_path.exists():
        print(
            f"Skip K={group.K:.2f}, delta={group.delta:.3f}, p={group.p:.2f}: "
            f"missing {npz_path}",
            file=sys.stderr,
        )
        return None

    with np.load(npz_path, allow_pickle=True) as data:
        omega = np.asarray(data["omega"], dtype=float)
        rotation_raw = np.asarray(data["rotation_number"], dtype=float)

    rotation_clean = clean_rotation_profile(omega, rotation_raw, clean_params)
    order = np.argsort(omega)
    omega_sorted = omega[order]
    raw_sorted = rotation_raw[order]
    clean_sorted = rotation_clean[order]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        omega_sorted,
        raw_sorted,
        s=8,
        alpha=0.35,
        color="#999999",
        marker="x",
        label=f"raw ({group.state_raw})",
    )
    clean_color = STATE_COLORS.get(group.state_cleaned, "#2166ac")
    ax.scatter(
        omega_sorted,
        clean_sorted,
        s=10,
        alpha=0.65,
        color=clean_color,
        label=f"cleaned ({group.state_cleaned})",
    )

    ax.set_xlabel(r"Natural frequency $\omega_i$")
    ax.set_ylabel(r"Rotation number $\rho_i$")
    ax.set_title(
        "Rotation number vs. natural frequency (clean changes state)\n"
        f"$K={group.K:.2f}$, $\\delta={group.delta:.3f}$, "
        f"$p={group.p:.2f}$, trial={group.trial:02d}\n"
        f"{group.state_raw} $\\rightarrow$ {group.state_cleaned} "
        f"({cleaning_label(clean_params)})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out_dir = figure_dir / group.dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / f"{Path(group.npz_file).stem}.png"
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def discover_csv_files(table_dir: Path, names: list[str] | None = None) -> list[Path]:
    if names:
        paths = [table_dir / name for name in names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            missing_list = "\n  ".join(str(p) for p in missing)
            raise FileNotFoundError(f"CSV file(s) not found:\n  {missing_list}")
        return paths

    paths = sorted(table_dir.glob("clean_state_changes_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No clean_state_changes_*.csv in {table_dir}/")
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw vs. cleaned profiles for groups listed in tables/*.csv."
        )
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help=f"Directory with state-change CSV files (default: {DEFAULT_TABLE_DIR.name}/)",
    )
    parser.add_argument(
        "--csv",
        nargs="*",
        default=None,
        metavar="FILE",
        help=(
            "Specific CSV filenames under tables-dir "
            f"(default: all {len(DEFAULT_CSV_FILES)} known files)"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Root output directory for figures (default: {DEFAULT_FIGURE_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution (default: 150)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    table_dir = args.tables_dir.resolve()
    figure_dir = args.figure_dir.resolve()

    if not table_dir.exists():
        print(f"error: tables directory not found: {table_dir}", file=sys.stderr)
        return 1

    csv_names = args.csv if args.csv is not None else list(DEFAULT_CSV_FILES)
    csv_paths = discover_csv_files(table_dir, csv_names)

    total_saved = 0
    total_skipped = 0

    print(f"Tables:  {table_dir}/")
    print(f"Figures: {figure_dir}/")
    print()

    for csv_path in csv_paths:
        groups = load_changed_groups(csv_path)
        clean_params = clean_params_for_csv(csv_path)
        print(
            f"=== {csv_path.name}: {len(groups)} changed group(s), "
            f"cleaning={cleaning_label(clean_params)} ==="
        )
        saved = 0
        skipped = 0
        for group in groups:
            figure_path = plot_raw_vs_cleaned(
                group, clean_params, figure_dir, args.dpi
            )
            if figure_path is None:
                skipped += 1
                continue
            saved += 1
            print(f"  Saved {figure_path.relative_to(figure_dir)}")
        print(f"  done: {saved} saved, {skipped} skipped")
        print()
        total_saved += saved
        total_skipped += skipped

    print(f"All done: {total_saved} figure(s) saved, {total_skipped} skipped")
    return 0 if total_skipped == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
