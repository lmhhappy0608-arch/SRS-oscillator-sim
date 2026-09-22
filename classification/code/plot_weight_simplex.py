#!/usr/bin/env python3
"""
Plot mixture weights (w_death, w_lock, w_drift) on the 2-simplex.

One triangle per p; each (K, delta) cell is one point.
Default scan: p = 0.2, 0.4, 0.6, 0.8, 1.0. Only the old
Partial Death / Partial Locking cells. The three vertices are
drawn as reference markers. Does not refit the mixture.

Example (run from classification_method/):
  python3 code/plot_weight_simplex.py
  python3 code/plot_weight_simplex.py --p 1.0
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import project_paths  # noqa: F401
from project_paths import (
    CLASSIFICATION_ROOT,
    CURSOR_ROOT,
    DEFAULT_P_VALUES,
    DEFAULT_RAW_DIR,
    DEFAULT_SIMPLEX_FIGURE_DIR,
    DEFAULT_TABLE_DIR,
    resolve_user_path,
)
from fit_rotation_mixture import load_trial_arrays, trial_path
from plot_rotation_vs_frequency import STATE_COLORS, ClassifyParams, classify_state
from rotation_cleaning import CleanParams

PARTIAL_STATES = ("Partial Death", "Partial Locking")
CODE_TO_STATE = {
    0: "Incoherence",
    1: "Locking",
    2: "Partial Locking",
    3: "Partial Death",
    4: "Death",
}
SQRT3_OVER_2 = 0.5 * np.sqrt(3.0)
VERTEX_XY = {
    "death": np.array([0.5, SQRT3_OVER_2]),
    "lock": np.array([0.0, 0.0]),
    "drift": np.array([1.0, 0.0]),
}


def barycentric_xy(
    w_death: np.ndarray, w_lock: np.ndarray, w_drift: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Death at the top, lock bottom-left, drift bottom-right."""
    x = w_lock * 0.0 + w_drift * 1.0 + w_death * 0.5
    y = w_lock * 0.0 + w_drift * 0.0 + w_death * SQRT3_OVER_2
    return x, y


def entropy(w_death: np.ndarray, w_lock: np.ndarray, w_drift: np.ndarray) -> np.ndarray:
    stacked = np.stack((w_death, w_lock, w_drift), axis=0)
    clipped = np.clip(stacked, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=0)


def _read_weight_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "w_death" not in reader.fieldnames:
            raise ValueError(f"{path} has no w_death column")
        has_old = "old_state" in reader.fieldnames
        for raw in reader:
            rows.append(
                {
                    "K": float(raw["K"]),
                    "delta": float(raw["delta"]),
                    "p": float(raw["p"]),
                    "w_death": float(raw["w_death"]),
                    "w_lock": float(raw["w_lock"]),
                    "w_drift": float(raw["w_drift"]),
                    "old_state": raw["old_state"] if has_old else "",
                }
            )
    return rows


def _cells_from_rows(rows: list[dict[str, float | str]]) -> dict[str, np.ndarray]:
    grouped: dict[tuple[float, float], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["K"]), float(row["delta"]))].append(row)

    Ks: list[float] = []
    deltas: list[float] = []
    wd: list[float] = []
    wl: list[float] = []
    wr: list[float] = []
    states: list[str] = []
    for (K, delta), items in grouped.items():
        Ks.append(K)
        deltas.append(delta)
        wd.append(float(np.mean([item["w_death"] for item in items])))
        wl.append(float(np.mean([item["w_lock"] for item in items])))
        wr.append(float(np.mean([item["w_drift"] for item in items])))
        labels = [str(item["old_state"]) for item in items if item["old_state"]]
        if labels:
            states.append(max(set(labels), key=labels.count))
        else:
            states.append("")
    return {
        "K": np.asarray(Ks),
        "delta": np.asarray(deltas),
        "w_death": np.asarray(wd),
        "w_lock": np.asarray(wl),
        "w_drift": np.asarray(wr),
        "old_state": np.asarray(states),
    }


def _cells_from_weight_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    k_axis = np.asarray(data["K_axis"], dtype=float)
    delta_axis = np.asarray(data["delta_axis"], dtype=float)
    w_death = np.asarray(data["w_death"], dtype=float)
    w_lock = np.asarray(data["w_lock"], dtype=float)
    w_drift = np.asarray(data["w_drift"], dtype=float)
    old_code = np.asarray(data["old_code"]) if "old_code" in data.files else None
    delta_grid, k_grid = np.meshgrid(delta_axis, k_axis, indexing="ij")
    finite = np.isfinite(w_death) & np.isfinite(w_lock) & np.isfinite(w_drift)
    states = np.full(w_death.shape, "", dtype=object)
    if old_code is not None:
        for code, name in CODE_TO_STATE.items():
            states[old_code == code] = name
    return {
        "K": k_grid[finite],
        "delta": delta_grid[finite],
        "w_death": w_death[finite],
        "w_lock": w_lock[finite],
        "w_drift": w_drift[finite],
        "old_state": states[finite],
    }


def load_cells(table_dir: Path, p: float) -> tuple[dict[str, np.ndarray], Path]:
    """Prefer mixture-weight tables (they carry old five-color labels)."""
    candidates = [
        table_dir / f"mixture_weights_p_{p:.2f}.csv",
        table_dir / f"mixture_weights_p_{p:.2f}.npz",
        table_dir / f"mixture_residuals_p_{p:.2f}.csv",
        table_dir / f"mixture_residuals_p_{p:.2f}.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".csv":
            cells = _cells_from_rows(_read_weight_csv(path))
        else:
            cells = _cells_from_weight_npz(path)
        return cells, path
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No mixture weight table for p={p:.2f} in {table_dir}\n{tried}"
    )


def attach_old_states_from_raw(
    cells: dict[str, np.ndarray],
    raw_dir: Path,
    p: float,
) -> dict[str, np.ndarray]:
    """Fill old five-color labels from stored trials. No mixture refit.

    Lookup uses the trial filename (K:.2f, delta:.2f/.3f), not exact
    equality of CSV floats. npz files store linspace δ (0.025, 0.075, …)
    while names were rounded (0.03, 0.08, …).
    """
    if np.any(cells["old_state"] != ""):
        return cells
    clean_params = CleanParams(method="none")
    classify_params = ClassifyParams()
    labels = np.array(cells["old_state"], dtype=object)
    n_missing = 0
    for i, (K, delta) in enumerate(zip(cells["K"], cells["delta"])):
        names: list[str] = []
        path = trial_path(raw_dir, float(K), float(delta), float(p), trial=0)
        if not path.exists():
            n_missing += 1
            continue
        omega, rho, _meta = load_trial_arrays(path, clean_params)
        names.append(classify_state(omega, rho, classify_params))
        labels[i] = names[0]
    if n_missing:
        print(f"warning: {n_missing} cells had no matching trial npz")
    out = dict(cells)
    out["old_state"] = labels
    return out


def select_partial(
    cells: dict[str, np.ndarray],
    keep_delta_zero: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    n_all = int(cells["K"].size)
    mask = np.ones(n_all, dtype=bool)
    if not keep_delta_zero:
        mask &= cells["delta"] != 0.0
    n_after_delta = int(np.sum(mask))

    labels = cells["old_state"]
    has_labels = np.any(labels != "")
    if has_labels:
        partial = np.isin(labels, PARTIAL_STATES)
        n_partial_labeled = int(np.sum(partial & mask))
        mask &= partial
    else:
        n_partial_labeled = -1

    picked = {key: value[mask] for key, value in cells.items()}
    counts = {
        "n_cells": n_all,
        "n_after_dropping_delta0": n_after_delta,
        "n_partial": int(picked["K"].size),
        "n_partial_labeled": n_partial_labeled,
        "has_labels": int(has_labels),
    }
    return picked, counts


def draw_simplex(ax) -> None:
    triangle = np.stack(
        (VERTEX_XY["lock"], VERTEX_XY["drift"], VERTEX_XY["death"]), axis=0
    )
    ax.add_patch(
        Polygon(
            triangle,
            closed=True,
            facecolor="#f4f6f8",
            edgecolor="#1f2937",
            linewidth=1.2,
            zorder=0,
        )
    )
    ax.plot(
        [VERTEX_XY["death"][0], VERTEX_XY["drift"][0]],
        [VERTEX_XY["death"][1], VERTEX_XY["drift"][1]],
        color=STATE_COLORS["Partial Death"],
        linewidth=2.0,
        zorder=1,
    )
    ax.plot(
        [VERTEX_XY["lock"][0], VERTEX_XY["drift"][0]],
        [VERTEX_XY["lock"][1], VERTEX_XY["drift"][1]],
        color=STATE_COLORS["Partial Locking"],
        linewidth=2.0,
        zorder=1,
    )
    ax.scatter(
        [VERTEX_XY["death"][0]],
        [VERTEX_XY["death"][1]],
        s=60,
        c=STATE_COLORS["Death"],
        zorder=4,
        edgecolors="k",
        linewidths=0.4,
    )
    ax.scatter(
        [VERTEX_XY["lock"][0]],
        [VERTEX_XY["lock"][1]],
        s=60,
        c=STATE_COLORS["Locking"],
        zorder=4,
        edgecolors="k",
        linewidths=0.4,
    )
    ax.scatter(
        [VERTEX_XY["drift"][0]],
        [VERTEX_XY["drift"][1]],
        s=60,
        c=STATE_COLORS["Incoherence"],
        zorder=4,
        edgecolors="k",
        linewidths=0.4,
    )
    ax.text(
        VERTEX_XY["death"][0],
        VERTEX_XY["death"][1] + 0.045,
        r"Death $(1,0,0)$",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    ax.text(
        VERTEX_XY["lock"][0] - 0.02,
        VERTEX_XY["lock"][1] - 0.045,
        r"Lock $(0,1,0)$",
        ha="left",
        va="top",
        fontsize=11,
    )
    ax.text(
        VERTEX_XY["drift"][0] + 0.02,
        VERTEX_XY["drift"][1] - 0.045,
        r"Drift $(0,0,1)$",
        ha="right",
        va="top",
        fontsize=11,
    )
    ax.text(
        0.78,
        0.38,
        "Partial Death",
        color=STATE_COLORS["Partial Death"],
        fontsize=10,
        rotation=-60,
        ha="center",
        va="center",
    )
    ax.text(
        0.50,
        -0.09,
        "Partial Locking",
        color=STATE_COLORS["Partial Locking"],
        fontsize=10,
        ha="center",
        va="top",
    )


def plot_simplex(
    picked: dict[str, np.ndarray],
    p: float,
    source: Path,
    figure_dir: Path,
    dpi: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    draw_simplex(ax)

    labels = picked["old_state"]
    x, y = barycentric_xy(picked["w_death"], picked["w_lock"], picked["w_drift"])
    if np.any(labels != ""):
        for name in PARTIAL_STATES:
            sel = labels == name
            if not np.any(sel):
                continue
            ax.scatter(
                x[sel],
                y[sel],
                s=18,
                alpha=0.75,
                c=STATE_COLORS[name],
                label=name,
                zorder=3,
                edgecolors="none",
            )
    else:
        ax.scatter(
            x,
            y,
            s=18,
            alpha=0.75,
            c="#5c6773",
            label="mixed cells (no old labels in table)",
            zorder=3,
            edgecolors="none",
        )

    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.18, SQRT3_OVER_2 + 0.14)
    ax.axis("off")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_title(
        r"Mixture-weight simplex, only Partial cells"
        f"\np={p:.2f}  (one point per (K, δ); vertices are reference only)"
        f"\nsource: {source.name}"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"mixture_simplex_partial_p_{p:.2f}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter Partial-state mixture weights on one 2-simplex. "
            "Reads saved tables; does not refit."
        )
    )
    parser.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=list(DEFAULT_P_VALUES),
        help=(
            "ER connection probabilities "
            f"(default: {' '.join(f'{p:.1f}' for p in DEFAULT_P_VALUES)})"
        ),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help=f"CSV/NPZ directory (default: {DEFAULT_TABLE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SIMPLEX_FIGURE_DIR,
        help=f"Figure directory (default: {DEFAULT_SIMPLEX_FIGURE_DIR})",
    )
    parser.add_argument(
        "--keep-delta-zero",
        action="store_true",
        help="Keep δ=0 cells (lock and drift are unidentifiable there)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=(
            "Trial npz directory, used only if the weight table has no old_state "
            f"(default: {DEFAULT_RAW_DIR})"
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_dir = resolve_user_path(args.table_dir, CLASSIFICATION_ROOT, CURSOR_ROOT)
    output_dir = resolve_user_path(
        args.output_dir, CLASSIFICATION_ROOT, CURSOR_ROOT
    )
    raw_dir = resolve_user_path(args.raw_dir, CLASSIFICATION_ROOT, CURSOR_ROOT)
    p_values = [float(value) for value in args.p]
    print("Scanning p = " + ", ".join(f"{p:.1f}" for p in p_values))
    for p in p_values:
        try:
            cells, source = load_cells(table_dir, p)
        except FileNotFoundError as exc:
            print(f"skip p={p:.2f}: {exc}")
            continue
        if not np.any(cells["old_state"] != ""):
            print(
                f"p={p:.2f}: {source.name} has no old_state; "
                "labeling Partial from stored trials (no mixture refit)"
            )
            cells = attach_old_states_from_raw(cells, raw_dir, p)
        picked, counts = select_partial(cells, keep_delta_zero=args.keep_delta_zero)
        if picked["K"].size == 0:
            print(f"skip p={p:.2f}: no Partial cells from {source}")
            continue
        path = plot_simplex(picked, p, source, output_dir, args.dpi)
        print(
            f"p={p:.2f}  cells={counts['n_cells']}  "
            f"after dropping δ=0: {counts['n_after_dropping_delta0']}  "
            f"plotted={counts['n_partial']}"
        )
        print(
            "mean H(w) of plotted points: "
            f"{float(np.mean(entropy(picked['w_death'], picked['w_lock'], picked['w_drift']))):.3f}"
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
