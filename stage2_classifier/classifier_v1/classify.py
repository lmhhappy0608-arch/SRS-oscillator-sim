"""Classify the Stage 1 pilot with the adopted v1.0.0-pilot rules.

Reads saved rotation profiles only. Does not integrate the ODE and does not
write the Stage 1 index.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-classifier-v11")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "penalty_calibration"))

from method_c_separated import load_method_c, segment_separated  # noqa: E402
from stage1.query import query_runs  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
VERSION = "v1.0.0-pilot"
STATES = ("Death", "Locking", "Partial Death", "Partial Locking", "Incoherence")
WINDOWS = (("W1", 8000.0, 10000.0), ("W2", 10000.0, 12000.0), ("W3", 12000.0, 14000.0))
STATE_COLOR = {
    "Death": "#332288",
    "Locking": "#0072B2",
    "Partial Death": "#CC79A7",
    "Partial Locking": "#E69F00",
    "Incoherence": "#009E73",
}
PHASE_STATES = (
    "Locking",
    "Partial Locking",
    "Incoherence",
    "Partial Death",
    "Death",
)
PHASE_COLORS = ("#2166ac", "#fdae61", "#1a9850", "#d73027", "#762a83")


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text())
    if config["classifier_version"] != VERSION:
        raise RuntimeError("config version does not match the script")
    return config


def consecutive_near_zero(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def classify_window(cost_audit, pilot, omega: np.ndarray, rho: np.ndarray, config: dict) -> dict:
    """Return the state and the segment record for one sorted profile."""
    near_tol = float(config["rule_4_near_zero_tolerance"]["value"])
    death_fraction = float(config["rule_5_death"]["value"])
    partial_run = int(config["rule_6_partial_death"]["minimum_consecutive_near_zero_oscillators"])
    lock_range = float(config["rule_7_locking"]["maximum_profile_rho_range"])
    rule3 = config["rule_3_valid_plateau"]
    min_points = int(rule3["minimum_oscillators"])
    min_rel = float(rule3["minimum_relative_omega_span"])
    max_rho_span = float(rule3["maximum_rho_span"])
    nonzero_level = float(rule3["nonzero_level_for_partial_locking"])
    min_len = int(config["method_c"]["minimum_segment_length"]["value"])
    alpha = float(config["alpha"]["value"])
    beta = float(config["beta"]["value"])

    near = np.abs(rho) < near_tol
    near_fraction = float(near.mean())
    run = consecutive_near_zero(near)
    profile_range = float(rho.max() - rho.min())
    sigma = float(pilot.estimate_sigma(rho))
    segments, fitted, objective, pen0, pen1, rss = segment_separated(
        cost_audit, omega, rho, sigma * sigma, alpha, beta, min_len
    )
    full_span = float(omega[-1] - omega[0])
    records = []
    nonzero_cover = np.zeros(rho.size, dtype=bool)
    for index, (start, end, mode) in enumerate(segments):
        piece = rho[start:end]
        level = float(np.mean(piece))
        rho_span = float(piece.max() - piece.min())
        omega_span = float(omega[end - 1] - omega[start])
        relative = float(omega_span / full_span) if full_span > 0 else 0.0
        passes = bool(
            mode == 0 and (end - start) >= min_points and relative >= min_rel and rho_span <= max_rho_span
        )
        nonzero = bool(passes and abs(level) >= nonzero_level)
        if nonzero:
            nonzero_cover[start:end] = True
        records.append(
            {
                "segment_index": index,
                "start": int(start),
                "end": int(end),
                "mode": "constant" if mode == 0 else "linear",
                "length": int(end - start),
                "omega_start": float(omega[start]),
                "omega_end": float(omega[end - 1]),
                "omega_span": omega_span,
                "relative_omega_span": relative,
                "rho_span": rho_span,
                "level": level,
                "passes_rule3": int(passes),
                "validated_nonzero_plateau": int(nonzero),
            }
        )
    if near_fraction >= death_fraction:
        state = "Death"
    elif profile_range <= lock_range:
        state = "Locking"
    elif run >= partial_run:
        state = "Partial Death"
    elif int(nonzero_cover.sum()) > 0:
        state = "Partial Locking"
    else:
        state = "Incoherence"
    if state == "Locking":
        locked_fraction = 1.0
    else:
        locked_fraction = float(nonzero_cover.mean())
    return {
        "state": state,
        "near_zero_fraction": near_fraction,
        "locked_fraction": locked_fraction,
        "consecutive_near_zero": run,
        "rho_range": profile_range,
        "sigma_hat": sigma,
        "sigma_hat_at_floor": int(sigma <= 1e-12 * 1.0000001),
        "objective": float(objective),
        "total_rss": float(rss),
        "penalty_constant": float(pen0),
        "penalty_linear": float(pen1),
        "n_segments": len(records),
        "candidate_plateau_count": sum(row["mode"] == "constant" for row in records),
        "validated_plateau_count": sum(row["validated_nonzero_plateau"] for row in records),
        "validated_rule3_count": sum(row["passes_rule3"] for row in records),
        "validated_near_zero_plateau_count": sum(
            row["passes_rule3"] and not row["validated_nonzero_plateau"] for row in records
        ),
        "nonzero_plateau_mask": nonzero_cover,
        "segments": records,
        "fitted": fitted,
    }


def review_for_windows(by_window: dict[str, dict]) -> tuple[int, str]:
    states = [by_window[name]["state"] for name, *_ in WINDOWS]
    counts = [by_window[name]["validated_plateau_count"] for name, *_ in WINDOWS]
    reasons = []
    if len(set(states)) > 1:
        reasons.append("temporal_state_disagreement")
    if len(set(counts)) > 1:
        reasons.append("temporal_plateau_count_disagreement")
    return int(bool(reasons)), ";".join(reasons)


def assert_priority_examples(cost_audit, pilot, config: dict) -> None:
    """Small constructed profiles. A failure here stops the pilot classification."""
    omega = np.linspace(0.8, 1.2, 1000)
    death = classify_window(cost_audit, pilot, omega, np.zeros(1000), config)
    if death["state"] != "Death":
        raise RuntimeError(f"all-zero profile returned {death['state']}")
    locked = classify_window(cost_audit, pilot, omega, np.full(1000, 0.8), config)
    if locked["state"] != "Locking":
        raise RuntimeError(f"flat nonzero profile returned {locked['state']}")
    partial = np.concatenate([np.zeros(200), np.linspace(0.2, 1.0, 800)])
    partial_state = classify_window(cost_audit, pilot, omega, partial, config)
    if partial_state["state"] != "Partial Death":
        raise RuntimeError(f"near-zero block returned {partial_state['state']}")
    incoherent = classify_window(cost_audit, pilot, omega, omega.copy(), config)
    if incoherent["state"] != "Incoherence":
        raise RuntimeError(f"rho = omega returned {incoherent['state']}")


def window_row(starts, ends, bounds) -> int:
    hits = [
        index
        for index, (start, end) in enumerate(zip(starts, ends))
        if abs(float(start) - bounds[0]) < 1e-6 and abs(float(end) - bounds[1]) < 1e-6
    ]
    if len(hits) != 1:
        raise RuntimeError(f"window {bounds} matched {hits}")
    return hits[0]


def classify_pilot(config: dict) -> tuple[list[dict], list[dict], dict]:
    pilot, cost_audit = load_method_c()
    assert_priority_examples(cost_audit, pilot, config)
    rows = []
    segments = []
    plot_cache = {}
    runs = query_runs(status="completed")
    for run in runs:
        record = {
            "run_id": run["run_id"],
            "classifier_version": VERSION,
            "K": run["K"],
            "p": run["p"],
            "Delta": run["Delta"],
            "N": run["N"],
            "alpha": config["alpha"]["value"],
            "beta": config["beta"]["value"],
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
            by_window = {}
            for name, start, end in WINDOWS:
                rho = rotation[window_row(starts, ends, (start, end))]
                result = classify_window(cost_audit, pilot, omega, rho, config)
                by_window[name] = result
                for segment in result["segments"]:
                    segments.append(
                        {
                            "run_id": run["run_id"],
                            "classifier_version": VERSION,
                            "window": name,
                            "K": run["K"],
                            "p": run["p"],
                            "Delta": run["Delta"],
                            **segment,
                        }
                    )
            primary = by_window["W3"]
            flag, reason = review_for_windows(by_window)
            record.update(
                {
                    "assigned_state": primary["state"],
                    "state_w1": by_window["W1"]["state"],
                    "state_w2": by_window["W2"]["state"],
                    "state_w3": primary["state"],
                    "near_zero_fraction": primary["near_zero_fraction"],
                    "locked_fraction": primary["locked_fraction"],
                    "consecutive_near_zero": primary["consecutive_near_zero"],
                    "rho_range": primary["rho_range"],
                    "sigma_hat": primary["sigma_hat"],
                    "sigma_hat_at_floor": primary["sigma_hat_at_floor"],
                    "candidate_plateau_count": primary["candidate_plateau_count"],
                    "validated_plateau_count": primary["validated_plateau_count"],
                    "validated_rule3_count": primary["validated_rule3_count"],
                    "validated_near_zero_plateau_count": primary["validated_near_zero_plateau_count"],
                    "validated_plateau_count_w1": by_window["W1"]["validated_plateau_count"],
                    "validated_plateau_count_w2": by_window["W2"]["validated_plateau_count"],
                    "validated_plateau_count_w3": primary["validated_plateau_count"],
                    "review_flag": flag,
                    "temporal_review_flag": flag,
                    "review_reason": reason,
                }
            )
            if run["run_id"] not in plot_cache:
                plot_cache[run["run_id"]] = {"omega": omega, "rho": rotation[window_row(starts, ends, (12000.0, 14000.0))], "segments": primary["segments"], "fitted": primary["fitted"]}
        except Exception as exc:
            record.update(
                {
                    "assigned_state": "",
                    "state_w1": "",
                    "state_w2": "",
                    "state_w3": "",
                    "near_zero_fraction": "",
                    "locked_fraction": "",
                    "consecutive_near_zero": "",
                    "rho_range": "",
                    "sigma_hat": "",
                    "sigma_hat_at_floor": "",
                    "candidate_plateau_count": "",
                    "validated_plateau_count": "",
                    "validated_rule3_count": "",
                    "validated_near_zero_plateau_count": "",
                    "validated_plateau_count_w1": "",
                    "validated_plateau_count_w2": "",
                    "validated_plateau_count_w3": "",
                    "review_flag": 1,
                    "temporal_review_flag": 0,
                    "review_reason": "classification_failed",
                    "execution_status": "failed",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(record)
    return rows, segments, plot_cache


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(results: list[dict]) -> list[dict]:
    ok = [row for row in results if row["execution_status"] == "ok"]
    total = len(ok)
    counts = Counter(row["assigned_state"] for row in ok)
    output = []
    for state in STATES:
        count = counts.get(state, 0)
        output.append(
            {
                "scope": "all",
                "p": "",
                "Delta": "",
                "assigned_state": state,
                "count": count,
                "fraction": count / total if total else 0.0,
                "classifier_version": VERSION,
            }
        )
    grouped = Counter((row["p"], row["Delta"], row["assigned_state"]) for row in ok)
    sizes = Counter((row["p"], row["Delta"]) for row in ok)
    for p, delta, state in sorted(grouped):
        count = grouped[(p, delta, state)]
        output.append(
            {
                "scope": "p_delta",
                "p": p,
                "Delta": delta,
                "assigned_state": state,
                "count": count,
                "fraction": count / sizes[(p, delta)],
                "classifier_version": VERSION,
            }
        )
    return output


def plot_distribution(results: list[dict], path: Path, title: str | None = None) -> None:
    ok = [row for row in results if row["execution_status"] == "ok"]
    counts = [sum(row["assigned_state"] == state for row in ok) for state in STATES]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = [STATE_COLOR[state] for state in STATES]
    bars = ax.bar(STATES, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")
    ax.set_ylabel("Profiles")
    ax.set_title(title or f"Classifier {VERSION} | alpha=2 | beta=1\nW3 state counts on the Stage 1 pilot")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_state_vs_k(results: list[dict], path: Path) -> None:
    ok = [row for row in results if row["execution_status"] == "ok"]
    p_values = [1.0, 0.8, 0.4, 0.2]
    deltas = [0.05, 0.2, 0.8]
    rank = {state: index for index, state in enumerate(STATES)}
    fig, axes = plt.subplots(len(deltas), len(p_values), figsize=(12.5, 7.2), sharex=True, sharey=True)
    for row_index, delta in enumerate(deltas):
        for col_index, p_value in enumerate(p_values):
            ax = axes[row_index, col_index]
            chosen = [item for item in ok if abs(item["p"] - p_value) < 1e-9 and abs(item["Delta"] - delta) < 1e-9]
            for state in STATES:
                xs = [item["K"] for item in chosen if item["assigned_state"] == state]
                ys = [rank[state]] * len(xs)
                ax.scatter(xs, ys, s=16, color=STATE_COLOR[state], label=state)
            ax.set_yticks(range(len(STATES)), STATES, fontsize=7)
            if row_index == 0:
                ax.set_title(f"p = {p_value:g}")
            if col_index == 0:
                ax.set_ylabel(f"Delta = {delta:g}")
            if row_index == len(deltas) - 1:
                ax.set_xlabel("K")
    fig.suptitle(f"Classifier {VERSION} | alpha=2 | beta=1\nW3 state against K", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _phase_cmap():
    code = {state: index for index, state in enumerate(PHASE_STATES)}
    cmap = ListedColormap(PHASE_COLORS)
    cmap.set_bad("#f0f0f0")
    norm = BoundaryNorm(np.arange(-0.5, len(PHASE_COLORS) + 0.5, 1), cmap.N)
    handles = [Patch(color=PHASE_COLORS[index], label=state) for index, state in enumerate(PHASE_STATES)]
    return code, cmap, norm, handles


def _draw_phase_panel(ax, rows: list[dict], p_value: float, code: dict, cmap, norm, ylabel: bool) -> str:
    chosen = [row for row in rows if abs(float(row["p"]) - p_value) < 1e-9]
    if not chosen:
        ax.set_axis_off()
        return ""
    k_axis = np.array(sorted({float(row["K"]) for row in chosen}))
    delta_axis = np.array(sorted({float(row["Delta"]) for row in chosen}))
    k_index = {round(value, 10): index for index, value in enumerate(k_axis)}
    delta_index = {round(value, 10): index for index, value in enumerate(delta_axis)}
    grid = np.full((delta_axis.size, k_axis.size), -1, dtype=int)
    for row in chosen:
        grid[delta_index[round(float(row["Delta"]), 10)], k_index[round(float(row["K"]), 10)]] = code[
            row["assigned_state"]
        ]
    display = np.ma.masked_less(grid.T, 0)
    ax.imshow(display, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(delta_axis.size))
    ax.set_xticklabels([f"{value:.2f}" for value in delta_axis])
    ax.set_yticks(np.arange(k_axis.size))
    ax.set_yticklabels([f"{value:.2f}" for value in k_axis])
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("Frequency width Δ")
    if ylabel:
        ax.set_ylabel("Coupling strength K")
    n_values = {int(float(row["N"])) for row in chosen}
    return str(next(iter(n_values))) if len(n_values) == 1 else "mixed"


def plot_phase_diagrams(
    results: list[dict],
    figure_dir: Path,
    label: str | None = None,
    filename_prefix: str = "phase_diagram",
) -> None:
    """Draw one (Delta, K) phase diagram per p from the W3 labels.

    Cells have equal width, matching the earlier imshow phase diagrams.
    The color order is Locking, Partial Locking, Incoherence, Partial Death, Death.
    """
    ok = [row for row in results if row["execution_status"] == "ok"]
    code, cmap, norm, handles = _phase_cmap()
    p_values = sorted({float(row["p"]) for row in ok}, reverse=True)
    for p_value in p_values:
        fig, ax = plt.subplots(figsize=(9, 7))
        n_label = _draw_phase_panel(ax, ok, p_value, code, cmap, norm, ylabel=True)
        heading = label or f"Classifier {VERSION} | alpha=2 | beta=1"
        ax.set_title(f"{heading}\nN={n_label}, p={p_value:g}, W3=[12000,14000], trials=1")
        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.38, 1.0))
        fig.tight_layout()
        fig.savefig(figure_dir / f"{filename_prefix}_p_{p_value:.2f}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_phase_comparison(
    v1_results: list[dict],
    summer_results: list[dict],
    figure_dir: Path,
    summer_label: str = "Summer classifier",
) -> None:
    """Place the v1 and summer phase diagrams for the same p side by side."""
    v1_ok = [row for row in v1_results if row["execution_status"] == "ok"]
    summer_ok = [row for row in summer_results if row["execution_status"] == "ok"]
    code, cmap, norm, handles = _phase_cmap()
    p_values = sorted({float(row["p"]) for row in v1_ok} | {float(row["p"]) for row in summer_ok}, reverse=True)
    v1_heading = f"Classifier {VERSION} | alpha=2 | beta=1"
    for p_value in p_values:
        fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2), sharey=True)
        n_label = _draw_phase_panel(axes[0], v1_ok, p_value, code, cmap, norm, ylabel=True)
        _draw_phase_panel(axes[1], summer_ok, p_value, code, cmap, norm, ylabel=False)
        axes[0].set_title(v1_heading)
        axes[1].set_title(summer_label)
        axes[1].legend(handles=handles, loc="upper right", bbox_to_anchor=(1.46, 1.0))
        fig.suptitle(f"N={n_label}, p={p_value:g}, W3=[12000,14000], trials=1", y=1.02)
        fig.tight_layout()
        fig.savefig(figure_dir / f"phase_diagram_compare_p_{p_value:.2f}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_representatives(results: list[dict], cache: dict, path: Path) -> None:
    table = ROOT / "stage1_analysis" / "pilot_exploration" / "representative_cases.csv"
    preferred = {}
    if table.is_file():
        with table.open() as handle:
            for row in csv.DictReader(handle):
                preferred.setdefault(row["run_id"], row["case_id"])
    by_state = {state: [] for state in STATES}
    for row in results:
        if row["execution_status"] == "ok":
            by_state[row["assigned_state"]].append(row)
    chosen = []
    for state in STATES:
        pool = by_state[state]
        named = [row for row in pool if row["run_id"] in preferred]
        pick = named[0] if named else (pool[0] if pool else None)
        if pick is not None:
            chosen.append(pick)
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4), sharex=False)
    for ax, row in zip(axes.ravel(), chosen):
        cached = cache[row["run_id"]]
        ax.plot(cached["omega"], cached["rho"], color="0.65", lw=0.8)
        for segment in cached["segments"]:
            sl = slice(segment["start"], segment["end"])
            color = "#0072B2" if segment["mode"] == "constant" else "#D55E00"
            ax.plot(cached["omega"][sl], cached["fitted"][sl], color=color, lw=1.5)
        case = preferred.get(row["run_id"], row["run_id"][:8])
        ax.set_title(
            f"{case}  {row['assigned_state']}\nK={row['K']:g}, p={row['p']:g}, Delta={row['Delta']:g}",
            fontsize=9,
        )
        ax.set_xlabel("omega")
        ax.set_ylabel("rotation number")
    for ax in axes.ravel()[len(chosen) :]:
        ax.axis("off")
    fig.suptitle(f"Classifier {VERSION} | alpha=2 | beta=1\nW3 profiles and Method C fit")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def write_manifest(results: list[dict]) -> dict:
    ok = sum(row["execution_status"] == "ok" for row in results)
    failed = sum(row["execution_status"] == "failed" for row in results)
    names = [
        "config.json",
        "algorithm_specification.md",
        "classify.py",
        "classification_results.csv",
        "segmentation_details.csv",
        "state_summary.csv",
        "review_candidates.csv",
        "README.md",
        "CHANGELOG.md",
    ]
    checksums = {name: sha256(HERE / name) for name in names if (HERE / name).is_file()}
    manifest = {
        "archive_status": "complete",
        "execution_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classifier_version": VERSION,
        "archive_note": "Completed in place. This directory replaces the earlier incomplete unlabeled archive.",
        "alpha": 2,
        "beta": 1,
        "git_commit": git_head(),
        "git_commit_created": False,
        "source_data_directory": str(ROOT / "stage1_runs"),
        "dataset_identifier": "stage1_pilot",
        "number_of_input_runs": len(results),
        "number_of_successfully_classified_runs": ok,
        "number_of_failed_runs": failed,
        "configuration_file": str(CONFIG_PATH),
        "algorithm_specification": str(HERE / "algorithm_specification.md"),
        "output_files": {name: str(HERE / name) for name in names},
        "checksums_sha256": checksums,
    }
    (HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    errors = []
    if len(results) != 732:
        errors.append("input count")
    if len({row["run_id"] for row in results}) != len(results):
        errors.append("run_id is not unique")
    if ok + failed != len(results):
        errors.append("count identity")
    if any(row["classifier_version"] != VERSION for row in results):
        errors.append("version")
    if any(row["execution_status"] == "ok" and (row["alpha"] != 2 or row["beta"] != 1) for row in results):
        errors.append("alpha beta")
    if any(row["execution_status"] == "ok" and row["assigned_state"] not in STATES for row in results):
        errors.append("state")
    report = {"passed": not errors, "errors": errors, "classified": ok, "failed": failed, "input": len(results)}
    (HERE / "archive_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    config = load_config()
    figure_dir = HERE / "figures"
    figure_dir.mkdir(exist_ok=True)
    results, segments, cache = classify_pilot(config)
    write_csv(HERE / "classification_results.csv", results)
    write_csv(HERE / "segmentation_details.csv", segments)
    write_csv(HERE / "state_summary.csv", summary_rows(results))
    reviews = [row for row in results if row["review_flag"] == 1]
    write_csv(HERE / "review_candidates.csv", reviews)
    plot_distribution(results, figure_dir / "state_distribution.png")
    plot_state_vs_k(results, figure_dir / "state_vs_K.png")
    plot_phase_diagrams(results, figure_dir)
    plot_representatives(results, cache, figure_dir / "representative_cases.png")
    report = write_manifest(results)
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
