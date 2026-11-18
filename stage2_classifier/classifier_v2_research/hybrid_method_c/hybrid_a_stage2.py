"""Stage 2 review of Hybrid A linear-segment eligibility.

Reads the Stage 1 summary, draws a stratified sample, reuses archived
Method C cuts, and writes the summary figure plus the case gallery.
It does not rerun the Winfree ODE, does not merge segments, and does not
change Rule 3 or the state priority.

Run from this directory after ``hybrid_a_validation.py``:

    python3 hybrid_a_validation.py --stage2
"""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-hybrid-method-c")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import hybrid_a_validation as stage1
import hybrid_method_c as hybrid
from calibrate_plateau_detector import catalog

HERE = Path(__file__).resolve().parent
GALLERY_PATH = HERE / "hybrid_a_case_gallery.pdf"
FIGURE_PATH = HERE / "hybrid_a_validation.png"
CACHE_PATH = Path("/tmp/hybrid_a_stage2_cache.pkl")
PNG_DIR = Path("/tmp/hybrid_a_cases")
DRIFT_BANDS = ((0.0, 0.30), (0.30, 0.60), (0.60, 0.90), (0.90, 1.01))
SANITY = (
    (1.0, 0.20, 0.65),
    (1.0, 0.15, 0.40),
)
STATE_CODE = {"Locking": 0, "Partial Locking": 1, "Incoherence": 2}


def load_summary() -> dict:
    if not stage1.SUMMARY_PATH.is_file():
        raise RuntimeError("Stage 1 summary is missing; run hybrid_a_validation.py first")
    return json.loads(stage1.SUMMARY_PATH.read_text(encoding="utf-8"))


def pick_delta_bands(pool: list[dict], blocked: set[str]) -> list[dict]:
    chosen = []
    used = set(blocked)
    for low, high in DRIFT_BANDS:
        band = [
            row
            for row in pool
            if low <= row["delta"] < high and row["run_id"] not in used
        ]
        if not band:
            continue
        present = {round(row["p"], 4) for row in chosen}
        band.sort(
            key=lambda row: (
                round(row["p"], 4) in present,
                abs(row["K"] - 1.0),
                row["delta"],
                row["run_id"],
            )
        )
        chosen.append(band[0])
        used.add(band[0]["run_id"])
    return chosen


def pick_one_per_p(pool: list[dict], count: int, blocked: set[str], delta_targets: tuple[float, ...]) -> list[dict]:
    remaining = [row for row in pool if row["run_id"] not in blocked]
    chosen = []
    for p_value in sorted({round(row["p"], 4) for row in remaining}):
        if len(chosen) >= count:
            break
        group = [row for row in remaining if round(row["p"], 4) == p_value]
        target = delta_targets[min(len(chosen), len(delta_targets) - 1)]
        group.sort(key=lambda row: (abs(row["delta"] - target), abs(row["K"] - 1.0), row["run_id"]))
        chosen.append(group[0])
    chosen_ids = {row["run_id"] for row in chosen}
    rest = [row for row in remaining if row["run_id"] not in chosen_ids]
    while len(chosen) < count and rest:
        used_delta = [row["delta"] for row in chosen]
        rest.sort(key=lambda row: (-min(abs(row["delta"] - value) for value in used_delta), abs(row["K"] - 1.0), row["run_id"]))
        chosen.append(rest.pop(0))
    return chosen[:count]


def build_sample(summary: dict, rows: list[dict]) -> list[dict]:
    by_id = {row["run_id"]: row for row in rows}
    blocked: set[str] = set()
    sample = []

    def take(run_ids: list[str], category: str, picked: list[dict] | None = None) -> None:
        chosen = picked
        if chosen is None:
            chosen = [by_id[run_id] for run_id in run_ids]
        for row in chosen:
            if row["run_id"] in blocked:
                continue
            item = dict(row)
            item["category"] = category
            item["sanity"] = False
            sample.append(item)
            blocked.add(row["run_id"])

    take(summary["new_partial_ids"], "new_partial_locking")
    take(summary["plateau_like_incoherence_ids"], "plateau_like_incoherence")
    near = pick_one_per_p(
        [by_id[run_id] for run_id in summary["near_tau_ids"]],
        4,
        blocked,
        (0.05, 0.10, 0.15, 0.20),
    )
    take([], "near_tau", near)
    drift = pick_delta_bands([by_id[run_id] for run_id in summary["drift_control_ids"]], blocked)
    take([], "drift_control", drift)
    previous = pick_one_per_p(
        [by_id[run_id] for run_id in summary["previous_partial_ids"]],
        4,
        blocked,
        (0.05, 0.20, 0.225, 0.40),
    )
    take([], "previous_partial_locking", previous)
    for p_value, delta, coupling in SANITY:
        match = [
            row
            for row in rows
            if round(row["p"], 4) == round(p_value, 4)
            and round(row["delta"], 4) == round(delta, 4)
            and round(row["K"], 4) == round(coupling, 4)
        ]
        if len(match) != 1:
            raise RuntimeError(f"expected one sanity run at p={p_value}, delta={delta}, K={coupling}")
        row = match[0]
        if row["run_id"] in blocked:
            for item in sample:
                if item["run_id"] == row["run_id"]:
                    item["sanity"] = True
            continue
        item = dict(row)
        item["category"] = "sanity_check"
        item["sanity"] = True
        sample.append(item)
        blocked.add(row["run_id"])
    known = {(1.0, 0.20, 0.60), (1.0, 0.20, 0.70), (1.0, 0.20, 0.75)}
    for item in sample:
        if (round(item["p"], 4), round(item["delta"], 4), round(item["K"], 4)) in known:
            item["sanity"] = True
    return sample


def neighbor_line(index: hybrid.ProfileIndex, segment_index: int) -> tuple[float | None, float | None, int]:
    """Mean absolute residual of a segment against the line fit to its neighbors."""
    spans = []
    if segment_index > 0:
        spans.append(index.segments[segment_index - 1][:2])
    if segment_index + 1 < len(index.segments):
        spans.append(index.segments[segment_index + 1][:2])
    if not spans:
        return None, None, 0
    xs = [index.omega[start:end] for start, end in spans]
    ys = [index.rho[start:end] for start, end in spans]
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    count = int(x.size)
    if count < 15 or float(np.max(x) - np.min(x)) <= 1e-15:
        return None, None, count
    slope, intercept = np.polyfit(x, y, 1)
    start, end, _mode = index.segments[segment_index]
    residual = float(np.mean(np.abs(index.rho[start:end] - (slope * index.omega[start:end] + intercept))))
    trend = float(abs(slope) * (float(x.max()) - float(x.min())))
    return residual, trend, count


def shape_stats(omega: np.ndarray, rho: np.ndarray, start: int, end: int) -> tuple[float, float]:
    x = omega[start:end]
    y = rho[start:end]
    span = float(x[-1] - x[0])
    quad = 0.0
    if x.size >= 4 and span > 1e-15:
        curvature = float(np.polyfit(x, y, 2)[0])
        quad = abs(curvature) * span * span
    spike = float(np.max(np.abs(y - np.median(y)))) if y.size else 0.0
    return quad, spike


def enrich(index: hybrid.ProfileIndex, item: dict, windows: list[hybrid.ProfileIndex], matrix: np.ndarray) -> dict:
    start, end = int(item["start"]), int(item["end"])
    segment_index = next(
        position
        for position, (left, right, _mode) in enumerate(index.segments)
        if left == start and right == end
    )
    residual, neighbor_s, neighbor_n = neighbor_line(index, segment_index)
    quad, spike = shape_stats(index.omega, index.rho, start, end)
    record = {
        "start": start,
        "end": end,
        "n": int(item["n"]),
        "relative_omega_span": float(item["relative_omega_span"]),
        "R": float(item["R"]),
        "b": float(item["b"]),
        "S": float(item["S"]),
        "mean": float(item["mean"]),
        "origin": item["origin"],
        "quad_amp": quad,
        "spike": spike,
        "neighbor_residual": residual,
        "neighbor_S": neighbor_s,
        "neighbor_n": neighbor_n,
        "on_neighbor_line": bool(
            residual is not None
            and neighbor_s is not None
            and residual <= max(0.002, 2.0 * float(item["R"]))
            and neighbor_s > 0.001
            and float(item["relative_omega_span"]) < 0.10
        ),
    }
    for label, window, series in (("w1", windows[0], matrix[0]), ("w2", windows[1], matrix[1])):
        passed, features = stage1.passes_on(window, start, end)
        member_gap = np.abs(matrix[2, start:end] - series[start:end])
        record[f"{label}_pass"] = passed
        record[f"{label}_mean"] = float(features["mean"])
        record[f"{label}_R"] = float(features["R"])
        record[f"{label}_S"] = float(features["S"])
        record[f"{label}_abs"] = float(member_gap.mean())
        record[f"{label}_p90"] = float(np.percentile(member_gap, 90))
    return record


def screen_population(rows: list[dict], selected: dict[str, dict], cost_audit, pilot) -> tuple[dict, list[dict]]:
    population = {
        "runs": 0,
        "linear_by_state": Counter(),
        "runs_by_state": Counter(),
        "continuing_by_state": Counter(),
        "broad_by_state": Counter(),
        "cells": Counter(),
        "new_points": [],
        "candidates": [],
        "accepted": [],
        "flips": [],
        "protected": [],
    }
    cases = {run_id: None for run_id in selected}
    started = __import__("time").perf_counter()
    for index, row in enumerate(rows, start=1):
        loaded = stage1.load_run(row["path"])
        if loaded is None:
            continue
        omega, matrix = loaded
        segments = stage1.segment_fast(cost_audit, pilot, omega, matrix[2])
        profile = hybrid.ProfileIndex(omega, matrix[2], segments)
        decision = stage1.classify_window(profile)
        pair = (decision["state_base"], decision["state_hybrid"])
        if pair[0] in {"Death", "Locking", "Partial Death"} and pair[0] != pair[1]:
            population["protected"].append(row["run_id"])
        if pair == ("Partial Locking", "Incoherence"):
            population["protected"].append(row["run_id"])
        population["runs"] += 1
        population["runs_by_state"][decision["state_base"]] += 1
        host = decision["state_base"]
        if pair == ("Incoherence", "Partial Locking"):
            host = "New Partial Locking"
        delta_key = round(row["delta"], 4)
        coupling_key = round(row["K"], 4)
        population["cells"][(delta_key, coupling_key)] += len(decision["linears"])
        if host == "New Partial Locking":
            population["new_points"].append(
                {"p": row["p"], "delta": row["delta"], "K": row["K"], "n": len(decision["linears"])}
            )
        min_s = min(decision["candidate_s"]) if decision["candidate_s"] else None
        x = omega - float(omega.mean())
        y = matrix[2] - float(matrix[2].mean())
        denom = float(np.dot(x, x) * np.dot(y, y))
        corr = float(np.dot(x, y) / np.sqrt(denom)) if denom > 0.0 else 0.0
        if decision["state_base"] == "Incoherence" and min_s is not None and min_s <= 5e-4:
            population["flips"].append(
                {
                    "run_id": row["run_id"],
                    "p": row["p"],
                    "delta": row["delta"],
                    "K": row["K"],
                    "min_S": min_s,
                    "n_segments": decision["n_segments"],
                    "rho_range": float(matrix[2].max() - matrix[2].min()),
                    "abs_corr": abs(corr),
                    "flipped_at_1e-4": min_s <= stage1.TAU,
                }
            )
        windows = None
        if row["run_id"] in selected:
            windows = [hybrid.ProfileIndex(omega, matrix[window], [(0, omega.size, 1)]) for window in (0, 1)]
        for item in decision["linears"]:
            population["linear_by_state"][host] += 1
            start, end = int(item["start"]), int(item["end"])
            segment_index = next(
                position
                for position, (left, right, _mode) in enumerate(profile.segments)
                if left == start and right == end
            )
            residual, neighbor_s, _count = neighbor_line(profile, segment_index)
            continuing = bool(
                residual is not None
                and neighbor_s is not None
                and residual <= max(0.002, 2.0 * float(item["R"]))
                and neighbor_s > 0.001
                and float(item["relative_omega_span"]) < 0.10
            )
            if continuing:
                population["continuing_by_state"][host] += 1
            if float(item["relative_omega_span"]) >= 0.10:
                population["broad_by_state"][host] += 1
            code = {"Locking": 0, "Partial Locking": 1, "New Partial Locking": 2}.get(host, 3)
            population["accepted"].append(
                (
                    float(item["R"]),
                    float(item["S"]),
                    int(item["n"]),
                    float(item["relative_omega_span"]),
                    code,
                    row["delta"],
                    row["K"],
                    row["p"],
                    continuing,
                )
            )
        for length, slope in decision["candidates"]:
            population["candidates"].append((length, slope))
        if row["run_id"] in selected:
            enriched = [enrich(profile, item, windows, matrix) for item in decision["linears"]]
            constants = [enrich(profile, item, windows, matrix) for item in decision["constants"]]
            meta = selected[row["run_id"]]
            cases[row["run_id"]] = {
                "category": meta["category"],
                "sanity": meta["sanity"],
                "run_id": row["run_id"],
                "N": row["N"],
                "p": row["p"],
                "delta": row["delta"],
                "K": row["K"],
                "trial": row["trial"],
                "state_base": decision["state_base"],
                "state_hybrid": decision["state_hybrid"],
                "n_segments": decision["n_segments"],
                "n_constant_segments": decision["n_constant_segments"],
                "n_linear_segments": decision["n_linear_segments"],
                "n_linear_fail_geometry": decision["fail_geometry"],
                "n_linear_fail_trend": decision["fail_trend"],
                "rho_range": float(matrix[2].max() - matrix[2].min()),
                "abs_corr": abs(corr),
                "min_candidate_s": min_s,
                "omega": omega,
                "matrix": matrix,
                "segments": segments,
                "linears": enriched,
                "constants": constants,
            }
        if index % 1000 == 0 or index == len(rows):
            print(f"stage2 scanned {index}/{len(rows)} in {__import__('time').perf_counter() - started:.1f}s", flush=True)
    if population["protected"]:
        raise RuntimeError(f"unexpected state changes: {population['protected'][:8]}")
    ordered = [cases[row["run_id"]] for row in selected.values()]
    if any(case is None for case in ordered):
        raise RuntimeError("a sampled run was not screened")
    return population, ordered


def draw_profile(ax, case: dict, title: str | None = None) -> None:
    omega = case["omega"]
    matrix = case["matrix"]
    for item in case["constants"]:
        ax.axvspan(omega[item["start"]], omega[item["end"] - 1], color="#4C78A8", alpha=0.28, lw=0)
    for item in case["linears"]:
        ax.axvspan(omega[item["start"]], omega[item["end"] - 1], color="#F58518", alpha=0.35, lw=0)
    if case["n_segments"] <= 160:
        for start, end, _mode in case["segments"]:
            ax.axvline(omega[start], color="#B0B0B0", lw=0.25, alpha=0.7)
    ax.plot(omega, matrix[0], color="#B3B3B3", lw=0.8, label="W1")
    ax.plot(omega, matrix[1], color="#6B6B6B", lw=0.8, label="W2")
    ax.plot(omega, matrix[2], color="black", lw=1.05, label="W3")
    ax.set_xlabel("natural frequency")
    ax.set_ylabel("rotation number")
    if title:
        ax.set_title(title, loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="best")


def case_title(case: dict) -> str:
    sanity = " | sanity cell" if case["sanity"] else ""
    return (
        f"{case['category']}{sanity} | p={case['p']:.3g}, delta={case['delta']:.3g}, "
        f"K={case['K']:.3g}, N={case['N']}, trial={case['trial']}\n"
        f"{case['state_base']} -> {case['state_hybrid']} | segments={case['n_segments']} "
        f"(linear {case['n_linear_segments']}, constant {case['n_constant_segments']}) | "
        f"range={case['rho_range']:.4g} | |corr|={case['abs_corr']:.3f}"
    )


def case_note(case: dict) -> str:
    def row(item: dict, kind: str) -> str:
        residual = "na" if item["neighbor_residual"] is None else f"{item['neighbor_residual']:.3g}"
        neighbor = "na" if item["neighbor_S"] is None else f"{item['neighbor_S']:.3g}"
        return (
            f"{kind} [{item['start']:4d},{item['end']:4d}) n={item['n']:4d} "
            f"rel={item['relative_omega_span']:.3f} R={item['R']:.3g} S={item['S']:.3g} "
            f"b={item['b']:.3g} mean={item['mean']:.4f} quad={item['quad_amp']:.3g} spike={item['spike']:.3g} "
            f"nbrS={neighbor} resid={residual} line={int(item['on_neighbor_line'])} "
            f"W1={int(item['w1_pass'])}/{item['w1_abs']:.3g} W2={int(item['w2_pass'])}/{item['w2_abs']:.3g}"
        )

    lines = [
        f"min geometry-passing linear S={case['min_candidate_s']}",
        f"linear segments failing geometry={case['n_linear_fail_geometry']}, failing trend={case['n_linear_fail_trend']}",
    ]
    shown = sorted(case["linears"], key=lambda item: item["n"], reverse=True)
    extra = max(0, len(shown) - 8)
    for item in shown[:8]:
        lines.append(row(item, "linear"))
    if extra:
        lines.append(f"... {extra} narrower linear plateaus omitted from this note")
    for item in sorted(case["constants"], key=lambda item: item["n"], reverse=True)[:4]:
        lines.append(row(item, "const "))
    covered = sum(item["n"] for item in case["linears"]) + sum(item["n"] for item in case["constants"])
    lines.append(
        f"accepted members={covered}/{case['N']}, continuing-line flags="
        f"{sum(item['on_neighbor_line'] for item in case['linears'])}/{len(case['linears'])}"
    )
    return "\n".join(lines)


def write_gallery(cases: list[dict]) -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(GALLERY_PATH) as pdf:
        for index, case in enumerate(cases, start=1):
            fig = plt.figure(figsize=(11.2, 8.0))
            ax = fig.add_axes((0.07, 0.58, 0.90, 0.32))
            draw_profile(ax, case, case_title(case))
            ax.set_xlabel("")
            fig.text(0.04, 0.54, case_note(case), family="monospace", fontsize=6.2, va="top")
            fig.text(0.05, 0.02, f"case {index}/{len(cases)}  {case['run_id']}", fontsize=8, color="#444444")
            pdf.savefig(fig)
            fig.savefig(PNG_DIR / f"case_{index:02d}.png", dpi=120)
            plt.close(fig)
    print(f"wrote {GALLERY_PATH}", flush=True)


def tau_quality(candidates: list[tuple[int, float]]) -> list[dict]:
    lengths = np.array([item[0] for item in candidates], dtype=float)
    slopes = np.array([item[1] for item in candidates], dtype=float)
    rows = []
    previous = -1.0
    for tau in stage1.TAU_GRID:
        kept = slopes <= tau
        values = slopes[kept]
        sizes = lengths[kept]
        marginal = (slopes > previous) & (slopes <= tau)
        previous = tau
        rows.append(
            {
                "tau_s": tau,
                "linear_plateaus": int(kept.sum()),
                "short_n_lt_30": int((sizes < 30).sum()),
                "narrow_rel_proxy_n_lt_40": int((sizes < 40).sum()),
                "median_n": float(np.median(sizes)) if sizes.size else None,
                "median_S": float(np.median(values)) if values.size else None,
                "marginal_count": int(marginal.sum()),
                "marginal_median_n": float(np.median(lengths[marginal])) if marginal.any() else None,
                "marginal_median_S": float(np.median(slopes[marginal])) if marginal.any() else None,
            }
        )
    return rows


def write_summary_figure(population: dict, cases: list[dict]) -> None:
    accepted = population["accepted"]
    figure = plt.figure(figsize=(12.4, 8.6))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.05, 1.0), hspace=0.38, wspace=0.32)

    axis = figure.add_subplot(grid[0, 0])
    cells = population["cells"]
    deltas = np.array(sorted({key[0] for key in cells}))
    couplings = np.array(sorted({key[1] for key in cells}))
    heat = np.zeros((couplings.size, deltas.size))
    delta_index = {value: index for index, value in enumerate(deltas)}
    coupling_index = {value: index for index, value in enumerate(couplings)}
    for (delta, coupling), count in cells.items():
        heat[coupling_index[coupling], delta_index[delta]] = count
    image = axis.imshow(
        np.log1p(heat),
        origin="lower",
        aspect="auto",
        cmap="Greys",
        extent=(deltas[0], deltas[-1], couplings[0], couplings[-1]),
    )
    markers = {0.2: "o", 0.4: "s", 0.8: "D", 1.0: "^"}
    for point in population["new_points"]:
            axis.scatter(
            point["delta"],
            point["K"],
            s=48,
            marker=markers.get(round(point["p"], 4), "o"),
            facecolors="none",
            edgecolors="#E45756",
            linewidths=1.4,
            zorder=3,
        )
    axis.set_xlabel("delta")
    axis.set_ylabel("K")
    axis.set_title("New Partial Locking on linear-plateau counts", loc="left", fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04).set_label("log(1+count)")

    axis = figure.add_subplot(grid[0, 1])
    colors = {0: "#9A9A9A", 1: "#4C78A8", 2: "#F58518"}
    labels = {0: "inside Locking", 1: "inside previous Partial Locking", 2: "on new Partial Locking"}
    plotted = set()
    for code in (0, 1, 2):
        group = [item for item in accepted if item[4] == code]
        if not group:
            continue
        axis.scatter(
            [item[1] for item in group],
            [item[0] for item in group],
            s=10,
            alpha=0.55,
            c=colors[code],
            label=labels[code],
            linewidths=0,
        )
        plotted.add(code)
    axis.axvline(stage1.TAU, color="#E45756", lw=0.8, label="tau_S=0.0001")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("S_I")
    axis.set_ylabel("R_I")
    axis.set_title("Accepted linear segments", loc="left", fontsize=9)
    axis.legend(frameon=False, fontsize=6.5)

    axis = figure.add_subplot(grid[0, 2])
    quality = tau_quality(population["candidates"])
    flip_counts = []
    for tau in stage1.TAU_GRID:
        flip_counts.append(sum(row["min_S"] <= tau for row in population["flips"]))
    positions = np.arange(len(stage1.TAU_GRID))
    counts = [row["linear_plateaus"] for row in quality]
    axis.bar(positions, counts, color="#4C78A8")
    for pos, count, flips in zip(positions, counts, flip_counts):
        axis.text(pos, count + 30, f"{flips} labels", ha="center", va="bottom", fontsize=7, color="#9B2C2C")
    axis.set_xticks(positions, [f"{tau:.5g}" for tau in stage1.TAU_GRID], rotation=20)
    axis.set_ylim(0, max(counts) * 1.16)
    axis.set_ylabel("linear plateaus")
    axis.set_title("tau_S sensitivity", loc="left", fontsize=9)
    axis.set_xlabel("tau_S")

    new_cases = [case for case in cases if case["category"] == "new_partial_locking"]
    broad = max(new_cases, key=lambda case: (sum(item["n"] for item in case["linears"]), -case["n_segments"], case["run_id"]))
    fragmented = max(new_cases, key=lambda case: (case["n_segments"], case["run_id"]))
    for slot, case, heading in (
        (grid[1, 0:2], broad, "Broad shelf accepted as linear plateaus"),
        (grid[1, 2], fragmented, "Short fragments on a shelf with a rising tail"),
    ):
        axis = figure.add_subplot(slot)
        draw_profile(
            axis,
            case,
            f"{heading}\np={case['p']:.3g}, delta={case['delta']:.3g}, K={case['K']:.3g}, segments={case['n_segments']}",
        )
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)
    print(f"wrote {FIGURE_PATH}", flush=True)


def score_profile(profile, index: hybrid.ProfileIndex, tau: float, linear: bool) -> dict:
    items = hybrid.accept_plateaus(index, tau, float("inf"), False)
    if not linear:
        items = [item for item in items if item["origin"] == "constant-rule3"]
    truth = profile.truth if profile.truth.ndim == 1 else profile.truth[2]
    truth_runs = hybrid.true_runs(np.asarray(truth, dtype=bool))
    pred_runs = [(int(item["start"]), int(item["end"])) for item in items]
    used = set()
    true_positive = 0
    false_negative = 0
    boundary = []
    for start, end in truth_runs:
        best = 0.0
        best_index = None
        for pred_index, run in enumerate(pred_runs):
            if pred_index in used:
                continue
            score = hybrid.interval_iou((start, end), run)
            if score > best:
                best = score
                best_index = pred_index
        if best >= 0.5 and best_index is not None:
            used.add(best_index)
            true_positive += 1
            if profile.sigma <= 1e-3 + 1e-15 and (end - start) >= 80:
                run = pred_runs[best_index]
                boundary.append(0.5 * (abs(run[0] - start) + abs(run[1] - end)))
        else:
            false_negative += 1
    false_runs = []
    for pred_index, item in enumerate(items):
        if pred_index in used:
            continue
        if truth_runs:
            best = max(hybrid.interval_iou((item["start"], item["end"]), run) for run in truth_runs)
            if best >= 0.5:
                continue
        false_runs.append(item)
    return {
        "split": profile.split,
        "family": profile.family,
        "kind": profile.kind,
        "sigma": float(profile.sigma),
        "length": int(getattr(profile, "length", 0)),
        "n_segments": len(index.segments),
        "truth_runs": len(truth_runs),
        "tp": true_positive,
        "fn": false_negative,
        "fp_runs": len(false_runs),
        "fp_short_linear": sum(item["origin"] == "linear-trend" and item["n"] < 30 for item in false_runs),
        "fp_linear": sum(item["origin"] == "linear-trend" for item in false_runs),
        "fp_linear_details": [
            {
                "n": int(item["n"]),
                "S": float(item["S"]),
                "R": float(item["R"]),
                "relative_omega_span": float(item["relative_omega_span"]),
            }
            for item in false_runs
            if item["origin"] == "linear-trend"
        ],
        "fp_profile": int(len(false_runs) > 0),
        "boundary": boundary,
        "accepted_linear_n": [int(item["n"]) for item in items if item["origin"] == "linear-trend"],
        "accepted_linear_s": [float(item["S"]) for item in items if item["origin"] == "linear-trend"],
    }


def synthetic_benchmark(cost_audit, pilot) -> dict:
    profiles = catalog("dev") + catalog("val") + catalog("test")
    cache, elapsed = hybrid.prepared_cache(cost_audit, pilot, profiles)
    confirmation = {"val", "test"}
    out = {"segment_seconds": elapsed, "by_tau": [], "false_profiles": []}
    for tau in stage1.TAU_GRID:
        scored = [score_profile(profile, cache[profile.profile_id], tau, True) for profile in profiles]
        baseline = [score_profile(profile, cache[profile.profile_id], tau, False) for profile in profiles] if tau == stage1.TAU else None
        def pack(rows: list[dict], split_filter: set[str] | None) -> dict:
            chosen = [row for row in rows if split_filter is None or row["split"] in split_filter]
            def subset(**filters):
                return [
                    row
                    for row in chosen
                    if all(row[key] == value for key, value in filters.items())
                ]

            def rates(group: list[dict]) -> dict:
                boundary = [value for row in group for value in row["boundary"]]
                return {
                    "profiles": len(group),
                    "truth_runs": sum(row["truth_runs"] for row in group),
                    "tp": sum(row["tp"] for row in group),
                    "fn": sum(row["fn"] for row in group),
                    "fp_runs": sum(row["fp_runs"] for row in group),
                    "fp_profiles": sum(row["fp_profile"] for row in group),
                    "fp_linear": sum(row["fp_linear"] for row in group),
                    "fp_short_linear": sum(row["fp_short_linear"] for row in group),
                    "boundary_median": float(np.median(boundary)) if boundary else None,
                    "boundary_n": len(boundary),
                }

            return {
                "embedded": rates(subset(family="embedded")),
                "embedded_sigma_0.01": rates([row for row in subset(family="embedded") if row["sigma"] >= 0.01 - 1e-15]),
                "flat": rates(subset(family="flat")),
                "tails": rates(subset(family="tails")),
                "nearglobal": rates(subset(family="nearglobal")),
                "slow": rates(subset(family="slow")),
                "quad": rates(subset(family="nonlinear", kind="quad")),
                "drift": rates(subset(family="drift")),
                "sine": rates(subset(family="nonlinear", kind="sine")),
            }

        entry = {"tau_s": tau, "val_test": pack(scored, confirmation), "dev": pack(scored, {"dev"})}
        if baseline is not None:
            entry["baseline_val_test"] = pack(baseline, confirmation)
            entry["baseline_dev"] = pack(baseline, {"dev"})
        out["by_tau"].append(entry)
        for profile, row in zip(profiles, scored):
            if row["fp_linear_details"] or (row["fp_profile"] and row["family"] in {"slow", "drift", "nonlinear", "tails", "embedded"}):
                out["false_profiles"].append(
                    {
                        "tau_s": tau,
                        "split": row["split"],
                        "family": row["family"],
                        "kind": row["kind"],
                        "sigma": row["sigma"],
                        "length": row["length"],
                        "n_segments": row["n_segments"],
                        "fp_runs": row["fp_runs"],
                        "fp_short_linear": row["fp_short_linear"],
                        "fp_linear": row["fp_linear_details"],
                    }
                )
    return out


def compact_case(case: dict) -> dict:
    keep = {
        key: case[key]
        for key in (
            "category",
            "sanity",
            "run_id",
            "N",
            "p",
            "delta",
            "K",
            "trial",
            "state_base",
            "state_hybrid",
            "n_segments",
            "n_constant_segments",
            "n_linear_segments",
            "n_linear_fail_geometry",
            "n_linear_fail_trend",
            "rho_range",
            "abs_corr",
            "min_candidate_s",
        )
    }
    keep["linear_count"] = len(case["linears"])
    keep["constant_count"] = len(case["constants"])
    keep["continuing"] = sum(item["on_neighbor_line"] for item in case["linears"])
    keep["covered"] = sum(item["n"] for item in case["linears"]) + sum(item["n"] for item in case["constants"])
    keep["linears"] = case["linears"]
    keep["constants"] = [
        {key: item[key] for key in ("start", "end", "n", "relative_omega_span", "R", "S", "mean", "w1_pass", "w2_pass", "w1_abs", "w2_abs")}
        for item in case["constants"]
    ]
    return keep


def main() -> None:
    hybrid.load_archived_constants()
    pilot, cost_audit = hybrid.load_method_c()
    summary = load_summary()
    _audit, rows = stage1.audit_rows()
    sample = build_sample(summary, rows)
    print(
        json.dumps(
            [
                {
                    "category": item["category"],
                    "sanity": item["sanity"],
                    "p": item["p"],
                    "delta": item["delta"],
                    "K": item["K"],
                    "run_id": item["run_id"],
                }
                for item in sample
            ],
            indent=2,
        ),
        flush=True,
    )
    selected = {item["run_id"]: item for item in sample}
    population, cases = screen_population(rows, selected, cost_audit, pilot)
    print("synthetic benchmark", flush=True)
    synthetic = synthetic_benchmark(cost_audit, pilot)
    write_gallery(cases)
    write_summary_figure(population, cases)
    payload = {
        "sample_size": len(cases),
        "linear_by_state": dict(population["linear_by_state"]),
        "continuing_by_state": dict(population["continuing_by_state"]),
        "broad_by_state": dict(population["broad_by_state"]),
        "tau_quality": tau_quality(population["candidates"]),
        "flips": population["flips"],
        "cases": [compact_case(case) for case in cases],
        "synthetic": synthetic,
    }
    CACHE_PATH.write_bytes(pickle.dumps({"population_accepted_n": len(population["accepted"]), "cases": cases}))
    numbers_path = Path("/tmp/hybrid_a_stage2_numbers.json")
    numbers_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {numbers_path}", flush=True)
    print(
        json.dumps(
            {
                "linear_by_state": payload["linear_by_state"],
                "continuing_by_state": payload["continuing_by_state"],
                "broad_by_state": payload["broad_by_state"],
                "tau_quality": payload["tau_quality"],
                "flip_count": len(payload["flips"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
