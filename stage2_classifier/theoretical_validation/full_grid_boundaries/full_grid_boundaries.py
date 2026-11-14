"""Full-grid annealed boundaries on the Classifier v2.0.0 (Delta, K) diagrams.

The Hopf values are roots of Ariaratnam and Strogatz Eq. (6) and of the
degree-based characteristic equation from the annealed ER calculation.
They are not quenched-network eigenvalues and they are not classifier boundaries.
Missing roots are left blank. Curves are not interpolated across those gaps.

The classifier table is read only. Its rules, labels, and figures are not edited.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
THEORY = HERE.parent
CLASSIFIER_CSV = ROOT / "stage2_classifier" / "classifier_v2" / "classification_results.csv"
STEP1_CSV = THEORY / "p1_hopf_linear_response" / "p1_hopf_linear_response.csv"
STEP2_CSV = THEORY / "er_annealed_hopf" / "er_annealed_hopf.csv"
for path in (
    THEORY / "eq7_hopf",
    THEORY / "death_boundary",
    THEORY / "er_annealed_hopf",
    ROOT / "stage2_classifier" / "classifier_v2",
    ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import annealed_degree_hopf as degree  # noqa: E402
import validate_death_boundary as death  # noqa: E402
import validate_eq7 as eq7  # noqa: E402
from classify import PHASE_COLORS, PHASE_STATES  # noqa: E402

OUT_CSV = HERE / "full_grid_boundaries.csv"
OUT_JSON = HERE / "full_grid_boundaries_summary.json"
OUT_DEVIATION = HERE / "hopf_degree_vs_scaling_deviation.png"
P_DIAGRAMS = (0.2, 0.4, 0.8)
P_THEORY = (1.0, 0.8, 0.4, 0.2)
ANCHORS = (0.05, 0.10, 0.20)
EQ7_MAX = 0.20

CSV_FIELDS = [
    "p",
    "delta",
    "K_H_p1",
    "K_H_scaling",
    "K_H_degree",
    "Omega_H_degree",
    "hopf_root_residual",
    "hopf_validity_status",
    "hopf_convergence_status",
    "degree_tail_mass",
    "pinned_q_mass",
    "continuation_uncertainty",
    "K_all_drifting_edge",
    "K_death_p1",
    "K_death_existence",
    "death_boundary_type",
    "K_death_endpoint_p1",
    "K_death_endpoint",
    "K_eq7_p1",
    "K_eq7_scaling",
    "inside_k_scan",
    "notes",
]


def blank(value, digits: str | None = None) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    if digits is None:
        return f"{number:.10g}"
    return format(number, digits)


def load_classifier() -> list[dict]:
    rows = []
    with CLASSIFIER_CSV.open() as handle:
        for row in csv.DictReader(handle):
            if row["execution_status"] != "ok":
                continue
            if row["trial_id"] != "0" or row["alpha"] != "2" or row["beta"] != "1":
                continue
            if int(float(row["N"])) != 1000:
                continue
            rows.append(
                {
                    "p": float(row["p"]),
                    "delta": float(row["Delta"]),
                    "K": float(row["K"]),
                    "state": row["assigned_state"],
                }
            )
    return rows


def grid_axes(rows: list[dict], probability: float) -> tuple[np.ndarray, np.ndarray]:
    chosen = [row for row in rows if abs(row["p"] - probability) < 1.0e-12]
    deltas = np.array(sorted({row["delta"] for row in chosen}), dtype=float)
    couplings = np.array(sorted({row["K"] for row in chosen}), dtype=float)
    return deltas, couplings


def solve_eq6_grid(deltas: np.ndarray) -> dict[float, dict]:
    found = {}
    for delta in deltas:
        key = float(delta)
        if key <= 0.0:
            found[key] = {
                "status": "outside_validity_range",
                "roots": [],
                "notes": "degenerate frequency support; the Eq. (6) integral is undefined at delta=0",
            }
            print("eq6 delta=0 outside validity", flush=True)
            continue
        solved = eq7.solve_eq6(key)
        usable = [root for root in solved["roots"] if root.get("branch_ok")]
        if solved["status"] == "ok" and len(usable) == 1:
            root = usable[0]
            found[key] = {
                "status": "valid_root",
                "K": float(root["K"]),
                "Omega": float(root["Omega"]),
                "residual": float(root["imag_residual"]),
                "notes": "single Eq. (6) root inside the all-drifting branch",
            }
            print(f"eq6 delta={key:.3f} K={root['K']:.8f} Omega={root['Omega']:.6f}", flush=True)
        elif not usable:
            found[key] = {
                "status": "no_root_detected",
                "roots": [],
                "notes": "Eq. (6) solver found no sign-changing root with 0<K<1-delta; this is not a nonexistence proof",
            }
            print(f"eq6 delta={key:.3f} no root detected", flush=True)
        else:
            found[key] = {
                "status": "unresolved",
                "roots": usable,
                "notes": f"Eq. (6) solver returned {len(usable)} branch-valid roots",
            }
            print(f"eq6 delta={key:.3f} unresolved n={len(usable)}", flush=True)
    return found


def refine_eq6_endpoint(found: dict[float, dict], deltas: np.ndarray) -> dict:
    positive = [float(delta) for delta in deltas if float(delta) > 0.0]
    last_valid = max((delta for delta in positive if found[delta]["status"] == "valid_root"), default=None)
    first_missing = min((delta for delta in positive if delta > (last_valid or -1) and found[delta]["status"] != "valid_root"), default=None)
    if last_valid is None or first_missing is None:
        return {"last_grid_root": last_valid, "first_grid_without_root": first_missing}
    left, right = last_valid, first_missing
    last_root = {"delta": last_valid, "K": found[last_valid]["K"], "Omega": found[last_valid]["Omega"]}
    for _ in range(8):
        mid = 0.5 * (left + right)
        solved = eq7.solve_eq6(mid)
        usable = [root for root in solved["roots"] if root.get("branch_ok")]
        if solved["status"] == "ok" and len(usable) == 1:
            left = mid
            last_root = {"delta": mid, "K": float(usable[0]["K"]), "Omega": float(usable[0]["Omega"])}
        else:
            right = mid
    return {
        "last_grid_root": last_valid,
        "first_grid_without_root": first_missing,
        "refined_bracket": [left, right],
        "last_refined_root": last_root,
    }


def death_row(delta: float) -> dict:
    solved = death.existence_boundary(float(delta))
    if solved.get("status") != "ok":
        return {
            "K_death_p1": None,
            "death_boundary_type": solved.get("status", "unresolved"),
            "K_death_endpoint_p1": None,
            "notes": solved.get("reason", ""),
        }
    return {
        "K_death_p1": float(solved["K_exist"]),
        "death_boundary_type": solved["mechanism"],
        "K_death_endpoint_p1": float(solved["K_endpoint"]),
        "notes": "",
    }


def degree_row(probability: float, delta: float, paper: dict) -> dict:
    try:
        solved = degree.assess_case(probability, delta, paper)
    except Exception as exc:
        return {
            "K_H_degree": None,
            "Omega_H_degree": None,
            "hopf_root_residual": None,
            "hopf_validity_status": "unresolved",
            "hopf_convergence_status": "not_converged",
            "degree_tail_mass": None,
            "pinned_q_mass": None,
            "continuation_uncertainty": None,
            "notes": f"{type(exc).__name__}: {exc}",
        }
    valid = solved["validity_status"] == "valid_all_drifting" and solved["convergence_status"] == "converged"
    return {
        "K_H_degree": solved["K_H_degree"] if valid else None,
        "Omega_H_degree": solved["Omega_H"] if valid else None,
        "hopf_root_residual": solved["residual"],
        "hopf_validity_status": "valid_all_drifting" if valid else solved["validity_status"],
        "hopf_convergence_status": solved["convergence_status"],
        "degree_tail_mass": solved.get("discarded_probability"),
        "pinned_q_mass": solved.get("pinned_q_mass"),
        "continuation_uncertainty": solved.get("continuation_uncertainty"),
        "notes": "" if valid else solved["validity_status"],
    }


def build_rows(deltas: np.ndarray, papers: dict[float, dict], k_max: float) -> list[dict]:
    rows = []
    for delta in deltas:
        key = float(delta)
        paper = papers[key]
        life = death_row(key)
        edge_factor = 1.0 - key
        eq7 = eq7_value(key)
        for probability in P_THEORY:
            notes = []
            record = {
                "p": probability,
                "delta": key,
                "K_H_p1": None,
                "K_H_scaling": None,
                "K_H_degree": None,
                "Omega_H_degree": None,
                "hopf_root_residual": None,
                "hopf_validity_status": paper["status"] if paper["status"] != "valid_root" else "unresolved",
                "hopf_convergence_status": "not_applicable",
                "degree_tail_mass": None,
                "pinned_q_mass": None,
                "continuation_uncertainty": None,
                "K_all_drifting_edge": edge_factor / probability,
                "K_death_p1": life["K_death_p1"],
                "K_death_existence": None if life["K_death_p1"] is None else life["K_death_p1"] / probability,
                "death_boundary_type": life["death_boundary_type"],
                "K_death_endpoint_p1": life["K_death_endpoint_p1"],
                "K_death_endpoint": None if life["K_death_endpoint_p1"] is None else life["K_death_endpoint_p1"] / probability,
                "K_eq7_p1": eq7,
                "K_eq7_scaling": None if eq7 is None else eq7 / probability,
                "inside_k_scan": "",
                "notes": "",
            }
            if paper["status"] == "valid_root":
                record["K_H_p1"] = paper["K"]
                record["K_H_scaling"] = paper["K"] / probability
                degree_result = degree_row(probability, key, {"K": paper["K"], "Omega": paper["Omega"]})
                record.update({item: degree_result[item] for item in degree_result if item != "notes"})
                if degree_result["notes"]:
                    notes.append(degree_result["notes"])
                if record["K_H_degree"] is not None:
                    record["inside_k_scan"] = "yes" if 0.0 <= record["K_H_degree"] <= k_max + 1.0e-12 else "no"
                    if record["inside_k_scan"] == "no":
                        notes.append("root is outside the classifier K scan")
            else:
                notes.append(paper.get("notes", paper["status"]))
                record["hopf_convergence_status"] = "not_applicable"
            if life["notes"]:
                notes.append(life["notes"])
            record["notes"] = "; ".join(note for note in notes if note)
            rows.append(record)
    return rows


def eq7_value(delta: float) -> float | None:
    if 0.0 < float(delta) <= EQ7_MAX + 1.0e-12:
        return float(eq7.series_eq7(float(delta)))
    return None


def write_csv(rows: list[dict]) -> None:
    numeric = [
        "K_H_p1",
        "K_H_scaling",
        "K_H_degree",
        "Omega_H_degree",
        "hopf_root_residual",
        "degree_tail_mass",
        "pinned_q_mass",
        "continuation_uncertainty",
        "K_all_drifting_edge",
        "K_death_p1",
        "K_death_existence",
        "K_death_endpoint_p1",
        "K_death_endpoint",
        "K_eq7_p1",
        "K_eq7_scaling",
    ]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            payload = {field: row.get(field, "") for field in CSV_FIELDS}
            payload["p"] = f"{float(row['p']):.1f}"
            payload["delta"] = f"{float(row['delta']):.3f}"
            for field in numeric:
                payload[field] = blank(row.get(field))
            writer.writerow(payload)


def index_of(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    if np.any(values < axis[0] - 1.0e-12) or np.any(values > axis[-1] + 1.0e-12):
        raise ValueError("theory point is outside the classifier axis")
    return np.interp(values, axis, np.arange(axis.size, dtype=float))


def segments(points: list[tuple[float, float]], max_gap: float) -> list[np.ndarray]:
    """Connect sorted points only when the delta gap does not exceed one grid step."""
    ordered = sorted(points, key=lambda item: item[0])
    groups: list[list[tuple[float, float]]] = []
    for point in ordered:
        if groups and point[0] - groups[-1][-1][0] <= max_gap + 1.0e-8:
            groups[-1].append(point)
        else:
            groups.append([point])
    return [np.array(group, dtype=float) for group in groups if group]


def draw_phase(
    classifier_rows: list[dict],
    theory_rows: list[dict],
    probability: float,
    deltas: np.ndarray,
    show_states: bool = True,
) -> None:
    axis_p = probability if show_states else 0.2
    chosen = [row for row in classifier_rows if abs(row["p"] - axis_p) < 1.0e-12]
    delta_axis = np.array(sorted({row["delta"] for row in chosen}), dtype=float)
    k_axis = np.array(sorted({row["K"] for row in chosen}), dtype=float)
    step = float(np.min(np.diff(delta_axis)))
    lookup = {(round(row["delta"], 10), round(row["K"], 10)): row["state"] for row in chosen} if show_states else {}
    code = {state: index for index, state in enumerate(PHASE_STATES)}
    grid = np.full((delta_axis.size, k_axis.size), -1, dtype=int)
    delta_index = {round(value, 10): index for index, value in enumerate(delta_axis)}
    k_index = {round(value, 10): index for index, value in enumerate(k_axis)}
    for (delta, coupling), state in lookup.items():
        grid[delta_index[delta], k_index[coupling]] = code[state]
    display = np.ma.masked_less(grid.T, 0)
    cmap = ListedColormap(PHASE_COLORS)
    cmap.set_bad("#f0f0f0")
    norm = BoundaryNorm(np.arange(-0.5, len(PHASE_COLORS) + 0.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(9.4, 7.8))
    ax.imshow(display, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", zorder=0)
    ax.set_xticks(np.arange(delta_axis.size))
    ax.set_xticklabels(
        [f"{value:.1f}" if abs(value / 0.1 - round(value / 0.1)) < 1.0e-6 else "" for value in delta_axis]
    )
    ax.set_yticks(np.arange(k_axis.size))
    ax.set_yticklabels(
        [f"{value:.1f}" if abs(value / 0.2 - round(value / 0.2)) < 1.0e-6 else "" for value in k_axis]
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("Frequency half-width δ")
    ax.set_ylabel("Coupling strength K")
    subset = [row for row in theory_rows if abs(row["p"] - probability) < 1.0e-12]
    def collect(field: str, status_ok) -> list[tuple[float, float]]:
        points = []
        for row in subset:
            if not status_ok(row):
                continue
            value = row[field]
            if value is None or not (k_axis[0] - 1.0e-12 <= float(value) <= k_axis[-1] + 1.0e-12):
                continue
            points.append((float(row["delta"]), float(value)))
        return points

    hopf_degree = collect(
        "K_H_degree",
        lambda row: row["hopf_validity_status"] == "valid_all_drifting" and row["hopf_convergence_status"] == "converged",
    )
    hopf_scaling = collect("K_H_scaling", lambda row: row["K_H_p1"] is not None)
    edge = collect("K_all_drifting_edge", lambda row: True)
    death_sn = collect("K_death_existence", lambda row: row["death_boundary_type"] == "saddle-node")
    death_ep = collect("K_death_existence", lambda row: row["death_boundary_type"] == "end-point")
    switch = float(brentq(death.endpoint_slope, 0.2, 0.4))
    switch_k = death.existence_boundary(switch)["K_exist"] / probability
    if k_axis[0] <= switch_k <= k_axis[-1]:
        death_sn.append((switch, switch_k))
        death_ep.append((switch, switch_k))
    styles = (
        (hopf_degree, "#085c5c", 2.5, "-", 1.0, None),
        (hopf_scaling, "#e66101", 1.5, "--", 0.95, None),
        (edge, "black", 1.3, (0, (1.0, 1.3)), 1.0, None),
        (death_sn, "#8c510a", 2.0, "-.", 1.0, None),
        (death_ep, "#8c510a", 2.0, (0, (6, 2)), 1.0, None),
    )
    for points, color, width, linestyle, alpha, _marker in styles:
        for group in segments(points, step):
            ax.plot(
                index_of(group[:, 0], delta_axis),
                index_of(group[:, 1], k_axis),
                color=color,
                lw=width,
                ls=linestyle,
                alpha=alpha,
                solid_capstyle="round",
                zorder=3,
            )
    ax.set_xlim(-0.5, delta_axis.size - 0.5)
    ax.set_ylim(-0.5, k_axis.size - 0.5)
    if show_states:
        ax.set_title(
            f"Classifier v2.0.0 with annealed boundaries\nN=1000, p={probability:g}, W3=[12000, 14000], α=2, β=1, trials=1"
        )
    else:
        ax.set_title(
            f"Annealed boundaries on the Classifier v2.0.0 grid\n"
            f"N=1000, p={probability:g}; no classifier cells at this p"
        )
    state_handles = [
        Patch(facecolor=color, edgecolor="0.35", linewidth=0.4, label=state)
        for state, color in zip(PHASE_STATES, PHASE_COLORS)
    ] if show_states else []
    curve_handles = [
        Line2D([0], [0], color="#085c5c", lw=2.5, label="Degree-based annealed Hopf"),
        Line2D([0], [0], color="#e66101", lw=1.5, ls="--", label="Naive pK Hopf"),
        Line2D([0], [0], color="black", lw=1.3, ls=(0, (1.0, 1.3)), label="All-drifting edge (1−δ)/p"),
        Line2D([0], [0], color="#8c510a", lw=2.0, ls="-.", label="Death existence, saddle-node (pK approx.)"),
        Line2D([0], [0], color="#8c510a", lw=2.0, ls=(0, (6, 2)), label="Death existence, end-point (pK approx.)"),
    ]
    fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
    fig.legend(
        handles=state_handles + curve_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Curves are annealed approximations, not quenched eigenvalues and not classifier boundaries."
            if show_states
            else "Same δ step 0.025 and K step 0.05 as p=0.2, 0.4, 0.8. Gray means no Classifier v2.0.0 cell."
        ),
        ha="center",
        fontsize=8,
    )
    fig.savefig(HERE / f"phase_diagram_theory_p_{probability:.2f}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_deviation(rows: list[dict], probabilities: tuple[float, ...] = P_DIAGRAMS) -> None:
    colors = {0.8: "#2166ac", 0.6: "#6a3d9a", 0.4: "#1b9e77", 0.2: "#d95f02"}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for probability in probabilities:
        subset = [
            row
            for row in rows
            if abs(row["p"] - probability) < 1.0e-12
            and row["K_H_degree"] is not None
            and row["K_H_scaling"] is not None
            and row["hopf_validity_status"] == "valid_all_drifting"
            and row["hopf_convergence_status"] == "converged"
        ]
        subset = sorted(subset, key=lambda row: row["delta"])
        if len(subset) < 2:
            continue
        delta = np.array([row["delta"] for row in subset])
        deviation = np.array([(row["K_H_degree"] - row["K_H_scaling"]) / row["K_H_scaling"] for row in subset])
        ax.plot(delta, 100.0 * deviation, marker="o", ms=5, color=colors[probability], lw=1.6, label=f"p={probability:.1f}")
        factor = 100.0 * (2.0 * probability - 1.0) / (1.0 + (degree.N_OSC - 2) * probability)
        ax.axhline(factor, color=colors[probability], lw=0.8, ls=":", alpha=0.9)
    ax.axhline(0.0, color="0.4", lw=0.7)
    ax.set_xlabel("Frequency half-width δ")
    ax.set_ylabel(r"$(K_H^{degree}-K_H(p{=}1)/p)\,/\,(K_H(p{=}1)/p)$ (%)")
    ax.set_title("Degree-based Hopf correction relative to naive pK scaling")
    ax.grid(True, color="0.9", lw=0.6)
    ax.legend(
        fontsize=8,
        title="dotted: finite-N mean degree",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )
    fig.savefig(OUT_DEVIATION, dpi=180, bbox_inches="tight")
    plt.close(fig)


def nearest_states(classifier_rows: list[dict], theory_rows: list[dict]) -> dict:
    summary = {}
    for probability in P_DIAGRAMS:
        cells = [row for row in classifier_rows if abs(row["p"] - probability) < 1.0e-12]
        by_delta: dict[float, list[dict]] = {}
        for row in cells:
            by_delta.setdefault(round(row["delta"], 10), []).append(row)
        visits = []
        for row in theory_rows:
            if abs(row["p"] - probability) > 1.0e-12 or row["K_H_degree"] is None:
                continue
            delta_key = round(float(row["delta"]), 10)
            column = sorted(by_delta.get(delta_key, []), key=lambda item: item["K"])
            if not column:
                visits.append({"delta": row["delta"], "K": row["K_H_degree"], "placement": "delta not in classifier grid"})
                continue
            below = [item for item in column if item["K"] <= row["K_H_degree"] + 1.0e-12]
            above = [item for item in column if item["K"] >= row["K_H_degree"] - 1.0e-12]
            lower = below[-1] if below else None
            upper = above[0] if above else None
            if lower and upper and abs(lower["K"] - upper["K"]) < 1.0e-12:
                placement = f"inside {lower['state']}"
            elif lower and upper:
                placement = f"between {lower['state']} (K={lower['K']:g}) and {upper['state']} (K={upper['K']:g})"
            elif lower:
                placement = f"above saved cells, last cell {lower['state']}"
            else:
                placement = f"below saved cells, first cell {upper['state']}"
            visits.append({"delta": row["delta"], "K_H_degree": row["K_H_degree"], "placement": placement})
        summary[f"{probability:.1f}"] = visits
    return summary


def recovery_checks(rows: list[dict]) -> dict:
    step1 = {}
    with STEP1_CSV.open() as handle:
        for row in csv.DictReader(handle):
            if row["paper_K_H"]:
                step1[float(row["delta"])] = float(row["paper_K_H"])
    step2 = {}
    with STEP2_CSV.open() as handle:
        for row in csv.DictReader(handle):
            step2[(float(row["p"]), float(row["delta"]))] = {
                "K_p1": float(row["K_H_p1"]),
                "K_degree": float(row["K_H_degree"]),
            }
    comparisons = []
    passed = True
    for delta in ANCHORS:
        ours = next(row for row in rows if abs(row["p"] - 1.0) < 1.0e-12 and abs(row["delta"] - delta) < 1.0e-12)
        gap_eq6 = abs(ours["K_H_p1"] - step1[delta]) / step1[delta]
        gap_step2_p1 = abs(ours["K_H_p1"] - step2[(1.0, delta)]["K_p1"]) / step2[(1.0, delta)]["K_p1"]
        passed = passed and gap_eq6 < 1.0e-6 and gap_step2_p1 < 1.0e-8
        comparisons.append({"delta": delta, "p": 1.0, "kind": "eq6_vs_step1", "relative_gap": gap_eq6, "passed": gap_eq6 < 1.0e-6})
        comparisons.append({"delta": delta, "p": 1.0, "kind": "eq6_vs_step2", "relative_gap": gap_step2_p1, "passed": gap_step2_p1 < 1.0e-8})
        for probability in P_THEORY:
            current = next(row for row in rows if abs(row["p"] - probability) < 1.0e-12 and abs(row["delta"] - delta) < 1.0e-12)
            reference = step2[(probability, delta)]["K_degree"]
            if current["K_H_degree"] is None:
                passed = False
                comparisons.append({"delta": delta, "p": probability, "kind": "degree_vs_step2", "passed": False})
                continue
            gap = abs(current["K_H_degree"] - reference) / reference
            ok = gap < 1.0e-4
            passed = passed and ok
            comparisons.append({"delta": delta, "p": probability, "kind": "degree_vs_step2", "relative_gap": gap, "passed": ok})
    return {"passed": passed, "comparisons": comparisons}


def deviation_summary(rows: list[dict]) -> dict:
    summary = {}
    for probability in P_DIAGRAMS:
        values = []
        for row in rows:
            if abs(row["p"] - probability) > 1.0e-12 or row["K_H_degree"] is None or row["K_H_scaling"] is None:
                continue
            if row["hopf_validity_status"] != "valid_all_drifting" or row["hopf_convergence_status"] != "converged":
                continue
            values.append((row["delta"], (row["K_H_degree"] - row["K_H_scaling"]) / row["K_H_scaling"]))
        if not values:
            summary[f"{probability:.1f}"] = None
            continue
        worst = max(values, key=lambda item: abs(item[1]))
        summary[f"{probability:.1f}"] = {
            "n": len(values),
            "max_abs_relative_deviation": abs(worst[1]),
            "delta_at_max_abs": worst[0],
            "signed_at_max_abs": worst[1],
            "min_signed": min(item[1] for item in values),
            "max_signed": max(item[1] for item in values),
        }
    return summary


def main() -> None:
    started = time.perf_counter()
    classifier_rows = load_classifier()
    axes = {probability: grid_axes(classifier_rows, probability) for probability in P_DIAGRAMS}
    delta_sets = [tuple(np.round(axis[0], 10)) for axis in axes.values()]
    if len(set(delta_sets)) != 1:
        raise RuntimeError("the three phase diagrams do not share a delta grid")
    deltas = axes[0.2][0]
    k_max = max(float(axis[1][-1]) for axis in axes.values())
    print("n_delta", deltas.size, "k_max", k_max, flush=True)
    papers = solve_eq6_grid(deltas)
    endpoint = refine_eq6_endpoint(papers, deltas)
    print("endpoint", endpoint, flush=True)
    rows = build_rows(deltas, papers, k_max)
    checks = recovery_checks(rows)
    print("recovery", checks["passed"], flush=True)
    regions = nearest_states(classifier_rows, rows)
    deviations = deviation_summary(rows)
    write_csv(rows)
    for probability in P_DIAGRAMS:
        draw_phase(classifier_rows, rows, probability, deltas)
    draw_deviation(rows)
    payload = {
        "elapsed_sec": time.perf_counter() - started,
        "n_delta": int(deltas.size),
        "deltas": [float(delta) for delta in deltas],
        "eq6_endpoint": endpoint,
        "recovery": checks,
        "relative_deviation": deviations,
        "classifier_regions": regions,
        "statement": (
            "Hopf values are annealed characteristic roots. "
            "Death values are existence thresholds scaled by 1/p, not attraction thresholds. "
            "No missing Hopf root was filled by interpolation."
        ),
    }
    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value

    OUT_JSON.write_text(json.dumps(convert(payload), indent=2) + "\n")
    print("wrote", OUT_CSV, flush=True)


def read_theory_csv() -> list[dict]:
    numeric = {
        "p",
        "delta",
        "K_H_p1",
        "K_H_scaling",
        "K_H_degree",
        "Omega_H_degree",
        "hopf_root_residual",
        "degree_tail_mass",
        "pinned_q_mass",
        "continuation_uncertainty",
        "K_all_drifting_edge",
        "K_death_p1",
        "K_death_existence",
        "K_death_endpoint_p1",
        "K_death_endpoint",
        "K_eq7_p1",
        "K_eq7_scaling",
    }
    rows = []
    with OUT_CSV.open() as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key in numeric:
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def add_p06() -> None:
    """Theory-only p=0.6 rows on the same delta grid. No classifier cells exist."""
    probability = 0.6
    rows = [row for row in read_theory_csv() if abs(row["p"] - probability) > 1.0e-12]
    reference = [row for row in rows if abs(row["p"] - 1.0) < 1.0e-12]
    k_max = 3.0
    added = []
    for source in reference:
        record = {
            "p": probability,
            "delta": source["delta"],
            "K_H_p1": source["K_H_p1"],
            "K_H_scaling": None if source["K_H_p1"] is None else source["K_H_p1"] / probability,
            "K_H_degree": None,
            "Omega_H_degree": None,
            "hopf_root_residual": None,
            "hopf_validity_status": source["hopf_validity_status"],
            "hopf_convergence_status": "not_applicable",
            "degree_tail_mass": None,
            "pinned_q_mass": None,
            "continuation_uncertainty": None,
            "K_all_drifting_edge": (1.0 - source["delta"]) / probability,
            "K_death_p1": source["K_death_p1"],
            "K_death_existence": None if source["K_death_p1"] is None else source["K_death_p1"] / probability,
            "death_boundary_type": source["death_boundary_type"],
            "K_death_endpoint_p1": source["K_death_endpoint_p1"],
            "K_death_endpoint": None if source["K_death_endpoint_p1"] is None else source["K_death_endpoint_p1"] / probability,
            "K_eq7_p1": source["K_eq7_p1"],
            "K_eq7_scaling": None if source["K_eq7_p1"] is None else source["K_eq7_p1"] / probability,
            "inside_k_scan": "",
            "notes": source["notes"] or "",
        }
        if source["K_H_p1"] is not None and source["Omega_H_degree"] is not None:
            solved = degree_row(
                probability,
                source["delta"],
                {"K": source["K_H_p1"], "Omega": source["Omega_H_degree"]},
            )
            record.update({item: solved[item] for item in solved if item != "notes"})
            if solved["notes"]:
                record["notes"] = solved["notes"]
            if record["K_H_degree"] is not None:
                record["inside_k_scan"] = "yes" if 0.0 <= record["K_H_degree"] <= k_max + 1.0e-12 else "no"
        added.append(record)
    step2 = {}
    with STEP2_CSV.open() as handle:
        for raw in csv.DictReader(handle):
            if abs(float(raw["p"]) - probability) < 1.0e-12:
                step2[round(float(raw["delta"]), 10)] = float(raw["K_H_degree"])
    for record in added:
        archived = step2.get(round(record["delta"], 10))
        if archived is None or record["K_H_degree"] is None:
            continue
        gap = abs(record["K_H_degree"] - archived) / archived
        print(f"step2 recovery delta={record['delta']:.3f} relative_gap={gap:.3e}", flush=True)
        if gap > 1.0e-4:
            raise RuntimeError(f"p=0.6 delta={record['delta']} does not recover Step 2")
    rows.extend(added)
    write_csv(rows)
    classifier_rows = load_classifier()
    deltas = np.array(sorted({row["delta"] for row in reference}), dtype=float)
    draw_phase(classifier_rows, added, probability, deltas, show_states=False)
    draw_deviation(rows, probabilities=(0.8, 0.6, 0.4, 0.2))
    print("wrote p=0.6", flush=True)


if __name__ == "__main__":
    if "--p06" in sys.argv:
        add_p06()
    else:
        main()
