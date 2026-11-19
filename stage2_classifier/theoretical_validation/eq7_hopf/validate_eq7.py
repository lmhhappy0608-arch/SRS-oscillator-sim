"""Compare Classifier v2 incoherence exits with the Ariaratnam–Strogatz Hopf boundary.

Reads saved Stage 1 p=1 profiles and labels them in memory with Classifier v2.
The Hopf threshold is the printed Eq. (7) truncation.
Eq. (6) is solved only at a few widths as a check on that truncation.
This script does not integrate the ODE, edit thresholds, or rewrite archives.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-eq7-hopf")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import quad
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLASSIFIER_DIR = ROOT / "stage2_classifier" / "classifier_v2"
sys.path.insert(0, str(CLASSIFIER_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "penalty_calibration"))

from classify import PHASE_COLORS, PHASE_STATES, classify_window, load_config, window_row  # noqa: E402
from method_c_separated import load_method_c  # noqa: E402
from stage1.model import make_adjacency, sha256_array  # noqa: E402
from stage1.query import query_runs  # noqa: E402

W3_BOUNDS = (12000.0, 14000.0)
FIGURE_PATH = HERE / "eq7_validation_figure.png"
MAIN_DELTA_MAX = 0.20
MISSING_COLOR = "#d0d0d0"
HOPF_COLOR = "#e66101"
EQ6_DELTAS = (0.05, 0.10, 0.15, 0.20, 0.22, 0.225, 0.25)
PI = float(np.pi)


def series_eq7(delta: float) -> float:
    """Printed Eq. (7), truncated before O(delta^7).

    kappa = (8*delta/pi) * [1 + 16*delta^2/pi^2 + 16*(pi^2+80)*delta^4/pi^4].
    The additive expansion 8*delta/pi + 16*delta^2/pi^2 + ... is not this formula.
    """
    delta = float(delta)
    factor = 1.0 + 16.0 * delta**2 / PI**2 + 16.0 * (PI**2 + 80.0) * delta**4 / PI**4
    return (8.0 * delta / PI) * factor


def additive_transcription(delta: float) -> float:
    """The expanded expression in the task note. Kept only as a transcription check."""
    delta = float(delta)
    return 8.0 * delta / PI + 16.0 * delta**2 / PI**2 + 16.0 * (PI**2 + 80.0) * delta**4 / PI**4


def incoherent_branch(delta: float, coupling: float) -> bool:
    """The stationary incoherent state requires K < 1 - delta, and delta > 0."""
    return float(delta) > 0.0 and float(coupling) < 1.0 - float(delta)


def _pole_pair(delta: float, z: float) -> tuple[float, float, float] | None:
    """Real-part (Landau) relation at a pole omega_p = 1 + delta*z.

    Returns (omega_p, Omega, K) only when 0 < K < 1 - delta and Omega > 0.
    """
    omega_p = 1.0 + float(delta) * float(z)
    if omega_p <= 0.0:
        return None
    radius = 16.0 * delta * delta / (PI * PI * omega_p * omega_p)
    candidates = []
    for root in np.roots([1.0, -1.0, radius, radius]):
        if abs(root.imag) < 1e-8 and 1e-8 < root.real < 1.0 - 1e-8:
            candidates.append(float(root.real))
    if not candidates:
        return None
    ratio = min(candidates, key=lambda value: abs(value - 1.0))
    coupling = omega_p * float(np.sqrt(max(0.0, 1.0 - ratio * ratio)))
    frequency = ratio * omega_p
    if not (0.0 < coupling < 1.0 - float(delta) and frequency > 0.0):
        return None
    return omega_p, frequency, coupling


def imaginary_residual(delta: float, z: float) -> float:
    """Principal-value imaginary part of Eq. (6) at lambda = i*Omega(z)."""
    pair = _pole_pair(delta, z)
    if pair is None:
        return np.nan
    omega_p, frequency, coupling = pair
    lower, upper = 1.0 - float(delta), 1.0 + float(delta)

    def numerator_slope(omega: float) -> float:
        speed = float(np.sqrt(max(0.0, omega * omega - coupling * coupling)))
        return omega * (omega - speed) / (omega + omega_p)

    level = numerator_slope(omega_p)

    def regularized(omega: float) -> float:
        gap = omega - omega_p
        if abs(gap) < 1e-8:
            return (numerator_slope(omega_p + 1e-6) - numerator_slope(omega_p - 1e-6)) / 2e-6
        return (numerator_slope(omega) - level) / gap

    integral, _ = quad(regularized, lower, upper, epsabs=1e-11, limit=500)
    logarithm = level * np.log((upper - omega_p) / (omega_p - lower))
    return frequency / (2.0 * float(delta)) * (integral + logarithm)


def solve_eq6(delta: float) -> dict:
    """Find lambda = i*Omega roots of Eq. (6) inside the incoherent branch."""
    samples = np.linspace(-0.98, 0.98, 121)
    values = np.array([imaginary_residual(delta, float(z)) for z in samples])
    roots = []
    for index in range(samples.size - 1):
        left, right = values[index], values[index + 1]
        if not (np.isfinite(left) and np.isfinite(right) and left * right < 0.0):
            continue
        location = float(
            brentq(lambda z: imaginary_residual(delta, z), float(samples[index]), float(samples[index + 1]))
        )
        pair = _pole_pair(delta, location)
        if pair is None:
            continue
        omega_p, frequency, coupling = pair
        residual = float(imaginary_residual(delta, location))
        residue = PI * frequency * (omega_p - frequency) / (4.0 * float(delta))
        shift = 1e-3
        up = _pole_pair(delta, location + shift)
        down = _pole_pair(delta, location - shift)
        sensitivity = None if up is None or down is None else (up[2] - down[2]) / (2.0 * shift)
        roots.append(
            {
                "z": location,
                "omega_p": omega_p,
                "Omega": frequency,
                "K": coupling,
                "imag_residual": residual,
                "real_gap": residue - coupling,
                "dK_dz": sensitivity,
                "branch_ok": incoherent_branch(delta, coupling),
            }
        )
    finite = int(np.isfinite(values).sum())
    if not roots:
        return {
            "status": "no valid root",
            "delta": float(delta),
            "n_sign_changes": 0,
            "n_finite_samples": finite,
            "roots": [],
        }
    return {
        "status": "ok" if len(roots) == 1 else "multiple roots",
        "delta": float(delta),
        "n_sign_changes": len(roots),
        "n_finite_samples": finite,
        "roots": roots,
    }


def complex_residual(delta: float, coupling: float, frequency: float, epsilon: float = 1e-5) -> tuple[float, float]:
    """Direct Eq. (6) integral at lambda = epsilon + i*Omega, split at the pole."""
    lower, upper = 1.0 - float(delta), 1.0 + float(delta)
    pole = float(np.sqrt(coupling * coupling + frequency * frequency))
    eigenvalue = epsilon + 1j * frequency

    def component(omega: float, part: str) -> float:
        speed = float(np.sqrt(max(0.0, omega * omega - coupling * coupling)))
        value = eigenvalue * omega * (omega - speed) / (eigenvalue**2 + omega * omega - coupling * coupling)
        return float(np.real(value) if part == "real" else np.imag(value)) / (2.0 * float(delta))

    real = 0.0
    imag = 0.0
    for part, target in (("real", "real"), ("imag", "imag")):
        left, _ = quad(lambda omega: component(omega, part), lower, pole, epsabs=1e-10, limit=400)
        right, _ = quad(lambda omega: component(omega, part), pole, upper, epsabs=1e-10, limit=400)
        if target == "real":
            real = left + right
        else:
            imag = left + right
    return real, imag


def load_records() -> tuple[list[dict], dict]:
    config = load_config()
    tolerance = float(config["rule_4_near_zero_tolerance"]["value"])
    death_level = float(config["rule_5_death"]["value"])
    lock_range = float(config["rule_7_locking"]["maximum_profile_rho_range"])
    if abs(tolerance - 0.001) > 1e-15 or abs(death_level - 1.0) > 1e-15:
        raise RuntimeError(f"archived near-zero rule is |rho|<{tolerance}, death fraction>={death_level}")
    if config["state_priority"] != ["Death", "Locking", "Partial Death", "Partial Locking", "Incoherence"]:
        raise RuntimeError(f"archived priority is {config['state_priority']}")
    if config["classifier_version"] != "v2.0.0":
        raise RuntimeError(config["classifier_version"])
    runs = query_runs(status="completed", p=1.0)
    if not runs:
        raise RuntimeError("no completed p=1 runs")
    pilot, cost_audit = load_method_c()
    records = []
    for run in runs:
        with np.load(run["output_npz"], allow_pickle=False) as data:
            omega = np.array(data["omega"], dtype=np.float64)
            rotation = np.array(data["rotation_numbers"], dtype=np.float64)
            starts = np.array(data["window_start"], dtype=np.float64)
            ends = np.array(data["window_end"], dtype=np.float64)
        if np.any(np.diff(omega) < 0):
            order = np.argsort(omega, kind="mergesort")
            omega = omega[order]
            rotation = rotation[:, order]
        rho = rotation[window_row(starts, ends, W3_BOUNDS)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
            classified = classify_window(cost_audit, pilot, omega, rho, config)
        segments = classified["segments"]
        plateaus = [row for row in segments if row["validated_nonzero_plateau"]]
        largest = max(plateaus, key=lambda row: row["length"]) if plateaus else None
        records.append(
            {
                "K": float(run["K"]),
                "Delta": float(run["Delta"]),
                "N": int(run["N"]),
                "trial_id": int(run["trial_id"]),
                "state": classified["state"],
                "near_fraction": float(classified["near_zero_fraction"]),
                "rho_range": float(classified["rho_range"]),
                "rho_min": float(rho.min()),
                "rho_max": float(rho.max()),
                "rho_mean": float(rho.mean()),
                "plateau_count": int(classified["validated_plateau_count"]),
                "largest_plateau_length": 0 if largest is None else int(largest["length"]),
                "largest_plateau_level": None if largest is None else float(largest["level"]),
                "n_segments": int(classified["n_segments"]),
                "sigma_floor": int(classified["sigma_hat_at_floor"]),
                "linear_segments": sum(row["mode"] == "linear" for row in segments),
                "max_segment_rho_span": 0.0 if not segments else max(float(row["rho_span"]) for row in segments),
                "max_span_length": 0 if not segments else int(max(segments, key=lambda row: row["rho_span"])["length"]),
                "lock_by_range": bool(classified["state"] == "Locking" and classified["rho_range"] <= lock_range),
                "partial_by_plateau": bool(classified["state"] == "Partial Locking" and classified["validated_plateau_count"] > 0),
            }
        )
    meta = {
        "config": config,
        "runs": runs,
        "tolerance": tolerance,
        "lock_range": lock_range,
        "n": int(records[0]["N"]) if records else 0,
    }
    return records, meta


def _runs(rows: list[dict]) -> list[dict]:
    compressed = []
    for row in rows:
        if not compressed or compressed[-1]["state"] != row["state"]:
            compressed.append({"state": row["state"], "k_start": row["K"], "k_end": row["K"], "n": 1})
        else:
            compressed[-1]["k_end"] = row["K"]
            compressed[-1]["n"] += 1
    return compressed


def transition_summary(rows: list[dict], coupling: float) -> dict:
    """Sustained exit from the initial Incoherence run. Returns are not assigned a K_exit."""
    rows = sorted(rows, key=lambda item: item["K"])
    runs = _runs(rows)
    applicable = incoherent_branch(rows[0]["Delta"], coupling)
    base = {
        "delta": rows[0]["Delta"],
        "K_H": coupling,
        "applicable": applicable,
        "runs": runs,
        "sequence": " | ".join(f"{item['state']} [{item['k_start']:.2f},{item['k_end']:.2f}]" for item in runs),
        "K_exit": None,
        "K_prev": None,
        "state_before": None,
        "state_after": None,
        "delta_K": None,
        "inside_gap": False,
        "grid_step": None,
        "status": "unset",
    }
    if abs(rows[0]["Delta"]) < 1e-15:
        base["status"] = "degenerate delta=0"
        return base
    if rows[0]["state"] != "Incoherence":
        base["status"] = "no initial incoherence"
        return base
    exit_index = next((index for index, row in enumerate(rows) if row["state"] != "Incoherence"), None)
    if exit_index is None:
        base["status"] = "remains incoherence"
        return base
    if any(row["state"] == "Incoherence" for row in rows[exit_index:]):
        base["status"] = "non-monotonic"
        base["state_before"] = rows[exit_index - 1]["state"]
        base["state_after"] = rows[exit_index]["state"]
        base["K_prev"] = rows[exit_index - 1]["K"]
        return base
    previous = rows[exit_index - 1]
    current = rows[exit_index]
    step = current["K"] - previous["K"]
    difference = current["K"] - coupling
    base.update(
        {
            "status": "sustained exit",
            "K_exit": current["K"],
            "K_prev": previous["K"],
            "state_before": previous["state"],
            "state_after": current["state"],
            "delta_K": difference,
            "grid_step": step,
            "inside_gap": previous["K"] < coupling <= current["K"] + 1e-12,
        }
    )
    return base


def scope_name(delta: float) -> str:
    if abs(delta) < 1e-15:
        return "degenerate"
    if delta <= MAIN_DELTA_MAX + 1e-12:
        return "main"
    return "exploratory"


def print_diagnostics(records: list[dict], summaries: list[dict]) -> None:
    wanted = {
        (0.05, 0.10),
        (0.05, 0.15),
        (0.10, 0.25),
        (0.10, 0.30),
        (0.15, 0.35),
        (0.15, 0.40),
        (0.15, 0.45),
        (0.20, 0.55),
        (0.20, 0.60),
        (0.20, 0.65),
        (0.20, 0.70),
        (0.20, 0.75),
        (0.225, 0.70),
        (0.225, 0.75),
        (0.225, 0.80),
        (0.25, 0.75),
        (0.25, 0.80),
        (0.25, 0.85),
    }
    print("ROTATION")
    for row in sorted(records, key=lambda item: (item["Delta"], item["K"])):
        key = (round(row["Delta"], 3), round(row["K"], 2))
        if key not in wanted:
            continue
        coverage = row["largest_plateau_length"] / row["N"]
        level = "NA" if row["largest_plateau_level"] is None else f"{row['largest_plateau_level']:.4f}"
        print(
            f"d={row['Delta']:.3f} K={row['K']:.2f} {row['state']} "
            f"range={row['rho_range']:.6f} near={row['near_fraction']:.4f} "
            f"rho=[{row['rho_min']:.4f},{row['rho_max']:.4f}] mean={row['rho_mean']:.4f} "
            f"plateaus={row['plateau_count']} largest={row['largest_plateau_length']} "
            f"cover={coverage:.3f} level={level} "
            f"lock_by_range={int(row['lock_by_range'])} partial_by_plateau={int(row['partial_by_plateau'])} "
            f"segments={row['n_segments']} linear={row['linear_segments']} "
            f"sigma_floor={row['sigma_floor']} max_seg_span={row['max_segment_rho_span']:.6f} "
            f"max_span_len={row['max_span_length']}"
        )
    print("TRANSITIONS")
    for item in summaries:
        difference = "NA" if item["delta_K"] is None else f"{item['delta_K']:+.4f}"
        exit_k = "NA" if item["K_exit"] is None else f"{item['K_exit']:.2f}"
        prev = "NA" if item["K_prev"] is None else f"{item['K_prev']:.2f}"
        print(
            f"d={item['delta']:.3f} scope={scope_name(item['delta'])} KH={item['K_H']:.4f} "
            f"applicable={int(item['applicable'])} status={item['status']} "
            f"Kprev={prev} Kexit={exit_k} before={item['state_before']} after={item['state_after']} "
            f"dK={difference} inside={int(item['inside_gap'])} step={item['grid_step']} "
            f"seq={item['sequence']}"
        )


def print_provenance(records: list[dict], meta: dict, eq6: list[dict]) -> None:
    runs = meta["runs"]
    database = Path(runs[0]["output_dir"]) / "index.sqlite"
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        totals = dict(
            connection.execute("SELECT COUNT(*) AS n, SUM(status='completed') AS completed FROM runs").fetchone()
        )
        provenance = list(
            connection.execute(
                """
                SELECT solver, dt, t_origin, t_end, master_seed, randomness_mode, seed_recipe,
                       network_seed, initial_phase_seed, frequency_seed, frequency_rule,
                       network_pairing_rule, phase_pairing_rule, model_version,
                       network_realization_id, initial_condition_id, paired_group_id, n, trial_id,
                       COUNT(*) AS rows
                FROM runs
                WHERE status='completed' AND ABS(p-1.0)<1e-9
                GROUP BY solver, dt, t_origin, t_end, master_seed, randomness_mode, seed_recipe,
                         network_seed, initial_phase_seed, frequency_seed, frequency_rule,
                         network_pairing_rule, phase_pairing_rule, model_version,
                         network_realization_id, initial_condition_id, paired_group_id, n, trial_id
                """
            ).fetchall()
        )
    print("DATA")
    print("index", totals)
    print("p1 profiles", len(records))
    print("provenance groups", len(provenance))
    for row in provenance:
        print(dict(row))
    k_values = np.array(sorted({row["K"] for row in records}))
    deltas = np.array(sorted({row["Delta"] for row in records}))
    print("K", k_values[0], k_values[-1], "steps", np.unique(np.round(np.diff(k_values), 10)).tolist(), "count", k_values.size)
    print("delta", deltas[0], deltas[-1], "steps", np.unique(np.round(np.diff(deltas), 10)).tolist(), "count", deltas.size)
    by_delta: dict[float, set[float]] = {}
    for row in records:
        by_delta.setdefault(row["Delta"], set()).add(row["K"])
    print(
        "K counts",
        {length: sum(len(values) == length for values in by_delta.values()) for length in sorted({len(v) for v in by_delta.values()})},
    )
    print("classifier", meta["config"]["classifier_version"], "tol", meta["tolerance"], "lock", meta["lock_range"], "path", CLASSIFIER_DIR)
    sample = runs[0]
    adjacency = make_adjacency(int(sample["N"]), 1.0, int(provenance[0]["network_seed"]))
    stored = json.loads(Path(sample["output_json"]).read_text())
    print(
        "graph",
        "diag",
        float(adjacency.diagonal().sum()),
        "upper",
        int(np.sum(adjacency[np.triu_indices(sample["N"], 1)])),
        "hash",
        sha256_array(adjacency) == stored["adjacency_sha256"],
    )
    counts: dict[str, int] = {}
    for row in records:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    print("states", counts)
    print("SERIES")
    for delta in deltas:
        printed = series_eq7(float(delta))
        added = additive_transcription(float(delta))
        print(
            f"d={delta:.3f} paper={printed:.5f} additive={added:.5f} "
            f"gap={printed - added:+.5f} branch={int(incoherent_branch(float(delta), printed))} edge={1.0 - delta:.3f}"
        )
    crossing = brentq(lambda value: series_eq7(value) - (1.0 - value), 0.2, 0.4)
    print(f"series_crosses_branch_edge {crossing:.4f}")
    print("EQ6")
    for item in eq6:
        if not item["roots"]:
            print(
                f"d={item['delta']:.3f} status={item['status']} finite={item['n_finite_samples']} "
                f"series={series_eq7(item['delta']):.5f}"
            )
            continue
        for root in item["roots"]:
            real, imag = complex_residual(item["delta"], root["K"], root["Omega"])
            print(
                f"d={item['delta']:.3f} status={item['status']} z={root['z']:.4f} K={root['K']:.5f} "
                f"Omega={root['Omega']:.4f} wp={root['omega_p']:.4f} ImPV={root['imag_residual']:.2e} "
                f"ReGap={root['real_gap']:.2e} dKdz={root['dK_dz']} "
                f"complexRe={real:.5f} complexIm={imag:.3e} series={series_eq7(item['delta']):.5f} "
                f"series_gap={root['K'] - series_eq7(item['delta']):+.5f}"
            )


def draw_figure(records: list[dict], summaries: list[dict], eq6: list[dict]) -> None:
    deltas = np.array(sorted({row["Delta"] for row in records}))
    ks = np.array(sorted({row["K"] for row in records}))
    delta_edges = np.concatenate(([deltas[0] - 0.0125], 0.5 * (deltas[:-1] + deltas[1:]), [deltas[-1] + 0.0125]))
    k_edges = np.concatenate(([ks[0] - 0.025], 0.5 * (ks[:-1] + ks[1:]), [ks[-1] + 0.025]))
    delta_index = {round(value, 10): index for index, value in enumerate(deltas)}
    k_index = {round(value, 10): index for index, value in enumerate(ks)}
    code = {state: index for index, state in enumerate(PHASE_STATES)}
    field = np.full((ks.size, deltas.size), np.nan)
    for row in records:
        field[k_index[round(row["K"], 10)], delta_index[round(row["Delta"], 10)]] = code[row["state"]]
    cmap = ListedColormap(PHASE_COLORS)
    cmap.set_bad(MISSING_COLOR)
    norm = BoundaryNorm(np.arange(-0.5, len(PHASE_COLORS) + 0.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.6))
    axes[0].pcolormesh(delta_edges, k_edges, np.ma.masked_invalid(field), cmap=cmap, norm=norm, shading="flat")
    fine = np.linspace(0.005, 0.45, 400)
    main_delta, main_k, extra_delta, extra_k = [], [], [], []
    for value in fine:
        coupling = series_eq7(float(value))
        if not incoherent_branch(float(value), coupling):
            continue
        if value <= MAIN_DELTA_MAX:
            main_delta.append(value)
            main_k.append(coupling)
        else:
            extra_delta.append(value)
            extra_k.append(coupling)
    axes[0].plot(main_delta, main_k, color=HOPF_COLOR, lw=2.2, label="Eq. (7), main range")
    axes[0].plot(extra_delta, extra_k, color=HOPF_COLOR, lw=2.0, ls="--", label="Eq. (7), exploratory")
    edge = np.linspace(0.0, 1.0, 200)
    axes[0].plot(edge, 1.0 - edge, color="0.15", lw=0.9, ls=":", label="Incoherent-branch edge K=1−δ")
    axes[0].axvline(MAIN_DELTA_MAX, color="0.2", lw=0.8, ls="--")
    eq6_delta, eq6_k = [], []
    for item in eq6:
        if item["status"] != "ok" or not item["roots"]:
            continue
        eq6_delta.append(item["delta"])
        eq6_k.append(item["roots"][0]["K"])
    axes[0].scatter(eq6_delta, eq6_k, s=28, marker="x", color="black", zorder=4, label="Eq. (6) root")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.05, 3.05)
    axes[0].set_xlabel("Frequency half-width δ")
    axes[0].set_ylabel("Coupling K")
    axes[0].set_title("Hopf boundary and observed states")
    handles = [Patch(facecolor=color, edgecolor="none", label=state) for state, color in zip(PHASE_STATES, PHASE_COLORS)]
    handles.append(Patch(facecolor=MISSING_COLOR, edgecolor="none", label="No data"))
    handles.extend(
        [
            Line2D([0], [0], color=HOPF_COLOR, lw=2.2, label="Eq. (7), 0<δ≤0.20"),
            Line2D([0], [0], color=HOPF_COLOR, lw=2.0, ls="--", label="Eq. (7), exploratory"),
            Line2D([0], [0], color="0.15", lw=0.9, ls=":", label="Incoherent-branch edge K=1−δ"),
            Line2D([0], [0], color="black", marker="x", ls="none", label="Eq. (6) root"),
            Line2D([0], [0], color="0.2", lw=0.8, ls="--", label="Main-scope edge δ=0.20"),
        ]
    )
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.46, 0.98), fontsize=7.2, framealpha=0.95)

    follow = []
    for item in summaries:
        if scope_name(item["delta"]) != "main" or not item["applicable"] or item["delta_K"] is None:
            if scope_name(item["delta"]) == "main" and item["status"] == "non-monotonic":
                follow.append(item["delta"])
            continue
        outside = abs(item["delta_K"]) > 0.05 + 1e-9
        axes[1].scatter(
            item["delta"],
            item["delta_K"],
            s=42 if outside else 32,
            marker="s" if outside else "o",
            color=HOPF_COLOR,
            zorder=3,
        )
    axes[1].axhline(0.0, color="black", lw=1.0)
    axes[1].axhspan(-0.05, 0.05, color="0.85", zorder=0)
    for value in follow:
        axes[1].axvline(value, color="0.35", lw=0.8, ls=":")
        axes[1].text(value - 0.004, 0.072, "no K_exit", rotation=90, va="top", ha="right", fontsize=7, color="0.25")
    axes[1].set_xlim(-0.01, 0.245)
    axes[1].set_ylim(-0.02, 0.08)
    axes[1].set_xlabel("Frequency half-width δ")
    axes[1].set_ylabel(r"$K_{\mathrm{exit}} - K_H^{(7)}$")
    axes[1].set_title("Hopf boundary discrepancy")
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.0, label="Zero difference"),
            Patch(facecolor="0.85", edgecolor="none", label="One K grid step (±0.05)"),
            Line2D([0], [0], color=HOPF_COLOR, marker="o", ls="none", label="Sustained exit, main scope"),
            Line2D([0], [0], color=HOPF_COLOR, marker="s", ls="none", label="Outside one grid step"),
            Line2D([0], [0], color="0.35", lw=0.8, ls=":", label="Non-monotonic, no ΔK"),
        ],
        loc="upper left",
        fontsize=8,
        framealpha=0.95,
    )
    fig.suptitle("N=1000, p=1, trial 0, W3=[12000, 14000], Classifier v2.0.0", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    eq6 = [solve_eq6(delta) for delta in EQ6_DELTAS]
    records, meta = load_records()
    grouped: dict[float, list[dict]] = {}
    for row in records:
        grouped.setdefault(row["Delta"], []).append(row)
    summaries = [transition_summary(grouped[delta], series_eq7(delta)) for delta in sorted(grouped)]
    print_provenance(records, meta, eq6)
    print_diagnostics(records, summaries)
    main_rows = [item for item in summaries if scope_name(item["delta"]) == "main"]
    scored = [item for item in main_rows if item["applicable"] and item["status"] == "sustained exit"]
    print(
        "MAIN",
        "n",
        len(main_rows),
        "applicable",
        sum(item["applicable"] for item in main_rows),
        "sustained",
        len(scored),
        "inside",
        sum(item["inside_gap"] for item in scored),
        "nonmonotonic",
        [item["delta"] for item in main_rows if item["status"] == "non-monotonic"],
    )
    if scored:
        diffs = np.array([item["delta_K"] for item in scored])
        print(
            "DK",
            "min",
            float(diffs.min()),
            "median",
            float(np.median(diffs)),
            "max",
            float(diffs.max()),
            "outside_step",
            [item["delta"] for item in scored if abs(item["delta_K"]) > 0.05 + 1e-9],
        )
        after: dict[str, int] = {}
        for item in scored:
            after[item["state_after"]] = after.get(item["state_after"], 0) + 1
        print("after", after)
    draw_figure(records, summaries, eq6)
    print(f"figure {FIGURE_PATH}")


if __name__ == "__main__":
    main()
