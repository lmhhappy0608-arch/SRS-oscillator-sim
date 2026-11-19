"""Compare Classifier v2 Death onsets with the Ariaratnam–Strogatz existence boundary.

Reads saved Stage 1 p=1 profiles and labels them with Classifier v2.
The existence threshold is computed from Eq. (8)–(9)
alone. This script does not integrate the ODE, edit thresholds, or rewrite
the classifier archive.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-death-boundary")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLASSIFIER_DIR = ROOT / "stage2_classifier" / "classifier_v2"
sys.path.insert(0, str(CLASSIFIER_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "penalty_calibration"))

from classify import classify_window, load_config, window_row  # noqa: E402
from method_c_separated import load_method_c  # noqa: E402
from stage1.model import make_adjacency, sha256_array  # noqa: E402
from stage1.query import query_runs  # noqa: E402

W3_BOUNDS = (12000.0, 14000.0)
FIGURE_PATH = HERE / "death_validation_figure.png"
DEATH_COLOR = "#762a83"
OTHER_COLOR = "#dceaf7"
MISSING_COLOR = "#d0d0d0"
SN_COLOR = "#d95f02"
EP_COLOR = "#1b9e77"


def mean_field_factor(sigma: float, delta: float) -> float:
    """F_delta(sigma) in Eq. (8). Sigma must be at least the largest frequency."""
    sigma = float(sigma)
    delta = float(delta)
    if delta == 0.0:
        if sigma < 1.0 - 1e-12:
            raise ValueError("sigma is below the frequency")
        return 1.0 + float(np.sqrt(max(0.0, 1.0 - 1.0 / sigma**2)))
    if delta <= 0.0:
        raise ValueError("delta must be non-negative")
    upper = 1.0 + delta
    lower = 1.0 - delta
    if sigma < upper - 1e-12:
        raise ValueError("sigma is below the largest frequency")

    def antiderivative(omega: float) -> float:
        radicand = max(0.0, sigma * sigma - omega * omega)
        return omega * np.sqrt(radicand) + sigma * sigma * np.arcsin(float(np.clip(omega / sigma, -1.0, 1.0)))

    integral = 0.5 * (antiderivative(upper) - antiderivative(lower))
    return 1.0 + integral / (2.0 * delta * sigma)


def slope_factor(sigma: float, delta: float) -> float:
    """F - sigma dF/dsigma. Its sign is the sign of dK/dsigma."""
    sigma = float(sigma)
    delta = float(delta)
    if delta == 0.0:
        radicand = 1.0 - 1.0 / sigma**2
        if radicand <= 1e-18:
            return -np.inf
        return 1.0 + (1.0 - 2.0 / sigma**2) / np.sqrt(radicand)
    upper = 1.0 + delta
    lower = 1.0 - delta
    upper_term = upper * np.sqrt(max(0.0, sigma * sigma - upper * upper))
    lower_term = lower * np.sqrt(max(0.0, sigma * sigma - lower * lower))
    return 1.0 + (upper_term - lower_term) / (2.0 * delta * sigma)


def endpoint_slope(delta: float) -> float:
    """Limit of F - sigma F' as sigma approaches the largest frequency from above."""
    delta = float(delta)
    if delta == 0.0:
        return -np.inf
    return 1.0 - (1.0 - delta) / (np.sqrt(delta) * (1.0 + delta))


def coupling_of(sigma: float, delta: float) -> float:
    return float(sigma) / mean_field_factor(sigma, delta)


def existence_boundary(delta: float, range_factor: float = 80.0) -> dict:
    """Return the death-solution existence threshold for one delta.

    The search coordinate is sigma = omega_max * cosh(u), so the square-root
    endpoint is analytic in u. A bounded minimization checks the critical points.
    """
    delta = float(delta)
    omega_max = 1.0 + delta if delta > 0.0 else 1.0
    try:
        slope_at_end = endpoint_slope(delta)
        u_max = float(np.arccosh(range_factor))
        samples = np.linspace(1e-8, u_max, 6000)
        sigmas = omega_max * np.cosh(samples)
        slopes = np.array([slope_factor(value, delta) for value in sigmas])
        roots: list[float] = []
        for index in range(samples.size - 1):
            left, right = slopes[index], slopes[index + 1]
            if not np.isfinite(left) or not np.isfinite(right) or left * right >= 0.0:
                continue
            root = float(
                brentq(lambda value: slope_factor(value, delta), float(sigmas[index]), float(sigmas[index + 1]), xtol=1e-14)
            )
            if not roots or abs(root - roots[-1]) > 1e-8:
                roots.append(root)
        candidates = [(omega_max, coupling_of(omega_max, delta), "endpoint")]
        for root in roots:
            kind = "local-min" if slope_factor(root + 1e-6, delta) > 0.0 else "local-max"
            candidates.append((root, coupling_of(root, delta), kind))
        minimized = minimize_scalar(
            lambda width: coupling_of(omega_max * np.cosh(width), delta),
            bounds=(0.0, u_max),
            method="bounded",
            options={"xatol": 1e-14},
        )
        if not minimized.success:
            return {"status": "unresolved", "reason": "bounded minimization failed"}
        sigma_opt = float(omega_max * np.cosh(minimized.x))
        best = min(candidates, key=lambda item: item[1])
        if abs(float(minimized.fun) - best[1]) > 1e-8:
            return {
                "status": "unresolved",
                "reason": f"minimizer {minimized.fun:.8f} disagrees with critical value {best[1]:.8f}",
            }
        if slope_factor(omega_max * range_factor, delta) <= 0.0:
            return {"status": "unresolved", "reason": "slope is still negative at the search edge"}
        mechanism = "end-point" if best[2] == "endpoint" else "saddle-node"
        return {
            "status": "ok",
            "delta": delta,
            "mechanism": mechanism,
            "sigma": float(best[0]),
            "K_exist": float(best[1]),
            "K_endpoint": coupling_of(omega_max, delta),
            "slope_at_endpoint": float(slope_at_end) if np.isfinite(slope_at_end) else -np.inf,
            "n_roots": len(roots),
            "optimizer_gap": float(minimized.fun) - float(best[1]),
            "sigma_optimizer": sigma_opt,
        }
    except Exception as exc:
        return {"status": "unresolved", "reason": f"{type(exc).__name__}: {exc}"}


def quadrature_check() -> list[str]:
    """Compare the antiderivative with scipy.quad away from and on the endpoint."""
    lines = []
    probes = ((0.5, 2.0), (0.5, 1.5), (0.1, 1.5), (0.025, 1.2), (1.0, 2.5))
    worst = 0.0
    for delta, sigma in probes:
        analytic = mean_field_factor(sigma, delta)
        upper = 1.0 + delta
        lower = 1.0 - delta
        numeric, error = quad(
            lambda omega: np.sqrt(max(0.0, 1.0 - (omega / sigma) ** 2)),
            lower,
            upper,
            epsabs=1e-12,
            limit=500,
        )
        numeric = 1.0 + numeric / (2.0 * delta)
        gap = abs(analytic - numeric)
        worst = max(worst, gap)
        lines.append(f"quad d={delta:.3f} sigma={sigma:.3f} |F_analytic-F_quad|={gap:.2e} quad_err={error:.1e}")
    # Delta -> 0 should recover the single-frequency factor.
    limit_gap = abs(mean_field_factor(1.5, 1e-4) - mean_field_factor(1.5, 0.0))
    lines.append(f"delta->0 gap at sigma=1.5: {limit_gap:.2e}")
    lines.append(f"worst quadrature gap: {worst:.2e}")
    narrow = existence_boundary(0.2, 40.0)
    wide = existence_boundary(0.2, 200.0)
    lines.append(
        "search range K(d=0.2): "
        f"{narrow['K_exist']:.10f} at x80, {wide['K_exist']:.10f} at x200, "
        f"diff={wide['K_exist'] - narrow['K_exist']:+.2e}"
    )
    zero = existence_boundary(0.0)
    exact = (2.0 / np.sqrt(3.0)) / 1.5
    lines.append(f"delta=0 K={zero['K_exist']:.10f} exact 4/(3*sqrt(3))={exact:.10f} diff={zero['K_exist'] - exact:+.2e}")
    return lines


def high_frequency_tail(rho: np.ndarray, tolerance: float) -> dict:
    alive = np.abs(rho) >= tolerance
    count = int(alive.sum())
    if count == 0:
        return {"n_alive": 0, "high_frequency_suffix": True, "alive_omega_min": None, "alive_omega_max": None}
    suffix = bool(np.all(alive[-count:])) and (count == rho.size or bool(np.all(~alive[:-count])))
    return {"n_alive": count, "high_frequency_suffix": suffix}


def load_records() -> tuple[list[dict], dict]:
    config = load_config()
    tolerance = float(config["rule_4_near_zero_tolerance"]["value"])
    death_level = float(config["rule_5_death"]["value"])
    if abs(tolerance - 0.001) > 1e-15 or abs(death_level - 1.0) > 1e-15:
        raise RuntimeError(f"archived death rule is |rho|<{tolerance}, fraction>={death_level}")
    if config["state_priority"][:2] != ["Death", "Locking"]:
        raise RuntimeError("archived priority does not start with Death, Locking")
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
        tail = high_frequency_tail(rho, tolerance)
        records.append(
            {
                "K": float(run["K"]),
                "Delta": float(run["Delta"]),
                "N": int(run["N"]),
                "trial_id": int(run["trial_id"]),
                "state": classified["state"],
                "near_fraction": float(classified["near_zero_fraction"]),
                "rho_range": float(classified["rho_range"]),
                "n_alive": tail["n_alive"],
                "high_frequency_suffix": tail["high_frequency_suffix"],
                "alive_omega_min": None if tail["n_alive"] == 0 else float(omega[np.abs(rho) >= tolerance].min()),
                "alive_omega_max": None if tail["n_alive"] == 0 else float(omega[np.abs(rho) >= tolerance].max()),
                "omega_max": float(omega[-1]),
            }
        )
    meta = {"config": config, "runs": runs, "tolerance": tolerance}
    return records, meta


def death_onsets(records: list[dict], theory: dict[float, dict]) -> list[dict]:
    grouped: dict[float, list[dict]] = {}
    for row in records:
        grouped.setdefault(row["Delta"], []).append(row)
    summaries = []
    for delta in sorted(grouped):
        rows = sorted(grouped[delta], key=lambda item: item["K"])
        death_indices = [index for index, row in enumerate(rows) if row["state"] == "Death"]
        predicted = theory[delta]
        if not death_indices:
            onset = {"status": "not observed", "K": None, "previous_K": None, "previous_state": None}
        elif death_indices[0] == 0:
            onset = {
                "status": "left-censored",
                "K": rows[0]["K"],
                "previous_K": None,
                "previous_state": None,
                "persists": death_indices == list(range(len(rows))),
            }
        else:
            index = death_indices[0]
            previous = rows[index - 1]
            later = rows[index:]
            persists = all(row["state"] == "Death" for row in later)
            reversals = [
                (later[pos - 1]["K"], later[pos]["K"], later[pos]["state"])
                for pos in range(1, len(later))
                if later[pos]["state"] != "Death"
            ]
            onset = {
                "status": "observed",
                "K": rows[index]["K"],
                "previous_K": previous["K"],
                "previous_state": previous["state"],
                "previous_fraction": previous["near_fraction"],
                "previous_alive": previous["n_alive"],
                "previous_suffix": previous["high_frequency_suffix"],
                "previous_alive_omega_min": previous["alive_omega_min"],
                "previous_alive_omega_max": previous["alive_omega_max"],
                "previous_omega_max": previous["omega_max"],
                "previous_range": previous["rho_range"],
                "persists": persists,
                "reversals": reversals,
            }
        step = None if onset["previous_K"] is None or onset["K"] is None else onset["K"] - onset["previous_K"]
        if predicted["status"] != "ok" or onset["status"] != "observed":
            difference = None
            inside = False
        else:
            difference = onset["K"] - predicted["K_exist"]
            inside = onset["previous_K"] < predicted["K_exist"] <= onset["K"] + 1e-12
        summaries.append(
            {
                "delta": delta,
                "theory": predicted,
                "onset": onset,
                "difference": difference,
                "inside_gap": inside,
                "grid_step": step,
                "rows": rows,
            }
        )
    return summaries


def print_report(records: list[dict], meta: dict, theory: dict[float, dict], summaries: list[dict], checks: list[str]) -> None:
    runs = meta["runs"]
    database = Path(runs[0]["output_dir"]) / "index.sqlite"
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        totals = dict(
            connection.execute(
                "SELECT COUNT(*) AS n, SUM(status='completed') AS completed FROM runs"
            ).fetchone()
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
    print("K", k_values[0], k_values[-1], "step", np.unique(np.round(np.diff(k_values), 10)).tolist(), "count", k_values.size)
    print("delta", deltas[0], deltas[-1], "step", np.unique(np.round(np.diff(deltas), 10)).tolist(), "count", deltas.size)
    by_delta = {}
    for row in records:
        by_delta.setdefault(row["Delta"], set()).add(row["K"])
    print("K counts", {length: sum(len(values) == length for values in by_delta.values()) for length in sorted({len(v) for v in by_delta.values()})})
    print("classifier", meta["config"]["classifier_version"], "tol", meta["tolerance"], "path", CLASSIFIER_DIR)
    sample = runs[0]
    adjacency = make_adjacency(int(sample["N"]), 1.0, int(provenance[0]["network_seed"]))
    stored = json.loads(Path(sample["output_json"]).read_text())
    print(
        "graph",
        "diag", float(adjacency.diagonal().sum()),
        "upper", int(np.sum(adjacency[np.triu_indices(sample["N"], 1)])),
        "hash", sha256_array(adjacency) == stored["adjacency_sha256"],
    )
    print("CHECKS")
    for line in checks:
        print(line)
    print("ONSETS")
    for item in summaries:
        predicted = item["theory"]
        onset = item["onset"]
        difference = "NA" if item["difference"] is None else f"{item['difference']:+.4f}"
        print(
            f"d={item['delta']:.3f} {predicted.get('mechanism', predicted['status'])} "
            f"Kexist={predicted.get('K_exist', float('nan')):.4f} "
            f"status={onset['status']} Kobs={onset['K']} prev={onset['previous_state']} "
            f"Kprev={onset['previous_K']} dK={difference} inside={int(item['inside_gap'])} "
            f"persist={onset.get('persists')} frac={onset.get('previous_fraction')} "
            f"alive={onset.get('previous_alive')} suffix={onset.get('previous_suffix')} "
            f"wmin={onset.get('previous_alive_omega_min')} wmax_alive={onset.get('previous_alive_omega_max')} "
            f"wmax={onset.get('previous_omega_max')} roots={predicted.get('n_roots')} "
            f"sigma={predicted.get('sigma')}"
        )
    valid = [item for item in summaries if item["difference"] is not None]
    diffs = np.array([item["difference"] for item in valid])
    print("SUMMARY n", len(valid), "median", float(np.median(diffs)), "max_abs", float(np.max(np.abs(diffs))))
    print("inside_gap", sum(item["inside_gap"] for item in valid), "/", len(valid))
    print("not observed", [item["delta"] for item in summaries if item["onset"]["status"] == "not observed"])
    print("censored", [item["delta"] for item in summaries if item["onset"]["status"] == "left-censored"])
    print("unresolved", [item["delta"] for item in summaries if item["theory"]["status"] != "ok"])
    print("reversals", [(item["delta"], item["onset"].get("reversals")) for item in summaries if item["onset"].get("reversals")])
    print("nonpersist", [item["delta"] for item in summaries if item["onset"].get("persists") is False])
    for mechanism in ("saddle-node", "end-point"):
        chosen = [item for item in valid if item["theory"]["mechanism"] == mechanism]
        states: dict[str, int] = {}
        for item in chosen:
            state = item["onset"]["previous_state"]
            states[state] = states.get(state, 0) + 1
        suffix = sum(bool(item["onset"].get("previous_suffix")) for item in chosen)
        print(mechanism, "n", len(chosen), "previous", states, "high_suffix", suffix, "deltas", [item["delta"] for item in chosen])


def draw_figure(records: list[dict], summaries: list[dict], switch: float) -> None:
    deltas = np.array(sorted({row["Delta"] for row in records}))
    ks = np.array(sorted({row["K"] for row in records}))
    delta_edges = np.concatenate(([deltas[0] - 0.0125], 0.5 * (deltas[:-1] + deltas[1:]), [deltas[-1] + 0.0125]))
    k_edges = np.concatenate(([ks[0] - 0.025], 0.5 * (ks[:-1] + ks[1:]), [ks[-1] + 0.025]))
    delta_index = {round(value, 10): index for index, value in enumerate(deltas)}
    k_index = {round(value, 10): index for index, value in enumerate(ks)}
    field = np.full((ks.size, deltas.size), np.nan)
    for row in records:
        field[k_index[round(row["K"], 10)], delta_index[round(row["Delta"], 10)]] = 1.0 if row["state"] == "Death" else 0.0
    cmap = ListedColormap([OTHER_COLOR, DEATH_COLOR])
    cmap.set_bad(MISSING_COLOR)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.6))
    axes[0].pcolormesh(delta_edges, k_edges, np.ma.masked_invalid(field), cmap=cmap, norm=norm, shading="flat")
    fine = np.linspace(0.0, 1.0, 401)
    fine_boundary = [existence_boundary(float(value)) for value in fine]
    sn_delta = [value for value, item in zip(fine, fine_boundary) if item["status"] == "ok" and item["mechanism"] == "saddle-node"]
    sn_k = [item["K_exist"] for item in fine_boundary if item["status"] == "ok" and item["mechanism"] == "saddle-node"]
    ep_delta = [value for value, item in zip(fine, fine_boundary) if item["status"] == "ok" and item["mechanism"] == "end-point"]
    ep_k = [item["K_exist"] for item in fine_boundary if item["status"] == "ok" and item["mechanism"] == "end-point"]
    axes[0].plot(sn_delta, sn_k, color=SN_COLOR, lw=2.0, label="Saddle-node existence boundary")
    axes[0].plot(ep_delta, ep_k, color=EP_COLOR, lw=2.0, ls="--", label="End-point existence boundary")
    axes[0].axvline(switch, color="0.25", lw=0.8, ls=":", label=f"Mechanism switch δ={switch:.3f}")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.05, 3.05)
    axes[0].set_xlabel("Frequency half-width δ")
    axes[0].set_ylabel("Coupling K")
    axes[0].set_title("Death region and theoretical existence boundary")
    handles = [
        Patch(facecolor=DEATH_COLOR, edgecolor="none", label="Classifier v1 Death"),
        Patch(facecolor=OTHER_COLOR, edgecolor="none", label="Classifier v1 non-Death"),
        Patch(facecolor=MISSING_COLOR, edgecolor="none", label="No data"),
        Line2D([0], [0], color=SN_COLOR, lw=2.0, label="Saddle-node existence boundary"),
        Line2D([0], [0], color=EP_COLOR, lw=2.0, ls="--", label="End-point existence boundary"),
        Line2D([0], [0], color="0.25", lw=0.8, ls=":", label=f"Mechanism switch δ={switch:.3f}"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.95)

    for item in summaries:
        if item["difference"] is None:
            continue
        color = SN_COLOR if item["theory"]["mechanism"] == "saddle-node" else EP_COLOR
        marker = "o" if item["theory"]["mechanism"] == "saddle-node" else "s"
        axes[1].scatter(item["delta"], item["difference"], color=color, marker=marker, s=28, zorder=3)
    axes[1].axhline(0.0, color="black", lw=1.0, label="Zero difference")
    axes[1].axhspan(-0.05, 0.05, color="0.85", zorder=0, label="One K grid step (±0.05)")
    axes[1].plot([], [], color=SN_COLOR, marker="o", ls="none", label="Saddle-node δ")
    axes[1].plot([], [], color=EP_COLOR, marker="s", ls="none", label="End-point δ")
    axes[1].axvline(switch, color="0.25", lw=0.8, ls=":")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.012, 0.068)
    axes[1].set_xlabel("Frequency half-width δ")
    axes[1].set_ylabel(r"$K_{\mathrm{observed}} - K_{\mathrm{exist}}$")
    axes[1].set_title("Boundary discrepancy")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle("N=1000, p=1, trial 0, W3=[12000, 14000], Classifier v2.0.0", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    checks = quadrature_check()
    records, meta = load_records()
    deltas = sorted({row["Delta"] for row in records})
    theory = {delta: existence_boundary(delta) for delta in deltas}
    # The switch is a property of the formula, not of the scanned deltas.
    switch = float(brentq(endpoint_slope, 0.2, 0.4))
    summaries = death_onsets(records, theory)
    print_report(records, meta, theory, summaries, checks)
    print(f"switch_delta {switch:.6f}")
    draw_figure(records, summaries, switch)
    print(f"figure {FIGURE_PATH}")


if __name__ == "__main__":
    main()
