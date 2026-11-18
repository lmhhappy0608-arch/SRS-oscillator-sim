"""Finite-network evidence near the annealed all-drifting Hopf boundary.

The continuum Hopf boundary is a discrete eigenvalue of a stationary phase
density. A finite quenched network does not have that density as a fixed
point. This pilot asks whether the collective pulse changes across the
known boundary. It does not estimate a quenched Hopf eigenvalue and it does
not refit the five-state classifier.

The amplitude rule was read from one p=1 probe (realization 0, T=1200) and
then frozen. A spectral peak near Omega_H is not by itself evidence of
instability: on the weakly damped side the finite-N pulse is already
narrowband. The discriminating observation is a jump in the demeaned
amplitude of the collective cosine.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-quenched-hopf-pilot")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stage1.model import make_adjacency, make_buffers, natural_frequencies, rk4_step_copy
from stage1.seeds import derive_seed

HERE = Path(__file__).resolve().parent
THEORY_CSV = HERE.parent / "er_annealed_hopf" / "er_annealed_hopf.csv"
CSV_PATH = HERE / "quenched_hopf_pilot.csv"
FIGURE_SERIES = HERE / "quenched_hopf_timeseries.png"
FIGURE_SCAN = HERE / "quenched_hopf_scan.png"
SUMMARY_PATH = HERE / "quenched_hopf_pilot_summary.json"

N_OSC = 1000
DT = 0.1
T_END = 1200.0
SAMPLE_DT = 0.2
P_VALUES = (1.0, 0.8, 0.4, 0.2)
DELTAS = (0.05, 0.20)
OFFSETS = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10)
REALIZATIONS = (0, 1, 2)
PERTURBATION = 0.05
SEPARATION_RATIO = 3.0
FREQUENCY_TOL = 0.06
NEAR_ONSET = 0.02
FREQUENCY_RULE = "deterministic_linspace_1_minus_delta_to_1_plus_delta"
MASTER = "quenched-hopf-pilot-v1"


def load_theory() -> dict[tuple[float, float], dict]:
    table = {}
    with THEORY_CSV.open() as handle:
        for row in csv.DictReader(handle):
            key = (float(row["p"]), float(row["delta"]))
            if key[0] in P_VALUES and key[1] in DELTAS:
                table[key] = {
                    "K_degree": float(row["K_H_degree"]),
                    "K_scaling": float(row["K_H_scaling"]),
                    "Omega": float(row["Omega_H"]),
                }
    expected = {(p, delta) for p in P_VALUES for delta in DELTAS}
    missing = expected - set(table)
    if missing:
        raise RuntimeError(f"theory table is missing {sorted(missing)}")
    return table


def sample_incoherent(omega: np.ndarray, local_a: np.ndarray, rng: np.random.Generator, nphi: int = 4096) -> tuple[np.ndarray, int]:
    """Draw phases from the drifting Adler density, or the stable lock if a>=omega.

    The density is the stationary law of dtheta/dt = omega - a sin(theta).
    It is the continuum incoherent profile only when every oscillator drifts
    and the collective pulse is exactly one. The returned count is the number
    of oscillators placed on a locked fixed point instead.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    sinphi = np.sin(phi)
    quantiles = rng.random(omega.size)
    theta = np.empty(omega.size, dtype=np.float64)
    pinned = 0
    for index, (frequency, coefficient) in enumerate(zip(omega, local_a)):
        frequency = float(frequency)
        coefficient = float(coefficient)
        if coefficient >= frequency - 1.0e-12:
            pinned += 1
            ratio = frequency / max(coefficient, 1.0e-12)
            theta[index] = np.arcsin(np.clip(ratio, -1.0, 1.0))
            continue
        beta = np.sqrt(max(frequency * frequency - coefficient * coefficient, 0.0))
        density = beta / (frequency - coefficient * sinphi)
        cumulative = np.cumsum(density)
        cumulative /= cumulative[-1]
        theta[index] = np.interp(float(quantiles[index]), cumulative, phi)
    return theta, pinned


def sampler_check() -> dict:
    rng = np.random.default_rng(1)
    frequency = np.full(20000, 1.0)
    coefficient = 0.30
    theta, pinned = sample_incoherent(frequency, np.full(frequency.shape, coefficient), rng)
    beta = np.sqrt(1.0 - coefficient * coefficient)
    expected_sin = coefficient / (1.0 + beta)
    cos_mean = float(np.mean(np.cos(theta)))
    sin_mean = float(np.mean(np.sin(theta)))
    passed = pinned == 0 and abs(cos_mean) < 0.02 and abs(sin_mean - expected_sin) < 0.01
    return {
        "pinned": pinned,
        "mean_cos": cos_mean,
        "mean_sin": sin_mean,
        "expected_sin": expected_sin,
        "passed": passed,
    }


def local_coefficients(coupling: float, adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    return coupling * degree / float(N_OSC)


def integrate_collective(
    theta: np.ndarray,
    omega: np.ndarray,
    adjacency: np.ndarray,
    coupling: float,
    dt: float,
    t_end: float,
    sample_dt: float,
) -> dict[str, np.ndarray]:
    buffers = make_buffers(N_OSC)
    sample_every = int(round(sample_dt / dt))
    if abs(sample_every * dt - sample_dt) > 1.0e-12:
        raise ValueError("sample_dt must be an integer multiple of dt")
    nsteps = int(round(t_end / dt))
    nsave = nsteps // sample_every + 1
    cosine = np.empty(nsave)
    field_mean = np.empty(nsave)
    field_std = np.empty(nsave)
    pulse = np.empty(N_OSC)
    field = np.empty(N_OSC)
    phase = np.array(theta, dtype=np.float64, copy=True)
    saved = 0
    for step in range(nsteps + 1):
        if step % sample_every == 0:
            np.cos(phase, out=pulse)
            cosine[saved] = float(pulse.mean())
            np.add(pulse, 1.0, out=pulse)
            np.matmul(adjacency, pulse, out=field)
            field /= float(N_OSC)
            field_mean[saved] = float(field.mean())
            field_std[saved] = float(field.std())
            saved += 1
        if step == nsteps:
            break
        phase = rk4_step_copy(phase, dt, omega, adjacency, coupling, buffers)
    times = np.arange(saved) * sample_dt
    return {
        "time": times,
        "C": cosine[:saved],
        "H": field_mean[:saved],
        "Hstd": field_std[:saved],
    }


def window_rms(times: np.ndarray, values: np.ndarray, start: float, end: float) -> tuple[float, float]:
    mask = (times >= start) & (times <= end)
    sample = values[mask]
    mean = float(np.mean(sample))
    centered = sample - mean
    return float(np.sqrt(np.mean(centered * centered))), mean


def spectral_peak(times: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    mask = (times >= start) & (times <= end)
    sample = values[mask]
    sample = sample - np.mean(sample)
    window = np.hanning(sample.size)
    power = np.abs(np.fft.rfft(sample * window)) ** 2
    angular = 2.0 * np.pi * np.fft.rfftfreq(sample.size, d=float(times[1] - times[0]))
    band = (angular >= 0.2) & (angular <= 1.8)
    if not np.any(band):
        return float("nan")
    return float(angular[band][np.argmax(power[band])])


def dt_agreement(omega: np.ndarray, adjacency: np.ndarray, coupling: float, theta: np.ndarray) -> dict:
    """Compare the collective cosine at the production step and at half that step."""
    coarse = integrate_collective(theta, omega, adjacency, coupling, DT, 40.0, SAMPLE_DT)
    fine = integrate_collective(theta, omega, adjacency, coupling, DT / 2.0, 40.0, SAMPLE_DT)
    gap = float(np.max(np.abs(coarse["C"] - fine["C"])))
    return {"max_abs_C": gap, "passed": gap < 0.02, "coupling": coupling}


def network_seed(realization: int) -> int:
    return derive_seed(MASTER, "network", realization)


def phase_seed(realization: int, kind: str) -> int:
    return derive_seed(MASTER, kind, realization)


def simulate_case(
    probability: float,
    delta: float,
    theory: dict,
    offset: float,
    realization: int,
    initial_condition: str,
) -> dict:
    coupling = theory["K_degree"] * (1.0 + offset)
    omega = np.array(natural_frequencies(delta, N_OSC), dtype=np.float64, copy=True)
    adjacency = make_adjacency(N_OSC, probability, network_seed(realization))
    coefficients = local_coefficients(coupling, adjacency)
    if initial_condition == "uniform":
        seed = phase_seed(realization, "uniform-phase")
        theta = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, N_OSC)
        pinned = int(np.sum(coefficients >= omega))
    else:
        seed = phase_seed(realization, "phase-quantile")
        theta, pinned = sample_incoherent(omega, coefficients, np.random.default_rng(seed))
        if initial_condition == "incoherent_perturbed":
            theta = theta + PERTURBATION * np.sin(theta)
    series = integrate_collective(theta, omega, adjacency, coupling, DT, T_END, SAMPLE_DT)
    late_rms, late_mean = window_rms(series["time"], series["C"], T_END / 2.0, T_END)
    quarter_rms, _ = window_rms(series["time"], series["C"], 0.75 * T_END, T_END)
    early_rms, _ = window_rms(series["time"], series["C"], 0.0, 200.0)
    late_h, _ = window_rms(series["time"], series["H"], T_END / 2.0, T_END)
    peak = spectral_peak(series["time"], series["C"], T_END / 2.0, T_END)
    late_std = float(np.mean(series["Hstd"][series["time"] >= T_END / 2.0]))
    return {
        "N": N_OSC,
        "p": probability,
        "delta": delta,
        "K": coupling,
        "r_offset": offset,
        "K_H_degree": theory["K_degree"],
        "K_H_scaling": theory["K_scaling"],
        "Omega_H": theory["Omega"],
        "network_seed": network_seed(realization),
        "frequency_seed": "",
        "frequency_rule": FREQUENCY_RULE,
        "initial_phase_seed": seed,
        "initial_condition": initial_condition,
        "realization": realization,
        "simulation_time": T_END,
        "dt": DT,
        "n_pinned": pinned,
        "a_max": float(np.max(coefficients)),
        "late_rms_C": late_rms,
        "quarter_rms_C": quarter_rms,
        "early_rms_C": early_rms,
        "late_mean_C": late_mean,
        "late_rms_H": late_h,
        "late_std_H": late_std,
        "peak_omega": peak,
        "peak_offset": peak - theory["Omega"],
        "series": series,
    }


def _median(rows: list[dict], key: str) -> float:
    return float(np.median([row[key] for row in rows]))


def classify(rows: list[dict]) -> dict:
    """Label incoherent-product scans. Other initial conditions are not votes."""
    groups: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        if row["initial_condition"] != "incoherent_product":
            continue
        groups.setdefault((row["p"], row["delta"]), []).append(row)
    summary = {}
    for key, group in groups.items():
        probability, delta = key
        per_seed = []
        for realization in REALIZATIONS:
            seed_rows = [row for row in group if row["realization"] == realization]
            below = [row for row in seed_rows if row["r_offset"] < -1.0e-12]
            above = [row for row in seed_rows if row["r_offset"] > 1.0e-12]
            onset = [row for row in seed_rows if abs(row["r_offset"] - NEAR_ONSET) < 1.0e-12]
            median_below = _median(below, "late_rms_C")
            median_above = _median(above, "late_rms_C")
            ratio = median_above / max(median_below, 1.0e-12)
            separated = min(row["late_rms_C"] for row in above) > max(row["late_rms_C"] for row in below)
            frequency_ok = bool(onset) and abs(onset[0]["peak_offset"]) <= FREQUENCY_TOL
            per_seed.append(
                {
                    "realization": realization,
                    "median_below": median_below,
                    "median_above": median_above,
                    "ratio": ratio,
                    "separated": separated,
                    "frequency_ok": frequency_ok,
                    "n_pinned_max": max(row["n_pinned"] for row in seed_rows),
                }
            )
            for row in seed_rows:
                row["contrast_ratio"] = ratio
                if separated and ratio >= SEPARATION_RATIO and row["r_offset"] < -1.0e-12:
                    row["classification_of_evidence"] = "subthreshold_fluctuation"
                elif separated and ratio >= SEPARATION_RATIO and row["r_offset"] > 1.0e-12:
                    row["classification_of_evidence"] = "collective_oscillation"
                elif abs(row["r_offset"]) <= 1.0e-12 and separated and ratio >= SEPARATION_RATIO:
                    row["classification_of_evidence"] = "near_boundary"
                elif ratio >= 2.0:
                    row["classification_of_evidence"] = "suggestive"
                else:
                    row["classification_of_evidence"] = "unresolved"
        n_separated = sum(item["separated"] and item["ratio"] >= SEPARATION_RATIO for item in per_seed)
        frequency_ok = all(item["frequency_ok"] for item in per_seed)
        amplitude_ok = n_separated == len(per_seed)
        if probability == 1.0 and amplitude_ok and frequency_ok:
            aggregate = "confirmed"
        elif probability == 1.0 and amplitude_ok:
            aggregate = "suggestive"
        elif probability < 1.0 and n_separated >= 2:
            aggregate = "suggestive"
        else:
            aggregate = "unresolved"
        for row in group:
            row["aggregate_status"] = aggregate
        summary[f"p={probability:.2f},delta={delta:.2f}"] = {
            "aggregate_status": aggregate,
            "realizations": per_seed,
        }
    for row in rows:
        if row["initial_condition"] == "incoherent_product":
            continue
        row["contrast_ratio"] = ""
        row["aggregate_status"] = ""
        row["classification_of_evidence"] = (
            "uniform_initial_condition" if row["initial_condition"] == "uniform" else "unresolved_perturbation"
        )
    return summary


def diagnostic_string(row: dict) -> str:
    return (
        f"late_rms_C={row['late_rms_C']:.6f};quarter_rms_C={row['quarter_rms_C']:.6f};"
        f"early_rms_C={row['early_rms_C']:.6f};late_mean_C={row['late_mean_C']:.6f};"
        f"late_rms_H={row['late_rms_H']:.6f};late_std_H={row['late_std_H']:.6f};"
        f"peak_omega={row['peak_omega']:.6f};peak_offset={row['peak_offset']:.6f};"
        f"n_pinned={row['n_pinned']};a_max={row['a_max']:.6f};"
        f"contrast_ratio={row.get('contrast_ratio', '')}"
    )


def write_csv(rows: list[dict]) -> None:
    fields = [
        "N",
        "p",
        "delta",
        "K",
        "r_offset",
        "K_H_degree",
        "K_H_scaling",
        "Omega_H",
        "network_seed",
        "frequency_seed",
        "frequency_rule",
        "initial_phase_seed",
        "initial_condition",
        "realization",
        "simulation_time",
        "dt",
        "diagnostic_metrics",
        "classification_of_evidence",
        "aggregate_status",
        "late_rms_C",
        "quarter_rms_C",
        "early_rms_C",
        "late_mean_C",
        "late_rms_H",
        "late_std_H",
        "peak_omega",
        "peak_offset",
        "n_pinned",
        "a_max",
        "contrast_ratio",
    ]
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {field: row.get(field, "") for field in fields}
            payload["diagnostic_metrics"] = diagnostic_string(row)
            payload["frequency_seed"] = ""
            for field in ("p", "delta", "K", "r_offset", "K_H_degree", "K_H_scaling", "Omega_H"):
                payload[field] = f"{float(row[field]):.10g}"
            for field in (
                "late_rms_C",
                "quarter_rms_C",
                "early_rms_C",
                "late_mean_C",
                "late_rms_H",
                "late_std_H",
                "peak_omega",
                "peak_offset",
                "a_max",
            ):
                payload[field] = f"{float(row[field]):.6e}"
            contrast = row.get("contrast_ratio", "")
            payload["contrast_ratio"] = "" if contrast == "" else f"{float(contrast):.6e}"
            writer.writerow(payload)


def _select(rows: list[dict], probability: float, delta: float, offset: float, realization: int = 0) -> dict:
    matches = [
        row
        for row in rows
        if row["initial_condition"] == "incoherent_product"
        and abs(row["p"] - probability) < 1.0e-12
        and abs(row["delta"] - delta) < 1.0e-12
        and abs(row["r_offset"] - offset) < 1.0e-12
        and row["realization"] == realization
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one series at p={probability}, delta={delta}, r={offset}")
    return matches[0]


def draw_series(rows: list[dict]) -> None:
    panels = [(1.0, 0.05), (1.0, 0.20)]
    if any(abs(row["p"] - 0.2) < 1.0e-12 and abs(row["delta"] - 0.20) < 1.0e-12 for row in rows):
        panels.append((0.2, 0.20))
    fig, axes = plt.subplots(len(panels), 2, figsize=(10.4, 3.15 * len(panels)), constrained_layout=True)
    if len(panels) == 1:
        axes = np.array([axes])
    for axis_row, (probability, delta) in zip(axes, panels):
        below = _select(rows, probability, delta, -0.10)
        above = _select(rows, probability, delta, 0.10)
        omega_h = below["Omega_H"]
        time_axis, spectrum_axis = axis_row
        for row, color, label in (
            (below, "#4d4d4d", r"$K=0.90\,K_H$"),
            (above, "#d95f02", r"$K=1.10\,K_H$"),
        ):
            time_axis.plot(row["series"]["time"], row["series"]["C"], color=color, lw=0.8, label=label)
            mask = row["series"]["time"] >= T_END / 2.0
            sample = row["series"]["C"][mask]
            sample = sample - np.mean(sample)
            window = np.hanning(sample.size)
            power = np.abs(np.fft.rfft(sample * window)) ** 2
            angular = 2.0 * np.pi * np.fft.rfftfreq(sample.size, d=SAMPLE_DT)
            spectrum_axis.semilogy(angular, np.maximum(power, 1.0e-16), color=color, lw=1.1, label=label)
        spectrum_axis.axvline(omega_h, color="#2166ac", lw=1.0, ls="--", label=r"$\Omega_H$")
        time_axis.set_xlim(0.0, T_END)
        time_axis.set_ylabel(r"collective cosine $C(t)$")
        time_axis.set_title(rf"$p={probability:.1f}$, $\delta={delta:.2f}$, realization 0")
        time_axis.grid(True, color="0.9", lw=0.5)
        time_axis.legend(fontsize=8, loc="upper right")
        spectrum_axis.set_xlim(0.2, 1.8)
        spectrum_axis.set_xlabel(r"angular frequency")
        spectrum_axis.set_ylabel("late-window power")
        spectrum_axis.grid(True, color="0.9", lw=0.5)
        spectrum_axis.legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("time")
    fig.suptitle("Collective pulse on either side of the degree-based Hopf prediction", fontsize=12)
    fig.savefig(FIGURE_SERIES, dpi=160)
    plt.close(fig)


def draw_scan(rows: list[dict]) -> None:
    colors = {1.0: "#222222", 0.8: "#2166ac", 0.4: "#1b9e77", 0.2: "#d95f02"}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), sharey=True, constrained_layout=True)
    incoherent = [row for row in rows if row["initial_condition"] == "incoherent_product"]
    for axis, delta in zip(axes, DELTAS):
        for probability in P_VALUES:
            subset = [
                row
                for row in incoherent
                if abs(row["p"] - probability) < 1.0e-12 and abs(row["delta"] - delta) < 1.0e-12
            ]
            if not subset:
                continue
            status = subset[0]["aggregate_status"]
            for realization in REALIZATIONS:
                seed_rows = sorted(
                    [row for row in subset if row["realization"] == realization],
                    key=lambda row: row["r_offset"],
                )
                axis.plot(
                    [row["r_offset"] for row in seed_rows],
                    [row["late_rms_C"] for row in seed_rows],
                    color=colors[probability],
                    lw=1.0,
                    alpha=0.9 if realization == 0 else 0.35,
                    marker="o",
                    ms=4.5,
                    label=f"p={probability:.1f}, {status}" if realization == 0 else None,
                )
        axis.axvline(0.0, color="0.2", lw=1.0, ls="--")
        axis.set_xlabel(r"$(K-K_H^{\mathrm{degree}})/K_H^{\mathrm{degree}}$")
        axis.set_title(f"δ={delta:.2f}")
        axis.grid(True, color="0.9", lw=0.5)
        axis.legend(fontsize=7.4, loc="upper left", framealpha=0.95)
    axes[0].set_ylabel(r"late demeaned RMS of $C(t)$")
    fig.suptitle("Finite-network pulse amplitude against the theoretical Hopf location", fontsize=12)
    fig.savefig(FIGURE_SCAN, dpi=160)
    plt.close(fig)


def p1_gate(summary: dict) -> dict:
    keys = [f"p=1.00,delta={delta:.2f}" for delta in DELTAS]
    passed = True
    for key in keys:
        item = summary[key]
        passed = passed and all(
            seed["separated"] and seed["ratio"] >= SEPARATION_RATIO for seed in item["realizations"]
        )
    return {"passed": passed, "groups": {key: summary[key] for key in keys}}


def drop_series(row: dict) -> dict:
    payload = {key: value for key, value in row.items() if key != "series"}
    return payload


def run() -> None:
    started = time.perf_counter()
    theory = load_theory()
    identity = sampler_check()
    print("sampler", identity, flush=True)
    if not identity["passed"]:
        raise SystemExit(2)
    omega = np.array(natural_frequencies(0.05, N_OSC), dtype=np.float64, copy=True)
    adjacency = make_adjacency(N_OSC, 1.0, network_seed(0))
    coefficients = local_coefficients(theory[(1.0, 0.05)]["K_degree"] * 1.10, adjacency)
    theta, _ = sample_incoherent(omega, coefficients, np.random.default_rng(phase_seed(0, "phase-quantile")))
    step_check = dt_agreement(omega, adjacency, theory[(1.0, 0.05)]["K_degree"] * 1.10, theta)
    print("dt_check_p1", step_check, flush=True)
    if not step_check["passed"]:
        raise SystemExit(3)

    rows: list[dict] = []

    def launch(probability: float, delta: float, offset: float, realization: int, kind: str) -> None:
        row = simulate_case(probability, delta, theory[(probability, delta)], offset, realization, kind)
        rows.append(row)
        print(
            f"p={probability:.1f} delta={delta:.2f} r={offset:+.2f} ic={kind} "
            f"real={realization} rms={row['late_rms_C']:.4f} peak={row['peak_omega']:.3f} "
            f"pinned={row['n_pinned']}",
            flush=True,
        )

    for delta in DELTAS:
        for realization in REALIZATIONS:
            for offset in OFFSETS:
                launch(1.0, delta, offset, realization, "incoherent_product")
        for offset in OFFSETS:
            launch(1.0, delta, offset, 0, "uniform")
        for offset in (-0.10, 0.0, 0.10):
            launch(1.0, delta, offset, 0, "incoherent_perturbed")

    summary = classify(rows)
    gate = p1_gate(summary)
    print("p1_gate", gate["passed"], flush=True)
    if not gate["passed"]:
        write_csv(rows)
        SUMMARY_PATH.write_text(json.dumps({"sampler": identity, "dt_check": step_check, "p1_gate": gate, "classification": summary}, indent=2) + "\n")
        raise SystemExit(4)

    stiff_omega = np.array(natural_frequencies(0.20, N_OSC), dtype=np.float64, copy=True)
    stiff_adjacency = make_adjacency(N_OSC, 0.2, network_seed(0))
    stiff_k = theory[(0.2, 0.20)]["K_degree"] * 1.10
    stiff_theta, _ = sample_incoherent(
        stiff_omega,
        local_coefficients(stiff_k, stiff_adjacency),
        np.random.default_rng(phase_seed(0, "phase-quantile")),
    )
    stiff_check = dt_agreement(stiff_omega, stiff_adjacency, stiff_k, stiff_theta)
    print("dt_check_p02", stiff_check, flush=True)

    for probability in (0.8, 0.4, 0.2):
        for delta in DELTAS:
            for realization in REALIZATIONS:
                for offset in OFFSETS:
                    launch(probability, delta, offset, realization, "incoherent_product")
            for offset in (-0.10, 0.10):
                launch(probability, delta, offset, 0, "uniform")
            if probability == 0.2 and abs(delta - 0.20) < 1.0e-12:
                for offset in (-0.10, 0.10):
                    launch(probability, delta, offset, 0, "incoherent_perturbed")

    summary = classify(rows)
    write_csv(rows)
    draw_series(rows)
    draw_scan(rows)
    payload = {
        "sampler": identity,
        "dt_check_p1": step_check,
        "dt_check_p02_delta020": stiff_check,
        "p1_gate": gate,
        "classification": summary,
        "elapsed_sec": time.perf_counter() - started,
        "statement": (
            "aggregate_status confirmed means the p=1 amplitude diagnostic "
            "separates the two sides of the known continuum boundary. "
            "It is not a measured quenched eigenvalue. "
            "p<1 is at most suggestive."
        ),
    }
    def _jsonable(value):
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return _jsonable(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value

    SUMMARY_PATH.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    print("wrote", CSV_PATH, flush=True)
    print("elapsed", payload["elapsed_sec"], flush=True)


if __name__ == "__main__":
    run()
