"""Compare Stage 1 p=1 W3 profiles with Ariaratnam & Strogatz Eq. (5).

Reads the Stage 1 index and saved rotation numbers. Assigns labels with
Classifier v2.0.0. Does not integrate the ODE, edit thresholds, or rewrite
the classifier archive.

Structural near-zero counts in this script use |rho| < 0.01, as specified
for this comparison. Classifier v2 itself uses the archived tolerance
|rho| < 0.001. Both numbers are reported.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-eq5-validation")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

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

ANALYSIS_TOLERANCE = 0.01
PARTIAL_DEATH_RUN = 15
N_REFERENCE = 1000
W3_BOUNDS = (12000.0, 14000.0)
FIGURE_PATH = HERE / "eq5_validation_figure.png"


def prefix_run(mask: np.ndarray) -> int:
    count = 0
    for value in mask:
        if not value:
            break
        count += 1
    return int(count)


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Return the longest True run and the start index of the earliest such run."""
    best = current = 0
    best_start = -1
    start = 0
    for index, value in enumerate(mask):
        if value:
            if current == 0:
                start = index
            current += 1
            if current > best:
                best = current
                best_start = start
        else:
            current = 0
    return int(best), int(best_start)


def k_theory(delta: float) -> float:
    return 1.0 - float(delta)


def k15(delta: float | np.ndarray, n: int = N_REFERENCE) -> float | np.ndarray:
    values = np.asarray(delta, dtype=float)
    out = 1.0 - values + (28.0 * values) / (n - 1)
    if np.ndim(delta) == 0:
        return float(out)
    return out


def first_true(values: list[dict], key: str) -> dict | None:
    for row in values:
        if row[key]:
            return row
    return None


def classify_onset(rows: list[dict], theory: float) -> dict:
    """Locate the first grid K where a flag is true. No interpolation."""
    hit = first_true(rows, "flag")
    if hit is None:
        return {
            "status": "not observed",
            "K": None,
            "K_previous": None,
            "difference": None,
            "bracket": "not observed",
        }
    index = rows.index(hit)
    previous = rows[index - 1]["K"] if index else None
    difference = float(hit["K"]) - theory
    if previous is None:
        bracket = "at or below first grid point" if theory <= hit["K"] + 1e-9 else "early"
    elif hit["K"] < theory - 1e-9:
        bracket = "early"
    elif previous < theory <= hit["K"] + 1e-9:
        bracket = "theory inside grid gap"
    else:
        bracket = "late"
    return {
        "status": "observed",
        "K": float(hit["K"]),
        "K_previous": None if previous is None else float(previous),
        "difference": difference,
        "bracket": bracket,
        "state": hit["state"],
        "low_run": hit["low_run"],
        "longest": hit["longest"],
    }


def load_profiles() -> tuple[list[dict], dict]:
    config = load_config()
    classifier_tol = float(config["rule_4_near_zero_tolerance"]["value"])
    if int(config["rule_6_partial_death"]["minimum_consecutive_near_zero_oscillators"]) != PARTIAL_DEATH_RUN:
        raise RuntimeError("archived partial-death run length is not 15")
    runs = query_runs(status="completed", p=1.0, N=N_REFERENCE, trial_id=0)
    if not runs:
        raise RuntimeError("no completed p=1 runs")
    pilot, cost_audit = load_method_c()
    records = []
    omega_error = 0.0
    degenerate = []
    for run in runs:
        with np.load(run["output_npz"], allow_pickle=False) as data:
            omega = np.array(data["omega"], dtype=np.float64)
            rotation = np.array(data["rotation_numbers"], dtype=np.float64)
            starts = np.array(data["window_start"], dtype=np.float64)
            ends = np.array(data["window_end"], dtype=np.float64)
            stored_dt = float(np.asarray(data["dt"]).item())
        if omega.shape != (N_REFERENCE,) or rotation.shape[1] != N_REFERENCE:
            raise RuntimeError(f"unexpected shape for {run['run_id']}")
        expected = np.linspace(1.0 - float(run["Delta"]), 1.0 + float(run["Delta"]), N_REFERENCE)
        omega_error = max(omega_error, float(np.max(np.abs(omega - expected))))
        if np.any(np.diff(omega) < 0):
            order = np.argsort(omega, kind="mergesort")
            omega = omega[order]
            rotation = rotation[:, order]
        rho = rotation[window_row(starts, ends, W3_BOUNDS)]
        analysis_mask = np.abs(rho) < ANALYSIS_TOLERANCE
        classifier_mask = np.abs(rho) < classifier_tol
        low_run = prefix_run(analysis_mask)
        longest, longest_start = longest_run(analysis_mask)
        classifier_low = prefix_run(classifier_mask)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
            classified = classify_window(cost_audit, pilot, omega, rho, config)
        unique_omega = int(np.unique(np.round(omega, 12)).size)
        if unique_omega < N_REFERENCE:
            degenerate.append(float(run["Delta"]))
        records.append(
            {
                "run_id": run["run_id"],
                "K": float(run["K"]),
                "Delta": float(run["Delta"]),
                "N": int(run["N"]),
                "trial_id": int(run["trial_id"]),
                "dt": stored_dt,
                "state": classified["state"],
                "classifier_near_fraction": float(classified["near_zero_fraction"]),
                "classifier_run": int(classified["consecutive_near_zero"]),
                "classifier_low_run": classifier_low,
                "rho_range": float(classified["rho_range"]),
                "near_zero_fraction": float(analysis_mask.mean()),
                "low_frequency_near_zero_run": low_run,
                "longest_near_zero_run": longest,
                "longest_start": longest_start,
                "interior_near_zero": int(analysis_mask.sum() - low_run),
                "unique_omega": unique_omega,
            }
        )
    meta = {
        "config": config,
        "classifier_tolerance": classifier_tol,
        "runs": runs,
        "omega_error": omega_error,
        "degenerate_deltas": sorted(set(degenerate)),
        "version": config["classifier_version"],
    }
    return records, meta


def grid_summary(records: list[dict], meta: dict) -> None:
    runs = meta["runs"]
    k_values = np.array(sorted({row["K"] for row in records}))
    delta_values = np.array(sorted({row["Delta"] for row in records}))
    steps_k = np.unique(np.round(np.diff(k_values), 10))
    steps_d = np.unique(np.round(np.diff(delta_values), 10))
    by_delta = {}
    for row in records:
        by_delta.setdefault(row["Delta"], set()).add(row["K"])
    print("DATA")
    print(f"profiles {len(records)}")
    print(f"N {sorted({row['N'] for row in records})}")
    print(f"p {[1.0]}")
    print(f"trial {sorted({row['trial_id'] for row in runs})}")
    print(f"K count {k_values.size} min {k_values[0]} max {k_values[-1]} steps {steps_k.tolist()}")
    print(f"delta count {delta_values.size} min {delta_values[0]} max {delta_values[-1]} steps {steps_d.tolist()}")
    print(
        "coverage",
        {length: sum(len(values) == length for values in by_delta.values()) for length in sorted({len(v) for v in by_delta.values()})},
    )
    high = sorted(delta for delta, values in by_delta.items() if max(values) > 2.0 + 1e-9)
    print("deltas with K>2", high)
    print(f"dt {sorted({row['dt'] for row in records})}")
    database = Path(runs[0]["output_dir"]) / "index.sqlite"
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        provenance = connection.execute(
            """
            SELECT solver, t_origin, t_end, master_seed, randomness_mode, seed_recipe,
                   network_seed, initial_phase_seed, frequency_seed, frequency_rule,
                   network_pairing_rule, phase_pairing_rule, model_version,
                   network_realization_id, initial_condition_id, paired_group_id,
                   COUNT(*) AS n
            FROM runs
            WHERE status='completed' AND ABS(p-1.0)<1e-9
            GROUP BY solver, t_origin, t_end, master_seed, randomness_mode, seed_recipe,
                     network_seed, initial_phase_seed, frequency_seed, frequency_rule,
                     network_pairing_rule, phase_pairing_rule, model_version,
                     network_realization_id, initial_condition_id, paired_group_id
            """
        ).fetchall()
    print(f"provenance groups {len(provenance)}")
    for row in provenance:
        print(dict(row))
    print(f"omega max abs error vs linspace {meta['omega_error']:.3e}")
    print(f"degenerate deltas {meta['degenerate_deltas']}")
    print(f"classifier {meta['version']} near_zero {meta['classifier_tolerance']}")
    sample = runs[0]
    adjacency = make_adjacency(sample["N"], 1.0, sample["network_seed"])
    rebuilt = sha256_array(adjacency)
    stored = json.loads(Path(sample["output_json"]).read_text())["adjacency_sha256"]
    off = int(np.sum(adjacency[np.triu_indices(sample["N"], k=1)]))
    print(
        "p=1 graph check",
        f"diagonal_sum {adjacency.diagonal().sum()}",
        f"upper_edges {off}",
        f"expected_upper {sample['N'] * (sample['N'] - 1) // 2}",
        f"hash_match {rebuilt == stored}",
    )


def per_delta(records: list[dict]) -> list[dict]:
    grouped: dict[float, list[dict]] = {}
    for row in records:
        grouped.setdefault(row["Delta"], []).append(row)
    summaries = []
    for delta in sorted(grouped):
        rows = sorted(grouped[delta], key=lambda item: item["K"])
        theory = k_theory(delta)
        reference = k15(delta, rows[0]["N"])
        below = [row for row in rows if row["K"] < theory - 1e-9]
        state_below = below[-1]["state"] if below else "no grid point below Kc"
        flags = {
            "ge1": [{**row, "flag": row["low_frequency_near_zero_run"] >= 1} for row in rows],
            "ge15": [{**row, "flag": row["low_frequency_near_zero_run"] >= PARTIAL_DEATH_RUN, "low_run": row["low_frequency_near_zero_run"], "longest": row["longest_near_zero_run"]} for row in rows],
            "pd": [{**row, "flag": row["state"] == "Partial Death", "low_run": row["low_frequency_near_zero_run"], "longest": row["longest_near_zero_run"]} for row in rows],
        }
        for name in ("ge1",):
            for row in flags[name]:
                row["low_run"] = row["low_frequency_near_zero_run"]
                row["longest"] = row["longest_near_zero_run"]
        runs_series = [row["low_frequency_near_zero_run"] for row in rows]
        drop = 0
        peak = 0
        for value in runs_series:
            peak = max(peak, value)
            drop = max(drop, peak - value)
        states = [row["state"] for row in rows]
        pd_indices = [index for index, state in enumerate(states) if state == "Partial Death"]
        pd_gap = False
        if len(pd_indices) >= 2 and pd_indices[-1] - pd_indices[0] + 1 != len(pd_indices):
            pd_gap = True
        left_branch = sorted({row["state"] for row in below if row["state"] != "Incoherence"})
        summaries.append(
            {
                "delta": delta,
                "Kc": theory,
                "K15": reference,
                "state_below": state_below,
                "left_branch_states": left_branch,
                "ge1": classify_onset(flags["ge1"], theory),
                "ge15": classify_onset(flags["ge15"], theory),
                "pd": classify_onset(flags["pd"], theory),
                "run_drop_after_peak": drop,
                "pd_gap": pd_gap,
                "pd_count": len(pd_indices),
                "states_at_or_above": sorted({row["state"] for row in rows if row["K"] + 1e-12 >= theory}),
                "n_below_with_run": sum(row["low_frequency_near_zero_run"] >= 1 for row in below),
                "n_above_incoherent_without_run": sum(
                    row["K"] + 1e-12 >= theory
                    and row["state"] == "Incoherence"
                    and row["low_frequency_near_zero_run"] == 0
                    for row in rows
                ),
                "rows": rows,
            }
        )
    return summaries


def print_onsets(summaries: list[dict]) -> None:
    print("ONSETS")
    header = (
        "delta Kc K15 state_below ge1 d1 bracket1 ge15 d15 bracket15 "
        "pd dpd bracketpd state_at_ge15 drop pd_gap below_run above_inc0"
    )
    print(header)
    for item in summaries:
        ge15_state = item["ge15"]["state"] if item["ge15"]["status"] == "observed" else "-"
        def fmt(onset: dict, digits: str = ".3f") -> str:
            if onset["K"] is None:
                return "NA NA NA"
            return f"{onset['K']:{digits}} {onset['difference']:+.3f} {onset['bracket'].replace(' ', '_')}"

        print(
            f"{item['delta']:.3f} {item['Kc']:.3f} {item['K15']:.4f} {item['state_below'].replace(' ', '_')} "
            f"{fmt(item['ge1'])} {fmt(item['ge15'])} {fmt(item['pd'])} {ge15_state.replace(' ', '_')} "
            f"{item['run_drop_after_peak']} {int(item['pd_gap'])} {item['n_below_with_run']} "
            f"{item['n_above_incoherent_without_run']} left={','.join(item['left_branch_states']) or '-'}"
        )


def print_cases(records: list[dict], summaries: list[dict]) -> None:
    print("CASES")
    counts: dict[str, int] = {}
    for row in records:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    print("state_counts", counts)
    analysis_would_pd = [
        row
        for row in records
        if row["longest_near_zero_run"] >= PARTIAL_DEATH_RUN
        and row["near_zero_fraction"] < 1.0
        and row["rho_range"] > 0.01
    ]
    reasons = {"Partial Death": 0, "Death": 0, "Locking": 0, "other": 0}
    for row in records:
        if row["longest_near_zero_run"] < PARTIAL_DEATH_RUN:
            continue
        if row["classifier_near_fraction"] >= 1.0:
            reasons["Death"] += 1
        elif row["rho_range"] <= 0.01:
            reasons["Locking"] += 1
        elif row["classifier_run"] >= PARTIAL_DEATH_RUN:
            reasons["Partial Death"] += 1
        else:
            reasons["other"] += 1
    print("cells with |rho|<0.01 run>=15, label by archived rule inputs", reasons)
    print("those cells", sum(reasons.values()))
    mismatch = [
        row
        for row in records
        if row["longest_near_zero_run"] >= PARTIAL_DEATH_RUN
        and row["classifier_run"] < PARTIAL_DEATH_RUN
        and row["classifier_near_fraction"] < 1.0
        and row["rho_range"] > 0.01
    ]
    print("0.01-run>=15 but archived 0.001-run<15 and not death/locking", len(mismatch))
    mismatch.sort(key=lambda row: (-(row["longest_near_zero_run"] - row["classifier_run"]), row["Delta"], row["K"]))
    for row in mismatch[:12]:
        print(
            " tol_gap",
            f"K={row['K']:.3f}",
            f"d={row['Delta']:.3f}",
            f"state={row['state']}",
            f"run01={row['longest_near_zero_run']}",
            f"low01={row['low_frequency_near_zero_run']}",
            f"run001={row['classifier_run']}",
            f"f001={row['classifier_near_fraction']:.4f}",
            f"range={row['rho_range']:.4f}",
        )
    middle = [
        row
        for row in records
        if row["state"] == "Partial Death" and row["longest_start"] > 0
    ]
    print("Partial Death whose earliest longest |rho|<0.01 run does not start at omega_min", len(middle))
    for row in sorted(middle, key=lambda item: (item["longest_start"], -item["longest_near_zero_run"]))[:12]:
        print(
            " middle_pd",
            f"K={row['K']:.3f}",
            f"d={row['Delta']:.3f}",
            f"low={row['low_frequency_near_zero_run']}",
            f"longest={row['longest_near_zero_run']}",
            f"start={row['longest_start']}",
            f"interior={row['interior_near_zero']}",
            f"run001={row['classifier_run']}",
        )
    low_end_pd = sum(
        row["state"] == "Partial Death" and row["low_frequency_near_zero_run"] >= PARTIAL_DEATH_RUN
        for row in records
    )
    pd_total = sum(row["state"] == "Partial Death" for row in records)
    print(f"partial_death {pd_total} of which low_frequency_run01>=15 {low_end_pd}")
    above_without = [
        row
        for row in records
        if row["Delta"] > 0
        and row["K"] > k_theory(row["Delta"]) + 1e-9
        and row["low_frequency_near_zero_run"] == 0
        and row["state"] == "Incoherence"
    ]
    above_without.sort(key=lambda row: row["K"] - k_theory(row["Delta"]), reverse=True)
    print("incoherent cells strictly above K=1-delta with low-frequency run 0", len(above_without))
    for row in above_without[:8]:
        print(
            " above_empty",
            f"K={row['K']:.3f}",
            f"d={row['Delta']:.3f}",
            f"Kc={k_theory(row['Delta']):.3f}",
            f"gap={row['K'] - k_theory(row['Delta']):+.3f}",
            f"longest={row['longest_near_zero_run']}",
            f"f={row['near_zero_fraction']:.4f}",
        )
    below_with = [
        row
        for row in records
        if row["Delta"] > 0 and row["K"] < k_theory(row["Delta"]) - 1e-9 and row["low_frequency_near_zero_run"] >= 1
    ]
    below_with.sort(key=lambda row: row["K"] - k_theory(row["Delta"]))
    print("cells strictly below K=1-delta with low-frequency run >=1", len(below_with))
    for row in below_with[:8]:
        print(
            " below_hit",
            f"K={row['K']:.3f}",
            f"d={row['Delta']:.3f}",
            f"Kc={k_theory(row['Delta']):.3f}",
            f"gap={row['K'] - k_theory(row['Delta']):+.3f}",
            f"low={row['low_frequency_near_zero_run']}",
            f"state={row['state']}",
            f"f={row['near_zero_fraction']:.4f}",
        )
    print("nonmonotonic run drop after peak, deltas", [item["delta"] for item in summaries if item["run_drop_after_peak"] > 0])
    print("PD label gaps", [item["delta"] for item in summaries if item["pd_gap"]])
    comparable = [
        item
        for item in summaries
        if item["delta"] > 0 and item["state_below"] == "Incoherence"
    ]
    print("comparison deltas (state immediately below Kc is Incoherence, delta>0)", [item["delta"] for item in comparable])
    excluded = [
        (item["delta"], item["state_below"], item["left_branch_states"])
        for item in summaries
        if item not in comparable
    ]
    print("excluded", excluded)

    def tally(name: str) -> None:
        brackets: dict[str, int] = {}
        diffs = []
        for item in comparable:
            onset = item[name]
            brackets[onset["bracket"] if onset["status"] == "observed" else "not observed"] = (
                brackets.get(onset["bracket"] if onset["status"] == "observed" else "not observed", 0) + 1
            )
            if onset["difference"] is not None:
                diffs.append(onset["difference"])
        diffs_arr = np.array(diffs) if diffs else np.array([])
        print(
            name,
            "n",
            len(comparable),
            "brackets",
            brackets,
            "diff_min",
            None if diffs_arr.size == 0 else float(diffs_arr.min()),
            "diff_median",
            None if diffs_arr.size == 0 else float(np.median(diffs_arr)),
            "diff_max",
            None if diffs_arr.size == 0 else float(diffs_arr.max()),
        )

    tally("ge1")
    tally("ge15")
    tally("pd")
    print("analysis_would_be_partial_death_count", len(analysis_would_pd))
    middle_classifier = [
        row
        for row in records
        if row["state"] == "Partial Death" and row["classifier_low_run"] < PARTIAL_DEATH_RUN
    ]
    print("Partial Death with archived low-frequency run <15", len(middle_classifier))
    for row in middle_classifier[:8]:
        print(
            " middle001",
            f"K={row['K']:.3f}",
            f"d={row['Delta']:.3f}",
            f"low001={row['classifier_low_run']}",
            f"run001={row['classifier_run']}",
            f"low01={row['low_frequency_near_zero_run']}",
        )
    below_partial = [
        row
        for row in below_with
        if row["low_frequency_near_zero_run"] < N_REFERENCE or row["state"] != "Death"
    ]
    print("below-line near-zero that is not full Death", len(below_partial))
    print_alignment(summaries)


def first_grid_above(grid: list[float], threshold: float) -> float | None:
    for value in grid:
        if value > threshold + 1e-9:
            return value
    return None


def print_alignment(summaries: list[dict]) -> None:
    print("ALIGNMENT")
    ge1_match = ge15_match = pd_match = 0
    comparable = []
    for item in summaries:
        grid = [row["K"] for row in item["rows"]]
        pred1 = first_grid_above(grid, item["Kc"])
        pred15 = first_grid_above(grid, item["K15"])
        item["pred_ge1"] = pred1
        item["pred_ge15"] = pred15
        if item["delta"] <= 0 or item["state_below"] != "Incoherence":
            continue
        comparable.append(item)
        ge1_match += int(item["ge1"]["K"] == pred1)
        ge15_match += int(item["ge15"]["K"] == pred15)
        pd_match += int(item["pd"]["K"] == pred15)
    print(
        "clean incoherent-below deltas",
        len(comparable),
        "ge1 equals first K>Kc",
        ge1_match,
        "ge15 equals first K>K15",
        ge15_match,
        "PD equals first K>K15",
        pd_match,
    )
    print("mismatches against those grid predictions")
    for item in comparable:
        flags = []
        if item["ge1"]["K"] != item["pred_ge1"]:
            flags.append(f"ge1 {item['ge1']['K']} pred {item['pred_ge1']}")
        if item["ge15"]["K"] != item["pred_ge15"]:
            flags.append(f"ge15 {item['ge15']['K']} pred {item['pred_ge15']}")
        if item["pd"]["K"] != item["pred_ge15"]:
            flags.append(f"pd {item['pd']['K']} pred {item['pred_ge15']}")
        if flags:
            onset = next(row for row in item["rows"] if row["K"] == item["ge1"]["K"])
            print(
                f"d={item['delta']:.3f}",
                f"Kc={item['Kc']:.4f}",
                f"K15={item['K15']:.4f}",
                " ".join(flags),
                f"onset_low={onset['low_frequency_near_zero_run']}",
                f"onset_state={onset['state']}",
                f"onset_f={onset['near_zero_fraction']:.4f}",
            )
    print("STATE SEQUENCES")
    for item in summaries:
        parts = []
        start = item["rows"][0]
        previous = start["state"]
        for row in item["rows"][1:]:
            if row["state"] != previous:
                parts.append(f"{previous}:{start['K']:.2f}-{item['rows'][item['rows'].index(row)-1]['K']:.2f}")
                start = row
                previous = row["state"]
        parts.append(f"{previous}:{start['K']:.2f}-{item['rows'][-1]['K']:.2f}")
        print(f"d={item['delta']:.3f}", " | ".join(parts))
    print("SELECTED PROFILES")
    wanted = {0.05, 0.2, 0.225, 0.5, 0.8, 0.925, 1.0}
    for item in summaries:
        if round(item["delta"], 3) not in wanted:
            continue
        print(f"-- delta {item['delta']:.3f}")
        for row in item["rows"]:
            if row["K"] > 1.25 + 1e-9 and row["Delta"] not in (0.05, 0.2):
                continue
            if row["Delta"] in (0.05, 0.2) and row["K"] > 1.25:
                continue
            print(
                f"  K={row['K']:.3f} state={row['state']} low01={row['low_frequency_near_zero_run']} "
                f"long01={row['longest_near_zero_run']} low001={row['classifier_low_run']} "
                f"run001={row['classifier_run']} f01={row['near_zero_fraction']:.4f} "
                f"f001={row['classifier_near_fraction']:.4f} range={row['rho_range']:.4f}"
            )


def draw_figure(records: list[dict], meta: dict) -> None:
    deltas = np.array(sorted({row["Delta"] for row in records}))
    ks = np.array(sorted({row["K"] for row in records}))
    # Display the completed rectangle, including K values present for only some delta.
    delta_edges = np.concatenate(
        ([deltas[0] - 0.0125], 0.5 * (deltas[:-1] + deltas[1:]), [deltas[-1] + 0.0125])
    )
    k_edges = np.concatenate(([ks[0] - 0.025], 0.5 * (ks[:-1] + ks[1:]), [ks[-1] + 0.025]))
    delta_index = {round(value, 10): index for index, value in enumerate(deltas)}
    k_index = {round(value, 10): index for index, value in enumerate(ks)}
    runs = np.full((ks.size, deltas.size), np.nan)
    states = np.full((ks.size, deltas.size), -1, dtype=int)
    code = {state: index for index, state in enumerate(PHASE_STATES)}
    for row in records:
        i = k_index[round(row["K"], 10)]
        j = delta_index[round(row["Delta"], 10)]
        runs[i, j] = row["low_frequency_near_zero_run"]
        states[i, j] = code[row["state"]]

    run_bounds = [0, 1, 15, 50, 150, 400, 1000]
    run_colors = ["#f7f7f7", "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    run_cmap = ListedColormap(run_colors)
    run_cmap.set_bad("#d9d9d9")
    run_norm = BoundaryNorm(run_bounds, run_cmap.N)
    state_cmap = ListedColormap(PHASE_COLORS)
    state_cmap.set_bad("#d9d9d9")
    state_norm = BoundaryNorm(np.arange(-0.5, len(PHASE_COLORS) + 0.5, 1), state_cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 8.4), sharex=True, sharey=True)
    line_delta = np.linspace(0.0, 1.0, 400)
    panels = (
        (
            axes[0],
            np.ma.masked_invalid(runs),
            run_cmap,
            run_norm,
            "Low-frequency near-zero run",
        ),
        (
            axes[1],
            np.ma.masked_less(states.astype(float), 0),
            state_cmap,
            state_norm,
            "Classifier v2.0.0",
        ),
    )
    for ax, field, cmap, norm, title in panels:
        mesh = ax.pcolormesh(
            delta_edges,
            k_edges,
            field,
            cmap=cmap,
            norm=norm,
            shading="flat",
        )
        ax.plot(line_delta, 1.0 - line_delta, color="black", lw=1.6, label="Eq. (5): K = 1 − δ")
        ax.plot(
            line_delta,
            k15(line_delta),
            color="black",
            lw=1.2,
            ls="--",
            label="K15: 15th discrete frequency (reference only)",
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 3.05)
        ax.set_xlabel("Frequency half-width δ")
        ax.set_title(title)
        ax.set_aspect("auto")
        mesh.set_rasterized(False)
    axes[0].set_ylabel("Coupling K")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=run_norm, cmap=run_cmap),
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        ticks=[0, 1, 15, 50, 150, 400, 1000],
    )
    colorbar.set_label("Oscillators from ω_min with |ρ| < 0.01")
    state_handles = [Patch(facecolor=PHASE_COLORS[index], edgecolor="none", label=state) for index, state in enumerate(PHASE_STATES)]
    state_handles.append(Patch(facecolor="#d9d9d9", edgecolor="none", label="No data"))
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.46, 0.98), fontsize=7.5, framealpha=0.95)
    axes[1].legend(
        handles=state_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        framealpha=0.95,
        borderaxespad=0.2,
    )
    fig.suptitle(
        f"N=1000, p=1, trial 0, W3=[12000, 14000]  |  "
        f"near-zero color uses |ρ|<0.01; labels use archived |ρ|<{meta['classifier_tolerance']:g}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    records, meta = load_profiles()
    grid_summary(records, meta)
    summaries = per_delta(records)
    print_onsets(summaries)
    print_cases(records, summaries)
    draw_figure(records, meta)
    print(f"figure {FIGURE_PATH}")


if __name__ == "__main__":
    main()
