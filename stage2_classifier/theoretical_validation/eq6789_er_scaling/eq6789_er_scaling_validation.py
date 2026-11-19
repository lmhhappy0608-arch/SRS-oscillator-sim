"""Annealed pK check of Ariaratnam–Strogatz Eqs. (6)–(9) on saved ER scans.

The p=1 formulas are the functions already used in the Eq. (7) and death-boundary
validations. This script does not retype those formulas, refit them, rerun the
ODE, or edit Classifier v1. Labels are the W3 labels stored in the Eq. (5) ER
cache, and they are accepted only after the published p=1 Hopf and death tables
are reproduced. Hopf comparisons are phase-diagram proxies. They are not
eigenvalue measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sqlite3
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-eq6789-er")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import format as npformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "stage2_classifier" / "theoretical_validation" / "eq7_hopf"))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "theoretical_validation" / "death_boundary"))
sys.path.insert(0, str(ROOT))

import validate_death_boundary as death_mod  # noqa: E402
import validate_eq7 as hopf_mod  # noqa: E402
from stage1.grid import quantize  # noqa: E402
from stage1.query import query_runs  # noqa: E402

FIGURE_PATH = HERE / "eq6789_er_scaling_validation.png"
REPORT_PATH = HERE / "eq6789_er_scaling_report.md"
CACHE_PATH = Path("/tmp/eq5_er_scaling_cache.pkl")
CACHE_VERSION = 1
DATABASE = ROOT / "stage1_runs" / "index.sqlite"
N_EXPECTED = 1000
GRID_STEP = 0.05
ABS_TOL = 1e-8
EARLY_TOL = 1e-4
MAIN_DELTA = 0.20
P_VALUES = (0.2, 0.4, 0.8, 1.0)
WINDOWS = (("W1", 8000.0, 10000.0), ("W2", 10000.0, 12000.0), ("W3", 12000.0, 14000.0))
NEAR_TOL = 0.001
P_COLORS = {0.2: "#0072B2", 0.4: "#E69F00", 0.8: "#009E73", 1.0: "#CC79A7"}
P_MARKERS = {0.2: "o", 0.4: "s", 0.8: "D", 1.0: "^"}
SHORT = {
    "Inc": "Incoherence",
    "PL": "Partial Locking",
    "Lock": "Locking",
    "PD": "Partial Death",
    "Death": "Death",
}
SHORT_OF = {value: key for key, value in SHORT.items()}
STATE_COLORS = dict(zip(hopf_mod.PHASE_STATES, hopf_mod.PHASE_COLORS))

SERIES_ANCHORS = {
    0.025: 0.0637,
    0.050: 0.12785,
    0.075: 0.1928,
    0.100: 0.25915,
    0.125: 0.3275,
    0.150: 0.39876,
    0.175: 0.4739,
    0.200: 0.55435,
}
EQ6_ROOTS = {0.05: 0.12785, 0.10: 0.25920, 0.15: 0.39991, 0.20: 0.56905, 0.22: 0.67792}
EQ6_ABSENT = (0.225, 0.25)
HOPF_EXITS = {
    0.025: ("sustained exit", 0.10, "Locking", True),
    0.050: ("sustained exit", 0.15, "Locking", True),
    0.075: ("sustained exit", 0.20, "Locking", True),
    0.100: ("sustained exit", 0.30, "Locking", True),
    0.125: ("sustained exit", 0.35, "Locking", True),
    0.150: ("sustained exit", 0.45, "Locking", False),
    0.175: ("sustained exit", 0.50, "Locking", True),
    0.200: ("non-monotonic", None, None, None),
}
DEATH_ANCHORS = {
    0.000: ("saddle-node", 0.7698, 0.80, "Locking"),
    0.200: ("saddle-node", 0.7902, 0.80, "Incoherence"),
    0.225: ("saddle-node", 0.7956, 0.80, "Partial Locking"),
    0.275: ("saddle-node", 0.8082, 0.85, "Partial Death"),
    0.300: ("end-point", 0.8155, 0.85, "Partial Death"),
    0.400: ("end-point", 0.8495, 0.90, "Partial Death"),
    0.500: ("end-point", 0.8889, 0.90, "Partial Death"),
    1.000: ("end-point", 1.1202, 1.15, "Partial Death"),
}
LATE_DEATH = {0.400, 0.525, 0.750}
P1_COUNTS = {
    "Incoherence": 329,
    "Partial Locking": 1,
    "Locking": 88,
    "Partial Death": 348,
    "Death": 975,
}


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def check(checks: list[dict], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + " | " + detail, flush=True)


def parse_sequence(text: str) -> list[dict]:
    segments = []
    for part in text.split(" → "):
        token, _, payload = part.partition("[")
        if payload:
            left, right = payload.rstrip("]").split(",")
            start = end = None
            start = float(left)
            end = float(right)
        else:
            token, _, payload = part.partition("@")
            start = end = float(payload)
        token = token.strip()
        if token not in SHORT:
            raise RuntimeError(f"unrecognized state token {token!r} in {text!r}")
        segments.append({"state": SHORT[token], "start": start, "end": end})
    return segments


def expand_column(column: dict) -> list[dict]:
    segments = parse_sequence(column["sequence"])
    rows = []
    for coupling in column["K"]:
        coupling = float(coupling)
        hits = [item for item in segments if item["start"] - 5e-4 <= coupling <= item["end"] + 5e-4]
        if len(hits) != 1:
            raise RuntimeError(
                f"sequence parse failed at p={column['p']} delta={column['delta']} K={coupling}: {column['sequence']}"
            )
        rows.append({"K": coupling, "state": hits[0]["state"], "Delta": float(column["delta"])})
    rows.sort(key=lambda item: item["K"])
    return rows


def first_state(rows: list[dict], state: str) -> dict | None:
    for row in rows:
        if row["state"] == state:
            return row
    return None


def first_strictly_above(couplings: list[float], prediction: float) -> float | None:
    for coupling in couplings:
        if coupling > prediction + ABS_TOL:
            return float(coupling)
    return None


def grid_steps(onset: float | None, prediction: float, couplings: list[float]) -> int | None:
    if onset is None:
        return None
    if onset < prediction - EARLY_TOL:
        reference = max((value for value in couplings if value < prediction - ABS_TOL), default=None)
        if reference is None:
            return -1
        return -int(round((reference - onset) / GRID_STEP)) - 1
    reference = first_strictly_above(couplings, prediction)
    if reference is None:
        if abs(onset - prediction) <= EARLY_TOL:
            return 0
        return None
    if abs(onset - prediction) <= EARLY_TOL:
        return 0
    return int(round((onset - reference) / GRID_STEP))


def scan_status(couplings: list[float], holes: list[float], theory: float) -> str:
    k_max = float(couplings[-1])
    if theory > k_max + ABS_TOL:
        return "out_of_range"
    if any(abs(float(hole) - theory) <= GRID_STEP + 1e-8 for hole in holes):
        return "insufficient_k_resolution"
    below = [value for value in couplings if value <= theory + ABS_TOL]
    above = [value for value in couplings if value > theory + ABS_TOL]
    if below and above and above[0] - below[-1] > GRID_STEP + 1e-6:
        return "insufficient_k_resolution"
    first_above = above[0] if above else None
    if first_above is None or abs(first_above - k_max) <= ABS_TOL:
        return "near_upper_boundary"
    return "in_range"


def hopf_theory_status(delta: float, series: float) -> str:
    if delta <= ABS_TOL:
        return "theory_not_applicable"
    if series >= 1.0 - delta - 1e-12:
        return "theory_not_applicable"
    if delta > MAIN_DELTA + 1e-12:
        return "theory_not_applicable"
    return "main"


def hopf_reason(delta: float, series: float) -> str:
    if delta <= ABS_TOL:
        return "delta=0 is degenerate"
    if series >= 1.0 - delta - 1e-12:
        return "Eq. (7) is outside the incoherent branch K<1-delta"
    if delta > MAIN_DELTA + 1e-12:
        return "Eq. (7) is outside the pre-set window 0<delta<=0.20"
    return "Eq. (7) main window"


def route_before_death(rows: list[dict]) -> list[str]:
    order = []
    for row in rows:
        if row["state"] == "Death":
            break
        if not order or order[-1] != row["state"]:
            order.append(row["state"])
    return order


def npy_shape(path: Path, key: str) -> tuple[int, ...]:
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + ".npy") as handle:
            version = npformat.read_magic(handle)
            if version == (1, 0):
                shape, _fortran, _dtype = npformat.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _fortran, _dtype = npformat.read_array_header_2_0(handle)
            else:
                shape, _fortran, _dtype = npformat.read_array_header_1_0(handle)
            return tuple(int(value) for value in shape)


def load_cache() -> dict:
    if not CACHE_PATH.is_file():
        raise RuntimeError(f"missing Eq. (5) label cache {CACHE_PATH}")
    with CACHE_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("version") != CACHE_VERSION:
        raise RuntimeError(f"unexpected cache version {payload.get('version')}")
    context = payload["context"]
    if len(context["columns"]) != 164:
        raise RuntimeError(f"expected 164 columns, found {len(context['columns'])}")
    return context


def index_status() -> dict:
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        status = {
            row["status"]: int(row["n"])
            for row in connection.execute("SELECT status, COUNT(*) AS n FROM runs GROUP BY status")
        }
        incomplete = [dict(row) for row in connection.execute(
            "SELECT run_id, status, K, p, delta FROM runs WHERE status != 'completed' ORDER BY p, delta, K"
        )]
    finally:
        connection.close()
    paths = [
        {
            "p": float(row["p"]),
            "delta": float(row["Delta"]),
            "K": float(row["K"]),
            "output_npz": row["output_npz"],
            "output_json": row["output_json"],
            "n": int(row["N"]),
            "trial_id": int(row["trial_id"]),
        }
        for row in query_runs(status="completed")
    ]
    return {"status": status, "incomplete": incomplete, "paths": paths}


def macro_census(paths: list[dict]) -> dict:
    flags = Counter()
    missing_json = []
    for row in paths:
        path = Path(row["output_json"])
        if not path.is_file():
            missing_json.append(str(path))
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        flags["sample_order_parameter_" + str(bool(document.get("sample_order_parameter"))).lower()] += 1
        flags["order_parameter_dt_none" if document.get("order_parameter_dt") is None else "order_parameter_dt_set"] += 1
    shapes = Counter()
    missing_npz = []
    nonempty = []
    for row in paths:
        path = Path(row["output_npz"])
        if not path.is_file():
            missing_npz.append(str(path))
            continue
        try:
            r_shape = npy_shape(path, "r_times")
            snap_shape = npy_shape(path, "snapshot_times")
            theta_shape = npy_shape(path, "theta_grid")
        except Exception as exc:
            nonempty.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        shapes[("r", r_shape, "snap", snap_shape, "theta", theta_shape)] += 1
        if r_shape != (0,) or snap_shape != (0,) or theta_shape != (N_EXPECTED,):
            nonempty.append(
                {"path": str(path), "r": r_shape, "snapshot": snap_shape, "theta": theta_shape, "p": row["p"], "K": row["K"], "delta": row["delta"]}
            )
    keys = []
    if paths:
        with np.load(paths[0]["output_npz"], allow_pickle=False) as data:
            keys = list(data.files)
    return {
        "json_flags": dict(flags),
        "missing_json": missing_json,
        "missing_npz": missing_npz,
        "shape_counts": [
            {"r_times": list(key[1]), "snapshot_times": list(key[3]), "theta_grid": list(key[5]), "n": count}
            for key, count in shapes.items()
        ],
        "nonempty_or_unexpected": nonempty,
        "npz_keys": keys,
        "direct_hopf_series": bool(nonempty),
    }


def window_counts(path: str) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as data:
        rotation = np.asarray(data["rotation_numbers"], dtype=np.float64)
        starts = np.asarray(data["window_start"], dtype=np.float64)
        ends = np.asarray(data["window_end"], dtype=np.float64)
    counts = {}
    for name, start, end in WINDOWS:
        rho = rotation[hopf_mod.window_row(starts, ends, (start, end))]
        counts[name] = int(np.count_nonzero(np.abs(rho) < NEAR_TOL))
    return counts


def theory_bundle(deltas: list[float]) -> dict:
    print("solving death existence on the scanned widths", flush=True)
    by_delta = {}
    for delta in deltas:
        item = death_mod.existence_boundary(float(delta))
        if item.get("status") != "ok":
            raise RuntimeError(f"death boundary unresolved at delta={delta}: {item}")
        by_delta[round(float(delta), 5)] = item
    switch = float(death_mod.brentq(death_mod.endpoint_slope, 0.2, 0.4))
    crossing = float(hopf_mod.brentq(lambda value: hopf_mod.series_eq7(value) - (1.0 - value), 0.2, 0.4))
    quadrature = death_mod.quadrature_check()
    print("solving Eq. (6) at the previously published widths", flush=True)
    eq6 = []
    for delta in hopf_mod.EQ6_DELTAS:
        print(f"  Eq. (6) delta={delta}", flush=True)
        eq6.append(hopf_mod.solve_eq6(float(delta)))
    return {"death": by_delta, "switch": switch, "crossing": crossing, "quadrature": quadrature, "eq6": eq6}


def death_of(bundle: dict, delta: float) -> dict:
    return bundle["death"][round(float(delta), 5)]


def reproduce_baselines(columns: list[dict], expanded: dict, bundle: dict, checks: list[dict]) -> None:
    p1 = [column for column in columns if abs(column["p"] - 1.0) <= ABS_TOL]
    counts = Counter()
    for column in p1:
        for row in expanded[id(column)]:
            counts[row["state"]] += 1
    check(checks, "p=1 completed profiles", sum(counts.values()) == 1741, f"found {sum(counts.values())}")
    for state, expected in P1_COUNTS.items():
        check(checks, f"p=1 {state} count", counts[state] == expected, f"found {counts[state]}, previous report {expected}")
    for delta, expected in SERIES_ANCHORS.items():
        series = hopf_mod.series_eq7(delta)
        digits = 5 if delta in (0.05, 0.10, 0.15, 0.20) else 4
        ok = abs(series - expected) <= 5e-5 if digits == 5 else abs(round(series, 4) - expected) <= 1e-4
        check(checks, f"Eq. (7) series at delta={delta:.3f}", ok, f"computed {series:.5f}, published {expected}")
    additive_gap = abs(hopf_mod.series_eq7(0.10) - hopf_mod.additive_transcription(0.10))
    check(checks, "Eq. (7) is the product form", abs(additive_gap - 0.013) < 2e-3, f"absolute gap at delta=0.10 is {additive_gap:.5f}")
    crossing = bundle["crossing"]
    check(checks, "Eq. (7) crosses 1-delta near 0.252", abs(crossing - 0.252) < 1.5e-3, f"crossing {crossing:.4f}")
    roots = {}
    for item in bundle["eq6"]:
        delta = float(item["delta"])
        if not item["roots"]:
            roots[round(delta, 5)] = None
            continue
        if item["status"] != "ok" or len(item["roots"]) != 1:
            roots[round(delta, 5)] = "multiple"
            continue
        roots[round(delta, 5)] = item["roots"][0]
    for delta, expected in EQ6_ROOTS.items():
        root = roots.get(round(delta, 5))
        ok = isinstance(root, dict) and abs(root["K"] - expected) <= 5e-4 and abs(root["imag_residual"]) < 1e-8
        detail = "missing" if not isinstance(root, dict) else f"K={root['K']:.5f}, Im={root['imag_residual']:.2e}, ReGap={root['real_gap']:.2e}"
        check(checks, f"Eq. (6) root at delta={delta:.3f}", ok, detail)
    for delta in EQ6_ABSENT:
        root = roots.get(round(delta, 5))
        check(checks, f"Eq. (6) has no valid root at delta={delta:.3f}", root is None, f"stored {root}")
    for column in p1:
        delta = float(column["delta"])
        if delta <= ABS_TOL or delta > MAIN_DELTA + 1e-12:
            continue
        summary = hopf_mod.transition_summary(expanded[id(column)], hopf_mod.series_eq7(delta))
        expected = HOPF_EXITS.get(round(delta, 5))
        if expected is None:
            continue
        status, k_exit, state_after, inside = expected
        ok = summary["status"] == status
        if k_exit is not None:
            ok = ok and summary["K_exit"] is not None and abs(summary["K_exit"] - k_exit) <= ABS_TOL
            ok = ok and summary["state_after"] == state_after and bool(summary["inside_gap"]) is inside
        detail = f"status={summary['status']} K_exit={summary['K_exit']} after={summary['state_after']} inside={summary['inside_gap']}"
        check(checks, f"p=1 Hopf label exit at delta={delta:.3f}", ok, detail)
    residuals = []
    late = []
    early = []
    for column in p1:
        delta = float(column["delta"])
        item = death_of(bundle, delta)
        rows = expanded[id(column)]
        onset = first_state(rows, "Death")
        if onset is None:
            check(checks, f"p=1 Death exists at delta={delta:.3f}", False, "no Death label")
            continue
        previous = None
        for row in rows:
            if row is onset:
                break
            previous = row
        residual = float(onset["K"] - item["K_exist"])
        residuals.append(residual)
        steps = grid_steps(float(onset["K"]), float(item["K_exist"]), [row["K"] for row in rows])
        if steps is not None and steps >= 1:
            late.append(round(delta, 5))
        if residual < -EARLY_TOL:
            early.append(round(delta, 5))
        anchor = DEATH_ANCHORS.get(round(delta, 5))
        if anchor is None:
            continue
        mechanism, k_exist, k_obs, predecessor = anchor
        predecessor_state = None if previous is None else previous["state"]
        ok = (
            item["mechanism"] == mechanism
            and abs(item["K_exist"] - k_exist) <= 5e-4
            and abs(onset["K"] - k_obs) <= ABS_TOL
            and predecessor_state == predecessor
        )
        check(
            checks,
            f"p=1 death anchor delta={delta:.3f}",
            ok,
            f"mechanism={item['mechanism']} K={item['K_exist']:.4f} observed={onset['K']:.2f} previous={predecessor_state}",
        )
    expected_mechanism_ok = True
    for column in p1:
        delta = float(column["delta"])
        item = death_of(bundle, delta)
        if delta < bundle["switch"]:
            expected_mechanism_ok = expected_mechanism_ok and item["mechanism"] == "saddle-node" and item["n_roots"] == 1
        else:
            expected_mechanism_ok = expected_mechanism_ok and item["mechanism"] == "end-point" and item["n_roots"] == 0
    check(checks, "p=1 death branch follows delta*", expected_mechanism_ok, f"delta*={bundle['switch']:.4f}")
    residuals = np.array(residuals, dtype=float)
    check(checks, "p=1 death residual sign", bool(residuals.size == 41 and np.all(residuals > 0.0) and not early), f"n={residuals.size} min={residuals.min():.4f}")
    check(
        checks,
        "p=1 death residual range",
        abs(residuals.min() - 0.0028) <= 6e-4 and abs(float(np.median(residuals)) - 0.0273) <= 1.5e-3 and abs(residuals.max() - 0.0507) <= 1.5e-3,
        f"min={residuals.min():.4f} median={float(np.median(residuals)):.4f} max={residuals.max():.4f}",
    )
    check(checks, "p=1 death late set", set(late) == {round(value, 5) for value in LATE_DEATH}, f"late={late}")
    zero = death_of(bundle, 0.0)
    exact = 4.0 / (3.0 * np.sqrt(3.0))
    check(checks, "delta=0 death closed form", abs(zero["K_exist"] - exact) < 1e-6, f"K={zero['K_exist']:.6f} exact={exact:.6f}")


def build_rows(columns: list[dict], expanded: dict, bundle: dict, path_index: dict) -> list[dict]:
    analyses = []
    for column in columns:
        rows = expanded[id(column)]
        probability = float(column["p"])
        delta = float(column["delta"])
        couplings = [row["K"] for row in rows]
        series = float(hopf_mod.series_eq7(delta))
        k_hopf = series / probability
        hopf_summary = hopf_mod.transition_summary(rows, k_hopf)
        death = death_of(bundle, delta)
        k_death = float(death["K_exist"]) / probability
        k_endpoint = float(death["K_endpoint"]) / probability
        k_sn = None if death["mechanism"] != "saddle-node" else k_death
        leave = next((row for row in rows if row["state"] != "Incoherence"), None)
        death_row = first_state(rows, "Death")
        previous = None
        if death_row is not None:
            for row in rows:
                if row is death_row:
                    break
                previous = row
        persists = death_row is not None and all(row["state"] == "Death" for row in rows if row["K"] >= death_row["K"] - ABS_TOL)
        windows = None
        if death_row is not None:
            key = (quantize(probability), quantize(delta), quantize(death_row["K"]))
            path = path_index.get(key)
            if path is None:
                raise RuntimeError(f"no archive for Death cell p={probability} delta={delta} K={death_row['K']}")
            windows = window_counts(path)
        route = route_before_death(rows)
        non_inc = [state for state in route if state != "Incoherence"]
        residual_leave = None if leave is None else probability * float(leave["K"]) - series
        residual_exit = None if hopf_summary["K_exit"] is None else probability * float(hopf_summary["K_exit"]) - series
        residual_death = None if death_row is None else probability * float(death_row["K"]) - float(death["K_exist"])
        analyses.append(
            {
                "p": probability,
                "delta": delta,
                "couplings": couplings,
                "k_max": float(column["K_max"]),
                "n_cells": len(rows),
                "sequence": column["sequence"],
                "route": route,
                "mixed_route": len(set(non_inc)) > 1,
                "returned": route.count("Incoherence") > 1,
                "series": series,
                "K_hopf": k_hopf,
                "hopf_status": hopf_theory_status(delta, series),
                "hopf_reason": hopf_reason(delta, series),
                "hopf_scan": scan_status(couplings, column["holes"], k_hopf),
                "leave": leave,
                "K_PL": None if first_state(rows, "Partial Locking") is None else first_state(rows, "Partial Locking")["K"],
                "K_Lock": None if first_state(rows, "Locking") is None else first_state(rows, "Locking")["K"],
                "K_PD": None if first_state(rows, "Partial Death") is None else first_state(rows, "Partial Death")["K"],
                "K_Death": None if death_row is None else death_row["K"],
                "exit_status": hopf_summary["status"],
                "K_exit": hopf_summary["K_exit"],
                "exit_state": hopf_summary["state_after"],
                "inside_gap": bool(hopf_summary["inside_gap"]),
                "residual_leave": residual_leave,
                "residual_exit": residual_exit,
                "leave_steps": None if leave is None else grid_steps(float(leave["K"]), k_hopf, couplings),
                "death_mechanism": death["mechanism"],
                "K_exist_1": float(death["K_exist"]),
                "K_endpoint_1": float(death["K_endpoint"]),
                "K_sn_1": None if k_sn is None else float(death["K_exist"]),
                "K_death_scaled": k_death,
                "K_endpoint_scaled": k_endpoint,
                "death_scan": scan_status(couplings, column["holes"], k_death),
                "death_previous": None if previous is None else previous["state"],
                "death_persists": persists,
                "residual_death": residual_death,
                "death_steps": None if death_row is None else grid_steps(float(death_row["K"]), k_death, couplings),
                "windows": windows,
                "eq5": (1.0 - delta) / probability,
                "eq5_scan": scan_status(couplings, column["holes"], (1.0 - delta) / probability),
            }
        )
    return analyses


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def overlap(analyses: list[dict], predicate) -> tuple[tuple[float, ...], list[float]]:
    chosen = list(P_VALUES)
    while chosen:
        sets = []
        for probability in chosen:
            sets.append({round(item["delta"], 5) for item in analyses if abs(item["p"] - probability) <= ABS_TOL and predicate(item)})
        common = set.intersection(*sets) if sets else set()
        if common:
            return tuple(chosen), sorted(common)
        chosen = chosen[1:]
    return tuple(), []


def summarize(analyses: list[dict], bundle: dict) -> dict:
    def hopf_main_in_range(item: dict) -> bool:
        return item["hopf_status"] == "main" and item["hopf_scan"] == "in_range"

    def death_in_range(item: dict) -> bool:
        return item["death_scan"] == "in_range"

    hopf_ps, hopf_deltas = overlap(analyses, hopf_main_in_range)
    death_ps, death_deltas = overlap(analyses, death_in_range)
    return {
        "hopf_overlap_p": hopf_ps,
        "hopf_overlap_deltas": hopf_deltas,
        "death_overlap_p": death_ps,
        "death_overlap_deltas": death_deltas,
        "switch": bundle["switch"],
        "crossing": bundle["crossing"],
    }


def luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def plot_figure(analyses: list[dict], bundle: dict, summary: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), constrained_layout=True)
    plot_hopf_plane(axes[0, 0], analyses, bundle)
    plot_hopf_states(axes[0, 1], analyses)
    plot_death_plane(axes[1, 0], analyses, bundle)
    plot_death_residual(axes[1, 1], analyses, summary)
    figure.suptitle("Annealed $pK$ scaling of Eqs. (6)–(9): classifier comparison, not an eigenvalue test", fontsize=12)
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


def plot_hopf_plane(ax, analyses: list[dict], bundle: dict) -> None:
    delta_line = np.linspace(0.0, 1.0, 400)
    ax.plot(delta_line, 1.0 - delta_line, color="black", lw=1.4, ls=(0, (1.2, 1.4)), label="Eq. (5): $pK=1-\\delta$", zorder=3)
    main_d, main_k, extra_d, extra_k = [], [], [], []
    for delta in np.linspace(0.001, 0.40, 500):
        series = hopf_mod.series_eq7(float(delta))
        if series >= 1.0 - float(delta):
            break
        if delta <= MAIN_DELTA:
            main_d.append(delta)
            main_k.append(series)
        else:
            extra_d.append(delta)
            extra_k.append(series)
    ax.plot(main_d, main_k, color="#e66101", lw=2.2, label="Eq. (7), $0<\\delta\\leq 0.20$", zorder=4)
    if extra_d:
        ax.plot(extra_d, extra_k, color="#e66101", lw=1.6, ls="--", label="Eq. (7) series outside that window", zorder=4)
    for item in bundle["eq6"]:
        if item["roots"] and item["status"] == "ok":
            root = item["roots"][0]
            ax.scatter([item["delta"]], [root["K"]], marker="x", s=36, color="black", zorder=5)
    ax.scatter([], [], marker="x", s=36, color="black", label="Eq. (6) root (diagnostic)")
    for probability in P_VALUES:
        for item in analyses:
            if abs(item["p"] - probability) > ABS_TOL or item["hopf_status"] != "main":
                continue
            if item["hopf_scan"] not in {"in_range", "near_upper_boundary"} or item["leave"] is None:
                continue
            color = STATE_COLORS[item["leave"]["state"]]
            filled = item["exit_status"] == "sustained exit" and item["exit_state"] == "Locking"
            ax.scatter(
                [item["delta"]],
                [probability * float(item["leave"]["K"])],
                marker=P_MARKERS[probability],
                s=28 if filled else 36,
                facecolor=color if filled else "none",
                edgecolor=color,
                linewidths=1.1,
                zorder=6,
            )
    ax.axvspan(0.0, MAIN_DELTA, color="#e66101", alpha=0.05, zorder=0)
    ax.set_xlim(0.0, 0.42)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"frequency half-width $\delta$")
    ax.set_ylabel(r"scaled coupling $pK$")
    ax.set_title("A. Eq. (7) and the first non-Incoherence label")
    ax.legend(loc="lower right", fontsize=6.4, frameon=True, framealpha=0.94)
    ax.text(
        0.02,
        0.78,
        "Marker shape is $p$: circle 0.2, square 0.4,\n"
        "diamond 0.8, triangle 1. Color is the first\n"
        "label that leaves Incoherence.\n"
        "Filled: sustained Locking exit. Open: other leave.\n"
        "Out-of-range columns are omitted.",
        transform=ax.transAxes,
        fontsize=6.3,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )


def plot_hopf_states(ax, analyses: list[dict]) -> None:
    deltas = [round(0.025 * index, 5) for index in range(1, 9)]
    image = np.ones((len(P_VALUES), len(deltas), 3))
    for row_index, probability in enumerate(P_VALUES):
        for col_index, delta in enumerate(deltas):
            item = next(entry for entry in analyses if abs(entry["p"] - probability) <= ABS_TOL and abs(entry["delta"] - delta) <= 1e-8)
            if item["hopf_scan"] == "out_of_range":
                image[row_index, col_index] = (0.82, 0.82, 0.82)
                text, color = "out", "black"
            else:
                state = None if item["leave"] is None else item["leave"]["state"]
                hex_color = "#f0f0f0" if state is None else STATE_COLORS[state]
                value = hex_color.lstrip("#")
                image[row_index, col_index] = tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
                text = "—" if state is None else SHORT_OF[state]
                if item["hopf_scan"] == "near_upper_boundary":
                    text += "*"
                color = "white" if luminance(hex_color) < 0.45 else "black"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8, color=color)
    ax.imshow(image, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([f"{delta:.3f}" for delta in deltas], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(P_VALUES)))
    ax.set_yticklabels([f"p={probability:.1f}" for probability in P_VALUES])
    ax.set_xlabel("Gray: scaled Eq. (7) is above the scan. Star: within one K step of the upper edge.")
    ax.set_title("B. Phase-diagram comparison only;\nHopf instability not directly measured.", fontsize=10)


def grid_scaled_onset(k_exist: float, probability: float, step: float = GRID_STEP) -> tuple[float, float]:
    """Ceiling of the continuous threshold onto the uniform K grid.

    p K_grid = p ΔK * ceil( K_exist^(1) / (p ΔK) ).
    The returned pair is (K_grid, p K_grid). A ratio that lands on an
    integer is left on that grid point.
    """
    unit = float(probability) * float(step)
    ratio = float(k_exist) / unit
    nearest = round(ratio)
    if abs(ratio - nearest) <= 1e-6:
        count = int(nearest)
    else:
        count = int(math.ceil(ratio - 1e-12))
    return count * float(step), count * unit


def plot_death_plane(ax, analyses: list[dict], bundle: dict) -> None:
    deltas = np.array(sorted({round(item["delta"], 5) for item in analyses}))
    envelope, endpoint, saddle_d, saddle_k = [], [], [], []
    for delta in deltas:
        item = death_of(bundle, float(delta))
        envelope.append(item["K_exist"])
        endpoint.append(item["K_endpoint"])
        if item["mechanism"] == "saddle-node":
            saddle_d.append(delta)
            saddle_k.append(item["K_exist"])
    ax.plot(deltas, endpoint, color="#9a9a9a", lw=1.2, label="Endpoint candidate, Eq. (9)", zorder=2)
    ax.plot(saddle_d, saddle_k, color="#d95f02", lw=1.2, ls=(0, (2, 2)), label="Saddle-node candidate", zorder=2)
    split = next(index for index, delta in enumerate(deltas) if delta >= bundle["switch"])
    ax.plot(deltas[: split + 1], envelope[: split + 1], color="#0b7d74", lw=2.3, label="Continuous existence boundary", zorder=3)
    ax.plot(deltas[split:], envelope[split:], color="#0b7d74", lw=2.3, ls="--", zorder=3)
    for probability in P_VALUES:
        grid_x, grid_y = [], []
        obs_x, obs_y = [], []
        obs_by_delta: dict[float, float] = {}
        for item in analyses:
            if abs(item["p"] - probability) > ABS_TOL:
                continue
            k_grid, pk_grid = grid_scaled_onset(item["K_exist_1"], probability)
            scanned = any(abs(float(coupling) - k_grid) <= 1e-6 for coupling in item["couplings"])
            if scanned:
                grid_x.append(item["delta"])
                grid_y.append(pk_grid)
            if item["K_Death"] is not None and item["death_scan"] != "out_of_range":
                observed = probability * float(item["K_Death"])
                obs_x.append(item["delta"])
                obs_y.append(observed)
                obs_by_delta[item["delta"]] = observed
        for delta, pk_grid in zip(grid_x, grid_y):
            if delta in obs_by_delta and obs_by_delta[delta] > pk_grid + 1e-6:
                ax.plot(
                    [delta, delta],
                    [pk_grid, obs_by_delta[delta]],
                    color=P_COLORS[probability],
                    lw=0.8,
                    alpha=0.85,
                    zorder=3,
                    solid_capstyle="round",
                )
        if grid_x:
            ax.scatter(
                grid_x,
                grid_y,
                marker=P_MARKERS[probability],
                s=42,
                facecolor="white",
                edgecolor=P_COLORS[probability],
                linewidths=1.15,
                zorder=4,
                label=f"$pK_{{\\mathrm{{grid}}}}$, $p={probability:.1f}$",
            )
        if obs_x:
            ax.scatter(
                obs_x,
                obs_y,
                marker=P_MARKERS[probability],
                s=18,
                color=P_COLORS[probability],
                zorder=5,
                label=f"observed Death, $p={probability:.1f}$",
            )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.65, 1.35)
    ax.set_xlabel(r"frequency half-width $\delta$")
    ax.set_ylabel(r"scaled coupling $pK$")
    ax.set_title("C. Continuous boundary, grid ceiling, and observed Death")
    ax.legend(
        fontsize=5.4,
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.94,
        borderpad=0.3,
        labelspacing=0.22,
        handlelength=1.5,
        columnspacing=0.8,
    )
    ax.text(
        0.015,
        0.985,
        r"\begin{tabular}{@{}l@{}}"
        r"$\displaystyle pK_{\mathrm{grid}}(\delta)=p\Delta K\left\lceil\frac{K_{\mathrm{exist}}^{(1)}(\delta)}{p\Delta K}\right\rceil$\\"
        r"Open marker: grid ceiling. Filled marker: observed Death.\\"
        r"Curve to open marker: quantization error.\\"
        r"Open marker to filled marker: extra delay.\\"
        r"Dashed teal: end-point branch.\\"
        r"$p=0.2$ ceiling lies outside the scan."
        r"\end{tabular}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        usetex=True,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.94},
    )


def plot_death_residual(ax, analyses: list[dict], summary: dict) -> None:
    for probability in P_VALUES:
        in_range_x, in_range_y, edge_x, edge_y = [], [], [], []
        for item in analyses:
            if abs(item["p"] - probability) > ABS_TOL:
                continue
            if item["death_scan"] == "out_of_range" or item["residual_death"] is None:
                continue
            if item["death_scan"] == "in_range":
                in_range_x.append(item["delta"])
                in_range_y.append(item["residual_death"])
            else:
                edge_x.append(item["delta"])
                edge_y.append(item["residual_death"])
        if in_range_x:
            ax.scatter(in_range_x, in_range_y, marker=P_MARKERS[probability], s=26, color=P_COLORS[probability], label=f"$p={probability:.1f}$, in range", zorder=3)
        if edge_x:
            ax.scatter(edge_x, edge_y, marker=P_MARKERS[probability], s=28, facecolor="none", edgecolor=P_COLORS[probability], linewidths=1.1, zorder=3)
        step = probability * GRID_STEP
        ax.axhline(step, color=P_COLORS[probability], lw=0.6, ls=":", alpha=0.8)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.scatter([], [], marker="o", facecolor="none", edgecolor="black", label="near upper edge")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(r"frequency half-width $\delta$")
    ax.set_ylabel(r"$pK_{\mathrm{Death,obs}}-K_{\mathrm{exist}}^{(1)}(\delta)$")
    ax.set_title("D. Observed-Death residual in $pK$")
    ax.legend(fontsize=6.2, loc="upper right", frameon=True, framealpha=0.94)
    ax.text(
        0.02,
        0.02,
        "Dotted lines: one $K$ step in $pK$ units.\n"
        "A zero-step onset lies between 0 and that line.\n"
        "Out of range, no residual: $p=0.2$ (41), $p=0.4$ (30).",
        transform=ax.transAxes,
        fontsize=6.2,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )


def status_counts(analyses: list[dict], field: str) -> dict[float, Counter]:
    counts = {probability: Counter() for probability in P_VALUES}
    for item in analyses:
        counts[item["p"]][item[field]] += 1
    return counts


def late_snapshot_order(path: str) -> tuple[float, float]:
    """Mean and range of |R| on the stored snapshots at t >= 10000."""
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["snapshot_times"], dtype=np.float64)
        theta = np.asarray(data["snapshot_theta"], dtype=np.float64)
        order = np.abs(np.mean(np.exp(1j * theta), axis=1))
        late = order[times >= 10000.0 - 1e-6]
        if late.size == 0:
            raise RuntimeError(f"no late snapshot in {path}")
        return float(late.mean()), float(late.max() - late.min())


def snapshot_contrast(analyses: list[dict], path_index: dict) -> list[dict]:
    """Compare sparse |R| just before and at the first non-Incoherence label.

    The archives store eight phases, 2000 time units apart. That interval is
    much longer than an oscillator period, so the values cannot show whether
    R(t) is periodic.
    """
    rows = []
    for item in analyses:
        if item["hopf_status"] != "main" or item["leave"] is None:
            continue
        if item["hopf_scan"] not in {"in_range", "near_upper_boundary"}:
            continue
        leave_k = float(item["leave"]["K"])
        previous = [value for value in item["couplings"] if value < leave_k - ABS_TOL]
        if not previous:
            continue
        previous_k = previous[-1]
        leave_path = path_index[(quantize(item["p"]), quantize(item["delta"]), quantize(leave_k))]
        previous_path = path_index[(quantize(item["p"]), quantize(item["delta"]), quantize(previous_k))]
        leave_mean, leave_span = late_snapshot_order(leave_path)
        previous_mean, previous_span = late_snapshot_order(previous_path)
        rows.append(
            {
                "p": item["p"],
                "delta": item["delta"],
                "K_prev": previous_k,
                "K_leave": leave_k,
                "leave_state": item["leave"]["state"],
                "R_prev": previous_mean,
                "R_leave": leave_mean,
                "span_prev": previous_span,
                "span_leave": leave_span,
            }
        )
    return rows


def render_report(context: dict, analyses: list[dict], bundle: dict, summary: dict, checks: list[dict], macro: dict, index: dict, snapshots: list[dict]) -> str:
    lines = []
    add = lines.append
    hopf_counts = status_counts(analyses, "hopf_scan")
    death_counts = status_counts(analyses, "death_scan")
    add("# Eqs. (6)–(9) on ER networks: annealed $pK$ scaling check")
    add("")
    add("## 1. Research question")
    add("")
    add(
        "Ariaratnam and Strogatz (Phys. Rev. Lett. 86, 4278, 2001; arXiv:nlin/0012017) give an incoherent-branch Hopf condition and a death-equilibrium existence condition for the all-to-all Winfree model. "
        "The completed Stage 1 Erdős–Rényi scans are used here to ask whether those $p=1$ boundaries remain consistent with the saved finite-graph runs after the annealed replacement $K\\to pK$. "
        "The relation under test is $pK_c(p,\\delta)\\approx K_c(1,\\delta)$. It is a mean-field scaling hypothesis. It is not a theorem for a finite ER graph, and it is not a statement that a classifier label is the same object as a linear instability or a self-consistent equilibrium."
    )
    add("")
    add("## 2. Original Eqs. (6)–(9) and their assumptions")
    add("")
    add(
        "The formulas below are the ones already implemented for the $p=1$ checks in `validate_eq7.py` and `validate_death_boundary.py`, as recorded in their reports. "
        "This script calls those functions. It does not introduce a new transcription of the 2001 paper. "
        "Where the paper's display equation was reduced to a numerical condition in that earlier code, the reduction used here is that code."
    )
    add("")
    add("**Eq. (6), Hopf condition.** The stored solver treats the boundary of the stationary incoherent state as an implicit condition on the imaginary axis. A candidate is kept only when it is a unique root, the imaginary frequency is nonzero, and $K<1-\\delta$. The real part is reduced to the cubic already coded for the Landau pole, and the imaginary part is a subtracted principal-value integral. These roots diagnose the truncation. They were not the $p=1$ comparison baseline, and they are not refit here. The solver returns no valid root at $\\delta=0.225$ and $\\delta=0.250$.")
    add("")
    add("**Eq. (7), small-$\\delta$ truncation.** The printed product form, truncated before $O(\\delta^7)$, is")
    add("")
    add(r"\[K_H^{(7)}(\delta)=\frac{8\delta}{\pi}\left[1+\frac{16\delta^2}{\pi^2}+\frac{16(\pi^2+80)\delta^4}{\pi^4}\right].\]")
    add("")
    add(
        f"The additive expansion that drops the outer factor from the second and third terms is not this formula. At $\\delta=0.10$ the two expressions differ by {abs(hopf_mod.series_eq7(0.10) - hopf_mod.additive_transcription(0.10)):.5f}. "
        f"The pre-set comparison window is $0<\\delta\\le 0.20$. The series remains below the incoherent-branch edge $K=1-\\delta$ until $\\delta={bundle['crossing']:.4f}$, and the dashed extension in that interval is exploratory. It is not a global exact boundary. $\\delta=0$ is degenerate because every natural frequency equals 1."
    )
    add("")
    add("**Eq. (8), death self-consistency.** For $\\sigma\\ge 1+\\delta$,")
    add("")
    add(r"\[F_\delta(\sigma)=1+\frac{1}{2\delta}\int_{1-\delta}^{1+\delta}\sqrt{1-\frac{\omega^2}{\sigma^2}}\,d\omega,\qquad K(\sigma,\delta)=\frac{\sigma}{F_\delta(\sigma)},\]")
    add("")
    add(
        r"and $K_{\mathrm{exist}}(\delta)=\min K$. At $\delta=0$ the factor collapses to $1+\sqrt{1-1/\sigma^2}$, whose minimum is the exact value $4/(3\sqrt{3})$. "
        r"Critical points solve $F-\sigma F'=0$. A positive slope at an interior root identifies a local minimum."
    )
    add("")
    add(
        f"**Eq. (9), endpoint, and the saddle-node.** The minimum is not assigned by using the endpoint alone. "
        f"The sign of $F-\\sigma F'$ at $\\sigma=1+\\delta$ changes at $\\delta_*={bundle['switch']:.4f}$. "
        r"Below $\delta_*$ the minimum is an interior saddle-node, at $\sigma_{\mathrm{SN}}>1+\delta$. "
        r"Above $\delta_*$ there is no interior root and the minimum is the endpoint $\sigma=1+\delta$. "
        "Both candidates are retained. The existence envelope is whichever candidate is lower. "
        "Existence of a stopped equilibrium is not the statement that the saved initial condition converges to it."
    )
    add("")
    add(
        "The incoherent branch used by Eq. (6) and Eq. (7) additionally requires $K<1-\\delta$. "
        "Eq. (5), $K=1-\\delta$, is that branch edge. It is drawn as a reference and is not retested here."
    )
    add("")
    add("## 3. Derivation of the $pK$ scaling")
    add("")
    add("The saved model is")
    add("")
    add(r"\[\dot\theta_i=\omega_i-\frac{K}{N}\sin\theta_i\sum_{j=1}^{N}A_{ij}(1+\cos\theta_j),\]")
    add("")
    add(
        r"with $N=1000$, a zero diagonal, and $\omega\in[1-\delta,1+\delta]$ on the deterministic grid. "
        r"Replacing $A_{ij}$ by its expectation $p$ for $i\neq j$ produces a mean interaction $pK(N-1)/N$. "
        r"The hypothesis tested here drops the factor $(N-1)/N$ and replaces $K$ by $pK$ in the $p=1$ condition. "
        r"No degree correction and no spectral correction are added."
    )
    add("")
    add(r"For the explicit Hopf truncation this is $K_H^{(p)}(\delta)=K_H^{(7)}(\delta)/p$ on the same $\delta$ window. The window is unchanged because $pK_H^{(p)}=K_H^{(7)}$ lies below $1-\delta$ exactly when the $p=1$ series does.")
    add("")
    add(
        r"For death, the auxiliary amplitude $\sigma$ stays inside $F_\delta$. The scaled self-consistency is $pK=\sigma/F_\delta(\sigma)$, so "
        r"$K_{\mathrm{Death,exist}}^{(p)}(\delta)=p^{-1}\min_{\sigma\ge 1+\delta}\sigma/F_\delta(\sigma)$. "
        r"Multiplying $F_\delta$ by $p$ again would double-count the mean-field factor. "
        r"The saddle-node and endpoint candidates scale by the same $1/p$. Their crossing $\delta_*$ does not depend on $K$ and is therefore the same at every $p$."
    )
    add("")
    add(r"Eq. (5) scales in the same way: $K_{\mathrm{Eq5}}^{(p)}(\delta)=(1-\delta)/p$. A one-line finite-$N$ shift of size $1/999$ is smaller than the coupling step 0.05 and is not fitted.")
    add("")
    add("## 4. Dataset audit")
    add("")
    add(
        f"The Stage 1 index `{DATABASE}` was reread for this check. Status counts: "
        + ", ".join(f"{key} {value}" for key, value in sorted(index["status"].items()))
        + f". The Eq. (5) label cache records {context['audit']['status']}. "
        "Labels below come from that cache, version 1, built with Classifier v2.0.0. "
        f"Config SHA-256 `{context['config_sha']}` matches the archive manifest: {context['config_matches_manifest']}. "
        f"`classify.py` SHA-256 in the cache is `{context['classify_sha']}`; the manifest checksum is `{context['manifest_classify_sha']}`. "
        "The Eq. (5) note already traced that worktree difference to phase-diagram drawing helpers. This script does not reclassify and does not change thresholds."
    )
    add("")
    if index["incomplete"]:
        add("Incomplete rows, excluded from every onset: " + "; ".join(
            f"`{row['run_id']}` ({row['status']}, p={row['p']}, delta={row['delta']}, K={row['K']})" for row in index["incomplete"]
        ) + ".")
        add("")
    add(
        f"Completed runs in the cache: {context['classify_meta']['n_runs']}. "
        f"Distinct $p$ values are 0.2, 0.4, 0.8, and 1.0. There is no $p=0.05$ and no $p=0.60$. "
        f"Maximum $|\u03c9|$ discrepancy recorded by the cache is {context['classify_meta']['omega_error']:.3e}. "
        f"Network seed {context['networks']['network_seed']}, nested graphs {context['networks']['nested']}, "
        f"adjacency mismatches {len(context['hashes']['mismatch'])} of {context['hashes']['checked']} metadata files. "
        f"Git HEAD at the cache build was `{context['git_head']}`. Git HEAD for this script is `{git_head()}`."
    )
    add("")
    coverage_rows = []
    for probability in P_VALUES:
        group = [item for item in analyses if abs(item["p"] - probability) <= ABS_TOL]
        k_counts = Counter(item["n_cells"] for item in group)
        coverage_rows.append([
            f"{probability:.1f}",
            str(len(group)),
            ", ".join(f"{count} cols × {width} K" for width, count in sorted(k_counts.items())),
            str(min(item["k_max"] for item in group)),
            str(max(item["k_max"] for item in group)),
        ])
    add(markdown_table(["p", "delta columns", "K counts", "smallest Kmax", "largest Kmax"], coverage_rows))
    add("")
    add("Coupling runs from 0 in steps of 0.05. Thirty-eight widths at each $p$ stop at $K=2$. The widths $\\delta\\in\\{0.05,0.20,0.80\\}$ continue to $K=3$, except that $p=0.2$, $\\delta=1$ stops at $K=1.90$ because $K=1.95$ is running and $K=2.00$ is pending. A prediction above the scanned $K$ is out of range. It is not counted as evidence against the formula.")
    add("")
    add("## 5. $p=1$ baseline reproduction")
    add("")
    add("The $p=1$ labels, the Eq. (7) series, the Eq. (6) roots, and the death-existence values were compared with the published validation notes before any $p<1$ residual was interpreted. The ODE was not rerun.")
    add("")
    add(markdown_table(["Result", "Check", "Detail"], [["PASS" if item["ok"] else "FAIL", item["name"], item["detail"]] for item in checks]))
    add("")
    add(
        f"Quadrature and search-range lines from the original death solver: {'; '.join(bundle['quadrature'])}. "
        "The baseline matches the earlier $p=1$ notes on the state counts, the eight main-window Hopf label exits, the Eq. (6) roots, the death anchors, the saddle-node/end-point split, and the sign and scale of the death residuals. "
        "The Eq. (7) series, not the Eq. (6) root, remains the comparison baseline inside $0<\\delta\\le 0.20$."
    )
    add("")
    add("## 6. Hopf phase-diagram consistency")
    add("")
    add(
        "A Hopf threshold is a linear-stability statement about the stationary incoherent state. "
        "The quantity available here is a W3 classifier label. "
        "$K_{\\mathrm{leave}}$ is the first scanned coupling whose label is not Incoherence. "
        "$K_{\\mathrm{exit}}$ is the sustained-exit coupling from the $p=1$ Eq. (7) note: the column must start in Incoherence, and Incoherence must not return. "
        "Neither number is a Hopf eigenvalue. "
        r"The plotted and tabulated residual is $\Delta(pK)=pK_{\mathrm{proxy}}-K_H^{(7)}(\delta)$. It is a label-proxy residual."
    )
    add("")
    add("Coverage of the scaled Eq. (7) curve. `theory not applicable` includes $\\delta=0$, every width outside $0<\\delta\\le 0.20$, and any series value that has crossed $1-\\delta$.")
    add("")
    hopf_rows = []
    for probability in P_VALUES:
        group = [item for item in analyses if abs(item["p"] - probability) <= ABS_TOL]
        applicable = [item for item in group if item["hopf_status"] == "main"]
        not_applicable = len(group) - len(applicable)
        scan = Counter(item["hopf_scan"] for item in applicable)
        hopf_rows.append([
            f"{probability:.1f}",
            str(scan["in_range"]),
            str(scan["near_upper_boundary"]),
            str(scan["out_of_range"]),
            str(scan["insufficient_k_resolution"]),
            str(not_applicable),
        ])
    add(markdown_table(["p", "In range", "Near upper boundary", "Out of range", "Insufficient K resolution", "Theory not applicable"], hopf_rows))
    add("")
    add("Main-window columns. The residual uses the first non-Incoherence label. A sustained Locking exit is reported separately and is still only a label change.")
    add("")
    proxy_rows = []
    for item in analyses:
        if item["hopf_status"] != "main":
            continue
        leave_state = None if item["leave"] is None else item["leave"]["state"]
        proxy_rows.append([
            f"{item['p']:.1f}",
            f"{item['delta']:.3f}",
            fmt(item["series"], 4),
            fmt(item["K_hopf"], 3),
            item["hopf_scan"].replace("_", " "),
            fmt(None if item["leave"] is None else item["leave"]["K"], 2),
            leave_state or "—",
            item["exit_status"],
            fmt(item["K_exit"], 2),
            fmt(item["residual_leave"], 3),
            "yes" if item["returned"] or item["mixed_route"] else "no",
        ])
    add(markdown_table(
        ["p", "delta", "K_H^(7)", "K_H(p)", "Scan", "K_leave", "First label", "Exit rule", "K_exit", "pK_leave - K_H^(7)", "Mixed or return"],
        proxy_rows,
    ))
    add("")
    add("State sequences in that window:")
    add("")
    for item in analyses:
        if item["hopf_status"] != "main":
            continue
        add(f"- p={item['p']:.1f}, delta={item['delta']:.3f}: {item['sequence']}")
    add("")
    add("## 7. Direct Hopf evidence")
    add("")
    shape_text = "; ".join(
        f"r_times={item['r_times']} snapshot_times={item['snapshot_times']} theta_grid={item['theta_grid']} on {item['n']} files"
        for item in macro["shape_counts"]
    ) or "no arrays read"
    add(
        "The completed metadata were read for every saved run. "
        f"Order-parameter flags: {macro['json_flags']}. Missing JSON files: {len(macro['missing_json'])}. Missing NPZ files: {len(macro['missing_npz'])}. "
        f"Array headers: {shape_text}. "
        f"NPZ keys in one completed archive: {', '.join(macro['npz_keys'])}. "
        "There is no stored mean-field $H(t)$, and `r_times` is empty in every completed archive: the run configuration did not sample $R(t)$. "
        "`theta_grid` is one phase vector. `snapshot_theta` has eight phases. In the archives inspected for the Hopf window those times are $t=0,2000,\\ldots,14000$."
    )
    add("")
    add(
        "Eight phases separated by 2000 time units are not a trajectory that can show a Hopf oscillation. "
        "An oscillator period is of order $2\\pi$, so a sample every 2000 time units aliases any cycle and cannot establish periodicity, growth of an oscillatory amplitude, or a clean distinction between a limit cycle and a slow transient. "
        "The only macroscopic quantity computed from these snapshots is the order-parameter magnitude $|R|$ at $t\\ge 10000$, averaged over those three late samples. "
        "It is reported below as a descriptive check of whether the first non-Incoherence label coincides with a large ordered snapshot. It is not a Hopf eigenvalue."
    )
    add("")
    add(snapshot_table(snapshots))
    add("")
    add("## 8. Death existence comparison")
    add("")
    add(
        r"The observed onset is the first W3 label with near-zero fraction exactly 1 at $|\rho|<0.001$. "
        r"The residual is $\Delta(pK)=pK_{\mathrm{Death,obs}}-K_{\mathrm{exist}}^{(1)}(\delta)$. "
        "It compares an attractor label with an existence threshold. "
        "A positive residual can mean that another attractor is still selected, that the finite-time run has not settled, or that the grid point above the threshold is the first available sample. "
        "A negative residual, Death while the stopped equilibrium does not yet exist, is the one-sided discrepancy singled out for inspection."
    )
    add("")
    add("Coverage of the scaled existence envelope:")
    add("")
    death_rows = []
    for probability in P_VALUES:
        group = [item for item in analyses if abs(item["p"] - probability) <= ABS_TOL]
        scan = Counter(item["death_scan"] for item in group)
        early = sum(1 for item in group if item["residual_death"] is not None and item["residual_death"] < -EARLY_TOL)
        no_death = sum(1 for item in group if item["K_Death"] is None and item["death_scan"] == "in_range")
        death_rows.append([
            f"{probability:.1f}",
            str(scan["in_range"]),
            str(scan["near_upper_boundary"]),
            str(scan["out_of_range"]),
            str(scan["insufficient_k_resolution"]),
            str(early),
            str(no_death),
        ])
    add(markdown_table(["p", "In range", "Near upper boundary", "Out of range", "Insufficient K resolution", "Death before existence", "In range but no Death"], death_rows))
    add("")
    window_bad = [item for item in analyses if item["windows"] is not None and (item["windows"]["W1"] != N_EXPECTED or item["windows"]["W2"] != N_EXPECTED or item["windows"]["W3"] != N_EXPECTED)]
    window_w3 = [item for item in analyses if item["windows"] is not None and item["windows"]["W3"] != N_EXPECTED]
    add(
        f"At every first Death cell the W3 near-zero count is {N_EXPECTED if not window_w3 else 'not ' + str(N_EXPECTED)}. "
        f"Cells whose W1 or W2 count is below {N_EXPECTED}: {len(window_bad)}. "
        + (
            "Every first Death cell is inside the cut on W1, W2, and W3, so the onset is not produced by reading only W3. "
            "Agreement of the three windows still does not identify the saved state with the self-consistent equilibrium."
            if not window_bad
            else "A short earlier window means the three rotation-number windows do not yet agree."
        )
    )
    add("")
    if window_bad:
        add(markdown_table(
            ["p", "delta", "K_Death", "W1", "W2", "W3"],
            [[f"{item['p']:.1f}", f"{item['delta']:.3f}", fmt(item["K_Death"], 2), item["windows"]["W1"], item["windows"]["W2"], item["windows"]["W3"]] for item in window_bad[:40]],
        ))
        add("")
    exceptions = [
        item for item in analyses
        if (item["residual_death"] is not None and item["death_steps"] is not None and abs(item["death_steps"]) > 1)
        or (item["residual_death"] is not None and item["residual_death"] < -EARLY_TOL)
        or (item["K_Death"] is None and item["death_scan"] == "in_range")
        or not item["death_persists"] and item["K_Death"] is not None
    ]
    add("Columns with an early onset, a delay of more than one $K$ step, a missing in-range onset, or a Death label that does not persist:")
    add("")
    if not exceptions:
        add("None.")
    else:
        add(markdown_table(
            ["p", "delta", "Scan", "K_exist(p)", "K_Death", "Previous", "Steps", "Residual pK", "Persists"],
            [[
                f"{item['p']:.1f}", f"{item['delta']:.3f}", item["death_scan"].replace("_", " "),
                fmt(item["K_death_scaled"], 3), fmt(item["K_Death"], 2), item["death_previous"] or "—",
                "—" if item["death_steps"] is None else str(item["death_steps"]), fmt(item["residual_death"], 3),
                "yes" if item["death_persists"] else "no",
            ] for item in exceptions],
        ))
    add("")
    add("## 9. Saddle-node versus endpoint")
    add("")
    add(
        f"The mechanism is a property of $F_\\delta$ and does not change with $p$. It switches at $\\delta_*={bundle['switch']:.4f}$. "
        "On the saddle-node side the endpoint value lies above the envelope. Using Eq. (9) alone would place the existence boundary too high wherever the two candidates differ by more than the grid. "
        "That separation is large only at very small $\\delta$: at $\\delta=0$ the endpoint is $K=1$ and the envelope is $0.7698$. "
        "By $\\delta=0.10$ the gap has fallen to about $0.016$, and by $\\delta=0.20$ it is about $0.001$. "
        "A count of which candidate lies closer to the observed onset is then decided by the $0.05$ bin, not by the mechanism. "
        "The mechanism itself remains the sign of $F-\\sigma F'$ at the endpoint."
    )
    add("")
    mechanism_rows = []
    for probability in P_VALUES:
        for mechanism in ("saddle-node", "end-point"):
            group = [item for item in analyses if abs(item["p"] - probability) <= ABS_TOL and item["death_mechanism"] == mechanism and item["death_scan"] == "in_range" and item["residual_death"] is not None]
            if not group:
                mechanism_rows.append([f"{probability:.1f}", mechanism, "0", "—", "—", "—"])
                continue
            closer = 0
            for item in group:
                if mechanism == "saddle-node":
                    endpoint_residual = abs(item["p"] * float(item["K_Death"]) - item["K_endpoint_1"])
                    envelope_residual = abs(item["residual_death"])
                    closer += int(envelope_residual < endpoint_residual - 1e-8)
            predecessors = Counter(item["death_previous"] for item in group)
            within = sum(1 for item in group if item["death_steps"] is not None and 0 <= item["death_steps"] <= 1)
            mechanism_rows.append([
                f"{probability:.1f}",
                mechanism,
                str(len(group)),
                ", ".join(f"{key or 'none'} {value}" for key, value in sorted(predecessors.items(), key=lambda pair: str(pair[0]))),
                f"{within}/{len(group)}",
                "—" if mechanism != "saddle-node" else f"{closer}/{len(group)}",
            ])
    add(markdown_table(
        ["p", "Mechanism", "In-range onsets", "State before Death", "Within 0 or 1 K step", "Envelope closer than endpoint"],
        mechanism_rows,
    ))
    add("")
    add("## 10. Grid resolution and coverage")
    add("")
    add(
        "The coupling step is 0.05. In the $pK$ coordinate one step has length $0.05p$: 0.010 at $p=0.2$, 0.020 at $p=0.4$, 0.040 at $p=0.8$, and 0.050 at $p=1$. "
        "A residual smaller than that length is what the upper edge of a bin produces when the threshold sits between two scanned points. "
        "It is not sub-grid precision. "
        "Near-upper columns are those whose first scanned point above the prediction is the last point in the column, or whose prediction equals that last point. "
        "They are drawn as open markers and are not pooled into the common-interval medians. "
        "Out-of-range columns have no residual."
    )
    add("")
    add("## 11. Comparison across $p$")
    add("")
    add(
        "Medians are not taken over every saved width. "
        f"For the Hopf proxy, the common in-range window uses p={', '.join(f'{value:.1f}' for value in summary['hopf_overlap_p']) or 'none'} "
        f"and {len(summary['hopf_overlap_deltas'])} shared widths. "
        f"For death, the common in-range window uses p={', '.join(f'{value:.1f}' for value in summary['death_overlap_p']) or 'none'} "
        f"and {len(summary['death_overlap_deltas'])} shared widths. "
        "If the four probabilities do not all qualify, the smallest $p$ is dropped until the intersection is nonempty. That partial overlap is labeled as such."
    )
    add("")
    add(comparison_table(analyses, summary))
    add("")
    add(conclusion_text(analyses, summary, snapshots))
    add("")
    add("## Source record")
    add("")
    add(
        f"Theory implementation: `stage2_classifier/theoretical_validation/eq7_hopf/validate_eq7.py` SHA-256 `{file_sha256(ROOT / 'stage2_classifier/theoretical_validation/eq7_hopf/validate_eq7.py')}`, "
        f"`stage2_classifier/theoretical_validation/death_boundary/validate_death_boundary.py` SHA-256 `{file_sha256(ROOT / 'stage2_classifier/theoretical_validation/death_boundary/validate_death_boundary.py')}`. "
        f"Classifier config SHA-256 `{file_sha256(ROOT / 'stage2_classifier/classifier_v2/config.json')}`. "
        f"This script SHA-256 `{file_sha256(Path(__file__))}`. "
        f"Label cache `{CACHE_PATH}`, version {CACHE_VERSION}. "
        "No ODE was rerun, no threshold was changed, and no new network correction was fitted."
    )
    add("")
    return "\n".join(lines) + "\n"


def comparison_table(analyses: list[dict], summary: dict) -> str:
    lines = ["Hopf label proxy on each $p$ separately, restricted to the Eq. (7) window and to in-range columns. These four medians are not one cross-$p$ average.", ""]
    rows = []
    for probability in P_VALUES:
        group = [item for item in analyses if abs(item["p"] - probability) <= ABS_TOL and item["hopf_status"] == "main" and item["hopf_scan"] == "in_range"]
        locking = [item for item in group if item["exit_status"] == "sustained exit" and item["exit_state"] == "Locking"]
        inside = [item for item in locking if item["inside_gap"]]
        residuals = [item["residual_leave"] for item in group if item["residual_leave"] is not None]
        rows.append([
            f"{probability:.1f}",
            str(len(group)),
            f"{len(inside)}/{len(locking) if locking else 0}",
            fmt(median(residuals), 3),
            fmt(min(residuals) if residuals else None, 3),
            fmt(max(residuals) if residuals else None, 3),
        ])
    lines.append(markdown_table(["p", "In-range widths", "Sustained Locking exit inside the gap", "Median leave residual", "Min", "Max"], rows))
    lines.append("")
    if summary["hopf_overlap_deltas"]:
        lines.append(
            "Shared Hopf widths, p=" + ", ".join(f"{value:.1f}" for value in summary["hopf_overlap_p"]) + ": "
            + ", ".join(f"{value:.3f}" for value in summary["hopf_overlap_deltas"]) + "."
        )
        lines.append("")
        collapse_rows = []
        for delta in summary["hopf_overlap_deltas"]:
            group = [
                item for item in analyses
                if round(item["delta"], 5) == round(delta, 5) and item["p"] in summary["hopf_overlap_p"] and item["residual_leave"] is not None
            ]
            values = [item["p"] * float(item["leave"]["K"]) for item in group if item["leave"] is not None]
            states = ", ".join(f"{item['p']:.1f}:{None if item['leave'] is None else SHORT_OF[item['leave']['state']]}" for item in sorted(group, key=lambda entry: entry["p"]))
            spread = None if len(values) < 2 else max(values) - min(values)
            collapse_rows.append([f"{delta:.3f}", str(len(values)), fmt(spread, 3), states])
        lines.append(markdown_table(["delta", "p with a leave", "Range of pK_leave", "First label by p"], collapse_rows))
        lines.append("")
    lines.append("Death onset on each $p$ separately, in-range columns only.")
    lines.append("")
    death_rows = []
    for probability in P_VALUES:
        group = [item for item in analyses if abs(item["p"] - probability) <= ABS_TOL and item["death_scan"] == "in_range" and item["residual_death"] is not None]
        residuals = [item["residual_death"] for item in group]
        within = sum(1 for item in group if item["death_steps"] is not None and 0 <= item["death_steps"] <= 1)
        beyond = sum(1 for item in group if item["death_steps"] is not None and item["death_steps"] > 1)
        early = sum(1 for item in residuals if item < -EARLY_TOL)
        death_rows.append([
            f"{probability:.1f}",
            str(len(group)),
            f"{within}/{len(group) if group else 0}",
            str(beyond),
            str(early),
            fmt(median(residuals), 3),
            fmt(max(residuals) if residuals else None, 3),
        ])
    lines.append(markdown_table(["p", "In-range onsets", "0 or 1 K step", "More than one K step", "Before existence", "Median residual", "Max residual"], death_rows))
    lines.append("")
    if summary["death_overlap_deltas"]:
        lines.append(
            "Shared death widths, p=" + ", ".join(f"{value:.1f}" for value in summary["death_overlap_p"]) + ": "
            + ", ".join(f"{value:.3f}" for value in summary["death_overlap_deltas"]) + "."
        )
        lines.append("")
        collapse_rows = []
        for delta in summary["death_overlap_deltas"]:
            group = [item for item in analyses if round(item["delta"], 5) == round(delta, 5) and item["p"] in summary["death_overlap_p"] and item["K_Death"] is not None]
            values = [item["p"] * float(item["K_Death"]) for item in group]
            spread = None if len(values) < 2 else max(values) - min(values)
            collapse_rows.append([
                f"{delta:.3f}",
                str(len(values)),
                fmt(spread, 3),
                ", ".join(f"{item['p']:.1f}:{fmt(item['p'] * item['K_Death'], 3)}" for item in sorted(group, key=lambda entry: entry["p"])),
            ])
        lines.append(markdown_table(["delta", "p with Death", "Range of pK_Death", "pK_Death by p"], collapse_rows))
    return "\n".join(lines)


def snapshot_table(rows: list[dict]) -> str:
    lines = [
        "Late-snapshot $|R|$ across the first label change in the Eq. (7) window. "
        "`Before` is the saved cell immediately below that label, and `At leave` is the first non-Incoherence cell. "
        "Each number is the mean of $|R|$ at $t=10000$, $12000$, and $14000$.",
        "",
    ]
    summary_rows = []
    for probability in P_VALUES:
        group = [row for row in rows if abs(row["p"] - probability) <= ABS_TOL]
        if not group:
            summary_rows.append([f"{probability:.1f}", "0", "—", "—", "—"])
            continue
        ordered = sum(1 for row in group if row["R_leave"] > 0.5)
        summary_rows.append([
            f"{probability:.1f}",
            str(len(group)),
            fmt(median([row["R_prev"] for row in group]), 3),
            fmt(median([row["R_leave"] for row in group]), 3),
            f"{ordered}/{len(group)}",
        ])
    lines.append(markdown_table(
        ["p", "Columns", "Median late |R| before", "Median late |R| at leave", "Leave cells with late |R|>0.5"],
        summary_rows,
    ))
    lines.append("")
    wanted = {(1.0, 0.10), (1.0, 0.20), (0.8, 0.10), (0.4, 0.05), (0.2, 0.05)}
    examples = [
        row for row in rows
        if (round(row["p"], 5), round(row["delta"], 5)) in {(round(p, 5), round(delta, 5)) for p, delta in wanted}
    ]
    lines.append("Examples of the same three-point average:")
    lines.append("")
    lines.append(markdown_table(
        ["p", "delta", "K before", "late |R|", "K leave", "Label", "late |R|"],
        [[
            f"{row['p']:.1f}", f"{row['delta']:.3f}", fmt(row["K_prev"], 2), fmt(row["R_prev"], 3),
            fmt(row["K_leave"], 2), row["leave_state"], fmt(row["R_leave"], 3),
        ] for row in examples],
    ))
    lines.append("")
    lines.append(
        "At $p=1$ every Locking cell in this window has late $|R|>0.5$ (median $0.882$). "
        "The preceding cell is usually near $0.1$, except at $\\delta=0.20$, where the Incoherence cell just below the single Locking point is already ordered. "
        "At $p\\le 0.8$ the first Partial Locking label usually does not raise late $|R|$ above $0.5$. "
        "That disagreement means the early Partial Locking label is not, by itself, evidence that a macroscopic oscillation has begun. "
        "The sampling interval still cannot show whether $|R|$ oscillates between the stored times."
    )
    return "\n".join(lines)


def conclusion_text(analyses: list[dict], summary: dict, snapshots: list[dict]) -> str:
    lines = ["## 12. Main conclusions and unresolved questions", ""]
    # The sources section is numbered 12 in render_report; this heading is inserted before it.
    # Renumber by returning body only. The caller uses heading 11's following text.
    lines[0] = "## 12. Main conclusions and unresolved questions"
    main = [item for item in analyses if item["hopf_status"] == "main" and item["hopf_scan"] == "in_range"]
    locking_inside = [item for item in main if item["exit_status"] == "sustained exit" and item["exit_state"] == "Locking" and item["inside_gap"]]
    other_leave = [item for item in main if item["leave"] is not None and item["leave"]["state"] not in {"Locking", "Partial Locking"}]
    early = [item for item in analyses if item["residual_death"] is not None and item["residual_death"] < -EARLY_TOL]
    late = [item for item in analyses if item["death_scan"] == "in_range" and item["death_steps"] is not None and item["death_steps"] > 1]
    out_hopf = Counter(item["p"] for item in analyses if item["hopf_status"] == "main" and item["hopf_scan"] == "out_of_range")
    out_death = Counter(item["p"] for item in analyses if item["death_scan"] == "out_of_range")
    negative = [item for item in main if item["residual_leave"] is not None and item["residual_leave"] < -EARLY_TOL]
    lines.append(
        f"1. The scaled Eq. (7) series is bracketed by a sustained Incoherence-to-Locking label in {len(locking_inside)} of {len(main)} in-range columns. "
        f"Six of those are the $p=1$ widths already reported, and $\\delta=0.15$ at $p=1$ still misses the stored gap by $0.001$ in $K$. "
        f"On the other in-range columns the first label is Partial Locking, and {len(negative)} of {len(main)} in-range leave residuals are negative: the label changes at a $pK$ below $K_H^{{(7)}}$. "
        "Panel A shows that failure to collapse. It is a statement about the classifier proxy, not about the Hopf eigenvalue."
    )
    lines.append("")
    if other_leave:
        listed = ", ".join(f"p={item['p']:.1f}, delta={item['delta']:.3f} ({item['leave']['state']})" for item in other_leave)
        lines.append(f"2. The first non-Incoherence label is not Locking or Partial Locking in {len(other_leave)} in-range main-window columns: {listed}.")
    else:
        small = [row for row in snapshots if row["p"] < 1.0 - ABS_TOL]
        small_ordered = sum(1 for row in small if row["R_leave"] > 0.5)
        lines.append(
            "2. Inside the Eq. (7) window and inside the scan, every recorded first departure from Incoherence is Locking or Partial Locking. "
            "Partial Death does not take that role on these columns. "
            f"Of the {len(small)} coarse snapshot pairs at $p\\le 0.8$, {small_ordered} have late $|R|>0.5$ on the Partial Locking or Locking cell. "
            "The early Partial Locking labels are therefore not a macroscopic ordering transition in the stored snapshots."
        )
    lines.append("")
    lines.append(
        "3. No finely sampled $R(t)$ or $H(t)$ is stored. The eight phase snapshots cannot resolve a cycle. "
        "Every quantitative Hopf statement in this note is a classifier-proxy comparison, supplemented only by the coarse $|R|$ check in section 7. "
        "The Hopf instability itself was not measured."
    )
    lines.append("")
    lines.append(
        f"4. Death before the existence envelope occurs in {len(early)} columns. "
        + ("No saved Death label precedes the scaled existence threshold." if not early else "Those columns are listed in section 8 and are the discrepancies that would challenge the existence calculation.")
        + f" Delays of more than one $K$ step on in-range columns: {len(late)}."
    )
    lines.append("")
    lines.append(
        "5. The saddle-node branch, not the Eq. (9) endpoint, determines the envelope for $\\delta<\\delta_*$. "
        "The endpoint remains the envelope only for $\\delta>\\delta_*$. The two candidates are drawn separately in panel C."
    )
    lines.append("")
    lines.append(
        "6. Out-of-range counts for the scaled Eq. (7) curve, among widths where the series itself is applicable: "
        + (", ".join(f"p={probability:.1f}: {count}" for probability, count in sorted(out_hopf.items())) or "none")
        + ". Out-of-range counts for the death envelope: "
        + (", ".join(f"p={probability:.1f}: {count}" for probability, count in sorted(out_death.items())) or "none")
        + ". These scans end before the predicted coupling. They are coverage limits."
    )
    lines.append("")
    if len(summary["hopf_overlap_p"]) < 4:
        lines.append(
            f"7. The four probabilities do not share an in-range Eq. (7) width. The overlap used above drops the smallest missing $p$ and keeps p={', '.join(f'{value:.1f}' for value in summary['hopf_overlap_p']) or 'none'}. "
            "A trend with decreasing $p$ is not estimated from the widths that only the larger $p$ can see."
        )
    else:
        lines.append("7. All four probabilities have a nonempty shared in-range Eq. (7) interval. The cross-$p$ comparison in section 11 is restricted to that interval.")
    lines.append("")
    lines.append(
        "8. The saved design has one network, one initial phase, and a finite observation time. "
        "At the first Death cell, W1, W2, and W3 are all entirely inside $|\\rho|<0.001$, so that onset is not an artifact of reading only W3. "
        "The $K/N$ normalization and the difference between existence and attraction can still move an onset by a fraction of a grid step. "
        "The archives do not separate those effects from an error in the annealed $pK$ hypothesis. "
        "A finer $K$ scan or a linear-stability calculation would be required before a label residual is called a Hopf error."
    )
    lines.append("")
    return "\n".join(lines)


def parser_checks(columns: list[dict], expanded: dict) -> None:
    for column in columns:
        rows = expanded[id(column)]
        if [round(row["K"], 5) for row in rows] != [round(float(value), 5) for value in column["K"]]:
            raise RuntimeError("expanded K grid does not match the cache")
        death = first_state(rows, "Death")
        cached = column["K_death"]
        found = None if death is None else death["K"]
        if (cached is None) != (found is None) or (found is not None and abs(float(cached) - found) > 1e-6):
            raise RuntimeError(f"Death parse mismatch at p={column['p']} delta={column['delta']}: cache {cached}, parsed {found}")
        leave = next((row for row in rows if row["state"] != "Incoherence"), None)
        cached_leave = column["K_leave_incoherence"]
        found_leave = None if leave is None else leave["K"]
        if column["initial_state"] == "Incoherence" and cached_leave is None and found_leave is not None:
            raise RuntimeError(f"leave parse mismatch at p={column['p']} delta={column['delta']}")
        if cached_leave is not None and found_leave is not None and abs(float(cached_leave) - found_leave) > 1e-6:
            raise RuntimeError(f"leave parse mismatch at p={column['p']} delta={column['delta']}: cache {cached_leave}, parsed {found_leave}")


def main() -> None:
    context = load_cache()
    index = index_status()
    if index["status"] != context["audit"]["status"]:
        raise RuntimeError(f"index status {index['status']} does not match the label cache {context['audit']['status']}")
    probabilities = sorted({round(float(column["p"]), 5) for column in context["columns"]})
    if probabilities != [round(value, 5) for value in P_VALUES]:
        raise RuntimeError(f"unexpected probabilities {probabilities}")
    expanded = {}
    for column in context["columns"]:
        expanded[id(column)] = expand_column(column)
    parser_checks(context["columns"], expanded)
    deltas = sorted({float(column["delta"]) for column in context["columns"] if abs(column["p"] - 1.0) <= ABS_TOL})
    checks: list[dict] = []
    print("reproducing series and p=1 label exits", flush=True)
    for delta, expected in SERIES_ANCHORS.items():
        series = hopf_mod.series_eq7(delta)
        digits = 5 if delta in (0.05, 0.10, 0.15, 0.20) else 4
        ok = abs(series - expected) <= 5e-5 if digits == 5 else abs(round(series, 4) - expected) <= 1e-4
        check(checks, f"Eq. (7) series at delta={delta:.3f}", ok, f"computed {series:.5f}, published {expected}")
    p1 = [column for column in context["columns"] if abs(column["p"] - 1.0) <= ABS_TOL]
    counts = Counter(row["state"] for column in p1 for row in expanded[id(column)])
    check(checks, "p=1 completed profiles", sum(counts.values()) == 1741, f"found {sum(counts.values())}")
    for state, expected in P1_COUNTS.items():
        check(checks, f"p=1 {state} count", counts[state] == expected, f"found {counts[state]}, previous report {expected}")
    for column in p1:
        delta = float(column["delta"])
        rounded = round(delta, 5)
        if rounded not in {round(value, 5) for value in HOPF_EXITS}:
            continue
        summary = hopf_mod.transition_summary(expanded[id(column)], hopf_mod.series_eq7(delta))
        status, k_exit, state_after, inside = HOPF_EXITS[rounded]
        ok = summary["status"] == status
        if k_exit is not None:
            ok = ok and summary["K_exit"] is not None and abs(summary["K_exit"] - k_exit) <= ABS_TOL
            ok = ok and summary["state_after"] == state_after and bool(summary["inside_gap"]) is bool(inside)
        check(checks, f"p=1 Hopf label exit at delta={delta:.3f}", ok, f"status={summary['status']} K_exit={summary['K_exit']} after={summary['state_after']} inside={summary['inside_gap']}")
    if any(not item["ok"] for item in checks):
        REPORT_PATH.write_text(failure_report(checks, "p=1 label or Eq. (7) baseline"), encoding="utf-8")
        raise SystemExit(1)
    bundle = theory_bundle(deltas)
    reproduce_baselines(context["columns"], expanded, bundle, checks)
    # series and label checks were recorded twice; keep one copy
    unique = []
    seen = set()
    for item in checks:
        if item["name"] in seen:
            continue
        seen.add(item["name"])
        unique.append(item)
    checks = unique
    if any(not item["ok"] for item in checks):
        REPORT_PATH.write_text(failure_report(checks, "p=1 theory baseline"), encoding="utf-8")
        raise SystemExit(1)
    print("baseline passed; building scaled comparison", flush=True)
    path_index = {
        (quantize(row["p"]), quantize(row["delta"]), quantize(row["K"])): row["output_npz"]
        for row in index["paths"]
    }
    macro = macro_census(index["paths"])
    analyses = build_rows(context["columns"], expanded, bundle, path_index)
    window_fail = [item for item in analyses if item["windows"] is not None and item["windows"]["W3"] != N_EXPECTED]
    if window_fail:
        check(checks, "W3 Death count", False, f"{len(window_fail)} Death cells are not entirely inside |rho|<0.001")
        REPORT_PATH.write_text(failure_report(checks, "Death label disagrees with W3"), encoding="utf-8")
        raise SystemExit(1)
    summary = summarize(analyses, bundle)
    snapshots = snapshot_contrast(analyses, path_index)
    plot_figure(analyses, bundle, summary)
    report = render_report(context, analyses, bundle, summary, checks, macro, index, snapshots)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"wrote {FIGURE_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


def failure_report(checks: list[dict], reason: str) -> str:
    lines = [
        "# Eqs. (6)–(9) ER scaling check stopped",
        "",
        f"The $p<1$ comparison was not run. Reason: {reason}.",
        "",
        "The $p=1$ baseline below did not match the published Eq. (7) or death-boundary notes, or the cached labels disagreed with the archives. No scaled boundary is interpreted from a failed baseline.",
        "",
        markdown_table(["Result", "Check", "Detail"], [["PASS" if item["ok"] else "FAIL", item["name"], item["detail"]] for item in checks]),
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
