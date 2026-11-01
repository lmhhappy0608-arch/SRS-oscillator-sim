"""Command line for the Stage 1 scan, continuation, query, and benchmark.

The benchmark subcommand estimates cost. It does not launch the production grid.
``scan`` does launch the configured grid; pass a subgrid before a full run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from stage1.benchmark import PRESET_NAMES, extrapolate, load_measurements, measure_representative_points, preset_report
from stage1.config import load_config, validate_config
from stage1.integrator import ContinuationError
from stage1.model import PROJECT_ROOT
from stage1.inspect import (
    audit_plan,
    format_audit,
    format_profile_report,
    plot_rotation_profile,
    selected_rows,
)
from stage1.query import load_profiles, load_run, query_runs
from stage1.scan import continue_trajectory, run_scan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stage1")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Run or resume the configured parameter scan")
    scan.add_argument("--config", required=True, type=Path)
    scan.add_argument("--output-dir", type=Path)
    scan.add_argument("--workers", type=int)
    scan.add_argument("--k-min", type=float)
    scan.add_argument("--k-max", type=float)
    scan.add_argument("--k-step", type=float)
    scan.add_argument("--k-values", type=float, nargs="+")
    scan.add_argument("--p-values", type=float, nargs="+")
    scan.add_argument("--delta-min", type=float)
    scan.add_argument("--delta-max", type=float)
    scan.add_argument("--delta-step", type=float)
    scan.add_argument("--delta-values", type=float, nargs="+")
    scan.add_argument("--trials", type=int)
    scan.add_argument(
        "--mode",
        choices=("paired", "network", "initial_condition", "independent"),
    )
    scan.add_argument("--t-end", type=float)
    scan.add_argument("--dt", type=float)
    scan.add_argument("--solver", choices=("rk4", "rk45"))
    scan.add_argument("--master-seed", type=int)
    scan.add_argument("--stale-seconds", type=float)
    scan.add_argument("--allow-incomplete-restart", action="store_true")

    cont = sub.add_parser("continue", help="Extend a completed RK4 trajectory")
    cont.add_argument("--output-dir", type=Path)
    cont.add_argument("--run-id", required=True)
    cont.add_argument("--t-end", type=float)
    cont.add_argument("--window", nargs=2, type=float, action="append", required=True)
    cont.add_argument("--stale-seconds", type=float, default=3600.0)
    cont.add_argument("--allow-incomplete-restart", action="store_true")

    bench = sub.add_parser("benchmark", help="Count or extrapolate runtime; measure only with --measure")
    bench.add_argument("--preset", choices=(*PRESET_NAMES, "all"), default="all")
    bench.add_argument("--measurements", type=Path)
    bench.add_argument("--measure", action="store_true")
    bench.add_argument("--measure-output", type=Path)
    bench.add_argument("--N", type=int, default=1000)
    bench.add_argument("--horizons", type=float, nargs=2, default=(20.0, 40.0))
    bench.add_argument("--workers", type=int, default=1)
    bench.add_argument("--efficiency", type=float)

    query = sub.add_parser("query", help="Query the run index or load one run")
    query.add_argument("--output-dir", type=Path)
    query.add_argument("--K", type=float)
    query.add_argument("--p", type=float)
    query.add_argument("--Delta", type=float)
    query.add_argument("--N", type=int)
    query.add_argument("--run-id")
    query.add_argument("--status")

    inspect = sub.add_parser("inspect", help="Check a planned grid, load profiles, or plot rho(omega)")
    inspect.add_argument("--config", required=True, type=Path)
    inspect.add_argument("--output-dir", type=Path)
    inspect.add_argument("--K", type=float)
    inspect.add_argument("--p", type=float)
    inspect.add_argument("--Delta", type=float)
    inspect.add_argument("--trial", type=int)
    inspect.add_argument("--run-id")
    inspect.add_argument("--plot", type=Path)
    inspect.add_argument("--check-files", action="store_true")
    inspect.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit 1 unless every planned run is completed and its files match the plan",
    )

    args = parser.parse_args(argv)
    if args.command == "scan":
        return _scan(args)
    if args.command == "continue":
        return _continue(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "query":
        return _query(args)
    if args.command == "inspect":
        return _inspect(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _scan(args: argparse.Namespace) -> int:
    config = _apply_scan_overrides(load_config(args.config), args)
    summary = run_scan(config)
    failed = int(summary["counts"].get("failed", 0)) + int(summary["counts"].get("blocked", 0))
    return 1 if failed else 0


def _apply_scan_overrides(config, args):
    if args.k_values is not None and (
        args.k_min is not None or args.k_max is not None or args.k_step is not None
    ):
        raise SystemExit("pass either --k-values or --k-min/--k-max/--k-step")
    if args.delta_values is not None and (
        args.delta_min is not None or args.delta_max is not None or args.delta_step is not None
    ):
        raise SystemExit("pass either --delta-values or --delta-min/--delta-max/--delta-step")
    updates = {}
    if args.output_dir is not None:
        path = Path(args.output_dir)
        updates["output_dir"] = path if path.is_absolute() else PROJECT_ROOT / path
    if args.workers is not None:
        updates["workers"] = args.workers
    if args.k_values is not None:
        updates["k_values"] = tuple(args.k_values)
    elif args.k_min is not None or args.k_max is not None or args.k_step is not None:
        updates["k_values"] = None
        updates["k_min"] = config.k_min if args.k_min is None else args.k_min
        updates["k_max"] = config.k_max if args.k_max is None else args.k_max
        updates["k_step"] = config.k_step if args.k_step is None else args.k_step
    if args.p_values is not None:
        updates["p_values"] = tuple(args.p_values)
    if args.delta_values is not None:
        updates["delta_values"] = tuple(args.delta_values)
    elif args.delta_min is not None or args.delta_max is not None or args.delta_step is not None:
        updates["delta_values"] = None
        updates["delta_min"] = config.delta_min if args.delta_min is None else args.delta_min
        updates["delta_max"] = config.delta_max if args.delta_max is None else args.delta_max
        updates["delta_step"] = config.delta_step if args.delta_step is None else args.delta_step
    if args.trials is not None:
        updates["n_trials"] = args.trials
    if args.mode is not None:
        updates["randomness_mode"] = args.mode
    if args.t_end is not None:
        updates["t_end"] = args.t_end
    if args.dt is not None:
        updates["dt"] = args.dt
    if args.solver is not None:
        updates["solver"] = args.solver
    if args.master_seed is not None:
        updates["master_seed"] = args.master_seed
    if args.stale_seconds is not None:
        updates["stale_seconds"] = args.stale_seconds
    if args.allow_incomplete_restart:
        updates["allow_incomplete_restart"] = True
    return validate_config(replace(config, **updates))


def _continue(args: argparse.Namespace) -> int:
    output = args.output_dir
    if output is not None and not Path(output).is_absolute():
        output = PROJECT_ROOT / output
    try:
        result = continue_trajectory(
            PROJECT_ROOT / "stage1_runs" if output is None else output,
            args.run_id,
            t_end=args.t_end,
            extra_windows=[(float(start), float(end)) for start, end in args.window],
            allow_incomplete_restart=bool(args.allow_incomplete_restart),
            stale_seconds=float(args.stale_seconds),
        )
    except ContinuationError as exc:
        print(f"continuation refused: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={result['status']} run_id={result['run_id']} "
        f"K={result['K']} p={result['p']} Delta={result['Delta']} trial={result['trial_id']}"
    )
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    measurements = None
    if args.measure:
        measurements = measure_representative_points(
            horizons=(float(args.horizons[0]), float(args.horizons[1])),
            N=int(args.N),
            output_path=args.measure_output,
        )
        print("measured representative points; these timings are not the full grid")
    elif args.measurements is not None:
        measurements = load_measurements(args.measurements)
    names = PRESET_NAMES if args.preset == "all" else (args.preset,)
    if measurements is None and args.preset == "all":
        document = preset_report()
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    reports = [
        extrapolate(name, measurements, workers=args.workers, efficiency=args.efficiency)
        for name in names
    ]
    print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2, sort_keys=True))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.output_dir is not None:
        path = Path(args.output_dir)
        output = path if path.is_absolute() else PROJECT_ROOT / path
        config = validate_config(replace(config, output_dir=output))
    parameter_flags = (args.K, args.p, args.Delta)
    if args.trial is not None and (any(value is None for value in parameter_flags) or args.run_id):
        raise SystemExit("pass --trial together with --K, --p, and --Delta")
    if any(value is not None for value in parameter_flags) and any(value is None for value in parameter_flags):
        raise SystemExit("pass --K, --p, and --Delta together")
    if args.plot is not None and args.run_id is None and args.K is None:
        raise SystemExit("pass --plot with --run-id or with --K, --p, and --Delta")
    check_files = bool(args.check_files or args.require_complete)
    report = audit_plan(config, check_files=check_files)
    print(format_audit(report))
    exit_code = 0
    if report["counts"].get("failed", 0) or report["files_bad"]:
        exit_code = 1
    if args.require_complete and not report["complete"]:
        exit_code = 1
    if args.run_id is None and args.K is None:
        return exit_code
    rows = selected_rows(
        config,
        K=args.K,
        p=args.p,
        Delta=args.Delta,
        trial_id=args.trial,
        run_id=args.run_id,
    )
    if not rows:
        print("0 runs")
        return 1
    for row in rows:
        if row["status"] != "completed":
            print(
                f"{row['status']} K={row['K']} p={row['p']} Delta={row['Delta']} "
                f"trial={row['trial_id']} run_id={row['run_id']}"
            )
    completed_ids = [row["run_id"] for row in rows if row["status"] == "completed"]
    records = load_profiles(completed_ids, output_dir=config.output_dir) if completed_ids else []
    for record in records:
        print(format_profile_report(record))
    if args.plot is None:
        return exit_code
    if len(records) != 1:
        print(
            f"plot needs exactly one completed run, found {len(records)}; "
            "pass --trial or --run-id"
        )
        return 1
    destination = plot_rotation_profile(records[0], args.plot)
    print(f"plot={destination}")
    return exit_code


def _query(args: argparse.Namespace) -> int:
    output = args.output_dir
    if output is not None and not Path(output).is_absolute():
        output = PROJECT_ROOT / output
    if args.run_id and args.K is None and args.p is None and args.Delta is None and args.status is None:
        loaded = load_run(args.run_id, output_dir=output)
        meta = loaded["metadata"]
        print(
            f"run_id={args.run_id} status={meta['status']} "
            f"omega={loaded['omega'].shape} rotation_numbers={loaded['rotation_numbers'].shape} "
            f"order_parameter={loaded['order_parameter'].shape}"
        )
        print(loaded["output_npz"])
        return 0
    rows = query_runs(
        K=args.K,
        p=args.p,
        Delta=args.Delta,
        N=args.N,
        run_id=args.run_id,
        status=args.status,
        output_dir=output,
    )
    print(f"{len(rows)} runs")
    for row in rows:
        print(
            f"{row['status']} K={row['K']} p={row['p']} Delta={row['Delta']} "
            f"trial={row['trial_id']} run_id={row['run_id']} path={row['output_npz']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
