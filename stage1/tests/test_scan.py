"""Scan, resume, query, and benchmark checks. No production grid is launched."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from stage1.benchmark import extrapolate, measure_representative_points, preset_report
from stage1.config import load_config
from stage1.integrator import ContinuationError
from stage1.model import PROJECT_ROOT, make_adjacency, make_initial_phases, natural_frequencies, sha256_array
from stage1.query import load_run, query_runs
from stage1.scan import build_run_specs, continue_trajectory, execute_run, run_scan
from stage1.stats import local_slope_statistics, near_zero_fraction, sort_by_frequency
from stage1.storage import checkpoint_arrays, connect, insert_planned, run_paths, write_checkpoint
from stage1.tests.test_stage1 import tiny_config


class ScanTests(unittest.TestCase):
    def test_metadata_query_resume_and_index_alignment(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = tiny_config(directory, k_values=(0.0, 0.5), master_seed=3)
            first = run_scan(config)
            self.assertEqual(first["counts"].get("completed"), 2)
            rows = query_runs(K=0.5, p=0.5, Delta=0.25, N=4, output_dir=directory)
            self.assertEqual(len(rows), 1)
            again = query_runs(K=0.50, p=0.50, Delta=0.250, N=4, output_dir=directory)
            self.assertEqual(again[0]["run_id"], rows[0]["run_id"])
            loaded = load_run(rows[0]["run_id"], output_dir=directory)
            self.assertEqual(loaded["omega"].shape, (4,))
            self.assertEqual(loaded["rotation_numbers"].shape, (2, 4))
            self.assertTrue(np.array_equal(loaded["oscillator_index"], np.arange(4)))
            metadata = loaded["metadata"]
            for key in (
                "network_seed",
                "initial_phase_seed",
                "frequency_seed",
                "frequency_rule",
                "network_realization_id",
                "initial_condition_id",
                "trial_id",
                "randomness_mode",
                "paired_group_id",
                "solver",
                "run_id",
                "lineage_id",
                "model_version",
            ):
                self.assertIn(key, metadata)
            self.assertIsNone(metadata["frequency_seed"])
            self.assertEqual(metadata["status"], "completed")
            path = Path(loaded["output_npz"])
            stamp = path.stat().st_mtime_ns
            second = run_scan(config)
            self.assertEqual(second["counts"].get("skipped"), 2)
            self.assertEqual(path.stat().st_mtime_ns, stamp)
            zero = load_run(query_runs(K=0.0, output_dir=directory)[0]["run_id"], output_dir=directory)
            self.assertTrue(np.allclose(zero["rotation_numbers"], zero["omega"], atol=1e-10))

    def test_interrupted_run_without_checkpoint_is_not_silently_restarted(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = tiny_config(directory, k_values=(0.2,))
            spec = build_run_specs(config)[0]
            connection = connect(directory)
            insert_planned(connection, spec, directory)
            connection.execute(
                "UPDATE runs SET status='running', heartbeat_unix=0, worker_pid=NULL WHERE run_id=?",
                (spec.run_id,),
            )
            connection.close()
            blocked = execute_run(spec, directory, stale_seconds=30.0, allow_incomplete_restart=False)
            self.assertEqual(blocked["status"], "blocked")
            self.assertIn("checkpoint", blocked["message"].lower())
            again = execute_run(spec, directory, stale_seconds=30.0, allow_incomplete_restart=False)
            self.assertEqual(again["status"], "blocked")
            self.assertFalse(Path(query_runs(output_dir=directory)[0]["output_npz"]).exists())
            finished = execute_run(spec, directory, stale_seconds=30.0, allow_incomplete_restart=True)
            self.assertEqual(finished["status"], "completed", finished["message"])

    def test_resume_uses_checkpoint_state_rather_than_a_new_draw(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = tiny_config(directory, k_values=(0.0,), observation_windows=((0.0, 1.0),))
            spec = build_run_specs(config)[0]
            omega = natural_frequencies(spec.Delta, spec.N)
            theta0 = make_initial_phases(spec.N, spec.initial_phase_seed)
            corrupted = theta0 + 1.0
            adjacency = make_adjacency(spec.N, spec.p, spec.network_seed)
            arrays = checkpoint_arrays(
                spec,
                theta_grid=corrupted,
                t_grid=0.0,
                step_index=0,
                omega=omega,
                snapshots={0.0.hex(): corrupted.copy()},
                order_parameter={},
                adjacency_sha256=sha256_array(adjacency),
                omega_sha256=sha256_array(omega),
                runtime_seconds=0.0,
            )
            write_checkpoint(run_paths(directory, spec.run_id)["checkpoint"], arrays)
            connection = connect(directory)
            insert_planned(connection, spec, directory)
            connection.close()
            result = execute_run(spec, directory, stale_seconds=30.0, allow_incomplete_restart=False)
            self.assertEqual(result["status"], "completed", result["message"])
            loaded = load_run(spec.run_id, output_dir=directory)
            expected = corrupted + omega * spec.t_end
            redrawn = theta0 + omega * spec.t_end
            self.assertTrue(np.allclose(loaded["theta_final"], expected, atol=1e-8))
            self.assertFalse(np.allclose(loaded["theta_final"], redrawn, atol=1e-6))
            self.assertTrue(np.allclose(loaded["rotation_number"], omega, atol=1e-8))

    def test_scan_continuation_matches_a_direct_long_run_and_refuses_missing_snapshots(self):
        with TemporaryDirectory() as temporary:
            short_dir = Path(temporary) / "short"
            long_dir = Path(temporary) / "long"
            short = tiny_config(short_dir, t_end=0.4, observation_windows=((0.0, 0.4),))
            long = tiny_config(
                long_dir,
                t_end=1.0,
                observation_windows=((0.0, 0.4), (0.4, 1.0)),
            )
            run_scan(short)
            run_scan(long)
            parent = query_runs(output_dir=short_dir)[0]
            parent_file = Path(parent["output_npz"])
            parent_bytes = parent_file.read_bytes()
            with self.assertRaises(ContinuationError):
                continue_trajectory(short_dir, parent["run_id"], extra_windows=[(0.1, 0.3)])
            self.assertEqual(parent_file.read_bytes(), parent_bytes)
            continued = continue_trajectory(
                short_dir,
                parent["run_id"],
                t_end=1.0,
                extra_windows=[(0.4, 1.0)],
            )
            self.assertEqual(continued["status"], "completed", continued["message"])
            self.assertEqual(parent_file.read_bytes(), parent_bytes)
            direct = load_run(query_runs(output_dir=long_dir)[0]["run_id"], output_dir=long_dir)
            child = load_run(continued["run_id"], output_dir=short_dir)
            self.assertEqual(child["run_id"], direct["run_id"])
            self.assertEqual(child["metadata"]["parent_run_id"], parent["run_id"])
            self.assertTrue(np.allclose(child["theta_final"], direct["theta_final"], atol=1e-10))
            self.assertTrue(np.allclose(child["rotation_numbers"], direct["rotation_numbers"], atol=1e-10))
            self.assertTrue(np.array_equal(child["oscillator_index"], direct["oscillator_index"]))

    def test_parallel_workers_match_a_serial_scan(self):
        with TemporaryDirectory() as temporary:
            serial_dir = Path(temporary) / "serial"
            parallel_dir = Path(temporary) / "parallel"
            common = dict(
                k_values=(0.25, 0.5, 0.75, 1.0),
                t_end=0.5,
                observation_windows=((0.0, 0.5),),
                dt=0.25,
            )
            run_scan(tiny_config(serial_dir, workers=1, **common))
            run_scan(tiny_config(parallel_dir, workers=2, **common))
            serial = {
                row["K"]: load_run(row["run_id"], output_dir=serial_dir)
                for row in query_runs(output_dir=serial_dir)
            }
            parallel = {
                row["K"]: load_run(row["run_id"], output_dir=parallel_dir)
                for row in query_runs(output_dir=parallel_dir)
            }
            self.assertEqual(set(serial), set(parallel))
            for coupling, loaded in serial.items():
                self.assertTrue(
                    np.allclose(loaded["rotation_numbers"], parallel[coupling]["rotation_numbers"])
                )


class StatsAndBenchmarkTests(unittest.TestCase):
    def test_descriptive_stats_keep_the_shared_permutation_and_the_tolerance(self):
        omega = np.array([3.0, 1.0, 2.0])
        rho = np.array([30.0, 10.0, 20.0])
        ordered = sort_by_frequency(omega, rho)
        self.assertTrue(np.array_equal(ordered["oscillator_index"], np.array([1, 2, 0])))
        self.assertTrue(np.array_equal(ordered["arrays"][0], np.array([10.0, 20.0, 30.0])))
        slopes = local_slope_statistics(omega, rho)
        self.assertEqual(int(slopes["oscillator_index_left"][0]), 1)
        self.assertEqual(int(slopes["oscillator_index_right"][0]), 2)
        self.assertAlmostEqual(float(slopes["slope"][0]), 10.0)
        near = near_zero_fraction(np.array([0.0, 0.02, 1.0]), 0.01)
        self.assertEqual(near["tolerance"], 0.01)
        self.assertAlmostEqual(near["value"], 1.0 / 3.0)
        self.assertNotIn("Death", json.dumps(near))
        self.assertEqual(near["interpretation"], "descriptive")

    def test_benchmark_keeps_measurements_and_extrapolation_apart(self):
        report = preset_report()
        dense = report["presets"]["dense"]
        extra = report["presets"]["dense_extra_trials"]
        moderate = report["presets"]["moderate"]
        self.assertEqual(dense["K_count"], 301)
        self.assertEqual(dense["p_count"], 11)
        self.assertEqual(dense["Delta_count"], 20)
        self.assertEqual(dense["run_count"], 301 * 11 * 20)
        self.assertEqual(extra["run_count"], 301 * 11 * 20 * 3)
        self.assertEqual(
            moderate["run_count"],
            moderate["K_count"] * moderate["p_count"] * moderate["Delta_count"],
        )
        self.assertEqual(dense["runtime"]["status"], "measurement_required")
        self.assertEqual(dense["disk_structural"]["label"], "structural_estimate_not_measured")
        missing = extrapolate("dense", None, workers=4)
        self.assertIsNone(missing["runtime"]["cpu_seconds"])
        self.assertEqual(missing["wall_time"]["efficiency_source"], "not_measured")
        self.assertIsNone(missing["wall_time"]["wall_seconds"])
        measured = measure_representative_points(
            horizons=(0.5, 1.0),
            N=4,
            dt=0.25,
            points=({"K": 0.4, "p": 0.5, "Delta": 0.3, "label": "tiny"},),
        )
        self.assertEqual(measured["kind"], "measured")
        self.assertEqual(measured["points"][0]["horizons"][0]["label"], "measured")
        estimated = extrapolate("moderate", measured, workers=2, efficiency=None)
        self.assertEqual(estimated["runtime"]["label"], "extrapolated")
        self.assertEqual(estimated["wall_time"]["status"], "efficiency_not_measured")
        scaled = extrapolate("moderate", measured, workers=2, efficiency=0.5)
        self.assertEqual(scaled["wall_time"]["label"], "extrapolated")
        self.assertEqual(scaled["wall_time"]["efficiency_source"], "user_supplied")
        self.assertAlmostEqual(
            scaled["wall_time"]["wall_seconds"],
            estimated["runtime"]["cpu_seconds_median"] / (2 * 0.5),
        )

    def test_default_config_and_benchmark_cli_do_not_launch_the_grid(self):
        config = load_config(PROJECT_ROOT / "stage1" / "configs" / "stage1_default.json")
        self.assertEqual(config.N, 1000)
        self.assertEqual(config.n_trials, 3)
        self.assertEqual(config.randomness_mode, "paired")
        self.assertEqual(config.solver, "rk4")
        self.assertEqual(len(config.resolved_k()), 301)
        self.assertEqual(len(config.resolved_p()), 11)
        self.assertEqual(len(config.resolved_delta()), 20)
        self.assertEqual(config.observation_windows[0], (8000.0, 10000.0))
        from stage1.__main__ import main

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["benchmark", "--preset", "dense"])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        self.assertIn("66220", text)
        self.assertIn("measurement_required", text)


if __name__ == "__main__":
    unittest.main()
