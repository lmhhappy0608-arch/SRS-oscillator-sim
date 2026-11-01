"""Pilot grid, inspection, and profile loading. No production scan is launched."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from stage1.config import load_config, validate_config
from stage1.grid import quantize
from stage1.inspect import audit_plan, compare_observation_windows, format_profile_report, plot_rotation_profile
from stage1.model import PROJECT_ROOT
from stage1.query import load_profiles, query_runs
from stage1.scan import build_run_specs, run_scan
from stage1.tests.test_stage1 import tiny_config


class PilotConfigTests(unittest.TestCase):
    def test_pilot_grid_has_732_runs_and_reuses_the_recorded_point(self):
        config = load_config(PROJECT_ROOT / "stage1" / "configs" / "stage1_pilot.json")
        coupling = config.resolved_k()
        self.assertEqual(len(coupling), 61)
        self.assertEqual(quantize(coupling[0]), quantize(0.0))
        self.assertEqual(quantize(coupling[-1]), quantize(3.0))
        self.assertEqual(quantize(coupling[14]), quantize(0.7))
        self.assertEqual(config.resolved_p(), (1.0, 0.8, 0.4, 0.2))
        self.assertEqual([quantize(value) for value in config.resolved_delta()], [quantize(v) for v in (0.05, 0.2, 0.8)])
        self.assertEqual(config.n_trials, 1)
        self.assertEqual(config.N, 1000)
        self.assertEqual(config.t_end, 14000.0)
        self.assertEqual(config.observation_windows[0], (8000.0, 10000.0))
        self.assertEqual(config.observation_windows[-1], (12000.0, 14000.0))
        self.assertEqual(len(coupling) * len(config.resolved_p()) * len(config.resolved_delta()) * config.n_trials, 732)
        self.assertEqual(config.output_dir, PROJECT_ROOT / "stage1_runs")
        point = validate_config(
            replace(config, k_values=(0.70,), p_values=(0.80,), delta_values=(0.20,), n_trials=1)
        )
        spec = build_run_specs(point)[0]
        self.assertEqual(spec.trial_id, 0)
        self.assertEqual(spec.run_id, "3b702160612d668d83e8bf57a45dfebb")

    def test_refinement_reuses_trial_zero_and_adds_a_new_trial_id(self):
        pilot = load_config(PROJECT_ROOT / "stage1" / "configs" / "stage1_pilot.json")
        refine = load_config(PROJECT_ROOT / "stage1" / "configs" / "stage1_refine.example.json")
        self.assertEqual(refine.output_dir, pilot.output_dir)
        self.assertEqual(refine.master_seed, pilot.master_seed)
        self.assertEqual(quantize(refine.resolved_k()[0]), quantize(0.4))
        self.assertEqual(quantize(refine.resolved_k()[-1]), quantize(0.6))
        self.assertEqual(len(refine.resolved_k()), 41)
        self.assertEqual(refine.n_trials, 3)
        pilot_point = validate_config(
            replace(pilot, k_values=(0.5,), p_values=(0.8,), delta_values=(0.2,), n_trials=1)
        )
        refine_point = validate_config(
            replace(refine, k_values=(0.5,), p_values=(0.8,), delta_values=(0.2,), n_trials=2)
        )
        pilot_id = build_run_specs(pilot_point)[0].run_id
        refine_specs = build_run_specs(refine_point)
        by_trial = {spec.trial_id: spec.run_id for spec in refine_specs}
        self.assertEqual(by_trial[0], pilot_id)
        self.assertNotEqual(by_trial[1], by_trial[0])

    def test_inspect_does_not_create_an_index(self):
        from stage1.__main__ import main

        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "empty"
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = main(
                    [
                        "inspect",
                        "--config",
                        str(PROJECT_ROOT / "stage1" / "configs" / "stage1_pilot.json"),
                        "--output-dir",
                        str(directory),
                        "--require-complete",
                    ]
                )
            self.assertEqual(code, 1)
            text = buffer.getvalue()
            self.assertIn("planned=732", text)
            self.assertIn("absent=732", text)
            self.assertNotIn("Death", text)
            self.assertFalse(directory.exists())


class PilotDataTests(unittest.TestCase):
    def test_finer_grid_skips_completed_runs_and_profiles_stay_aligned(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            coarse = tiny_config(directory, k_values=(0.0, 0.5))
            first = run_scan(coarse)
            self.assertEqual(first["counts"].get("completed"), 2)
            finer = tiny_config(directory, k_values=(0.0, 0.25, 0.5), n_trials=2)
            before = audit_plan(finer, check_files=True)
            self.assertEqual(before["planned"], 6)
            self.assertEqual(before["counts"]["completed"], 2)
            self.assertEqual(before["counts"]["absent"], 4)
            self.assertEqual(before["files_ok"], 2)
            self.assertEqual(before["files_bad"], 0)
            second = run_scan(finer)
            self.assertEqual(second["counts"].get("skipped"), 2)
            self.assertEqual(second["counts"].get("completed"), 4)
            after = audit_plan(finer, check_files=True)
            self.assertTrue(after["complete"])
            self.assertEqual(after["files_ok"], 6)

            rows = query_runs(output_dir=directory)
            ordered_ids = [rows[1]["run_id"], rows[0]["run_id"]]
            records = load_profiles(ordered_ids, output_dir=directory)
            self.assertEqual([record["run_id"] for record in records], ordered_ids)
            record = records[0]
            self.assertTrue(np.array_equal(record["oscillator_index"], np.arange(record["parameters"]["N"])))
            self.assertEqual(record["rotation_numbers"].shape[1], record["omega"].shape[0])
            self.assertEqual(set(record["parameters"]), {"K", "p", "Delta", "N"})
            self.assertIn("network_seed", record["realization"])
            self.assertIn("initial_phase_seed", record["realization"])
            self.assertEqual(record["realization"]["randomness_mode"], "paired")
            comparisons = compare_observation_windows(record)
            self.assertEqual(len(comparisons), 1)
            self.assertEqual(comparisons[0]["interpretation"], "descriptive")
            report = format_profile_report(record)
            self.assertNotIn("Death", report)
            self.assertNotIn("Locking", report)
            figure = directory / "rho.png"
            plot_rotation_profile(record, figure)
            self.assertGreater(figure.stat().st_size, 0)

            saved = Path(rows[0]["output_npz"])
            saved.unlink()
            broken = audit_plan(coarse, check_files=True)
            self.assertEqual(broken["files_bad"], 1)
            self.assertFalse(broken["complete"])

    def test_inspect_command_prints_a_profile_and_writes_a_plot(self):
        from stage1.__main__ import main

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = {
                "N": 4,
                "K": {"values": [0.0]},
                "p": [0.5],
                "Delta": {"values": [0.25]},
                "n_trials": 1,
                "randomness_mode": "paired",
                "master_seed": 11,
                "solver": "rk4",
                "dt": 0.25,
                "t_start": 0.0,
                "t_end": 1.0,
                "observation_windows": [[0.0, 0.5], [0.5, 1.0]],
                "checkpoint_interval": 10.0,
                "output_dir": str(directory),
                "workers": 1,
                "frequency": {"mode": "deterministic_linspace"},
            }
            config_path = directory / "tiny.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            run_scan(load_config(config_path))
            figure = directory / "rho.png"
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = main(
                    [
                        "inspect",
                        "--config",
                        str(config_path),
                        "--K",
                        "0",
                        "--p",
                        "0.5",
                        "--Delta",
                        "0.25",
                        "--trial",
                        "0",
                        "--check-files",
                        "--plot",
                        str(figure),
                    ]
                )
            self.assertEqual(code, 0)
            text = buffer.getvalue()
            self.assertIn("planned=1", text)
            self.assertIn("completed=1", text)
            self.assertIn("files_ok=1", text)
            self.assertIn("rotation_numbers=(2, 4)", text)
            self.assertIn("max_abs=", text)
            self.assertTrue(figure.is_file())


if __name__ == "__main__":
    unittest.main()
