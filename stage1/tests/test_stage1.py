"""Lightweight checks for the Stage 1 scan. These do not run the production grid."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from stage1.benchmark import extrapolate, measure_representative_points, preset_report
from stage1.config import ScanConfig, load_config, validate_config
from stage1.grid import canonicalize_values, inclusive_range, quantize
from stage1.integrator import ContinuationError, advance_rk4, rotation_numbers
from stage1.model import (
    PROJECT_ROOT,
    make_adjacency,
    make_buffers,
    make_initial_phases,
    natural_frequencies,
    rk4_step_copy,
    sha256_array,
)
from stage1.query import load_run, query_runs
from stage1.scan import build_run_specs, continue_trajectory, execute_run, run_scan
from stage1.seeds import network_and_phase_seeds
from stage1.stats import local_slope_statistics, near_zero_fraction, sort_by_frequency
from stage1.storage import checkpoint_arrays, connect, insert_planned, run_paths, write_checkpoint


def tiny_config(directory: Path, **overrides) -> ScanConfig:
    config = ScanConfig(
        N=4,
        k_values=(0.5,),
        p_values=(0.5,),
        delta_values=(0.25,),
        n_trials=1,
        randomness_mode="paired",
        master_seed=11,
        solver="rk4",
        dt=0.25,
        t_start=0.0,
        t_end=1.0,
        observation_windows=((0.0, 0.5), (0.5, 1.0)),
        sample_order_parameter=False,
        checkpoint_interval=10.0,
        output_dir=directory,
        workers=1,
        stale_seconds=3600.0,
        allow_incomplete_restart=False,
    )
    if overrides:
        config = replace(config, **overrides)
    return validate_config(config)


class GridTests(unittest.TestCase):
    def test_dense_axes_hit_endpoints_without_duplicates(self):
        coupling = inclusive_range(0.0, 3.0, 0.01)
        delta = inclusive_range(0.05, 1.0, 0.05)
        self.assertEqual(len(coupling), 301)
        self.assertEqual(quantize(coupling[0]), quantize(0.0))
        self.assertEqual(quantize(coupling[-1]), quantize(3.0))
        self.assertEqual(quantize(coupling[200]), quantize(2.0))
        self.assertEqual(len({quantize(value) for value in coupling}), 301)
        self.assertEqual(len(delta), 20)
        self.assertEqual(quantize(delta[0]), quantize(0.05))
        self.assertEqual(quantize(delta[-1]), quantize(1.0))

    def test_range_rejects_a_step_that_misses_the_endpoint(self):
        with self.assertRaises(ValueError):
            inclusive_range(0.0, 1.0, 0.3)

    def test_explicit_values_reject_quantized_duplicates(self):
        with self.assertRaises(ValueError):
            canonicalize_values([0.1, 0.10])


class PairingTests(unittest.TestCase):
    def test_paired_seeds_are_shared_across_parameters_but_not_identical_inputs(self):
        config = tiny_config(
            Path("."),
            N=16,
            k_values=(0.25, 0.75),
            p_values=(0.2, 0.8),
            delta_values=(0.25, 0.75),
            output_dir=PROJECT_ROOT / "stage1_runs",
        )
        specs = build_run_specs(config)
        by_key = {(spec.K, spec.p, spec.Delta): spec for spec in specs}
        anchor = by_key[(0.25, 0.2, 0.25)]
        other_k = by_key[(0.75, 0.2, 0.25)]
        other_p = by_key[(0.25, 0.8, 0.25)]
        other_delta = by_key[(0.25, 0.2, 0.75)]
        self.assertEqual(anchor.network_seed, other_k.network_seed)
        self.assertEqual(anchor.initial_phase_seed, other_k.initial_phase_seed)
        self.assertEqual(anchor.network_seed, other_p.network_seed)
        self.assertIsNone(anchor.frequency_seed)
        self.assertNotEqual(anchor.run_id, other_k.run_id)
        self.assertEqual(anchor.run_id, build_run_specs(config)[0].run_id)

        phases_a = make_initial_phases(config.N, anchor.initial_phase_seed)
        phases_b = make_initial_phases(config.N, other_k.initial_phase_seed)
        network_low = make_adjacency(config.N, anchor.p, anchor.network_seed)
        network_high = make_adjacency(config.N, other_p.p, other_p.network_seed)
        omega_low = natural_frequencies(anchor.Delta, config.N)
        omega_high = natural_frequencies(other_delta.Delta, config.N)
        self.assertTrue(np.array_equal(phases_a, phases_b))
        self.assertTrue(np.all(network_low <= network_high))
        self.assertTrue(np.array_equal(network_low, network_low.T))
        self.assertGreater(float(network_high.sum()), float(network_low.sum()))
        quantiles = np.linspace(-1.0, 1.0, config.N)
        self.assertTrue(np.allclose(omega_low, 1.0 + anchor.Delta * quantiles, atol=1e-12))
        self.assertTrue(np.allclose(omega_high, 1.0 + other_delta.Delta * quantiles, atol=1e-12))

    def test_variability_modes_touch_one_source(self):
        directory = PROJECT_ROOT / "stage1_runs"
        network = tiny_config(directory, randomness_mode="network", n_trials=2)
        initial = tiny_config(directory, randomness_mode="initial_condition", n_trials=2)
        independent = tiny_config(
            directory,
            randomness_mode="independent",
            k_values=(0.25, 0.75),
        )
        network_specs = build_run_specs(network)
        initial_specs = build_run_specs(initial)
        independent_specs = build_run_specs(independent)
        self.assertNotEqual(network_specs[0].network_seed, network_specs[1].network_seed)
        self.assertEqual(network_specs[0].initial_phase_seed, network_specs[1].initial_phase_seed)
        self.assertEqual(initial_specs[0].network_seed, initial_specs[1].network_seed)
        self.assertNotEqual(initial_specs[0].initial_phase_seed, initial_specs[1].initial_phase_seed)
        self.assertNotEqual(independent_specs[0].network_seed, independent_specs[1].network_seed)
        self.assertNotEqual(independent_specs[0].initial_phase_seed, independent_specs[1].initial_phase_seed)
        self.assertEqual(len({spec.run_id for spec in network_specs}), 2)

    def test_seed_recipe_is_stable(self):
        first, _phase, rule, _phase_rule = network_and_phase_seeds(
            "paired", 5, 0, 0, 0, "0.5", "0.5", "0.25"
        )
        again, _, _, _ = network_and_phase_seeds("paired", 5, 0, 0, 0, "0.5", "0.5", "0.25")
        self.assertEqual(first, again)
        self.assertEqual(rule, "shared_upper_triangle_uniforms_thresholded_by_p")


class ModelConventionTests(unittest.TestCase):
    def test_adjacency_matches_one_upper_triangle_stream_and_p_equals_one(self):
        n = 6
        seed = 99
        adjacency = make_adjacency(n, 0.3, seed)
        rng = np.random.default_rng(seed)
        expected = np.zeros((n, n), dtype=np.float64)
        index = np.triu_indices(n, k=1)
        expected[index] = rng.random(index[0].size) < 0.3
        expected += expected.T
        self.assertTrue(np.array_equal(adjacency, expected))
        self.assertTrue(np.all(np.diag(adjacency) == 0))
        complete = make_adjacency(n, 1.0, seed)
        self.assertTrue(np.array_equal(complete, np.ones((n, n)) - np.eye(n)))

    def test_rhs_uses_k_over_n_without_self_coupling(self):
        import generate_raw_grid as legacy

        theta = np.array([0.3, 1.1, 2.0])
        omega = np.array([0.8, 1.0, 1.2])
        adjacency = np.ones((3, 3)) - np.eye(3)
        coupling = 0.7
        buffers = make_buffers(3)
        out = np.zeros(3)
        legacy.winfree_rhs(theta, omega, adjacency, coupling, 3, buffers, out)
        pulse = 1.0 + np.cos(theta)
        manual = omega - (coupling / 3.0) * np.sin(theta) * (adjacency @ pulse)
        rescaled = omega - (coupling / 2.0) * np.sin(theta) * (adjacency @ pulse)
        with_self = omega - (coupling / 3.0) * np.sin(theta) * ((adjacency + np.eye(3)) @ pulse)
        self.assertTrue(np.allclose(out, manual))
        self.assertFalse(np.allclose(out, rescaled))
        self.assertFalse(np.allclose(out, with_self))
        stepped = rk4_step_copy(theta, 0.25, omega, adjacency, coupling, buffers)
        self.assertEqual(stepped.shape, theta.shape)

    def test_production_graph_is_undirected(self):
        adjacency = make_adjacency(5, 0.4, 3)
        self.assertTrue(np.array_equal(adjacency, adjacency.T))
        directed = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.assertFalse(np.array_equal(directed, directed.T))


class IntegratorTests(unittest.TestCase):
    def test_zero_coupling_keeps_each_oscillator_with_its_frequency(self):
        with TemporaryDirectory() as temporary:
            spec = build_run_specs(tiny_config(Path(temporary), k_values=(0.0,)))[0]
        omega = np.array([0.2, 1.5, 0.4, 1.1])
        theta0 = np.zeros(4)
        adjacency = np.zeros((4, 4))
        state = advance_rk4(
            spec,
            theta0,
            0,
            omega,
            adjacency,
            {float(spec.t_origin).hex(): theta0.copy()},
            {},
            buffers=make_buffers(4),
        )
        rotation = rotation_numbers(state, spec.observation_windows)
        self.assertTrue(np.allclose(rotation, omega, atol=1e-10))
        self.assertGreater(rotation[0, 1], rotation[0, 2])

    def test_matches_legacy_rk4_capture(self):
        import generate_raw_grid as legacy

        with TemporaryDirectory() as temporary:
            spec = build_run_specs(tiny_config(Path(temporary)))[0]
        rng = np.random.default_rng(4)
        theta0 = rng.uniform(0.0, 2.0 * np.pi, spec.N)
        omega = natural_frequencies(spec.Delta, spec.N)
        adjacency = make_adjacency(spec.N, spec.p, 5)
        state = advance_rk4(
            spec,
            theta0,
            0,
            omega,
            adjacency,
            {0.0.hex(): theta0.copy()},
            {},
            buffers=make_buffers(spec.N),
        )
        legacy_t0, legacy_t = legacy.integrate_rk4_to_times(
            theta0,
            omega,
            adjacency,
            spec.K,
            spec.N,
            make_buffers(spec.N),
            t0=0.5,
            t_end=1.0,
            dt=spec.dt,
        )
        self.assertTrue(np.allclose(state.snapshot_at(0.5), legacy_t0, atol=1e-10))
        self.assertTrue(np.allclose(state.snapshot_at(1.0), legacy_t, atol=1e-10))
        rotation = rotation_numbers(state, ((0.5, 1.0),))
        self.assertTrue(np.allclose(rotation[0], (legacy_t - legacy_t0) / 0.5, atol=1e-10))

    def test_continuation_from_last_full_step_matches_uninterrupted_run(self):
        with TemporaryDirectory() as temporary:
            long_spec = build_run_specs(
                tiny_config(
                    Path(temporary),
                    t_end=1.0,
                    observation_windows=((0.0, 0.4), (0.4, 1.0)),
                )
            )[0]
        short_spec = replace(long_spec, t_end=0.4, observation_windows=((0.0, 0.4),))
        theta0 = make_initial_phases(long_spec.N, long_spec.initial_phase_seed)
        omega = natural_frequencies(long_spec.Delta, long_spec.N)
        adjacency = make_adjacency(long_spec.N, long_spec.p, long_spec.network_seed)
        origin = {0.0.hex(): theta0.copy()}
        direct = advance_rk4(
            long_spec, theta0, 0, omega, adjacency, dict(origin), {}, buffers=make_buffers(long_spec.N)
        )
        short = advance_rk4(
            short_spec, theta0, 0, omega, adjacency, dict(origin), {}, buffers=make_buffers(long_spec.N)
        )
        self.assertAlmostEqual(short.t_grid, 0.25)
        self.assertEqual(short.step_index, 1)
        kept = {
            key: value
            for key, value in short.snapshots.items()
            if float.fromhex(key) <= short.t_grid + 1e-8
        }
        continued = advance_rk4(
            long_spec,
            short.theta_grid,
            short.step_index,
            omega,
            adjacency,
            kept,
            {},
            buffers=make_buffers(long_spec.N),
        )
        self.assertTrue(np.allclose(continued.theta_final, direct.theta_final, atol=1e-10))
        self.assertTrue(
            np.allclose(
                rotation_numbers(continued, long_spec.observation_windows),
                rotation_numbers(direct, long_spec.observation_windows),
                atol=1e-10,
            )
        )

    def test_rk45_continuation_is_refused(self):
        with TemporaryDirectory() as temporary:
            config = tiny_config(
                Path(temporary),
                solver="rk45",
                k_values=(0.0,),
                t_end=0.5,
                observation_windows=((0.0, 0.5),),
            )
            summary = run_scan(config)
            self.assertEqual(summary["counts"].get("completed"), 1)
            rows = query_runs(output_dir=temporary)
            self.assertFalse(rows[0]["continuable"])
            with self.assertRaises(ContinuationError):
                continue_trajectory(
                    Path(temporary),
                    rows[0]["run_id"],
                    t_end=1.0,
                    extra_windows=[(0.5, 1.0)],
                )
            loaded = load_run(rows[0]["run_id"], output_dir=temporary)
            self.assertTrue(np.allclose(loaded["rotation_number"], loaded["omega"], atol=1e-4))
