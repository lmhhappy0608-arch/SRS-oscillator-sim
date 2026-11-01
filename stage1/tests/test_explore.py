"""Synthetic checks for descriptive exploration. The pilot grid is not loaded."""

from __future__ import annotations

import unittest

import numpy as np

from stage1.explore import (
    DIVERSITY_FEATURES,
    InventoryError,
    _add_neighbor_distances,
    assign_window_profiles,
    feature_row,
    rolling_slopes,
    select_cases,
)


class SlopeTests(unittest.TestCase):
    def test_linear_profile_keeps_identity_order_and_recovers_the_slope(self):
        omega = np.array([1.2, 0.2, 0.9, 1.7, 0.4, 1.5, 0.7, 1.1], dtype=np.float64)
        rho = -1.5 * omega + 0.25
        original_omega = omega.copy()
        original_rho = rho.copy()
        slopes = rolling_slopes(omega, rho, window=5)
        self.assertTrue(np.allclose(slopes, -1.5, atol=1e-8))
        self.assertTrue(np.array_equal(omega, original_omega))
        self.assertTrue(np.array_equal(rho, original_rho))

    def test_short_profile_has_no_slope_and_does_not_fill_zero(self):
        omega = np.linspace(0.2, 1.4, 6)
        slopes = rolling_slopes(omega, np.zeros_like(omega), window=11)
        self.assertEqual(slopes.size, 0)


class FeatureTests(unittest.TestCase):
    def test_windows_follow_bounds_and_differences_stay_undefined_when_absent(self):
        omega = np.linspace(0.5, 1.5, 40)
        rho = 0.3 * omega + 2.0
        later = rho + 0.02
        record = {
            "run_id": "abc",
            "omega": omega,
            "oscillator_index": np.arange(40, dtype=np.int32),
            "rotation_numbers": np.vstack([later, rho, later]),
            "window_start": np.array([12000.0, 8000.0, 12000.0]),
            "window_end": np.array([14000.0, 10000.0, 14000.0]),
            "parameters": {"K": 0.5, "p": 0.8, "Delta": 0.2, "N": 40},
            "realization": {
                "trial_id": 0,
                "network_seed": 7,
                "initial_phase_seed": 9,
                "frequency_seed": None,
                "network_realization_id": 0,
                "initial_condition_id": 0,
                "paired_group_id": "paired-net-0-ic-0",
            },
        }
        # The third stored row repeats w3 bounds, so assignment must fail.
        with self.assertRaises(InventoryError):
            assign_window_profiles(record)
        record["rotation_numbers"] = np.vstack([later, rho, rho + 0.02])
        record["window_start"] = np.array([12000.0, 8000.0, 10000.0])
        record["window_end"] = np.array([14000.0, 10000.0, 12000.0])
        # w2 and w3 were both later. Give w2 the +0.02 profile and w3 the same +0.02.
        record["rotation_numbers"] = np.vstack([rho + 0.02, rho, rho + 0.02])
        row = feature_row(record)
        self.assertAlmostEqual(row["slope11_abs_q50_w1"], 0.3, places=6)
        self.assertAlmostEqual(row["slope31_abs_q50_w3"], 0.3, places=6)
        self.assertEqual(row["zero_frac_eps0p01_w3"], 0.0)
        self.assertAlmostEqual(row["D13_median"], 0.02, places=8)
        self.assertEqual(row["D13_frac_gt_0p01"], 1.0)
        self.assertEqual(row["D13_frac_gt_0p05"], 0.0)
        self.assertAlmostEqual(row["delta_rho_mean_w3_minus_w1"], 0.02, places=8)
        self.assertTrue(np.isnan(row["slope_scale_gap_w3"]) or row["slope_scale_gap_w3"] < 1e-8)


class SelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_covers_strata(self):
        rows = []
        for p_value in (1.0, 0.5):
            for delta in (0.2, 0.8):
                for coupling in (0.0, 1.0, 2.0):
                    row = {name: 0.0 for name in DIVERSITY_FEATURES}
                    row.update(
                        {
                            "run_id": f"p{p_value}-d{delta}-k{coupling}",
                            "K": coupling,
                            "p": p_value,
                            "Delta": delta,
                            "trial_id": 0,
                            "network_seed": 1,
                            "initial_phase_seed": 2,
                            "D12_p95": 0.0,
                            "rho_std_w3": 0.0,
                        }
                    )
                    rows.append(row)
        outlier = next(row for row in rows if row["run_id"] == "p1.0-d0.2-k2.0")
        outlier["D13_p95"] = 10.0
        outlier["rho_std_w3"] = 10.0
        outlier["zero_frac_eps_sensitivity_w3"] = 0.4
        _add_neighbor_distances(rows)
        first_cases, first_reviews = select_cases(rows)
        second_cases, _second_reviews = select_cases(rows)
        self.assertEqual([case["run_id"] for case in first_cases], [case["run_id"] for case in second_cases])
        covered = {(case["p"], case["Delta"]) for case in first_cases}
        self.assertEqual(covered, {(1.0, 0.2), (1.0, 0.8), (0.5, 0.2), (0.5, 0.8)})
        selected = {case["run_id"] for case in first_cases}
        self.assertIn("p1.0-d0.2-k2.0", selected)
        self.assertTrue(first_reviews)
        self.assertNotIn("Death", first_cases[0]["selection_detail"])
        self.assertNotIn("Locking", first_reviews[0]["review_detail"])


if __name__ == "__main__":
    unittest.main()
