from __future__ import annotations

import unittest

import numpy as np

from analyze_v14_dual_timescale_diagnostics import paired_metrics
from diagnose_world_model_multistep import (
    future_terminal_prediction_table,
    future_terminal_rows,
    spearman_correlation,
)


class V14DiagnosticTests(unittest.TestCase):
    def test_spearman_handles_tied_values(self):
        actual = np.asarray([1.0, 1.0, 2.0, 3.0])
        predicted = np.asarray([2.0, 2.0, 4.0, 6.0])
        self.assertAlmostEqual(spearman_correlation(actual, predicted), 1.0)

    def test_terminal_diagnostics_compare_direct_head_with_short_extrapolation(self):
        sample_count = 6
        rollout_horizon = 2
        actual = np.asarray(
            [[100.0 + 10.0 * index, 2.0 + index % 2, float(index % 2)] for index in range(sample_count)],
            dtype=np.float32,
        )
        data = {
            "episode_id": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
            "start_transition_id": np.arange(sample_count, dtype=np.int64),
            "target_future_terminal_kpi": np.repeat(actual[:, None, :], rollout_horizon, axis=1),
            "target_future_terminal_kpi_mask": np.ones((sample_count, rollout_horizon, 3), dtype=np.float32),
            "pred_future_terminal_kpi": np.repeat(actual[:, None, :], rollout_horizon, axis=1),
            "pred_kpi": np.zeros((sample_count, rollout_horizon, 6), dtype=np.float32),
            "pred_congestion_kpi": np.zeros((sample_count, rollout_horizon, 2), dtype=np.float32),
        }
        data["pred_kpi"][:, :, 2] = 0.5
        data["pred_kpi"][:, :, 5] = 0.5
        rows = future_terminal_rows(
            data,
            horizons=[1, 2],
            agv_count=3,
            forecast_window_steps=80,
            terminal_scales=np.asarray([200.0, 4.0, 2.0]),
        )
        self.assertEqual(len(rows), 15)
        predictions = future_terminal_prediction_table(
            data,
            horizons=[1, 2],
            agv_count=3,
            forecast_window_steps=80,
        )
        self.assertEqual(len(predictions), sample_count)
        metrics = paired_metrics(predictions)
        self.assertAlmostEqual(metrics["direct_eer_mae"], 0.0)
        self.assertLess(metrics["eer_mae_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
