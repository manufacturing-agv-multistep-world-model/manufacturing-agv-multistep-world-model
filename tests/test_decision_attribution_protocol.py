import unittest

import numpy as np

from agv_case_env import AGV_A_Charge_Env
from run_experiments import heuristic_action, run_episode


class DecisionAttributionProtocolTests(unittest.TestCase):
    def test_arrival_signature_is_action_independent_at_shared_cutoff(self):
        cutoff_sec = 1800.0
        for seed in (73101, 73102, 73103):
            with self.subTest(seed=seed):
                first = AGV_A_Charge_Env(seed=seed, scenario="rush", capacity_mode="stress")
                second = AGV_A_Charge_Env(seed=seed, scenario="rush", capacity_mode="stress")
                first.reset(seed=seed)
                second.reset(seed=seed)
                while first.metrics.total_time_sec < cutoff_sec:
                    first.step(np.zeros(first.agv_count, dtype=np.int64))
                while second.metrics.total_time_sec < cutoff_sec:
                    second.step(heuristic_action(second))
                self.assertEqual(
                    first.arrival_trace_count(cutoff_sec),
                    second.arrival_trace_count(cutoff_sec),
                )
                self.assertEqual(
                    first.arrival_trace_signature(cutoff_sec),
                    second.arrival_trace_signature(cutoff_sec),
                )

    def test_runner_reports_timing_overshoot_and_paired_arrivals(self):
        spec = {
            "experiment": "test",
            "method": "DT-aware",
            "env_variant": "full",
            "reward_mode": "hybrid",
            "scenario": "rush",
            "dispatch_rule": "dt_aware",
            "capacity_mode": "stress",
            "agv_count": 3,
            "policy_override": "heuristic",
            "fixed_time_target_h": 1.0 / 60.0,
            "fixed_time_target_sec": 60.0,
            "max_released_jobs": None,
        }
        summary, trace, _ = run_episode(
            spec, episode_id=0, seed=73102, max_steps=100, policy="heuristic"
        )
        self.assertEqual(summary["fixed_time_reached"], 1.0)
        self.assertGreaterEqual(summary["fixed_time_overshoot_sec"], 0.0)
        self.assertEqual(len(summary["paired_arrival_signature"]), 64)
        self.assertIn("mean_decision_compute_sec", summary)
        self.assertTrue(all(row["decision_compute_sec"] >= 0.0 for row in trace))


if __name__ == "__main__":
    unittest.main()
