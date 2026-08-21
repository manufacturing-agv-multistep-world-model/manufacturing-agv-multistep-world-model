from __future__ import annotations

import unittest

import numpy as np

from agv_case_env import AGV_A_Charge_Env
from physics_graph_world_model import estimate_action_physics
from run_experiments import heuristic_action


class AGVCapacityInvariantTests(unittest.TestCase):
    def make_env(
        self,
        seed: int = 42,
        capacity_mode: str = "baseline",
        charge_node_capacity: int = 2,
    ) -> AGV_A_Charge_Env:
        env = AGV_A_Charge_Env(
            agv_count=3,
            env_variant="full",
            reward_mode="hybrid",
            scenario="rush",
            dispatch_rule="nearest",
            capacity_mode=capacity_mode,
            max_steps=5000,
            max_released_jobs=None,
            seed=seed,
            charge_node_capacity=charge_node_capacity,
        )
        env.policy_variant = "full"
        env.reset(seed=seed)
        return env

    def assert_node_capacities(self, env: AGV_A_Charge_Env) -> None:
        occupancy = np.bincount(env.agv_positions, minlength=len(env.node_map))
        violations = [
            (env.nodes[node].name, int(occupancy[node]), env._node_capacity(node))
            for node in range(len(occupancy))
            if occupancy[node] > env._node_capacity(node)
        ]
        self.assertEqual([], violations)

    def test_single_capacity_head_on_swap_blocks_both_agvs(self) -> None:
        env = self.make_env(capacity_mode="stress")
        edge = next(
            item
            for item in env.edges
            if env._edge_capacity(item.edge_id) == 1
            and env._node_capacity(item.from_node) == 1
            and env._node_capacity(item.to_node) == 1
        )
        third_node = next(
            node
            for node in env.home_nodes
            if node not in {edge.from_node, edge.to_node}
        )
        env.agv_positions = [edge.from_node, edge.to_node, third_node]
        proposals = [edge.to_node, edge.from_node, third_node]
        edge_ids = [edge.edge_id, edge.edge_id, None]

        blocked, _ = env._detect_conflicts(proposals, edge_ids)

        self.assertTrue(blocked[0])
        self.assertTrue(blocked[1])

    def test_wide_endpoint_allows_safe_single_lane_yield(self) -> None:
        env = self.make_env(capacity_mode="baseline")
        edge = next(
            item
            for item in env.edges
            if env._edge_capacity(item.edge_id) == 1
            and max(env._node_capacity(item.from_node), env._node_capacity(item.to_node)) >= 2
        )
        wide_node = max(
            (edge.from_node, edge.to_node),
            key=env._node_capacity,
        )
        narrow_node = edge.to_node if wide_node == edge.from_node else edge.from_node
        third_node = next(
            node
            for node in env.home_nodes
            if node not in {wide_node, narrow_node}
        )
        env.agv_positions = [wide_node, narrow_node, third_node]
        proposals = [narrow_node, wide_node, third_node]
        edge_ids = [edge.edge_id, edge.edge_id, None]

        blocked, _ = env._detect_conflicts(proposals, edge_ids)
        effective = [
            env.agv_positions[i] if blocked[i] else proposals[i]
            for i in range(env.agv_count)
        ]
        occupancy = np.bincount(effective, minlength=len(env.node_map))

        self.assertEqual(1, int(blocked[0]) + int(blocked[1]))
        for node in {wide_node, narrow_node}:
            self.assertLessEqual(occupancy[node], env._node_capacity(node))

    def test_one_hour_rush_never_exceeds_node_capacity(self) -> None:
        env = self.make_env()
        while env.metrics.total_time_sec < 3600.0:
            env.step(heuristic_action(env))
            self.assert_node_capacities(env)

        summary = env.summary()
        self.assertLessEqual(summary["blocking_onset_count"], summary["blocked_agent_steps"])
        self.assertGreaterEqual(
            summary["blocked_time_sec"],
            summary["blocked_agent_steps"] * env.config.wait_time_s,
        )
        self.assertAlmostEqual(
            env.node_occupancy_time_sec.sum(),
            env.agv_count * env.metrics.total_time_sec,
            places=6,
        )
        self.assertAlmostEqual(
            summary["blocked_time_ratio"],
            summary["blocked_time_sec"] / (env.agv_count * summary["real_time_sec"]),
            places=9,
        )
        self.assertGreaterEqual(summary["blocked_time_ratio"], 0.0)
        self.assertLessEqual(summary["blocked_time_ratio"], 1.0)

    def test_full_charger_is_reported_as_charge_queue_not_route_blocking(self) -> None:
        env = self.make_env(charge_node_capacity=2)
        env.agv_positions = [env.CHARGE_NODE, env.CHARGE_NODE, env.node_by_name["A"]]
        env.agv_batteries = [40.0, 45.0, 10.0]

        env.step(np.asarray([0, 0, 3], dtype=np.int64))
        summary = env.summary()

        self.assertEqual(summary["charge_queue_onset_count"], 1.0)
        self.assertEqual(summary["charge_queue_blocked_agent_steps"], 1.0)
        self.assertEqual(summary["route_blocking_onset_count"], 0.0)
        self.assertEqual(summary["route_blocked_agent_steps"], 0.0)

    def test_poisson_model_starts_empty_with_positive_first_arrivals(self) -> None:
        env = self.make_env()

        self.assertEqual([], env.jobs)
        self.assertTrue(all(value > 0.0 for value in env.next_arrival_by_template.values()))

    def test_new_assignment_is_visible_before_its_first_executed_action(self) -> None:
        env = self.make_env()
        next_arrival = min(env.next_arrival_by_template.values())
        env.metrics.total_time_sec = next_arrival - 1.0
        positions_before = list(env.agv_positions)

        env.step(np.zeros(env.agv_count, dtype=np.int64))

        self.assertEqual(positions_before, env.agv_positions)
        self.assertTrue(any(job.status == "assigned" for job in env.jobs))
        self.assertTrue(any(job_id is not None for job_id in env.agv_job_ids))

    def test_charging_progress_is_not_counted_as_deadlock(self) -> None:
        env = self.make_env()
        env.metrics.total_time_sec = max(env.next_arrival_by_template.values()) + 1.0
        env._release_jobs()
        env.agv_positions = [env.CHARGE_NODE, env.CHARGE_NODE, env.node_by_name["A"]]
        env.agv_phases = ["to_charge", "to_charge", "to_charge"]
        env.agv_batteries = [15.0, 16.0, 15.0]

        for _ in range(env.config.deadlock_soft_steps + 1):
            env.step(np.asarray([0, 0, 3], dtype=np.int64))

        self.assertEqual(0, env.metrics.deadlock_count)
        self.assertGreater(env.agv_batteries[0], 15.0)
        self.assertGreater(env.agv_batteries[1], 16.0)

    def test_fidelity_ablation_does_not_leak_battery_awareness(self) -> None:
        env = self.make_env()
        env.fidelity_dispatch_mode = True
        env.agv_job_ids[0] = None
        env.agv_phases[0] = "idle"
        env.agv_batteries[0] = env.config.low_battery_soc - 1.0
        current = env.agv_positions[0]

        env.policy_variant = "kinematics"
        self.assertEqual(current, env._target_for_action(0, 0))

        env.policy_variant = "full"
        self.assertEqual(env.CHARGE_NODE, env._target_for_action(0, 0))

    def test_charging_and_idle_energy_use_global_physical_step_time(self) -> None:
        env = self.make_env()
        env.agv_positions = [env.CHARGE_NODE, env.node_by_name["Home1"], env.node_by_name["Home3"]]
        env.agv_batteries = [20.0, 100.0, 100.0]

        battery_before = env.agv_batteries[0]
        idle_energy_before = env.energy_by_agent[2]
        _, _, _, _, info = env.step(np.asarray([0, 3, 0], dtype=np.int64))
        step_time = float(info["step_time_sec"])

        self.assertGreater(step_time, env.config.wait_time_s)
        expected_charge_gain = env.config.charge_soc_per_min * step_time / 60.0
        expected_idle_soc_drop = 100.0 * env.config.idle_wh_per_s * step_time / env.config.battery_capacity_wh
        self.assertAlmostEqual(
            env.agv_batteries[0] - battery_before,
            expected_charge_gain - expected_idle_soc_drop,
            places=6,
        )
        self.assertAlmostEqual(
            env.energy_by_agent[2] - idle_energy_before,
            env.config.idle_wh_per_s * step_time,
            places=6,
        )
        self.assertAlmostEqual(
            env.node_occupancy_time_sec.sum(),
            env.agv_count * step_time,
            places=6,
        )

    def test_analytical_action_physics_matches_parallel_step_accounting(self) -> None:
        env = self.make_env()
        env.agv_positions = [env.CHARGE_NODE, env.node_by_name["Home1"], env.node_by_name["Home3"]]
        actions = np.asarray([0, 3, 0], dtype=np.int64)
        estimate = estimate_action_physics(env, actions)
        energy_before = env.metrics.total_energy_wh

        _, _, _, _, info = env.step(actions)

        self.assertAlmostEqual(estimate["time_sec"], info["step_time_sec"], places=6)
        self.assertAlmostEqual(
            estimate["energy_wh"],
            env.metrics.total_energy_wh - energy_before,
            places=6,
        )

    def test_charge_node_capacity_is_explicit_and_isolated(self) -> None:
        for capacity in (1, 2, 3):
            env = self.make_env(charge_node_capacity=capacity)
            self.assertEqual(capacity, env._node_capacity(env.CHARGE_NODE))
            self.assertEqual(2, env._node_capacity(env.node_by_name["P1_Packaging"]))

    def test_charge_node_capacity_must_be_positive_integer(self) -> None:
        for invalid in (0, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.make_env(charge_node_capacity=invalid)


if __name__ == "__main__":
    unittest.main()
