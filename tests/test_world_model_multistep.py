from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agv_case_env import AGV_A_Charge_Env
from agv_dt_env import Job
from jms_parameter_registry import MPC_UTILITY_WEIGHTS
from physics_graph_world_model import WorldModelMetadata
from physics_graph_world_model_multistep import (
    MODEL_VERSION,
    MultiStepPhysicsInformedMPCPolicy,
    PhysicsInformedGraphWorldModelMultiStep,
    PhysicsOnlyRiskPolicy,
    build_sequence_samples,
    analytical_charge_staggering_opportunity,
    candidate_joint_actions,
    classify_override_evidence,
    immediate_physical_risk,
    load_multistep_world_model_policy,
    multistep_world_model_loss,
    project_physical_kpis,
    save_multistep_world_model,
    stable_graph_attention,
)
from physics_graph_world_model_multistep_v10 import (
    KPI_COMPONENT_WEIGHTS,
    MODEL_VERSION as V10_MODEL_VERSION,
    PhysicsInformedGraphWorldModelMultiStepV10,
    load_multistep_world_model_policy_v10,
    multistep_world_model_loss_v10,
    save_multistep_world_model_v10,
)
from physics_graph_world_model_multistep_v11 import (
    EDGE_PHYSICAL_NAMES,
    MODEL_VERSION as V11_MODEL_VERSION,
    NODE_PHYSICAL_NAMES,
    PhysicsInformedGraphWorldModelMultiStepV11,
    load_multistep_world_model_policy_v11,
    multistep_world_model_loss_v11,
    save_multistep_world_model_v11,
)
from physics_graph_world_model_multistep_v12 import (
    PhysicsInformedGraphWorldModelMultiStepV12,
    multistep_world_model_loss_v12,
)
from physics_graph_world_model_multistep_v13 import (
    MODEL_VERSION as V13_MODEL_VERSION,
    PhysicsInformedGraphWorldModelMultiStepV13,
    annotate_future_congestion_risk,
    future_risk_positive_weights,
    load_multistep_world_model_policy_v13,
    multistep_world_model_loss_v13,
    save_multistep_world_model_v13,
)
from physics_graph_world_model_multistep_v14 import (
    MODEL_VERSION as V14_MODEL_VERSION,
    PhysicsInformedGraphWorldModelMultiStepV14,
    annotate_future_terminal_kpis,
    future_terminal_positive_weights,
    future_terminal_scales,
    load_multistep_world_model_v14,
    multistep_world_model_loss_v14,
    save_multistep_world_model_v14,
)
from train_world_model_multistep import split_v12_congestion_stratified


TEST_NODE_COUNT = 20


def transition(episode: int, step: int, done: bool = False):
    return {
        "episode_id": np.asarray(episode, dtype=np.int64),
        "transition_id": np.asarray(step, dtype=np.int64),
        "done": np.asarray(done, dtype=np.float32),
        "agent_features": np.full((3, 10), step, dtype=np.float32),
        "node_features": np.full((TEST_NODE_COUNT, 7), step, dtype=np.float32),
        "adjacency_matrix": np.eye(TEST_NODE_COUNT, dtype=np.float32),
        "global_features": np.full(10, step, dtype=np.float32),
        "actions": np.asarray([step % 4, 1, 0], dtype=np.int64),
        "next_agent_features": np.full((3, 10), step + 1, dtype=np.float32),
        "next_node_features": np.full((TEST_NODE_COUNT, 7), step + 1, dtype=np.float32),
        "next_global_features": np.full(10, step + 1, dtype=np.float32),
        "kpi": np.full(6, 0.01 * (step + 1), dtype=np.float32),
        "physics_kpi": np.full(6, 0.01 * (step + 1), dtype=np.float32),
        "congestion_kpi": np.asarray([0.0, step % 2], dtype=np.float32),
    }


class MultiStepWorldModelTests(unittest.TestCase):
    def setUp(self):
        self.metadata = WorldModelMetadata(
            agv_count=3,
            node_count=TEST_NODE_COUNT,
            agent_dim=10,
            node_dim=7,
            global_dim=10,
            hidden_dim=16,
        )
        self.node_physical = np.random.default_rng(1).random(
            (TEST_NODE_COUNT, len(NODE_PHYSICAL_NAMES)), dtype=np.float32
        )
        self.edge_physical = np.random.default_rng(2).random(
            (TEST_NODE_COUNT, TEST_NODE_COUNT, len(EDGE_PHYSICAL_NAMES)), dtype=np.float32
        )

    def test_windows_never_cross_episode_or_terminal_boundary(self):
        rows = [transition(0, step, done=step == 3) for step in range(4)]
        rows += [transition(1, step, done=step == 4) for step in range(5)]
        windows = build_sequence_samples(rows, horizon=3)
        self.assertEqual(len(windows), 5)
        for window in windows:
            self.assertEqual(window["actions"].shape, (3, 3))
            self.assertEqual(window["target_node_features"].shape, (3, TEST_NODE_COUNT, 7))
            self.assertEqual(window["target_congestion_kpi"].shape, (3, 2))

    def test_v12_episode_split_keeps_charge_events_in_train_and_validation(self):
        rows = [transition(episode, step) for episode in range(4) for step in range(3)]
        for row in rows:
            episode = int(row["episode_id"])
            if episode in {0, 1} and int(row["transition_id"]) == 1:
                row["congestion_kpi"][1] = 1.0
        train_rows, valid_rows = split_v12_congestion_stratified(rows, seed=7)
        self.assertTrue(any(row["congestion_kpi"][1] > 0 for row in train_rows))
        self.assertTrue(any(row["congestion_kpi"][1] > 0 for row in valid_rows))

    def test_open_loop_rollout_shapes_and_gradients(self):
        model = PhysicsInformedGraphWorldModelMultiStep(self.metadata)
        batch = {
            "agent_features": torch.rand(2, 3, 10),
            "node_features": torch.rand(2, TEST_NODE_COUNT, 7),
            "adjacency_matrix": torch.eye(TEST_NODE_COUNT).repeat(2, 1, 1),
            "global_features": torch.rand(2, 10),
            "actions": torch.randint(0, 4, (2, 5, 3)),
            "target_agent_features": torch.rand(2, 5, 3, 10),
            "target_node_features": torch.rand(2, 5, TEST_NODE_COUNT, 7),
            "target_global_features": torch.rand(2, 5, 10),
            "target_kpi": torch.rand(2, 5, 6),
            "target_physics_kpi": torch.rand(2, 5, 6),
        }
        output = model.rollout(batch, teacher_forcing_ratio=0.0)
        self.assertEqual(output["pred_agent_features"].shape, (2, 5, 3, 10))
        self.assertEqual(output["pred_node_features"].shape, (2, 5, TEST_NODE_COUNT, 7))
        self.assertEqual(output["pred_global_features"].shape, (2, 5, 10))
        self.assertEqual(output["pred_kpi"].shape, (2, 5, 6))
        loss, _ = multistep_world_model_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_half_precision_attention_mask_does_not_overflow(self):
        adjacency = torch.zeros(1, TEST_NODE_COUNT, TEST_NODE_COUNT, dtype=torch.float16)
        for node in range(TEST_NODE_COUNT):
            adjacency[0, node, node] = 1.0
            if node + 1 < TEST_NODE_COUNT:
                adjacency[0, node, node + 1] = 1.0
                adjacency[0, node + 1, node] = 1.0
        query = torch.rand(1, TEST_NODE_COUNT, 16, dtype=torch.float16)
        key = torch.rand(1, TEST_NODE_COUNT, 16, dtype=torch.float16)
        node_tokens = torch.rand(1, TEST_NODE_COUNT, 16, dtype=torch.float16)
        output = stable_graph_attention(query, key, node_tokens, adjacency)
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())

    def test_physical_kpi_projection_enforces_event_support(self):
        raw = torch.tensor([[-2.0, -1.0, -3.0, -0.5, 2.0, 9.0]])
        projected = project_physical_kpis(raw, agv_count=3)
        self.assertEqual(float(projected[0, 0]), -2.0)
        torch.testing.assert_close(
            projected[0, 1:], torch.tensor([0.0, 0.0, 0.0, 1.0, 3.0])
        )
        self.assertEqual(MPC_UTILITY_WEIGHTS["model_risk_reduction_gate"], 0.5)

    def test_override_gate_requires_physical_safety_and_learned_evidence(self):
        safe = {"blocked_count": 0.0, "conflict_events": 0.0}
        immediate_conflict = {"blocked_count": 1.0, "conflict_events": 1.0}
        self.assertEqual(
            classify_override_evidence(safe, immediate_conflict, 2.0, 0.75),
            "reject_physical",
        )
        self.assertEqual(
            classify_override_evidence(safe, safe, 0.50, 0.75),
            "reject_insufficient_evidence",
        )
        self.assertEqual(
            classify_override_evidence(
                {**safe, "progress_m": 0.0},
                {**safe, "progress_m": 0.0},
                0.0,
                0.75,
                predicted_throughput_delta=0.0,
                operational_energy_action=True,
                predicted_charge_queue_reduction=0.75,
                charge_queue_gate_threshold=0.5,
                analytical_charge_staggering=True,
            ),
            "accept_charge_stagger",
        )
        self.assertEqual(
            classify_override_evidence(
                {**safe, "progress_m": 0.0},
                {**safe, "progress_m": 0.0},
                0.0,
                0.75,
                predicted_energy_reduction_wh=3.0,
                predicted_throughput_delta=0.0,
                energy_gate_threshold_wh=2.75,
                operational_energy_action=True,
                analytical_charge_staggering=True,
                dedicated_charge_gate_required=True,
            ),
            "reject_insufficient_evidence",
        )
        self.assertEqual(
            classify_override_evidence(
                safe,
                safe,
                0.75,
                0.75,
                analytical_future_risk=True,
            ),
            "accept_risk",
        )
        self.assertEqual(
            classify_override_evidence(safe, safe, 2.0, 0.75),
            "reject_insufficient_evidence",
        )
        self.assertEqual(
            classify_override_evidence(
                safe,
                safe,
                0.0,
                0.75,
                predicted_energy_reduction_wh=3.0,
                predicted_throughput_delta=0.0,
                predicted_time_increase_sec=0.0,
                energy_gate_threshold_wh=2.75,
            ),
            "accept_energy",
        )
        self.assertEqual(
            classify_override_evidence(
                safe,
                safe,
                0.0,
                0.75,
                predicted_energy_reduction_wh=3.0,
                predicted_throughput_delta=0.0,
                predicted_time_increase_sec=0.0,
                energy_gate_threshold_wh=2.75,
                operational_energy_action=False,
            ),
            "reject_insufficient_evidence",
        )
        self.assertEqual(
            classify_override_evidence(
                safe,
                safe,
                0.0,
                0.75,
                predicted_energy_reduction_wh=3.0,
                predicted_throughput_delta=-0.01,
                energy_gate_threshold_wh=2.75,
            ),
            "reject_insufficient_evidence",
        )
        self.assertEqual(
            immediate_physical_risk(
                {"blocked_count": 2.0, "conflict_events": 1.0}, agv_count=3
            ),
            5.0,
        )

    def test_candidate_actions_only_allow_charging_below_threshold(self):
        env = AGV_A_Charge_Env()
        env.reset(seed=17)
        env.agv_batteries = [90.0] * env.agv_count
        high_soc_candidates = candidate_joint_actions(env)
        self.assertTrue(
            all(np.all(candidate != 3) for candidate in high_soc_candidates),
            "High-SOC AGVs must not receive return-to-charge candidates.",
        )

        env.agv_batteries = [90.0] * env.agv_count
        env.agv_batteries[0] = 10.0
        low_soc_candidates = candidate_joint_actions(env)
        self.assertTrue(any(candidate[0] == 3 for candidate in low_soc_candidates))
        self.assertTrue(all(candidate[1] != 3 and candidate[2] != 3 for candidate in low_soc_candidates))

        env.agv_batteries = [27.0, 90.0, 90.0]
        proactive_candidates = candidate_joint_actions(
            env,
            allow_proactive_charge=True,
        )
        self.assertTrue(any(candidate[0] == 3 for candidate in proactive_candidates))

        env.agv_batteries = [27.0, 28.0, 29.0]
        proactive_candidates = candidate_joint_actions(env, allow_proactive_charge=True)
        self.assertFalse(any(np.all(candidate == 3) for candidate in proactive_candidates))

    def test_charge_staggering_requires_one_free_slot_and_multiple_near_threshold(self):
        env = AGV_A_Charge_Env()
        env.reset(seed=23)
        env.agv_batteries = [27.0, 28.0, 29.0]
        base = np.ones(env.agv_count, dtype=np.int64)
        planned = base.copy()
        planned[0] = 3
        opportunity, pressure, slots = analytical_charge_staggering_opportunity(
            env, base, planned
        )
        self.assertTrue(opportunity)
        self.assertEqual(pressure, 3)
        self.assertEqual(slots, 2)

        env.agv_positions[1] = env.CHARGE_NODE
        env.agv_positions[2] = env.CHARGE_NODE
        opportunity, _, slots = analytical_charge_staggering_opportunity(env, base, planned)
        self.assertFalse(opportunity)
        self.assertEqual(slots, 0)

    def test_physics_only_ablation_returns_joint_action(self):
        env = AGV_A_Charge_Env()
        env.reset(seed=31)
        policy = PhysicsOnlyRiskPolicy(env.agv_count)
        action = policy.predict_guarded(env)
        self.assertEqual(action.shape, (env.agv_count,))
        self.assertTrue(np.all((0 <= action) & (action < 4)))

    def test_rejected_world_model_action_uses_identical_physics_baseline(self):
        physics_env = AGV_A_Charge_Env()
        physics_env.reset(seed=37)
        model_env = AGV_A_Charge_Env()
        model_obs, _ = model_env.reset(seed=37)
        physics_action = PhysicsOnlyRiskPolicy(physics_env.agv_count).predict_guarded(
            physics_env
        )
        metadata = WorldModelMetadata(
            agv_count=model_env.agv_count,
            node_count=model_obs["node_features"].shape[0],
            agent_dim=model_obs["agent_features"].shape[1],
            node_dim=model_obs["node_features"].shape[1],
            global_dim=model_obs["global_features"].shape[0],
            hidden_dim=16,
        )
        model = PhysicsInformedGraphWorldModelMultiStep(metadata)
        policy = MultiStepPhysicsInformedMPCPolicy(
            model,
            planning_horizon=1,
            beam_width=4,
            risk_gate_threshold=1.0e9,
            energy_gate_threshold_wh=1.0e9,
        )
        model_action = policy.predict_guarded(model_env)
        np.testing.assert_array_equal(
            np.asarray(policy.last_plan["baseline_action"]), physics_action
        )
        np.testing.assert_array_equal(model_action, physics_action)
        self.assertFalse(policy.last_plan["override_accepted"])

    def test_candidate_actions_reject_zero_progress_when_safe_work_exists(self):
        env = AGV_A_Charge_Env()
        env.reset(seed=21)
        job = Job(
            job_id=0,
            template_id="test",
            origin=env.node_by_name["A"],
            destination=env.node_by_name["P1_Packaging"],
            task_class="test",
            load_kg=50.0,
            priority=1,
            release_time_sec=0.0,
            assigned_agv=0,
            status="assigned",
        )
        env.jobs = [job]
        env.agv_job_ids[0] = job.job_id
        env.agv_phases[0] = "to_origin"
        candidates = candidate_joint_actions(env)
        self.assertTrue(candidates)
        self.assertTrue(all(np.any(candidate != 0) for candidate in candidates))
        self.assertTrue(all(np.all(candidate != 2) for candidate in candidates))

    def test_checkpoint_round_trip_preserves_v9_identity(self):
        model = PhysicsInformedGraphWorldModelMultiStep(self.metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_multistep_world_model(
                path,
                model,
                self.metadata,
                history=[],
                args={"planning_horizon": 3, "beam_width": 4, "planning_discount": 0.95},
            )
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["model_version"], MODEL_VERSION)
            policy = load_multistep_world_model_policy(path)
            self.assertEqual(policy.planning_horizon, 3)
            self.assertEqual(policy.beam_width, 4)

    def test_v10_local_actions_change_agent_predictions(self):
        model = PhysicsInformedGraphWorldModelMultiStepV10(self.metadata)
        base = {
            "agent_features": torch.rand(1, 3, 10).repeat(2, 1, 1),
            "node_features": torch.rand(1, TEST_NODE_COUNT, 7).repeat(2, 1, 1),
            "adjacency_matrix": torch.eye(TEST_NODE_COUNT).repeat(2, 1, 1),
            "global_features": torch.rand(1, 10).repeat(2, 1),
            "actions": torch.tensor([[0, 1, 1], [3, 1, 1]]),
        }
        output = model.forward_step(base)
        self.assertFalse(
            torch.allclose(
                output["next_agent_features"][0, 0],
                output["next_agent_features"][1, 0],
            )
        )

    def test_v10_engineering_weights_prioritize_energy_over_reward(self):
        self.assertGreater(KPI_COMPONENT_WEIGHTS[2], KPI_COMPONENT_WEIGHTS[0])
        output = {
            "pred_agent_features": torch.zeros(1, 2, 3, 10),
            "pred_node_features": torch.zeros(1, 2, TEST_NODE_COUNT, 7),
            "pred_global_features": torch.zeros(1, 2, 10),
            "pred_kpi": torch.zeros(1, 2, 6),
        }
        batch = {
            "target_agent_features": torch.zeros_like(output["pred_agent_features"]),
            "target_node_features": torch.zeros_like(output["pred_node_features"]),
            "target_global_features": torch.zeros_like(output["pred_global_features"]),
            "target_kpi": torch.zeros_like(output["pred_kpi"]),
            "target_physics_kpi": torch.zeros_like(output["pred_kpi"]),
        }
        reward_batch = {key: value.clone() for key, value in batch.items()}
        reward_batch["target_kpi"][:, :, 0] = 1.0
        energy_batch = {key: value.clone() for key, value in batch.items()}
        energy_batch["target_kpi"][:, :, 2] = 1.0
        reward_loss, _ = multistep_world_model_loss_v10(output, reward_batch, physics_weight=0.0)
        energy_loss, _ = multistep_world_model_loss_v10(output, energy_batch, physics_weight=0.0)
        self.assertGreater(float(energy_loss), float(reward_loss))

    def test_checkpoint_round_trip_preserves_v10_identity(self):
        model = PhysicsInformedGraphWorldModelMultiStepV10(self.metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_multistep_world_model_v10(
                path,
                model,
                self.metadata,
                history=[],
                args={"planning_horizon": 3, "beam_width": 4, "planning_discount": 0.95},
            )
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["model_version"], V10_MODEL_VERSION)
            policy = load_multistep_world_model_policy_v10(path)
            self.assertEqual(policy.planning_horizon, 3)
            self.assertEqual(policy.beam_width, 4)

    def test_v11_rollout_uses_discrete_node_predictions(self):
        model = PhysicsInformedGraphWorldModelMultiStepV11(
            self.metadata, self.node_physical, self.edge_physical
        )
        batch = {
            "agent_features": torch.rand(2, 3, 10),
            "node_features": torch.rand(2, TEST_NODE_COUNT, 7),
            "adjacency_matrix": torch.ones(2, TEST_NODE_COUNT, TEST_NODE_COUNT),
            "global_features": torch.rand(2, 10),
            "actions": torch.randint(0, 4, (2, 5, 3)),
            "target_agent_features": torch.rand(2, 5, 3, 10),
            "target_node_features": torch.rand(2, 5, TEST_NODE_COUNT, 7),
            "target_global_features": torch.rand(2, 5, 10),
            "target_kpi": torch.rand(2, 5, 6),
            "target_physics_kpi": torch.rand(2, 5, 6),
        }
        output = model.rollout(batch, teacher_forcing_ratio=0.0)
        scaled_positions = output["pred_agent_features"][:, :, :, 0] * (TEST_NODE_COUNT - 1)
        self.assertTrue(torch.allclose(scaled_positions, scaled_positions.round(), atol=1.0e-5))
        self.assertEqual(output["pred_position_logits"].shape, (2, 5, 3, TEST_NODE_COUNT))
        loss, _ = multistep_world_model_loss_v11(output, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_checkpoint_round_trip_preserves_v11_physical_graph(self):
        model = PhysicsInformedGraphWorldModelMultiStepV11(
            self.metadata, self.node_physical, self.edge_physical
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_multistep_world_model_v11(
                path,
                model,
                self.metadata,
                history=[],
                args={"planning_horizon": 3, "beam_width": 4, "planning_discount": 0.95},
            )
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["model_version"], V11_MODEL_VERSION)
            np.testing.assert_allclose(
                checkpoint["edge_physical_features"].numpy(), self.edge_physical
            )
            policy = load_multistep_world_model_policy_v11(path, risk_gate_threshold=0.75)
            self.assertEqual(policy.planning_horizon, 3)
            self.assertEqual(policy.beam_width, 4)
            self.assertEqual(policy.risk_gate_threshold, 0.75)

    def test_v12_predicts_separate_route_and_charge_congestion(self):
        model = PhysicsInformedGraphWorldModelMultiStepV12(
            self.metadata, self.node_physical, self.edge_physical
        )
        batch = {
            "agent_features": torch.rand(2, 3, 10),
            "node_features": torch.rand(2, TEST_NODE_COUNT, 7),
            "adjacency_matrix": torch.ones(2, TEST_NODE_COUNT, TEST_NODE_COUNT),
            "global_features": torch.rand(2, 10),
            "actions": torch.randint(0, 4, (2, 5, 3)),
            "target_agent_features": torch.rand(2, 5, 3, 10),
            "target_node_features": torch.rand(2, 5, TEST_NODE_COUNT, 7),
            "target_global_features": torch.rand(2, 5, 10),
            "target_kpi": torch.rand(2, 5, 6),
            "target_physics_kpi": torch.rand(2, 5, 6),
            "target_congestion_kpi": torch.rand(2, 5, 2),
        }
        output = model.rollout(batch, teacher_forcing_ratio=0.0)
        self.assertEqual(output["pred_congestion_kpi"].shape, (2, 5, 2))
        loss, parts = multistep_world_model_loss_v12(output, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("congestion_loss", parts)
        loss.backward()

    def test_v13_future_labels_use_complete_within_episode_windows(self):
        rows = [transition(0, step) for step in range(7)]
        for row in rows:
            row["congestion_kpi"][:] = 0.0
        rows[3]["congestion_kpi"][1] = 1.0
        annotated = annotate_future_congestion_risk(rows, horizon=3)
        self.assertEqual(annotated[0]["future_congestion_risk"][0], 1.0)
        self.assertEqual(annotated[1]["future_congestion_risk"][0], 1.0)
        self.assertEqual(annotated[2]["future_congestion_risk"][0], 1.0)
        self.assertEqual(annotated[3]["future_congestion_risk_mask"][0], 0.0)
        np.testing.assert_array_equal(
            annotated[4]["future_congestion_risk_mask"], np.zeros(1)
        )
        np.testing.assert_array_equal(
            annotated[6]["future_congestion_risk_mask"], np.zeros(1)
        )
        sequences = build_sequence_samples(annotated, horizon=2)
        self.assertEqual(sequences[0]["target_future_congestion_risk"].shape, (2, 1))
        self.assertEqual(
            sequences[0]["target_future_congestion_risk_mask"].shape, (2, 1)
        )

    def test_v13_predicts_long_horizon_risk_with_masked_loss(self):
        model = PhysicsInformedGraphWorldModelMultiStepV13(
            self.metadata,
            self.node_physical,
            self.edge_physical,
            future_risk_pos_weight=np.asarray([4.0], dtype=np.float32),
            future_risk_horizon=80,
        )
        batch = {
            "agent_features": torch.rand(2, 3, 10),
            "node_features": torch.rand(2, TEST_NODE_COUNT, 7),
            "adjacency_matrix": torch.ones(2, TEST_NODE_COUNT, TEST_NODE_COUNT),
            "global_features": torch.rand(2, 10),
            "actions": torch.randint(0, 4, (2, 5, 3)),
            "target_agent_features": torch.rand(2, 5, 3, 10),
            "target_node_features": torch.rand(2, 5, TEST_NODE_COUNT, 7),
            "target_global_features": torch.rand(2, 5, 10),
            "target_kpi": torch.rand(2, 5, 6),
            "target_physics_kpi": torch.rand(2, 5, 6),
            "target_congestion_kpi": torch.rand(2, 5, 2),
            "target_future_congestion_risk": torch.randint(0, 2, (2, 5, 1)).float(),
            "target_future_congestion_risk_mask": torch.ones(2, 5, 1),
        }
        output = model.rollout(batch, teacher_forcing_ratio=0.0)
        self.assertEqual(
            output["pred_future_congestion_risk_logits"].shape, (2, 5, 1)
        )
        loss, parts = multistep_world_model_loss_v13(output, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("future_risk_loss", parts)
        loss.backward()

    def test_v13_checkpoint_preserves_forecast_window_and_weights(self):
        model = PhysicsInformedGraphWorldModelMultiStepV13(
            self.metadata,
            self.node_physical,
            self.edge_physical,
            future_risk_pos_weight=np.asarray([3.5], dtype=np.float32),
            future_risk_horizon=80,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_multistep_world_model_v13(
                path,
                model,
                self.metadata,
                history=[],
                args={"planning_horizon": 5, "beam_width": 8, "planning_discount": 0.95},
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_version"], V13_MODEL_VERSION)
            self.assertEqual(checkpoint["future_risk_horizon"], 80)
            policy = load_multistep_world_model_policy_v13(path)
            self.assertEqual(policy.model.future_risk_horizon, 80)
            np.testing.assert_allclose(
                policy.model.future_risk_pos_weight.detach().numpy(), [3.5]
            )

    def test_v13_positive_weights_are_derived_from_training_labels(self):
        rows = [transition(0, step) for step in range(8)]
        for row in rows:
            row["congestion_kpi"][:] = 0.0
        rows[3]["congestion_kpi"][1] = 1.0
        annotated = annotate_future_congestion_risk(rows, horizon=2)
        weights = future_risk_positive_weights(annotated)
        self.assertEqual(weights.shape, (1,))
        self.assertTrue(np.all(weights >= 1.0))

    def test_v14_terminal_labels_restore_physical_units_and_require_complete_windows(self):
        rows = [transition(0, step) for step in range(5)]
        for index, row in enumerate(rows):
            row["kpi"][:] = 0.0
            row["kpi"][2] = 0.5
            row["kpi"][5] = 1.0 if index == 1 else 0.0
            row["congestion_kpi"][:] = 0.0
            row["congestion_kpi"][1] = 1.0 if index == 2 else 0.0
        annotated = annotate_future_terminal_kpis(rows, horizon=3)
        np.testing.assert_allclose(annotated[0]["future_terminal_kpi"], [30.0, 1.0, 3.0])
        np.testing.assert_array_equal(
            annotated[0]["future_terminal_kpi_mask"], np.ones(3)
        )
        np.testing.assert_array_equal(
            annotated[3]["future_terminal_kpi_mask"], np.zeros(3)
        )
        sequences = build_sequence_samples(annotated, horizon=2)
        self.assertEqual(sequences[0]["target_future_terminal_kpi"].shape, (2, 3))
        self.assertEqual(
            sequences[0]["target_future_terminal_kpi_mask"].shape, (2, 3)
        )

    def test_v14_terminal_scales_use_only_supplied_training_rows(self):
        train_rows = [transition(0, step) for step in range(5)]
        valid_rows = [transition(1, step) for step in range(5)]
        for row in train_rows:
            row["kpi"][:] = 0.0
            row["kpi"][2] = 0.1
        for row in valid_rows:
            row["kpi"][:] = 0.0
            row["kpi"][2] = 10.0
        train_annotated = annotate_future_terminal_kpis(train_rows, horizon=2)
        valid_annotated = annotate_future_terminal_kpis(valid_rows, horizon=2)
        scales_before = future_terminal_scales(train_annotated)
        scales_after = future_terminal_scales(train_annotated + valid_annotated)
        self.assertLess(scales_before[0], scales_after[0])
        self.assertEqual(scales_before.shape, (3,))
        self.assertTrue(np.all(scales_before >= 1.0))

    def test_v14_sparse_charge_terminal_weight_is_training_derived_and_capped(self):
        rows = [transition(0, step) for step in range(30)]
        for row in rows:
            row["kpi"][:] = 0.0
            row["congestion_kpi"][:] = 0.0
        rows[10]["congestion_kpi"][1] = 1.0
        annotated = annotate_future_terminal_kpis(rows, horizon=3)
        weights = future_terminal_positive_weights(annotated, maximum=10.0)
        np.testing.assert_allclose(weights[:2], [1.0, 1.0])
        self.assertGreater(weights[2], 1.0)
        self.assertLessEqual(weights[2], 10.0)

    def test_v14_predicts_nonnegative_terminal_kpis_with_finite_loss(self):
        model = PhysicsInformedGraphWorldModelMultiStepV14(
            self.metadata,
            self.node_physical,
            self.edge_physical,
            future_risk_pos_weight=np.asarray([4.0], dtype=np.float32),
            future_risk_horizon=80,
            future_terminal_scale=np.asarray([100.0, 5.0, 20.0], dtype=np.float32),
            future_terminal_positive_weight=np.asarray(
                [1.0, 1.0, 8.0], dtype=np.float32
            ),
            future_terminal_horizon=80,
        )
        batch = {
            "agent_features": torch.rand(2, 3, 10),
            "node_features": torch.rand(2, TEST_NODE_COUNT, 7),
            "adjacency_matrix": torch.ones(2, TEST_NODE_COUNT, TEST_NODE_COUNT),
            "global_features": torch.rand(2, 10),
            "actions": torch.randint(0, 4, (2, 5, 3)),
            "target_agent_features": torch.rand(2, 5, 3, 10),
            "target_node_features": torch.rand(2, 5, TEST_NODE_COUNT, 7),
            "target_global_features": torch.rand(2, 5, 10),
            "target_kpi": torch.rand(2, 5, 6),
            "target_physics_kpi": torch.rand(2, 5, 6),
            "target_congestion_kpi": torch.rand(2, 5, 2),
            "target_future_congestion_risk": torch.randint(0, 2, (2, 5, 1)).float(),
            "target_future_congestion_risk_mask": torch.ones(2, 5, 1),
            "target_future_terminal_kpi": torch.rand(2, 5, 3) * 10.0,
            "target_future_terminal_kpi_mask": torch.ones(2, 5, 3),
        }
        output = model.rollout(batch, teacher_forcing_ratio=0.0)
        self.assertEqual(output["pred_future_terminal_kpi"].shape, (2, 5, 3))
        self.assertTrue(torch.all(output["pred_future_terminal_kpi"] >= 0.0))
        loss, parts = multistep_world_model_loss_v14(output, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("future_terminal_loss", parts)
        loss.backward()

    def test_v14_checkpoint_preserves_both_timescales(self):
        model = PhysicsInformedGraphWorldModelMultiStepV14(
            self.metadata,
            self.node_physical,
            self.edge_physical,
            future_risk_pos_weight=np.asarray([3.5], dtype=np.float32),
            future_risk_horizon=80,
            future_terminal_scale=np.asarray([120.0, 6.0, 24.0], dtype=np.float32),
            future_terminal_positive_weight=np.asarray(
                [1.0, 1.0, 7.0], dtype=np.float32
            ),
            future_terminal_horizon=80,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_multistep_world_model_v14(
                path,
                model,
                self.metadata,
                history=[],
                args={"planning_horizon": 5, "beam_width": 8},
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_version"], V14_MODEL_VERSION)
            loaded = load_multistep_world_model_v14(path)
            self.assertEqual(loaded.future_risk_horizon, 80)
            self.assertEqual(loaded.future_terminal_horizon, 80)
            np.testing.assert_allclose(
                loaded.future_terminal_scale.detach().numpy(), [120.0, 6.0, 24.0]
            )
            np.testing.assert_allclose(
                loaded.future_terminal_positive_weight.detach().numpy(),
                [1.0, 1.0, 7.0],
            )

    def test_risk_gate_rejects_negative_threshold(self):
        model = PhysicsInformedGraphWorldModelMultiStep(self.metadata)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            from physics_graph_world_model_multistep import MultiStepPhysicsInformedMPCPolicy

            MultiStepPhysicsInformedMPCPolicy(model, risk_gate_threshold=-0.1)


if __name__ == "__main__":
    unittest.main()
