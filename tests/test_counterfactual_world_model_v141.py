from __future__ import annotations

import numpy as np
import torch

from agv_case_env import AGV_A_Charge_Env
from counterfactual_rollout_v141 import rollout_fixed_policy
from physics_graph_world_model import WorldModelMetadata, baseline_dt_aware_action
from physics_graph_world_model_counterfactual_v141 import (
    PhysicsGraphWorldModelCounterfactualV141,
)
from physics_graph_world_model_multistep_v11 import (
    EDGE_PHYSICAL_NAMES,
    NODE_PHYSICAL_NAMES,
)
from train_counterfactual_world_model_v141 import _grouped_split


def _small_model() -> PhysicsGraphWorldModelCounterfactualV141:
    metadata = WorldModelMetadata(
        agv_count=3,
        node_count=5,
        agent_dim=10,
        node_dim=7,
        global_dim=10,
        action_dim=4,
        hidden_dim=16,
    )
    return PhysicsGraphWorldModelCounterfactualV141(
        metadata,
        torch.zeros(5, len(NODE_PHYSICAL_NAMES)),
        torch.zeros(5, 5, len(EDGE_PHYSICAL_NAMES)),
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(141)
    return {
        "agent_features": torch.rand(4, 3, 10, generator=generator),
        "node_features": torch.rand(4, 5, 7, generator=generator),
        "adjacency_matrix": torch.ones(4, 5, 5),
        "global_features": torch.rand(4, 10, generator=generator),
        "baseline_actions": torch.tensor(
            [[0, 1, 2], [1, 1, 0], [2, 3, 1], [0, 0, 0]]
        ),
        "candidate_actions": torch.tensor(
            [[1, 1, 2], [1, 0, 0], [2, 1, 1], [3, 0, 0]]
        ),
    }


def test_identical_actions_have_exactly_zero_effect_in_training_mode() -> None:
    model = _small_model().train()
    batch = _batch()
    batch["candidate_actions"] = batch["baseline_actions"].clone()
    prediction = model.forward_counterfactual(batch)["counterfactual_delta"]
    assert torch.equal(prediction, torch.zeros_like(prediction))


def test_swapping_action_pair_negates_prediction() -> None:
    model = _small_model().train()
    batch = _batch()
    forward = model.forward_counterfactual(batch)["counterfactual_delta"]
    swapped = dict(batch)
    swapped["candidate_actions"] = batch["baseline_actions"]
    swapped["baseline_actions"] = batch["candidate_actions"]
    reverse = model.forward_counterfactual(swapped)["counterfactual_delta"]
    torch.testing.assert_close(forward, -reverse, rtol=0.0, atol=0.0)


def test_grouped_split_never_leaks_an_episode() -> None:
    samples = [
        {"episode_id": np.asarray(episode, dtype=np.int64)}
        for episode in range(6)
        for _ in range(3)
    ]
    train, valid, train_ids, valid_ids = _grouped_split(samples, 0.25, 141)
    assert set(train_ids).isdisjoint(valid_ids)
    assert {int(row["episode_id"]) for row in train} == set(train_ids)
    assert {int(row["episode_id"]) for row in valid} == set(valid_ids)


def test_deep_copied_equal_branches_preserve_common_random_numbers() -> None:
    env = AGV_A_Charge_Env(seed=141, max_steps=200)
    env.reset(seed=141)
    action = baseline_dt_aware_action(env)
    first = rollout_fixed_policy(env, action, (30.0,), max_rollout_steps=50)
    second = rollout_fixed_policy(env, action, (30.0,), max_rollout_steps=50)
    np.testing.assert_array_equal(first["raw_outcomes"], second["raw_outcomes"])
    np.testing.assert_array_equal(first["elapsed_sec"], second["elapsed_sec"])
