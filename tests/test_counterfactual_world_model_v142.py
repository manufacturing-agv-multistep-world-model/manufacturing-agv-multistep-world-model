from __future__ import annotations

import numpy as np
import torch

from physics_graph_world_model import WorldModelMetadata
from physics_graph_world_model_counterfactual_v142 import (
    PhysicsGraphWorldModelCounterfactualV142,
    counterfactual_loss_v142,
)
from physics_graph_world_model_multistep_v11 import (
    EDGE_PHYSICAL_NAMES,
    NODE_PHYSICAL_NAMES,
)
from train_counterfactual_world_model_v142 import StateGroupedBatchSampler


def _model() -> PhysicsGraphWorldModelCounterfactualV142:
    metadata = WorldModelMetadata(3, 5, 10, 7, 10, 4, 16)
    return PhysicsGraphWorldModelCounterfactualV142(
        metadata,
        torch.zeros(5, len(NODE_PHYSICAL_NAMES)),
        torch.zeros(5, 5, len(EDGE_PHYSICAL_NAMES)),
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(142)
    return {
        "episode_id": torch.tensor([0, 0, 1, 1]),
        "state_id": torch.tensor([3, 3, 4, 4]),
        "agent_features": torch.rand(4, 3, 10, generator=generator),
        "node_features": torch.rand(4, 5, 7, generator=generator),
        "adjacency_matrix": torch.ones(4, 5, 5),
        "global_features": torch.rand(4, 10, generator=generator),
        "baseline_actions": torch.tensor(
            [[0, 1, 2], [0, 1, 2], [1, 0, 1], [1, 0, 1]]
        ),
        "candidate_actions": torch.tensor(
            [[1, 1, 2], [0, 0, 2], [1, 2, 1], [3, 0, 1]]
        ),
        "target_delta": torch.rand(4, 3, 3, generator=generator) - 0.5,
        "target_mask": torch.ones(4, 3, 3),
    }


def test_v142_same_action_is_exactly_zero_with_hard_and_soft_gate() -> None:
    model = _model().train()
    batch = _batch()
    batch["candidate_actions"] = batch["baseline_actions"].clone()
    output = model.forward_counterfactual(batch)
    assert torch.equal(
        output["counterfactual_delta"],
        torch.zeros_like(output["counterfactual_delta"]),
    )
    assert torch.equal(
        output["hard_counterfactual_delta"],
        torch.zeros_like(output["hard_counterfactual_delta"]),
    )


def test_v142_swap_negates_effect_and_preserves_gate_probability() -> None:
    model = _model().train()
    batch = _batch()
    forward = model.forward_counterfactual(batch)
    swapped = dict(batch)
    swapped["candidate_actions"] = batch["baseline_actions"]
    swapped["baseline_actions"] = batch["candidate_actions"]
    reverse = model.forward_counterfactual(swapped)
    torch.testing.assert_close(
        forward["counterfactual_delta"],
        -reverse["counterfactual_delta"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        forward["counterfactual_gate_probability"],
        reverse["counterfactual_gate_probability"],
        rtol=0.0,
        atol=0.0,
    )


def test_v142_loss_is_finite_and_backpropagates() -> None:
    model = _model().train()
    batch = _batch()
    loss, parts = counterfactual_loss_v142(
        model.forward_counterfactual(batch), batch
    )
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in parts.values())
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.counterfactual_gate_head.parameters()
    )


def test_grouped_sampler_never_splits_a_decision_state() -> None:
    samples = [
        {
            "episode_id": np.asarray(episode, dtype=np.int64),
            "state_id": np.asarray(state, dtype=np.int64),
        }
        for episode in range(2)
        for state in range(4)
        for _ in range(3)
    ]
    sampler = StateGroupedBatchSampler(samples, batch_size=7, seed=142, shuffle=True)
    seen = {}
    for batch_index, indices in enumerate(sampler):
        for index in indices:
            key = (int(samples[index]["episode_id"]), int(samples[index]["state_id"]))
            seen.setdefault(key, set()).add(batch_index)
    assert all(len(batch_ids) == 1 for batch_ids in seen.values())
