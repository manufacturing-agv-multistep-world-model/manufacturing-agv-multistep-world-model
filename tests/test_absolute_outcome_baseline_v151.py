from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from compare_paired_vs_absolute_v151 import _exact_two_sided_sign_test
from physics_graph_world_model import WorldModelMetadata
from physics_graph_world_model_absolute_v151 import (
    PhysicsGraphWorldModelAbsoluteV151,
    absolute_outcome_loss_v151,
    absolute_outcome_target_statistics,
    load_absolute_model_v151,
    save_absolute_model_v151,
)
from physics_graph_world_model_multistep_v11 import (
    EDGE_PHYSICAL_NAMES,
    NODE_PHYSICAL_NAMES,
)


def _metadata() -> WorldModelMetadata:
    return WorldModelMetadata(
        agv_count=3,
        node_count=5,
        agent_dim=10,
        node_dim=7,
        global_dim=10,
        action_dim=4,
        hidden_dim=16,
    )


def _small_model() -> PhysicsGraphWorldModelAbsoluteV151:
    metadata = _metadata()
    return PhysicsGraphWorldModelAbsoluteV151(
        metadata,
        torch.zeros(5, len(NODE_PHYSICAL_NAMES)),
        torch.zeros(5, 5, len(EDGE_PHYSICAL_NAMES)),
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(151)
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
        "baseline_outcomes": torch.rand(4, 3, 3, generator=generator),
        "candidate_outcomes": torch.rand(4, 3, 3, generator=generator),
        "target_mask": torch.ones(4, 3, 3),
    }


def test_absolute_formulation_subtracts_its_two_branch_predictions() -> None:
    model = _small_model().eval()
    batch = _batch()
    absolute = model.forward_absolute_outcomes(batch)
    effect = model.forward_counterfactual(batch)["counterfactual_delta"]
    torch.testing.assert_close(
        effect,
        absolute["candidate_outcomes"] - absolute["baseline_outcomes"],
    )


def test_identical_actions_produce_zero_inferred_effect() -> None:
    model = _small_model().train()
    batch = _batch()
    batch["candidate_actions"] = batch["baseline_actions"].clone()
    effect = model.forward_counterfactual(batch)["counterfactual_delta"]
    assert torch.equal(effect, torch.zeros_like(effect))


def test_absolute_loss_is_finite_and_backpropagates() -> None:
    model = _small_model().train()
    batch = _batch()
    output = model.forward_absolute_outcomes(batch)
    loss, report = absolute_outcome_loss_v151(output, batch)
    assert torch.isfinite(loss)
    assert report["absolute_outcome_mae"] >= 0.0
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.counterfactual_value_head.parameters()
    )


def test_statistics_use_both_absolute_branches() -> None:
    sample = {
        "baseline_outcomes": np.asarray(
            [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 2.0, 4.0]],
            dtype=np.float32,
        ),
        "candidate_outcomes": np.asarray(
            [[2.0, 0.0, 0.0], [4.0, 2.0, 0.0], [6.0, 3.0, 8.0]],
            dtype=np.float32,
        ),
        "target_mask": np.ones((3, 3), dtype=np.float32),
    }
    scales, event_weights = absolute_outcome_target_statistics([sample])
    assert scales.shape == (3, 3)
    assert event_weights.shape == (3, 3)
    assert np.all(scales > 0.0)
    assert np.all(event_weights >= 1.0)


def test_exact_sign_test_matches_twelve_of_twelve_case() -> None:
    result = _exact_two_sided_sign_test([1.0] * 12)
    assert result["positive_episodes"] == 12
    assert result["nonzero_episodes"] == 12
    assert np.isclose(result["p_value"], 2.0 / 4096.0)


def test_checkpoint_round_trip_preserves_predictions(tmp_path: Path) -> None:
    model = _small_model().eval()
    batch = _batch()
    expected = model.forward_counterfactual(batch)["counterfactual_delta"].detach()
    checkpoint = tmp_path / "absolute.pt"
    save_absolute_model_v151(
        checkpoint,
        model,
        _metadata(),
        history=[],
        args={"test": True},
        initialization_checkpoint="test-v13.pt",
    )
    restored = load_absolute_model_v151(checkpoint).eval()
    actual = restored.forward_counterfactual(batch)["counterfactual_delta"].detach()
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)
