from __future__ import annotations

import numpy as np
import torch

from compare_counterfactual_architectures_v150 import (
    _episode_comparison,
    _paired_bootstrap,
    _paired_rows,
)
from flat_counterfactual_baseline_v150 import (
    EXPECTED_TRAINABLE_PARAMETERS,
    FlatCounterfactualBaselineV150,
)


def _model() -> FlatCounterfactualBaselineV150:
    return FlatCounterfactualBaselineV150(
        agv_count=3,
        node_count=20,
        agent_dim=10,
        node_dim=7,
        global_dim=10,
        action_dim=4,
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(150)
    return {
        "agent_features": torch.rand(4, 3, 10, generator=generator),
        "node_features": torch.rand(4, 20, 7, generator=generator),
        "global_features": torch.rand(4, 10, generator=generator),
        "adjacency_matrix": torch.zeros(4, 20, 20),
        "baseline_actions": torch.tensor(
            [[0, 1, 2], [1, 1, 0], [2, 3, 1], [0, 0, 0]]
        ),
        "candidate_actions": torch.tensor(
            [[1, 1, 2], [1, 0, 0], [2, 1, 1], [3, 0, 0]]
        ),
    }


def test_parameter_budget_exactly_matches_v141_counterfactual_head() -> None:
    model = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 56_457
    assert EXPECTED_TRAINABLE_PARAMETERS == 56_457


def test_identical_actions_are_zero_and_swapping_negates_prediction() -> None:
    model = _model().train()
    batch = _batch()
    forward = model.forward_counterfactual(batch)["counterfactual_delta"]
    swapped = dict(batch)
    swapped["candidate_actions"] = batch["baseline_actions"]
    swapped["baseline_actions"] = batch["candidate_actions"]
    reverse = model.forward_counterfactual(swapped)["counterfactual_delta"]
    torch.testing.assert_close(forward, -reverse, rtol=0.0, atol=0.0)
    identical = dict(batch)
    identical["candidate_actions"] = batch["baseline_actions"].clone()
    zero = model.forward_counterfactual(identical)["counterfactual_delta"]
    assert torch.equal(zero, torch.zeros_like(zero))


def test_adjacency_is_deliberately_unused() -> None:
    model = _model().eval()
    batch = _batch()
    first = model.forward_counterfactual(batch)["counterfactual_delta"]
    changed = dict(batch)
    changed["adjacency_matrix"] = torch.ones_like(batch["adjacency_matrix"])
    second = model.forward_counterfactual(changed)["counterfactual_delta"]
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_paired_bootstrap_preserves_trajectory_groups() -> None:
    graph_rows = []
    flat_rows = []
    for episode in range(3):
        for state in range(4):
            common = {
                "episode_id": float(episode),
                "state_id": float(state),
                "baseline_regret": 1.0,
                "top1_agreement": 1.0,
            }
            graph_rows.append({**common, "model_regret": 0.2})
            flat_rows.append({**common, "model_regret": 0.5})
    paired = _paired_rows(graph_rows, flat_rows)
    episodes = _episode_comparison(paired)
    bootstrap = _paired_bootstrap(paired, replicates=250, seed=17499)
    assert len(episodes) == 3
    assert all(row["decision_states"] == 4 for row in episodes)
    np.testing.assert_allclose(bootstrap["ci_low"], 0.3)
    np.testing.assert_allclose(bootstrap["ci_high"], 0.3)

