from __future__ import annotations

import numpy as np

from diagnose_counterfactual_shadow_v145 import (
    FROZEN_UTILITY_MARGIN,
    _shadow_rows,
    _summary,
    _trajectory_bootstrap,
)


def test_unanimous_positive_margin_issues_beneficial_recommendation():
    samples = [
        {"episode_id": np.asarray(0), "state_id": np.asarray(0)},
        {"episode_id": np.asarray(0), "state_id": np.asarray(0)},
    ]
    target = np.zeros((2, 3, 3), dtype=np.float32)
    target[1, -1, 1] = 1.0
    mask = np.ones_like(target, dtype=bool)
    predictions = np.zeros((3, 2, 3, 3), dtype=np.float32)
    predictions[:, 1, -1, 1] = FROZEN_UTILITY_MARGIN + 0.1
    scales = np.ones((3, 3, 3), dtype=np.float32)
    rows = _shadow_rows(samples, predictions, target, mask, scales)
    summary = _summary(rows)
    assert summary["recommendations"] == 1
    assert summary["benefit_precision"] == 1.0
    assert summary["mean_true_gain"] == 1.0


def test_disagreement_abstains_and_bootstrap_uses_trajectories():
    samples = []
    target = []
    predictions = np.zeros((3, 6, 3, 3), dtype=np.float32)
    for episode in range(3):
        for candidate in range(2):
            samples.append(
                {"episode_id": np.asarray(episode), "state_id": np.asarray(0)}
            )
            value = np.zeros((3, 3), dtype=np.float32)
            if candidate == 1:
                value[-1, 1] = 1.0
            target.append(value)
    target_array = np.stack(target)
    mask = np.ones_like(target_array, dtype=bool)
    predictions[:, 1::2, -1, 1] = 0.5
    predictions[2, 1::2, -1, 1] = -0.5
    scales = np.ones((3, 3, 3), dtype=np.float32)
    rows = _shadow_rows(samples, predictions, target_array, mask, scales)
    assert _summary(rows)["recommendations"] == 0

    for row in rows:
        row["recommended"] = 1.0
        row["true_gain"] = 1.0
        row["beneficial"] = 1.0
    bootstrap = _trajectory_bootstrap(rows, replicates=250, seed=15699)
    assert bootstrap["mean_gain_ci95"] == [1.0, 1.0]
