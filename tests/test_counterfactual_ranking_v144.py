from __future__ import annotations

import numpy as np

from diagnose_counterfactual_ranking_v144 import (
    _bootstrap,
    _episode_summary,
    _ranking_rows,
    _summarize_rows,
)


def _synthetic_samples():
    samples = []
    for episode in range(3):
        for state in range(4):
            for candidate in range(2):
                samples.append(
                    {
                        "episode_id": np.asarray(episode),
                        "state_id": np.asarray(state),
                        "candidate": candidate,
                    }
                )
    return samples


def test_perfect_ranking_has_zero_regret_and_full_reduction():
    samples = _synthetic_samples()
    target = np.zeros((len(samples), 3, 3), dtype=np.float32)
    mask = np.ones_like(target, dtype=bool)
    for index, sample in enumerate(samples):
        candidate = sample["candidate"]
        target[index, -1] = [0.5 - candidate, 0.5 + candidate, 0.2]
    scale = np.ones((3, 3), dtype=np.float32)
    rows = _ranking_rows(samples, target.copy(), target, mask, scale)
    summary = _summarize_rows(rows)
    assert summary["model_mean_regret"] == 0.0
    assert summary["regret_reduction"] == 1.0
    assert summary["top1_agreement"] == 1.0


def test_episode_bootstrap_preserves_complete_trajectory_groups():
    samples = _synthetic_samples()
    target = np.zeros((len(samples), 3, 3), dtype=np.float32)
    mask = np.ones_like(target, dtype=bool)
    for index, sample in enumerate(samples):
        candidate = sample["candidate"]
        target[index, -1] = [0.5 - candidate, 0.5 + candidate, 0.2]
    scale = np.ones((3, 3), dtype=np.float32)
    rows = _ranking_rows(samples, target.copy(), target, mask, scale)
    episodes = _episode_summary(rows)
    bootstrap = _bootstrap(rows, replicates=250, seed=15499)
    assert len(episodes) == 3
    assert all(row["decision_states"] == 4 for row in episodes)
    assert bootstrap["ci_low"] == 1.0
    assert bootstrap["ci_high"] == 1.0
