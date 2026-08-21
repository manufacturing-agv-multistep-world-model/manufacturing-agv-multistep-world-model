from __future__ import annotations

import unittest

import numpy as np

from train_world_model import split_samples


def sample(episode_id: int, transition_id: int) -> dict[str, np.ndarray]:
    return {
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "transition_id": np.asarray(transition_id, dtype=np.int64),
    }


class WorldModelSplitTests(unittest.TestCase):
    def test_episode_groups_do_not_cross_train_validation_boundary(self) -> None:
        samples = [
            sample(episode_id, transition_id)
            for episode_id in range(10)
            for transition_id in range(5)
        ]

        train, valid = split_samples(samples, seed=42, train_ratio=0.8)
        train_ids = {int(item["episode_id"]) for item in train}
        valid_ids = {int(item["episode_id"]) for item in valid}

        self.assertTrue(train_ids)
        self.assertTrue(valid_ids)
        self.assertTrue(train_ids.isdisjoint(valid_ids))
        self.assertEqual(train_ids | valid_ids, set(range(10)))
        self.assertEqual(len(train) + len(valid), len(samples))

    def test_at_least_two_episodes_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two episodes"):
            split_samples([sample(0, 0), sample(0, 1)], seed=42)


if __name__ == "__main__":
    unittest.main()
