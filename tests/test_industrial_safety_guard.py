import unittest

import numpy as np

from industrial_safety_guard import apply_required_actions, safety_intervention_rate


class IndustrialSafetyGuardTests(unittest.TestCase):
    def test_constraint_occupancy_does_not_imply_intervention(self) -> None:
        proposed = np.array([0, 1, 3], dtype=np.int64)
        required = np.array([0, -1, 3], dtype=np.int64)

        self.assertEqual(safety_intervention_rate(proposed, required), 0.0)

    def test_intervention_rate_counts_only_changed_constrained_actions(self) -> None:
        proposed = np.array([1, 2, 0], dtype=np.int64)
        required = np.array([0, -1, 3], dtype=np.int64)

        self.assertTrue(np.isclose(safety_intervention_rate(proposed, required), 2.0 / 3.0))
        np.testing.assert_array_equal(
            apply_required_actions(proposed, required),
            np.array([0, 2, 3], dtype=np.int64),
        )


if __name__ == "__main__":
    unittest.main()
