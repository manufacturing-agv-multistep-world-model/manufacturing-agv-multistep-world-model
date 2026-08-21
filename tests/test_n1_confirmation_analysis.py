import unittest

from analyze_n1_confirmation import aggregate_mean_relative_change, paired_bootstrap


class N1ConfirmationAnalysisTests(unittest.TestCase):
    def test_relative_bootstrap_preserves_paired_improvement(self):
        left = {seed: value * 0.95 for seed, value in enumerate((10.0, 20.0, 30.0), 1)}
        right = {seed: value for seed, value in enumerate((10.0, 20.0, 30.0), 1)}
        result = paired_bootstrap(left, right, 2000, 7, relative=True)
        self.assertAlmostEqual(result["delta_mean"], -0.05)
        self.assertLess(result["ci_high"], 0.0)

    def test_bootstrap_rejects_unpaired_seed_sets(self):
        with self.assertRaises(ValueError):
            paired_bootstrap({1: 1.0}, {2: 1.0}, 1000, 7, relative=False)

    def test_aggregate_effect_uses_ratio_of_method_means(self):
        left = {1: 44.0, 2: 46.0}
        right = {1: 45.0, 2: 47.0}
        self.assertAlmostEqual(
            aggregate_mean_relative_change(left, right), (45.0 - 46.0) / 46.0
        )


if __name__ == "__main__":
    unittest.main()
