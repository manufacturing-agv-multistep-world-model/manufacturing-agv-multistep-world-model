import unittest

import numpy as np

from multistep_ensemble_policy import EnsembleAgreementPolicy


class DummyPolicy:
    def __init__(self, action, baseline=(0, 0, 0), **plan):
        self.action = np.asarray(action, dtype=np.int64)
        self.baseline = list(baseline)
        self.plan = plan
        self.last_plan = {}

    def predict_guarded(self, env):
        self.last_plan = {
            "baseline_action": self.baseline,
            "predicted_energy_reduction_wh": 1.0,
            "candidate_count": 3,
            **self.plan,
        }
        return self.action.copy()


class MultiStepEnsemblePolicyTests(unittest.TestCase):
    def test_two_of_three_agreement_executes_majority_action(self):
        policy = EnsembleAgreementPolicy(
            [DummyPolicy((1, 0, 0)), DummyPolicy((1, 0, 0)), DummyPolicy((0, 1, 0))]
        )
        action = policy.predict_guarded(env=object())
        np.testing.assert_array_equal(action, np.asarray([1, 0, 0]))
        self.assertEqual(policy.last_plan["ensemble_agreement_count"], 2)
        self.assertTrue(policy.last_plan["override_accepted"])

    def test_three_way_disagreement_falls_back_to_physics_baseline(self):
        policy = EnsembleAgreementPolicy(
            [DummyPolicy((1, 0, 0)), DummyPolicy((0, 1, 0)), DummyPolicy((0, 0, 1))]
        )
        action = policy.predict_guarded(env=object())
        np.testing.assert_array_equal(action, np.asarray([0, 0, 0]))
        self.assertEqual(policy.last_plan["ensemble_agreement_count"], 1)
        self.assertFalse(policy.last_plan["override_accepted"])

    def test_bounded_evidence_rejects_slow_majority_action(self):
        members = [
            DummyPolicy(
                (1, 0, 0),
                predicted_risk_reduction=1.0,
                predicted_throughput_delta=0.1,
                predicted_time_increase_sec=3.0,
                risk_gate_threshold=0.75,
            )
            for _ in range(3)
        ]
        policy = EnsembleAgreementPolicy(members, decision_mode="bounded_evidence")
        action = policy.predict_guarded(env=object())
        np.testing.assert_array_equal(action, np.asarray([0, 0, 0]))
        self.assertFalse(policy.last_plan["override_accepted"])

    def test_bounded_evidence_accepts_risk_reduction_with_bounded_time(self):
        members = [
            DummyPolicy(
                (1, 0, 0),
                predicted_risk_reduction=1.0,
                predicted_throughput_delta=0.1,
                predicted_time_increase_sec=2.0,
                risk_gate_threshold=0.75,
            )
            for _ in range(3)
        ]
        policy = EnsembleAgreementPolicy(members, decision_mode="bounded_evidence")
        action = policy.predict_guarded(env=object())
        np.testing.assert_array_equal(action, np.asarray([1, 0, 0]))
        self.assertTrue(policy.last_plan["override_accepted"])
        self.assertEqual(policy.last_plan["override_evidence"], "accept_bounded_risk")

    def test_energy_neutral_gate_rejects_risk_action_with_energy_increase(self):
        members = [
            DummyPolicy(
                (1, 0, 0),
                predicted_risk_reduction=1.0,
                predicted_energy_reduction_wh=-0.1,
                predicted_throughput_delta=0.0,
                predicted_time_increase_sec=0.0,
                risk_gate_threshold=0.75,
            )
            for _ in range(3)
        ]
        policy = EnsembleAgreementPolicy(
            members, decision_mode="energy_neutral_bounded_evidence"
        )
        action = policy.predict_guarded(env=object())
        np.testing.assert_array_equal(action, np.asarray([0, 0, 0]))
        self.assertFalse(policy.last_plan["override_accepted"])

    def test_energy_neutral_gate_accepts_nonnegative_risk_action(self):
        members = [
            DummyPolicy(
                (1, 0, 0),
                predicted_risk_reduction=1.0,
                predicted_energy_reduction_wh=0.0,
                predicted_throughput_delta=0.0,
                predicted_time_increase_sec=0.0,
                risk_gate_threshold=0.75,
            )
            for _ in range(3)
        ]
        policy = EnsembleAgreementPolicy(
            members, decision_mode="energy_neutral_bounded_evidence"
        )
        action = policy.predict_guarded(env=object())
        np.testing.assert_array_equal(action, np.asarray([1, 0, 0]))
        self.assertEqual(
            policy.last_plan["override_evidence"], "accept_energy_neutral_risk"
        )


if __name__ == "__main__":
    unittest.main()
