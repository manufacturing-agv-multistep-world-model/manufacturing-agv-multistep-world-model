from __future__ import annotations

import numpy as np

from guarded_counterfactual_policy_v146 import (
    FROZEN_UTILITY_MARGIN,
    V146AuthorityLimits,
)


def test_v146_authority_limits_are_frozen_and_bounded() -> None:
    limits = V146AuthorityLimits()
    assert limits.utility_margin == FROZEN_UTILITY_MARGIN == 0.15
    assert limits.cooldown_sec == 60.0
    assert limits.maximum_overrides == 12
    assert limits.maximum_action_hamming_distance == 3


def test_joint_action_scope_matches_the_validated_three_agv_decision_space() -> None:
    baseline = np.asarray([1, 1, 1])
    joint_action = np.asarray([0, 2, 3])
    assert np.count_nonzero(joint_action != baseline) == 3
    assert np.count_nonzero(joint_action != baseline) <= V146AuthorityLimits().maximum_action_hamming_distance
