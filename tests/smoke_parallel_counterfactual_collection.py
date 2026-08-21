from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from counterfactual_rollout_v141 import (
    CounterfactualCollectionConfig,
    collect_counterfactual_samples_parallel,
)


def _assert_equal(left: list[dict], right: list[dict]) -> None:
    assert len(left) == len(right) and len(left) > 0
    for left_sample, right_sample in zip(left, right):
        assert left_sample.keys() == right_sample.keys()
        for key in left_sample:
            np.testing.assert_array_equal(
                np.asarray(left_sample[key]),
                np.asarray(right_sample[key]),
                err_msg=key,
            )


def main() -> None:
    config = CounterfactualCollectionConfig(
        episodes=2,
        behavior_steps=20,
        warmup_steps=0,
        sample_stride=2,
        candidates_per_state=1,
        horizons_sec=(1.0, 2.0, 3.0),
        max_rollout_steps=20,
        maximum_relative_overshoot=100.0,
        maximum_absolute_overshoot_sec=300.0,
        seed=99145,
    )
    serial = collect_counterfactual_samples_parallel(
        config, parallel_episodes=1
    )
    parallel = collect_counterfactual_samples_parallel(
        config, parallel_episodes=2
    )
    _assert_equal(serial, parallel)
    print(
        f"Parallel collection determinism: PASS ({len(serial)} paired samples)"
    )


if __name__ == "__main__":
    main()
