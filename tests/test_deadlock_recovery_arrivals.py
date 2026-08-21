from __future__ import annotations

import numpy as np

from agv_case_env import AGV_A_Charge_Env


def test_arrivals_during_deadlock_recovery_are_visible_next_decision() -> None:
    env = AGV_A_Charge_Env(
        agv_count=3,
        env_variant="full",
        reward_mode="hybrid",
        scenario="rush",
        dispatch_rule="dt_aware",
        capacity_mode="stress",
        max_steps=100,
        max_released_jobs=None,
        seed=16002,
    )
    env.reset(seed=16002)

    env.metrics.total_time_sec = 100.0
    for template_id in env.next_arrival_by_template:
        env.next_arrival_by_template[template_id] = 1.0e9
    target_template = env.task_templates[0].task_id
    env.next_arrival_by_template[target_template] = 110.0

    env.deadlock_timer = env.config.deadlock_hard_steps
    env._update_deadlock_state = lambda *args, **kwargs: False
    env._recover_deadlock = lambda: 30.0

    jobs_before = len(env.jobs)
    env.step(np.zeros(env.agv_count, dtype=np.int64))

    assert env.metrics.total_time_sec == 100.0 + env.config.wait_time_s + 30.0
    assert len(env.jobs) == jobs_before + 1
    assert any(
        job.template_id == target_template and job.release_time_sec == 110.0
        for job in env.jobs
    )
