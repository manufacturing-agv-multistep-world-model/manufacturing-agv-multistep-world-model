from __future__ import annotations

from typing import Any

from agv_dt_env import AGVDigitalTwinEnv, DigitalTwinConfig


class AGV_A_Charge_Env(AGVDigitalTwinEnv):
    """Project entry point for the CAD-derived Dong'e AGV digital-twin case."""

    def __init__(
        self,
        agv_count: int = 3,
        env_variant: str = "full",
        reward_mode: str = "hybrid",
        scenario: str = "steady",
        dispatch_rule: str = "dt_aware",
        capacity_mode: str = "stress",
        arrival_process: str = "poisson",
        max_steps: int = 2000,
        max_released_jobs: int | None = None,
        seed: int | None = None,
        **config_overrides: Any,
    ):
        config = DigitalTwinConfig(
            agv_count=agv_count,
            env_variant=env_variant,
            reward_mode=reward_mode,
            scenario=scenario,
            dispatch_rule=dispatch_rule,
            capacity_mode=capacity_mode,
            arrival_process=arrival_process,
            max_steps=max_steps,
            max_released_jobs=max_released_jobs,
            seed=seed,
            **config_overrides,
        )
        super().__init__(config=config)
