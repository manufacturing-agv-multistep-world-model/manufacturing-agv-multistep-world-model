from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from physics_graph_world_model import (
    battery_reserve_wh,
    charge_start_soc,
    estimate_loaded_mission_energy_wh,
)


@dataclass(frozen=True)
class SafetyGuardReport:
    forced_action_count: int
    forced_action_rate: float
    min_battery_soc: float
    low_battery_risk: float


def _resume_soc(env: Any, charge_resume_soc: float | None) -> float:
    if charge_resume_soc is None:
        charge_resume_soc = 80.0
    return max(float(charge_resume_soc), charge_start_soc(env))


def low_battery_risk(env: Any) -> float:
    """Normalized risk below the dispatch-to-charge SOC threshold."""

    threshold = max(charge_start_soc(env), 1.0)
    deficits = [max(0.0, threshold - float(b)) / threshold for b in env.agv_batteries]
    return float(np.mean(deficits)) if deficits else 0.0


def required_safety_actions(env: Any, charge_resume_soc: float | None = None) -> np.ndarray:
    """Return hard safety actions for each AGV, using -1 when unconstrained.

    Action convention:
    - `0`: wait / charge if located at the charger.
    - `3`: travel to the charger.

    The shield models a standard industrial control layer: learned policies may
    optimize flow, but they cannot override battery reserve and charging-dwell
    requirements.
    """

    required = -np.ones(env.agv_count, dtype=np.int64)
    resume_soc = _resume_soc(env, charge_resume_soc)
    critical_soc = max(8.0, 0.5 * float(env.config.low_battery_soc))

    for agv_id, position in enumerate(env.agv_positions):
        battery_soc = float(env.agv_batteries[agv_id])
        loaded = bool(env._agv_loaded(agv_id))
        job = env._current_job(agv_id)
        at_charger = position == env.CHARGE_NODE

        if at_charger:
            if loaded and job is not None:
                capacity_wh = max(float(env.config.battery_capacity_wh), 1e-6)
                required_wh = estimate_loaded_mission_energy_wh(env, agv_id, job) + battery_reserve_wh(env)
                required_soc = min(100.0, 100.0 * required_wh / capacity_wh)
                if battery_soc < max(required_soc, charge_start_soc(env)):
                    required[agv_id] = 0
                    continue
            elif battery_soc < resume_soc:
                required[agv_id] = 0
                continue

        if battery_soc < critical_soc:
            required[agv_id] = 0 if at_charger else 3
            continue

        if not loaded and battery_soc < charge_start_soc(env):
            required[agv_id] = 0 if at_charger else 3
            continue

        if loaded and job is not None:
            battery_wh = battery_soc * float(env.config.battery_capacity_wh) / 100.0
            required_wh = estimate_loaded_mission_energy_wh(env, agv_id, job) + battery_reserve_wh(env)
            if battery_wh < required_wh:
                required[agv_id] = 0 if at_charger else 3

    return required


def apply_required_actions(actions: np.ndarray, required_actions: np.ndarray) -> np.ndarray:
    executed = np.asarray(actions, dtype=np.int64).reshape(required_actions.shape).copy()
    constrained = required_actions >= 0
    executed[constrained] = required_actions[constrained]
    return np.clip(executed, 0, 3).astype(np.int64)


def safety_intervention_rate(actions: np.ndarray, required_actions: np.ndarray) -> float:
    """Fraction of AGV actions that the safety layer actually changes.

    This differs from the safety-constraint occupancy: a charging dwell can keep
    an AGV constrained for many steps even when the learned policy already
    selects the required action.
    """

    required = np.asarray(required_actions, dtype=np.int64)
    proposed = np.asarray(actions, dtype=np.int64).reshape(required.shape)
    constrained = required >= 0
    changed = constrained & (proposed != required)
    return float(np.sum(changed) / max(required.size, 1))


def apply_industrial_safety_guard(
    env: Any,
    actions: np.ndarray,
    charge_resume_soc: float | None = None,
    enabled: bool = True,
) -> np.ndarray:
    if not enabled:
        return np.asarray(actions, dtype=np.int64).reshape(env.agv_count)
    required = required_safety_actions(env, charge_resume_soc=charge_resume_soc)
    return apply_required_actions(actions, required)


def safety_guard_report(env: Any, required_actions: np.ndarray) -> SafetyGuardReport:
    forced_count = int(np.sum(np.asarray(required_actions) >= 0))
    agv_count = max(int(env.agv_count), 1)
    min_battery = min(float(b) for b in env.agv_batteries) if env.agv_batteries else 0.0
    return SafetyGuardReport(
        forced_action_count=forced_count,
        forced_action_rate=float(forced_count / agv_count),
        min_battery_soc=float(min_battery),
        low_battery_risk=low_battery_risk(env),
    )
