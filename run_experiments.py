from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from agv_case_env import AGV_A_Charge_Env
from industrial_safety_guard import (
    apply_industrial_safety_guard,
    required_safety_actions,
    safety_guard_report,
    safety_intervention_rate,
)


ROOT = Path(__file__).resolve().parent

CONFIG_OVERRIDE_KEYS = {
    "speed_max_mps",
    "edge_speed_multiplier",
    "acceleration_mps2",
    "jerk_mps3",
    "wait_time_s",
    "lift_time_s",
    "rotate_time_s",
    "battery_capacity_wh",
    "base_drive_wh_per_s",
    "rolling_wh_per_m",
    "acceleration_wh",
    "idle_wh_per_s",
    "loaded_energy_factor",
    "charge_soc_per_min",
    "low_battery_soc",
    "charge_node_capacity",
    "arrival_rate_multiplier",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Route-B high-fidelity DT and industrial AGV dispatch evaluation experiments."
    )
    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "physical",
            "physical_fixed_time",
            "dispatch",
            "reward",
            "reward_fixed_time",
            "capacity",
            "radar",
            "world_model",
            "fixed_time",
            "charge_capacity_fixed_time",
        ],
        default="all",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy",
        choices=["heuristic", "random", "ppo", "physics_only_guarded", "world_model", "world_model_guarded", "graph_mappo"],
        default="heuristic",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--graph-policy-path", default=None)
    parser.add_argument("--world-model-planning-horizon", type=int, default=None)
    parser.add_argument("--world-model-beam-width", type=int, default=None)
    parser.add_argument(
        "--world-model-risk-gate",
        type=float,
        default=None,
        help="Minimum predicted blocking-equivalent reduction required to override a safe progressing baseline action.",
    )
    parser.add_argument(
        "--world-model-device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--output-dir", default="experiment_results_route_b")
    parser.add_argument("--agv-count", type=int, default=None, help="Override AGV count for non-capacity suites.")
    parser.add_argument(
        "--max-released-jobs",
        type=int,
        default=None,
        help="Optional positive job-release cap for smoke tests. Leave unset for publication runs.",
    )
    parser.add_argument(
        "--disable-graph-safety-shield",
        action="store_true",
        help="Evaluate Graph-MAPPO raw actions without the hard industrial safety shield.",
    )
    parser.add_argument(
        "--fixed-time-hours",
        default="1,4,8",
        help="Comma-separated physical-time horizons for --suite fixed_time, in hours.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Optional comma-separated method labels used to filter an experiment suite.",
    )
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test matrix.")
    return parser


def parse_fixed_time_hours(value: str) -> List[float]:
    hours: List[float] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        parsed = float(text)
        if parsed <= 0:
            raise ValueError("--fixed-time-hours values must be positive")
        hours.append(parsed)
    if not hours:
        raise ValueError("--fixed-time-hours must include at least one duration")
    return hours


def experiment_grid(suite: str, quick: bool = False, fixed_time_hours: List[float] | None = None) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    if suite in ("all", "physical"):
        for variant in ["ideal", "kinematics", "full"]:
            specs.append(
                {
                    "experiment": "E1_physical_ablation",
                    "env_variant": variant,
                    "execution_env_variant": "full",
                    "policy_variant": variant,
                    "reward_mode": "hybrid",
                    "scenario": "rush",
                    "dispatch_rule": "dt_aware",
                    "capacity_mode": "stress",
                    "agv_count": 3,
                }
            )

    if suite == "physical_fixed_time":
        horizons = fixed_time_hours or [1.0, 4.0, 8.0]
        if quick:
            horizons = horizons[:1]
        for hours in horizons:
            for label, variant in [
                ("Ideal Sim", "ideal"),
                ("Kinematics DT", "kinematics"),
                ("Full High-Fidelity DT", "full"),
            ]:
                specs.append(
                    {
                        "experiment": "E1_physical_fidelity_fixed_time",
                        "method": label,
                        "env_variant": variant,
                        "execution_env_variant": "full",
                        "policy_variant": variant,
                        "reward_mode": "hybrid",
                        "scenario": "rush",
                        "dispatch_rule": "dt_aware",
                        "capacity_mode": "baseline",
                        "agv_count": 3,
                        "policy_override": "heuristic",
                        "fidelity_dispatch_mode": True,
                        "fixed_time_target_h": float(hours),
                        "fixed_time_target_sec": float(hours) * 3600.0,
                        "max_released_jobs": None,
                    }
                )

    if suite in ("all", "dispatch"):
        for rule in ["fcfs", "nearest", "priority", "dt_aware"]:
            specs.append(
                {
                    "experiment": "E2_industrial_dispatch_policy",
                    "env_variant": "full",
                    "reward_mode": "hybrid",
                    "scenario": "rush",
                    "dispatch_rule": rule,
                    "capacity_mode": "stress",
                    "agv_count": 3,
                }
            )

    if suite in ("all", "reward"):
        for reward_mode in ["individual", "global", "hybrid"]:
            specs.append(
                {
                    "experiment": "E3_coordination_mechanism",
                    "env_variant": "full",
                    "reward_mode": reward_mode,
                    "scenario": "rush",
                    "dispatch_rule": "dt_aware",
                    "capacity_mode": "stress",
                    "agv_count": 3,
                }
            )

    if suite == "reward_fixed_time":
        horizons = fixed_time_hours or [1.0, 4.0, 8.0]
        if quick:
            horizons = horizons[:1]
        for hours in horizons:
            for label, reward_mode in [
                ("Individual reward", "individual"),
                ("Global reward", "global"),
                ("Hybrid reward", "hybrid"),
            ]:
                specs.append(
                    {
                        "experiment": "E2_reward_coordination_fixed_time",
                        "method": label,
                        "env_variant": "full",
                        "execution_env_variant": "full",
                        "policy_variant": "full",
                        "reward_mode": reward_mode,
                        "scenario": "rush",
                        "dispatch_rule": "dt_aware",
                        "capacity_mode": "stress",
                        "agv_count": 3,
                        "policy_override": "heuristic",
                        "fixed_time_target_h": float(hours),
                        "fixed_time_target_sec": float(hours) * 3600.0,
                        "max_released_jobs": None,
                    }
                )

    if suite in ("all", "capacity"):
        upper = 5 if quick else 8
        for agv_count in range(1, upper + 1):
            specs.append(
                {
                    "experiment": "E_capacity_paradox",
                    "env_variant": "full",
                    "reward_mode": "hybrid",
                    "scenario": "rush",
                    "dispatch_rule": "dt_aware",
                    "capacity_mode": "stress",
                    "agv_count": agv_count,
                }
            )

    if suite in ("all", "radar"):
        for variant, rule, capacity in [
            ("ideal", "fcfs", "baseline"),
            ("kinematics", "nearest", "baseline"),
            ("full", "dt_aware", "stress"),
        ]:
            specs.append(
                {
                    "experiment": "E4_industrial_kpi_radar",
                    "env_variant": variant,
                    "reward_mode": "hybrid",
                    "scenario": "rush",
                    "dispatch_rule": rule,
                    "capacity_mode": capacity,
                    "agv_count": 3,
                }
            )

    if suite == "world_model":
        for label, rule, policy_override in [
            ("Nearest", "nearest", "heuristic"),
            ("DT-aware", "dt_aware", "heuristic"),
            ("PI-only safety control", "dt_aware", "physics_only_guarded"),
            ("PI-GWM-MPC", "dt_aware", "world_model"),
            ("PI-GWM-MPC-G", "dt_aware", "world_model_guarded"),
            ("PI-GWM-GMAPPO", "dt_aware", "graph_mappo"),
        ]:
            specs.append(
                {
                    "experiment": "E5_physics_graph_world_model",
                    "method": label,
                    "env_variant": "full",
                    "reward_mode": "hybrid",
                    "scenario": "rush",
                    "dispatch_rule": rule,
                    "capacity_mode": "stress",
                    "agv_count": 3,
                    "policy_override": policy_override,
                }
            )

    if suite == "fixed_time":
        horizons = fixed_time_hours or [1.0, 4.0, 8.0]
        if quick:
            horizons = horizons[:1]
        for hours in horizons:
            for label, rule, policy_override in [
                ("Nearest", "nearest", "heuristic"),
                ("DT-aware", "dt_aware", "heuristic"),
                ("PI-only safety control", "dt_aware", "physics_only_guarded"),
                ("PI-GWM-MPC", "dt_aware", "world_model"),
                ("PI-GWM-MPC-G", "dt_aware", "world_model_guarded"),
                ("PI-GWM-GMAPPO", "dt_aware", "graph_mappo"),
            ]:
                specs.append(
                    {
                        "experiment": "E6_fixed_physical_time_jms",
                        "method": label,
                        "env_variant": "full",
                        "reward_mode": "hybrid",
                        "scenario": "rush",
                        "dispatch_rule": rule,
                        "capacity_mode": "baseline",
                        "agv_count": 3,
                        "policy_override": policy_override,
                        "fixed_time_target_h": float(hours),
                        "fixed_time_target_sec": float(hours) * 3600.0,
                        "max_released_jobs": None,
                    }
                )

    if suite == "charge_capacity_fixed_time":
        horizons = fixed_time_hours or [8.0]
        if quick:
            horizons = horizons[:1]
        for hours in horizons:
            for charge_capacity in [1, 2, 3]:
                for label, rule, policy_override in [
                    ("DT-aware", "dt_aware", "heuristic"),
                    ("PI-GWM-MPC-G", "dt_aware", "world_model_guarded"),
                    ("PI-GWM-GMAPPO", "dt_aware", "graph_mappo"),
                ]:
                    specs.append(
                        {
                            "experiment": "E7_charge_capacity_what_if",
                            "method": label,
                            "env_variant": "full",
                            "reward_mode": "hybrid",
                            "scenario": "rush",
                            "dispatch_rule": rule,
                            "capacity_mode": "baseline",
                            "charge_node_capacity": charge_capacity,
                            "agv_count": 3,
                            "policy_override": policy_override,
                            "fixed_time_target_h": float(hours),
                            "fixed_time_target_sec": float(hours) * 3600.0,
                            "max_released_jobs": None,
                        }
                    )

    if quick:
        return specs[: min(len(specs), 8)]
    return specs


def apply_agv_count_override(specs: List[Dict[str, Any]], agv_count: int | None) -> List[Dict[str, Any]]:
    if agv_count is None:
        return specs
    updated: List[Dict[str, Any]] = []
    for spec in specs:
        copied = dict(spec)
        if copied.get("experiment") != "E_capacity_paradox":
            copied["agv_count"] = int(agv_count)
        updated.append(copied)
    return updated


def heuristic_action(env: AGV_A_Charge_Env) -> np.ndarray:
    """Rule action policy used to evaluate dispatch rules reproducibly.

    Actions:
    0 = wait/charge, 1 = follow task target, 2 = move to passing buffer, 3 = go charge.

    For Route B, reward modes are interpreted as coordination behavior settings:
    individual = aggressive/selfish, global = conservative/shared, hybrid = balanced.
    The hybrid setting uses a one-step local congestion penalty proxy: lower-priority
    empty vehicles yield at single-capacity bottlenecks instead of blindly taking
    the same next node as loaded or long-waiting vehicles.
    """

    actions = np.ones(env.agv_count, dtype=np.int64)
    policy_variant = getattr(env, "policy_variant", env.config.env_variant)
    coordination_candidates: List[int] = []
    for i, position in enumerate(env.agv_positions):
        loaded = bool(env._agv_loaded(i))
        job = env._current_job(i)
        if policy_variant == "full" and env.agv_batteries[i] < env.config.low_battery_soc and not loaded:
            actions[i] = 0 if position == env.CHARGE_NODE else 3
            continue
        if job is None:
            actions[i] = 0
            continue

        if policy_variant == "ideal":
            actions[i] = 1
            continue

        if env.config.reward_mode == "individual":
            actions[i] = 1
            continue

        if env.config.reward_mode == "global" and i == env.agv_count - 1 and not loaded:
            actions[i] = 0
            continue

        if env.config.dispatch_rule != "dt_aware":
            actions[i] = 1
            continue

        actions[i] = 1
        if env.config.reward_mode == "hybrid" and policy_variant in {"kinematics", "full"}:
            coordination_candidates.append(i)

    if env.config.reward_mode != "hybrid" or not coordination_candidates:
        return actions

    def coordination_priority(agv_id: int) -> float:
        job = env._current_job(agv_id)
        loaded_bonus = 100.0 if env._agv_loaded(agv_id) else 0.0
        wait_bonus = float(env.wait_steps[agv_id])
        if job is None:
            return loaded_bonus + wait_bonus
        age_sec = max(0.0, env.metrics.total_time_sec - job.release_time_sec)
        job_priority_bonus = 8.0 / max(float(job.priority), 1.0)
        return loaded_bonus + wait_bonus + 0.01 * age_sec + job_priority_bonus

    def conflict_contenders(agv_id: int, candidate_actions: np.ndarray) -> List[int]:
        proposals, _, _ = env._propose_positions(candidate_actions)
        proposed = proposals[agv_id]
        current = env.agv_positions[agv_id]
        if proposed == current or env._node_capacity(proposed) > 1:
            return []

        contenders: List[int] = []
        for other_id, other_position in enumerate(env.agv_positions):
            if other_id == agv_id:
                continue
            other_proposed = proposals[other_id]
            same_next_node = other_proposed == proposed
            occupied_not_leaving = other_position == proposed and other_proposed == other_position
            head_on_swap = other_position == proposed and other_proposed == current
            priority_gap = coordination_priority(other_id) - coordination_priority(agv_id)
            other_loaded = env._agv_loaded(other_id)
            self_empty = not env._agv_loaded(agv_id)
            congestion_has_started = env.wait_steps[agv_id] >= 2 or env.wait_steps[other_id] >= 2
            should_yield = (
                (same_next_node and self_empty and other_loaded and congestion_has_started)
                or (head_on_swap and (self_empty and other_loaded or priority_gap >= 4.0))
                or (same_next_node and priority_gap >= 8.0 and congestion_has_started)
                or (occupied_not_leaving and self_empty and other_loaded and env.wait_steps[agv_id] >= 2)
            )
            if should_yield:
                contenders.append(other_id)
        return contenders

    def buffer_action_is_useful(agv_id: int, current_actions: np.ndarray) -> bool:
        if env.agv_positions[agv_id] == env.PASSING_BUFFER_NODE:
            return False
        buffer_occupancy = sum(1 for pos in env.agv_positions if pos == env.PASSING_BUFFER_NODE)
        if buffer_occupancy >= env._node_capacity(env.PASSING_BUFFER_NODE):
            return False
        trial_actions = current_actions.copy()
        trial_actions[agv_id] = 2
        trial_proposals, _, _ = env._propose_positions(trial_actions)
        trial_next = trial_proposals[agv_id]
        if trial_next == env.agv_positions[agv_id]:
            return False
        return all(pos != trial_next for j, pos in enumerate(env.agv_positions) if j != agv_id)

    # Evaluate lower-priority vehicles first so they make space for loaded or delayed AGVs.
    for i in sorted(coordination_candidates, key=coordination_priority):
        contenders = conflict_contenders(i, actions)
        if not contenders:
            continue
        best = max([i, *contenders], key=coordination_priority)
        if best == i:
            continue
        if not env._agv_loaded(i) and buffer_action_is_useful(i, actions):
            actions[i] = 2
        else:
            # Do not stop on a single-lane trunk just to be polite: it occupies
            # the bottleneck and can create the deadlock the hybrid rule avoids.
            actions[i] = 1
    return actions


def load_ppo_model(model_path: str | None):
    if not model_path:
        raise ValueError("--model-path is required when --policy=ppo")
    from stable_baselines3 import PPO

    return PPO.load(model_path)


def load_world_model(
    model_path: str | None,
    planning_horizon: int | None = None,
    beam_width: int | None = None,
    risk_gate_threshold: float | None = None,
    device: str = "auto",
):
    if not model_path:
        raise ValueError("--model-path is required when using a world-model policy or --suite=world_model")
    import torch

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto" or (resolved_device == "cuda" and not torch.cuda.is_available()):
        resolved_device = "cpu"
    checkpoint = torch.load(model_path, map_location=resolved_device)
    model_version = checkpoint.get("model_version")
    if model_version == "pi_gwm_multistep_v12_charge_aware":
        from physics_graph_world_model_multistep_v12 import (
            load_multistep_world_model_policy_v12,
        )

        return load_multistep_world_model_policy_v12(
            model_path,
            device=resolved_device,
            planning_horizon=planning_horizon,
            beam_width=beam_width,
            risk_gate_threshold=risk_gate_threshold,
        )
    if model_version == "pi_gwm_multistep_v11_physical_edges":
        from physics_graph_world_model_multistep_v11 import (
            load_multistep_world_model_policy_v11,
        )

        return load_multistep_world_model_policy_v11(
            model_path,
            device=resolved_device,
            planning_horizon=planning_horizon,
            beam_width=beam_width,
            risk_gate_threshold=risk_gate_threshold,
        )
    if model_version == "pi_gwm_multistep_v10_action_conditioned":
        from physics_graph_world_model_multistep_v10 import (
            load_multistep_world_model_policy_v10,
        )

        return load_multistep_world_model_policy_v10(
            model_path,
            device=resolved_device,
            planning_horizon=planning_horizon,
            beam_width=beam_width,
            risk_gate_threshold=risk_gate_threshold,
        )
    if model_version == "pi_gwm_multistep_v9":
        from physics_graph_world_model_multistep import load_multistep_world_model_policy

        return load_multistep_world_model_policy(
            model_path,
            device=resolved_device,
            planning_horizon=planning_horizon,
            beam_width=beam_width,
            risk_gate_threshold=risk_gate_threshold,
        )
    from physics_graph_world_model import load_world_model_policy

    return load_world_model_policy(model_path)


def load_graph_mappo_model(model_path: str | None):
    if not model_path:
        raise ValueError("--graph-policy-path or --model-path is required when using graph_mappo")
    from graph_mappo_policy import load_graph_mappo_policy

    return load_graph_mappo_policy(model_path)


def run_episode(
    spec: Dict[str, Any],
    episode_id: int,
    seed: int,
    max_steps: int,
    policy: str,
    models: Dict[str, Any] | None = None,
    disable_graph_safety_shield: bool = False,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    execution_env_variant = spec.get("execution_env_variant", spec["env_variant"])
    fixed_time_target_sec = spec.get("fixed_time_target_sec")
    episode_step_cap = int(max_steps)
    if fixed_time_target_sec is not None:
        # Safety cap for fixed-time studies: at least enough 2-second wait decisions
        # to reach the requested physical horizon even under heavy congestion.
        episode_step_cap = max(episode_step_cap, int(math.ceil(float(fixed_time_target_sec) / 2.0)) + 100)
    config_overrides = {
        key: value
        for key, value in spec.get("config_overrides", {}).items()
        if key in CONFIG_OVERRIDE_KEYS
    }
    for key in CONFIG_OVERRIDE_KEYS:
        if key in spec:
            config_overrides[key] = spec[key]
    env = AGV_A_Charge_Env(
        agv_count=spec["agv_count"],
        env_variant=execution_env_variant,
        reward_mode=spec["reward_mode"],
        scenario=spec["scenario"],
        dispatch_rule=spec["dispatch_rule"],
        capacity_mode=spec["capacity_mode"],
        max_steps=episode_step_cap,
        max_released_jobs=spec.get("max_released_jobs"),
        seed=seed,
        **config_overrides,
    )
    env.policy_variant = spec.get("policy_variant", spec["env_variant"])
    env.fidelity_dispatch_mode = bool(spec.get("fidelity_dispatch_mode", False))
    obs, _ = env.reset(seed=seed)
    trace_rows: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []
    total_reward = 0.0
    episode_policy = spec.get("policy_override", policy)
    model_registry = models or {}

    for step in range(episode_step_cap):
        decision_start = time.perf_counter()
        raw_action = None
        plan_info: Dict[str, Any] = {}
        guard_enabled = 0.0
        guard_forced_rate = 0.0
        guard_required_rate = 0.0
        guard_intervention_rate = 0.0
        guard_constraint_rate = 0.0
        guard_low_battery_risk = 0.0
        if episode_policy == "ppo":
            action, _ = model_registry["ppo"].predict(obs, deterministic=True)
        elif episode_policy == "physics_only_guarded":
            raw_action = model_registry["physics_only"].predict_guarded(env)
            plan_info = dict(getattr(model_registry["physics_only"], "last_plan", {}))
            required_action = required_safety_actions(env)
            guard_report = safety_guard_report(env, required_action)
            guard_enabled = 1.0
            guard_intervention_rate = safety_intervention_rate(raw_action, required_action)
            guard_constraint_rate = guard_report.forced_action_rate
            guard_forced_rate = guard_intervention_rate
            guard_required_rate = guard_constraint_rate
            guard_low_battery_risk = guard_report.low_battery_risk
            action = apply_industrial_safety_guard(env, raw_action)
        elif episode_policy == "world_model":
            action = model_registry["world_model"].predict(env)
        elif episode_policy == "world_model_guarded":
            raw_action = model_registry["world_model"].predict_guarded(env)
            plan_info = dict(getattr(model_registry["world_model"], "last_plan", {}))
            required_action = required_safety_actions(env)
            guard_report = safety_guard_report(env, required_action)
            guard_enabled = 1.0
            guard_intervention_rate = safety_intervention_rate(raw_action, required_action)
            guard_constraint_rate = guard_report.forced_action_rate
            guard_forced_rate = guard_intervention_rate
            guard_required_rate = guard_constraint_rate
            guard_low_battery_risk = guard_report.low_battery_risk
            action = apply_industrial_safety_guard(env, raw_action)
        elif episode_policy == "graph_mappo":
            raw_action = model_registry["graph_mappo"].predict(obs, deterministic=True)
            required_action = required_safety_actions(env)
            guard_report = safety_guard_report(env, required_action)
            guard_enabled = 0.0 if disable_graph_safety_shield else 1.0
            guard_intervention_rate = (
                safety_intervention_rate(raw_action, required_action) if guard_enabled else 0.0
            )
            guard_constraint_rate = guard_report.forced_action_rate
            guard_forced_rate = guard_intervention_rate
            guard_required_rate = guard_constraint_rate
            guard_low_battery_risk = guard_report.low_battery_risk
            action = apply_industrial_safety_guard(
                env,
                raw_action,
                enabled=not disable_graph_safety_shield,
            )
        elif episode_policy == "random":
            action = env.action_space.sample()
        else:
            action = heuristic_action(env)

        if raw_action is None:
            raw_action = np.asarray(action, dtype=np.int64)
        action = np.asarray(action, dtype=np.int64)
        decision_compute_sec = float(time.perf_counter() - decision_start)
        action_override_rate = float(np.mean(raw_action != action))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        metrics = info["metrics"]
        reward_components = info.get("reward_components", {})
        trace_rows.append(
            {
                **_spec_prefix(spec, episode_id, seed, episode_policy),
                "step": step,
                "reward": reward,
                **{f"reward_component_{key}": float(value) for key, value in reward_components.items()},
                "real_time_sec": metrics["real_time_sec"],
                "throughput": metrics["throughput"],
                "uph": metrics["uph"],
                "active_jobs": metrics["active_jobs"],
                "released_jobs": metrics["released_jobs"],
                "positions": "|".join(str(p) for p in info["positions"]),
                "position_names": "|".join(info["position_names"]),
                "phases": "|".join(info["phases"]),
                "batteries": "|".join(f"{b:.2f}" for b in info["batteries"]),
                "raw_actions": "|".join(str(int(a)) for a in raw_action),
                "executed_actions": "|".join(str(int(a)) for a in action),
                "action_override_rate": action_override_rate,
                "decision_compute_sec": decision_compute_sec,
                "world_model_raw_planned_actions": "|".join(
                    str(int(value)) for value in plan_info.get("raw_planned_action", [])
                ),
                "world_model_planned_actions": "|".join(
                    str(int(value)) for value in plan_info.get("executed_planned_action", [])
                ),
                "world_model_baseline_actions": "|".join(
                    str(int(value)) for value in plan_info.get("baseline_action", [])
                ),
                "world_model_predicted_risk_reduction": float(
                    plan_info.get("predicted_risk_reduction", 0.0)
                ),
                "world_model_predicted_energy_reduction_wh": float(
                    plan_info.get("predicted_energy_reduction_wh", 0.0)
                ),
                "world_model_predicted_throughput_delta": float(
                    plan_info.get("predicted_throughput_delta", 0.0)
                ),
                "world_model_predicted_time_increase_sec": float(
                    plan_info.get("predicted_time_increase_sec", 0.0)
                ),
                "world_model_predicted_route_blocking_reduction_agent_steps": float(
                    plan_info.get("predicted_route_blocking_reduction_agent_steps", 0.0)
                ),
                "world_model_predicted_charge_queue_reduction_agent_steps": float(
                    plan_info.get("predicted_charge_queue_reduction_agent_steps", 0.0)
                ),
                "world_model_risk_gate_threshold": float(
                    plan_info.get("risk_gate_threshold", 0.0)
                ),
                "world_model_energy_gate_threshold_wh": float(
                    plan_info.get("energy_gate_threshold_wh", 0.0)
                ),
                "world_model_charge_queue_gate_threshold_agent_steps": float(
                    plan_info.get("charge_queue_gate_threshold_agent_steps", 0.0)
                ),
                "world_model_risk_gate_applied": float(
                    bool(plan_info.get("risk_gate_applied", False))
                ),
                "world_model_energy_gate_applied": float(
                    bool(plan_info.get("energy_gate_applied", False))
                ),
                "world_model_physical_gate_applied": float(
                    bool(plan_info.get("physical_gate_applied", False))
                ),
                "world_model_override_accepted": float(
                    bool(plan_info.get("override_accepted", False))
                ),
                "world_model_override_evidence": str(
                    plan_info.get("override_evidence", "baseline")
                ),
                "world_model_override_mode": str(
                    plan_info.get("override_mode", "not_applicable")
                ),
                "world_model_ensemble_size": float(plan_info.get("ensemble_size", 0)),
                "world_model_ensemble_agreement_count": float(
                    plan_info.get("ensemble_agreement_count", 0)
                ),
                "world_model_ensemble_agreement_fraction": float(
                    plan_info.get("ensemble_agreement_fraction", 0.0)
                ),
                "world_model_ensemble_member_actions": ";".join(
                    "|".join(str(int(value)) for value in action)
                    for action in plan_info.get("ensemble_member_actions", [])
                ),
                "world_model_analytical_future_conflict_agent_count": float(
                    plan_info.get("analytical_future_conflict_agent_count", 0)
                ),
                "world_model_operational_energy_action": float(
                    bool(plan_info.get("operational_energy_action", False))
                ),
                "world_model_analytical_charge_staggering": float(
                    bool(plan_info.get("analytical_charge_staggering", False))
                ),
                "world_model_dedicated_charge_gate_required": float(
                    bool(plan_info.get("dedicated_charge_gate_required", False))
                ),
                "world_model_analytical_charge_pressure_agent_count": float(
                    plan_info.get("analytical_charge_pressure_agent_count", 0)
                ),
                "world_model_available_charge_slots": float(
                    plan_info.get("available_charge_slots", 0)
                ),
                "world_model_anti_stagnation_applied": float(
                    bool(plan_info.get("anti_stagnation_applied", False))
                ),
                "world_model_unsafe_candidate_filter_applied": float(
                    bool(plan_info.get("unsafe_candidate_filter_applied", False))
                ),
                "v146_shadow_recommended": float(
                    bool(plan_info.get("v146_shadow_recommended", False))
                ),
                "v146_predicted_normalized_gain": float(
                    plan_info.get("v146_predicted_normalized_gain", 0.0)
                ),
                "v146_candidate_count": float(plan_info.get("v146_candidate_count", 0)),
                "v146_rejection_reason": str(
                    plan_info.get("v146_rejection_reason", "not_applicable")
                ),
                "v146_override_count": float(plan_info.get("v146_override_count", 0)),
                "v146_permanent_fallback": float(
                    bool(plan_info.get("v146_permanent_fallback", False))
                ),
                "deadlock_count": metrics["deadlock_count"],
                "conflict_count": metrics["conflict_count"],
                "blocking_onset_count": metrics["blocking_onset_count"],
                "blocked_count": metrics["blocked_count"],
                "blocked_agent_steps": metrics["blocked_agent_steps"],
                "blocked_time_sec": metrics["blocked_time_sec"],
                "route_blocking_onset_count": metrics["route_blocking_onset_count"],
                "route_blocked_agent_steps": metrics["route_blocked_agent_steps"],
                "route_blocked_time_sec": metrics["route_blocked_time_sec"],
                "charge_queue_onset_count": metrics["charge_queue_onset_count"],
                "charge_queue_blocked_agent_steps": metrics[
                    "charge_queue_blocked_agent_steps"
                ],
                "charge_queue_time_sec": metrics["charge_queue_time_sec"],
                "guard_enabled": guard_enabled,
                "guard_forced_rate": guard_forced_rate,
                "guard_required_rate": guard_required_rate,
                "guard_intervention_rate": guard_intervention_rate,
                "guard_constraint_rate": guard_constraint_rate,
                "guard_low_battery_risk": guard_low_battery_risk,
                "agv_utilization": metrics["agv_utilization"],
                "avg_task_wait_time": metrics["avg_task_wait_time"],
            }
        )

        if step % 20 == 0:
            attention = info["attention_weights"]
            for i in range(attention.shape[0]):
                for j in range(attention.shape[1]):
                    if i != j and attention[i, j] > 0:
                        attention_rows.append(
                            {
                                **_spec_prefix(spec, episode_id, seed, episode_policy),
                                "step": step,
                                "source_agv": i,
                                "target_agv": j,
                                "attention_weight": float(attention[i, j]),
                            }
                        )

        reached_fixed_time = (
            fixed_time_target_sec is not None
            and float(metrics["real_time_sec"]) >= float(fixed_time_target_sec)
        )
        if terminated or truncated or reached_fixed_time:
            break

    summary = {**env.summary(), **_spec_prefix(spec, episode_id, seed, episode_policy), "episode_reward": total_reward}
    if trace_rows:
        summary["mean_guard_enabled"] = float(np.mean([row["guard_enabled"] for row in trace_rows]))
        summary["mean_guard_forced_rate"] = float(np.mean([row["guard_forced_rate"] for row in trace_rows]))
        summary["mean_guard_required_rate"] = float(np.mean([row["guard_required_rate"] for row in trace_rows]))
        summary["mean_guard_intervention_rate"] = float(
            np.mean([row["guard_intervention_rate"] for row in trace_rows])
        )
        summary["mean_guard_constraint_rate"] = float(
            np.mean([row["guard_constraint_rate"] for row in trace_rows])
        )
        summary["mean_guard_low_battery_risk"] = float(np.mean([row["guard_low_battery_risk"] for row in trace_rows]))
        summary["mean_action_override_rate"] = float(
            np.mean([row["action_override_rate"] for row in trace_rows])
        )
        decision_times = np.asarray(
            [row["decision_compute_sec"] for row in trace_rows], dtype=np.float64
        )
        summary["mean_decision_compute_sec"] = float(np.mean(decision_times))
        summary["p95_decision_compute_sec"] = float(np.quantile(decision_times, 0.95))
        summary["max_decision_compute_sec"] = float(np.max(decision_times))
        summary["mean_world_model_predicted_risk_reduction"] = float(
            np.mean([row["world_model_predicted_risk_reduction"] for row in trace_rows])
        )
        summary["mean_world_model_predicted_energy_reduction_wh"] = float(
            np.mean([row["world_model_predicted_energy_reduction_wh"] for row in trace_rows])
        )
        summary["mean_world_model_predicted_throughput_delta"] = float(
            np.mean([row["world_model_predicted_throughput_delta"] for row in trace_rows])
        )
        summary["mean_world_model_predicted_time_increase_sec"] = float(
            np.mean([row["world_model_predicted_time_increase_sec"] for row in trace_rows])
        )
        summary["mean_world_model_predicted_route_blocking_reduction_agent_steps"] = float(
            np.mean(
                [
                    row["world_model_predicted_route_blocking_reduction_agent_steps"]
                    for row in trace_rows
                ]
            )
        )
        summary["mean_world_model_predicted_charge_queue_reduction_agent_steps"] = float(
            np.mean(
                [
                    row["world_model_predicted_charge_queue_reduction_agent_steps"]
                    for row in trace_rows
                ]
            )
        )
        summary["world_model_risk_gate_threshold"] = float(
            np.max([row["world_model_risk_gate_threshold"] for row in trace_rows])
        )
        summary["world_model_energy_gate_threshold_wh"] = float(
            np.max([row["world_model_energy_gate_threshold_wh"] for row in trace_rows])
        )
        summary["world_model_charge_queue_gate_threshold_agent_steps"] = float(
            np.max(
                [
                    row["world_model_charge_queue_gate_threshold_agent_steps"]
                    for row in trace_rows
                ]
            )
        )
        summary["mean_world_model_risk_gate_applied"] = float(
            np.mean([row["world_model_risk_gate_applied"] for row in trace_rows])
        )
        summary["mean_world_model_energy_gate_applied"] = float(
            np.mean([row["world_model_energy_gate_applied"] for row in trace_rows])
        )
        summary["mean_world_model_physical_gate_applied"] = float(
            np.mean([row["world_model_physical_gate_applied"] for row in trace_rows])
        )
        summary["mean_world_model_override_accepted"] = float(
            np.mean([row["world_model_override_accepted"] for row in trace_rows])
        )
        summary["mean_world_model_risk_evidence_accepted"] = float(
            np.mean(
                [row["world_model_override_evidence"] == "accept_risk" for row in trace_rows]
            )
        )
        summary["mean_world_model_energy_evidence_accepted"] = float(
            np.mean(
                [row["world_model_override_evidence"] == "accept_energy" for row in trace_rows]
            )
        )
        summary["mean_world_model_charge_stagger_evidence_accepted"] = float(
            np.mean(
                [
                    row["world_model_override_evidence"] == "accept_charge_stagger"
                    for row in trace_rows
                ]
            )
        )
        summary["mean_world_model_safe_argmax_accepted"] = float(
            np.mean(
                [
                    row["world_model_override_evidence"] == "accept_safe_argmax"
                    for row in trace_rows
                ]
            )
        )
        summary["mean_world_model_ensemble_agreement_fraction"] = float(
            np.mean(
                [row["world_model_ensemble_agreement_fraction"] for row in trace_rows]
            )
        )
        summary["mean_world_model_ensemble_agreement_accepted"] = float(
            np.mean(
                [
                    row["world_model_override_evidence"]
                    == "accept_ensemble_agreement"
                    for row in trace_rows
                ]
            )
        )
        summary["mean_world_model_analytical_future_conflict_agent_count"] = float(
            np.mean(
                [
                    row["world_model_analytical_future_conflict_agent_count"]
                    for row in trace_rows
                ]
            )
        )
        summary["mean_world_model_operational_energy_action"] = float(
            np.mean([row["world_model_operational_energy_action"] for row in trace_rows])
        )
        summary["mean_world_model_analytical_charge_staggering"] = float(
            np.mean([row["world_model_analytical_charge_staggering"] for row in trace_rows])
        )
        summary["mean_world_model_anti_stagnation_applied"] = float(
            np.mean([row["world_model_anti_stagnation_applied"] for row in trace_rows])
        )
        summary["mean_world_model_unsafe_candidate_filter_applied"] = float(
            np.mean([row["world_model_unsafe_candidate_filter_applied"] for row in trace_rows])
        )
        summary["mean_v146_shadow_recommended"] = float(
            np.mean([row["v146_shadow_recommended"] for row in trace_rows])
        )
        summary["mean_v146_predicted_normalized_gain"] = float(
            np.mean([row["v146_predicted_normalized_gain"] for row in trace_rows])
        )
        summary["mean_v146_candidate_count"] = float(
            np.mean([row["v146_candidate_count"] for row in trace_rows])
        )
        summary["v146_final_override_count"] = float(
            np.max([row["v146_override_count"] for row in trace_rows])
        )
        summary["mean_v146_permanent_fallback"] = float(
            np.mean([row["v146_permanent_fallback"] for row in trace_rows])
        )
        reward_component_keys = [
            key for key in trace_rows[0].keys() if key.startswith("reward_component_")
        ]
        for key in reward_component_keys:
            summary[f"mean_{key}"] = float(np.mean([float(row[key]) for row in trace_rows if key in row]))
    if fixed_time_target_sec is not None:
        summary["fixed_time_reached"] = float(summary["real_time_sec"] >= float(fixed_time_target_sec))
        summary["fixed_time_target_sec"] = float(fixed_time_target_sec)
        summary["fixed_time_overshoot_sec"] = float(
            max(0.0, summary["real_time_sec"] - float(fixed_time_target_sec))
        )
        summary["paired_arrival_count"] = float(
            env.arrival_trace_count(float(fixed_time_target_sec))
        )
        summary["paired_arrival_signature"] = env.arrival_trace_signature(
            float(fixed_time_target_sec)
        )
    return summary, trace_rows, attention_rows


def _spec_prefix(spec: Dict[str, Any], episode_id: int, seed: int, policy: str) -> Dict[str, Any]:
    prefix = {
        "experiment": spec["experiment"],
        "method": spec.get("method", spec.get("dispatch_rule", "")),
        "episode": episode_id,
        "seed": seed,
        "policy": policy,
        "env_variant": spec["env_variant"],
        "execution_env_variant": spec.get("execution_env_variant", spec["env_variant"]),
        "policy_variant": spec.get("policy_variant", spec["env_variant"]),
        "fidelity_dispatch_mode": bool(spec.get("fidelity_dispatch_mode", False)),
        "reward_mode": spec["reward_mode"],
        "scenario": spec["scenario"],
        "dispatch_rule": spec["dispatch_rule"],
        "capacity_mode": spec["capacity_mode"],
        "agv_count": spec["agv_count"],
    }
    if "fixed_time_target_h" in spec:
        prefix["fixed_time_target_h"] = spec["fixed_time_target_h"]
        prefix["fixed_time_target_sec"] = spec["fixed_time_target_sec"]
    if "max_released_jobs" in spec:
        prefix["job_release_cap"] = "none" if spec["max_released_jobs"] is None else spec["max_released_jobs"]
    for optional_key in (
        "sensitivity_parameter",
        "sensitivity_label",
        "sensitivity_unit",
        "sensitivity_level",
        "sensitivity_value",
        "sensitivity_multiplier",
        "charge_node_capacity",
        "world_model_planning_horizon",
    ):
        if optional_key in spec:
            prefix[optional_key] = spec[optional_key]
    return prefix


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault(row["experiment"], []).append(row)

    def fmt(value: Any, digits: int = 2) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isnan(number):
            return "N/A"
        return f"{number:.{digits}f}"

    lines = [
        "# AGV Digital Twin Experiment Summary",
        "",
        "Generated by `run_experiments.py` using the high-fidelity CAD-derived DT scenario.",
        "",
    ]
    for experiment, rows in grouped.items():
        lines.append(f"## {experiment}")
        lines.append("")
        lines.append(
            "| setting | UPH | throughput | DRT(s) | EER(Wh/SKU) | FDE | empty | util | wait(s) | stepwise conflict detections | block onsets | blocked vehicle-time(s) | deadlocks |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            horizon = ""
            if row.get("fixed_time_target_h", "") != "":
                horizon = f"/{fmt(row['fixed_time_target_h'], 1)}h"
            method = row.get("method", row.get("policy", "unknown"))
            policy_variant = row.get("policy_variant", "")
            variant_suffix = f"/{policy_variant}" if policy_variant else ""
            sensitivity = ""
            if row.get("sensitivity_parameter", "") != "":
                sensitivity = f"{row['sensitivity_parameter']}={row.get('sensitivity_level', row.get('sensitivity_value', ''))}/"
            setting = (
                f"{sensitivity}{method}{variant_suffix}/{row['env_variant']}/{row['dispatch_rule']}/{row['reward_mode']}/"
                f"{row['scenario']}/{int(float(row['agv_count']))}AGV{horizon}/e{row['episode']}"
            )
            lines.append(
                "| "
                f"{setting} | {fmt(row['uph'])} | {fmt(row['throughput'], 0)} | "
                f"{fmt(row['deadlock_resolution_time'])} | "
                f"{fmt(row['energy_efficiency_wh_per_sku'])} | "
                f"{fmt(row['fleet_distribution_entropy'], 3)} | "
                f"{fmt(row['empty_running_ratio'], 3)} | "
                f"{fmt(row['agv_utilization'], 3)} | "
                f"{fmt(row['avg_task_wait_time'], 1)} | "
                f"{fmt(row['conflict_count'], 0)} | "
                f"{fmt(row.get('blocking_onset_count', float('nan')), 0)} | "
                f"{fmt(row.get('blocked_time_sec', float('nan')), 1)} | "
                f"{fmt(row['deadlock_count'], 0)} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    fixed_time_hours = parse_fixed_time_hours(args.fixed_time_hours)
    specs = apply_agv_count_override(
        experiment_grid(args.suite, quick=args.quick, fixed_time_hours=fixed_time_hours),
        args.agv_count,
    )
    if args.methods:
        requested_methods = {value.strip() for value in args.methods.split(",") if value.strip()}
        specs = [spec for spec in specs if spec.get("method") in requested_methods]
        if not specs:
            raise ValueError(f"No experiment methods matched --methods={args.methods!r}")
    if args.max_released_jobs is not None:
        for spec in specs:
            spec["max_released_jobs"] = int(args.max_released_jobs)
    if args.world_model_planning_horizon is not None:
        if args.world_model_planning_horizon < 1:
            raise ValueError("--world-model-planning-horizon must be at least 1")
        for spec in specs:
            if spec.get("policy_override") in {"world_model", "world_model_guarded"}:
                spec["world_model_planning_horizon"] = args.world_model_planning_horizon
                spec["method"] = f"{spec.get('method', 'PI-GWM-MPC')}-H{args.world_model_planning_horizon}"
    needs_world_model = args.policy in {"world_model", "world_model_guarded"} or any(
        spec.get("policy_override") in {"world_model", "world_model_guarded"} for spec in specs
    )
    needs_graph_mappo = args.policy == "graph_mappo" or any(
        spec.get("policy_override") == "graph_mappo" for spec in specs
    )
    needs_physics_only = args.policy == "physics_only_guarded" or any(
        spec.get("policy_override") == "physics_only_guarded" for spec in specs
    )
    models: Dict[str, Any] = {}
    if needs_physics_only:
        from physics_graph_world_model_multistep import PhysicsOnlyRiskPolicy

        models["physics_only"] = PhysicsOnlyRiskPolicy(agv_count=3)
    if args.policy == "ppo":
        models["ppo"] = load_ppo_model(args.model_path)
    if needs_world_model:
        models["world_model"] = load_world_model(
            args.model_path,
            planning_horizon=args.world_model_planning_horizon,
            beam_width=args.world_model_beam_width,
            risk_gate_threshold=args.world_model_risk_gate,
            device=args.world_model_device,
        )
    if needs_graph_mappo:
        models["graph_mappo"] = load_graph_mappo_model(args.graph_policy_path or args.model_path)

    summaries: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []

    for spec in specs:
        for episode in range(args.episodes):
            seed = args.seed + episode
            summary, trace, attention = run_episode(
                spec=spec,
                episode_id=episode,
                seed=seed,
                max_steps=args.max_steps,
                policy=args.policy,
                models=models,
                disable_graph_safety_shield=args.disable_graph_safety_shield,
            )
            summaries.append(summary)
            traces.extend(trace)
            attention_rows.extend(attention)
            horizon_text = (
                f"target={float(spec['fixed_time_target_h']):.2f}h | "
                if "fixed_time_target_h" in spec
                else ""
            )
            print(
                f"{spec['experiment']} | {spec.get('method', spec['dispatch_rule'])} | "
                f"{spec['env_variant']}/{spec['dispatch_rule']} | "
                f"{spec['agv_count']} AGV | episode {episode} | "
                f"{horizon_text}"
                f"real_h={float(summary['real_time_sec']) / 3600.0:.2f} | "
                f"UPH={float(summary['uph']):.2f} | throughput={float(summary['throughput']):.0f}"
            )

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "trace.csv", traces)
    write_csv(output_dir / "attention_samples.csv", attention_rows)
    write_markdown_summary(output_dir / "summary.md", summaries)
    print(f"Experiment data saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
