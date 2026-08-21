from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ParameterRecord:
    name: str
    value: float | int | str | None
    unit: str
    category: str
    rationale: str
    source: str
    sensitivity_values: tuple[float, ...] = ()


def _record(
    name: str,
    value: float | int | str | None,
    unit: str,
    category: str,
    rationale: str,
    source: str,
    sensitivity_values: tuple[float, ...] = (),
) -> ParameterRecord:
    return ParameterRecord(
        name=name,
        value=value,
        unit=unit,
        category=category,
        rationale=rationale,
        source=source,
        sensitivity_values=sensitivity_values,
    )


PARAMETER_REGISTRY: List[ParameterRecord] = [
    _record(
        "speed_max_mps",
        1.2,
        "m/s",
        "physical_kinematics",
        "Nominal industrial AGV/AMR corridor speed used as the route-level speed limit.",
        "CAD-DT baseline; sensitivity around the baseline is required.",
        (0.96, 1.08, 1.32, 1.44),
    ),
    _record(
        "acceleration_mps2",
        0.5,
        "m/s^2",
        "physical_kinematics",
        "Moderate acceleration for loaded indoor AGV motion; avoids assuming instantaneous velocity changes.",
        "High-fidelity DT assumption; sensitivity around the baseline is required.",
        (0.40, 0.45, 0.55, 0.60),
    ),
    _record(
        "jerk_mps3",
        0.8,
        "m/s^3",
        "physical_kinematics",
        "Reserved parameter for a future S-curve extension; it is not active in the reported trapezoidal/triangular kinematics.",
        "Inactive in the reported experiments; retained only for configuration compatibility.",
        (),
    ),
    _record(
        "wait_time_s",
        2.0,
        "s",
        "physical_operation",
        "Discrete decision interval and minimum wait/charge dwell duration.",
        "Simulation control interval.",
        (),
    ),
    _record(
        "lift_time_s",
        8.0,
        "s",
        "physical_operation",
        "Pickup/drop-off handling dwell time for material-transfer operations.",
        "Factory-operation assumption; can be replaced by measured MES/WCS data.",
        (),
    ),
    _record(
        "rotate_time_s",
        4.0,
        "s",
        "physical_operation",
        "Route-change penalty representing non-instant turning/alignment at control points.",
        "High-fidelity DT assumption.",
        (),
    ),
    _record(
        "battery_capacity_wh",
        2400.0,
        "Wh",
        "energy_model",
        "Nominal battery capacity for a 48V/50Ah-class indoor AGV/AMR used under sustained high-demand logistics.",
        "Energy-calibrated DT baseline; replace with AGV nameplate data when available.",
        (1920.0, 2160.0, 2640.0, 2880.0),
    ),
    _record(
        "base_drive_wh_per_s",
        0.06,
        "Wh/s",
        "energy_model",
        "Base traction/control energy during motion.",
        "Calibrated DT coefficient; sensitivity can be bundled into EER analysis.",
        (),
    ),
    _record(
        "rolling_wh_per_m",
        0.14,
        "Wh/m",
        "energy_model",
        "Distance-dependent rolling and drivetrain energy coefficient.",
        "Calibrated DT coefficient; sensitivity can be bundled into EER analysis.",
        (),
    ),
    _record(
        "acceleration_wh",
        0.35,
        "Wh/event",
        "energy_model",
        "Start/stop energy penalty for acceleration and route-change maneuvers.",
        "Calibrated DT coefficient.",
        (),
    ),
    _record(
        "idle_wh_per_s",
        0.01,
        "Wh/s",
        "energy_model",
        "Controller/standby energy while waiting, charging, or handling.",
        "Calibrated DT coefficient.",
        (),
    ),
    _record(
        "loaded_energy_factor",
        1.55,
        "ratio",
        "energy_model",
        "Loaded motion consumes more energy than empty motion; factor is capped by payload ratio in the model.",
        "DT baseline; sensitivity around the baseline is required.",
        (1.24, 1.395, 1.705, 1.86),
    ),
    _record(
        "charge_soc_per_min",
        2.0,
        "%SOC/min",
        "energy_model",
        "Fast opportunity-charging dock; 18% to 80% SOC takes about 31 minutes.",
        "Public AMR charging benchmarks plus sensitivity analysis.",
        (0.5, 1.0, 2.0, 3.0),
    ),
    _record(
        "low_battery_soc",
        18.0,
        "%SOC",
        "energy_model",
        "Dispatch threshold that sends unloaded AGVs to charge before battery depletion risk.",
        "Industrial reserve rule.",
        (),
    ),
    _record(
        "charge_resume_soc",
        80.0,
        "%SOC",
        "energy_model",
        "Minimum SOC before an unloaded AGV leaves the charging dock under the safety shield.",
        "Opportunity-charging operating rule; sensitivity can be tested with charging policy ablations.",
        (70.0, 80.0, 90.0),
    ),
    _record(
        "charge_node_capacity",
        2,
        "AGVs",
        "physical_operation",
        "Effective simultaneous occupancy/service-staging capacity of the Charge node and its immediate dock area.",
        "CAD-DT baseline assumption; capacities 1/2/3 form an infrastructure what-if experiment without policy retraining.",
        (1.0, 2.0, 3.0),
    ),
    _record(
        "delivery_reward",
        100.0,
        "utility/SKU",
        "reward",
        "Defines one completed SKU delivery as 100 normalized utility points.",
        "Industrial utility normalization.",
        (),
    ),
    _record(
        "pickup_reward",
        10.0,
        "utility/pickup",
        "reward",
        "Intermediate pickup milestone equals 10% of a completed delivery.",
        "Potential-based shaping relative to delivery utility.",
        (),
    ),
    _record(
        "conflict_penalty",
        25.0,
        "utility/event",
        "reward",
        "One route conflict/blocking event costs 25% of one completed delivery.",
        "Normalized industrial utility; sensitivity through safety ablation.",
        (),
    ),
    _record(
        "deadlock_penalty",
        120.0,
        "utility/event",
        "reward",
        "Deadlock onset costs slightly more than one delivery to prioritize flow recovery.",
        "Normalized industrial utility; sensitivity through safety ablation.",
        (),
    ),
    _record(
        "out_of_battery_penalty",
        200.0,
        "utility/event",
        "reward",
        "Battery depletion is treated as a severe service failure worth two deliveries.",
        "Normalized industrial utility.",
        (),
    ),
    _record(
        "time_penalty",
        0.10,
        "utility/s",
        "reward",
        "System time cost; one idle minute costs six utility points.",
        "Delay-cost normalization.",
        (),
    ),
    _record(
        "energy_penalty",
        0.035,
        "utility/Wh",
        "reward",
        "Energy cost links policy learning to EER without dominating throughput.",
        "Energy-aware normalized utility.",
        (),
    ),
    _record(
        "wait_penalty",
        0.4,
        "utility/decision",
        "reward",
        "Discourages unnecessary waiting while still allowing charging and yielding.",
        "Potential-based shaping.",
        (),
    ),
    _record(
        "assignment_bonus",
        1.0,
        "utility/job",
        "reward",
        "Small bonus for converting waiting jobs into assigned work.",
        "Queue-management shaping.",
        (),
    ),
    _record(
        "progress_reward_per_m",
        1.2,
        "utility/m",
        "reward",
        "Dense route-progress shaping toward the active task target.",
        "Potential-based shaping; reported separately from KPI evaluation.",
        (),
    ),
    _record(
        "wrong_progress_penalty_per_m",
        0.6,
        "utility/m",
        "reward",
        "Penalty for moving away from the active task target; lower than positive progress to allow detours.",
        "Potential-based shaping.",
        (),
    ),
    _record(
        "deadlock_recovery_penalty",
        80.0,
        "utility/event",
        "reward",
        "Explicit penalty for invoking the recovery controller.",
        "Normalized industrial utility.",
        (),
    ),
]


GRAPH_MAPPO_DEFAULTS: Dict[str, Any] = {
    "total_steps": 200_000,
    "rollout_steps": 1024,
    "ppo_epochs": 4,
    "minibatch_size": 256,
    "learning_rate": 2e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "value_clip_range": 0.20,
    "value_huber_delta": 10.0,
    "target_kl": 0.03,
    "value_coef": 0.50,
    "entropy_coef": 0.010,
    "safe_bc_coef": 0.20,
    "reward_scale": 0.02,
    "throughput_bonus_scale": 0.15,
    "blocked_penalty_scale": 0.01,
    "blocked_time_penalty_scale": 0.02,
    "deadlock_penalty_scale": 1.00,
    "terminal_risk_penalty_scale": 10.00,
    "low_battery_penalty_scale": 0.25,
    "charge_resume_soc": 80.0,
    "max_grad_norm": 0.80,
    "bc_steps": 8000,
    "bc_epochs": 8,
    "hidden_dim": 96,
    "env_max_steps": 2000,
    "eval_episodes": 10,
    "validation_interval_steps": 20_480,
    "validation_episodes": 5,
    "validation_max_intervention_rate": 0.05,
    "validation_max_deadlock_mean": 0.50,
}


WORLD_MODEL_DEFAULTS: Dict[str, Any] = {
    "episodes": 40,
    "max_steps": 400,
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 8e-4,
    "physics_weight": 0.35,
    "exploration_rate": 0.30,
}


MULTISTEP_WORLD_MODEL_DEFAULTS: Dict[str, Any] = {
    "episodes": 60,
    "max_steps": 400,
    "epochs": 80,
    "batch_size": 256,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "physics_weight": 0.35,
    "rollout_discount": 0.90,
    "training_horizon": 5,
    "sequence_stride": 1,
    "teacher_forcing_start": 0.90,
    "teacher_forcing_end": 0.10,
    "exploration_rate": 0.25,
    "hidden_dim": 96,
    "planning_horizon": 3,
    "beam_width": 8,
    "planning_discount": 0.95,
}


PARAMETER_REGISTRY.extend(
    [
        _record(
            "world_model_training_horizon",
            MULTISTEP_WORLD_MODEL_DEFAULTS["training_horizon"],
            "decision steps",
            "world_model_multistep",
            "Five-step supervision exposes compounding state error while retaining stable minibatch training.",
            "V9 design choice; evaluated at open-loop horizons 1, 3, 5, and 10.",
            (3.0, 5.0, 10.0),
        ),
        _record(
            "world_model_rollout_discount",
            MULTISTEP_WORLD_MODEL_DEFAULTS["rollout_discount"],
            "dimensionless",
            "world_model_multistep",
            "Later rollout errors remain influential without dominating the one-step physical fit.",
            "Geometric temporal weighting; fixed before final evaluation.",
            (0.85, 0.90, 0.95),
        ),
        _record(
            "world_model_planning_horizon",
            MULTISTEP_WORLD_MODEL_DEFAULTS["planning_horizon"],
            "decision steps",
            "world_model_multistep",
            "Three-step receding-horizon control balances anticipatory coordination and model-error accumulation.",
            "Direct controller ablation H=1, 3, and 5.",
            (1.0, 3.0, 5.0),
        ),
        _record(
            "world_model_beam_width",
            MULTISTEP_WORLD_MODEL_DEFAULTS["beam_width"],
            "candidate sequences",
            "world_model_multistep",
            "Eight retained action sequences provide bounded planning cost for online dispatch.",
            "Computational-budget design; report controller wall-clock latency.",
            (4.0, 8.0, 16.0),
        ),
        _record(
            "world_model_teacher_forcing_schedule",
            "0.90->0.10",
            "ratio",
            "world_model_multistep",
            "Scheduled sampling transitions from stable supervised fitting to predominantly self-generated rollout states.",
            "Validation always uses zero teacher forcing.",
            (),
        ),
    ]
)


MPC_UTILITY_WEIGHTS: Dict[str, float] = {
    "predicted_reward": 1.0,
    "throughput_sku": 80.0,
    "fleet_distribution_entropy": 3.0,
    "battery_soc": 1.0,
    "predicted_time_s": -0.12,
    "predicted_energy_wh": -0.08,
    "predicted_blocked_event": -12.0,
    "predicted_deadlock_event": -50.0,
    "predicted_route_blocked_agent_step": -12.0,
    "predicted_charge_queue_agent_step": -16.0,
    "physics_blocked_event": -10.0,
    "physics_time_s": -0.04,
    "physics_energy_wh": -0.04,
    "dt_aware_tie_break": 2.0,
    "model_risk_reduction_gate": 0.5,
    # H=3 held-out energy MAE is about 1.35 Wh. A 2.75 Wh gate is a
    # conservative two-error-margin threshold for pairwise action evidence.
    "model_energy_reduction_gate_wh": 2.75,
    "model_throughput_drop_tolerance_sku": 0.0,
    "model_time_increase_tolerance_s": 0.0,
    "proactive_charge_margin_soc": 8.0,
    # One half agent-step is deliberately above numerical noise while still
    # allowing a one-slot charger to act before a full queue materializes.
    "model_charge_queue_reduction_gate_agent_steps": 0.5,
}


def parameter_values(category: str | None = None) -> Dict[str, Any]:
    records = PARAMETER_REGISTRY if category is None else [r for r in PARAMETER_REGISTRY if r.category == category]
    return {record.name: record.value for record in records}


def env_default_values() -> Dict[str, Any]:
    categories = {"physical_kinematics", "physical_operation", "energy_model", "reward"}
    return {record.name: record.value for record in PARAMETER_REGISTRY if record.category in categories}


def parameter_table_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in PARAMETER_REGISTRY:
        row = asdict(record)
        row["sensitivity_values"] = "|".join(str(v) for v in record.sensitivity_values)
        rows.append(row)
    return rows
