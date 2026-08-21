from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from diagnose_counterfactual_ranking_v144 import _select_device
from guarded_counterfactual_policy_v146 import (
    FROZEN_UTILITY_MARGIN,
    GuardedCounterfactualPolicyV146,
    GuardedDTBaselinePolicy,
    V146AuthorityLimits,
)
from physics_graph_world_model_counterfactual_v141 import load_counterfactual_model_v141
from run_experiments import write_csv
from run_guarded_counterfactual_pilot_v146 import _method_means, _run, _write_markdown


ROOT = Path(__file__).resolve().parent
MODEL_SEEDS = (42, 43, 44)
COOLDOWN_SEC = 720.0
STUDIES = {
    "one_hour": {
        "protocol": "v147_horizon_isolated_guarded_closed_loop_development_v1",
        "variant": "V14.7",
        "environment_seeds": tuple(range(15901, 15906)),
        "hours": 1.0,
        "maximum_overrides": 4,
        "experiment": "N9_horizon_isolated_closed_loop_pilot",
        "audit_stem": "V147_DEVELOPMENT_AUDIT",
        "scenario": "rush",
        "capacity_mode": "stress",
        "minimum_completed_tasks": 1,
    },
    "four_hour": {
        "protocol": "v147b_horizon_isolated_guarded_closed_loop_4h_development_v1",
        "variant": "V14.7b",
        "environment_seeds": tuple(range(16001, 16006)),
        "hours": 4.0,
        "maximum_overrides": 8,
        "experiment": "N10_horizon_isolated_4h_closed_loop_pilot",
        "audit_stem": "V147B_4H_DEVELOPMENT_AUDIT",
        "scenario": "rush",
        "capacity_mode": "stress",
        "minimum_completed_tasks": 1,
    },
    "four_hour_recovery_sync": {
        "protocol": "v147b_r1_recovery_synchronized_4h_development_v2",
        "variant": "V14.7b-R1",
        "environment_seeds": tuple(range(16001, 16006)),
        "hours": 4.0,
        "maximum_overrides": 8,
        "experiment": "N10_horizon_isolated_4h_closed_loop_pilot_recovery_sync",
        "audit_stem": "V147B_R1_4H_DEVELOPMENT_AUDIT",
        "scenario": "rush",
        "capacity_mode": "stress",
        "minimum_completed_tasks": 1,
    },
    "nominal_four_hour": {
        "protocol": "v148_nominal_steady_4h_development_v1",
        "variant": "V14.8",
        "environment_seeds": tuple(range(17001, 17006)),
        "hours": 4.0,
        "maximum_overrides": 8,
        "experiment": "N11_nominal_steady_4h_closed_loop_development",
        "audit_stem": "V148_NOMINAL_4H_DEVELOPMENT_AUDIT",
        "scenario": "steady",
        "capacity_mode": "baseline",
        "minimum_completed_tasks": 20,
    },
    "steady_stress_four_hour": {
        "protocol": "v149_steady_arrival_stress_capacity_4h_development_v1",
        "variant": "V14.9-S",
        "environment_seeds": tuple(range(18001, 18006)),
        "hours": 4.0,
        "maximum_overrides": 8,
        "experiment": "N12a_steady_arrival_stress_capacity_boundary",
        "audit_stem": "V149S_BOUNDARY_4H_DEVELOPMENT_AUDIT",
        "scenario": "steady",
        "capacity_mode": "stress",
        "minimum_completed_tasks": 20,
    },
    "rush_baseline_four_hour": {
        "protocol": "v149_rush_arrival_baseline_capacity_4h_development_v1",
        "variant": "V14.9-D",
        "environment_seeds": tuple(range(18001, 18006)),
        "hours": 4.0,
        "maximum_overrides": 8,
        "experiment": "N12b_rush_arrival_baseline_capacity_boundary",
        "audit_stem": "V149D_BOUNDARY_4H_DEVELOPMENT_AUDIT",
        "scenario": "rush",
        "capacity_mode": "baseline",
        "minimum_completed_tasks": 20,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V14.7 horizon-isolated development.")
    parser.add_argument("--study", choices=tuple(STUDIES), default="one_hour")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _checkpoint(seed: int) -> Path:
    return ROOT / "world_model_runs" / f"pi_gwm_counterfactual_v141_seed{seed}" / "physics_graph_world_model_counterfactual.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> Path:
    if args.max_steps < 1:
        raise ValueError("Step cap must be positive")
    device = _select_device(args.device, args.require_cuda)
    study = STUDIES[args.study]
    protocol = str(study["protocol"])
    variant = str(study["variant"])
    environment_seeds = tuple(int(seed) for seed in study["environment_seeds"])
    hours = float(study["hours"])
    maximum_overrides = int(study["maximum_overrides"])
    experiment = str(study["experiment"])
    audit_stem = str(study["audit_stem"])
    scenario = str(study["scenario"])
    capacity_mode = str(study["capacity_mode"])
    minimum_completed_tasks = int(study["minimum_completed_tasks"])
    spec_overrides = {"scenario": scenario, "capacity_mode": capacity_mode}
    model_method = f"{variant} horizon-isolated world model"
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [_checkpoint(seed) for seed in MODEL_SEEDS]
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
    models = [load_counterfactual_model_v141(path, device=device) for path in checkpoints]
    limits = V146AuthorityLimits(
        utility_margin=FROZEN_UTILITY_MARGIN,
        cooldown_sec=COOLDOWN_SEC,
        maximum_overrides=maximum_overrides,
        maximum_action_hamming_distance=3,
    )
    summaries: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    for env_seed in environment_seeds:
        baseline_summary, baseline_trace = _run(
            "Guarded DT-aware baseline",
            GuardedDTBaselinePolicy(),
            env_seed,
            hours,
            args.max_steps,
            "horizon_isolated_development",
            protocol=protocol,
            experiment=experiment,
            spec_overrides=spec_overrides,
        )
        controller_summary, controller_trace = _run(
            model_method,
            GuardedCounterfactualPolicyV146(
                models,
                limits,
                policy_label=f"{variant.lower().replace('.', '')}_horizon_isolated_unanimous",
            ),
            env_seed,
            hours,
            args.max_steps,
            "horizon_isolated_development",
            protocol=protocol,
            experiment=experiment,
            spec_overrides=spec_overrides,
        )
        summaries.extend([baseline_summary, controller_summary])
        traces.extend([*baseline_trace, *controller_trace])
    write_csv(output_dir / "episode_summary.csv", summaries)
    write_csv(output_dir / "decision_trace.csv", traces)

    pairs = []
    for seed in environment_seeds:
        rows = [row for row in summaries if int(row["seed"]) == seed]
        pairs.append(
            (
                next(row for row in rows if row["method"] == "Guarded DT-aware baseline"),
                next(row for row in rows if row["method"] == model_method),
            )
        )
    total_overrides = int(sum(right["v146_accepted_override_count"] for _, right in pairs))
    total_decisions = int(sum(right["v146_decision_steps"] for _, right in pairs))
    baseline_block_time = float(np.mean([left["route_blocked_time_sec"] for left, _ in pairs]))
    model_block_time = float(np.mean([right["route_blocked_time_sec"] for _, right in pairs]))
    criteria = [
        {
            "criterion": f"Every paired run reaches the {hours:g}-hour physical horizon",
            "passed": all(left["fixed_time_reached"] > 0.5 and right["fixed_time_reached"] > 0.5 for left, right in pairs),
        },
        {
            "criterion": "Every policy pair receives an identical exogenous task stream",
            "passed": all(
                left["paired_arrival_signature"] == right["paired_arrival_signature"]
                and left["paired_arrival_count"] == right["paired_arrival_count"]
                for left, right in pairs
            ),
        },
        {
            "criterion": f"Every paired method completes at least {minimum_completed_tasks} tasks so operational KPIs are identifiable",
            "passed": all(
                left["throughput"] >= minimum_completed_tasks
                and right["throughput"] >= minimum_completed_tasks
                for left, right in pairs
            ),
        },
        {
            "criterion": f"No {variant} run has an out-of-battery event or timeout",
            "passed": all(right["out_of_battery_rate"] == 0.0 and right["timeout_rate"] == 0.0 for _, right in pairs),
        },
        {
            "criterion": f"{variant} introduces no additional deadlocks in any paired run",
            "passed": all(right["deadlock_count"] <= left["deadlock_count"] for left, right in pairs),
        },
        {
            "criterion": "Mean physical route-blocked time is no more than 101% of the guarded baseline",
            "passed": model_block_time <= 1.01 * baseline_block_time,
        },
        {
            "criterion": "Mean conflict events are no more than 105% of the guarded baseline",
            "passed": float(np.mean([right["conflict_count"] for _, right in pairs]))
            <= 1.05 * float(np.mean([left["conflict_count"] for left, _ in pairs])),
        },
        {
            "criterion": f"{variant} UPH is at least 95% of the guarded baseline in every run",
            "passed": all(right["uph"] >= 0.95 * left["uph"] for left, right in pairs),
        },
        {
            "criterion": f"{variant} EER is no more than 105% of the guarded baseline in every run",
            "passed": all(
                np.isfinite(right["energy_efficiency_wh_per_sku"])
                and right["energy_efficiency_wh_per_sku"] <= 1.05 * left["energy_efficiency_wh_per_sku"]
                for left, right in pairs
            ),
        },
        {
            "criterion": "At least three isolated overrides execute and total authority remains below 5%",
            "passed": total_overrides >= 3 and 0.0 < total_overrides / max(total_decisions, 1) < 0.05,
        },
        {
            "criterion": f"Every run respects the {maximum_overrides}-override authority budget",
            "passed": all(right["v146_accepted_override_count"] <= maximum_overrides for _, right in pairs),
        },
        {
            "criterion": f"Mean {variant} P95 decision time is at most 2 seconds",
            "passed": float(np.mean([right["p95_decision_compute_sec"] for _, right in pairs])) <= 2.0,
        },
    ]
    audit = {
        "protocol": protocol,
        "model_variant": variant,
        "phase": "horizon_isolated_development",
        "hours": hours,
        "scenario": scenario,
        "capacity_mode": capacity_mode,
        "minimum_completed_tasks": minimum_completed_tasks,
        "environment_seeds": list(environment_seeds),
        "model_seeds": list(MODEL_SEEDS),
        "checkpoint_sha256": {str(seed): _sha256(path) for seed, path in zip(MODEL_SEEDS, checkpoints)},
        "environment_source_sha256": _sha256(ROOT / "agv_dt_env.py"),
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "model_parameters_updated": False,
        "authority_limits": {
            "unanimous_models_required": 3,
            "prediction_and_cooldown_horizon_sec": COOLDOWN_SEC,
            "utility_margin": FROZEN_UTILITY_MARGIN,
            "maximum_joint_action_hamming_distance": 3,
            "maximum_overrides_per_run": maximum_overrides,
            "deadlock_recovery_takeover_prohibited": True,
            "hard_safety_guard_enabled_for_both_methods": True,
        },
        "total_accepted_overrides": total_overrides,
        "total_controller_decisions": total_decisions,
        "accepted_override_rate": total_overrides / max(total_decisions, 1),
        "mean_route_blocked_time_baseline_sec": baseline_block_time,
        "mean_route_blocked_time_v147_sec": model_block_time,
        "method_means": _method_means(summaries),
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
        "claim_status": "development_only_no_confirmatory_claim",
    }
    (output_dir / f"{audit_stem}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = output_dir / f"{audit_stem}.md"
    _write_markdown(markdown, audit)
    print(markdown.read_text(encoding="utf-8"))
    return output_dir


if __name__ == "__main__":
    main(build_parser().parse_args())
