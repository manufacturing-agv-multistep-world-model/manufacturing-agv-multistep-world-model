from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from diagnose_counterfactual_ranking_v144 import _select_device
from guarded_counterfactual_policy_v146 import (
    FROZEN_UTILITY_MARGIN,
    GuardedCounterfactualPolicyV146,
    GuardedDTBaselinePolicy,
    V146AuthorityLimits,
)
from physics_graph_world_model_counterfactual_v141 import (
    load_counterfactual_model_v141,
)
from run_experiments import run_episode, write_csv


ROOT = Path(__file__).resolve().parent
PROTOCOL = "v146_bounded_guarded_closed_loop_development_v1"
MODEL_SEEDS = (42, 43, 44)
DEVELOPMENT_ENV_SEEDS = tuple(range(15801, 15806))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded V14.6 closed-loop development pilot.")
    parser.add_argument("--phase", choices=("smoke", "development"), required=True)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--env-seed-start", type=int, default=15801)
    parser.add_argument("--env-seed-count", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2200)
    parser.add_argument("--cooldown-sec", type=float, default=60.0)
    parser.add_argument("--maximum-overrides", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(seed: int) -> Path:
    return (
        ROOT
        / "world_model_runs"
        / f"pi_gwm_counterfactual_v141_seed{seed}"
        / "physics_graph_world_model_counterfactual.pt"
    )


def _base_spec(
    method: str,
    hours: float,
    experiment: str = "N8_guarded_closed_loop_pilot",
) -> Dict[str, Any]:
    return {
        "experiment": experiment,
        "method": method,
        "env_variant": "full",
        "reward_mode": "hybrid",
        "scenario": "rush",
        "dispatch_rule": "dt_aware",
        "capacity_mode": "stress",
        "agv_count": 3,
        "policy_override": "world_model_guarded",
        "fixed_time_target_h": float(hours),
        "fixed_time_target_sec": float(hours) * 3600.0,
        "max_released_jobs": None,
    }


def _run(
    method: str,
    policy: Any,
    env_seed: int,
    hours: float,
    max_steps: int,
    phase: str,
    protocol: str = PROTOCOL,
    experiment: str = "N8_guarded_closed_loop_pilot",
    spec_overrides: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spec = _base_spec(method, hours, experiment=experiment)
    spec.update(spec_overrides or {})
    summary, trace, _ = run_episode(
        spec,
        episode_id=env_seed,
        seed=env_seed,
        max_steps=max_steps,
        policy="world_model_guarded",
        models={"world_model": policy},
    )
    for row in [summary, *trace]:
        row["v146_protocol"] = protocol
        row["v146_phase"] = phase
    summary["v146_decision_steps"] = len(trace)
    summary["v146_shadow_recommendation_count"] = int(
        sum(row["v146_shadow_recommended"] > 0.5 for row in trace)
    )
    summary["v146_accepted_override_count"] = int(
        sum(row["world_model_override_accepted"] > 0.5 for row in trace)
    )
    summary["v146_accepted_override_rate"] = float(
        summary["v146_accepted_override_count"] / max(len(trace), 1)
    )
    return summary, trace


def _method_means(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metrics = (
        "uph",
        "energy_efficiency_wh_per_sku",
        "avg_task_wait_time",
        "conflict_count",
        "blocking_onset_count",
        "route_blocking_onset_count",
        "route_blocked_time_sec",
        "deadlock_count",
        "empty_running_ratio",
        "p95_decision_compute_sec",
    )
    output: Dict[str, Dict[str, float]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        output[method] = {
            metric: float(np.mean([float(row[metric]) for row in subset]))
            for metric in metrics
        }
    return output


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    methods = audit["method_means"]
    limits = audit.get("authority_limits", {})
    cooldown_sec = float(
        limits.get(
            "prediction_and_cooldown_horizon_sec",
            limits.get("cooldown_sec", 60.0),
        )
    )
    lines = [
        f"# {audit.get('model_variant', 'V14.6')} bounded guarded closed-loop development audit",
        "",
        (
            f"Phase: {audit['phase']}; physical horizon: {audit['hours']:.3f} h; "
            f"environment seeds: {audit['environment_seeds']}."
        ),
        "",
        "Three frozen V14.1 models may execute one complete joint AGV action only when their choice is unanimous, "
        f"the normalized 720-second utility gain is at least {FROZEN_UTILITY_MARGIN:.2f}, no hard "
        f"safety action is active, the {cooldown_sec:.0f}-second cooldown has elapsed, and the per-run authority "
        "budget is not exhausted. All other decisions execute the guarded DT-aware fallback.",
        "",
        "| Method | UPH | EER | Wait (s) | Conflicts | Blocking | Deadlocks | P95 decision (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in methods.items():
        lines.append(
            f"| {method} | {values['uph']:.3f} | "
            f"{values['energy_efficiency_wh_per_sku']:.3f} | "
            f"{values['avg_task_wait_time']:.2f} | {values['conflict_count']:.3f} | "
            f"{values['blocking_onset_count']:.3f} | {values['deadlock_count']:.3f} | "
            f"{values['p95_decision_compute_sec']:.4f} |"
        )
    lines.extend(["", "## Development continuation criteria", ""])
    lines.extend(
        f"- [{'x' if item['passed'] else ' '}] {item['criterion']}"
        for item in audit["criteria"]
    )
    lines.extend(
        [
            "",
            "Proceed to a separately frozen confirmation protocol: "
            f"**{'YES' if audit['passed'] else 'NO'}**.",
            "",
            "This is development evidence only and must not be reported as a confirmatory "
            "closed-loop performance result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> Path:
    if args.hours <= 0.0 or args.env_seed_count < 1 or args.max_steps < 1:
        raise ValueError("Physical horizon, seed count, and step cap must be positive")
    if args.phase == "development":
        frozen = (args.hours, args.env_seed_start, args.env_seed_count, args.cooldown_sec, args.maximum_overrides)
        expected = (1.0, 15801, 5, 60.0, 12)
        if frozen != expected:
            raise ValueError(f"V14.6 development protocol is frozen at {expected}, received {frozen}")
    device = _select_device(args.device, args.require_cuda)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [_checkpoint_path(seed) for seed in MODEL_SEEDS]
    for checkpoint in checkpoint_paths:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    models = [load_counterfactual_model_v141(path, device=device) for path in checkpoint_paths]
    limits = V146AuthorityLimits(
        utility_margin=FROZEN_UTILITY_MARGIN,
        cooldown_sec=args.cooldown_sec,
        maximum_overrides=args.maximum_overrides,
        maximum_action_hamming_distance=3,
    )

    summaries: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    env_seeds = list(range(args.env_seed_start, args.env_seed_start + args.env_seed_count))
    for env_seed in env_seeds:
        baseline_summary, baseline_trace = _run(
            "Guarded DT-aware baseline",
            GuardedDTBaselinePolicy(),
            env_seed,
            args.hours,
            args.max_steps,
            args.phase,
        )
        controller_summary, controller_trace = _run(
            "V14.6 bounded world-model guard",
            GuardedCounterfactualPolicyV146(models, limits),
            env_seed,
            args.hours,
            args.max_steps,
            args.phase,
        )
        summaries.extend([baseline_summary, controller_summary])
        traces.extend([*baseline_trace, *controller_trace])

    write_csv(output_dir / "episode_summary.csv", summaries)
    write_csv(output_dir / "decision_trace.csv", traces)
    by_seed = {
        seed: {row["method"]: row for row in summaries if int(row["seed"]) == seed}
        for seed in env_seeds
    }
    pairs = [
        (
            rows["Guarded DT-aware baseline"],
            rows["V14.6 bounded world-model guard"],
        )
        for rows in by_seed.values()
    ]
    total_overrides = int(sum(pair[1]["v146_accepted_override_count"] for pair in pairs))
    total_steps = int(sum(pair[1]["v146_decision_steps"] for pair in pairs))
    criteria = [
        {
            "criterion": "Every paired run reaches the fixed physical horizon",
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
            "criterion": "No V14.6 run has an out-of-battery event or timeout",
            "passed": all(right["out_of_battery_rate"] == 0.0 and right["timeout_rate"] == 0.0 for _, right in pairs),
        },
        {
            "criterion": "V14.6 introduces no additional deadlocks in any paired run",
            "passed": all(right["deadlock_count"] <= left["deadlock_count"] for left, right in pairs),
        },
        {
            "criterion": "Mean V14.6 route-blocking onsets do not exceed the guarded baseline",
            "passed": float(np.mean([right["route_blocking_onset_count"] for left, right in pairs]))
            <= float(np.mean([left["route_blocking_onset_count"] for left, right in pairs])),
        },
        {
            "criterion": "Mean V14.6 conflict events are no more than 105% of the guarded baseline",
            "passed": float(np.mean([right["conflict_count"] for left, right in pairs]))
            <= 1.05 * float(np.mean([left["conflict_count"] for left, right in pairs])),
        },
        {
            "criterion": "V14.6 UPH is at least 95% of the guarded baseline in every run",
            "passed": all(right["uph"] >= 0.95 * left["uph"] for left, right in pairs),
        },
        {
            "criterion": "V14.6 EER is no more than 105% of the guarded baseline in every run",
            "passed": all(
                np.isfinite(right["energy_efficiency_wh_per_sku"])
                and right["energy_efficiency_wh_per_sku"] <= 1.05 * left["energy_efficiency_wh_per_sku"]
                for left, right in pairs
            ),
        },
        {
            "criterion": "Bounded model authority is active but below 20% of decisions",
            "passed": total_overrides >= len(env_seeds) and 0.0 < total_overrides / max(total_steps, 1) < 0.20,
        },
        {
            "criterion": "Every V14.6 run respects the frozen override budget",
            "passed": all(right["v146_accepted_override_count"] <= args.maximum_overrides for _, right in pairs),
        },
        {
            "criterion": "Mean V14.6 P95 decision time is at most 2 seconds",
            "passed": float(np.mean([right["p95_decision_compute_sec"] for _, right in pairs])) <= 2.0,
        },
    ]
    audit = {
        "protocol": PROTOCOL,
        "phase": args.phase,
        "hours": args.hours,
        "environment_seeds": env_seeds,
        "model_seeds": list(MODEL_SEEDS),
        "checkpoint_sha256": {str(seed): _sha256(path) for seed, path in zip(MODEL_SEEDS, checkpoint_paths)},
        "model_parameters_updated": False,
        "authority_limits": {
            "unanimous_models_required": 3,
            "terminal_horizon_sec": 720.0,
            "utility_margin": FROZEN_UTILITY_MARGIN,
            "maximum_action_hamming_distance": 3,
            "cooldown_sec": args.cooldown_sec,
            "maximum_overrides_per_run": args.maximum_overrides,
            "hard_safety_guard_enabled_for_both_methods": True,
            "automatic_permanent_fallback_after_post_override_deadlock": True,
        },
        "total_accepted_overrides": total_overrides,
        "total_controller_decisions": total_steps,
        "accepted_override_rate": total_overrides / max(total_steps, 1),
        "method_means": _method_means(summaries),
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
        "claim_status": "development_only_no_confirmatory_claim",
    }
    json_path = output_dir / "V146_DEVELOPMENT_AUDIT.json"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output_dir / "V146_DEVELOPMENT_AUDIT.md"
    _write_markdown(markdown_path, audit)
    print(markdown_path.read_text(encoding="utf-8"))
    return output_dir


if __name__ == "__main__":
    main(build_parser().parse_args())
