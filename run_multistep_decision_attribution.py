from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from physics_graph_world_model_multistep import PhysicsOnlyRiskPolicy
from multistep_ensemble_policy import EnsembleAgreementPolicy
from run_experiments import load_world_model, run_episode, write_csv


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "world_model_runs"
EXPECTED_TRANSITION_SCHEMA = (
    "assignment_visible_congestion_independent_arrival_streams_v4"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired multistep world-model decision-attribution experiments."
    )
    parser.add_argument(
        "--phase", choices=["smoke", "development", "confirmation"], required=True
    )
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--env-seed-start", type=int, required=True)
    parser.add_argument("--env-seed-count", type=int, required=True)
    parser.add_argument("--model-seeds", default="42")
    parser.add_argument(
        "--control-mode", choices=["single", "ensemble"], default="ensemble"
    )
    parser.add_argument("--minimum-ensemble-agreement", type=int, default=2)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--risk-gate", type=float, default=0.75)
    parser.add_argument(
        "--override-mode",
        choices=["evidence_gated", "energy_neutral_gated", "safe_argmax"],
        default="safe_argmax",
    )
    parser.add_argument("--capacity-mode", choices=["baseline", "stress"], default="stress")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_model_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--model-seeds must contain unique comma-separated integers")
    return seeds


def checkpoint_path(condition: str, seed: int) -> Path:
    directory = {
        "full": f"pi_gwm_multistep_v11_arrival_v4_full_seed{seed}",
        "data_only": f"pi_gwm_multistep_v11_arrival_v4_data_only_seed{seed}",
    }[condition]
    return MODEL_ROOT / directory / "physics_graph_world_model_multistep.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_audit(path: Path, condition: str, seed: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = dict(checkpoint.get("args", {}))
    state_dict = checkpoint["state_dict"]
    audit = {
        "condition": condition,
        "model_seed": seed,
        "path": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "model_version": checkpoint.get("model_version"),
        "parameter_count": int(sum(value.numel() for value in state_dict.values())),
        "state_shape_signature": [
            [name, list(value.shape)] for name, value in sorted(state_dict.items())
        ],
        "physical_feature_mode": args.get("physical_feature_mode"),
        "physics_weight": float(args.get("physics_weight", -1.0)),
        "data_seed": int(args.get("resolved_data_seed", args.get("data_seed", -1))),
        "split_seed": int(args.get("resolved_split_seed", args.get("split_seed", -1))),
        "training_horizon": int(args.get("training_horizon", -1)),
        "hidden_dim": int(args.get("hidden_dim", -1)),
        "transition_schema_version": args.get("transition_schema_version"),
    }
    expected_mode = "full" if condition == "full" else "zero"
    if audit["model_version"] != "pi_gwm_multistep_v11_physical_edges":
        raise ValueError(f"Unexpected model version in {path}")
    if audit["transition_schema_version"] != EXPECTED_TRANSITION_SCHEMA:
        raise ValueError(f"Checkpoint does not use the independent-arrival v4 schema: {path}")
    if audit["physical_feature_mode"] != expected_mode:
        raise ValueError(f"Unexpected physical feature mode in {path}")
    if condition == "full" and audit["physics_weight"] <= 0.0:
        raise ValueError(f"Full checkpoint has no physics loss: {path}")
    if condition == "data_only" and audit["physics_weight"] != 0.0:
        raise ValueError(f"Data-only checkpoint has a nonzero physics loss: {path}")
    return audit


def validate_checkpoint_pair(full: dict[str, Any], data_only: dict[str, Any]) -> None:
    equal_fields = (
        "parameter_count",
        "state_shape_signature",
        "data_seed",
        "split_seed",
        "training_horizon",
        "hidden_dim",
        "transition_schema_version",
    )
    mismatches = [field for field in equal_fields if full[field] != data_only[field]]
    if mismatches:
        raise ValueError(f"Unfair checkpoint pair; mismatched fields: {mismatches}")


def base_spec(method: str, policy: str, hours: float, capacity_mode: str) -> dict[str, Any]:
    return {
        "experiment": "N1_multistep_decision_attribution",
        "method": method,
        "env_variant": "full",
        "reward_mode": "hybrid",
        "scenario": "rush",
        "dispatch_rule": "dt_aware",
        "capacity_mode": capacity_mode,
        "agv_count": 3,
        "policy_override": policy,
        "fixed_time_target_h": float(hours),
        "fixed_time_target_sec": float(hours) * 3600.0,
        "max_released_jobs": None,
    }


def annotate(
    rows: list[dict[str, Any]],
    phase: str,
    model_seed: int,
    condition: str,
    checkpoint_hash: str,
) -> None:
    for row in rows:
        row["attribution_phase"] = phase
        row["model_seed"] = model_seed
        row["checkpoint_condition"] = condition
        row["checkpoint_sha256"] = checkpoint_hash


def run_method(
    *,
    spec: dict[str, Any],
    env_seed: int,
    max_steps: int,
    models: dict[str, Any],
    phase: str,
    model_seed: int,
    condition: str,
    checkpoint_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, trace, _ = run_episode(
        spec,
        episode_id=env_seed,
        seed=env_seed,
        max_steps=max_steps,
        policy=spec["policy_override"],
        models=models,
    )
    annotate([summary], phase, model_seed, condition, checkpoint_hash)
    annotate(trace, phase, model_seed, condition, checkpoint_hash)
    return summary, trace


def main(args: argparse.Namespace) -> None:
    if args.hours <= 0.0 or args.env_seed_count <= 0:
        raise ValueError("Hours and environment-seed count must be positive")
    model_seeds = parse_model_seeds(args.model_seeds)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows: list[dict[str, Any]] = []
    policies: dict[tuple[str, int], Any] = {}
    member_override_mode = (
        "safe_argmax"
        if args.control_mode == "ensemble"
        and args.override_mode in {"evidence_gated", "energy_neutral_gated"}
        else args.override_mode
    )
    for model_seed in model_seeds:
        full_path = checkpoint_path("full", model_seed)
        data_path = checkpoint_path("data_only", model_seed)
        full_audit = checkpoint_audit(full_path, "full", model_seed)
        data_audit = checkpoint_audit(data_path, "data_only", model_seed)
        validate_checkpoint_pair(full_audit, data_audit)
        checkpoint_rows.extend([full_audit, data_audit])
        policies[("full", model_seed)] = load_world_model(
            str(full_path),
            planning_horizon=args.planning_horizon,
            beam_width=args.beam_width,
            risk_gate_threshold=args.risk_gate,
            device=args.device,
        )
        policies[("full", model_seed)].override_mode = member_override_mode
        policies[("data_only", model_seed)] = load_world_model(
            str(data_path),
            planning_horizon=args.planning_horizon,
            beam_width=args.beam_width,
            risk_gate_threshold=args.risk_gate,
            device=args.device,
        )
        policies[("data_only", model_seed)].override_mode = member_override_mode

    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    env_seeds = list(
        range(args.env_seed_start, args.env_seed_start + args.env_seed_count)
    )
    for env_seed in env_seeds:
        for method, policy, models in (
            ("DT-aware", "heuristic", {}),
            (
                "PI-only guard",
                "physics_only_guarded",
                {"physics_only": PhysicsOnlyRiskPolicy(agv_count=3)},
            ),
        ):
            summary, trace = run_method(
                spec=base_spec(method, policy, args.hours, args.capacity_mode),
                env_seed=env_seed,
                max_steps=args.max_steps,
                models=models,
                phase=args.phase,
                model_seed=-1,
                condition="none",
                checkpoint_hash="none",
            )
            summaries.append(summary)
            traces.extend(trace)

        if args.control_mode == "ensemble":
            model_runs = [
                (
                    condition,
                    method,
                    -2,
                    EnsembleAgreementPolicy(
                        [policies[(condition, seed)] for seed in model_seeds],
                        minimum_agreement=args.minimum_ensemble_agreement,
                        decision_mode=(
                            "bounded_evidence"
                            if args.override_mode == "evidence_gated"
                            else (
                                "energy_neutral_bounded_evidence"
                                if args.override_mode == "energy_neutral_gated"
                                else "agreement_only"
                            )
                        ),
                    ),
                    hashlib.sha256(
                        "|".join(
                            next(
                                row["sha256"]
                                for row in checkpoint_rows
                                if row["condition"] == condition
                                and row["model_seed"] == seed
                            )
                            for seed in model_seeds
                        ).encode("ascii")
                    ).hexdigest(),
                )
                for condition, method in (
                    ("data_only", "Data-only graph MPC"),
                    ("full", "Full V11 physics-graph MPC"),
                )
            ]
        else:
            model_runs = [
                (
                    condition,
                    method,
                    model_seed,
                    policies[(condition, model_seed)],
                    next(
                        row["sha256"]
                        for row in checkpoint_rows
                        if row["condition"] == condition
                        and row["model_seed"] == model_seed
                    ),
                )
                for model_seed in model_seeds
                for condition, method in (
                    ("data_only", "Data-only graph MPC"),
                    ("full", "Full V11 physics-graph MPC"),
                )
            ]
        for condition, method, model_seed, policy_model, checkpoint_hash in model_runs:
            summary, trace = run_method(
                    spec=base_spec(
                        method, "world_model_guarded", args.hours, args.capacity_mode
                    ),
                    env_seed=env_seed,
                    max_steps=args.max_steps,
                    models={"world_model": policy_model},
                    phase=args.phase,
                    model_seed=model_seed,
                    condition=condition,
                    checkpoint_hash=checkpoint_hash,
                )
            summaries.append(summary)
            traces.extend(trace)

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "trace.csv", traces)
    manifest = {
        "protocol": "N1_multistep_decision_attribution_v2_arrival_v4",
        "transition_schema_version": EXPECTED_TRANSITION_SCHEMA,
        "phase": args.phase,
        "hours": float(args.hours),
        "env_seeds": env_seeds,
        "model_seeds": model_seeds,
        "control_mode": args.control_mode,
        "minimum_ensemble_agreement": int(args.minimum_ensemble_agreement),
        "scenario": "rush",
        "capacity_mode": args.capacity_mode,
        "planning_horizon": int(args.planning_horizon),
        "beam_width": int(args.beam_width),
        "risk_gate": float(args.risk_gate),
        "override_mode": args.override_mode,
        "device": args.device,
        "checkpoint_audit": checkpoint_rows,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Decision-attribution data saved to {output_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
