from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from agv_case_env import AGV_A_Charge_Env
from diagnose_world_model_multistep import collect_predictions, write_csv
from physics_graph_world_model import (
    KPI_NAMES,
    collect_world_model_transitions,
    kpi_scale,
)
from physics_graph_world_model_multistep import (
    MultiStepSequenceDataset,
    build_sequence_samples,
)
from physics_graph_world_model_multistep_v11 import (
    MODEL_VERSION,
    load_multistep_world_model_policy_v11,
)


ROOT = Path(__file__).resolve().parent
MODEL_SEEDS = (42, 43, 44)
HORIZONS = (1, 3, 5, 10)
PRIMARY_METRICS = ("delta_time_sec", "delta_energy_wh", "blocked_delta")
EXPECTED_TRANSITION_SCHEMA = (
    "assignment_visible_congestion_independent_arrival_streams_v4"
)
CONDITIONS: Dict[str, Tuple[str, str, float]] = {
    "Full": ("pi_gwm_multistep_v11_arrival_v4_full", "full", 0.5),
    "No physics loss": (
        "pi_gwm_multistep_v11_arrival_v4_no_physics_loss",
        "full",
        0.0,
    ),
    "No physical features": (
        "pi_gwm_multistep_v11_arrival_v4_no_physical_features",
        "zero",
        0.5,
    ),
    "Data-only graph": (
        "pi_gwm_multistep_v11_arrival_v4_data_only",
        "zero",
        0.0,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired independent evaluation of the V11 physics factorial."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--sequence-stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=27413)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(
    checkpoint: Dict[str, object], expected_mode: str, expected_weight: float
) -> None:
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("Factorial evaluation requires V11 checkpoints")
    args = checkpoint.get("args", {})
    if args.get("transition_schema_version") != EXPECTED_TRANSITION_SCHEMA:
        raise ValueError("Checkpoint does not use the independent-arrival v4 schema")
    if args.get("physical_feature_mode") != expected_mode:
        raise ValueError("Checkpoint physical-feature mode does not match condition")
    if abs(float(args.get("physics_weight", -1.0)) - expected_weight) > 1.0e-12:
        raise ValueError("Checkpoint physics weight does not match condition")
    node_features = checkpoint["node_physical_features"]
    edge_features = checkpoint["edge_physical_features"]
    is_zero = bool(
        int(torch.count_nonzero(node_features)) == 0
        and int(torch.count_nonzero(edge_features)) == 0
    )
    if is_zero != (expected_mode == "zero"):
        raise ValueError("Checkpoint physical buffers do not match declared mode")


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def env_factory(seed: int) -> AGV_A_Charge_Env:
        return AGV_A_Charge_Env(
            agv_count=3,
            env_variant="full",
            reward_mode="hybrid",
            scenario="rush",
            dispatch_rule="dt_aware",
            capacity_mode="stress",
            max_steps=args.max_steps,
            seed=seed,
        )

    print("Collecting one shared set of independent test trajectories...")
    transitions = collect_world_model_transitions(
        env_factory=env_factory,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        exploration_rate=0.25,
    )
    sequences = build_sequence_samples(
        transitions, horizon=max(HORIZONS), stride=args.sequence_stride
    )
    loader = DataLoader(
        MultiStepSequenceDataset(sequences),
        batch_size=args.batch_size,
        shuffle=False,
    )
    episode_ids = np.asarray([int(row["episode_id"]) for row in sequences])
    transition_ids = np.asarray(
        [int(row["start_transition_id"]) for row in sequences]
    )
    scale = kpi_scale(3)
    metric_indices = {name: KPI_NAMES.index(name) for name in PRIMARY_METRICS}
    prediction_rows: List[Dict[str, object]] = []
    position_rows: List[Dict[str, object]] = []
    checkpoint_rows: List[Dict[str, object]] = []
    reference_targets = None
    reference_positions = None

    for condition, (stem, expected_mode, expected_weight) in CONDITIONS.items():
        for model_seed in MODEL_SEEDS:
            checkpoint_path = (
                ROOT
                / "world_model_runs"
                / f"{stem}_seed{model_seed}"
                / "physics_graph_world_model_multistep.pt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            validate_checkpoint(checkpoint, expected_mode, expected_weight)
            policy = load_multistep_world_model_policy_v11(
                checkpoint_path, device=args.device
            )
            parameter_count = sum(
                parameter.numel() for parameter in policy.model.parameters()
            )
            data = collect_predictions(policy, loader, args.device)
            actual_kpi = data["target_kpi"] * scale
            predicted_kpi = data["pred_kpi"] * scale
            target_positions = data["target_agent_features"][:, :, :, 0]
            predicted_positions = data["pred_agent_features"][:, :, :, 0]
            if reference_targets is None:
                reference_targets = actual_kpi.copy()
                reference_positions = target_positions.copy()
            elif not (
                np.array_equal(reference_targets, actual_kpi)
                and np.array_equal(reference_positions, target_positions)
            ):
                raise RuntimeError("Models did not receive identical target trajectories")

            node_count = int(data["target_node_features"].shape[2])
            for horizon in HORIZONS:
                step = horizon - 1
                for metric, metric_index in metric_indices.items():
                    actual = actual_kpi[:, step, metric_index]
                    predicted = predicted_kpi[:, step, metric_index]
                    for index in range(len(sequences)):
                        prediction_rows.append(
                            {
                                "condition": condition,
                                "model_seed": model_seed,
                                "episode_id": int(episode_ids[index]),
                                "start_transition_id": int(transition_ids[index]),
                                "horizon_steps": horizon,
                                "metric": metric,
                                "actual": float(actual[index]),
                                "predicted": float(predicted[index]),
                                "absolute_error": float(
                                    abs(predicted[index] - actual[index])
                                ),
                            }
                        )
                actual_nodes = np.rint(
                    target_positions[:, step] * max(node_count - 1, 1)
                ).astype(np.int64)
                predicted_nodes = np.clip(
                    np.rint(predicted_positions[:, step] * max(node_count - 1, 1)),
                    0,
                    node_count - 1,
                ).astype(np.int64)
                correct = np.mean(predicted_nodes == actual_nodes, axis=1)
                for index in range(len(sequences)):
                    position_rows.append(
                        {
                            "condition": condition,
                            "model_seed": model_seed,
                            "episode_id": int(episode_ids[index]),
                            "start_transition_id": int(transition_ids[index]),
                            "horizon_steps": horizon,
                            "position_node_accuracy": float(correct[index]),
                        }
                    )
            checkpoint_rows.append(
                {
                    "condition": condition,
                    "model_seed": model_seed,
                    "physical_feature_mode": expected_mode,
                    "physics_weight": expected_weight,
                    "transition_schema_version": checkpoint["args"][
                        "transition_schema_version"
                    ],
                    "data_seed": checkpoint["args"]["resolved_data_seed"],
                    "split_seed": checkpoint["args"]["resolved_split_seed"],
                    "parameter_count": parameter_count,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "sha256": checkpoint_sha256(checkpoint_path),
                }
            )
            print(f"Evaluated {condition}, seed {model_seed}.")

    parameter_counts = {int(row["parameter_count"]) for row in checkpoint_rows}
    if len(parameter_counts) != 1:
        raise RuntimeError("Factorial checkpoints do not have equal parameter counts")
    write_csv(output_dir / "paired_physical_predictions.csv", prediction_rows)
    write_csv(output_dir / "paired_position_predictions.csv", position_rows)
    write_csv(output_dir / "checkpoint_audit.csv", checkpoint_rows)
    manifest = {
        "protocol": "v11_arrival_v4_physics_factorial_independent_open_loop",
        "transition_schema_version": EXPECTED_TRANSITION_SCHEMA,
        "test_seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "sequence_stride": args.sequence_stride,
        "horizons": list(HORIZONS),
        "primary_metrics": list(PRIMARY_METRICS),
        "transition_count": len(transitions),
        "sequence_count": len(sequences),
        "model_seeds": list(MODEL_SEEDS),
        "conditions": list(CONDITIONS),
        "parameter_count": parameter_counts.pop(),
        "device": args.device,
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"V11 factorial predictions saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
