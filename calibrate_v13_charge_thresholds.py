from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from agv_case_env import AGV_A_Charge_Env
from diagnose_world_model_multistep import binary_average_precision, binary_roc_auc
from physics_graph_world_model import collect_world_model_transitions
from physics_graph_world_model_multistep import MultiStepSequenceDataset, build_sequence_samples
from physics_graph_world_model_multistep_v13 import (
    annotate_future_congestion_risk,
    load_multistep_world_model_policy_v13,
)
from train_world_model_multistep import move_batch


ROOT = Path(__file__).resolve().parent
MODEL_SEEDS = (42, 43, 44)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate V13 charge-risk thresholds on dedicated trajectories."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--sequence-stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=22313)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def classification_metrics(
    actual: np.ndarray, score: np.ndarray, threshold: float
) -> Dict[str, float]:
    predicted = score >= threshold
    true_positive = int(np.sum(actual & predicted))
    positives = int(np.sum(actual))
    precision = true_positive / max(int(np.sum(predicted)), 1)
    recall = true_positive / max(positives, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def choose_threshold(actual: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    candidates: List[Dict[str, float]] = []
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = classification_metrics(actual, score, float(threshold))
        if metrics["precision"] >= 0.55 and metrics["recall"] >= 0.65:
            candidates.append({"threshold": float(threshold), **metrics})
    if not candidates:
        raise RuntimeError(
            "No calibration threshold satisfies precision >= 0.55 and recall >= 0.65"
        )
    return max(
        candidates,
        key=lambda row: (row["f1"], row["recall"], -row["threshold"]),
    )


def collect_probabilities(policy, loader: DataLoader, device: str):
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    policy.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = policy.model.rollout(batch, teacher_forcing_ratio=0.0)
            probability = torch.sigmoid(
                output["pred_future_congestion_risk_logits"][:, 0, 0]
            )
            probabilities.append(probability.detach().cpu().numpy())
            targets.append(
                batch["target_future_congestion_risk"][:, 0, 0].detach().cpu().numpy()
            )
            masks.append(
                batch["target_future_congestion_risk_mask"][:, 0, 0]
                .detach()
                .cpu()
                .numpy()
            )
    probability = np.concatenate(probabilities)
    target = np.concatenate(targets) > 0.5
    valid = np.concatenate(masks) > 0.5
    return target[valid], probability[valid]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration requested but unavailable")
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

    transitions = collect_world_model_transitions(
        env_factory=env_factory,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        exploration_rate=0.35,
    )
    transitions = annotate_future_congestion_risk(transitions, horizon=80)
    sequences = build_sequence_samples(
        transitions, horizon=10, stride=args.sequence_stride
    )
    loader = DataLoader(
        MultiStepSequenceDataset(sequences),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_results: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, object]] = []
    for model_seed in MODEL_SEEDS:
        model_path = (
            ROOT
            / "world_model_runs"
            / f"pi_gwm_multistep_v13_multiscale_v2_seed{model_seed}"
            / "physics_graph_world_model_multistep.pt"
        )
        policy = load_multistep_world_model_policy_v13(model_path, device=args.device)
        actual, score = collect_probabilities(policy, loader, args.device)
        selected = choose_threshold(actual, score)
        row = {
            "model_seed": model_seed,
            "threshold": selected["threshold"],
            "precision": selected["precision"],
            "recall": selected["recall"],
            "f1": selected["f1"],
            "event_prevalence": float(np.mean(actual)),
            "event_count": int(np.sum(actual)),
            "sample_count": int(actual.size),
            "roc_auc": binary_roc_auc(actual, score),
            "average_precision": binary_average_precision(actual, score),
        }
        rows.append(row)
        model_results[str(model_seed)] = {
            key: float(value) for key, value in row.items() if key != "model_seed"
        }

    write_csv(output_dir / "threshold_calibration.csv", rows)
    payload = {
        "protocol": "dedicated_calibration_seed_without_model_updates",
        "calibration_seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "future_risk_horizon": 80,
        "threshold_grid": {"minimum": 0.05, "maximum": 0.95, "step": 0.005},
        "selection_constraints": {"precision_min": 0.55, "recall_min": 0.65},
        "models": model_results,
    }
    (output_dir / "calibrated_thresholds.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
