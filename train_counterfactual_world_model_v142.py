from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from counterfactual_rollout_v141 import COUNTERFACTUAL_METRIC_NAMES
from physics_graph_world_model import WorldModelTransitionDataset
from physics_graph_world_model_counterfactual_v142 import (
    MATERIAL_EFFECT_FRACTION,
    TERMINAL_UTILITY_WEIGHTS,
    counterfactual_loss_v142,
    freeze_v142_backbone,
    gate_positive_weights,
    initialize_v142_from_v141,
    save_counterfactual_model_v142,
)
from train_counterfactual_world_model_v141 import (
    _grouped_split,
    _install_numpy_pickle_compatibility,
    _load_or_collect,
    _move_batch,
    _select_device,
)


PROTOCOL = "v142_zero_inflated_rank_aware_training_v1"


class StateGroupedBatchSampler(Sampler[List[int]]):
    """Keep every candidate from a decision state in the same mini-batch."""

    def __init__(
        self,
        samples: Sequence[Dict[str, np.ndarray]],
        batch_size: int,
        seed: int,
        shuffle: bool,
    ):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        groups: Dict[Tuple[int, int], List[int]] = {}
        for index, sample in enumerate(samples):
            key = (int(sample["episode_id"]), int(sample["state_id"]))
            groups.setdefault(key, []).append(index)
        self.groups = list(groups.values())
        if any(len(group) > batch_size for group in self.groups):
            raise ValueError("Batch size cannot split a decision-state candidate group")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        order = np.arange(len(self.groups))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        batch: List[int] = []
        for group_index in order:
            group = self.groups[int(group_index)]
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch

    def __len__(self) -> int:
        batches = 0
        used = 0
        for group in self.groups:
            if used and used + len(group) > self.batch_size:
                batches += 1
                used = 0
            used += len(group)
        return batches + int(used > 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train V14.2 zero-inflated, rank-aware physical effects."
    )
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counterfactual-cache", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--behavior-steps", type=int, default=4000)
    parser.add_argument("--warmup-steps", type=int, default=1200)
    parser.add_argument("--sample-stride", type=int, default=60)
    parser.add_argument("--candidates-per-state", type=int, default=3)
    parser.add_argument("--max-rollout-steps", type=int, default=500)
    parser.add_argument("--exploration-rate", type=float, default=0.35)
    parser.add_argument("--data-seed", type=int, default=14100)
    parser.add_argument("--split-seed", type=int, default=14100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    if not args.counterfactual_cache.is_file():
        raise FileNotFoundError(args.counterfactual_cache)
    if args.episodes < 3 or not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("V14.2 requires grouped training and validation episodes")
    if args.warmup_steps >= args.behavior_steps:
        raise ValueError("Warm-up must be shorter than the behavior trajectory")


def _ranking_report(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    samples: Sequence[Dict[str, np.ndarray]],
    scale: np.ndarray,
) -> Dict[str, float]:
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, sample in enumerate(samples):
        if bool(np.all(mask[index, -1])):
            key = (int(sample["episode_id"]), int(sample["state_id"]))
            groups.setdefault(key, []).append(index)
    utility_weights = np.asarray(TERMINAL_UTILITY_WEIGHTS)
    regrets = []
    baseline_regrets = []
    agreement = []
    for indices in groups.values():
        true_scores = np.sum(
            target[indices, -1] / scale[-1] * utility_weights, axis=1
        )
        predicted_scores = np.sum(
            prediction[indices, -1] / scale[-1] * utility_weights, axis=1
        )
        true_values = np.concatenate([[0.0], true_scores])
        predicted_values = np.concatenate([[0.0], predicted_scores])
        true_best = int(np.argmax(true_values))
        predicted_best = int(np.argmax(predicted_values))
        regrets.append(float(true_values[true_best] - true_values[predicted_best]))
        baseline_regrets.append(float(true_values[true_best]))
        agreement.append(float(true_best == predicted_best))
    mean_regret = float(np.mean(regrets))
    baseline_regret = float(np.mean(baseline_regrets))
    return {
        "decision_states": float(len(groups)),
        "top1_agreement": float(np.mean(agreement)),
        "mean_regret": mean_regret,
        "baseline_mean_regret": baseline_regret,
        "regret_reduction": (
            1.0 - mean_regret / baseline_regret
            if baseline_regret > 1.0e-12
            else float("nan")
        ),
    }


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    samples: Sequence[Dict[str, np.ndarray]],
    device: str,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    losses = []
    predictions = []
    soft_predictions = []
    targets = []
    masks = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model.forward_counterfactual(batch)
            loss, _ = counterfactual_loss_v142(output, batch)
            losses.append(float(loss.cpu()))
            predictions.append(output["hard_counterfactual_delta"].cpu().numpy())
            soft_predictions.append(output["counterfactual_delta"].cpu().numpy())
            targets.append(batch["target_delta"].cpu().numpy())
            masks.append(batch["target_mask"].cpu().numpy())
    prediction = np.concatenate(predictions)
    soft_prediction = np.concatenate(soft_predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks) > 0.0
    scale = model.counterfactual_scale.detach().cpu().numpy()
    components = []
    for horizon_index, horizon in enumerate(model.counterfactual_horizons_sec):
        for metric_index, metric in enumerate(COUNTERFACTUAL_METRIC_NAMES):
            valid = mask[:, horizon_index, metric_index]
            truth = target[valid, horizon_index, metric_index]
            estimate = prediction[valid, horizon_index, metric_index]
            soft_estimate = soft_prediction[valid, horizon_index, metric_index]
            error = np.abs(estimate - truth)
            zero_error = np.abs(truth)
            material = np.abs(truth) >= (
                MATERIAL_EFFECT_FRACTION * scale[horizon_index, metric_index]
            )
            components.append(
                {
                    "horizon_sec": float(horizon),
                    "metric": metric,
                    "samples": int(truth.size),
                    "mae": float(np.mean(error)),
                    "soft_mae": float(np.mean(np.abs(soft_estimate - truth))),
                    "zero_mae": float(np.mean(zero_error)),
                    "mae_gain_over_zero": float(np.mean(zero_error - error)),
                    "nonzero_prediction_rate": float(np.mean(np.abs(estimate) > 1.0e-9)),
                    "material_sign_accuracy": (
                        float(
                            np.mean(
                                np.sign(estimate[material])
                                == np.sign(truth[material])
                            )
                        )
                        if np.any(material)
                        else float("nan")
                    ),
                }
            )
    return float(np.mean(losses)), {
        "components": components,
        "ranking": _ranking_report(prediction, target, mask, samples, scale),
    }


def train(args: argparse.Namespace) -> Path:
    _validate_args(args)
    _install_numpy_pickle_compatibility()
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _select_device(args.device, args.require_cuda)
    amp_enabled = bool(args.amp and device == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, cache_signature, cache_source = _load_or_collect(args)
    train_samples, valid_samples, train_ids, valid_ids = _grouped_split(
        samples, args.validation_fraction, args.split_seed
    )
    initialization = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    scales = np.asarray(initialization["counterfactual_scale"], dtype=np.float32)
    gate_weights = gate_positive_weights(train_samples, scales)
    model, metadata, _ = initialize_v142_from_v141(
        args.init_checkpoint, gate_weights, device=device
    )
    freeze_v142_backbone(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = DataLoader(
        WorldModelTransitionDataset(train_samples),
        batch_sampler=StateGroupedBatchSampler(
            train_samples, args.batch_size, args.seed, shuffle=True
        ),
        num_workers=0,
        pin_memory=device == "cuda",
    )
    valid_loader = DataLoader(
        WorldModelTransitionDataset(valid_samples),
        batch_sampler=StateGroupedBatchSampler(
            valid_samples, args.batch_size, args.seed, shuffle=False
        ),
        num_workers=0,
        pin_memory=device == "cuda",
    )

    best_loss = float("inf")
    best_state = None
    stale = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=amp_enabled
            ):
                output = model.forward_counterfactual(batch)
                loss, parts = counterfactual_loss_v142(output, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        validation_loss, validation = _evaluate(
            model, valid_loader, valid_samples, device
        )
        row = {
            "epoch": float(epoch),
            **{f"train_{key}": value / batches for key, value in totals.items()},
            "validation_loss": validation_loss,
            "validation_regret_reduction": validation["ranking"]["regret_reduction"],
        }
        history.append(row)
        print(
            f"seed={args.seed} epoch={epoch} train={row['train_loss']:.6f} "
            f"valid={validation_loss:.6f} "
            f"valid_regret_reduction={row['validation_regret_reduction']:.4f}"
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("V14.2 training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    final_loss, validation = _evaluate(model, valid_loader, valid_samples, device)
    checkpoint = args.output_dir / "physics_graph_world_model_counterfactual_v142.pt"
    save_counterfactual_model_v142(
        checkpoint,
        model,
        metadata,
        history,
        vars(args),
        str(args.init_checkpoint),
    )
    with (args.output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    audit = {
        "protocol": PROTOCOL,
        "model_seed": args.seed,
        "device": device,
        "mixed_precision": amp_enabled,
        "cache_source": cache_source,
        "cache_signature": cache_signature,
        "training_episode_ids": train_ids,
        "validation_episode_ids": valid_ids,
        "training_samples": len(train_samples),
        "validation_samples": len(valid_samples),
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "gate_positive_weights": gate_weights.tolist(),
        "best_validation_loss": best_loss,
        "final_validation_loss": final_loss,
        "validation_report": validation,
    }
    (args.output_dir / "training_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"V14.2 model saved to {checkpoint}")
    return checkpoint


if __name__ == "__main__":
    train(build_parser().parse_args())
