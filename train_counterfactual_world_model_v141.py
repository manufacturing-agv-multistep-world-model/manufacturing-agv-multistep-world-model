from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    COUNTERFACTUAL_METRIC_NAMES,
    CounterfactualCollectionConfig,
    collect_counterfactual_samples,
    summarize_counterfactual_samples,
)
from physics_graph_world_model import WorldModelTransitionDataset
from physics_graph_world_model_counterfactual_v141 import (
    counterfactual_loss_v141,
    counterfactual_target_statistics,
    freeze_v141_backbone,
    initialize_v141_from_v13,
    save_counterfactual_model_v141,
)


CACHE_SCHEMA = "v141_paired_crn_material_action_fixed_physical_horizon_v2"


def _install_numpy_pickle_compatibility() -> None:
    """Read NumPy 2 caches on NumPy 1 installations without changing payloads."""

    if "numpy._core" not in sys.modules:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules["numpy._core"] = numpy_core
        sys.modules["numpy._core.multiarray"] = numpy_multiarray
        sys.modules["numpy._core.numeric"] = numpy_numeric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the V14.1 paired counterfactual long-horizon effect head."
    )
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counterfactual-cache", type=Path)
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 3:
        raise ValueError("At least three episodes are required for grouped splitting")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("Validation fraction must be in (0, 0.5)")
    if args.warmup_steps >= args.behavior_steps:
        raise ValueError("Warm-up must be shorter than the behavior trajectory")
    for name in (
        "behavior_steps",
        "sample_stride",
        "candidates_per_state",
        "max_rollout_steps",
        "epochs",
        "patience",
        "batch_size",
        "cpu_threads",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)


def _select_device(name: str, require_cuda: bool) -> str:
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA was required but is unavailable")
    if name == "auto":
        return "cuda" if cuda_available else "cpu"
    if name == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was selected but is unavailable")
    return name


def _collection_config(args: argparse.Namespace) -> CounterfactualCollectionConfig:
    return CounterfactualCollectionConfig(
        episodes=args.episodes,
        behavior_steps=args.behavior_steps,
        warmup_steps=args.warmup_steps,
        sample_stride=args.sample_stride,
        candidates_per_state=args.candidates_per_state,
        exploration_rate=args.exploration_rate,
        horizons_sec=COUNTERFACTUAL_HORIZONS_SEC,
        max_rollout_steps=args.max_rollout_steps,
        seed=args.data_seed,
    )


def _cache_signature(config: CounterfactualCollectionConfig) -> Dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA,
        "collection": asdict(config),
        "scenario": "rush",
        "capacity_mode": "stress",
        "continuation_policy": "dt_aware_after_first_action",
        "common_random_numbers": True,
        "metrics": COUNTERFACTUAL_METRIC_NAMES,
    }


def _load_or_collect(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any], str]:
    config = _collection_config(args)
    signature = _cache_signature(config)
    cache = args.counterfactual_cache
    if cache is not None and cache.is_file():
        _install_numpy_pickle_compatibility()
        with gzip.open(cache, "rb") as stream:
            payload = pickle.load(stream)
        if payload.get("signature") != signature:
            raise ValueError(
                "Counterfactual cache settings do not match this run; use a new cache"
            )
        return payload["samples"], signature, "cache"

    samples = collect_counterfactual_samples(config)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wb", compresslevel=4) as stream:
            pickle.dump(
                {"signature": signature, "samples": samples},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return samples, signature, "collected"


def _grouped_split(
    samples: Sequence[Dict[str, np.ndarray]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[int], List[int]]:
    episode_ids = sorted({int(sample["episode_id"]) for sample in samples})
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(episode_ids, dtype=np.int64)
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_ids = sorted(int(value) for value in shuffled[:validation_count])
    training_ids = sorted(int(value) for value in shuffled[validation_count:])
    if not training_ids:
        raise ValueError("Grouped split left no training episodes")
    validation_set = set(validation_ids)
    train = [
        sample for sample in samples if int(sample["episode_id"]) not in validation_set
    ]
    valid = [
        sample for sample in samples if int(sample["episode_id"]) in validation_set
    ]
    return train, valid, training_ids, validation_ids


def _move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    losses: List[float] = []
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model.forward_counterfactual(batch)
            loss, _ = counterfactual_loss_v141(output, batch)
            losses.append(float(loss.cpu()))
            predictions.append(output["counterfactual_delta"].cpu().numpy())
            targets.append(batch["target_delta"].cpu().numpy())
            masks.append(batch["target_mask"].cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks) > 0.0
    scale = model.counterfactual_scale.detach().cpu().numpy()
    component_rows = []
    for horizon_index, horizon in enumerate(model.counterfactual_horizons_sec):
        for metric_index, metric in enumerate(COUNTERFACTUAL_METRIC_NAMES):
            valid = mask[:, horizon_index, metric_index]
            error = np.abs(
                prediction[valid, horizon_index, metric_index]
                - target[valid, horizon_index, metric_index]
            )
            zero_error = np.abs(target[valid, horizon_index, metric_index])
            mae = float(np.mean(error)) if error.size else float("nan")
            zero_mae = float(np.mean(zero_error)) if zero_error.size else float("nan")
            component_rows.append(
                {
                    "horizon_sec": float(horizon),
                    "metric": metric,
                    "valid_samples": int(np.sum(valid)),
                    "mae": mae,
                    "normalized_mae": mae / float(scale[horizon_index, metric_index]),
                    "zero_baseline_mae": zero_mae,
                    "mae_improvement_over_zero": (
                        1.0 - mae / zero_mae if zero_mae > 1.0e-9 else float("nan")
                    ),
                    "nonzero_rate": float(np.mean(zero_error > 1.0e-6))
                    if zero_error.size
                    else float("nan"),
                }
            )
    return float(np.mean(losses)), {"components": component_rows}


def _write_history(path: Path, history: Sequence[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(args: argparse.Namespace) -> Path:
    _validate_args(args)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _select_device(args.device, args.require_cuda)
    amp_enabled = bool(args.amp and device == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, cache_signature, source = _load_or_collect(args)
    train_samples, valid_samples, train_ids, valid_ids = _grouped_split(
        samples, args.validation_fraction, args.split_seed
    )
    scales, event_weights = counterfactual_target_statistics(train_samples)
    model, metadata, _ = initialize_v141_from_v13(
        args.init_checkpoint,
        scales,
        event_weights,
        device=device,
    )
    freeze_v141_backbone(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = DataLoader(
        WorldModelTransitionDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    valid_loader = DataLoader(
        WorldModelTransitionDataset(valid_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    best_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model.forward_counterfactual(batch)
                loss, _ = counterfactual_loss_v141(output, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        validation_loss, _ = _evaluate(model, valid_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation_loss,
        }
        history.append(row)
        print(
            f"seed={args.seed} epoch={epoch} train={row['train_loss']:.6f} "
            f"valid={validation_loss:.6f}"
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("V14.1 training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    final_validation_loss, validation_report = _evaluate(model, valid_loader, device)
    checkpoint_path = args.output_dir / "physics_graph_world_model_counterfactual.pt"
    save_counterfactual_model_v141(
        checkpoint_path,
        model,
        metadata,
        history,
        vars(args),
        str(args.init_checkpoint),
    )
    _write_history(args.output_dir / "training_history.csv", history)
    audit = {
        "protocol": CACHE_SCHEMA,
        "device": device,
        "mixed_precision": amp_enabled,
        "model_seed": args.seed,
        "cache_source": source,
        "cache_signature": cache_signature,
        "training_episode_ids": train_ids,
        "validation_episode_ids": valid_ids,
        "training_samples": len(train_samples),
        "validation_samples": len(valid_samples),
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "counterfactual_scale": scales.tolist(),
        "counterfactual_event_weight": event_weights.tolist(),
        "dataset_summary": summarize_counterfactual_samples(samples),
        "best_validation_loss": best_loss,
        "final_validation_loss": final_validation_loss,
        "validation_report": validation_report,
    }
    (args.output_dir / "training_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"V14.1 counterfactual model saved to {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train(build_parser().parse_args())
