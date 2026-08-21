from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from flat_counterfactual_baseline_v150 import (
    EXPECTED_TRAINABLE_PARAMETERS,
    FlatCounterfactualBaselineV150,
    dimensions_from_sample,
    save_flat_counterfactual_baseline_v150,
)
from physics_graph_world_model import WorldModelTransitionDataset
from physics_graph_world_model_counterfactual_v141 import (
    counterfactual_loss_v141,
    counterfactual_target_statistics,
)
from train_counterfactual_world_model_v141 import (
    _evaluate,
    _grouped_split,
    _load_or_collect,
    _move_batch,
    _select_device,
    _write_history,
)


PROTOCOL = "v150_flat_mlp_trainable_budget_matched_counterfactual_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the frozen 56,457-parameter flat counterfactual baseline."
    )
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.data_seed != 14100 or args.split_seed != 14100:
        raise ValueError("V15.0 training is frozen to data/split seed 14100")
    if args.episodes != 12 or args.validation_fraction != 0.20:
        raise ValueError("V15.0 training requires 12 grouped episodes and 20% validation")
    if not args.counterfactual_cache.is_file():
        raise FileNotFoundError(
            "V15.0 must reuse the frozen V14.1 paired training cache"
        )
    if args.warmup_steps >= args.behavior_steps:
        raise ValueError("Warm-up must be shorter than the behavior trajectory")
    for name in ("epochs", "patience", "batch_size", "cpu_threads"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")


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
    model = FlatCounterfactualBaselineV150(
        **dimensions_from_sample(samples[0]),
        counterfactual_scale=scales,
        counterfactual_event_weight=event_weights,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = int(sum(parameter.numel() for parameter in trainable))
    if trainable_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("V15.0 trainable parameter budget is not frozen")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
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
                device_type=device, dtype=torch.float16, enabled=amp_enabled
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
            f"flat seed={args.seed} epoch={epoch} "
            f"train={row['train_loss']:.6f} valid={validation_loss:.6f}"
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
        raise RuntimeError("V15.0 training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    final_validation_loss, validation_report = _evaluate(model, valid_loader, device)
    checkpoint = args.output_dir / "flat_counterfactual_baseline.pt"
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    save_flat_counterfactual_baseline_v150(
        checkpoint, model, history, serializable_args
    )
    _write_history(args.output_dir / "training_history.csv", history)
    audit = {
        "protocol": PROTOCOL,
        "device": device,
        "mixed_precision": amp_enabled,
        "model_seed": args.seed,
        "cache_source": source,
        "cache_signature": cache_signature,
        "training_episode_ids": train_ids,
        "validation_episode_ids": valid_ids,
        "training_samples": len(train_samples),
        "validation_samples": len(valid_samples),
        "trainable_parameters": trainable_count,
        "v141_counterfactual_head_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "parameter_budget_exact_match": True,
        "uses_adjacency": False,
        "uses_static_physical_features": False,
        "counterfactual_scale": scales.tolist(),
        "counterfactual_event_weight": event_weights.tolist(),
        "best_validation_loss": best_loss,
        "final_validation_loss": final_validation_loss,
        "validation_report": validation_report,
    }
    (args.output_dir / "training_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"V15.0 flat baseline saved to {checkpoint}")
    return checkpoint


if __name__ == "__main__":
    train(build_parser().parse_args())

