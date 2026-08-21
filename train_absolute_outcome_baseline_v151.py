from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    COUNTERFACTUAL_METRIC_NAMES,
    summarize_counterfactual_samples,
)
from physics_graph_world_model import WorldModelTransitionDataset
from physics_graph_world_model_absolute_v151 import (
    absolute_outcome_loss_v151,
    absolute_outcome_target_statistics,
    initialize_absolute_v151_from_v13,
    save_absolute_model_v151,
)
from physics_graph_world_model_counterfactual_v141 import freeze_v141_backbone
from train_counterfactual_world_model_v141 import (
    _grouped_split,
    _load_or_collect,
    _move_batch,
    _select_device,
    _validate_args,
    build_parser as build_v141_parser,
)


PROTOCOL = "v151_absolute_outcome_then_difference_train_only_v1"


def build_parser():
    parser = build_v141_parser()
    parser.description = (
        "Train the V15.1 absolute-outcome-then-difference formulation baseline."
    )
    return parser


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
            absolute = model.forward_absolute_outcomes(batch)
            loss, _ = absolute_outcome_loss_v151(absolute, batch)
            losses.append(float(loss.detach().cpu()))
            predictions.append(
                (
                    absolute["candidate_outcomes"]
                    - absolute["baseline_outcomes"]
                ).detach().cpu().numpy()
            )
            targets.append(batch["target_delta"].detach().cpu().numpy())
            masks.append(batch["target_mask"].detach().cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks) > 0.0
    rows = []
    for horizon_index, horizon in enumerate(COUNTERFACTUAL_HORIZONS_SEC):
        for metric_index, metric in enumerate(COUNTERFACTUAL_METRIC_NAMES):
            valid = mask[:, horizon_index, metric_index]
            error = np.abs(
                prediction[valid, horizon_index, metric_index]
                - target[valid, horizon_index, metric_index]
            )
            zero_error = np.abs(target[valid, horizon_index, metric_index])
            mae = float(np.mean(error)) if error.size else float("nan")
            zero_mae = float(np.mean(zero_error)) if zero_error.size else float("nan")
            rows.append(
                {
                    "horizon_sec": float(horizon),
                    "metric": metric,
                    "valid_samples": int(np.sum(valid)),
                    "inferred_delta_mae": mae,
                    "zero_delta_mae": zero_mae,
                    "delta_mae_improvement_over_zero": (
                        1.0 - mae / zero_mae
                        if zero_mae > 1.0e-9
                        else float("nan")
                    ),
                }
            )
    return float(np.mean(losses)), {"inferred_delta_components": rows}


def _write_history(path: Path, history: Sequence[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(args) -> Path:
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
    scales, event_weights = absolute_outcome_target_statistics(train_samples)
    model, metadata, _ = initialize_absolute_v151_from_v13(
        args.init_checkpoint,
        scales,
        event_weights,
        device=device,
    )
    freeze_v141_backbone(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = int(sum(parameter.numel() for parameter in trainable))
    if trainable_count != 56_457:
        raise RuntimeError(
            f"V15.1 must match the 56,457-parameter paired head, got {trainable_count}"
        )
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
                output = model.forward_absolute_outcomes(batch)
                loss, _ = absolute_outcome_loss_v151(output, batch)
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
        raise RuntimeError("V15.1 training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    final_validation_loss, validation_report = _evaluate(
        model, valid_loader, device
    )
    checkpoint_path = args.output_dir / "absolute_outcome_graph_baseline.pt"
    save_absolute_model_v151(
        checkpoint_path,
        model,
        metadata,
        history,
        vars(args),
        str(args.init_checkpoint),
    )
    _write_history(args.output_dir / "training_history.csv", history)
    audit = {
        "protocol": PROTOCOL,
        "scientific_role": "absolute_outcome_then_difference_formulation_baseline",
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
        "outcome_scale": scales.tolist(),
        "outcome_event_weight": event_weights.tolist(),
        "dataset_summary": summarize_counterfactual_samples(samples),
        "best_validation_loss": best_loss,
        "final_validation_loss": final_validation_loss,
        "validation_report": validation_report,
    }
    (args.output_dir / "training_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"V15.1 absolute-outcome model saved to {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train(build_parser().parse_args())
