from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    COUNTERFACTUAL_METRIC_NAMES,
)
from physics_graph_world_model import WorldModelMetadata
from physics_graph_world_model_counterfactual_v141 import (
    PhysicsGraphWorldModelCounterfactualV141,
)
from physics_graph_world_model_multistep_v13 import MODEL_VERSION as V13_MODEL_VERSION


MODEL_VERSION = "pi_gwm_v151_absolute_outcome_then_difference_v1"


def absolute_outcome_target_statistics(
    samples: Sequence[Dict[str, np.ndarray]],
    quantile: float = 0.75,
    maximum_event_weight: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive normalization and sparse-outcome weights from training episodes only."""

    if not samples:
        raise ValueError("Absolute-outcome statistics require training samples")
    outcomes = np.concatenate(
        [
            np.stack([sample["baseline_outcomes"] for sample in samples]),
            np.stack([sample["candidate_outcomes"] for sample in samples]),
        ],
        axis=0,
    ).astype(np.float64)
    pair_masks = np.stack([sample["target_mask"] for sample in samples]) > 0.0
    masks = np.concatenate([pair_masks, pair_masks], axis=0)
    scales = np.ones(outcomes.shape[1:], dtype=np.float32)
    event_weights = np.ones_like(scales)
    for horizon in range(outcomes.shape[1]):
        for metric in range(outcomes.shape[2]):
            valid = masks[:, horizon, metric]
            absolute = np.abs(outcomes[valid, horizon, metric])
            nonzero = absolute[absolute > 1.0e-6]
            if nonzero.size:
                scales[horizon, metric] = max(
                    float(np.quantile(nonzero, quantile)), 1.0e-3
                )
            positives = float(nonzero.size)
            zeros = float(max(int(np.sum(valid)) - int(positives), 0))
            if positives > 0.0:
                event_weights[horizon, metric] = min(
                    max(float(np.sqrt(zeros / positives)), 1.0),
                    maximum_event_weight,
                )
    return scales, event_weights


class PhysicsGraphWorldModelAbsoluteV151(
    PhysicsGraphWorldModelCounterfactualV141
):
    """Predict each branch outcome, then subtract predictions at inference."""

    def forward_absolute_outcomes(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        candidate_normalized = self._action_value(
            batch, batch["candidate_actions"].long()
        )
        baseline_normalized = self._action_value(
            batch, batch["baseline_actions"].long()
        )
        return {
            "candidate_outcomes": candidate_normalized * self.counterfactual_scale,
            "baseline_outcomes": baseline_normalized * self.counterfactual_scale,
            "candidate_normalized": candidate_normalized,
            "baseline_normalized": baseline_normalized,
            "outcome_scale": self.counterfactual_scale,
            "outcome_event_weight": self.counterfactual_event_weight,
        }

    def forward_counterfactual(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        absolute = self.forward_absolute_outcomes(batch)
        delta = absolute["candidate_outcomes"] - absolute["baseline_outcomes"]
        return {
            "counterfactual_delta": delta,
            "normalized_counterfactual_delta": (
                absolute["candidate_normalized"]
                - absolute["baseline_normalized"]
            ),
            "counterfactual_scale": self.counterfactual_scale,
            "counterfactual_event_weight": self.counterfactual_event_weight,
        }


def absolute_outcome_loss_v151(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    scales = output["outcome_scale"].float()
    event_weights = output["outcome_event_weight"].float()
    mask = batch["target_mask"].float()
    weighted_losses = []
    absolute_errors = []
    weight_totals = []
    for branch in ("baseline", "candidate"):
        prediction = output[f"{branch}_outcomes"].float()
        target = batch[f"{branch}_outcomes"].float()
        per_component = F.smooth_l1_loss(
            prediction / scales,
            target / scales,
            reduction="none",
        )
        sparse_weights = torch.where(
            target.abs() > 1.0e-6,
            event_weights,
            torch.ones_like(target),
        )
        weights = mask * sparse_weights
        weighted_losses.append((per_component * weights).sum())
        absolute_errors.append((prediction.sub(target).abs() * mask).sum())
        weight_totals.append(weights.sum())
    loss = torch.stack(weighted_losses).sum() / torch.stack(weight_totals).sum().clamp_min(1.0)
    mae = torch.stack(absolute_errors).sum() / (2.0 * mask.sum()).clamp_min(1.0)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "absolute_outcome_mae": float(mae.detach().cpu()),
    }


def initialize_absolute_v151_from_v13(
    checkpoint_path: str | Path,
    outcome_scale: np.ndarray,
    outcome_event_weight: np.ndarray,
    device: str = "cpu",
) -> Tuple[PhysicsGraphWorldModelAbsoluteV151, WorldModelMetadata, Dict[str, Any]]:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=False
    )
    if checkpoint.get("model_version") != V13_MODEL_VERSION:
        raise ValueError("V15.1 must be initialized from the frozen V13 checkpoint")
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsGraphWorldModelAbsoluteV151(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
        future_risk_pos_weight=checkpoint["future_risk_pos_weight"],
        future_risk_horizon=int(checkpoint["future_risk_horizon"]),
        counterfactual_scale=outcome_scale,
        counterfactual_event_weight=outcome_event_weight,
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    expected_missing = {
        "counterfactual_scale",
        "counterfactual_event_weight",
        *{
            f"counterfactual_value_head.{name}"
            for name in model.counterfactual_value_head.state_dict()
        },
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected V13-to-V15.1 state mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device), metadata, checkpoint


def save_absolute_model_v151(
    path: str | Path,
    model: PhysicsGraphWorldModelAbsoluteV151,
    metadata: WorldModelMetadata,
    history: List[Dict[str, float]],
    args: Dict[str, Any],
    initialization_checkpoint: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "state_dict": model.state_dict(),
            "metadata": asdict(metadata),
            "history": history,
            "args": args,
            "initialization_checkpoint": initialization_checkpoint,
            "counterfactual_horizons_sec": model.counterfactual_horizons_sec,
            "counterfactual_metric_names": COUNTERFACTUAL_METRIC_NAMES,
            "outcome_scale": model.counterfactual_scale.detach().cpu(),
            "outcome_event_weight": (
                model.counterfactual_event_weight.detach().cpu()
            ),
            "future_risk_horizon": model.future_risk_horizon,
            "future_risk_pos_weight": model.future_risk_pos_weight.detach().cpu(),
            "node_physical_features": model.node_physical_features.detach().cpu(),
            "edge_physical_features": model.edge_physical_features.detach().cpu(),
        },
        destination,
    )


def load_absolute_model_v151(
    path: str | Path, device: str = "cpu"
) -> PhysicsGraphWorldModelAbsoluteV151:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Checkpoint is not a {MODEL_VERSION} model")
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsGraphWorldModelAbsoluteV151(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
        future_risk_pos_weight=checkpoint["future_risk_pos_weight"],
        future_risk_horizon=int(checkpoint["future_risk_horizon"]),
        counterfactual_scale=checkpoint["outcome_scale"],
        counterfactual_event_weight=checkpoint["outcome_event_weight"],
        counterfactual_horizons_sec=checkpoint["counterfactual_horizons_sec"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device)
