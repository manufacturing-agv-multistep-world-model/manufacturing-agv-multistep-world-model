from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    COUNTERFACTUAL_METRIC_NAMES,
)
from physics_graph_world_model import WorldModelMetadata
from physics_graph_world_model_multistep_v13 import (
    MODEL_VERSION as V13_MODEL_VERSION,
    PhysicsInformedGraphWorldModelMultiStepV13,
)


MODEL_VERSION = "pi_gwm_multistep_v141_paired_counterfactual_effect_v1"


def counterfactual_target_statistics(
    samples: Sequence[Dict[str, np.ndarray]],
    quantile: float = 0.75,
    maximum_event_weight: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate leakage-free scales and sparse-effect weights from training pairs."""

    if not samples:
        raise ValueError("Counterfactual target statistics require training samples")
    targets = np.stack([sample["target_delta"] for sample in samples]).astype(
        np.float64
    )
    masks = np.stack([sample["target_mask"] for sample in samples]) > 0.0
    scales = np.ones(targets.shape[1:], dtype=np.float32)
    event_weights = np.ones_like(scales)
    for horizon in range(targets.shape[1]):
        for metric in range(targets.shape[2]):
            valid = masks[:, horizon, metric]
            absolute = np.abs(targets[valid, horizon, metric])
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


class PhysicsGraphWorldModelCounterfactualV141(
    PhysicsInformedGraphWorldModelMultiStepV13
):
    """Predict paired long-horizon effects under a shared physical continuation."""

    def __init__(
        self,
        metadata: WorldModelMetadata,
        node_physical_features: torch.Tensor,
        edge_physical_features: torch.Tensor,
        future_risk_pos_weight: torch.Tensor | np.ndarray | None = None,
        future_risk_horizon: int = 80,
        counterfactual_scale: torch.Tensor | np.ndarray | None = None,
        counterfactual_event_weight: torch.Tensor | np.ndarray | None = None,
        counterfactual_horizons_sec: Sequence[float] = COUNTERFACTUAL_HORIZONS_SEC,
    ):
        super().__init__(
            metadata,
            node_physical_features,
            edge_physical_features,
            future_risk_pos_weight=future_risk_pos_weight,
            future_risk_horizon=future_risk_horizon,
        )
        horizons = tuple(float(value) for value in counterfactual_horizons_sec)
        if not horizons or any(value <= 0.0 for value in horizons):
            raise ValueError("Counterfactual horizons must be positive")
        self.counterfactual_horizons_sec = horizons
        output_shape = (len(horizons), len(COUNTERFACTUAL_METRIC_NAMES))
        scales = (
            torch.ones(output_shape, dtype=torch.float32)
            if counterfactual_scale is None
            else torch.as_tensor(counterfactual_scale, dtype=torch.float32)
        )
        event_weights = (
            torch.ones(output_shape, dtype=torch.float32)
            if counterfactual_event_weight is None
            else torch.as_tensor(counterfactual_event_weight, dtype=torch.float32)
        )
        if tuple(scales.shape) != output_shape:
            raise ValueError("Counterfactual scales do not match output dimensions")
        if tuple(event_weights.shape) != output_shape:
            raise ValueError("Counterfactual event weights do not match outputs")
        if not torch.isfinite(scales).all() or torch.any(scales <= 0.0):
            raise ValueError("Counterfactual scales must be finite and positive")
        if not torch.isfinite(event_weights).all() or torch.any(event_weights < 1.0):
            raise ValueError("Counterfactual event weights must be finite and at least one")
        self.register_buffer("counterfactual_scale", scales.clone())
        self.register_buffer("counterfactual_event_weight", event_weights.clone())
        hidden = metadata.hidden_dim
        self.counterfactual_value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(np.prod(output_shape))),
        )

    def _action_value(
        self, batch: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> torch.Tensor:
        action_batch = dict(batch)
        action_batch["actions"] = actions
        output = super().forward_step(action_batch)
        values = self.counterfactual_value_head(output["_transition_context"])
        return values.reshape(
            -1,
            len(self.counterfactual_horizons_sec),
            len(COUNTERFACTUAL_METRIC_NAMES),
        )

    def forward_counterfactual(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        candidate_value = self._action_value(batch, batch["candidate_actions"].long())
        baseline_value = self._action_value(batch, batch["baseline_actions"].long())
        normalized_delta = candidate_value - baseline_value
        return {
            "counterfactual_delta": normalized_delta * self.counterfactual_scale,
            "normalized_counterfactual_delta": normalized_delta,
            "counterfactual_scale": self.counterfactual_scale,
            "counterfactual_event_weight": self.counterfactual_event_weight,
        }


def counterfactual_loss_v141(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    prediction = output["counterfactual_delta"].float()
    target = batch["target_delta"].float()
    mask = batch["target_mask"].float()
    scales = output["counterfactual_scale"].float()
    event_weights = output["counterfactual_event_weight"].float()
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
    loss = (per_component * weights).sum() / weights.sum().clamp_min(1.0)
    mae = (prediction.sub(target).abs() * mask).sum() / mask.sum().clamp_min(1.0)
    zero_mae = (target.abs() * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "zero_baseline_mae": float(zero_mae.detach().cpu()),
    }


def initialize_v141_from_v13(
    checkpoint_path: str | Path,
    counterfactual_scale: np.ndarray,
    counterfactual_event_weight: np.ndarray,
    device: str = "cpu",
) -> Tuple[PhysicsGraphWorldModelCounterfactualV141, WorldModelMetadata, Dict[str, Any]]:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=False
    )
    if checkpoint.get("model_version") != V13_MODEL_VERSION:
        raise ValueError("V14.1 must be initialized from the frozen V13 checkpoint")
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsGraphWorldModelCounterfactualV141(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
        future_risk_pos_weight=checkpoint["future_risk_pos_weight"],
        future_risk_horizon=int(checkpoint["future_risk_horizon"]),
        counterfactual_scale=counterfactual_scale,
        counterfactual_event_weight=counterfactual_event_weight,
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
            "Unexpected V13-to-V14.1 state mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device), metadata, checkpoint


def freeze_v141_backbone(model: PhysicsGraphWorldModelCounterfactualV141) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.counterfactual_value_head.parameters():
        parameter.requires_grad = True


def save_counterfactual_model_v141(
    path: str | Path,
    model: PhysicsGraphWorldModelCounterfactualV141,
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
            "counterfactual_scale": model.counterfactual_scale.detach().cpu(),
            "counterfactual_event_weight": (
                model.counterfactual_event_weight.detach().cpu()
            ),
            "future_risk_horizon": model.future_risk_horizon,
            "future_risk_pos_weight": model.future_risk_pos_weight.detach().cpu(),
            "node_physical_features": model.node_physical_features.detach().cpu(),
            "edge_physical_features": model.edge_physical_features.detach().cpu(),
        },
        destination,
    )


def load_counterfactual_model_v141(
    path: str | Path, device: str = "cpu"
) -> PhysicsGraphWorldModelCounterfactualV141:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Checkpoint is not a {MODEL_VERSION} model")
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsGraphWorldModelCounterfactualV141(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
        future_risk_pos_weight=checkpoint["future_risk_pos_weight"],
        future_risk_horizon=int(checkpoint["future_risk_horizon"]),
        counterfactual_scale=checkpoint["counterfactual_scale"],
        counterfactual_event_weight=checkpoint["counterfactual_event_weight"],
        counterfactual_horizons_sec=checkpoint["counterfactual_horizons_sec"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
