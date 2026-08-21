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
from physics_graph_world_model_counterfactual_v141 import (
    MODEL_VERSION as V141_MODEL_VERSION,
    PhysicsGraphWorldModelCounterfactualV141,
)


MODEL_VERSION = "pi_gwm_multistep_v142_zero_inflated_rank_aware_v1"
MATERIAL_EFFECT_FRACTION = 0.05
HARD_GATE_THRESHOLD = 0.50
REGRESSION_LOSS_WEIGHT = 1.00
GATE_LOSS_WEIGHT = 0.15
SIGN_LOSS_WEIGHT = 0.10
RANKING_LOSS_WEIGHT = 0.25
RANKING_MARGIN = 0.05
RANKING_TEMPERATURE = 0.20
TERMINAL_UTILITY_WEIGHTS = (-1.0, 1.0, -1.0)


def gate_positive_weights(
    samples: Sequence[Dict[str, np.ndarray]],
    scales: np.ndarray,
    maximum: float = 4.0,
) -> np.ndarray:
    targets = np.stack([sample["target_delta"] for sample in samples])
    masks = np.stack([sample["target_mask"] for sample in samples]) > 0.0
    material = np.abs(targets) >= MATERIAL_EFFECT_FRACTION * scales[None, :, :]
    weights = np.ones_like(scales, dtype=np.float32)
    for horizon in range(targets.shape[1]):
        for metric in range(targets.shape[2]):
            valid = masks[:, horizon, metric]
            positives = float(np.sum(material[valid, horizon, metric]))
            negatives = float(np.sum(valid) - positives)
            if positives > 0.0:
                weights[horizon, metric] = min(
                    max(float(np.sqrt(negatives / positives)), 1.0), maximum
                )
    return weights


class PhysicsGraphWorldModelCounterfactualV142(
    PhysicsGraphWorldModelCounterfactualV141
):
    """Zero-inflated, rank-aware paired physical-effect model."""

    def __init__(
        self,
        metadata: WorldModelMetadata,
        node_physical_features: torch.Tensor,
        edge_physical_features: torch.Tensor,
        future_risk_pos_weight: torch.Tensor | np.ndarray | None = None,
        future_risk_horizon: int = 80,
        counterfactual_scale: torch.Tensor | np.ndarray | None = None,
        counterfactual_event_weight: torch.Tensor | np.ndarray | None = None,
        counterfactual_gate_pos_weight: torch.Tensor | np.ndarray | None = None,
        counterfactual_horizons_sec: Sequence[float] = COUNTERFACTUAL_HORIZONS_SEC,
    ):
        super().__init__(
            metadata,
            node_physical_features,
            edge_physical_features,
            future_risk_pos_weight=future_risk_pos_weight,
            future_risk_horizon=future_risk_horizon,
            counterfactual_scale=counterfactual_scale,
            counterfactual_event_weight=counterfactual_event_weight,
            counterfactual_horizons_sec=counterfactual_horizons_sec,
        )
        output_shape = (
            len(self.counterfactual_horizons_sec),
            len(COUNTERFACTUAL_METRIC_NAMES),
        )
        gate_weights = (
            torch.ones(output_shape, dtype=torch.float32)
            if counterfactual_gate_pos_weight is None
            else torch.as_tensor(counterfactual_gate_pos_weight, dtype=torch.float32)
        )
        if tuple(gate_weights.shape) != output_shape:
            raise ValueError("Gate positive weights do not match physical outputs")
        if not torch.isfinite(gate_weights).all() or torch.any(gate_weights < 1.0):
            raise ValueError("Gate positive weights must be finite and at least one")
        self.register_buffer(
            "counterfactual_gate_pos_weight", gate_weights.clone()
        )
        hidden = metadata.hidden_dim
        self.counterfactual_gate_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(np.prod(output_shape))),
        )

    def _action_context(self, batch: Dict[str, torch.Tensor], actions: torch.Tensor):
        action_batch = dict(batch)
        action_batch["actions"] = actions.long()
        return super(PhysicsGraphWorldModelCounterfactualV141, self).forward_step(
            action_batch
        )["_transition_context"]

    def forward_counterfactual(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        candidate_context = self._action_context(
            batch, batch["candidate_actions"]
        )
        baseline_context = self._action_context(
            batch, batch["baseline_actions"]
        )
        output_shape = (
            -1,
            len(self.counterfactual_horizons_sec),
            len(COUNTERFACTUAL_METRIC_NAMES),
        )
        candidate_value = self.counterfactual_value_head(candidate_context).reshape(
            output_shape
        )
        baseline_value = self.counterfactual_value_head(baseline_context).reshape(
            output_shape
        )
        normalized_raw_delta = candidate_value - baseline_value
        symmetric_context = torch.cat(
            [
                torch.abs(candidate_context - baseline_context),
                0.5 * (candidate_context + baseline_context),
            ],
            dim=-1,
        )
        gate_logits = self.counterfactual_gate_head(symmetric_context).reshape(
            output_shape
        )
        gate_probability = torch.sigmoid(gate_logits)
        normalized_soft_delta = normalized_raw_delta * gate_probability
        hard_gate = (gate_probability >= HARD_GATE_THRESHOLD).to(
            normalized_raw_delta.dtype
        )
        normalized_hard_delta = normalized_raw_delta * hard_gate
        return {
            "counterfactual_delta": (
                normalized_soft_delta * self.counterfactual_scale
            ),
            "hard_counterfactual_delta": (
                normalized_hard_delta * self.counterfactual_scale
            ),
            "raw_counterfactual_delta": (
                normalized_raw_delta * self.counterfactual_scale
            ),
            "normalized_counterfactual_delta": normalized_soft_delta,
            "normalized_raw_counterfactual_delta": normalized_raw_delta,
            "counterfactual_gate_logits": gate_logits,
            "counterfactual_gate_probability": gate_probability,
            "counterfactual_scale": self.counterfactual_scale,
            "counterfactual_gate_pos_weight": (
                self.counterfactual_gate_pos_weight
            ),
        }


def _ranking_loss(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    mask: torch.Tensor,
    episode_ids: torch.Tensor,
    state_ids: torch.Tensor,
) -> torch.Tensor:
    terminal_prediction = normalized_prediction[:, -1]
    terminal_target = normalized_target[:, -1]
    terminal_mask = mask[:, -1].all(dim=-1)
    utility_weights = torch.as_tensor(
        TERMINAL_UTILITY_WEIGHTS,
        dtype=terminal_prediction.dtype,
        device=terminal_prediction.device,
    )
    predicted_utility = (terminal_prediction * utility_weights).sum(dim=-1)
    target_utility = (terminal_target * utility_weights).sum(dim=-1)
    keys = torch.stack([episode_ids.long(), state_ids.long()], dim=-1)
    losses = []
    for key in torch.unique(keys, dim=0):
        group = (keys == key).all(dim=-1) & terminal_mask
        candidate_prediction = predicted_utility[group]
        candidate_target = target_utility[group]
        if candidate_prediction.numel() == 0:
            continue
        values_prediction = torch.cat(
            [candidate_prediction.new_zeros(1), candidate_prediction]
        )
        values_target = torch.cat([candidate_target.new_zeros(1), candidate_target])
        count = values_target.numel()
        left, right = torch.triu_indices(
            count, count, offset=1, device=values_target.device
        )
        target_difference = values_target[left] - values_target[right]
        material = target_difference.abs() >= RANKING_MARGIN
        if not material.any():
            continue
        prediction_difference = values_prediction[left] - values_prediction[right]
        direction = torch.sign(target_difference[material])
        losses.append(
            F.softplus(
                -direction
                * prediction_difference[material]
                / RANKING_TEMPERATURE
            ).mean()
        )
    if not losses:
        return normalized_prediction.sum() * 0.0
    return torch.stack(losses).mean()


def counterfactual_loss_v142(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    scale = output["counterfactual_scale"].float()
    prediction = output["counterfactual_delta"].float()
    normalized_prediction = prediction / scale
    target = batch["target_delta"].float()
    normalized_target = target / scale
    mask = batch["target_mask"].float()
    material = (
        normalized_target.abs() >= MATERIAL_EFFECT_FRACTION
    ).float()

    regression_loss = (
        (normalized_prediction - normalized_target).abs() * mask
    ).sum() / mask.sum().clamp_min(1.0)
    gate_loss_raw = F.binary_cross_entropy_with_logits(
        output["counterfactual_gate_logits"].float(),
        material,
        pos_weight=output["counterfactual_gate_pos_weight"].float(),
        reduction="none",
    )
    gate_loss = (gate_loss_raw * mask).sum() / mask.sum().clamp_min(1.0)
    sign_mask = mask * material
    sign_margin = torch.sign(normalized_target) * normalized_prediction
    sign_loss = (
        F.softplus(-sign_margin / RANKING_TEMPERATURE) * sign_mask
    ).sum() / sign_mask.sum().clamp_min(1.0)
    ranking_loss = _ranking_loss(
        normalized_prediction,
        normalized_target,
        mask > 0.0,
        batch["episode_id"],
        batch["state_id"],
    )
    total = (
        REGRESSION_LOSS_WEIGHT * regression_loss
        + GATE_LOSS_WEIGHT * gate_loss
        + SIGN_LOSS_WEIGHT * sign_loss
        + RANKING_LOSS_WEIGHT * ranking_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "regression_loss": float(regression_loss.detach().cpu()),
        "gate_loss": float(gate_loss.detach().cpu()),
        "sign_loss": float(sign_loss.detach().cpu()),
        "ranking_loss": float(ranking_loss.detach().cpu()),
    }


def initialize_v142_from_v141(
    checkpoint_path: str | Path,
    gate_pos_weight: np.ndarray,
    device: str = "cpu",
) -> Tuple[PhysicsGraphWorldModelCounterfactualV142, WorldModelMetadata, Dict[str, Any]]:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=False
    )
    if checkpoint.get("model_version") != V141_MODEL_VERSION:
        raise ValueError("V14.2 must be initialized from a frozen V14.1 checkpoint")
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsGraphWorldModelCounterfactualV142(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
        future_risk_pos_weight=checkpoint["future_risk_pos_weight"],
        future_risk_horizon=int(checkpoint["future_risk_horizon"]),
        counterfactual_scale=checkpoint["counterfactual_scale"],
        counterfactual_event_weight=checkpoint["counterfactual_event_weight"],
        counterfactual_gate_pos_weight=gate_pos_weight,
        counterfactual_horizons_sec=checkpoint["counterfactual_horizons_sec"],
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    expected_missing = {
        "counterfactual_gate_pos_weight",
        *{
            f"counterfactual_gate_head.{name}"
            for name in model.counterfactual_gate_head.state_dict()
        },
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected V14.1-to-V14.2 state mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device), metadata, checkpoint


def freeze_v142_backbone(model: PhysicsGraphWorldModelCounterfactualV142) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for head in (
        model.counterfactual_value_head,
        model.counterfactual_gate_head,
    ):
        for parameter in head.parameters():
            parameter.requires_grad = True


def save_counterfactual_model_v142(
    path: str | Path,
    model: PhysicsGraphWorldModelCounterfactualV142,
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
            "counterfactual_gate_pos_weight": (
                model.counterfactual_gate_pos_weight.detach().cpu()
            ),
            "material_effect_fraction": MATERIAL_EFFECT_FRACTION,
            "hard_gate_threshold": HARD_GATE_THRESHOLD,
            "loss_weights": {
                "regression": REGRESSION_LOSS_WEIGHT,
                "gate": GATE_LOSS_WEIGHT,
                "sign": SIGN_LOSS_WEIGHT,
                "ranking": RANKING_LOSS_WEIGHT,
            },
            "ranking_margin": RANKING_MARGIN,
            "ranking_temperature": RANKING_TEMPERATURE,
            "terminal_utility_weights": TERMINAL_UTILITY_WEIGHTS,
            "future_risk_horizon": model.future_risk_horizon,
            "future_risk_pos_weight": model.future_risk_pos_weight.detach().cpu(),
            "node_physical_features": model.node_physical_features.detach().cpu(),
            "edge_physical_features": model.edge_physical_features.detach().cpu(),
        },
        destination,
    )


def load_counterfactual_model_v142(
    path: str | Path, device: str = "cpu"
) -> PhysicsGraphWorldModelCounterfactualV142:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Checkpoint is not a {MODEL_VERSION} model")
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsGraphWorldModelCounterfactualV142(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
        future_risk_pos_weight=checkpoint["future_risk_pos_weight"],
        future_risk_horizon=int(checkpoint["future_risk_horizon"]),
        counterfactual_scale=checkpoint["counterfactual_scale"],
        counterfactual_event_weight=checkpoint["counterfactual_event_weight"],
        counterfactual_gate_pos_weight=checkpoint[
            "counterfactual_gate_pos_weight"
        ],
        counterfactual_horizons_sec=checkpoint["counterfactual_horizons_sec"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
