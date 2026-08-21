from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jms_parameter_registry import MPC_UTILITY_WEIGHTS
from physics_graph_world_model import CONGESTION_KPI_NAMES, KPI_NAMES, WorldModelMetadata
from physics_graph_world_model_multistep import MultiStepPhysicsInformedMPCPolicy
from physics_graph_world_model_multistep_v10 import KPI_COMPONENT_WEIGHTS
from physics_graph_world_model_multistep_v11 import EDGE_PHYSICAL_NAMES, NODE_PHYSICAL_NAMES
from physics_graph_world_model_multistep_v12 import (
    CONGESTION_COMPONENT_WEIGHTS,
    PhysicsInformedGraphWorldModelMultiStepV12,
    multistep_world_model_loss_v12,
)


MODEL_VERSION = "pi_gwm_multistep_v13_multiscale_charge_onset_v2"
FUTURE_RISK_NAMES = ("future_charge_queue_risk",)
FUTURE_RISK_COMPONENT_WEIGHTS = (1.0,)
FUTURE_RISK_LOSS_WEIGHT = 1.0
CHARGE_CONGESTION_INDEX = 1


def annotate_future_congestion_risk(
    transitions: Sequence[Dict[str, np.ndarray]],
    horizon: int = 80,
) -> List[Dict[str, np.ndarray]]:
    """Label event onset in the next window while excluding active current events."""

    if horizon < 1:
        raise ValueError("Future-risk horizon must be positive")
    episodes: Dict[int, List[Dict[str, np.ndarray]]] = {}
    for fallback_index, transition in enumerate(transitions):
        episode_id = int(np.asarray(transition["episode_id"]).item())
        item = dict(transition)
        item.setdefault("transition_id", np.asarray(fallback_index, dtype=np.int64))
        episodes.setdefault(episode_id, []).append(item)

    annotated: List[Dict[str, np.ndarray]] = []
    for _, rows in sorted(episodes.items()):
        rows.sort(key=lambda row: int(np.asarray(row["transition_id"]).item()))
        event_values = np.stack(
            [
                np.asarray(
                    [
                        row.get(
                            "congestion_kpi",
                            np.zeros(len(CONGESTION_KPI_NAMES), dtype=np.float32),
                        )[CHARGE_CONGESTION_INDEX]
                    ],
                    dtype=np.float32,
                )
                for row in rows
            ]
        ) > 0.0
        cumulative_events = np.concatenate(
            [
                np.zeros((1, event_values.shape[1]), dtype=np.int64),
                np.cumsum(event_values, axis=0, dtype=np.int64),
            ],
            axis=0,
        )
        for index, row in enumerate(rows):
            # Mutate the in-memory cache object to avoid duplicating tens of
            # thousands of large graph-state dictionaries during annotation.
            item = row
            window_end = index + horizon + 1
            complete_window = window_end <= len(rows)
            if complete_window:
                risk = (
                    cumulative_events[window_end] - cumulative_events[index + 1] > 0
                ).astype(np.float32)
                # An already-active event measures persistence, not anticipation.
                mask = (~event_values[index]).astype(np.float32)
            else:
                risk = np.zeros(len(FUTURE_RISK_NAMES), dtype=np.float32)
                mask = np.zeros(len(FUTURE_RISK_NAMES), dtype=np.float32)
            item["future_congestion_risk"] = risk
            item["future_congestion_risk_mask"] = mask
            annotated.append(item)
    return annotated


def future_risk_positive_weights(
    transitions: Sequence[Dict[str, np.ndarray]],
    maximum: float = 20.0,
) -> np.ndarray:
    """Derive bounded class weights only from the training split."""

    targets = np.stack([row["future_congestion_risk"] for row in transitions])
    masks = np.stack([row["future_congestion_risk_mask"] for row in transitions]) > 0.0
    weights = []
    for component in range(targets.shape[1]):
        valid = masks[:, component]
        positives = float(np.sum(targets[valid, component] > 0.0))
        negatives = float(np.sum(targets[valid, component] <= 0.0))
        if positives <= 0.0:
            raise ValueError(
                f"Future-risk component {component} has no positive training labels"
            )
        weights.append(min(max(negatives / positives, 1.0), maximum))
    return np.asarray(weights, dtype=np.float32)


class PhysicsInformedGraphWorldModelMultiStepV13(
    PhysicsInformedGraphWorldModelMultiStepV12
):
    """Short-horizon physical rollout plus direct long-horizon congestion risk."""

    def __init__(
        self,
        metadata: WorldModelMetadata,
        node_physical_features: torch.Tensor,
        edge_physical_features: torch.Tensor,
        future_risk_pos_weight: torch.Tensor | np.ndarray | None = None,
        future_risk_horizon: int = 80,
    ):
        super().__init__(metadata, node_physical_features, edge_physical_features)
        if future_risk_horizon < 1:
            raise ValueError("Future-risk horizon must be positive")
        self.future_risk_horizon = int(future_risk_horizon)
        weights = (
            torch.ones(len(FUTURE_RISK_NAMES), dtype=torch.float32)
            if future_risk_pos_weight is None
            else torch.as_tensor(future_risk_pos_weight, dtype=torch.float32)
        )
        if tuple(weights.shape) != (len(FUTURE_RISK_NAMES),):
            raise ValueError("Future-risk positive weights must match future-risk outputs")
        self.register_buffer("future_risk_pos_weight", weights.clone())
        hidden = metadata.hidden_dim
        self.future_risk_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, len(FUTURE_RISK_NAMES)),
        )

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = super().forward_step(batch)
        output["future_congestion_risk_logits"] = self.future_risk_head(
            output["_transition_context"]
        )
        return output

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        output = super().rollout(batch, teacher_forcing_ratio=teacher_forcing_ratio)
        output["future_risk_pos_weight"] = self.future_risk_pos_weight
        return output


def multistep_world_model_loss_v13(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    physics_weight: float = 0.5,
    discount: float = 0.9,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine V12 physical losses with masked long-horizon event supervision."""

    base_loss, parts = multistep_world_model_loss_v12(
        output,
        batch,
        physics_weight=physics_weight,
        discount=discount,
    )
    logits = output["pred_future_congestion_risk_logits"]
    target = batch["target_future_congestion_risk"].float()
    mask = batch["target_future_congestion_risk_mask"].float()
    horizon = logits.shape[1]
    step_weights = torch.pow(
        torch.as_tensor(discount, dtype=torch.float32, device=logits.device),
        torch.arange(horizon, dtype=torch.float32, device=logits.device),
    )
    component_weights = torch.as_tensor(
        FUTURE_RISK_COMPONENT_WEIGHTS,
        dtype=torch.float32,
        device=logits.device,
    )
    positive_weights = output["future_risk_pos_weight"].to(
        device=logits.device, dtype=torch.float32
    )
    per_component = F.binary_cross_entropy_with_logits(
        logits.float(),
        target,
        pos_weight=positive_weights,
        reduction="none",
    )
    weights = (
        mask
        * step_weights.view(1, horizon, 1)
        * component_weights.view(1, 1, -1)
    )
    future_risk_loss = (per_component * weights).sum() / weights.sum().clamp_min(1.0)
    total = base_loss + FUTURE_RISK_LOSS_WEIGHT * future_risk_loss
    parts = dict(parts)
    parts["future_risk_loss"] = float(future_risk_loss.detach().cpu())
    parts["loss"] = float(total.detach().cpu())
    return total, parts


def save_multistep_world_model_v13(
    path: Path,
    model: PhysicsInformedGraphWorldModelMultiStepV13,
    metadata: WorldModelMetadata,
    history: List[Dict[str, float]],
    args: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "state_dict": model.state_dict(),
            "metadata": asdict(metadata),
            "history": history,
            "args": args,
            "kpi_names": KPI_NAMES,
            "congestion_kpi_names": CONGESTION_KPI_NAMES,
            "kpi_component_weights": KPI_COMPONENT_WEIGHTS,
            "congestion_component_weights": CONGESTION_COMPONENT_WEIGHTS,
            "future_risk_component_weights": FUTURE_RISK_COMPONENT_WEIGHTS,
            "future_risk_names": FUTURE_RISK_NAMES,
            "future_risk_loss_weight": FUTURE_RISK_LOSS_WEIGHT,
            "future_risk_horizon": model.future_risk_horizon,
            "future_risk_pos_weight": model.future_risk_pos_weight.detach().cpu(),
            "node_physical_names": NODE_PHYSICAL_NAMES,
            "edge_physical_names": EDGE_PHYSICAL_NAMES,
            "node_physical_features": model.node_physical_features.detach().cpu(),
            "edge_physical_features": model.edge_physical_features.detach().cpu(),
            "mpc_utility_weights": dict(MPC_UTILITY_WEIGHTS),
        },
        path,
    )


def load_multistep_world_model_policy_v13(
    path: str | Path,
    device: str = "cpu",
) -> MultiStepPhysicsInformedMPCPolicy:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Checkpoint is not a {MODEL_VERSION} model")
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsInformedGraphWorldModelMultiStepV13(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
        future_risk_pos_weight=checkpoint["future_risk_pos_weight"],
        future_risk_horizon=int(checkpoint["future_risk_horizon"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    args = checkpoint.get("args", {})
    return MultiStepPhysicsInformedMPCPolicy(
        model,
        device=device,
        planning_horizon=int(args.get("planning_horizon", 5)),
        beam_width=int(args.get("beam_width", 8)),
        discount=float(args.get("planning_discount", 0.95)),
    )
