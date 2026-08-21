from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jms_parameter_registry import MPC_UTILITY_WEIGHTS
from physics_graph_world_model import (
    CONGESTION_KPI_NAMES,
    KPI_NAMES,
    WorldModelMetadata,
    kpi_scale,
)
from physics_graph_world_model_multistep_v10 import KPI_COMPONENT_WEIGHTS
from physics_graph_world_model_multistep_v11 import EDGE_PHYSICAL_NAMES, NODE_PHYSICAL_NAMES
from physics_graph_world_model_multistep_v12 import CONGESTION_COMPONENT_WEIGHTS
from physics_graph_world_model_multistep_v13 import (
    FUTURE_RISK_COMPONENT_WEIGHTS,
    FUTURE_RISK_LOSS_WEIGHT,
    FUTURE_RISK_NAMES,
    PhysicsInformedGraphWorldModelMultiStepV13,
    multistep_world_model_loss_v13,
)


MODEL_VERSION = "pi_gwm_multistep_v14_dual_timescale_terminal_efficiency_v1"
FUTURE_TERMINAL_KPI_NAMES = (
    "future_cumulative_energy_wh",
    "future_cumulative_completed_tasks",
    "future_cumulative_charge_queue_agent_steps",
)
FUTURE_TERMINAL_COMPONENT_WEIGHTS = (1.0, 1.0, 1.0)
FUTURE_TERMINAL_LOSS_WEIGHT = 1.0
ENERGY_KPI_INDEX = 2
THROUGHPUT_KPI_INDEX = 5
CHARGE_CONGESTION_INDEX = 1


def _raw_terminal_increment(row: Dict[str, np.ndarray]) -> np.ndarray:
    agv_count = int(np.asarray(row["agent_features"]).shape[0])
    raw_kpi = np.asarray(row["kpi"], dtype=np.float32) * kpi_scale(agv_count)
    congestion = np.asarray(
        row.get(
            "congestion_kpi",
            np.zeros(len(CONGESTION_KPI_NAMES), dtype=np.float32),
        ),
        dtype=np.float32,
    )
    return np.asarray(
        [
            max(float(raw_kpi[ENERGY_KPI_INDEX]), 0.0),
            max(float(raw_kpi[THROUGHPUT_KPI_INDEX]), 0.0),
            max(float(congestion[CHARGE_CONGESTION_INDEX]) * agv_count, 0.0),
        ],
        dtype=np.float32,
    )


def annotate_future_terminal_kpis(
    transitions: Sequence[Dict[str, np.ndarray]],
    horizon: int = 80,
) -> List[Dict[str, np.ndarray]]:
    """Attach action-conditioned cumulative physical targets over a complete window."""

    if horizon < 1:
        raise ValueError("Future-terminal horizon must be positive")
    episodes: Dict[int, List[Dict[str, np.ndarray]]] = {}
    for fallback_index, transition in enumerate(transitions):
        episode_id = int(np.asarray(transition["episode_id"]).item())
        item = dict(transition)
        item.setdefault("transition_id", np.asarray(fallback_index, dtype=np.int64))
        episodes.setdefault(episode_id, []).append(item)

    annotated: List[Dict[str, np.ndarray]] = []
    width = len(FUTURE_TERMINAL_KPI_NAMES)
    for _, rows in sorted(episodes.items()):
        rows.sort(key=lambda row: int(np.asarray(row["transition_id"]).item()))
        increments = np.stack([_raw_terminal_increment(row) for row in rows])
        prefix = np.concatenate(
            [
                np.zeros((1, width), dtype=np.float64),
                np.cumsum(increments, axis=0, dtype=np.float64),
            ],
            axis=0,
        )
        transition_ids = [
            int(np.asarray(row["transition_id"]).item()) for row in rows
        ]
        for index, row in enumerate(rows):
            window_end = index + horizon
            complete_window = window_end <= len(rows)
            if complete_window:
                ids = transition_ids[index:window_end]
                complete_window = all(
                    right == left + 1 for left, right in zip(ids, ids[1:])
                ) and not any(
                    bool(np.asarray(rows[offset].get("done", 0.0)).item())
                    for offset in range(index, window_end - 1)
                )
            if complete_window:
                target = (prefix[window_end] - prefix[index]).astype(np.float32)
                mask = np.ones(width, dtype=np.float32)
            else:
                target = np.zeros(width, dtype=np.float32)
                mask = np.zeros(width, dtype=np.float32)
            row["future_terminal_kpi"] = target
            row["future_terminal_kpi_mask"] = mask
            annotated.append(row)
    return annotated


def future_terminal_scales(
    transitions: Sequence[Dict[str, np.ndarray]],
    quantile: float = 0.75,
) -> np.ndarray:
    """Derive robust target scales from the training split only."""

    if not 0.0 < quantile <= 1.0:
        raise ValueError("Terminal-target scale quantile must be in (0, 1]")
    targets = np.stack([row["future_terminal_kpi"] for row in transitions])
    masks = np.stack([row["future_terminal_kpi_mask"] for row in transitions]) > 0.0
    scales = []
    for component in range(targets.shape[1]):
        valid = targets[masks[:, component], component]
        if valid.size == 0:
            raise ValueError(
                f"Future-terminal component {component} has no complete training labels"
            )
        scales.append(max(float(np.quantile(valid, quantile)), 1.0))
    return np.asarray(scales, dtype=np.float32)


def future_terminal_positive_weights(
    transitions: Sequence[Dict[str, np.ndarray]],
    maximum: float = 20.0,
) -> np.ndarray:
    """Balance sparse positive terminal events using training labels only."""

    if maximum < 1.0:
        raise ValueError("Maximum terminal positive weight must be at least one")
    targets = np.stack([row["future_terminal_kpi"] for row in transitions])
    masks = np.stack([row["future_terminal_kpi_mask"] for row in transitions]) > 0.0
    weights = np.ones(targets.shape[1], dtype=np.float32)
    component = FUTURE_TERMINAL_KPI_NAMES.index(
        "future_cumulative_charge_queue_agent_steps"
    )
    valid = masks[:, component]
    positives = float(np.sum(targets[valid, component] > 0.0))
    negatives = float(np.sum(targets[valid, component] <= 0.0))
    if positives <= 0.0:
        raise ValueError("Future-terminal charge-queue target has no positive labels")
    weights[component] = min(max(negatives / positives, 1.0), maximum)
    return weights


class PhysicsInformedGraphWorldModelMultiStepV14(
    PhysicsInformedGraphWorldModelMultiStepV13
):
    """Dual-timescale model with short rollouts and direct terminal efficiency heads."""

    def __init__(
        self,
        metadata: WorldModelMetadata,
        node_physical_features: torch.Tensor,
        edge_physical_features: torch.Tensor,
        future_risk_pos_weight: torch.Tensor | np.ndarray | None = None,
        future_risk_horizon: int = 80,
        future_terminal_scale: torch.Tensor | np.ndarray | None = None,
        future_terminal_positive_weight: torch.Tensor | np.ndarray | None = None,
        future_terminal_horizon: int = 80,
    ):
        super().__init__(
            metadata,
            node_physical_features,
            edge_physical_features,
            future_risk_pos_weight=future_risk_pos_weight,
            future_risk_horizon=future_risk_horizon,
        )
        if future_terminal_horizon < 1:
            raise ValueError("Future-terminal horizon must be positive")
        self.future_terminal_horizon = int(future_terminal_horizon)
        scales = (
            torch.ones(len(FUTURE_TERMINAL_KPI_NAMES), dtype=torch.float32)
            if future_terminal_scale is None
            else torch.as_tensor(future_terminal_scale, dtype=torch.float32)
        )
        if tuple(scales.shape) != (len(FUTURE_TERMINAL_KPI_NAMES),):
            raise ValueError("Future-terminal scales must match terminal outputs")
        if not torch.isfinite(scales).all() or torch.any(scales <= 0.0):
            raise ValueError("Future-terminal scales must be finite and positive")
        self.register_buffer("future_terminal_scale", scales.clone())
        positive_weights = (
            torch.ones(len(FUTURE_TERMINAL_KPI_NAMES), dtype=torch.float32)
            if future_terminal_positive_weight is None
            else torch.as_tensor(future_terminal_positive_weight, dtype=torch.float32)
        )
        if tuple(positive_weights.shape) != (len(FUTURE_TERMINAL_KPI_NAMES),):
            raise ValueError("Future-terminal positive weights must match terminal outputs")
        if not torch.isfinite(positive_weights).all() or torch.any(positive_weights < 1.0):
            raise ValueError("Future-terminal positive weights must be finite and at least one")
        self.register_buffer(
            "future_terminal_positive_weight", positive_weights.clone()
        )
        hidden = metadata.hidden_dim
        self.future_terminal_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, len(FUTURE_TERMINAL_KPI_NAMES)),
        )

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = super().forward_step(batch)
        normalized = F.softplus(self.future_terminal_head(output["_transition_context"]))
        output["future_terminal_kpi"] = normalized * self.future_terminal_scale
        return output

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        output = super().rollout(batch, teacher_forcing_ratio=teacher_forcing_ratio)
        output["future_terminal_scale"] = self.future_terminal_scale
        output["future_terminal_positive_weight"] = (
            self.future_terminal_positive_weight
        )
        return output


def multistep_world_model_loss_v14(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    physics_weight: float = 0.5,
    discount: float = 0.9,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Add leakage-free, scale-balanced terminal supervision to the V13 objective."""

    base_loss, parts = multistep_world_model_loss_v13(
        output,
        batch,
        physics_weight=physics_weight,
        discount=discount,
    )
    prediction = output["pred_future_terminal_kpi"].float()
    target = batch["target_future_terminal_kpi"].float()
    mask = batch["target_future_terminal_kpi_mask"].float()
    scales = output["future_terminal_scale"].to(
        device=prediction.device, dtype=torch.float32
    )
    horizon = prediction.shape[1]
    step_weights = torch.pow(
        torch.as_tensor(discount, dtype=torch.float32, device=prediction.device),
        torch.arange(horizon, dtype=torch.float32, device=prediction.device),
    )
    component_weights = torch.as_tensor(
        FUTURE_TERMINAL_COMPONENT_WEIGHTS,
        dtype=torch.float32,
        device=prediction.device,
    )
    positive_weights = output["future_terminal_positive_weight"].to(
        device=prediction.device, dtype=torch.float32
    )
    per_component = F.smooth_l1_loss(
        prediction / scales.view(1, 1, -1),
        target / scales.view(1, 1, -1),
        reduction="none",
    )
    event_weights = torch.where(
        target > 0.0,
        positive_weights.view(1, 1, -1),
        torch.ones_like(target),
    )
    weights = (
        mask
        * step_weights.view(1, horizon, 1)
        * component_weights.view(1, 1, -1)
        * event_weights
    )
    terminal_loss = (per_component * weights).sum() / weights.sum().clamp_min(1.0)
    total = base_loss + FUTURE_TERMINAL_LOSS_WEIGHT * terminal_loss
    parts = dict(parts)
    parts["future_terminal_loss"] = float(terminal_loss.detach().cpu())
    parts["loss"] = float(total.detach().cpu())
    return total, parts


def save_multistep_world_model_v14(
    path: Path,
    model: PhysicsInformedGraphWorldModelMultiStepV14,
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
            "future_terminal_kpi_names": FUTURE_TERMINAL_KPI_NAMES,
            "future_terminal_component_weights": FUTURE_TERMINAL_COMPONENT_WEIGHTS,
            "future_terminal_loss_weight": FUTURE_TERMINAL_LOSS_WEIGHT,
            "future_terminal_horizon": model.future_terminal_horizon,
            "future_terminal_scale": model.future_terminal_scale.detach().cpu(),
            "future_terminal_positive_weight": (
                model.future_terminal_positive_weight.detach().cpu()
            ),
            "node_physical_names": NODE_PHYSICAL_NAMES,
            "edge_physical_names": EDGE_PHYSICAL_NAMES,
            "node_physical_features": model.node_physical_features.detach().cpu(),
            "edge_physical_features": model.edge_physical_features.detach().cpu(),
            "mpc_utility_weights": dict(MPC_UTILITY_WEIGHTS),
        },
        path,
    )


def load_multistep_world_model_v14(
    path: str | Path,
    device: str = "cpu",
) -> PhysicsInformedGraphWorldModelMultiStepV14:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Checkpoint is not a {MODEL_VERSION} model")
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsInformedGraphWorldModelMultiStepV14(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
        future_risk_pos_weight=checkpoint["future_risk_pos_weight"],
        future_risk_horizon=int(checkpoint["future_risk_horizon"]),
        future_terminal_scale=checkpoint["future_terminal_scale"],
        future_terminal_positive_weight=checkpoint[
            "future_terminal_positive_weight"
        ],
        future_terminal_horizon=int(checkpoint["future_terminal_horizon"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
