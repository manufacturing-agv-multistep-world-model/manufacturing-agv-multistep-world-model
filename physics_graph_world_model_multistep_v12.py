from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from jms_parameter_registry import MPC_UTILITY_WEIGHTS
from physics_graph_world_model import CONGESTION_KPI_NAMES, KPI_NAMES, WorldModelMetadata
from physics_graph_world_model_multistep import MultiStepPhysicsInformedMPCPolicy
from physics_graph_world_model_multistep_v10 import KPI_COMPONENT_WEIGHTS
from physics_graph_world_model_multistep_v11 import (
    EDGE_PHYSICAL_NAMES,
    NODE_PHYSICAL_NAMES,
    PhysicsInformedGraphWorldModelMultiStepV11,
    multistep_world_model_loss_v11,
)


MODEL_VERSION = "pi_gwm_multistep_v12_charge_aware"
CONGESTION_COMPONENT_WEIGHTS = (1.0, 2.0)
POSITIVE_CONGESTION_WEIGHT = 4.0


class PhysicsInformedGraphWorldModelMultiStepV12(
    PhysicsInformedGraphWorldModelMultiStepV11
):
    """V11 physical graph model with route/charger congestion attribution."""

    def __init__(
        self,
        metadata: WorldModelMetadata,
        node_physical_features: torch.Tensor,
        edge_physical_features: torch.Tensor,
    ):
        super().__init__(metadata, node_physical_features, edge_physical_features)
        hidden = metadata.hidden_dim
        self.congestion_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(CONGESTION_KPI_NAMES)),
        )

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = super().forward_step(batch)
        output["congestion_kpi"] = self.congestion_head(output["_transition_context"])
        return output

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        actions = batch["actions"].long()
        if actions.ndim != 3:
            raise ValueError("Multi-step actions must have shape [batch, horizon, agv]")
        state = {
            "agent_features": batch["agent_features"].float(),
            "node_features": batch["node_features"].float(),
            "global_features": batch["global_features"].float(),
        }
        adjacency = batch["adjacency_matrix"].float()
        outputs: Dict[str, List[torch.Tensor]] = {
            "pred_agent_features": [],
            "pred_node_features": [],
            "pred_global_features": [],
            "pred_kpi": [],
            "pred_congestion_kpi": [],
            "pred_position_logits": [],
            "pred_target_logits": [],
            "pred_future_congestion_risk_logits": [],
            "pred_future_terminal_kpi": [],
        }
        for step in range(actions.shape[1]):
            output = self.forward_step(
                {**state, "adjacency_matrix": adjacency, "actions": actions[:, step]}
            )
            outputs["pred_agent_features"].append(output["next_agent_features"])
            outputs["pred_node_features"].append(output["next_node_features"])
            outputs["pred_global_features"].append(output["next_global_features"])
            outputs["pred_kpi"].append(output["kpi"])
            outputs["pred_congestion_kpi"].append(output["congestion_kpi"])
            outputs["pred_position_logits"].append(output["position_logits"])
            outputs["pred_target_logits"].append(output["target_logits"])
            if "future_congestion_risk_logits" in output:
                outputs["pred_future_congestion_risk_logits"].append(
                    output["future_congestion_risk_logits"]
                )
            if "future_terminal_kpi" in output:
                outputs["pred_future_terminal_kpi"].append(
                    output["future_terminal_kpi"]
                )
            predicted_state = self._bounded_state(output)
            if self.training and teacher_forcing_ratio > 0.0 and "target_agent_features" in batch:
                use_truth = (
                    torch.rand(actions.shape[0], 1, 1, device=actions.device)
                    < teacher_forcing_ratio
                )
                state = {
                    "agent_features": torch.where(
                        use_truth,
                        batch["target_agent_features"][:, step],
                        predicted_state["agent_features"],
                    ),
                    "node_features": torch.where(
                        use_truth,
                        batch["target_node_features"][:, step],
                        predicted_state["node_features"],
                    ),
                    "global_features": torch.where(
                        use_truth.squeeze(-1),
                        batch["target_global_features"][:, step],
                        predicted_state["global_features"],
                    ),
                }
            else:
                state = predicted_state
        return {
            key: torch.stack(value, dim=1)
            for key, value in outputs.items()
            if value
        }


def multistep_world_model_loss_v12(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    physics_weight: float = 0.5,
    discount: float = 0.9,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Add event-balanced route and charger congestion supervision to V11."""

    base_loss, parts = multistep_world_model_loss_v11(
        output,
        batch,
        physics_weight=physics_weight,
        discount=discount,
    )
    prediction = output["pred_congestion_kpi"]
    target = batch["target_congestion_kpi"].float()
    horizon = prediction.shape[1]
    step_weights = torch.pow(
        torch.as_tensor(discount, dtype=torch.float32, device=prediction.device),
        torch.arange(horizon, dtype=torch.float32, device=prediction.device),
    )
    step_weights = step_weights / step_weights.sum()
    component_weights = torch.as_tensor(
        CONGESTION_COMPONENT_WEIGHTS,
        dtype=torch.float32,
        device=prediction.device,
    )
    event_weights = torch.where(
        target > 0.0,
        torch.full_like(target, POSITIVE_CONGESTION_WEIGHT),
        torch.ones_like(target),
    )
    per_component = F.smooth_l1_loss(prediction, target, reduction="none")
    per_step = (
        per_component * event_weights * component_weights
    ).sum(dim=2) / component_weights.sum()
    congestion_loss = (per_step * step_weights.unsqueeze(0)).sum(dim=1).mean()
    total = base_loss + 2.0 * congestion_loss
    parts = dict(parts)
    parts["congestion_loss"] = float(congestion_loss.detach().cpu())
    parts["loss"] = float(total.detach().cpu())
    return total, parts


def save_multistep_world_model_v12(
    path: Path,
    model: PhysicsInformedGraphWorldModelMultiStepV12,
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
            "positive_congestion_weight": POSITIVE_CONGESTION_WEIGHT,
            "node_physical_names": NODE_PHYSICAL_NAMES,
            "edge_physical_names": EDGE_PHYSICAL_NAMES,
            "node_physical_features": model.node_physical_features.detach().cpu(),
            "edge_physical_features": model.edge_physical_features.detach().cpu(),
            "mpc_utility_weights": dict(MPC_UTILITY_WEIGHTS),
        },
        path,
    )


def load_multistep_world_model_policy_v12(
    path: str | Path,
    device: str = "cpu",
    planning_horizon: int | None = None,
    beam_width: int | None = None,
    risk_gate_threshold: float | None = None,
) -> MultiStepPhysicsInformedMPCPolicy:
    checkpoint = torch.load(Path(path), map_location=device)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Checkpoint is not a {MODEL_VERSION} model")
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsInformedGraphWorldModelMultiStepV12(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    args = checkpoint.get("args", {})
    return MultiStepPhysicsInformedMPCPolicy(
        model,
        device=device,
        planning_horizon=int(planning_horizon or args.get("planning_horizon", 5)),
        beam_width=int(beam_width or args.get("beam_width", 8)),
        discount=float(args.get("planning_discount", 0.95)),
        risk_gate_threshold=risk_gate_threshold,
    )
