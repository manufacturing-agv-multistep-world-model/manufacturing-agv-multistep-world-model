from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from jms_parameter_registry import MPC_UTILITY_WEIGHTS
from physics_graph_world_model import KPI_NAMES, WorldModelMetadata
from physics_graph_world_model_multistep import (
    MultiStepPhysicsInformedMPCPolicy,
    stable_graph_attention,
)


MODEL_VERSION = "pi_gwm_multistep_v10_action_conditioned"

# Fixed, dimensionless engineering-priority weights inherited before the
# factorial study. They emphasize energy, then time, while limiting the
# influence of the more frequent normalized blocking signal; they are not
# learned or tuned on evaluation trajectories.
KPI_COMPONENT_WEIGHTS = (0.5, 8.0, 32.0, 2.0, 2.0, 2.0)


class PhysicsInformedGraphWorldModelMultiStepV10(nn.Module):
    """Action-conditioned graph world model with engineering-balanced KPI loss."""

    def __init__(self, metadata: WorldModelMetadata):
        super().__init__()
        self.metadata = metadata
        hidden = metadata.hidden_dim

        self.agent_encoder = nn.Sequential(
            nn.Linear(metadata.agent_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(metadata.node_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.node_query = nn.Linear(hidden, hidden, bias=False)
        self.node_key = nn.Linear(hidden, hidden, bias=False)
        self.local_action_encoder = nn.Sequential(nn.Linear(metadata.action_dim, hidden), nn.SiLU())
        self.joint_action_encoder = nn.Sequential(
            nn.Linear(metadata.agv_count * metadata.action_dim, hidden), nn.SiLU()
        )
        self.global_encoder = nn.Sequential(nn.Linear(metadata.global_dim, hidden), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden), nn.SiLU()
        )
        self.agent_delta_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, metadata.agent_dim)
        )
        self.node_delta_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, metadata.node_dim)
        )
        self.global_delta_head = nn.Linear(hidden, metadata.global_dim)
        self.kpi_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, len(KPI_NAMES))
        )

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        agent_features = batch["agent_features"].float()
        node_features = batch["node_features"].float()
        global_features = batch["global_features"].float()
        adjacency = batch["adjacency_matrix"].float()
        actions = batch["actions"].long()

        agent_tokens = self.agent_encoder(agent_features)
        node_tokens = self.node_encoder(node_features)
        action_one_hot = torch.nn.functional.one_hot(
            actions, num_classes=self.metadata.action_dim
        ).float()
        local_action_tokens = self.local_action_encoder(action_one_hot)
        joint_action_context = self.joint_action_encoder(
            action_one_hot.reshape(actions.shape[0], -1)
        )

        query = self.node_query(node_tokens)
        key = self.node_key(node_tokens)
        graph_tokens = stable_graph_attention(query, key, node_tokens, adjacency)
        fleet_context = agent_tokens.mean(dim=1)
        graph_context = graph_tokens.mean(dim=1)
        global_context = self.global_encoder(global_features)
        latent = self.fusion(
            torch.cat(
                [fleet_context, graph_context, global_context, joint_action_context], dim=-1
            )
        )

        expanded_latent = latent.unsqueeze(1).expand(-1, self.metadata.agv_count, -1)
        expanded_graph = graph_context.unsqueeze(1).expand(-1, self.metadata.agv_count, -1)
        agent_delta = self.agent_delta_head(
            torch.cat(
                [agent_tokens, local_action_tokens, expanded_graph, expanded_latent], dim=-1
            )
        )
        expanded_node_latent = latent.unsqueeze(1).expand(-1, self.metadata.node_count, -1)
        node_delta = self.node_delta_head(
            torch.cat([node_tokens, graph_tokens, expanded_node_latent], dim=-1)
        )
        return {
            "next_agent_features": agent_features + agent_delta,
            "next_node_features": node_features + node_delta,
            "next_global_features": global_features + self.global_delta_head(latent),
            "kpi": self.kpi_head(latent),
        }

    @staticmethod
    def _bounded_state(output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "agent_features": output["next_agent_features"].clamp(-0.1, 1.5),
            "node_features": output["next_node_features"].clamp(-0.1, 1.5),
            "global_features": output["next_global_features"].clamp(-0.5, 100.0),
        }

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
        predicted_agents: List[torch.Tensor] = []
        predicted_nodes: List[torch.Tensor] = []
        predicted_globals: List[torch.Tensor] = []
        predicted_kpis: List[torch.Tensor] = []

        for step in range(actions.shape[1]):
            output = self.forward_step(
                {**state, "adjacency_matrix": adjacency, "actions": actions[:, step]}
            )
            predicted_agents.append(output["next_agent_features"])
            predicted_nodes.append(output["next_node_features"])
            predicted_globals.append(output["next_global_features"])
            predicted_kpis.append(output["kpi"])
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
            "pred_agent_features": torch.stack(predicted_agents, dim=1),
            "pred_node_features": torch.stack(predicted_nodes, dim=1),
            "pred_global_features": torch.stack(predicted_globals, dim=1),
            "pred_kpi": torch.stack(predicted_kpis, dim=1),
        }


def multistep_world_model_loss_v10(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    physics_weight: float = 0.5,
    discount: float = 0.9,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    horizon = output["pred_kpi"].shape[1]
    step_weights = torch.pow(
        torch.as_tensor(discount, dtype=torch.float32, device=output["pred_kpi"].device),
        torch.arange(horizon, dtype=torch.float32, device=output["pred_kpi"].device),
    )
    step_weights = step_weights / step_weights.sum()

    def discounted_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_step = (prediction - target.float()).pow(2).flatten(start_dim=2).mean(dim=2)
        return (per_step * step_weights.unsqueeze(0)).sum(dim=1).mean()

    def weighted_kpi_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        component_weights: torch.Tensor,
    ) -> torch.Tensor:
        squared = (prediction - target.float()).pow(2)
        per_step = (squared * component_weights).sum(dim=2) / component_weights.sum()
        return (per_step * step_weights.unsqueeze(0)).sum(dim=1).mean()

    device = output["pred_kpi"].device
    kpi_weights = torch.as_tensor(KPI_COMPONENT_WEIGHTS, dtype=torch.float32, device=device)
    physics_weights = kpi_weights[[1, 2, 3]]
    agent_loss = discounted_mse(output["pred_agent_features"], batch["target_agent_features"])
    node_loss = discounted_mse(output["pred_node_features"], batch["target_node_features"])
    global_loss = discounted_mse(output["pred_global_features"], batch["target_global_features"])
    kpi_loss = weighted_kpi_mse(output["pred_kpi"], batch["target_kpi"], kpi_weights)
    physics_loss = weighted_kpi_mse(
        output["pred_kpi"][:, :, [1, 2, 3]],
        batch["target_physics_kpi"][:, :, [1, 2, 3]],
        physics_weights,
    )
    total = agent_loss + node_loss + global_loss + 4.0 * kpi_loss + physics_weight * physics_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "agent_loss": float(agent_loss.detach().cpu()),
        "node_loss": float(node_loss.detach().cpu()),
        "global_loss": float(global_loss.detach().cpu()),
        "kpi_loss": float(kpi_loss.detach().cpu()),
        "physics_loss": float(physics_loss.detach().cpu()),
    }


def save_multistep_world_model_v10(
    path: Path,
    model: PhysicsInformedGraphWorldModelMultiStepV10,
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
            "kpi_component_weights": KPI_COMPONENT_WEIGHTS,
            "mpc_utility_weights": dict(MPC_UTILITY_WEIGHTS),
        },
        path,
    )


def load_multistep_world_model_policy_v10(
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
    model = PhysicsInformedGraphWorldModelMultiStepV10(metadata)
    model.load_state_dict(checkpoint["state_dict"])
    args = checkpoint.get("args", {})
    return MultiStepPhysicsInformedMPCPolicy(
        model,
        device=device,
        planning_horizon=int(planning_horizon or args.get("planning_horizon", 3)),
        beam_width=int(beam_width or args.get("beam_width", 8)),
        discount=float(args.get("planning_discount", 0.95)),
        risk_gate_threshold=risk_gate_threshold,
    )
