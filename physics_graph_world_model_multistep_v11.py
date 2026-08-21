from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jms_parameter_registry import MPC_UTILITY_WEIGHTS
from physics_graph_world_model import KPI_NAMES, WorldModelMetadata
from physics_graph_world_model_multistep import MultiStepPhysicsInformedMPCPolicy
from physics_graph_world_model_multistep_v10 import KPI_COMPONENT_WEIGHTS


MODEL_VERSION = "pi_gwm_multistep_v11_physical_edges"
NODE_PHYSICAL_NAMES = ("x_coordinate", "y_coordinate", "node_degree")
EDGE_PHYSICAL_NAMES = ("distance", "speed_limit", "capacity", "nominal_kinematic_time")


def build_physical_graph_features(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Build normalized, static graph features directly from the DT scenario."""

    node_count = len(env.node_map)
    x = np.asarray([node.x_m for node in env.nodes], dtype=np.float32)
    y = np.asarray([node.y_m for node in env.nodes], dtype=np.float32)
    x_span = max(float(x.max() - x.min()), 1.0)
    y_span = max(float(y.max() - y.min()), 1.0)
    degree = np.asarray(
        [float(np.count_nonzero(env.adjacency_matrix[node]) - 1) for node in range(node_count)],
        dtype=np.float32,
    )
    node_physical = np.stack(
        [
            (x - float(x.min())) / x_span,
            (y - float(y.min())) / y_span,
            degree / max(float(degree.max()), 1.0),
        ],
        axis=1,
    ).astype(np.float32)

    max_distance = max(float(edge.distance_m) for edge in env.edges)
    max_speed = max(float(edge.speed_limit_mps) for edge in env.edges)
    nominal_times = {
        pair: env.get_kinematic_time(edge.distance_m, edge.speed_limit_mps, False)
        for pair, edge in env.edge_by_pair.items()
    }
    max_time = max(nominal_times.values())
    edge_physical = np.zeros((node_count, node_count, len(EDGE_PHYSICAL_NAMES)), dtype=np.float32)
    for (source, target), edge in env.edge_by_pair.items():
        capacity = (
            edge.capacity_stress
            if env.config.capacity_mode == "stress"
            else edge.capacity_baseline
        )
        edge_physical[source, target] = np.asarray(
            [
                float(edge.distance_m) / max(max_distance, 1.0),
                float(edge.speed_limit_mps) / max(max_speed, 1.0e-6),
                float(capacity) / max(float(env.agv_count), 1.0),
                float(nominal_times[(source, target)]) / max(float(max_time), 1.0),
            ],
            dtype=np.float32,
        )
    return node_physical, edge_physical


class PhysicsInformedGraphWorldModelMultiStepV11(nn.Module):
    """Physical-edge graph model with local AGV context and discrete node dynamics."""

    def __init__(
        self,
        metadata: WorldModelMetadata,
        node_physical_features: np.ndarray | torch.Tensor,
        edge_physical_features: np.ndarray | torch.Tensor,
    ):
        super().__init__()
        self.metadata = metadata
        hidden = metadata.hidden_dim
        node_physical = torch.as_tensor(node_physical_features, dtype=torch.float32)
        edge_physical = torch.as_tensor(edge_physical_features, dtype=torch.float32)
        if node_physical.shape != (metadata.node_count, len(NODE_PHYSICAL_NAMES)):
            raise ValueError("Unexpected node physical-feature shape")
        if edge_physical.shape != (
            metadata.node_count,
            metadata.node_count,
            len(EDGE_PHYSICAL_NAMES),
        ):
            raise ValueError("Unexpected edge physical-feature shape")
        self.register_buffer("node_physical_features", node_physical)
        self.register_buffer("edge_physical_features", edge_physical)

        self.agent_encoder = nn.Sequential(
            nn.Linear(metadata.agent_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(metadata.node_dim + len(NODE_PHYSICAL_NAMES), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.node_query = nn.Linear(hidden, hidden, bias=False)
        self.node_key = nn.Linear(hidden, hidden, bias=False)
        self.edge_bias = nn.Sequential(
            nn.Linear(len(EDGE_PHYSICAL_NAMES), hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1, bias=False),
        )
        self.edge_value = nn.Sequential(
            nn.Linear(len(EDGE_PHYSICAL_NAMES), hidden), nn.SiLU()
        )
        self.local_action_encoder = nn.Sequential(nn.Linear(metadata.action_dim, hidden), nn.SiLU())
        self.joint_action_encoder = nn.Sequential(
            nn.Linear(metadata.agv_count * metadata.action_dim, hidden), nn.SiLU()
        )
        self.global_encoder = nn.Sequential(nn.Linear(metadata.global_dim, hidden), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden), nn.SiLU()
        )
        self.agent_transition_encoder = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden), nn.SiLU()
        )
        self.agent_delta_head = nn.Linear(hidden, metadata.agent_dim)
        self.position_head = nn.Linear(hidden, metadata.node_count)
        self.target_head = nn.Linear(hidden, metadata.node_count)
        self.node_delta_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, metadata.node_dim)
        )
        self.global_delta_head = nn.Linear(hidden, metadata.global_dim)
        self.kpi_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, len(KPI_NAMES))
        )

    def _physical_graph_attention(
        self,
        node_tokens: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        query = self.node_query(node_tokens)
        key = self.node_key(node_tokens)
        logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(
            float(query.shape[-1])
        )
        edge_features = self.edge_physical_features.unsqueeze(0).expand(node_tokens.shape[0], -1, -1, -1)
        logits = logits + self.edge_bias(edge_features.float()).squeeze(-1)
        logits = logits.masked_fill(adjacency <= 0.0, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        neighbor_values = node_tokens.unsqueeze(1) + self.edge_value(edge_features).to(node_tokens.dtype)
        return (attention.unsqueeze(-1) * neighbor_values.float()).sum(dim=2).to(node_tokens.dtype)

    def _node_indices(self, normalized: torch.Tensor) -> torch.Tensor:
        return torch.round(normalized * max(self.metadata.node_count - 1, 1)).long().clamp(
            0, self.metadata.node_count - 1
        )

    @staticmethod
    def _gather_nodes(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            tokens,
            1,
            indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )

    def _straight_through_node_value(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        hard = F.one_hot(probabilities.argmax(dim=-1), num_classes=self.metadata.node_count).float()
        straight_through = hard + probabilities - probabilities.detach()
        node_values = torch.linspace(
            0.0, 1.0, self.metadata.node_count, dtype=logits.dtype, device=logits.device
        )
        return (straight_through * node_values).sum(dim=-1)

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        agent_features = batch["agent_features"].float()
        node_features = batch["node_features"].float()
        global_features = batch["global_features"].float()
        adjacency = batch["adjacency_matrix"].float()
        actions = batch["actions"].long()
        batch_size = agent_features.shape[0]

        node_physical = self.node_physical_features.unsqueeze(0).expand(batch_size, -1, -1)
        agent_tokens = self.agent_encoder(agent_features)
        node_tokens = self.node_encoder(torch.cat([node_features, node_physical], dim=-1))
        graph_tokens = self._physical_graph_attention(node_tokens, adjacency)
        action_one_hot = F.one_hot(actions, num_classes=self.metadata.action_dim).float()
        local_action_tokens = self.local_action_encoder(action_one_hot)
        joint_action_context = self.joint_action_encoder(action_one_hot.reshape(batch_size, -1))

        fleet_context = agent_tokens.mean(dim=1)
        graph_context = graph_tokens.mean(dim=1)
        global_context = self.global_encoder(global_features)
        latent = self.fusion(
            torch.cat([fleet_context, graph_context, global_context, joint_action_context], dim=-1)
        )
        position_indices = self._node_indices(agent_features[:, :, 0])
        target_indices = self._node_indices(agent_features[:, :, 3])
        local_graph = self._gather_nodes(graph_tokens, position_indices)
        target_graph = self._gather_nodes(graph_tokens, target_indices)
        expanded_latent = latent.unsqueeze(1).expand(-1, self.metadata.agv_count, -1)
        transition_tokens = self.agent_transition_encoder(
            torch.cat(
                [agent_tokens, local_action_tokens, local_graph, target_graph, expanded_latent],
                dim=-1,
            )
        )
        next_agent = agent_features + self.agent_delta_head(transition_tokens)
        position_logits = self.position_head(transition_tokens)
        target_logits = self.target_head(transition_tokens)
        predicted_position = self._straight_through_node_value(position_logits)
        predicted_target = self._straight_through_node_value(target_logits)
        next_agent = torch.cat(
            [
                predicted_position.unsqueeze(-1),
                next_agent[:, :, 1:3],
                predicted_target.unsqueeze(-1),
                next_agent[:, :, 4:6],
                (actions.float() / 3.0).unsqueeze(-1),
                next_agent[:, :, 7:],
            ],
            dim=-1,
        )

        expanded_node_latent = latent.unsqueeze(1).expand(-1, self.metadata.node_count, -1)
        node_delta = self.node_delta_head(
            torch.cat([node_tokens, graph_tokens, expanded_node_latent], dim=-1)
        )
        fleet_transition = transition_tokens.mean(dim=1)
        return {
            "next_agent_features": next_agent,
            "next_node_features": node_features + node_delta,
            "next_global_features": global_features + self.global_delta_head(latent),
            "kpi": self.kpi_head(torch.cat([latent, fleet_transition], dim=-1)),
            "position_logits": position_logits,
            "target_logits": target_logits,
            # Private transition context lets later model versions add
            # interpretable auxiliary heads without duplicating the encoder.
            "_transition_context": torch.cat([latent, fleet_transition], dim=-1),
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
        outputs: Dict[str, List[torch.Tensor]] = {
            "pred_agent_features": [],
            "pred_node_features": [],
            "pred_global_features": [],
            "pred_kpi": [],
            "pred_position_logits": [],
            "pred_target_logits": [],
        }
        for step in range(actions.shape[1]):
            output = self.forward_step(
                {**state, "adjacency_matrix": adjacency, "actions": actions[:, step]}
            )
            outputs["pred_agent_features"].append(output["next_agent_features"])
            outputs["pred_node_features"].append(output["next_node_features"])
            outputs["pred_global_features"].append(output["next_global_features"])
            outputs["pred_kpi"].append(output["kpi"])
            outputs["pred_position_logits"].append(output["position_logits"])
            outputs["pred_target_logits"].append(output["target_logits"])
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
        return {key: torch.stack(value, dim=1) for key, value in outputs.items()}


def multistep_world_model_loss_v11(
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

    def reduce_steps(per_step: torch.Tensor) -> torch.Tensor:
        return (per_step * step_weights.unsqueeze(0)).sum(dim=1).mean()

    def discounted_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return reduce_steps((prediction - target.float()).pow(2).flatten(start_dim=2).mean(dim=2))

    def weighted_kpi_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        component_weights: torch.Tensor,
    ) -> torch.Tensor:
        squared = (prediction - target.float()).pow(2)
        return reduce_steps((squared * component_weights).sum(dim=2) / component_weights.sum())

    def node_classification_loss(logits: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        node_count = logits.shape[-1]
        target_indices = torch.round(target_values * max(node_count - 1, 1)).long().clamp(
            0, node_count - 1
        )
        per_item = F.cross_entropy(
            logits.reshape(-1, node_count), target_indices.reshape(-1), reduction="none"
        ).reshape(logits.shape[0], logits.shape[1], logits.shape[2])
        return reduce_steps(per_item.mean(dim=2))

    device = output["pred_kpi"].device
    kpi_weights = torch.as_tensor(KPI_COMPONENT_WEIGHTS, dtype=torch.float32, device=device)
    physics_weights = kpi_weights[[1, 2, 3]]
    continuous_indices = [1, 2, 4, 5, 7, 8, 9]
    agent_loss = discounted_mse(
        output["pred_agent_features"][:, :, :, continuous_indices],
        batch["target_agent_features"][:, :, :, continuous_indices],
    )
    position_loss = node_classification_loss(
        output["pred_position_logits"], batch["target_agent_features"][:, :, :, 0]
    )
    target_loss = node_classification_loss(
        output["pred_target_logits"], batch["target_agent_features"][:, :, :, 3]
    )
    node_loss = discounted_mse(output["pred_node_features"], batch["target_node_features"])
    global_loss = discounted_mse(output["pred_global_features"], batch["target_global_features"])
    kpi_loss = weighted_kpi_mse(output["pred_kpi"], batch["target_kpi"], kpi_weights)
    physics_loss = weighted_kpi_mse(
        output["pred_kpi"][:, :, [1, 2, 3]],
        batch["target_physics_kpi"][:, :, [1, 2, 3]],
        physics_weights,
    )
    total = (
        agent_loss
        + 0.05 * position_loss
        + 0.05 * target_loss
        + node_loss
        + global_loss
        + 4.0 * kpi_loss
        + physics_weight * physics_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "agent_loss": float(agent_loss.detach().cpu()),
        "position_loss": float(position_loss.detach().cpu()),
        "target_loss": float(target_loss.detach().cpu()),
        "node_loss": float(node_loss.detach().cpu()),
        "global_loss": float(global_loss.detach().cpu()),
        "kpi_loss": float(kpi_loss.detach().cpu()),
        "physics_loss": float(physics_loss.detach().cpu()),
    }


def save_multistep_world_model_v11(
    path: Path,
    model: PhysicsInformedGraphWorldModelMultiStepV11,
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
            "node_physical_names": NODE_PHYSICAL_NAMES,
            "edge_physical_names": EDGE_PHYSICAL_NAMES,
            "node_physical_features": model.node_physical_features.detach().cpu(),
            "edge_physical_features": model.edge_physical_features.detach().cpu(),
            "mpc_utility_weights": dict(MPC_UTILITY_WEIGHTS),
        },
        path,
    )


def load_multistep_world_model_policy_v11(
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
    model = PhysicsInformedGraphWorldModelMultiStepV11(
        metadata,
        checkpoint["node_physical_features"],
        checkpoint["edge_physical_features"],
    )
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
