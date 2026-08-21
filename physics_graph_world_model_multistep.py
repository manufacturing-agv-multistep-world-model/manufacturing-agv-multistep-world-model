from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from jms_parameter_registry import MPC_UTILITY_WEIGHTS
from physics_graph_world_model import (
    KPI_NAMES,
    WorldModelMetadata,
    baseline_dt_aware_action,
    charge_start_soc,
    estimate_action_physics,
    kpi_scale,
    torch_kpi_scale,
)


MODEL_VERSION = "pi_gwm_multistep_v9"


def project_physical_kpis(kpis: torch.Tensor, agv_count: int) -> torch.Tensor:
    """Project decoded KPIs onto their physically admissible support."""

    reward, dt_sec, energy_wh, blocked, deadlock, throughput = kpis.unbind(dim=-1)
    return torch.stack(
        [
            reward,
            dt_sec.clamp_min(0.0),
            energy_wh.clamp_min(0.0),
            blocked.clamp(0.0, float(agv_count)),
            deadlock.clamp(0.0, 1.0),
            throughput.clamp(0.0, float(agv_count)),
        ],
        dim=-1,
    )


def stable_graph_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    node_tokens: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    """Apply an adjacency mask without creating FP16 overflow under autocast."""

    logits = torch.matmul(
        query.float(), key.float().transpose(-1, -2)
    ) / math.sqrt(float(query.shape[-1]))
    logits = logits.masked_fill(adjacency <= 0.0, torch.finfo(logits.dtype).min)
    attention = torch.softmax(logits, dim=-1)
    return torch.matmul(attention, node_tokens.float()).to(node_tokens.dtype)


class MultiStepSequenceDataset(Dataset):
    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        if not samples:
            raise ValueError("MultiStepSequenceDataset requires at least one sequence")
        # Stack once instead of converting every field of every sample in each
        # epoch. This removes the dominant Python collation overhead on GPU runs.
        self.tensors = {
            key: torch.as_tensor(np.stack([sample[key] for sample in samples]))
            for key in samples[0]
        }
        self.length = len(samples)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def build_sequence_samples(
    transitions: Sequence[Dict[str, np.ndarray]],
    horizon: int,
    stride: int = 1,
) -> List[Dict[str, np.ndarray]]:
    """Create fixed-horizon windows without crossing episode boundaries."""

    if horizon < 2:
        raise ValueError("A multi-step horizon must be at least 2")
    if stride < 1:
        raise ValueError("Sequence stride must be positive")

    episodes: Dict[int, List[Dict[str, np.ndarray]]] = {}
    for fallback_index, transition in enumerate(transitions):
        episode_id = int(np.asarray(transition["episode_id"]).item())
        item = dict(transition)
        item.setdefault("transition_id", np.asarray(fallback_index, dtype=np.int64))
        episodes.setdefault(episode_id, []).append(item)

    sequences: List[Dict[str, np.ndarray]] = []
    for episode_id, rows in sorted(episodes.items()):
        rows.sort(key=lambda row: int(np.asarray(row["transition_id"]).item()))
        for start in range(0, len(rows) - horizon + 1, stride):
            window = rows[start : start + horizon]
            ids = [int(np.asarray(row["transition_id"]).item()) for row in window]
            if any(right != left + 1 for left, right in zip(ids, ids[1:])):
                continue
            if any(bool(np.asarray(row.get("done", 0.0)).item()) for row in window[:-1]):
                continue
            first = window[0]
            if "next_node_features" not in first:
                raise KeyError("Transitions must include next_node_features for multi-step training")
            sequence = {
                    "episode_id": np.asarray(episode_id, dtype=np.int64),
                    "start_transition_id": np.asarray(ids[0], dtype=np.int64),
                    "agent_features": first["agent_features"].astype(np.float32),
                    "node_features": first["node_features"].astype(np.float32),
                    "adjacency_matrix": first["adjacency_matrix"].astype(np.float32),
                    "global_features": first["global_features"].astype(np.float32),
                    "actions": np.stack([row["actions"] for row in window]).astype(np.int64),
                    "target_agent_features": np.stack(
                        [row["next_agent_features"] for row in window]
                    ).astype(np.float32),
                    "target_node_features": np.stack(
                        [row["next_node_features"] for row in window]
                    ).astype(np.float32),
                    "target_global_features": np.stack(
                        [row["next_global_features"] for row in window]
                    ).astype(np.float32),
                    "target_kpi": np.stack([row["kpi"] for row in window]).astype(np.float32),
                    "target_physics_kpi": np.stack(
                        [row["physics_kpi"] for row in window]
                    ).astype(np.float32),
                    "target_congestion_kpi": np.stack(
                        [
                            row.get("congestion_kpi", np.zeros(2, dtype=np.float32))
                            for row in window
                        ]
                    ).astype(np.float32),
                    "done": np.asarray(
                        [float(np.asarray(row.get("done", 0.0)).item()) for row in window],
                        dtype=np.float32,
                    ),
                }
            if "future_congestion_risk" in first:
                sequence["target_future_congestion_risk"] = np.stack(
                    [row["future_congestion_risk"] for row in window]
                ).astype(np.float32)
                sequence["target_future_congestion_risk_mask"] = np.stack(
                    [row["future_congestion_risk_mask"] for row in window]
                ).astype(np.float32)
            if "future_terminal_kpi" in first:
                sequence["target_future_terminal_kpi"] = np.stack(
                    [row["future_terminal_kpi"] for row in window]
                ).astype(np.float32)
                sequence["target_future_terminal_kpi_mask"] = np.stack(
                    [row["future_terminal_kpi_mask"] for row in window]
                ).astype(np.float32)
            sequences.append(sequence)
    if not sequences:
        raise ValueError("No valid multi-step sequences could be constructed")
    return sequences


class PhysicsInformedGraphWorldModelMultiStep(nn.Module):
    """Graph world model trained for autoregressive physical-state imagination."""

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
        self.action_encoder = nn.Sequential(
            nn.Linear(metadata.agv_count * metadata.action_dim, hidden), nn.SiLU()
        )
        self.global_encoder = nn.Sequential(nn.Linear(metadata.global_dim, hidden), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden), nn.SiLU()
        )
        self.agent_delta_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, metadata.agent_dim)
        )
        self.node_delta_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, metadata.node_dim)
        )
        self.global_delta_head = nn.Linear(hidden, metadata.global_dim)
        self.kpi_head = nn.Linear(hidden, len(KPI_NAMES))

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
        action_context = self.action_encoder(action_one_hot.reshape(actions.shape[0], -1))

        query = self.node_query(node_tokens)
        key = self.node_key(node_tokens)
        # Keep graph attention normalization in FP32 because a large negative
        # adjacency mask cannot be represented by CUDA FP16.
        graph_tokens = stable_graph_attention(query, key, node_tokens, adjacency)

        fleet_context = agent_tokens.mean(dim=1)
        graph_context = graph_tokens.mean(dim=1)
        global_context = self.global_encoder(global_features)
        latent = self.fusion(
            torch.cat([fleet_context, graph_context, global_context, action_context], dim=-1)
        )

        expanded_agent_latent = latent.unsqueeze(1).expand(-1, self.metadata.agv_count, -1)
        expanded_agent_graph = graph_context.unsqueeze(1).expand(-1, self.metadata.agv_count, -1)
        agent_delta = self.agent_delta_head(
            torch.cat([agent_tokens, expanded_agent_graph, expanded_agent_latent], dim=-1)
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
            output = self.forward_step({**state, "adjacency_matrix": adjacency, "actions": actions[:, step]})
            predicted_agents.append(output["next_agent_features"])
            predicted_nodes.append(output["next_node_features"])
            predicted_globals.append(output["next_global_features"])
            predicted_kpis.append(output["kpi"])
            predicted_state = self._bounded_state(output)

            if self.training and teacher_forcing_ratio > 0.0 and "target_agent_features" in batch:
                use_truth = (
                    torch.rand(actions.shape[0], 1, 1, device=actions.device) < teacher_forcing_ratio
                )
                use_truth_global = use_truth.squeeze(-1)
                state = {
                    "agent_features": torch.where(
                        use_truth, batch["target_agent_features"][:, step], predicted_state["agent_features"]
                    ),
                    "node_features": torch.where(
                        use_truth, batch["target_node_features"][:, step], predicted_state["node_features"]
                    ),
                    "global_features": torch.where(
                        use_truth_global,
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


def multistep_world_model_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    physics_weight: float = 0.35,
    discount: float = 0.9,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    horizon = output["pred_kpi"].shape[1]
    weights = torch.pow(
        torch.as_tensor(discount, dtype=torch.float32, device=output["pred_kpi"].device),
        torch.arange(horizon, device=output["pred_kpi"].device, dtype=torch.float32),
    )
    weights = weights / weights.sum()

    def discounted_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_step = (prediction - target.float()).pow(2).flatten(start_dim=2).mean(dim=2)
        return (per_step * weights.unsqueeze(0)).sum(dim=1).mean()

    agent_loss = discounted_mse(output["pred_agent_features"], batch["target_agent_features"])
    node_loss = discounted_mse(output["pred_node_features"], batch["target_node_features"])
    global_loss = discounted_mse(output["pred_global_features"], batch["target_global_features"])
    kpi_loss = discounted_mse(output["pred_kpi"], batch["target_kpi"])
    physics_loss = discounted_mse(
        output["pred_kpi"][:, :, [1, 2, 3]], batch["target_physics_kpi"][:, :, [1, 2, 3]]
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


def candidate_joint_actions(
    env: Any,
    *,
    allow_proactive_yield: bool = False,
    allow_proactive_charge: bool = False,
) -> List[np.ndarray]:
    base = baseline_dt_aware_action(env)
    candidates: List[np.ndarray] = [base.copy()]
    base_physics = estimate_action_physics(env, base)
    require_positive_progress = (
        any(env._current_job(i) is not None for i in range(env.agv_count))
        and base_physics["progress_m"] > 0.0
        and base_physics["blocked_count"] <= 0.0
        and base_physics["conflict_events"] <= 0.0
    )
    allow_unprompted_yield = (
        base_physics["blocked_count"] > 0.0 or base_physics["conflict_events"] > 0.0
    )
    proactive_charge_soc = charge_start_soc(env) + MPC_UTILITY_WEIGHTS["proactive_charge_margin_soc"]
    charge_allowed = np.asarray(
        [
            int(base[i]) == 3
            or float(env.agv_batteries[i]) <= charge_start_soc(env)
            or (
                allow_proactive_charge
                and not bool(env._agv_loaded(i))
                and float(env.agv_batteries[i]) <= proactive_charge_soc
            )
            for i in range(env.agv_count)
        ],
        dtype=bool,
    )
    for i in range(env.agv_count):
        for action in range(4):
            # Returning to charge at high SOC is physically feasible but
            # operationally inadmissible because it creates route oscillation.
            if action == 3 and not charge_allowed[i]:
                continue
            # Action 2 is a passing-buffer yield, not a normal routing mode.
            if action == 2 and int(base[i]) != 2 and not allow_unprompted_yield:
                proactive_yield_is_admissible = (
                    allow_proactive_yield
                    and env._current_job(i) is not None
                    and not bool(env._agv_loaded(i))
                    and int(env.agv_positions[i]) != int(env.PASSING_BUFFER_NODE)
                )
                if not proactive_yield_is_admissible:
                    continue
            candidate = base.copy()
            candidate[i] = action
            candidates.append(candidate)
    candidates.extend(
        [
            np.ones(env.agv_count, dtype=np.int64),
            np.zeros(env.agv_count, dtype=np.int64),
        ]
    )
    if allow_unprompted_yield:
        candidates.append(np.full(env.agv_count, 2, dtype=np.int64))
    # A fleet-wide proactive charge request is inadmissible when the charger
    # cannot serve the whole fleet. Mandatory low-SOC actions remain available
    # through the analytical baseline and single-agent candidates.
    if bool(charge_allowed.all()) and int(env.config.charge_node_capacity) >= env.agv_count:
        candidates.append(np.full(env.agv_count, 3, dtype=np.int64))
    unique: List[np.ndarray] = []
    seen = set()
    for candidate in candidates:
        if (
            not allow_unprompted_yield
            and not allow_proactive_yield
            and np.any((candidate == 2) & (base != 2))
        ):
            continue
        # If the safe dispatch baseline can advance assigned work, a fleet-wide
        # zero-progress action is dominated. Mixed yielding remains admissible.
        if require_positive_progress:
            candidate_physics = estimate_action_physics(env, candidate)
            if candidate_physics["progress_m"] <= 0.0:
                continue
        key = tuple(int(value) for value in candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate.astype(np.int64))
    return unique


def analytical_future_conflict_agents(
    env: Any,
    base_actions: np.ndarray | None = None,
    horizon: int = 3,
) -> set[int]:
    """Identify topology-supported route threats under the fallback action modes."""

    actions = (
        baseline_dt_aware_action(env)
        if base_actions is None
        else np.asarray(base_actions, dtype=np.int64).reshape(env.agv_count)
    )
    paths: List[List[int]] = []
    edge_paths: List[List[int | None]] = []
    for i in range(env.agv_count):
        current = int(env.agv_positions[i])
        target = int(env._target_for_action(i, int(actions[i])))
        node_path = [current]
        edge_path: List[int | None] = []
        for _ in range(max(int(horizon), 1)):
            nxt = int(env._next_node_on_shortest_path(current, target))
            edge = env.edge_by_pair.get((current, nxt)) if nxt != current else None
            edge_path.append(None if edge is None else int(edge.edge_id))
            node_path.append(nxt)
            current = nxt
        paths.append(node_path)
        edge_paths.append(edge_path)

    threatened: set[int] = set()
    for depth in range(1, max(int(horizon), 1) + 1):
        for i in range(env.agv_count):
            for j in range(i + 1, env.agv_count):
                same_node = paths[i][depth] == paths[j][depth]
                node_is_route_capacity = (
                    paths[i][depth] != int(env.CHARGE_NODE)
                    and env._node_capacity(paths[i][depth]) <= 1
                )
                head_on = (
                    paths[i][depth - 1] == paths[j][depth]
                    and paths[j][depth - 1] == paths[i][depth]
                    and paths[i][depth] != paths[i][depth - 1]
                )
                shared_single_edge = (
                    edge_paths[i][depth - 1] is not None
                    and edge_paths[i][depth - 1] == edge_paths[j][depth - 1]
                    and env._edge_capacity(edge_paths[i][depth - 1]) <= 1
                )
                if (same_node and node_is_route_capacity) or (head_on and shared_single_edge):
                    threatened.update((i, j))
    return threatened


def analytical_charge_staggering_opportunity(
    env: Any,
    base_action: np.ndarray,
    planned_action: np.ndarray,
) -> Tuple[bool, int, int]:
    """Validate a single proactive charge move against charger capacity and SOC."""

    base = np.asarray(base_action, dtype=np.int64).reshape(env.agv_count)
    planned = np.asarray(planned_action, dtype=np.int64).reshape(env.agv_count)
    new_charge_agents = np.flatnonzero((planned == 3) & (base != 3))
    if len(new_charge_agents) != 1:
        return False, 0, 0

    agv_id = int(new_charge_agents[0])
    start_soc = charge_start_soc(env)
    proactive_soc = start_soc + MPC_UTILITY_WEIGHTS["proactive_charge_margin_soc"]
    battery_soc = float(env.agv_batteries[agv_id])
    genuinely_proactive = (
        not bool(env._agv_loaded(agv_id))
        and start_soc < battery_soc <= proactive_soc
    )
    if not genuinely_proactive:
        return False, 0, 0

    occupied = sum(int(position) == int(env.CHARGE_NODE) for position in env.agv_positions)
    incoming = sum(
        int(base[index]) == 3 and int(env.agv_positions[index]) != int(env.CHARGE_NODE)
        for index in range(env.agv_count)
    )
    available_slots = max(int(env.config.charge_node_capacity) - occupied - incoming, 0)
    pressure_count = sum(
        not bool(env._agv_loaded(index))
        and float(env.agv_batteries[index]) <= proactive_soc
        and int(env.agv_positions[index]) != int(env.CHARGE_NODE)
        for index in range(env.agv_count)
    )
    opportunity = (
        available_slots >= 1
        and pressure_count > int(env.config.charge_node_capacity)
    )
    return bool(opportunity), int(pressure_count), int(available_slots)


def classify_override_evidence(
    base_physics: Dict[str, float],
    planned_physics: Dict[str, float],
    predicted_risk_reduction: float,
    risk_gate_threshold: float,
    predicted_energy_reduction_wh: float = 0.0,
    predicted_throughput_delta: float = 0.0,
    predicted_time_increase_sec: float = 0.0,
    energy_gate_threshold_wh: float | None = None,
    throughput_drop_tolerance_sku: float = 0.0,
    time_increase_tolerance_sec: float = 0.0,
    analytical_future_risk: bool = False,
    operational_energy_action: bool = True,
    predicted_charge_queue_reduction: float = 0.0,
    charge_queue_gate_threshold: float | None = None,
    analytical_charge_staggering: bool = False,
    dedicated_charge_gate_required: bool = False,
) -> str:
    """Classify an override using physical safety and conservative learned evidence."""

    if (
        planned_physics["blocked_count"] > base_physics["blocked_count"]
        or planned_physics["conflict_events"] > base_physics["conflict_events"]
    ):
        return "reject_physical"
    if analytical_future_risk and predicted_risk_reduction >= risk_gate_threshold:
        return "accept_risk"
    progress_preserved = (
        planned_physics.get("progress_m", 0.0) + 1e-9
        >= base_physics.get("progress_m", 0.0)
    )
    energy_evidence = (
        energy_gate_threshold_wh is not None
        and predicted_energy_reduction_wh >= energy_gate_threshold_wh
        and predicted_throughput_delta >= -throughput_drop_tolerance_sku
        and predicted_time_increase_sec <= time_increase_tolerance_sec
        and progress_preserved
        and operational_energy_action
        and not dedicated_charge_gate_required
    )
    if energy_evidence:
        return "accept_energy"
    charge_staggering_evidence = (
        operational_energy_action
        and analytical_charge_staggering
        and charge_queue_gate_threshold is not None
        and predicted_charge_queue_reduction >= charge_queue_gate_threshold
        and predicted_throughput_delta >= -throughput_drop_tolerance_sku
        and progress_preserved
    )
    if charge_staggering_evidence:
        return "accept_charge_stagger"
    return "reject_insufficient_evidence"


def immediate_physical_risk(physics: Dict[str, float], agv_count: int) -> float:
    """Express immediate conflict and blocking in blocked-agent equivalents."""

    return float(physics["blocked_count"] + agv_count * physics["conflict_events"])


def select_physics_only_action(
    env: Any,
    candidates: List[np.ndarray],
    repeated_no_motion: bool,
    agv_count: int,
) -> Tuple[np.ndarray, List[np.ndarray], bool, bool]:
    """Return the shared analytical baseline and its filtered feasible set."""

    base = baseline_dt_aware_action(env).astype(np.int64)
    filtered = list(candidates)
    original_count = len(filtered)
    if repeated_no_motion and any(env._current_job(i) is not None for i in range(env.agv_count)):
        moving = [candidate for candidate in filtered if np.any(candidate != 0)]
        if moving:
            filtered = moving
    anti_stagnation = len(filtered) < original_count

    base_risk = immediate_physical_risk(estimate_action_physics(env, base), agv_count)
    unsafe_filter = False
    selected = base
    if base_risk > 0.0:
        scored = []
        for candidate in filtered:
            physics = estimate_action_physics(env, candidate)
            risk = immediate_physical_risk(physics, agv_count)
            if risk < base_risk:
                scored.append(
                    (
                        risk,
                        -physics["progress_m"],
                        physics["time_sec"],
                        physics["energy_wh"],
                        tuple(int(value) for value in candidate),
                        candidate,
                    )
                )
        if scored:
            scored.sort(key=lambda item: item[:-1])
            selected = scored[0][-1].astype(np.int64)
            selected_risk = immediate_physical_risk(
                estimate_action_physics(env, selected), agv_count
            )
            safer = [
                candidate
                for candidate in filtered
                if immediate_physical_risk(
                    estimate_action_physics(env, candidate), agv_count
                )
                <= selected_risk
            ]
            if safer:
                filtered = safer
            unsafe_filter = True
    if not any(np.array_equal(candidate, selected) for candidate in filtered):
        filtered.append(selected.copy())
    return selected, filtered, anti_stagnation, unsafe_filter


class MultiStepPhysicsInformedMPCPolicy:
    """Receding-horizon beam-search control over graph-world-model imagination."""

    def __init__(
        self,
        model: PhysicsInformedGraphWorldModelMultiStep,
        device: str = "cpu",
        planning_horizon: int = 3,
        beam_width: int = 8,
        discount: float = 0.95,
        risk_gate_threshold: float | None = None,
        energy_gate_threshold_wh: float | None = None,
        throughput_drop_tolerance_sku: float | None = None,
        time_increase_tolerance_sec: float | None = None,
        charge_queue_gate_threshold: float | None = None,
        override_mode: str = "evidence_gated",
    ):
        if planning_horizon < 1:
            raise ValueError("planning_horizon must be at least 1")
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if risk_gate_threshold is not None and risk_gate_threshold < 0.0:
            raise ValueError("risk_gate_threshold must be non-negative")
        if override_mode not in {"evidence_gated", "safe_argmax"}:
            raise ValueError("override_mode must be 'evidence_gated' or 'safe_argmax'")
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.planning_horizon = int(planning_horizon)
        self.beam_width = int(beam_width)
        self.discount = float(discount)
        self.override_mode = override_mode
        self.risk_gate_threshold = float(
            MPC_UTILITY_WEIGHTS["model_risk_reduction_gate"]
            if risk_gate_threshold is None
            else risk_gate_threshold
        )
        self.energy_gate_threshold_wh = float(
            MPC_UTILITY_WEIGHTS["model_energy_reduction_gate_wh"]
            if energy_gate_threshold_wh is None
            else energy_gate_threshold_wh
        )
        self.throughput_drop_tolerance_sku = float(
            MPC_UTILITY_WEIGHTS["model_throughput_drop_tolerance_sku"]
            if throughput_drop_tolerance_sku is None
            else throughput_drop_tolerance_sku
        )
        self.time_increase_tolerance_sec = float(
            MPC_UTILITY_WEIGHTS["model_time_increase_tolerance_s"]
            if time_increase_tolerance_sec is None
            else time_increase_tolerance_sec
        )
        self.charge_queue_gate_threshold = float(
            MPC_UTILITY_WEIGHTS["model_charge_queue_reduction_gate_agent_steps"]
            if charge_queue_gate_threshold is None
            else charge_queue_gate_threshold
        )
        self.last_plan: Dict[str, Any] = {}
        self._last_positions: Tuple[int, ...] | None = None
        self._last_env_time = -1.0

    def _tensor_state(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "agent_features": torch.as_tensor(obs["agent_features"][None], dtype=torch.float32, device=self.device),
            "node_features": torch.as_tensor(obs["node_features"][None], dtype=torch.float32, device=self.device),
            "global_features": torch.as_tensor(obs["global_features"][None], dtype=torch.float32, device=self.device),
            "adjacency_matrix": torch.as_tensor(
                obs["adjacency_matrix"][None], dtype=torch.float32, device=self.device
            ),
        }

    def _utility(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        kpis = output["kpi"] * torch_kpi_scale(self.model.metadata.agv_count, self.device)
        kpis = project_physical_kpis(kpis, self.model.metadata.agv_count)
        reward, dt_sec, energy_wh, blocked, deadlock, throughput = kpis.unbind(dim=1)
        next_global = output["next_global_features"]
        fde = next_global[:, 7] if next_global.shape[1] > 7 else torch.zeros_like(reward)
        battery = next_global[:, 5] if next_global.shape[1] > 5 else torch.zeros_like(reward)
        utility = (
            MPC_UTILITY_WEIGHTS["predicted_reward"] * reward
            + MPC_UTILITY_WEIGHTS["throughput_sku"] * throughput
            + MPC_UTILITY_WEIGHTS["fleet_distribution_entropy"] * fde
            + MPC_UTILITY_WEIGHTS["battery_soc"] * battery
            + MPC_UTILITY_WEIGHTS["predicted_time_s"] * dt_sec.clamp_min(0.0)
            + MPC_UTILITY_WEIGHTS["predicted_energy_wh"] * energy_wh.clamp_min(0.0)
            + MPC_UTILITY_WEIGHTS["predicted_blocked_event"] * blocked.clamp_min(0.0)
            + MPC_UTILITY_WEIGHTS["predicted_deadlock_event"] * deadlock.clamp_min(0.0)
        )
        if "congestion_kpi" in output:
            congestion = output["congestion_kpi"].clamp_min(0.0) * float(
                self.model.metadata.agv_count
            )
            utility = (
                utility
                + MPC_UTILITY_WEIGHTS["predicted_route_blocked_agent_step"] * congestion[:, 0]
                + MPC_UTILITY_WEIGHTS["predicted_charge_queue_agent_step"] * congestion[:, 1]
            )
        return utility

    @torch.no_grad()
    def predict(self, env: Any) -> np.ndarray:
        env_time = float(env.metrics.total_time_sec)
        positions = tuple(int(value) for value in env.agv_positions)
        if env_time <= self._last_env_time:
            self._last_positions = None
        repeated_no_motion = self._last_positions == positions
        candidates = candidate_joint_actions(
            env,
            allow_proactive_yield=True,
            allow_proactive_charge=True,
        )
        base_action_np, candidates, anti_stagnation_applied, unsafe_candidate_filter_applied = (
            select_physics_only_action(
                env,
                candidates,
                repeated_no_motion,
                self.model.metadata.agv_count,
            )
        )
        base_physics_at_root = estimate_action_physics(env, base_action_np)
        future_conflict_agents = analytical_future_conflict_agents(
            env,
            base_actions=base_action_np,
            horizon=self.planning_horizon,
        )
        candidate_tensor = torch.as_tensor(np.stack(candidates), dtype=torch.long, device=self.device)
        physics_baseline_action = torch.as_tensor(
            base_action_np, dtype=torch.long, device=self.device
        )
        initial = self._tensor_state(env._get_obs())
        root_kpis: torch.Tensor | None = None
        beams: List[
            Tuple[
                Dict[str, torch.Tensor],
                List[np.ndarray],
                float,
                List[float],
                np.ndarray,
                np.ndarray,
            ]
        ] = [
            (
                initial,
                [],
                0.0,
                [],
                np.zeros(6, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
            )
        ]

        for depth in range(self.planning_horizon):
            expanded: List[
                Tuple[
                    Dict[str, torch.Tensor],
                    List[np.ndarray],
                    float,
                    List[float],
                    np.ndarray,
                    np.ndarray,
                ]
            ] = []
            for (
                state,
                sequence,
                cumulative_score,
                step_scores,
                cumulative_kpis,
                cumulative_congestion,
            ) in beams:
                count = len(candidates)
                batch = {
                    key: value.repeat(count, *([1] * (value.ndim - 1)))
                    for key, value in state.items()
                }
                batch["actions"] = candidate_tensor
                output = self.model.forward_step(batch)
                utilities = self._utility(output)
                physical_kpis = project_physical_kpis(
                    output["kpi"]
                    * torch_kpi_scale(self.model.metadata.agv_count, self.device),
                    self.model.metadata.agv_count,
                )
                congestion_kpis = (
                    output["congestion_kpi"].clamp_min(0.0)
                    * float(self.model.metadata.agv_count)
                    if "congestion_kpi" in output
                    else torch.zeros((count, 2), dtype=torch.float32, device=self.device)
                )
                if depth == 0:
                    root_kpis = physical_kpis
                    agreement = (
                        candidate_tensor == physics_baseline_action.unsqueeze(0)
                    ).float().mean(dim=1)
                    utilities = utilities + MPC_UTILITY_WEIGHTS["dt_aware_tie_break"] * agreement
                next_state_all = self.model._bounded_state(output)
                for index, candidate in enumerate(candidates):
                    step_score = float(utilities[index].cpu())
                    if depth == 0:
                        physics = estimate_action_physics(env, candidate)
                        step_score += MPC_UTILITY_WEIGHTS["physics_blocked_event"] * physics["blocked_count"]
                        step_score += MPC_UTILITY_WEIGHTS["physics_time_s"] * physics["time_sec"]
                        step_score += MPC_UTILITY_WEIGHTS["physics_energy_wh"] * physics["energy_wh"]
                    next_state = {
                        "agent_features": next_state_all["agent_features"][index : index + 1],
                        "node_features": next_state_all["node_features"][index : index + 1],
                        "global_features": next_state_all["global_features"][index : index + 1],
                        "adjacency_matrix": state["adjacency_matrix"],
                    }
                    expanded.append(
                        (
                            next_state,
                            sequence + [candidate],
                            cumulative_score + (self.discount**depth) * step_score,
                            step_scores + [step_score],
                            cumulative_kpis
                            + (self.discount**depth)
                            * physical_kpis[index].detach().cpu().numpy().astype(np.float64),
                            cumulative_congestion
                            + (self.discount**depth)
                            * congestion_kpis[index].detach().cpu().numpy().astype(np.float64),
                        )
                    )
            expanded.sort(key=lambda item: item[2], reverse=True)
            beams = expanded[: self.beam_width]

        (
            best_state,
            best_sequence,
            best_score,
            best_step_scores,
            best_cumulative_kpis,
            best_cumulative_congestion,
        ) = beams[0]
        del best_state
        baseline_state = initial
        baseline_cumulative_kpis = np.zeros(6, dtype=np.float64)
        baseline_cumulative_congestion = np.zeros(2, dtype=np.float64)
        for depth in range(self.planning_horizon):
            baseline_batch = dict(baseline_state)
            baseline_batch["actions"] = physics_baseline_action.unsqueeze(0)
            baseline_output = self.model.forward_step(baseline_batch)
            baseline_kpis = project_physical_kpis(
                baseline_output["kpi"]
                * torch_kpi_scale(self.model.metadata.agv_count, self.device),
                self.model.metadata.agv_count,
            )[0]
            baseline_cumulative_kpis += (
                (self.discount**depth)
                * baseline_kpis.detach().cpu().numpy().astype(np.float64)
            )
            if "congestion_kpi" in baseline_output:
                baseline_congestion = (
                    baseline_output["congestion_kpi"][0].clamp_min(0.0)
                    * float(self.model.metadata.agv_count)
                )
                baseline_cumulative_congestion += (
                    (self.discount**depth)
                    * baseline_congestion.detach().cpu().numpy().astype(np.float64)
                )
            bounded = self.model._bounded_state(baseline_output)
            baseline_state = {
                **bounded,
                "adjacency_matrix": baseline_state["adjacency_matrix"],
            }
        planned_action = best_sequence[0].astype(np.int64)
        base_action = physics_baseline_action.detach().cpu().numpy().astype(np.int64)
        risk_reduction = 0.0
        energy_reduction_wh = 0.0
        throughput_delta = 0.0
        time_increase_sec = 0.0
        route_blocking_reduction = 0.0
        charge_queue_reduction = 0.0
        risk_gate_applied = False
        energy_gate_applied = False
        physical_gate_applied = False
        override_accepted = False
        override_evidence = "baseline"
        deviates_from_baseline = not np.array_equal(planned_action, base_action)
        base_physics = estimate_action_physics(env, base_action)
        planned_physics = estimate_action_physics(env, planned_action)
        (
            analytical_charge_staggering,
            charge_pressure_count,
            available_charge_slots,
        ) = analytical_charge_staggering_opportunity(
            env,
            base_action,
            planned_action,
        )
        operational_energy_action = bool(analytical_charge_staggering)
        base_is_immediately_safe = (
            base_physics["blocked_count"] <= 0.0
            and base_physics["conflict_events"] <= 0.0
        )
        if deviates_from_baseline and base_is_immediately_safe and root_kpis is not None:
            blocked_reduction = float(
                baseline_cumulative_kpis[3] - best_cumulative_kpis[3]
            )
            deadlock_reduction = float(
                baseline_cumulative_kpis[4] - best_cumulative_kpis[4]
            )
            risk_reduction = blocked_reduction + self.model.metadata.agv_count * deadlock_reduction
            energy_reduction_wh = float(
                baseline_cumulative_kpis[2] - best_cumulative_kpis[2]
            )
            throughput_delta = float(
                best_cumulative_kpis[5] - baseline_cumulative_kpis[5]
            )
            time_increase_sec = float(
                best_cumulative_kpis[1] - baseline_cumulative_kpis[1]
            )
            route_blocking_reduction = float(
                baseline_cumulative_congestion[0] - best_cumulative_congestion[0]
            )
            charge_queue_reduction = float(
                baseline_cumulative_congestion[1] - best_cumulative_congestion[1]
            )
            gate_decision = classify_override_evidence(
                base_physics,
                planned_physics,
                risk_reduction,
                self.risk_gate_threshold,
                predicted_energy_reduction_wh=energy_reduction_wh,
                predicted_throughput_delta=throughput_delta,
                predicted_time_increase_sec=time_increase_sec,
                energy_gate_threshold_wh=self.energy_gate_threshold_wh,
                throughput_drop_tolerance_sku=self.throughput_drop_tolerance_sku,
                time_increase_tolerance_sec=self.time_increase_tolerance_sec,
                analytical_future_risk=bool(future_conflict_agents),
                operational_energy_action=operational_energy_action,
                predicted_charge_queue_reduction=charge_queue_reduction,
                charge_queue_gate_threshold=self.charge_queue_gate_threshold,
                analytical_charge_staggering=analytical_charge_staggering,
                dedicated_charge_gate_required=hasattr(self.model, "congestion_head"),
            )
            if gate_decision == "reject_physical":
                planned_action = base_action.copy()
                physical_gate_applied = True
            elif self.override_mode == "safe_argmax":
                override_accepted = True
                override_evidence = "accept_safe_argmax"
            elif gate_decision == "reject_insufficient_evidence":
                planned_action = base_action.copy()
                risk_gate_applied = True
                energy_gate_applied = True
            else:
                override_accepted = True
                override_evidence = gate_decision
        self.last_plan = {
            "horizon": self.planning_horizon,
            "beam_width": self.beam_width,
            "score": best_score,
            "step_scores": best_step_scores,
            "sequence": [action.tolist() for action in best_sequence],
            "raw_planned_action": best_sequence[0].tolist(),
            "executed_planned_action": planned_action.tolist(),
            "baseline_action": base_action.tolist(),
            "predicted_risk_reduction": risk_reduction,
            "predicted_energy_reduction_wh": energy_reduction_wh,
            "predicted_throughput_delta": throughput_delta,
            "predicted_time_increase_sec": time_increase_sec,
            "predicted_route_blocking_reduction_agent_steps": route_blocking_reduction,
            "predicted_charge_queue_reduction_agent_steps": charge_queue_reduction,
            "risk_gate_threshold": self.risk_gate_threshold,
            "energy_gate_threshold_wh": self.energy_gate_threshold_wh,
            "charge_queue_gate_threshold_agent_steps": self.charge_queue_gate_threshold,
            "risk_gate_applied": risk_gate_applied,
            "energy_gate_applied": energy_gate_applied,
            "physical_gate_applied": physical_gate_applied,
            "override_accepted": override_accepted,
            "override_evidence": override_evidence,
            "override_mode": self.override_mode,
            "analytical_future_conflict_agent_count": len(future_conflict_agents),
            "operational_energy_action": operational_energy_action,
            "analytical_charge_staggering": analytical_charge_staggering,
            "dedicated_charge_gate_required": hasattr(self.model, "congestion_head"),
            "analytical_charge_pressure_agent_count": charge_pressure_count,
            "available_charge_slots": available_charge_slots,
            "anti_stagnation_applied": anti_stagnation_applied,
            "unsafe_candidate_filter_applied": unsafe_candidate_filter_applied,
            "candidate_count": len(candidates),
        }
        self._last_positions = positions
        self._last_env_time = env_time
        return planned_action

    def predict_guarded(self, env: Any) -> np.ndarray:
        """Return the first receding-horizon action; the shared industrial guard is applied by the runner."""

        return self.predict(env)


class PhysicsOnlyRiskPolicy:
    """Ablation using the same analytical feasible set without learned rollouts."""

    def __init__(self, agv_count: int = 3):
        self.agv_count = int(agv_count)
        self._last_positions: Tuple[int, ...] | None = None
        self._last_env_time = -1.0
        self.last_plan: Dict[str, Any] = {}

    def predict_guarded(self, env: Any) -> np.ndarray:
        env_time = float(env.metrics.total_time_sec)
        positions = tuple(int(value) for value in env.agv_positions)
        if env_time <= self._last_env_time:
            self._last_positions = None
        repeated_no_motion = self._last_positions == positions
        base = baseline_dt_aware_action(env).astype(np.int64)
        candidates = candidate_joint_actions(
            env,
            allow_proactive_yield=True,
            allow_proactive_charge=True,
        )
        selected, candidates, anti_stagnation, unsafe_filter = select_physics_only_action(
            env,
            candidates,
            repeated_no_motion,
            self.agv_count,
        )
        override = not np.array_equal(selected, base)
        self.last_plan = {
            "raw_planned_action": selected.tolist(),
            "executed_planned_action": selected.tolist(),
            "baseline_action": base.tolist(),
            "predicted_risk_reduction": 0.0,
            "predicted_energy_reduction_wh": 0.0,
            "predicted_throughput_delta": 0.0,
            "predicted_time_increase_sec": 0.0,
            "risk_gate_threshold": 0.0,
            "energy_gate_threshold_wh": 0.0,
            "risk_gate_applied": False,
            "energy_gate_applied": False,
            "physical_gate_applied": False,
            "override_accepted": override,
            "override_evidence": "immediate_physics" if override else "baseline",
            "analytical_future_conflict_agent_count": 0,
            "anti_stagnation_applied": anti_stagnation,
            "unsafe_candidate_filter_applied": unsafe_filter,
            "candidate_count": len(candidates),
        }
        self._last_positions = positions
        self._last_env_time = env_time
        return selected


def save_multistep_world_model(
    path: Path,
    model: PhysicsInformedGraphWorldModelMultiStep,
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
            "mpc_utility_weights": dict(MPC_UTILITY_WEIGHTS),
        },
        path,
    )


def load_multistep_world_model_policy(
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
    model = PhysicsInformedGraphWorldModelMultiStep(metadata)
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
