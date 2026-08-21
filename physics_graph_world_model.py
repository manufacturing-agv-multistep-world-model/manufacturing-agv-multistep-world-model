from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from jms_parameter_registry import MPC_UTILITY_WEIGHTS, WORLD_MODEL_DEFAULTS


KPI_NAMES = (
    "reward",
    "delta_time_sec",
    "delta_energy_wh",
    "blocked_delta",
    "deadlock_delta",
    "throughput_delta",
)
CONGESTION_KPI_NAMES = (
    "route_blocked_agent_steps",
    "charge_queue_blocked_agent_steps",
)


@dataclass
class WorldModelMetadata:
    agv_count: int
    node_count: int
    agent_dim: int
    node_dim: int
    global_dim: int
    action_dim: int = 4
    hidden_dim: int = 96


def kpi_scale(agv_count: int) -> np.ndarray:
    return np.array([100.0, 100.0, 20.0, max(float(agv_count), 1.0), 1.0, 1.0], dtype=np.float32)


def torch_kpi_scale(agv_count: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(kpi_scale(agv_count), dtype=torch.float32, device=device)


class PhysicsInformedGraphWorldModel(nn.Module):
    """Predicts one-step AGV DT dynamics from graph state and joint action."""

    def __init__(self, metadata: WorldModelMetadata):
        super().__init__()
        self.metadata = metadata
        hidden = metadata.hidden_dim

        self.agent_encoder = nn.Sequential(
            nn.Linear(metadata.agent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(metadata.node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.node_query = nn.Linear(hidden, hidden, bias=False)
        self.node_key = nn.Linear(hidden, hidden, bias=False)
        self.action_encoder = nn.Sequential(
            nn.Linear(metadata.agv_count * metadata.action_dim, hidden),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(metadata.global_dim, hidden),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.agent_head = nn.Linear(hidden, metadata.agv_count * metadata.agent_dim)
        self.global_head = nn.Linear(hidden, metadata.global_dim)
        self.kpi_head = nn.Linear(hidden, len(KPI_NAMES))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        agent_tokens = self.agent_encoder(batch["agent_features"].float())
        node_tokens = self.node_encoder(batch["node_features"].float())
        adjacency = batch["adjacency_matrix"].float()
        actions = batch["actions"].long()

        action_one_hot = torch.nn.functional.one_hot(actions, num_classes=self.metadata.action_dim).float()
        action_context = self.action_encoder(action_one_hot.reshape(actions.shape[0], -1))

        query = self.node_query(node_tokens)
        key = self.node_key(node_tokens)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
        logits = logits.masked_fill(adjacency <= 0.0, -1e9)
        graph_tokens = torch.matmul(torch.softmax(logits, dim=-1), node_tokens)

        fleet_context = agent_tokens.mean(dim=1)
        graph_context = graph_tokens.mean(dim=1)
        global_context = self.global_encoder(batch["global_features"].float())
        latent = self.fusion(torch.cat([fleet_context, graph_context, global_context, action_context], dim=-1))

        return {
            "next_agent_features": self.agent_head(latent).reshape(
                -1, self.metadata.agv_count, self.metadata.agent_dim
            ),
            "next_global_features": self.global_head(latent),
            "kpi": self.kpi_head(latent),
        }


class WorldModelTransitionDataset(Dataset):
    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        if not samples:
            raise ValueError("WorldModelTransitionDataset requires at least one sample")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            key: torch.as_tensor(value)
            for key, value in sample.items()
        }


def battery_reserve_wh(env: Any) -> float:
    return max(0.06 * float(env.config.battery_capacity_wh), 70.0)


def charge_start_soc(env: Any) -> float:
    return max(float(env.config.low_battery_soc) + 3.0, 21.0)


def baseline_dt_aware_action(env: Any) -> np.ndarray:
    """A safe industrial policy used for data collection and MPC fallback."""

    actions = np.ones(env.agv_count, dtype=np.int64)
    for i, position in enumerate(env.agv_positions):
        loaded = bool(env._agv_loaded(i))
        job = env._current_job(i)
        battery_wh = env.agv_batteries[i] * env.config.battery_capacity_wh / 100.0
        if not loaded and env.agv_batteries[i] < charge_start_soc(env):
            actions[i] = 0 if position == env.CHARGE_NODE else 3
            continue
        if loaded and job is not None:
            needed_wh = estimate_route_energy_wh(env, i, job.destination, loaded=True)
            if battery_wh < needed_wh + battery_reserve_wh(env):
                actions[i] = 0 if position == env.CHARGE_NODE else 3
                continue
        if job is None:
            actions[i] = 0
            continue

        target = env._target_for_action(i, 1)
        next_node = env._next_node_on_shortest_path(position, target)
        next_occupied = any(p == next_node for j, p in enumerate(env.agv_positions) if j != i)
        if (
            not loaded
            and env.PASSING_BUFFER_NODE != position
            and env._node_capacity(next_node) <= 1
            and next_occupied
            and env.wait_steps[i] >= 8
        ):
            actions[i] = 2
        else:
            actions[i] = 1
    return actions


def behavior_action(env: Any, rng: np.random.Generator, exploration_rate: float = 0.25) -> np.ndarray:
    action = baseline_dt_aware_action(env)
    if rng.random() < exploration_rate:
        for i in range(env.agv_count):
            if rng.random() < 0.45:
                action[i] = int(rng.integers(0, 4))
    elif rng.random() < 0.35:
        i = int(rng.integers(0, env.agv_count))
        action[i] = int(rng.choice([0, 1, 2, 3]))
    return action.astype(np.int64)


def estimate_action_physics(env: Any, action: np.ndarray) -> Dict[str, float]:
    action = np.asarray(action, dtype=np.int64).reshape(env.agv_count)
    proposals, targets, edge_ids = env._propose_positions(action)
    blocked, conflict_events = env._detect_conflicts(proposals, edge_ids)
    agent_times = np.full(env.agv_count, env.config.wait_time_s, dtype=np.float64)
    energy_by_agent = np.zeros(env.agv_count, dtype=np.float64)
    progress_m = 0.0
    moving_count = 0.0

    for i, proposed in enumerate(proposals):
        current = env.agv_positions[i]
        if blocked[i]:
            energy_by_agent[i] = env._idle_energy(env.config.wait_time_s)
            continue

        job = env._current_job(i)
        handling_time = 0.0
        if int(action[i]) == 1 and job is not None:
            reaches_origin = env.agv_phases[i] == "to_origin" and proposed == job.origin
            reaches_destination = (
                env.agv_phases[i] == "to_destination" and proposed == job.destination
            )
            if reaches_origin or reaches_destination:
                handling_time = env._handling_time()

        if proposed == current:
            agent_times[i] = max(env.config.wait_time_s, handling_time)
            energy_by_agent[i] = env._idle_energy(agent_times[i])
            continue

        moving_count += 1.0
        edge = env.edge_by_pair.get((current, proposed))
        if edge is None:
            energy_by_agent[i] = env._idle_energy(env.config.wait_time_s)
            continue
        changed = env.last_actions[i] not in (0, int(action[i]))
        travel_time = env.get_kinematic_time(edge.distance_m, edge.speed_limit_mps, changed)
        agent_times[i] = max(env.config.wait_time_s, travel_time + handling_time)
        loaded = bool(env._agv_loaded(i))
        energy_by_agent[i] = env._move_energy(
            edge.distance_m,
            travel_time,
            i,
            loaded=loaded,
            action_changed=changed,
        )
        energy_by_agent[i] += env._idle_energy(handling_time)
        progress_m += env._path_distance(current, targets[i]) - env._path_distance(proposed, targets[i])

    step_time_sec = float(max(env.config.wait_time_s, float(agent_times.max())))
    residual_idle_sec = np.maximum(0.0, step_time_sec - agent_times)
    energy_by_agent += np.asarray(
        [env._idle_energy(float(duration)) for duration in residual_idle_sec],
        dtype=np.float64,
    )

    return {
        "time_sec": step_time_sec,
        "energy_wh": float(energy_by_agent.sum()),
        "blocked_count": float(blocked.sum()),
        "conflict_events": float(conflict_events),
        "progress_m": float(progress_m),
        "moving_count": float(moving_count),
    }


def estimate_route_energy_wh(env: Any, agv_id: int, target: int, loaded: bool, start: int | None = None) -> float:
    current = env.agv_positions[agv_id] if start is None else int(start)
    energy_wh = 0.0
    visited = set()
    action_changed = env.last_actions[agv_id] not in (0, 1) if start is None else False
    while current != target and current not in visited and len(visited) <= len(env.node_map):
        visited.add(current)
        nxt = env._next_node_on_shortest_path(current, target)
        if nxt == current:
            break
        edge = env.edge_by_pair.get((current, nxt))
        if edge is None:
            break
        travel_time = env.get_kinematic_time(edge.distance_m, edge.speed_limit_mps, action_changed)
        energy_wh += env._move_energy(
            edge.distance_m,
            travel_time,
            agv_id,
            loaded=loaded,
            action_changed=action_changed,
        )
        current = nxt
        action_changed = False
    return float(energy_wh)


def estimate_loaded_mission_energy_wh(env: Any, agv_id: int, job: Any) -> float:
    """Energy needed to deliver the current load and still reach the charger."""

    to_destination = estimate_route_energy_wh(env, agv_id, job.destination, loaded=True)
    destination_to_charge = estimate_route_energy_wh(
        env,
        agv_id,
        env.CHARGE_NODE,
        loaded=False,
        start=job.destination,
    )
    return float(to_destination + destination_to_charge)


def make_transition_sample(
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    next_obs: Dict[str, np.ndarray],
    reward: float,
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    physics: Dict[str, float],
    agv_count: int,
    episode_id: int,
    transition_id: int = 0,
    done: bool = False,
) -> Dict[str, np.ndarray]:
    scale = kpi_scale(agv_count)
    kpi = np.array(
        [
            float(reward),
            float(after_metrics["real_time_sec"]) - float(before_metrics["real_time_sec"]),
            float(after_metrics["total_energy_wh"]) - float(before_metrics["total_energy_wh"]),
            float(after_metrics["blocked_count"]) - float(before_metrics["blocked_count"]),
            float(after_metrics["deadlock_count"]) - float(before_metrics["deadlock_count"]),
            float(after_metrics["throughput"]) - float(before_metrics["throughput"]),
        ],
        dtype=np.float32,
    )
    physics_kpi = np.array(
        [
            0.0,
            physics["time_sec"],
            physics["energy_wh"],
            physics["blocked_count"],
            physics["conflict_events"],
            max(physics["progress_m"], 0.0) / 25.0,
        ],
        dtype=np.float32,
    )
    congestion_kpi = np.array(
        [
            float(after_metrics.get("route_blocked_agent_steps", 0.0))
            - float(before_metrics.get("route_blocked_agent_steps", 0.0)),
            float(after_metrics.get("charge_queue_blocked_agent_steps", 0.0))
            - float(before_metrics.get("charge_queue_blocked_agent_steps", 0.0)),
        ],
        dtype=np.float32,
    )
    return {
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "transition_id": np.asarray(transition_id, dtype=np.int64),
        "done": np.asarray(done, dtype=np.float32),
        "agent_features": obs["agent_features"].astype(np.float32),
        "node_features": obs["node_features"].astype(np.float32),
        "adjacency_matrix": obs["adjacency_matrix"].astype(np.float32),
        "global_features": obs["global_features"].astype(np.float32),
        "actions": action.astype(np.int64),
        "next_agent_features": next_obs["agent_features"].astype(np.float32),
        "next_node_features": next_obs["node_features"].astype(np.float32),
        "next_global_features": next_obs["global_features"].astype(np.float32),
        "kpi": kpi / scale,
        "physics_kpi": physics_kpi / scale,
        # Agent-step counts are bounded by fleet size at each transition.
        "congestion_kpi": congestion_kpi / max(float(agv_count), 1.0),
    }


def collect_world_model_transitions(
    env_factory: Any,
    episodes: int,
    max_steps: int,
    seed: int,
    exploration_rate: float = 0.25,
) -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, np.ndarray]] = []
    for episode in range(episodes):
        env = env_factory(seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        for transition_id in range(max_steps):
            before_metrics = env.summary()
            action = behavior_action(env, rng, exploration_rate=exploration_rate)
            physics = estimate_action_physics(env, action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            after_metrics = info["metrics"]
            samples.append(
                make_transition_sample(
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    reward=float(reward),
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    physics=physics,
                    agv_count=env.agv_count,
                    episode_id=episode,
                    transition_id=transition_id,
                    done=bool(terminated or truncated),
                )
            )
            obs = next_obs
            if terminated or truncated:
                break
    return samples


def world_model_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    agv_count: int,
    physics_weight: float = WORLD_MODEL_DEFAULTS["physics_weight"],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mse = nn.functional.mse_loss
    state_loss = mse(output["next_agent_features"], batch["next_agent_features"].float())
    global_loss = mse(output["next_global_features"], batch["next_global_features"].float())
    kpi_loss = mse(output["kpi"], batch["kpi"].float())
    physics_loss = mse(output["kpi"][:, [1, 2, 3]], batch["physics_kpi"].float()[:, [1, 2, 3]])
    total = state_loss + global_loss + 4.0 * kpi_loss + physics_weight * physics_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "state_loss": float(state_loss.detach().cpu()),
        "global_loss": float(global_loss.detach().cpu()),
        "kpi_loss": float(kpi_loss.detach().cpu()),
        "physics_loss": float(physics_loss.detach().cpu()),
    }


class PhysicsInformedMPCPolicy:
    """One-step model-predictive dispatcher using the learned graph world model."""

    def __init__(self, model: PhysicsInformedGraphWorldModel, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)

    def predict(self, env: Any) -> np.ndarray:
        obs = env._get_obs()
        candidates = self._candidate_actions(env)
        batch = {
            "agent_features": torch.as_tensor(
                np.repeat(obs["agent_features"][None, ...], len(candidates), axis=0),
                dtype=torch.float32,
            ).to(self.device),
            "node_features": torch.as_tensor(
                np.repeat(obs["node_features"][None, ...], len(candidates), axis=0),
                dtype=torch.float32,
            ).to(self.device),
            "adjacency_matrix": torch.as_tensor(
                np.repeat(obs["adjacency_matrix"][None, ...], len(candidates), axis=0),
                dtype=torch.float32,
            ).to(self.device),
            "global_features": torch.as_tensor(
                np.repeat(obs["global_features"][None, ...], len(candidates), axis=0),
                dtype=torch.float32,
            ).to(self.device),
            "actions": torch.as_tensor(np.stack(candidates), dtype=torch.long, device=self.device),
        }
        with torch.no_grad():
            output = self.model(batch)

        kpis = output["kpi"] * torch_kpi_scale(env.agv_count, self.device)
        next_global = output["next_global_features"]
        scores = []
        base = baseline_dt_aware_action(env)
        base_index = 0
        physics_rows = []
        for idx, candidate in enumerate(candidates):
            if np.array_equal(candidate, base):
                base_index = idx
            reward, dt_sec, energy_wh, blocked, deadlock, throughput = kpis[idx].detach().cpu().numpy()
            physics = estimate_action_physics(env, candidate)
            physics_rows.append(physics)
            fde = float(next_global[idx, 7].detach().cpu()) if next_global.shape[1] > 7 else 0.0
            battery = float(next_global[idx, 5].detach().cpu()) if next_global.shape[1] > 5 else 0.0
            score = (
                MPC_UTILITY_WEIGHTS["predicted_reward"] * reward
                + MPC_UTILITY_WEIGHTS["throughput_sku"] * throughput
                + MPC_UTILITY_WEIGHTS["fleet_distribution_entropy"] * fde
                + MPC_UTILITY_WEIGHTS["battery_soc"] * battery
                + MPC_UTILITY_WEIGHTS["predicted_time_s"] * max(dt_sec, 0.0)
                + MPC_UTILITY_WEIGHTS["predicted_energy_wh"] * max(energy_wh, 0.0)
                + MPC_UTILITY_WEIGHTS["predicted_blocked_event"] * max(blocked, 0.0)
                + MPC_UTILITY_WEIGHTS["predicted_deadlock_event"] * max(deadlock, 0.0)
                + MPC_UTILITY_WEIGHTS["physics_blocked_event"] * physics["blocked_count"]
                + MPC_UTILITY_WEIGHTS["physics_time_s"] * physics["time_sec"]
                + MPC_UTILITY_WEIGHTS["physics_energy_wh"] * physics["energy_wh"]
            )
            if np.array_equal(candidate, base):
                score += MPC_UTILITY_WEIGHTS["dt_aware_tie_break"]
            scores.append(float(score))
        best_index = int(np.argmax(scores))
        base_physics = physics_rows[base_index]
        if base_physics["blocked_count"] > 0:
            safer = [
                idx
                for idx, physics in enumerate(physics_rows)
                if physics["blocked_count"] < base_physics["blocked_count"]
            ]
            if safer:
                best_index = max(
                    safer,
                    key=lambda idx: (
                        -physics_rows[idx]["blocked_count"],
                        physics_rows[idx]["progress_m"],
                        scores[idx],
                    ),
                )
        if best_index != base_index:
            best_physics = physics_rows[best_index]
            unsafe = (
                scores[best_index] < scores[base_index] + 3.0
                and best_physics["blocked_count"] >= base_physics["blocked_count"] - 0.1
            ) or (
                best_physics["blocked_count"] > base_physics["blocked_count"] + 0.1
            ) or (
                best_physics["progress_m"] < base_physics["progress_m"] - 2.0
                and best_physics["progress_m"] <= 0.0
                and best_physics["blocked_count"] >= base_physics["blocked_count"]
            ) or (
                base_physics["moving_count"] > 0
                and best_physics["moving_count"] <= 0
                and best_physics["blocked_count"] >= base_physics["blocked_count"]
            )
            if unsafe:
                best_index = base_index
        return candidates[best_index].astype(np.int64)

    def predict_guarded(
        self,
        env: Any,
        target_delivery_rate: float = 0.30,
        min_guard_step: int = 0,
    ) -> np.ndarray:
        """MPC with a throughput guard for safety-throughput tradeoff experiments.

        The base MPC is deliberately conservative. This guard keeps its safety
        screening, but temporarily falls back to the DT-aware industrial rule
        when the episode is falling behind a minimum delivery pace.
        """

        mpc_action = self.predict(env)

        service_action = mpc_action.copy()
        for i, position in enumerate(env.agv_positions):
            if env._current_job(i) is not None and position == env._target_for_action(i, 1):
                service_action[i] = 1
        if not np.array_equal(service_action, mpc_action):
            return service_action.astype(np.int64)

        emergency_action = mpc_action.copy()
        for i, position in enumerate(env.agv_positions):
            loaded = bool(env._agv_loaded(i))
            battery_wh = env.agv_batteries[i] * env.config.battery_capacity_wh / 100.0
            critical_soc = max(8.0, 0.5 * env.config.low_battery_soc)
            if position == env.CHARGE_NODE and not loaded and env.agv_batteries[i] < 80.0:
                emergency_action[i] = 0
                continue
            if env.agv_batteries[i] < critical_soc:
                emergency_action[i] = 0 if position == env.CHARGE_NODE else 3
                continue
            if not loaded and env.agv_batteries[i] < charge_start_soc(env):
                emergency_action[i] = 0 if position == env.CHARGE_NODE else 3
                continue
            job = env._current_job(i)
            if loaded and job is not None:
                needed_wh = estimate_loaded_mission_energy_wh(env, i, job)
                reserve_wh = battery_reserve_wh(env)
                if battery_wh < needed_wh + reserve_wh:
                    emergency_action[i] = 0 if position == env.CHARGE_NODE else 3
        if not np.array_equal(emergency_action, mpc_action):
            return emergency_action.astype(np.int64)

        soft_deadlock_steps = max(int(getattr(env.config, "deadlock_soft_steps", 6)), 1)
        deadlock_pressure = int(getattr(env, "deadlock_timer", 0)) >= max(1, soft_deadlock_steps // 3)
        if deadlock_pressure:
            candidates = self._candidate_actions(env)
            physics_rows = [estimate_action_physics(env, candidate) for candidate in candidates]
            viable = [
                idx
                for idx, physics in enumerate(physics_rows)
                if physics["moving_count"] > 0.0
                and physics["progress_m"] > 0.0
                and physics["blocked_count"] <= max(float(env.agv_count - 1), 1.0)
            ]
            if viable:
                best_index = max(
                    viable,
                    key=lambda idx: (
                        physics_rows[idx]["progress_m"],
                        -physics_rows[idx]["blocked_count"],
                        -physics_rows[idx]["conflict_events"],
                        -physics_rows[idx]["time_sec"],
                    ),
                )
                return candidates[best_index].astype(np.int64)

        step = int(getattr(env, "current_step", 0))
        throughput = float(getattr(env, "throughput", 0))
        if step < min_guard_step:
            return mpc_action

        active_jobs = sum(
            1
            for job in getattr(env, "jobs", [])
            if getattr(job, "status", None) in {"waiting", "assigned", "in_transit"}
        )
        target_throughput = max(1.0, target_delivery_rate * float(step))
        if active_jobs <= 0 or throughput >= target_throughput:
            return mpc_action

        base_action = baseline_dt_aware_action(env)
        if np.array_equal(base_action, mpc_action):
            return mpc_action

        base_physics = estimate_action_physics(env, base_action)
        mpc_physics = estimate_action_physics(env, mpc_action)
        urgent = throughput < target_throughput
        base_has_controlled_progress = (
            base_physics["moving_count"] > 0.0
            and base_physics["progress_m"] > 0.0
            and base_physics["blocked_count"] <= max(float(env.agv_count - 1), 1.0)
            and base_physics["conflict_events"] <= max(float(env.agv_count - 1), 1.0)
        )
        urgent_base_is_safe = (
            base_has_controlled_progress
            or (
                base_physics["conflict_events"] <= max(1.0, mpc_physics["conflict_events"])
                and base_physics["blocked_count"] <= mpc_physics["blocked_count"] + 1.0
            )
        )
        if urgent and urgent_base_is_safe:
            return base_action.astype(np.int64)

        deadlock_pressure = int(getattr(env, "deadlock_timer", 0)) >= max(2, soft_deadlock_steps // 2)

        if deadlock_pressure and base_physics["blocked_count"] > mpc_physics["blocked_count"]:
            return mpc_action

        base_is_progressive = (
            base_physics["conflict_events"] <= 0.0
            and base_physics["moving_count"] >= max(1.0, mpc_physics["moving_count"])
            and base_physics["progress_m"] >= mpc_physics["progress_m"] - 0.5
        )
        base_is_tolerably_safe = base_physics["blocked_count"] <= max(1.0, mpc_physics["blocked_count"] + 0.1)

        if base_is_progressive and base_is_tolerably_safe and urgent:
            return base_action.astype(np.int64)
        if base_is_progressive and base_physics["blocked_count"] <= mpc_physics["blocked_count"] + 0.1:
            return base_action.astype(np.int64)

        return mpc_action

    def _candidate_actions(self, env: Any) -> List[np.ndarray]:
        base = baseline_dt_aware_action(env)
        candidates: List[np.ndarray] = [base.copy()]
        for i in range(env.agv_count):
            for action in range(4):
                candidate = base.copy()
                candidate[i] = action
                candidates.append(candidate)
        for i in range(env.agv_count):
            candidate = np.zeros(env.agv_count, dtype=np.int64)
            candidate[i] = int(base[i])
            candidates.append(candidate)
            candidate = np.full(env.agv_count, 2, dtype=np.int64)
            candidate[i] = int(base[i])
            candidates.append(candidate)
        candidates.append(np.ones(env.agv_count, dtype=np.int64))
        candidates.append(np.zeros(env.agv_count, dtype=np.int64))
        unique: List[np.ndarray] = []
        seen = set()
        for candidate in candidates:
            key = tuple(int(x) for x in candidate)
            if key not in seen:
                unique.append(candidate)
                seen.add(key)
        return unique


def save_world_model(
    path: Path,
    model: PhysicsInformedGraphWorldModel,
    metadata: WorldModelMetadata,
    history: List[Dict[str, float]],
    args: Dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata.__dict__,
            "history": history,
            "args": args or {},
            "mpc_utility_weights": dict(MPC_UTILITY_WEIGHTS),
            "kpi_names": KPI_NAMES,
        },
        path,
    )


def load_world_model_policy(path: str | Path, device: str = "cpu") -> PhysicsInformedMPCPolicy:
    checkpoint = torch.load(Path(path), map_location=device)
    metadata = WorldModelMetadata(**checkpoint["metadata"])
    model = PhysicsInformedGraphWorldModel(metadata)
    model.load_state_dict(checkpoint["state_dict"])
    return PhysicsInformedMPCPolicy(model, device=device)
