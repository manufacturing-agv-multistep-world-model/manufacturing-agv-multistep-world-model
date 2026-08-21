from __future__ import annotations

import csv
import hashlib
import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from jms_parameter_registry import env_default_values


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENARIO_DIR = ROOT / "agv-test2" / "simplified_cad_scenario"
ENV_DEFAULTS = env_default_values()


def canonical_dispatch_rule(rule: str) -> str:
    """Keep old experiment files runnable while publishing the safer Route-B name."""

    aliases = {
        "dt_marl": "dt_aware",
    }
    return aliases.get(rule, rule)


@dataclass(frozen=True)
class ScenarioNode:
    node_id: int
    name: str
    x_m: float
    y_m: float
    role: str
    cad_basis: str = ""
    note: str = ""


@dataclass(frozen=True)
class ScenarioEdge:
    edge_id: int
    from_node: int
    from_name: str
    to_node: int
    to_name: str
    distance_m: float
    edge_type: str
    bidirectional: bool
    capacity_baseline: int
    capacity_stress: int
    speed_limit_mps: float
    note: str = ""


@dataclass(frozen=True)
class TaskTemplate:
    task_id: str
    origin: str
    destination: str
    task_class: str
    steady_rate_per_hour: float
    rush_rate_per_hour: float
    load_kg: float
    priority: int


@dataclass
class Job:
    job_id: int
    template_id: str
    origin: int
    destination: int
    task_class: str
    load_kg: float
    priority: int
    release_time_sec: float
    assigned_agv: int | None = None
    pickup_time_sec: float | None = None
    completion_time_sec: float | None = None
    status: str = "waiting"  # waiting, assigned, in_transit, done


@dataclass
class ScenarioData:
    nodes: List[ScenarioNode]
    edges: List[ScenarioEdge]
    task_templates: List[TaskTemplate]


@dataclass
class DigitalTwinConfig:
    """Configuration for the CAD-derived AGV digital-twin experiment."""

    parameter_profile: str = "jms_interpretable_v2_energy_calibrated"
    agv_count: int = 3
    max_steps: int = 2000
    env_variant: str = "full"  # ideal, kinematics, full
    reward_mode: str = "hybrid"  # individual, global, hybrid
    velocity_profile: str = "s_curve"  # ideal, trapezoid, s_curve
    scenario: str = "steady"  # steady, rush
    dispatch_rule: str = "dt_aware"  # fcfs, nearest, priority, dt_aware
    capacity_mode: str = "stress"  # baseline, stress
    arrival_process: str = "poisson"  # deterministic, poisson
    arrival_rate_multiplier: float = 1.0
    seed: int | None = None
    scenario_dir: Path | str = DEFAULT_SCENARIO_DIR
    initial_backlog_per_template: int = 0

    speed_max_mps: float = float(ENV_DEFAULTS["speed_max_mps"])
    edge_speed_multiplier: float = 1.0
    acceleration_mps2: float = float(ENV_DEFAULTS["acceleration_mps2"])
    jerk_mps3: float = float(ENV_DEFAULTS["jerk_mps3"])
    wait_time_s: float = float(ENV_DEFAULTS["wait_time_s"])
    lift_time_s: float = float(ENV_DEFAULTS["lift_time_s"])
    rotate_time_s: float = float(ENV_DEFAULTS["rotate_time_s"])
    ideal_move_time_s: float = 1.0

    battery_capacity_wh: float = float(ENV_DEFAULTS["battery_capacity_wh"])
    base_drive_wh_per_s: float = float(ENV_DEFAULTS["base_drive_wh_per_s"])
    rolling_wh_per_m: float = float(ENV_DEFAULTS["rolling_wh_per_m"])
    acceleration_wh: float = float(ENV_DEFAULTS["acceleration_wh"])
    idle_wh_per_s: float = float(ENV_DEFAULTS["idle_wh_per_s"])
    loaded_energy_factor: float = float(ENV_DEFAULTS["loaded_energy_factor"])
    charge_soc_per_min: float = float(ENV_DEFAULTS["charge_soc_per_min"])
    low_battery_soc: float = float(ENV_DEFAULTS["low_battery_soc"])
    charge_node_capacity: int = int(ENV_DEFAULTS["charge_node_capacity"])

    deadlock_soft_steps: int = 6
    deadlock_hard_steps: int = 40
    min_required_throughput: int = 3
    # No release cap by default. Use a positive value only for smoke tests.
    max_released_jobs: int | None = None

    delivery_reward: float = float(ENV_DEFAULTS["delivery_reward"])
    pickup_reward: float = float(ENV_DEFAULTS["pickup_reward"])
    conflict_penalty: float = float(ENV_DEFAULTS["conflict_penalty"])
    deadlock_penalty: float = float(ENV_DEFAULTS["deadlock_penalty"])
    out_of_battery_penalty: float = float(ENV_DEFAULTS["out_of_battery_penalty"])
    time_penalty: float = float(ENV_DEFAULTS["time_penalty"])
    energy_penalty: float = float(ENV_DEFAULTS["energy_penalty"])
    wait_penalty: float = float(ENV_DEFAULTS["wait_penalty"])
    assignment_bonus: float = float(ENV_DEFAULTS["assignment_bonus"])
    progress_reward_per_m: float = float(ENV_DEFAULTS["progress_reward_per_m"])
    wrong_progress_penalty_per_m: float = float(ENV_DEFAULTS["wrong_progress_penalty_per_m"])
    deadlock_recovery_penalty: float = float(ENV_DEFAULTS["deadlock_recovery_penalty"])


@dataclass
class EpisodeMetrics:
    total_time_sec: float = 0.0
    total_energy_wh: float = 0.0
    throughput: int = 0
    conflict_count: int = 0
    blocking_onset_count: int = 0
    blocked_count: int = 0
    blocked_time_sec: float = 0.0
    route_blocking_onset_count: int = 0
    route_blocked_agent_steps: int = 0
    route_blocked_time_sec: float = 0.0
    charge_queue_onset_count: int = 0
    charge_queue_blocked_agent_steps: int = 0
    charge_queue_time_sec: float = 0.0
    deadlock_count: int = 0
    out_of_battery_count: int = 0
    timeout_count: int = 0
    deadlock_resolution_times: List[float] = field(default_factory=list)
    completed_wait_times: List[float] = field(default_factory=list)
    completed_cycle_times: List[float] = field(default_factory=list)


def _as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def _as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value == "" else int(float(value))


def load_scenario_data(scenario_dir: Path | str = DEFAULT_SCENARIO_DIR) -> ScenarioData:
    """Load the final simplified CAD scenario generated for this project."""

    directory = Path(scenario_dir)
    node_path = directory / "simplified_nodes.csv"
    edge_path = directory / "simplified_edges.csv"
    task_path = directory / "simplified_task_flows.csv"

    if not node_path.exists() or not edge_path.exists() or not task_path.exists():
        raise FileNotFoundError(
            "Simplified CAD scenario files are missing. Run draw_simplified_cad_scenario.py first."
        )

    with node_path.open("r", encoding="utf-8-sig") as f:
        nodes = [
            ScenarioNode(
                node_id=_as_int(row, "node_id"),
                name=row["node_name"],
                x_m=_as_float(row, "x_m"),
                y_m=_as_float(row, "y_m"),
                role=row["role"],
                cad_basis=row.get("cad_basis", ""),
                note=row.get("note", ""),
            )
            for row in csv.DictReader(f)
        ]

    with edge_path.open("r", encoding="utf-8-sig") as f:
        edges = [
            ScenarioEdge(
                edge_id=_as_int(row, "edge_id"),
                from_node=_as_int(row, "from_node"),
                from_name=row["from_name"],
                to_node=_as_int(row, "to_node"),
                to_name=row["to_name"],
                distance_m=_as_float(row, "distance_m"),
                edge_type=row["edge_type"],
                bidirectional=bool(_as_int(row, "bidirectional", 1)),
                capacity_baseline=_as_int(row, "capacity_baseline", 1),
                capacity_stress=_as_int(row, "capacity_stress", 1),
                speed_limit_mps=_as_float(row, "speed_limit_mps", 1.2),
                note=row.get("note", ""),
            )
            for row in csv.DictReader(f)
        ]

    with task_path.open("r", encoding="utf-8-sig") as f:
        task_templates = [
            TaskTemplate(
                task_id=row["task_id"],
                origin=row["origin"],
                destination=row["destination"],
                task_class=row["task_class"],
                steady_rate_per_hour=_as_float(row, "steady_rate_per_hour"),
                rush_rate_per_hour=_as_float(row, "rush_rate_per_hour"),
                load_kg=_as_float(row, "load_kg"),
                priority=_as_int(row, "priority", 1),
            )
            for row in csv.DictReader(f)
        ]

    return ScenarioData(nodes=nodes, edges=edges, task_templates=task_templates)


class AGVDigitalTwinEnv(gym.Env):
    """High-fidelity graph-based AGV digital twin for the CAD-derived case."""

    metadata = {"render_modes": []}

    def __init__(self, config: DigitalTwinConfig | None = None):
        super().__init__()
        self.config = config or DigitalTwinConfig()
        self.config.dispatch_rule = canonical_dispatch_rule(self.config.dispatch_rule)
        self._validate_config()
        self.scenario_data = load_scenario_data(self.config.scenario_dir)

        self.nodes = self.scenario_data.nodes
        self.edges = self.scenario_data.edges
        self.task_templates = self.scenario_data.task_templates
        self.node_map = [node.name for node in self.nodes]
        self.node_roles = {node.node_id: node.role for node in self.nodes}
        self.node_by_name = {node.name: node.node_id for node in self.nodes}
        self.edge_by_pair: Dict[Tuple[int, int], ScenarioEdge] = {}
        self.graph = self._build_graph()
        self.adjacency_matrix = self._build_adjacency_matrix()
        self.next_hop_cache: Dict[Tuple[int, int], int] = {}

        self.agv_count = self.config.agv_count
        self.WAREHOUSE_NODE = self.node_by_name["B"]
        self.A_NODE = self.node_by_name["A"]
        self.CHARGE_NODE = self.node_by_name["Charge"]
        self.PASSING_BUFFER_NODE = self.node_by_name["PassingBuffer"]
        self.home_nodes = [self.node_by_name[f"Home{i}"] for i in range(1, 4)]
        self.AGV_SPEED_MAX = self.config.speed_max_mps
        self.ACCEL = self.config.acceleration_mps2
        self.max_route_distance = max(self._graph_diameter_estimate(), 1.0)

        self.action_space = spaces.MultiDiscrete([4] * self.agv_count)
        self.agent_feature_dim = 10
        self.node_feature_dim = 7
        node_count = len(self.node_map)
        self.observation_space = spaces.Dict(
            {
                "agent_features": spaces.Box(
                    low=-10.0,
                    high=100.0,
                    shape=(self.agv_count, self.agent_feature_dim),
                    dtype=np.float32,
                ),
                "node_features": spaces.Box(
                    low=-10.0,
                    high=100.0,
                    shape=(node_count, self.node_feature_dim),
                    dtype=np.float32,
                ),
                "adjacency_matrix": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(node_count, node_count),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    low=-100.0,
                    high=1000.0,
                    shape=(10,),
                    dtype=np.float32,
                ),
            }
        )

        self.np_random = np.random.default_rng(self.config.seed)
        self.reset(seed=self.config.seed)

    def _validate_config(self) -> None:
        variants = {"ideal", "kinematics", "full"}
        reward_modes = {"individual", "global", "hybrid"}
        profiles = {"ideal", "trapezoid", "s_curve"}
        scenarios = {"steady", "rush"}
        dispatch_rules = {"fcfs", "nearest", "priority", "dt_aware"}
        capacity_modes = {"baseline", "stress"}
        arrival_processes = {"deterministic", "poisson"}
        if self.config.env_variant not in variants:
            raise ValueError(f"env_variant must be one of {sorted(variants)}")
        if self.config.reward_mode not in reward_modes:
            raise ValueError(f"reward_mode must be one of {sorted(reward_modes)}")
        if self.config.velocity_profile not in profiles:
            raise ValueError(f"velocity_profile must be one of {sorted(profiles)}")
        if self.config.scenario not in scenarios:
            raise ValueError(f"scenario must be one of {sorted(scenarios)}")
        if self.config.dispatch_rule not in dispatch_rules:
            raise ValueError(f"dispatch_rule must be one of {sorted(dispatch_rules)}")
        if self.config.capacity_mode not in capacity_modes:
            raise ValueError(f"capacity_mode must be one of {sorted(capacity_modes)}")
        if self.config.arrival_process not in arrival_processes:
            raise ValueError(f"arrival_process must be one of {sorted(arrival_processes)}")
        if self.config.arrival_rate_multiplier <= 0:
            raise ValueError("arrival_rate_multiplier must be positive")
        if self.config.edge_speed_multiplier <= 0:
            raise ValueError("edge_speed_multiplier must be positive")
        if int(self.config.charge_node_capacity) != self.config.charge_node_capacity:
            raise ValueError("charge_node_capacity must be an integer")
        if self.config.charge_node_capacity <= 0:
            raise ValueError("charge_node_capacity must be positive")

    def _build_graph(self) -> Dict[int, List[Tuple[int, int, float]]]:
        graph: Dict[int, List[Tuple[int, int, float]]] = {}
        for edge in self.edges:
            graph.setdefault(edge.from_node, []).append((edge.to_node, edge.edge_id, edge.distance_m))
            self.edge_by_pair[(edge.from_node, edge.to_node)] = edge
            if edge.bidirectional:
                graph.setdefault(edge.to_node, []).append((edge.from_node, edge.edge_id, edge.distance_m))
                self.edge_by_pair[(edge.to_node, edge.from_node)] = edge
        for node in self.nodes:
            graph.setdefault(node.node_id, [])
        return graph

    def _build_adjacency_matrix(self) -> np.ndarray:
        n = len(self.nodes)
        adjacency = np.eye(n, dtype=np.float32)
        for edge in self.edges:
            adjacency[edge.from_node, edge.to_node] = 1.0
            if edge.bidirectional:
                adjacency[edge.to_node, edge.from_node] = 1.0
        return adjacency

    def _graph_diameter_estimate(self) -> float:
        total = 0.0
        for source in range(len(self.nodes)):
            distances = self._dijkstra_distances(source)
            finite = [d for d in distances.values() if math.isfinite(d)]
            if finite:
                total = max(total, max(finite))
        return total

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        resolved_seed = seed if seed is not None else self.config.seed
        if resolved_seed is None:
            resolved_seed = int(self.np_random.integers(0, 2**31 - 1))
        arrival_seed_sequences = np.random.SeedSequence(int(resolved_seed)).spawn(
            len(self.task_templates)
        )
        self.arrival_rng_by_template = {
            template.task_id: np.random.default_rng(template_seed)
            for template, template_seed in zip(
                self.task_templates, arrival_seed_sequences
            )
        }

        self.agv_positions = [self.home_nodes[i % len(self.home_nodes)] for i in range(self.agv_count)]
        self.agv_batteries = [100.0 for _ in range(self.agv_count)]
        self.agv_job_ids: List[int | None] = [None for _ in range(self.agv_count)]
        self.agv_phases: List[str] = ["idle" for _ in range(self.agv_count)]
        self.last_actions = [0 for _ in range(self.agv_count)]
        self.wait_steps = [0 for _ in range(self.agv_count)]
        self.previously_blocked = np.zeros(self.agv_count, dtype=bool)
        self.previously_route_blocked = np.zeros(self.agv_count, dtype=bool)
        self.previously_charge_queued = np.zeros(self.agv_count, dtype=bool)
        self.completed_by_agent = np.zeros(self.agv_count, dtype=np.int32)
        self.distance_by_agent = np.zeros(self.agv_count, dtype=np.float64)
        self.empty_distance_by_agent = np.zeros(self.agv_count, dtype=np.float64)
        self.loaded_distance_by_agent = np.zeros(self.agv_count, dtype=np.float64)
        self.energy_by_agent = np.zeros(self.agv_count, dtype=np.float64)
        self.busy_time_by_agent = np.zeros(self.agv_count, dtype=np.float64)
        self.node_visit_counts = np.zeros(len(self.node_map), dtype=np.float64)
        self.node_occupancy_time_sec = np.zeros(len(self.node_map), dtype=np.float64)

        self.current_step = 0
        self.throughput = 0
        self.deadlock_timer = 0
        self.deadlock_active = False
        self.deadlock_start_time = 0.0
        self.metrics = EpisodeMetrics()
        self.last_attention = np.zeros((self.agv_count, self.agv_count), dtype=np.float32)
        self.jobs: List[Job] = []
        self.next_job_id = 0
        self.next_arrival_by_template = self._initial_arrival_times()
        self._seed_initial_backlog()
        self._release_jobs()
        self._assign_jobs()
        return self._get_obs(), self._info()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(self.agv_count)
        actions = np.clip(actions, 0, 3)
        self.current_step += 1
        assigned_this_step = 0

        old_positions = list(self.agv_positions)
        proposals, targets, edge_ids = self._propose_positions(actions)
        blocked, conflict_events = self._detect_conflicts(proposals, edge_ids)
        charge_queued = blocked & np.asarray(
            [
                proposed == self.CHARGE_NODE and old_positions[i] != self.CHARGE_NODE
                for i, proposed in enumerate(proposals)
            ],
            dtype=bool,
        )
        route_blocked = blocked & ~charge_queued

        local_rewards = np.zeros(self.agv_count, dtype=np.float64)
        moved = np.zeros(self.agv_count, dtype=bool)
        charging_progress = False
        step_times = [self.config.wait_time_s]
        agent_action_time_sec = np.full(self.agv_count, self.config.wait_time_s, dtype=np.float64)
        travel_time_by_agent = np.zeros(self.agv_count, dtype=np.float64)
        charging_agents = np.zeros(self.agv_count, dtype=bool)
        step_energy_wh = np.zeros(self.agv_count, dtype=np.float64)
        deliveries_this_step = 0
        pickups_this_step = 0

        for i, action in enumerate(actions):
            current = self.agv_positions[i]
            proposed = proposals[i]
            loaded = self._agv_loaded(i)

            if blocked[i]:
                self.wait_steps[i] += 1
                local_rewards[i] -= self.config.conflict_penalty
                step_energy_wh[i] += self._idle_energy(self.config.wait_time_s)
                continue

            if proposed == current:
                station_service = (
                    int(action) == 1
                    and self._current_job(i) is not None
                    and current == self._target_for_action(i, 1)
                )
                if station_service:
                    pickup, delivered, handling_time = self._handle_job_event(
                        i,
                        event_time_sec=self.metrics.total_time_sec + self._handling_time(),
                    )
                    pickups_this_step += pickup
                    deliveries_this_step += delivered
                    if pickup:
                        local_rewards[i] += self.config.pickup_reward
                    if delivered:
                        local_rewards[i] += self.config.delivery_reward * self._scenario_pressure()
                    if pickup or delivered:
                        self.wait_steps[i] = 0
                        agent_action_time_sec[i] = max(self.config.wait_time_s, handling_time)
                        if handling_time > 0:
                            step_times.append(handling_time)
                            self.busy_time_by_agent[i] += handling_time
                        step_energy_wh[i] += self._idle_energy(max(self.config.wait_time_s, handling_time))
                        continue

                if action in (0, 3) and current == self.CHARGE_NODE:
                    charging_agents[i] = True
                else:
                    self.wait_steps[i] += 1
                local_rewards[i] -= self.config.wait_penalty
                step_energy_wh[i] += self._idle_energy(self.config.wait_time_s)
                continue

            edge = self.edge_by_pair[(current, proposed)]
            distance = edge.distance_m
            action_changed = self.last_actions[i] not in (0, int(action))
            travel_time = self.get_kinematic_time(distance, edge.speed_limit_mps, action_changed)
            agent_time = travel_time
            travel_time_by_agent[i] = travel_time

            self.agv_positions[i] = proposed
            self.wait_steps[i] = 0
            moved[i] = True
            self.distance_by_agent[i] += distance
            self.busy_time_by_agent[i] += travel_time

            if loaded:
                self.loaded_distance_by_agent[i] += distance
            else:
                self.empty_distance_by_agent[i] += distance

            progress_m = self._path_distance(current, targets[i]) - self._path_distance(proposed, targets[i])
            if progress_m >= 0:
                local_rewards[i] += self.config.progress_reward_per_m * progress_m
            else:
                local_rewards[i] += self.config.wrong_progress_penalty_per_m * progress_m

            local_rewards[i] -= self.config.time_penalty * travel_time
            energy = self._move_energy(distance, travel_time, i, loaded=loaded, action_changed=action_changed)
            step_energy_wh[i] += energy
            local_rewards[i] -= self.config.energy_penalty * energy

            pickup, delivered, handling_time = self._handle_job_event(
                i,
                event_time_sec=self.metrics.total_time_sec + travel_time + self._handling_time(),
            )
            pickups_this_step += pickup
            deliveries_this_step += delivered
            if pickup:
                local_rewards[i] += self.config.pickup_reward
            if delivered:
                local_rewards[i] += self.config.delivery_reward * self._scenario_pressure()
            if handling_time > 0:
                agent_time += handling_time
                self.busy_time_by_agent[i] += handling_time
                step_energy_wh[i] += self._idle_energy(handling_time)
            step_times.append(agent_time)
            agent_action_time_sec[i] = max(self.config.wait_time_s, agent_time)

        step_time_sec = float(max(step_times))

        # Every AGV experiences the same global physical interval. Account for
        # residual idle/charging time when another AGV has the longest action.
        residual_idle_sec = np.maximum(0.0, step_time_sec - agent_action_time_sec)
        for i, residual_sec in enumerate(residual_idle_sec):
            if residual_sec > 0.0:
                step_energy_wh[i] += self._idle_energy(float(residual_sec))
        for i in np.flatnonzero(charging_agents):
            battery_before = self.agv_batteries[i]
            self._charge(int(i), step_time_sec)
            if self.agv_batteries[i] > battery_before:
                charging_progress = True
                self.wait_steps[i] = 0
            else:
                self.wait_steps[i] += 1

        self._apply_energy(step_energy_wh)
        self.metrics.total_time_sec += step_time_sec
        # Exogenous arrivals and dispatch decisions become part of the next
        # observable state; the action selected for this step cannot act on a
        # job that was invisible when that action was chosen.
        self._release_jobs()
        assigned_this_step = self._assign_jobs()
        if assigned_this_step:
            local_rewards += (
                self.config.assignment_bonus * assigned_this_step / max(self.agv_count, 1)
            )
        self.metrics.total_energy_wh += float(step_energy_wh.sum())
        self.metrics.conflict_count += conflict_events
        blocked_agent_steps = int(blocked.sum())
        self.metrics.blocking_onset_count += int(np.sum(blocked & ~self.previously_blocked))
        self.metrics.blocked_count += blocked_agent_steps
        self.metrics.blocked_time_sec += blocked_agent_steps * step_time_sec
        route_blocked_steps = int(route_blocked.sum())
        charge_queue_steps = int(charge_queued.sum())
        self.metrics.route_blocking_onset_count += int(
            np.sum(route_blocked & ~self.previously_route_blocked)
        )
        self.metrics.route_blocked_agent_steps += route_blocked_steps
        self.metrics.route_blocked_time_sec += route_blocked_steps * step_time_sec
        self.metrics.charge_queue_onset_count += int(
            np.sum(charge_queued & ~self.previously_charge_queued)
        )
        self.metrics.charge_queue_blocked_agent_steps += charge_queue_steps
        self.metrics.charge_queue_time_sec += charge_queue_steps * step_time_sec
        self.previously_blocked = blocked.copy()
        self.previously_route_blocked = route_blocked.copy()
        self.previously_charge_queued = charge_queued.copy()
        self.node_visit_counts += np.bincount(self.agv_positions, minlength=len(self.node_map))
        self._accumulate_node_occupancy_time(
            old_positions,
            moved,
            travel_time_by_agent,
            step_time_sec,
        )
        self.last_attention = self._attention_proxy(proposals, targets)
        self.last_actions = actions.tolist()

        productive_progress = bool(
            moved.any()
            or pickups_this_step
            or deliveries_this_step
            or charging_progress
        )
        deadlock_started = self._update_deadlock_state(
            old_positions,
            moved,
            blocked,
            productive_progress=productive_progress,
        )
        out_of_battery = any(b <= 0.0 for b in self.agv_batteries)
        terminated = False
        truncated = False

        if deadlock_started:
            local_rewards -= self.config.deadlock_penalty / max(self.agv_count, 1)
        if out_of_battery:
            self.metrics.out_of_battery_count += 1
            local_rewards -= self.config.out_of_battery_penalty / max(self.agv_count, 1)
            terminated = True
        deadlock_recovered = False
        if self.deadlock_timer >= self.config.deadlock_hard_steps:
            recovery_time = self._recover_deadlock()
            step_time_sec += recovery_time
            self.metrics.total_time_sec += recovery_time
            # Recovery is physical downtime. Jobs released during that interval
            # must be visible before the next dispatch decision.
            self._release_jobs()
            recovery_assignments = self._assign_jobs()
            if recovery_assignments:
                assigned_this_step += recovery_assignments
                local_rewards += (
                    self.config.assignment_bonus
                    * recovery_assignments
                    / max(self.agv_count, 1)
                )
            local_rewards -= self.config.deadlock_recovery_penalty / max(self.agv_count, 1)
            deadlock_recovered = True
        if self.current_step >= self.config.max_steps:
            truncated = True
            if self.throughput < self.config.min_required_throughput:
                self.metrics.timeout_count = 1

        reward_components = self._reward_components(local_rewards, deliveries_this_step, conflict_events, step_time_sec)
        reward = reward_components["composed_reward"]
        info = self._info(
            step_time_sec=step_time_sec,
            local_rewards=local_rewards,
            reward_components=reward_components,
            deliveries_this_step=deliveries_this_step,
            pickups_this_step=pickups_this_step,
            blocked=blocked,
            conflict_events=conflict_events,
            assigned_this_step=assigned_this_step,
            deadlock_recovered=deadlock_recovered,
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    def _seed_initial_backlog(self) -> None:
        backlog = max(int(self.config.initial_backlog_per_template), 0)
        if backlog <= 0:
            return
        for template in self.task_templates:
            for _ in range(backlog):
                if self._job_release_cap_reached():
                    return
                self.jobs.append(
                    Job(
                        job_id=self.next_job_id,
                        template_id=template.task_id,
                        origin=self.node_by_name[template.origin],
                        destination=self.node_by_name[template.destination],
                        task_class=template.task_class,
                        load_kg=template.load_kg,
                        priority=template.priority,
                        release_time_sec=0.0,
                    )
                )
                self.next_job_id += 1

    def _recover_deadlock(self) -> float:
        agv_id = int(np.argmax(self.wait_steps)) if self.wait_steps else 0
        current = self.agv_positions[agv_id]
        occupancy = np.bincount(self.agv_positions, minlength=len(self.node_map))
        candidates = [self.PASSING_BUFFER_NODE, self.CHARGE_NODE, *self.home_nodes]
        feasible = [node for node in candidates if node == current or occupancy[node] < self._node_capacity(node)]
        if not feasible:
            feasible = candidates
        movable = [node for node in feasible if node != current]
        target_pool = movable if movable else feasible
        target = min(target_pool, key=lambda node: self._path_distance(current, node))
        distance = self._path_distance(current, target)
        recovery_time = self.get_kinematic_time(distance, self.config.speed_max_mps, action_changed=True)
        was_loaded = self._agv_loaded(agv_id)

        positions_before_recovery = list(self.agv_positions)
        self.agv_positions[agv_id] = target
        self.wait_steps[agv_id] = 0
        self.distance_by_agent[agv_id] += distance
        self.empty_distance_by_agent[agv_id] += 0.0 if was_loaded else distance
        self.loaded_distance_by_agent[agv_id] += distance if was_loaded else 0.0
        self.busy_time_by_agent[agv_id] += recovery_time
        recovery_energy = np.asarray(
            [self._idle_energy(recovery_time) for _ in range(self.agv_count)],
            dtype=np.float64,
        )
        recovery_energy[agv_id] = self._move_energy(
            distance,
            recovery_time,
            agv_id,
            loaded=was_loaded,
            action_changed=True,
        )
        self._apply_energy(recovery_energy)
        self.metrics.total_energy_wh += float(recovery_energy.sum())
        for other_id, position in enumerate(self.agv_positions):
            if other_id != agv_id and position == self.CHARGE_NODE:
                self._charge(other_id, recovery_time)

        recovery_moved = np.zeros(self.agv_count, dtype=bool)
        recovery_moved[agv_id] = target != current
        recovery_travel_times = np.zeros(self.agv_count, dtype=np.float64)
        recovery_travel_times[agv_id] = recovery_time
        self._accumulate_node_occupancy_time(
            positions_before_recovery,
            recovery_moved,
            recovery_travel_times,
            recovery_time,
        )

        if self.deadlock_active:
            duration = self.metrics.total_time_sec - self.deadlock_start_time + recovery_time
            self.metrics.deadlock_resolution_times.append(float(duration))
        self.deadlock_timer = 0
        self.deadlock_active = False
        self.node_visit_counts[target] += 1
        return float(recovery_time)

    def _accumulate_node_occupancy_time(
        self,
        old_positions: List[int],
        moved: np.ndarray,
        travel_time_by_agent: np.ndarray,
        step_time_sec: float,
    ) -> None:
        """Accumulate node occupancy in physical seconds for each AGV.

        Travel is split equally between its source and destination nodes; any
        handling or synchronization residual is assigned to the destination.
        This preserves exactly ``agv_count * step_time_sec`` occupancy-seconds.
        """
        for i, current in enumerate(self.agv_positions):
            if moved[i]:
                travel_sec = min(float(travel_time_by_agent[i]), step_time_sec)
                source_sec = 0.5 * travel_sec
                self.node_occupancy_time_sec[old_positions[i]] += source_sec
                self.node_occupancy_time_sec[current] += step_time_sec - source_sec
            else:
                self.node_occupancy_time_sec[current] += step_time_sec

    def _initial_arrival_times(self) -> Dict[str, float]:
        first_arrivals: Dict[str, float] = {}
        for template in self.task_templates:
            rate = max(self._arrival_rate(template), 0.0)
            mean_interval = 3600.0 / rate if rate > 0 else 3600.0
            if self.config.arrival_process == "deterministic":
                first_arrivals[template.task_id] = float(mean_interval)
            else:
                first_arrivals[template.task_id] = self._sample_interarrival(
                    rate, template.task_id
                )
        return first_arrivals

    def _arrival_rate(self, template: TaskTemplate) -> float:
        base_rate = template.rush_rate_per_hour if self.config.scenario == "rush" else template.steady_rate_per_hour
        return float(base_rate * self.config.arrival_rate_multiplier)

    def _sample_interarrival(self, rate_per_hour: float, template_id: str) -> float:
        mean_interval = 3600.0 / max(rate_per_hour, 1e-9)
        if self.config.arrival_process == "deterministic":
            return mean_interval
        return float(
            max(
                1.0,
                self.arrival_rng_by_template[template_id].exponential(mean_interval),
            )
        )

    def _job_release_cap_reached(self) -> bool:
        cap = self.config.max_released_jobs
        return cap is not None and cap > 0 and len(self.jobs) >= cap

    def _release_jobs(self) -> None:
        now = self.metrics.total_time_sec
        while not self._job_release_cap_reached():
            due_templates = [
                template
                for template in self.task_templates
                if self._arrival_rate(template) > 0.0
                and self.next_arrival_by_template[template.task_id] <= now
            ]
            if not due_templates:
                break
            template = min(
                due_templates,
                key=lambda item: (
                    self.next_arrival_by_template[item.task_id],
                    item.task_id,
                ),
            )
            release_time = self.next_arrival_by_template[template.task_id]
            self.jobs.append(
                Job(
                    job_id=self.next_job_id,
                    template_id=template.task_id,
                    origin=self.node_by_name[template.origin],
                    destination=self.node_by_name[template.destination],
                    task_class=template.task_class,
                    load_kg=template.load_kg,
                    priority=template.priority,
                    release_time_sec=release_time,
                )
            )
            self.next_job_id += 1
            self.next_arrival_by_template[template.task_id] += self._sample_interarrival(
                self._arrival_rate(template), template.task_id
            )

    def _assign_jobs(self) -> int:
        waiting = [job for job in self.jobs if job.status == "waiting"]
        if not waiting:
            return 0
        assigned = 0
        for agv_id in range(self.agv_count):
            if not waiting:
                break
            if self.agv_job_ids[agv_id] is not None:
                continue
            fidelity_mode = bool(getattr(self, "fidelity_dispatch_mode", False))
            policy_variant = getattr(self, "policy_variant", self.config.env_variant)
            battery_aware = not fidelity_mode or policy_variant == "full"
            if battery_aware and self.agv_batteries[agv_id] < self.config.low_battery_soc:
                self.agv_phases[agv_id] = "to_charge"
                continue
            job = self._select_job_for_agv(agv_id, waiting)
            if job is None:
                continue
            job.assigned_agv = agv_id
            job.status = "assigned"
            self.agv_job_ids[agv_id] = job.job_id
            self.agv_phases[agv_id] = "to_origin"
            waiting.remove(job)
            assigned += 1
        return assigned

    def _select_job_for_agv(self, agv_id: int, waiting: List[Job]) -> Job | None:
        if not waiting:
            return None
        position = self.agv_positions[agv_id]
        if self.config.dispatch_rule == "fcfs":
            return min(waiting, key=lambda job: (job.release_time_sec, job.priority, job.job_id))
        if self.config.dispatch_rule == "priority":
            return min(waiting, key=lambda job: (job.priority, job.release_time_sec, job.job_id))
        if self.config.dispatch_rule == "nearest":
            return min(waiting, key=lambda job: (self._path_distance(position, job.origin), job.release_time_sec))

        fidelity_mode = bool(getattr(self, "fidelity_dispatch_mode", False))
        policy_variant = getattr(self, "policy_variant", self.config.env_variant)

        def route_kinematic_time(source: int, target: int) -> float:
            current = source
            previous_heading: Tuple[int, int] | None = None
            total_time = 0.0
            visited = set()
            while current != target and current not in visited and len(visited) < len(self.nodes):
                visited.add(current)
                nxt = self._next_node_on_shortest_path(current, target)
                edge = self.edge_by_pair.get((current, nxt))
                if edge is None:
                    break
                source_node = self.nodes[current]
                target_node = self.nodes[nxt]
                dx = target_node.x_m - source_node.x_m
                dy = target_node.y_m - source_node.y_m
                heading = (
                    0 if abs(dx) < 1e-9 else (1 if dx > 0 else -1),
                    0 if abs(dy) < 1e-9 else (1 if dy > 0 else -1),
                )
                turns = previous_heading is not None and heading != previous_heading
                total_time += self.get_kinematic_time(
                    edge.distance_m,
                    edge.speed_limit_mps,
                    action_changed=turns,
                )
                previous_heading = heading
                current = nxt
            return total_time

        def fidelity_score(job: Job) -> float:
            to_origin_distance = self._path_distance(position, job.origin)
            loaded_distance = self._path_distance(job.origin, job.destination)
            if policy_variant == "ideal":
                estimated_time = (to_origin_distance + loaded_distance) / max(self.config.speed_max_mps, 1e-6)
            else:
                estimated_time = (
                    route_kinematic_time(position, job.origin)
                    + route_kinematic_time(job.origin, job.destination)
                    + 2.0 * self.config.lift_time_s
                )

            age = max(0.0, self.metrics.total_time_sec - job.release_time_sec)
            priority_credit = 4.0 / max(float(job.priority), 1.0)
            score = estimated_time - 0.02 * age - priority_credit
            if policy_variant == "full":
                remaining_wh = self.config.battery_capacity_wh * self.agv_batteries[agv_id] / 100.0
                load_factor = 1.0 + (self.config.loaded_energy_factor - 1.0) * min(job.load_kg / 120.0, 1.4)
                estimated_wh = (
                    self.config.rolling_wh_per_m * to_origin_distance
                    + self.config.rolling_wh_per_m * loaded_distance * load_factor
                    + 4.0 * self.config.acceleration_wh
                )
                reserve_wh = 0.15 * self.config.battery_capacity_wh
                energy_shortfall = max(0.0, estimated_wh + reserve_wh - remaining_wh)
                score += 2.0 * energy_shortfall
            return score

        if fidelity_mode:
            return min(waiting, key=lambda job: (fidelity_score(job), job.release_time_sec, job.job_id))

        def dt_score(job: Job) -> float:
            origin_distance = self._path_distance(position, job.origin)
            age = max(0.0, self.metrics.total_time_sec - job.release_time_sec)
            priority_bonus = 1.0 / max(job.priority, 1)
            battery_risk = 25.0 if self.agv_batteries[agv_id] < 35.0 and job.destination != self.CHARGE_NODE else 0.0
            return origin_distance - 0.02 * age - priority_bonus + battery_risk

        return min(waiting, key=lambda job: (dt_score(job), job.release_time_sec, job.job_id))

    def _propose_positions(self, actions: np.ndarray) -> Tuple[List[int], List[int], List[int | None]]:
        proposals: List[int] = []
        targets: List[int] = []
        edge_ids: List[int | None] = []
        for agv_id, action in enumerate(actions):
            current = self.agv_positions[agv_id]
            target = self._target_for_action(agv_id, int(action))
            targets.append(target)
            if target == current or int(action) == 0:
                proposals.append(current)
                edge_ids.append(None)
                continue
            next_node = self._next_node_on_shortest_path(current, target)
            proposals.append(next_node)
            edge = self.edge_by_pair.get((current, next_node))
            edge_ids.append(edge.edge_id if edge else None)
        return proposals, targets, edge_ids

    def _target_for_action(self, agv_id: int, action: int) -> int:
        if action == 3:
            return self.CHARGE_NODE
        if action == 2:
            job = self._current_job(agv_id)
            if job is None or self.agv_phases[agv_id] in {"idle", "to_origin"}:
                return self.PASSING_BUFFER_NODE
        job = self._current_job(agv_id)
        if job is None:
            fidelity_mode = bool(getattr(self, "fidelity_dispatch_mode", False))
            policy_variant = getattr(self, "policy_variant", self.config.env_variant)
            battery_aware = not fidelity_mode or policy_variant == "full"
            if battery_aware and self.agv_batteries[agv_id] < self.config.low_battery_soc:
                return self.CHARGE_NODE
            return self.agv_positions[agv_id]
        phase = self.agv_phases[agv_id]
        if phase == "to_origin":
            return job.origin
        if phase == "to_destination":
            return job.destination
        if phase == "to_charge":
            return self.CHARGE_NODE
        return self.agv_positions[agv_id]

    def _current_job(self, agv_id: int) -> Job | None:
        job_id = self.agv_job_ids[agv_id]
        if job_id is None:
            return None
        return self.jobs[job_id]

    def _agv_loaded(self, agv_id: int) -> bool:
        return self.agv_phases[agv_id] == "to_destination"

    def _handle_job_event(self, agv_id: int, event_time_sec: float | None = None) -> Tuple[int, int, float]:
        job = self._current_job(agv_id)
        if job is None:
            if self.agv_phases[agv_id] == "to_charge" and self.agv_positions[agv_id] == self.CHARGE_NODE:
                self.agv_phases[agv_id] = "idle"
            return 0, 0, 0.0

        timestamp = self.metrics.total_time_sec if event_time_sec is None else float(event_time_sec)

        if self.agv_phases[agv_id] == "to_origin" and self.agv_positions[agv_id] == job.origin:
            job.status = "in_transit"
            job.pickup_time_sec = timestamp
            self.metrics.completed_wait_times.append(job.pickup_time_sec - job.release_time_sec)
            self.agv_phases[agv_id] = "to_destination"
            return 1, 0, self._handling_time()

        if self.agv_phases[agv_id] == "to_destination" and self.agv_positions[agv_id] == job.destination:
            job.status = "done"
            job.completion_time_sec = timestamp
            self.metrics.completed_cycle_times.append(job.completion_time_sec - job.release_time_sec)
            self.agv_job_ids[agv_id] = None
            self.agv_phases[agv_id] = "idle"
            self.throughput += 1
            self.metrics.throughput = self.throughput
            self.completed_by_agent[agv_id] += 1
            return 0, 1, self._handling_time()

        return 0, 0, 0.0

    def _detect_conflicts(self, proposals: List[int], edge_ids: List[int | None]) -> Tuple[np.ndarray, int]:
        blocked = np.zeros(self.agv_count, dtype=bool)
        conflict_events = 0

        def priority_key(agent_id: int) -> Tuple[int, int]:
            return (-int(self.wait_steps[agent_id]), agent_id)

        def block_agents(agent_ids: Iterable[int]) -> bool:
            nonlocal conflict_events
            newly_blocked = [i for i in agent_ids if not blocked[i]]
            if newly_blocked:
                blocked[newly_blocked] = True
                conflict_events += 1
                return True
            return False

        for edge_id in sorted({eid for eid in edge_ids if eid is not None}):
            moving_agents = [i for i, eid in enumerate(edge_ids) if eid == edge_id and proposals[i] != self.agv_positions[i]]
            capacity = self._edge_capacity(edge_id)
            if len(moving_agents) > capacity:
                allowed = set(sorted(moving_agents, key=priority_key)[:capacity])
                block_agents(i for i in moving_agents if i not in allowed)

        for i in range(self.agv_count):
            for j in range(i + 1, self.agv_count):
                swap = proposals[i] == self.agv_positions[j] and proposals[j] == self.agv_positions[i]
                if swap and proposals[i] != self.agv_positions[i] and proposals[j] != self.agv_positions[j]:
                    shared_edge = edge_ids[i] is not None and edge_ids[i] == edge_ids[j]
                    if shared_edge and self._edge_capacity(edge_ids[i]) <= 1:
                        occupancy = np.bincount(self.agv_positions, minlength=len(self.node_map))
                        i_target = self.agv_positions[j]
                        j_target = self.agv_positions[i]
                        i_can_move = occupancy[i_target] + 1 <= self._node_capacity(i_target)
                        j_can_move = occupancy[j_target] + 1 <= self._node_capacity(j_target)

                        if i_can_move or j_can_move:
                            feasible_movers = [
                                agent_id
                                for agent_id, feasible in ((i, i_can_move), (j, j_can_move))
                                if feasible
                            ]
                            winner = min(feasible_movers, key=priority_key)
                            block_agents([j if winner == i else i])
                        else:
                            # Neither endpoint can safely hold the yielding and
                            # exiting AGV together, so both must wait for a
                            # corridor-level retreat or reservation decision.
                            block_agents([i, j])

        # Blocking one move makes that AGV stay at its current node, which can
        # invalidate another previously accepted move. Resolve these dependencies
        # iteratively until the effective next positions satisfy node capacities.
        for _ in range(self.agv_count + 1):
            effective_positions = [
                self.agv_positions[i] if blocked[i] else proposals[i]
                for i in range(self.agv_count)
            ]
            changed = False
            for node in range(len(self.node_map)):
                occupying = [i for i, target in enumerate(effective_positions) if target == node]
                capacity = self._node_capacity(node)
                if len(occupying) <= capacity:
                    continue

                staying = [i for i in occupying if self.agv_positions[i] == node]
                incoming = [i for i in occupying if self.agv_positions[i] != node]
                remaining = max(capacity - len(staying), 0)
                allowed_incoming = set(sorted(incoming, key=priority_key)[:remaining])
                changed |= block_agents(i for i in incoming if i not in allowed_incoming)

            if not changed:
                break

        effective_positions = [
            self.agv_positions[i] if blocked[i] else proposals[i]
            for i in range(self.agv_count)
        ]
        occupancy = np.bincount(effective_positions, minlength=len(self.node_map))
        violations = [
            (self.nodes[node].name, int(occupancy[node]), self._node_capacity(node))
            for node in range(len(occupancy))
            if occupancy[node] > self._node_capacity(node)
        ]
        if violations:
            raise RuntimeError(f"Conflict resolver produced node-capacity violations: {violations}")

        return blocked, conflict_events

    def _node_capacity(self, node: int) -> int:
        role = self.node_roles[node]
        if role in {"home", "warehouse", "warehouse_slot"}:
            return max(2, self.agv_count)
        if role == "pickup":
            return 2
        if role == "charge":
            return int(self.config.charge_node_capacity)
        if role in {"workstation", "buffer"}:
            return 1 if self.config.capacity_mode == "stress" else 2
        adjacent_caps = [self._edge_capacity(edge_id) for _, edge_id, _ in self.graph[node]]
        if adjacent_caps and min(adjacent_caps) <= 1:
            return 1 if self.config.capacity_mode == "stress" else 2
        return 2

    def _edge_capacity(self, edge_id: int | None) -> int:
        if edge_id is None:
            return max(self.agv_count, 1)
        edge = self.edges[edge_id]
        if self.config.capacity_mode == "stress":
            return max(edge.capacity_stress, 1)
        return max(edge.capacity_baseline, 1)

    def get_kinematic_time(self, dist: float, speed_limit: float | None = None, action_changed: bool = False) -> float:
        if dist <= 0.0:
            return self.config.wait_time_s
        if self.config.env_variant == "ideal" or self.config.velocity_profile == "ideal":
            return self.config.ideal_move_time_s

        edge_limit = (speed_limit * self.config.edge_speed_multiplier) if speed_limit else self.config.speed_max_mps
        v_max = min(self.config.speed_max_mps, edge_limit)
        accel = self.config.acceleration_mps2
        t_acc = v_max / accel
        d_acc = 0.5 * accel * (t_acc**2)
        if dist < 2.0 * d_acc:
            base_time = 2.0 * math.sqrt(dist / accel)
        else:
            base_time = (dist - 2.0 * d_acc) / v_max + 2.0 * t_acc

        if self.config.velocity_profile == "s_curve":
            base_time += 2.0 * accel / max(self.config.jerk_mps3, 1e-6)
        if action_changed:
            base_time += self.config.rotate_time_s
        return float(base_time)

    def _handling_time(self) -> float:
        if self.config.env_variant == "ideal":
            return 0.0
        return self.config.lift_time_s

    def _move_energy(self, distance: float, travel_time: float, agv_id: int, loaded: bool, action_changed: bool) -> float:
        if self.config.env_variant != "full":
            return 0.0
        job = self._current_job(agv_id)
        load_kg = job.load_kg if job and loaded else 0.0
        load_factor = 1.0 + (self.config.loaded_energy_factor - 1.0) * min(load_kg / 120.0, 1.4)
        accel_events = 2.0 + (1.0 if action_changed else 0.0)
        drive = self.config.base_drive_wh_per_s * travel_time
        rolling = self.config.rolling_wh_per_m * distance
        accel = self.config.acceleration_wh * accel_events
        return float((drive + rolling + accel) * load_factor)

    def _idle_energy(self, duration: float) -> float:
        if self.config.env_variant != "full":
            return 0.0
        return float(self.config.idle_wh_per_s * duration)

    def _apply_energy(self, step_energy_wh: np.ndarray) -> None:
        self.energy_by_agent += step_energy_wh
        if self.config.env_variant != "full":
            return
        for i, energy_wh in enumerate(step_energy_wh):
            soc_drop = 100.0 * float(energy_wh) / max(self.config.battery_capacity_wh, 1e-6)
            self.agv_batteries[i] = max(0.0, self.agv_batteries[i] - soc_drop)

    def _charge(self, agent_id: int, duration_s: float) -> None:
        if self.config.env_variant != "full":
            return
        soc_gain = self.config.charge_soc_per_min * duration_s / 60.0
        self.agv_batteries[agent_id] = min(100.0, self.agv_batteries[agent_id] + soc_gain)

    def _scenario_pressure(self) -> float:
        if self.config.scenario == "steady":
            return 1.0
        phase = self.current_step / max(self.config.max_steps, 1)
        wave = 1.0 + 0.20 * math.sin(2.0 * math.pi * 3.0 * phase)
        surge = 0.50 if 0.30 <= phase <= 0.62 else 0.0
        return float(max(0.8, wave + surge))

    def _update_deadlock_state(
        self,
        old_positions: List[int],
        moved: np.ndarray,
        blocked: np.ndarray,
        productive_progress: bool = False,
    ) -> bool:
        active_work = any(job.status in {"waiting", "assigned", "in_transit"} for job in self.jobs)
        no_progress = (
            not productive_progress
            and old_positions == self.agv_positions
            and active_work
            and (blocked.any() or any(a != 0 for a in self.last_actions))
        )
        if no_progress:
            self.deadlock_timer += 1
        else:
            if self.deadlock_active:
                duration = self.metrics.total_time_sec - self.deadlock_start_time
                self.metrics.deadlock_resolution_times.append(float(duration))
            self.deadlock_timer = 0
            self.deadlock_active = False

        deadlock_started = False
        if self.deadlock_timer == self.config.deadlock_soft_steps and not self.deadlock_active:
            self.deadlock_active = True
            self.deadlock_start_time = self.metrics.total_time_sec
            self.metrics.deadlock_count += 1
            deadlock_started = True
        return deadlock_started

    def _compose_reward(
        self,
        local_rewards: np.ndarray,
        deliveries_this_step: int,
        conflict_events: int,
        step_time_sec: float,
    ) -> float:
        return float(
            self._reward_components(
                local_rewards,
                deliveries_this_step,
                conflict_events,
                step_time_sec,
            )["composed_reward"]
        )

    def _reward_components(
        self,
        local_rewards: np.ndarray,
        deliveries_this_step: int,
        conflict_events: int,
        step_time_sec: float,
    ) -> Dict[str, float]:
        waiting_jobs = sum(1 for job in self.jobs if job.status == "waiting")
        local_sum = float(local_rewards.sum())
        entropy_bonus = 2.0 * self.fleet_distribution_entropy()
        balance_penalty = 0.2 * float(np.var(self.completed_by_agent))
        global_reward = (
            deliveries_this_step * self.config.delivery_reward * self._scenario_pressure()
            - self.config.time_penalty * step_time_sec
            - self.config.conflict_penalty * conflict_events
            - 0.15 * waiting_jobs
        )
        if self.config.reward_mode == "individual":
            composed_reward = local_sum
        elif self.config.reward_mode == "global":
            composed_reward = float(global_reward)
        else:
            composed_reward = float(global_reward + local_sum + entropy_bonus - balance_penalty)
        return {
            "composed_reward": float(composed_reward),
            "local_reward_sum": float(local_sum),
            "local_reward_mean": float(local_rewards.mean()) if len(local_rewards) else 0.0,
            "global_reward": float(global_reward),
            "entropy_bonus": float(entropy_bonus),
            "balance_penalty": float(balance_penalty),
            "waiting_jobs_penalty": float(0.15 * waiting_jobs),
            "deliveries_this_step": float(deliveries_this_step),
            "conflict_events": float(conflict_events),
            "step_time_sec": float(step_time_sec),
        }

    def _attention_proxy(self, proposals: List[int], targets: List[int]) -> np.ndarray:
        scores = np.zeros((self.agv_count, self.agv_count), dtype=np.float32)
        for i in range(self.agv_count):
            for j in range(self.agv_count):
                if i == j:
                    continue
                distance = max(self._path_distance(self.agv_positions[i], self.agv_positions[j]), 1.0)
                target_conflict = 1.0 if proposals[i] == proposals[j] else 0.0
                swap_conflict = (
                    1.0
                    if proposals[i] == self.agv_positions[j] and proposals[j] == self.agv_positions[i]
                    else 0.0
                )
                shared_goal = 1.0 if targets[i] == targets[j] else 0.0
                bottleneck = 1.0 if self._node_capacity(proposals[i]) <= 1 else 0.0
                scores[i, j] = 1.0 / distance + 2.0 * target_conflict + 3.0 * swap_conflict + shared_goal + bottleneck
        row_sum = scores.sum(axis=1, keepdims=True)
        return np.divide(scores, row_sum, out=np.zeros_like(scores), where=row_sum > 0)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        node_occupancy = np.bincount(self.agv_positions, minlength=len(self.node_map)).astype(np.float32)
        waiting_origin_counts = np.zeros(len(self.node_map), dtype=np.float32)
        waiting_destination_counts = np.zeros(len(self.node_map), dtype=np.float32)
        for job in self.jobs:
            if job.status == "waiting":
                waiting_origin_counts[job.origin] += 1
                waiting_destination_counts[job.destination] += 1

        node_features = np.zeros((len(self.node_map), self.node_feature_dim), dtype=np.float32)
        for node in range(len(self.node_map)):
            role = self.node_roles[node]
            node_features[node] = np.array(
                [
                    node_occupancy[node] / max(self.agv_count, 1),
                    self._node_capacity(node) / max(self.agv_count, 1),
                    1.0 if self._node_capacity(node) <= 1 else 0.0,
                    1.0 if role == "charge" else 0.0,
                    1.0 if role in {"pickup", "workstation", "buffer"} else 0.0,
                    min(waiting_origin_counts[node] / 5.0, 1.0),
                    min(waiting_destination_counts[node] / 5.0, 1.0),
                ],
                dtype=np.float32,
            )

        agent_features = np.zeros((self.agv_count, self.agent_feature_dim), dtype=np.float32)
        for i in range(self.agv_count):
            position = self.agv_positions[i]
            target = self._target_for_action(i, 1)
            distance_to_target = self._path_distance(position, target) / self.max_route_distance
            pressure = node_occupancy[position] / max(self._node_capacity(position), 1)
            job = self._current_job(i)
            load_kg = job.load_kg if job else 0.0
            agent_features[i] = np.array(
                [
                    position / max(len(self.node_map) - 1, 1),
                    self.agv_batteries[i] / 100.0,
                    float(self._agv_loaded(i)),
                    target / max(len(self.node_map) - 1, 1),
                    distance_to_target,
                    self.wait_steps[i] / max(self.config.deadlock_hard_steps, 1),
                    self.last_actions[i] / 3.0,
                    pressure,
                    load_kg / 120.0,
                    1.0 if self.agv_job_ids[i] is None else 0.0,
                ],
                dtype=np.float32,
            )

        waiting_jobs = sum(1 for job in self.jobs if job.status == "waiting")
        active_jobs = sum(1 for job in self.jobs if job.status in {"waiting", "assigned", "in_transit"})
        global_features = np.array(
            [
                self.current_step / max(self.config.max_steps, 1),
                self.throughput / max(self.agv_count, 1),
                self.metrics.total_time_sec / 3600.0,
                float(self.deadlock_active),
                self.agv_count / 10.0,
                float(np.mean(self.agv_batteries)) / 100.0,
                self._scenario_pressure(),
                self.fleet_distribution_entropy(),
                waiting_jobs / 20.0,
                active_jobs / 20.0,
            ],
            dtype=np.float32,
        )

        return {
            "agent_features": agent_features,
            "node_features": node_features,
            "adjacency_matrix": self.adjacency_matrix.copy(),
            "global_features": global_features,
        }

    def _next_node_on_shortest_path(self, source: int, target: int) -> int:
        if source == target:
            return source
        key = (source, target)
        if key in self.next_hop_cache:
            return self.next_hop_cache[key]

        queue: List[Tuple[float, int, int | None]] = [(0.0, source, None)]
        best = {source: 0.0}
        first_hop: Dict[int, int] = {source: source}
        while queue:
            cost, node, _ = heapq.heappop(queue)
            if node == target:
                hop = first_hop[node]
                self.next_hop_cache[key] = hop
                return hop
            if cost > best[node]:
                continue
            for neighbor, _, distance in self.graph[node]:
                new_cost = cost + distance
                if new_cost < best.get(neighbor, math.inf):
                    best[neighbor] = new_cost
                    first_hop[neighbor] = neighbor if node == source else first_hop[node]
                    heapq.heappush(queue, (new_cost, neighbor, node))
        return source

    def _dijkstra_distances(self, source: int) -> Dict[int, float]:
        queue = [(0.0, source)]
        best = {source: 0.0}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > best[node]:
                continue
            for neighbor, _, distance in self.graph[node]:
                new_cost = cost + distance
                if new_cost < best.get(neighbor, math.inf):
                    best[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor))
        return best

    def _path_distance(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        return float(self._dijkstra_distances(source).get(target, self.max_route_distance))

    def _route_narrow_ratio(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        current = source
        narrow = 0
        total = 0
        visited = set()
        while current != target and current not in visited and total < len(self.nodes):
            visited.add(current)
            nxt = self._next_node_on_shortest_path(current, target)
            edge = self.edge_by_pair.get((current, nxt))
            if not edge:
                break
            total += 1
            narrow += int(self._edge_capacity(edge.edge_id) <= 1)
            current = nxt
        return float(narrow / total) if total else 0.0

    def _normalized_distribution_entropy(self, weights: np.ndarray) -> float:
        total = weights.sum()
        if total <= 0:
            return 0.0
        p = weights / total
        p = p[p > 0]
        if len(p) <= 1:
            return 0.0
        return float(-np.sum(p * np.log(p)) / np.log(len(self.node_map)))

    def fleet_distribution_entropy(self) -> float:
        """Physical-time-weighted fleet distribution entropy."""
        return self._normalized_distribution_entropy(self.node_occupancy_time_sec)

    def node_visit_entropy(self) -> float:
        """Legacy decision-step visit entropy retained for traceability."""
        return self._normalized_distribution_entropy(self.node_visit_counts)

    def summary(self) -> Dict[str, float | str]:
        elapsed_h = self.metrics.total_time_sec / 3600.0
        total_distance = float(self.distance_by_agent.sum())
        empty_distance = float(self.empty_distance_by_agent.sum())
        throughput = max(self.throughput, 0)
        energy_model_available = self.config.env_variant == "full"
        if energy_model_available and throughput > 0:
            eer = float(self.metrics.total_energy_wh / throughput)
        elif energy_model_available:
            eer = math.nan
        else:
            eer = math.nan
        drt_values = self.metrics.deadlock_resolution_times
        utilization = float(self.busy_time_by_agent.sum() / max(self.metrics.total_time_sec * self.agv_count, 1e-9))
        blocked_time_ratio = float(
            self.metrics.blocked_time_sec
            / max(self.metrics.total_time_sec * self.agv_count, 1e-9)
        )
        active_jobs = sum(1 for job in self.jobs if job.status in {"waiting", "assigned", "in_transit"})
        return {
            "parameter_profile": self.config.parameter_profile,
            "env_variant": self.config.env_variant,
            "reward_mode": self.config.reward_mode,
            "scenario": self.config.scenario,
            "dispatch_rule": self.config.dispatch_rule,
            "capacity_mode": self.config.capacity_mode,
            "arrival_process": self.config.arrival_process,
            "arrival_stream_scheme": "independent_per_template_seedsequence_v1",
            "arrival_rate_multiplier": float(self.config.arrival_rate_multiplier),
            "agv_count": float(self.agv_count),
            "steps": float(self.current_step),
            "real_time_sec": float(self.metrics.total_time_sec),
            "uph": float(throughput / elapsed_h) if elapsed_h > 0 else 0.0,
            "throughput": float(throughput),
            "active_jobs": float(active_jobs),
            "released_jobs": float(len(self.jobs)),
            "arrival_trace_count": float(self.arrival_trace_count()),
            "arrival_trace_signature": self.arrival_trace_signature(),
            "max_released_jobs": float(self.config.max_released_jobs)
            if self.config.max_released_jobs is not None
            else -1.0,
            "release_cap_reached": float(self._job_release_cap_reached()),
            "conflict_count": float(self.metrics.conflict_count),
            "blocking_onset_count": float(self.metrics.blocking_onset_count),
            "blocked_count": float(self.metrics.blocked_count),
            "blocked_agent_steps": float(self.metrics.blocked_count),
            "blocked_time_sec": float(self.metrics.blocked_time_sec),
            "blocked_time_ratio": blocked_time_ratio,
            "route_blocking_onset_count": float(self.metrics.route_blocking_onset_count),
            "route_blocked_agent_steps": float(self.metrics.route_blocked_agent_steps),
            "route_blocked_time_sec": float(self.metrics.route_blocked_time_sec),
            "charge_queue_onset_count": float(self.metrics.charge_queue_onset_count),
            "charge_queue_blocked_agent_steps": float(
                self.metrics.charge_queue_blocked_agent_steps
            ),
            "charge_queue_time_sec": float(self.metrics.charge_queue_time_sec),
            "deadlock_count": float(self.metrics.deadlock_count),
            "deadlock_resolution_time": float(np.mean(drt_values)) if drt_values else 0.0,
            "energy_model_available": float(energy_model_available),
            "energy_efficiency_wh_per_sku": eer,
            "fleet_distribution_entropy": self.fleet_distribution_entropy(),
            "node_visit_entropy": self.node_visit_entropy(),
            "fleet_distribution_entropy_basis": "physical_time_seconds",
            "empty_running_ratio": float(empty_distance / total_distance) if total_distance > 0 else 0.0,
            "load_balancing_variance": float(np.var(self.completed_by_agent)),
            "agv_utilization": utilization,
            "avg_task_wait_time": float(np.mean(self.metrics.completed_wait_times))
            if self.metrics.completed_wait_times
            else 0.0,
            "avg_task_cycle_time": float(np.mean(self.metrics.completed_cycle_times))
            if self.metrics.completed_cycle_times
            else 0.0,
            "out_of_battery_rate": float(self.metrics.out_of_battery_count > 0),
            "timeout_rate": float(self.metrics.timeout_count > 0),
            "total_energy_wh": float(self.metrics.total_energy_wh),
            "total_distance_m": total_distance,
            "speed_max_mps": float(self.config.speed_max_mps),
            "edge_speed_multiplier": float(self.config.edge_speed_multiplier),
            "acceleration_mps2": float(self.config.acceleration_mps2),
            "jerk_mps3": float(self.config.jerk_mps3),
            "wait_time_s": float(self.config.wait_time_s),
            "lift_time_s": float(self.config.lift_time_s),
            "rotate_time_s": float(self.config.rotate_time_s),
            "battery_capacity_wh": float(self.config.battery_capacity_wh),
            "base_drive_wh_per_s": float(self.config.base_drive_wh_per_s),
            "rolling_wh_per_m": float(self.config.rolling_wh_per_m),
            "acceleration_wh": float(self.config.acceleration_wh),
            "idle_wh_per_s": float(self.config.idle_wh_per_s),
            "loaded_energy_factor": float(self.config.loaded_energy_factor),
            "charge_soc_per_min": float(self.config.charge_soc_per_min),
            "charge_node_capacity": float(self.config.charge_node_capacity),
            "deadlock_soft_steps": float(self.config.deadlock_soft_steps),
            "deadlock_hard_steps": float(self.config.deadlock_hard_steps),
        }

    def arrival_trace_count(self, cutoff_sec: float | None = None) -> int:
        """Count exogenous releases up to a shared physical-time cutoff."""

        cutoff = math.inf if cutoff_sec is None else float(cutoff_sec)
        return sum(job.release_time_sec <= cutoff + 1e-9 for job in self.jobs)

    def arrival_trace_signature(self, cutoff_sec: float | None = None) -> str:
        """Hash immutable release data so paired-policy task streams can be audited."""

        cutoff = math.inf if cutoff_sec is None else float(cutoff_sec)
        records = [
            (
                int(job.job_id),
                str(job.template_id),
                int(job.origin),
                int(job.destination),
                str(job.task_class),
                f"{float(job.load_kg):.9f}",
                int(job.priority),
                f"{float(job.release_time_sec):.9f}",
            )
            for job in self.jobs
            if job.release_time_sec <= cutoff + 1e-9
        ]
        payload = "\n".join("|".join(map(str, record)) for record in records)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _info(self, **kwargs: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "metrics": self.summary(),
            "attention_weights": self.last_attention.copy(),
            "positions": tuple(self.agv_positions),
            "position_names": tuple(self.node_map[p] for p in self.agv_positions),
            "batteries": tuple(float(b) for b in self.agv_batteries),
            "tasks": tuple(1 if self._agv_loaded(i) else 0 for i in range(self.agv_count)),
            "phases": tuple(self.agv_phases),
        }
        info.update(kwargs)
        return info

    def iter_node_rows(self) -> Iterable[Dict[str, Any]]:
        for node in self.nodes:
            yield {
                "node_id": node.node_id,
                "node_name": node.name,
                "x_m": node.x_m,
                "y_m": node.y_m,
                "role": node.role,
                "cad_basis": node.cad_basis,
                "is_warehouse": int(node.role in {"warehouse", "warehouse_slot"}),
                "is_pickup": int(node.role in {"pickup", "workstation"}),
                "is_charge": int(node.role == "charge"),
                "is_home": int(node.role == "home"),
                "is_buffer": int(node.role == "buffer"),
            }

    def iter_edge_rows(self) -> Iterable[Dict[str, Any]]:
        for edge in self.edges:
            yield {
                "edge_id": edge.edge_id,
                "from_node": edge.from_node,
                "from_name": edge.from_name,
                "to_node": edge.to_node,
                "to_name": edge.to_name,
                "distance_m": edge.distance_m,
                "edge_type": edge.edge_type,
                "bidirectional": int(edge.bidirectional),
                "capacity_baseline": edge.capacity_baseline,
                "capacity_stress": edge.capacity_stress,
                "speed_limit_mps": edge.speed_limit_mps,
            }
