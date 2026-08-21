from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from agv_case_env import AGV_A_Charge_Env
from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    COUNTERFACTUAL_METRIC_NAMES,
    _different_candidates,
    _validate_horizons,
    rollout_fixed_policy,
)
from physics_graph_world_model import baseline_dt_aware_action, behavior_action


QUEUE_STATUS_NAMES = ("waiting", "assigned", "in_transit")


@dataclass(frozen=True)
class ExpectedEffectCollectionConfig:
    episodes: int = 8
    behavior_steps: int = 4000
    warmup_steps: int = 1200
    sample_stride: int = 80
    candidates_per_state: int = 3
    future_replications: int = 4
    minimum_valid_replications: int = 3
    exploration_rate: float = 0.35
    horizons_sec: Tuple[float, ...] = COUNTERFACTUAL_HORIZONS_SEC
    max_rollout_steps: int = 500
    maximum_relative_overshoot: float = 0.35
    maximum_absolute_overshoot_sec: float = 60.0
    behavior_seed: int = 14300
    future_seed: int = 14350


def counterfactual_aux_features(env: AGV_A_Charge_Env) -> np.ndarray:
    """Expose queue composition hidden by the clipped base observation."""

    node_count = len(env.node_map)
    queue_features = []
    fixed_log_scale = np.log1p(100.0)
    for status in QUEUE_STATUS_NAMES:
        origin = np.zeros(node_count, dtype=np.float32)
        destination = np.zeros(node_count, dtype=np.float32)
        for job in env.jobs:
            if job.status == status:
                origin[job.origin] += 1.0
                destination[job.destination] += 1.0
        queue_features.extend(
            [
                np.log1p(origin) / fixed_log_scale,
                np.log1p(destination) / fixed_log_scale,
            ]
        )

    agent_features = []
    node_denominator = max(node_count - 1, 1)
    now = float(env.metrics.total_time_sec)
    for agent_id in range(env.agv_count):
        job = env._current_job(agent_id)
        if job is None:
            agent_features.extend([0.0] * 8)
            continue
        phase = env.agv_phases[agent_id]
        agent_features.extend(
            [
                float(job.origin) / node_denominator,
                float(job.destination) / node_denominator,
                float(phase == "to_origin"),
                float(phase == "to_destination"),
                min(max(now - float(job.release_time_sec), 0.0) / 3600.0, 4.0),
                float(job.priority) / 5.0,
                float(job.load_kg) / 120.0,
                float(job.status == "in_transit"),
            ]
        )
    status_totals = np.asarray(
        [
            sum(job.status == status for job in env.jobs)
            for status in (*QUEUE_STATUS_NAMES, "done")
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            *queue_features,
            np.asarray(agent_features, dtype=np.float32),
            np.log1p(status_totals) / fixed_log_scale,
        ]
    ).astype(np.float32)


def reseed_future_arrivals(env: AGV_A_Charge_Env, seed: int) -> None:
    """Draw a fresh conditional Poisson future without changing the current state."""

    sequences = np.random.SeedSequence(int(seed)).spawn(len(env.task_templates))
    env.arrival_rng_by_template = {
        template.task_id: np.random.default_rng(sequence)
        for template, sequence in zip(env.task_templates, sequences)
    }
    now = float(env.metrics.total_time_sec)
    env.next_arrival_by_template = {
        template.task_id: now
        + env._sample_interarrival(
            env._arrival_rate(template), template.task_id
        )
        for template in env.task_templates
    }


def _future_replication_seed(
    base_seed: int, episode: int, state_id: int, replication: int
) -> int:
    return int(
        np.random.SeedSequence(
            [int(base_seed), int(episode), int(state_id), int(replication)]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _aggregate_pair(
    obs: Dict[str, np.ndarray],
    baseline_action: np.ndarray,
    candidate_action: np.ndarray,
    baseline_results: Sequence[Dict[str, np.ndarray]],
    candidate_results: Sequence[Dict[str, np.ndarray]],
    horizons_sec: Sequence[float],
    config: ExpectedEffectCollectionConfig,
    episode_id: int,
    state_id: int,
    candidate_id: int,
    replication_seeds: Sequence[int],
    auxiliary_features: np.ndarray,
) -> Dict[str, np.ndarray]:
    horizons = np.asarray(tuple(horizons_sec), dtype=np.float32)
    allowed = np.minimum(
        config.maximum_absolute_overshoot_sec,
        config.maximum_relative_overshoot * horizons,
    )
    baseline_outcomes = np.stack([result["outcomes"] for result in baseline_results])
    candidate_outcomes = np.stack([result["outcomes"] for result in candidate_results])
    delta_replications = candidate_outcomes - baseline_outcomes
    overshoot = np.maximum(
        np.stack([result["overshoot_sec"] for result in baseline_results]),
        np.stack([result["overshoot_sec"] for result in candidate_results]),
    )
    valid = overshoot <= allowed[None, :]
    target = np.zeros(
        (len(horizons), len(COUNTERFACTUAL_METRIC_NAMES)), dtype=np.float32
    )
    target_std = np.zeros_like(target)
    target_mask = np.zeros_like(target)
    for horizon in range(len(horizons)):
        valid_replications = valid[:, horizon]
        if int(np.sum(valid_replications)) < config.minimum_valid_replications:
            continue
        values = delta_replications[valid_replications, horizon]
        target[horizon] = np.mean(values, axis=0)
        target_std[horizon] = np.std(values, axis=0, ddof=0)
        target_mask[horizon] = 1.0
    return {
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "state_id": np.asarray(state_id, dtype=np.int64),
        "candidate_id": np.asarray(candidate_id, dtype=np.int64),
        "agent_features": obs["agent_features"].astype(np.float32),
        "node_features": obs["node_features"].astype(np.float32),
        "adjacency_matrix": obs["adjacency_matrix"].astype(np.float32),
        "global_features": obs["global_features"].astype(np.float32),
        "counterfactual_aux_features": auxiliary_features.astype(np.float32),
        "baseline_actions": baseline_action.astype(np.int64),
        "candidate_actions": candidate_action.astype(np.int64),
        "action_hamming_distance": np.asarray(
            np.count_nonzero(candidate_action != baseline_action), dtype=np.int64
        ),
        "target_delta": target,
        "target_delta_replication_std": target_std,
        "target_mask": target_mask,
        "valid_replication_count": valid.sum(axis=0).astype(np.int64),
        "future_replications": np.asarray(
            config.future_replications, dtype=np.int64
        ),
        "replication_seeds": np.asarray(replication_seeds, dtype=np.uint32),
        "maximum_overshoot_sec": np.asarray(float(np.max(overshoot)), dtype=np.float32),
    }


def collect_expected_effect_samples(
    config: ExpectedEffectCollectionConfig,
    env_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, np.ndarray]]:
    horizons = _validate_horizons(config.horizons_sec)
    if config.future_replications < 2:
        raise ValueError("Expected-effect targets require at least two future replications")
    if not 1 <= config.minimum_valid_replications <= config.future_replications:
        raise ValueError("Invalid minimum valid-replication requirement")
    kwargs = dict(env_kwargs or {})
    kwargs.setdefault("scenario", "rush")
    kwargs.setdefault("capacity_mode", "stress")
    kwargs.setdefault(
        "max_steps", config.behavior_steps + config.max_rollout_steps + 100
    )
    behavior_rng = np.random.default_rng(config.behavior_seed)
    samples: List[Dict[str, np.ndarray]] = []
    for episode in range(config.episodes):
        episode_seed = config.behavior_seed + episode
        env = AGV_A_Charge_Env(seed=episode_seed, **kwargs)
        obs, _ = env.reset(seed=episode_seed)
        for state_id in range(config.behavior_steps):
            should_sample = (
                state_id >= config.warmup_steps
                and (state_id - config.warmup_steps) % config.sample_stride == 0
            )
            if should_sample:
                candidates = _different_candidates(env)
                if candidates:
                    order = behavior_rng.permutation(len(candidates))
                    selected = [
                        candidates[int(index)]
                        for index in order[: config.candidates_per_state]
                    ]
                    baseline_action = baseline_dt_aware_action(env)
                    auxiliary_features = counterfactual_aux_features(env)
                    replication_envs = []
                    baseline_results = []
                    replication_seeds = []
                    for replication in range(config.future_replications):
                        future_seed = _future_replication_seed(
                            config.future_seed, episode, state_id, replication
                        )
                        replicated_env = copy.deepcopy(env)
                        reseed_future_arrivals(replicated_env, future_seed)
                        replication_envs.append(replicated_env)
                        replication_seeds.append(future_seed)
                        baseline_results.append(
                            rollout_fixed_policy(
                                replicated_env,
                                baseline_action,
                                horizons,
                                config.max_rollout_steps,
                            )
                        )
                    for candidate_id, candidate in enumerate(selected):
                        candidate_results = [
                            rollout_fixed_policy(
                                replicated_env,
                                candidate,
                                horizons,
                                config.max_rollout_steps,
                            )
                            for replicated_env in replication_envs
                        ]
                        samples.append(
                            _aggregate_pair(
                                obs,
                                baseline_action,
                                candidate,
                                baseline_results,
                                candidate_results,
                                horizons,
                                config,
                                episode,
                                state_id,
                                candidate_id,
                                replication_seeds,
                                auxiliary_features,
                            )
                        )
            action = behavior_action(
                env, behavior_rng, exploration_rate=config.exploration_rate
            )
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    if not samples:
        raise RuntimeError("Expected-effect collection produced no samples")
    return samples


def summarize_expected_effect_samples(samples: Sequence[Dict[str, np.ndarray]]) -> dict:
    targets = np.stack([sample["target_delta"] for sample in samples])
    masks = np.stack([sample["target_mask"] for sample in samples]) > 0.0
    replication_std = np.stack(
        [sample["target_delta_replication_std"] for sample in samples]
    )
    return {
        "samples": len(samples),
        "episodes": len({int(sample["episode_id"]) for sample in samples}),
        "valid_target_rate": masks.mean(axis=0).tolist(),
        "nonzero_rate": (
            np.sum((np.abs(targets) > 1.0e-6) & masks, axis=0)
            / np.maximum(np.sum(masks, axis=0), 1)
        ).tolist(),
        "mean_replication_std": np.mean(replication_std, axis=0).tolist(),
        "maximum_overshoot_sec": float(
            max(float(sample["maximum_overshoot_sec"]) for sample in samples)
        ),
    }
