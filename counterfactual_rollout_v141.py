from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from agv_case_env import AGV_A_Charge_Env
from physics_graph_world_model import baseline_dt_aware_action, behavior_action
from physics_graph_world_model_multistep import candidate_joint_actions


COUNTERFACTUAL_HORIZONS_SEC = (120.0, 360.0, 720.0)
COUNTERFACTUAL_METRIC_NAMES = (
    "energy_wh",
    "completed_tasks",
    "charge_queue_time_sec",
)


@dataclass(frozen=True)
class CounterfactualCollectionConfig:
    episodes: int = 2
    behavior_steps: int = 80
    warmup_steps: int = 10
    sample_stride: int = 10
    candidates_per_state: int = 2
    exploration_rate: float = 0.35
    horizons_sec: Tuple[float, ...] = COUNTERFACTUAL_HORIZONS_SEC
    max_rollout_steps: int = 500
    maximum_relative_overshoot: float = 0.35
    maximum_absolute_overshoot_sec: float = 60.0
    seed: int = 14100


def _metric_vector(summary: Dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(summary["total_energy_wh"]),
            float(summary["throughput"]),
            float(summary["charge_queue_time_sec"]),
        ],
        dtype=np.float64,
    )


def _validate_horizons(horizons_sec: Sequence[float]) -> Tuple[float, ...]:
    horizons = tuple(float(value) for value in horizons_sec)
    if not horizons or any(value <= 0.0 for value in horizons):
        raise ValueError("Counterfactual horizons must be positive")
    if any(right <= left for left, right in zip(horizons, horizons[1:])):
        raise ValueError("Counterfactual horizons must be strictly increasing")
    return horizons


def rollout_fixed_policy(
    initial_env: AGV_A_Charge_Env,
    first_action: np.ndarray,
    horizons_sec: Sequence[float],
    max_rollout_steps: int,
) -> Dict[str, np.ndarray]:
    """Apply one action, then the frozen DT-aware policy to fixed physical horizons."""

    horizons = _validate_horizons(horizons_sec)
    if max_rollout_steps < 1:
        raise ValueError("Maximum rollout steps must be positive")
    env = copy.deepcopy(initial_env)
    start = env.summary()
    start_time = float(start["real_time_sec"])
    start_metrics = _metric_vector(start)
    outcomes = np.full((len(horizons), len(COUNTERFACTUAL_METRIC_NAMES)), np.nan)
    elapsed = np.full(len(horizons), np.nan)
    overshoot = np.full(len(horizons), np.nan)
    recorded = 0
    action = np.asarray(first_action, dtype=np.int64).reshape(env.agv_count)

    for rollout_step in range(max_rollout_steps):
        _, _, terminated, truncated, _ = env.step(action)
        summary = env.summary()
        physical_elapsed = float(summary["real_time_sec"]) - start_time
        while recorded < len(horizons) and physical_elapsed >= horizons[recorded]:
            outcomes[recorded] = _metric_vector(summary) - start_metrics
            elapsed[recorded] = physical_elapsed
            overshoot[recorded] = physical_elapsed - horizons[recorded]
            recorded += 1
        if recorded == len(horizons):
            break
        if terminated or truncated:
            raise RuntimeError(
                "Counterfactual branch ended before its longest physical horizon"
            )
        action = baseline_dt_aware_action(env)

    if recorded != len(horizons):
        raise RuntimeError(
            f"Counterfactual branch reached {recorded}/{len(horizons)} horizons "
            f"within {max_rollout_steps} steps"
        )
    horizon_array = np.asarray(horizons, dtype=np.float64)
    equivalent_outcomes = outcomes * (horizon_array / elapsed)[:, None]
    return {
        "outcomes": equivalent_outcomes.astype(np.float32),
        "raw_outcomes": outcomes.astype(np.float32),
        "elapsed_sec": elapsed.astype(np.float32),
        "overshoot_sec": overshoot.astype(np.float32),
        "steps": np.asarray(rollout_step + 1, dtype=np.int64),
    }


def paired_counterfactual_sample(
    env: AGV_A_Charge_Env,
    obs: Dict[str, np.ndarray],
    candidate_action: np.ndarray,
    horizons_sec: Sequence[float],
    max_rollout_steps: int,
    episode_id: int,
    state_id: int,
    candidate_id: int,
    maximum_relative_overshoot: float = 0.35,
    maximum_absolute_overshoot_sec: float = 60.0,
) -> Dict[str, np.ndarray]:
    baseline_action = baseline_dt_aware_action(env)
    candidate = np.asarray(candidate_action, dtype=np.int64).reshape(env.agv_count)
    if np.array_equal(candidate, baseline_action):
        raise ValueError("Counterfactual candidate must differ from the baseline")

    baseline = rollout_fixed_policy(
        env, baseline_action, horizons_sec, max_rollout_steps
    )
    candidate_result = rollout_fixed_policy(
        env, candidate, horizons_sec, max_rollout_steps
    )
    delta = candidate_result["outcomes"] - baseline["outcomes"]
    horizons = np.asarray(tuple(horizons_sec), dtype=np.float32)
    allowed_overshoot = np.minimum(
        maximum_absolute_overshoot_sec,
        maximum_relative_overshoot * horizons,
    )
    maximum_observed_overshoot = np.maximum(
        baseline["overshoot_sec"], candidate_result["overshoot_sec"]
    )
    target_mask = (maximum_observed_overshoot <= allowed_overshoot).astype(np.float32)
    return {
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "state_id": np.asarray(state_id, dtype=np.int64),
        "candidate_id": np.asarray(candidate_id, dtype=np.int64),
        "agent_features": obs["agent_features"].astype(np.float32),
        "node_features": obs["node_features"].astype(np.float32),
        "adjacency_matrix": obs["adjacency_matrix"].astype(np.float32),
        "global_features": obs["global_features"].astype(np.float32),
        "baseline_actions": baseline_action.astype(np.int64),
        "candidate_actions": candidate.astype(np.int64),
        "action_hamming_distance": np.asarray(
            np.count_nonzero(candidate != baseline_action), dtype=np.int64
        ),
        "baseline_outcomes": baseline["outcomes"],
        "candidate_outcomes": candidate_result["outcomes"],
        "baseline_raw_outcomes": baseline["raw_outcomes"],
        "candidate_raw_outcomes": candidate_result["raw_outcomes"],
        "target_delta": delta.astype(np.float32),
        "target_mask": np.repeat(
            target_mask[:, None], len(COUNTERFACTUAL_METRIC_NAMES), axis=1
        ).astype(np.float32),
        "baseline_elapsed_sec": baseline["elapsed_sec"],
        "candidate_elapsed_sec": candidate_result["elapsed_sec"],
        "baseline_overshoot_sec": baseline["overshoot_sec"],
        "candidate_overshoot_sec": candidate_result["overshoot_sec"],
        "baseline_rollout_steps": baseline["steps"],
        "candidate_rollout_steps": candidate_result["steps"],
    }


def _different_candidates(env: AGV_A_Charge_Env) -> List[np.ndarray]:
    baseline = baseline_dt_aware_action(env)
    baseline_proposals, baseline_targets, baseline_edges = env._propose_positions(
        baseline
    )
    candidates = candidate_joint_actions(
        env,
        allow_proactive_yield=True,
        allow_proactive_charge=True,
    )
    material_candidates: List[np.ndarray] = []
    baseline_signature = (
        tuple(int(value) for value in baseline_proposals),
        tuple(int(value) for value in baseline_targets),
        tuple(None if value is None else int(value) for value in baseline_edges),
    )
    for action in candidates:
        candidate = np.asarray(action, dtype=np.int64)
        if np.array_equal(candidate, baseline):
            continue
        proposals, targets, edges = env._propose_positions(candidate)
        candidate_signature = (
            tuple(int(value) for value in proposals),
            tuple(int(value) for value in targets),
            tuple(None if value is None else int(value) for value in edges),
        )
        if candidate_signature != baseline_signature:
            material_candidates.append(candidate)
    return material_candidates


def collect_counterfactual_samples(
    config: CounterfactualCollectionConfig,
    env_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, np.ndarray]]:
    horizons = _validate_horizons(config.horizons_sec)
    if config.episodes < 1 or config.behavior_steps < 1:
        raise ValueError("Episodes and behavior steps must be positive")
    if config.sample_stride < 1 or config.candidates_per_state < 1:
        raise ValueError("Sampling controls must be positive")
    kwargs = dict(env_kwargs or {})
    kwargs.setdefault("scenario", "rush")
    kwargs.setdefault("capacity_mode", "stress")
    kwargs.setdefault("max_steps", config.behavior_steps + config.max_rollout_steps + 100)
    rng = np.random.default_rng(config.seed)
    samples: List[Dict[str, np.ndarray]] = []

    for episode in range(config.episodes):
        episode_seed = config.seed + episode
        env = AGV_A_Charge_Env(seed=episode_seed, **kwargs)
        obs, _ = env.reset(seed=episode_seed)
        for state_id in range(config.behavior_steps):
            if (
                state_id >= config.warmup_steps
                and (state_id - config.warmup_steps) % config.sample_stride == 0
            ):
                candidates = _different_candidates(env)
                if candidates:
                    order = rng.permutation(len(candidates))
                    selected = order[: config.candidates_per_state]
                    for candidate_id, index in enumerate(selected):
                        samples.append(
                            paired_counterfactual_sample(
                                env=env,
                                obs=obs,
                                candidate_action=candidates[int(index)],
                                horizons_sec=horizons,
                                max_rollout_steps=config.max_rollout_steps,
                                episode_id=episode,
                                state_id=state_id,
                                candidate_id=candidate_id,
                                maximum_relative_overshoot=(
                                    config.maximum_relative_overshoot
                                ),
                                maximum_absolute_overshoot_sec=(
                                    config.maximum_absolute_overshoot_sec
                                ),
                            )
                        )
            action = behavior_action(
                env, rng, exploration_rate=config.exploration_rate
            )
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    if not samples:
        raise RuntimeError("Counterfactual collection produced no action pairs")
    return samples


def _independent_episode_rng(seed: int, episode: int) -> np.random.Generator:
    """Create a trajectory-local stream that is invariant to worker scheduling."""

    sequence = np.random.SeedSequence([int(seed), int(episode), 145])
    return np.random.default_rng(sequence)


def _collect_counterfactual_episode(
    config: CounterfactualCollectionConfig,
    env_kwargs: Dict[str, Any],
    episode: int,
) -> List[Dict[str, np.ndarray]]:
    horizons = _validate_horizons(config.horizons_sec)
    episode_seed = config.seed + episode
    rng = _independent_episode_rng(config.seed, episode)
    env = AGV_A_Charge_Env(seed=episode_seed, **env_kwargs)
    obs, _ = env.reset(seed=episode_seed)
    samples: List[Dict[str, np.ndarray]] = []
    for state_id in range(config.behavior_steps):
        if (
            state_id >= config.warmup_steps
            and (state_id - config.warmup_steps) % config.sample_stride == 0
        ):
            candidates = _different_candidates(env)
            if candidates:
                order = rng.permutation(len(candidates))
                selected = order[: config.candidates_per_state]
                for candidate_id, index in enumerate(selected):
                    samples.append(
                        paired_counterfactual_sample(
                            env=env,
                            obs=obs,
                            candidate_action=candidates[int(index)],
                            horizons_sec=horizons,
                            max_rollout_steps=config.max_rollout_steps,
                            episode_id=episode,
                            state_id=state_id,
                            candidate_id=candidate_id,
                            maximum_relative_overshoot=(
                                config.maximum_relative_overshoot
                            ),
                            maximum_absolute_overshoot_sec=(
                                config.maximum_absolute_overshoot_sec
                            ),
                        )
                    )
        action = behavior_action(
            env, rng, exploration_rate=config.exploration_rate
        )
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return samples


def collect_counterfactual_samples_parallel(
    config: CounterfactualCollectionConfig,
    env_kwargs: Dict[str, Any] | None = None,
    parallel_episodes: int = 1,
) -> List[Dict[str, np.ndarray]]:
    """Collect complete trajectories in parallel without changing their random streams."""

    _validate_horizons(config.horizons_sec)
    if config.episodes < 1 or config.behavior_steps < 1:
        raise ValueError("Episodes and behavior steps must be positive")
    if config.sample_stride < 1 or config.candidates_per_state < 1:
        raise ValueError("Sampling controls must be positive")
    if parallel_episodes < 1:
        raise ValueError("Parallel episode count must be positive")
    kwargs = dict(env_kwargs or {})
    kwargs.setdefault("scenario", "rush")
    kwargs.setdefault("capacity_mode", "stress")
    kwargs.setdefault(
        "max_steps", config.behavior_steps + config.max_rollout_steps + 100
    )
    worker_count = min(int(parallel_episodes), int(config.episodes))
    by_episode: Dict[int, List[Dict[str, np.ndarray]]] = {}
    if worker_count == 1:
        for episode in range(config.episodes):
            by_episode[episode] = _collect_counterfactual_episode(
                config, kwargs, episode
            )
            print(
                f"Counterfactual trajectory {episode + 1}/{config.episodes} "
                f"finished with {len(by_episode[episode])} pairs.",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_episode = {
                executor.submit(
                    _collect_counterfactual_episode, config, kwargs, episode
                ): episode
                for episode in range(config.episodes)
            }
            completed = 0
            try:
                for future in as_completed(future_to_episode):
                    episode = future_to_episode[future]
                    by_episode[episode] = future.result()
                    completed += 1
                    print(
                        f"Counterfactual trajectory {completed}/{config.episodes} "
                        f"finished (episode {episode}, "
                        f"{len(by_episode[episode])} pairs).",
                        flush=True,
                    )
            except BaseException:
                for future in future_to_episode:
                    future.cancel()
                raise
    samples = [
        sample
        for episode in range(config.episodes)
        for sample in by_episode.get(episode, [])
    ]
    samples.sort(
        key=lambda sample: (
            int(sample["episode_id"]),
            int(sample["state_id"]),
            int(sample["candidate_id"]),
        )
    )
    if not samples:
        raise RuntimeError("Parallel counterfactual collection produced no action pairs")
    return samples


def summarize_counterfactual_samples(
    samples: Iterable[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    rows = list(samples)
    if not rows:
        raise ValueError("Cannot summarize an empty counterfactual dataset")
    targets = np.stack([row["target_delta"] for row in rows]).astype(np.float64)
    masks = np.stack([row["target_mask"] for row in rows]) > 0.0
    overshoot = np.maximum(
        np.stack([row["baseline_overshoot_sec"] for row in rows]),
        np.stack([row["candidate_overshoot_sec"] for row in rows]),
    )
    nonzero = np.abs(targets) > 1.0e-6
    masked_targets = np.where(masks, targets, np.nan)
    return {
        "samples": len(rows),
        "episodes": len({int(row["episode_id"]) for row in rows}),
        "target_mean": np.nanmean(masked_targets, axis=0).tolist(),
        "target_std": np.nanstd(masked_targets, axis=0).tolist(),
        "target_nonzero_rate": (
            np.sum(nonzero & masks, axis=0) / np.maximum(np.sum(masks, axis=0), 1)
        ).tolist(),
        "target_abs_q75": [
            [
                float(np.quantile(np.abs(targets[:, horizon, metric][masks[:, horizon, metric]]), 0.75))
                if np.any(masks[:, horizon, metric])
                else float("nan")
                for metric in range(targets.shape[2])
            ]
            for horizon in range(targets.shape[1])
        ],
        "valid_target_rate": masks.mean(axis=0).tolist(),
        "maximum_overshoot_sec": float(np.max(overshoot)),
        "mean_action_hamming_distance": float(
            np.mean([int(row["action_hamming_distance"]) for row in rows])
        ),
    }
