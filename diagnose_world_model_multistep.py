from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from agv_case_env import AGV_A_Charge_Env
from physics_graph_world_model import (
    CONGESTION_KPI_NAMES,
    KPI_NAMES,
    collect_world_model_transitions,
    kpi_scale,
)
from physics_graph_world_model_multistep import (
    MODEL_VERSION as V9_MODEL_VERSION,
    MultiStepSequenceDataset,
    build_sequence_samples,
    load_multistep_world_model_policy,
)
from physics_graph_world_model_multistep_v10 import (
    MODEL_VERSION as V10_MODEL_VERSION,
    load_multistep_world_model_policy_v10,
)
from physics_graph_world_model_multistep_v11 import (
    MODEL_VERSION as V11_MODEL_VERSION,
    load_multistep_world_model_policy_v11,
)
from physics_graph_world_model_multistep_v12 import (
    MODEL_VERSION as V12_MODEL_VERSION,
    load_multistep_world_model_policy_v12,
)
from physics_graph_world_model_multistep_v13 import (
    MODEL_VERSION as V13_MODEL_VERSION,
    annotate_future_congestion_risk,
    load_multistep_world_model_policy_v13,
)
from physics_graph_world_model_multistep_v14 import (
    MODEL_VERSION as V14_MODEL_VERSION,
    FUTURE_TERMINAL_KPI_NAMES,
    annotate_future_terminal_kpis,
    load_multistep_world_model_v14,
)
from train_world_model_multistep import move_batch


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Held-out open-loop diagnostics for a multi-step graph world model."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="experiment_results/world_model_multistep_v9")
    parser.add_argument("--horizons", default="1,3,5,10")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-stride", type=int, default=5)
    parser.add_argument("--exploration-rate", type=float, default=0.25)
    parser.add_argument("--agv-count", type=int, default=3)
    parser.add_argument("--scenario", choices=["steady", "rush"], default="rush")
    parser.add_argument("--capacity-mode", choices=["baseline", "stress"], default="stress")
    parser.add_argument("--seed", type=int, default=9042)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--transition-cache",
        default=None,
        help="Optional trusted fresh-seed trajectory cache shared across model seeds.",
    )
    parser.add_argument(
        "--future-risk-threshold",
        type=float,
        default=0.5,
        help="Precommitted decision threshold for the long-horizon risk head.",
    )
    return parser


def parse_horizons(value: str) -> List[int]:
    horizons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not horizons or horizons[0] < 1:
        raise ValueError("At least one positive diagnostic horizon is required")
    return horizons


def make_env_factory(args: argparse.Namespace):
    def factory(seed: int) -> AGV_A_Charge_Env:
        return AGV_A_Charge_Env(
            agv_count=args.agv_count,
            env_variant="full",
            reward_mode="hybrid",
            scenario=args.scenario,
            dispatch_rule="dt_aware",
            capacity_mode=args.capacity_mode,
            max_steps=args.max_steps,
            seed=seed,
        )

    return factory


def diagnostic_cache_signature(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema": "v14_fresh_seed_open_loop_cache_v1",
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "exploration_rate": args.exploration_rate,
        "agv_count": args.agv_count,
        "scenario": args.scenario,
        "capacity_mode": args.capacity_mode,
        "seed": args.seed,
        "env_variant": "full",
        "reward_mode": "hybrid",
        "dispatch_rule": "dt_aware",
    }


def load_or_collect_diagnostic_transitions(
    args: argparse.Namespace,
) -> tuple[List[Dict[str, np.ndarray]], str]:
    cache_path = Path(args.transition_cache) if args.transition_cache else None
    if cache_path is not None and not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    signature = diagnostic_cache_signature(args)
    if cache_path is not None and cache_path.exists():
        try:
            importlib.import_module("numpy._core")
            needs_numpy_pickle_alias = False
        except ModuleNotFoundError:
            needs_numpy_pickle_alias = True
        if needs_numpy_pickle_alias:
            for modern_name, legacy_name in {
                "numpy._core": "numpy.core",
                "numpy._core.multiarray": "numpy.core.multiarray",
                "numpy._core.numeric": "numpy.core.numeric",
                "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
            }.items():
                sys.modules.setdefault(modern_name, importlib.import_module(legacy_name))
        with gzip.open(cache_path, "rb") as stream:
            cached = pickle.load(stream)
        if cached.get("signature") != signature:
            raise ValueError("Diagnostic trajectory-cache settings do not match this run")
        return cached["transitions"], "loaded"
    transitions = collect_world_model_transitions(
        env_factory=make_env_factory(args),
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        exploration_rate=args.exploration_rate,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "wb", compresslevel=3) as stream:
            pickle.dump(
                {"signature": signature, "transitions": transitions},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return transitions, "generated"


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 1.0e-12 else float("nan")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "r2": r2,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_predictions(
    policy,
    loader: DataLoader,
    device: str,
) -> Dict[str, np.ndarray]:
    collected: Dict[str, List[np.ndarray]] = {}
    policy.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = policy.model.rollout(batch, teacher_forcing_ratio=0.0)
            values = {
                **output,
                "episode_id": batch["episode_id"],
                "start_transition_id": batch["start_transition_id"],
                "target_agent_features": batch["target_agent_features"],
                "target_node_features": batch["target_node_features"],
                "target_global_features": batch["target_global_features"],
                "target_kpi": batch["target_kpi"],
            }
            if "pred_congestion_kpi" in output:
                values["target_congestion_kpi"] = batch["target_congestion_kpi"]
            if "pred_future_congestion_risk_logits" in output:
                values["target_future_congestion_risk"] = batch[
                    "target_future_congestion_risk"
                ]
                values["target_future_congestion_risk_mask"] = batch[
                    "target_future_congestion_risk_mask"
                ]
            if "pred_future_terminal_kpi" in output:
                values["target_future_terminal_kpi"] = batch[
                    "target_future_terminal_kpi"
                ]
                values["target_future_terminal_kpi_mask"] = batch[
                    "target_future_terminal_kpi_mask"
                ]
            for key, value in values.items():
                collected.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in collected.items()}


def diagnostic_rows(data: Dict[str, np.ndarray], horizons: List[int], agv_count: int):
    scale = kpi_scale(agv_count)
    node_count = int(data["target_node_features"].shape[2])
    summary: List[Dict[str, object]] = []
    per_kpi: List[Dict[str, object]] = []
    for horizon in horizons:
        index = horizon - 1
        state_specs = [
            ("agent_state", "pred_agent_features", "target_agent_features"),
            ("node_state", "pred_node_features", "target_node_features"),
            ("global_state", "pred_global_features", "target_global_features"),
        ]
        row: Dict[str, object] = {"horizon_steps": horizon}
        for label, pred_key, target_key in state_specs:
            stats = regression_metrics(data[target_key][:, index], data[pred_key][:, index])
            row[f"{label}_mae"] = stats["mae"]
            row[f"{label}_rmse"] = stats["rmse"]

        predicted_agents = data["pred_agent_features"][:, index]
        target_agents = data["target_agent_features"][:, index]
        for label, feature_index in (("position_node", 0), ("target_node", 3)):
            raw_predicted = predicted_agents[:, :, feature_index]
            actual_indices = np.rint(
                target_agents[:, :, feature_index] * max(node_count - 1, 1)
            ).astype(np.int64)
            predicted_indices = np.clip(
                np.rint(raw_predicted * max(node_count - 1, 1)), 0, node_count - 1
            ).astype(np.int64)
            row[f"{label}_accuracy"] = float(np.mean(predicted_indices == actual_indices))
            row[f"{label}_mae_nodes"] = float(
                np.mean(np.abs(predicted_indices - actual_indices))
            )
            row[f"{label}_invalid_rate"] = float(
                np.mean((raw_predicted < 0.0) | (raw_predicted > 1.0))
            )

        actual_kpi = data["target_kpi"][:, index] * scale
        predicted_kpi = data["pred_kpi"][:, index] * scale
        kpi_stats = regression_metrics(actual_kpi, predicted_kpi)
        row["kpi_mae_all_physical_units"] = kpi_stats["mae"]
        row["kpi_rmse_all_physical_units"] = kpi_stats["rmse"]
        summary.append(row)

        for metric_index, metric_name in enumerate(KPI_NAMES):
            stats = regression_metrics(
                actual_kpi[:, metric_index], predicted_kpi[:, metric_index]
            )
            per_kpi.append(
                {
                    "horizon_steps": horizon,
                    "metric": metric_name,
                    **stats,
                    "actual_mean": float(actual_kpi[:, metric_index].mean()),
                    "predicted_mean": float(predicted_kpi[:, metric_index].mean()),
                    "sample_count": int(actual_kpi.shape[0]),
                }
            )
        if "pred_congestion_kpi" in data:
            actual_congestion = data["target_congestion_kpi"][:, index] * float(agv_count)
            predicted_congestion = np.maximum(
                data["pred_congestion_kpi"][:, index], 0.0
            ) * float(agv_count)
            for metric_index, metric_name in enumerate(CONGESTION_KPI_NAMES):
                actual_values = actual_congestion[:, metric_index]
                predicted_values = predicted_congestion[:, metric_index]
                stats = regression_metrics(
                    actual_values,
                    predicted_values,
                )
                actual_event = actual_values > 0.0
                predicted_event = predicted_values >= 0.5
                true_positive = int(np.sum(actual_event & predicted_event))
                predicted_positive = int(np.sum(predicted_event))
                actual_positive = int(np.sum(actual_event))
                precision = true_positive / max(predicted_positive, 1)
                recall = true_positive / max(actual_positive, 1)
                f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
                per_kpi.append(
                    {
                        "horizon_steps": horizon,
                        "metric": metric_name,
                        **stats,
                        "actual_mean": float(actual_congestion[:, metric_index].mean()),
                        "predicted_mean": float(predicted_congestion[:, metric_index].mean()),
                        "sample_count": int(actual_congestion.shape[0]),
                        "event_prevalence": float(np.mean(actual_event)),
                        "event_precision_at_0_5": float(precision),
                        "event_recall_at_0_5": float(recall),
                        "event_f1_at_0_5": float(f1),
                        "positive_event_mae": (
                            float(np.mean(np.abs(predicted_values[actual_event] - actual_values[actual_event])))
                            if actual_positive
                            else float("nan")
                        ),
                    }
                )
    return summary, per_kpi


def binary_roc_auc(actual: np.ndarray, score: np.ndarray) -> float:
    positives = int(np.sum(actual))
    negatives = int(actual.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(actual.size, dtype=np.float64)
    start = 0
    while start < actual.size:
        end = start + 1
        while end < actual.size and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(np.sum(ranks[actual]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_average_precision(actual: np.ndarray, score: np.ndarray) -> float:
    positives = int(np.sum(actual))
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sorted_actual = actual[order].astype(np.float64)
    precision = np.cumsum(sorted_actual) / np.arange(1, actual.size + 1)
    return float(np.sum(precision * sorted_actual) / positives)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.size < 2:
        return float("nan")
    actual_rank = average_ranks(actual)
    predicted_rank = average_ranks(predicted)
    if np.std(actual_rank) <= 1.0e-12 or np.std(predicted_rank) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(actual_rank, predicted_rank)[0, 1])


def _terminal_metric_rows(
    actual: np.ndarray,
    predicted: np.ndarray,
    masks: np.ndarray,
    scales: np.ndarray,
    source: str,
    rollout_horizon: int,
    forecast_window_steps: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for component, metric_name in enumerate(FUTURE_TERMINAL_KPI_NAMES):
        valid = masks[:, component] > 0.0
        actual_values = actual[valid, component]
        predicted_values = predicted[valid, component]
        stats = regression_metrics(actual_values, predicted_values)
        rows.append(
            {
                "source": source,
                "rollout_horizon_steps": rollout_horizon,
                "forecast_window_steps": forecast_window_steps,
                "metric": metric_name,
                **stats,
                "normalized_mae": stats["mae"] / max(float(scales[component]), 1.0e-12),
                "spearman": spearman_correlation(actual_values, predicted_values),
                "actual_mean": float(np.mean(actual_values)),
                "predicted_mean": float(np.mean(predicted_values)),
                "sample_count": int(actual_values.size),
            }
        )

    valid_efficiency = np.all(masks > 0.0, axis=1) & (actual[:, 1] > 0.0)
    actual_efficiency = actual[valid_efficiency, 0] / actual[valid_efficiency, 1]
    predicted_efficiency = predicted[valid_efficiency, 0] / np.maximum(
        predicted[valid_efficiency, 1], 0.25
    )
    efficiency_stats = regression_metrics(actual_efficiency, predicted_efficiency)
    rows.append(
        {
            "source": source,
            "rollout_horizon_steps": rollout_horizon,
            "forecast_window_steps": forecast_window_steps,
            "metric": "future_energy_per_completed_task_wh",
            **efficiency_stats,
            "normalized_mae": float("nan"),
            "spearman": spearman_correlation(
                actual_efficiency, predicted_efficiency
            ),
            "actual_mean": float(np.mean(actual_efficiency)),
            "predicted_mean": float(np.mean(predicted_efficiency)),
            "sample_count": int(actual_efficiency.size),
        }
    )

    queue_valid = masks[:, 2] > 0.0
    queue_actual = actual[queue_valid, 2] > 0.0
    queue_score = predicted[queue_valid, 2]
    rows.append(
        {
            "source": source,
            "rollout_horizon_steps": rollout_horizon,
            "forecast_window_steps": forecast_window_steps,
            "metric": "future_charge_queue_event",
            "sample_count": int(queue_actual.size),
            "event_count": int(np.sum(queue_actual)),
            "event_prevalence": float(np.mean(queue_actual)),
            "roc_auc": binary_roc_auc(queue_actual, queue_score),
            "average_precision": binary_average_precision(queue_actual, queue_score),
        }
    )
    return rows


def future_terminal_rows(
    data: Dict[str, np.ndarray],
    horizons: List[int],
    agv_count: int,
    forecast_window_steps: int,
    terminal_scales: np.ndarray,
) -> List[Dict[str, object]]:
    if "pred_future_terminal_kpi" not in data:
        return []
    rows: List[Dict[str, object]] = []
    for horizon in horizons:
        index = horizon - 1
        rows.extend(
            _terminal_metric_rows(
                data["target_future_terminal_kpi"][:, index],
                data["pred_future_terminal_kpi"][:, index],
                data["target_future_terminal_kpi_mask"][:, index],
                terminal_scales,
                source="v14_direct_terminal_head",
                rollout_horizon=horizon,
                forecast_window_steps=forecast_window_steps,
            )
        )

    extrapolation_horizon = max(horizons)
    physical_kpis = data["pred_kpi"][:, :extrapolation_horizon] * kpi_scale(
        agv_count
    ).reshape(1, 1, -1)
    physical_congestion = np.maximum(
        data["pred_congestion_kpi"][:, :extrapolation_horizon], 0.0
    ) * float(agv_count)
    extrapolation_factor = forecast_window_steps / float(extrapolation_horizon)
    extrapolated = np.stack(
        [
            physical_kpis[:, :, 2].sum(axis=1) * extrapolation_factor,
            physical_kpis[:, :, 5].sum(axis=1) * extrapolation_factor,
            physical_congestion[:, :, 1].sum(axis=1) * extrapolation_factor,
        ],
        axis=1,
    )
    rows.extend(
        _terminal_metric_rows(
            data["target_future_terminal_kpi"][:, 0],
            extrapolated,
            data["target_future_terminal_kpi_mask"][:, 0],
            terminal_scales,
            source=f"v13_short_rollout_h{extrapolation_horizon}_extrapolated",
            rollout_horizon=1,
            forecast_window_steps=forecast_window_steps,
        )
    )
    return rows


def future_terminal_prediction_table(
    data: Dict[str, np.ndarray],
    horizons: List[int],
    agv_count: int,
    forecast_window_steps: int,
) -> List[Dict[str, object]]:
    if "pred_future_terminal_kpi" not in data:
        return []
    extrapolation_horizon = max(horizons)
    physical_kpis = data["pred_kpi"][:, :extrapolation_horizon] * kpi_scale(
        agv_count
    ).reshape(1, 1, -1)
    physical_congestion = np.maximum(
        data["pred_congestion_kpi"][:, :extrapolation_horizon], 0.0
    ) * float(agv_count)
    factor = forecast_window_steps / float(extrapolation_horizon)
    extrapolated = np.stack(
        [
            physical_kpis[:, :, 2].sum(axis=1) * factor,
            physical_kpis[:, :, 5].sum(axis=1) * factor,
            physical_congestion[:, :, 1].sum(axis=1) * factor,
        ],
        axis=1,
    )
    actual = data["target_future_terminal_kpi"][:, 0]
    direct = data["pred_future_terminal_kpi"][:, 0]
    masks = data["target_future_terminal_kpi_mask"][:, 0]
    rows: List[Dict[str, object]] = []
    for index in range(actual.shape[0]):
        if not bool(np.all(masks[index] > 0.0)):
            continue
        row: Dict[str, object] = {
            "episode_id": int(np.asarray(data["episode_id"][index]).item()),
            "start_transition_id": int(
                np.asarray(data["start_transition_id"][index]).item()
            ),
        }
        for component, name in enumerate(("energy_wh", "completed_tasks", "charge_queue_steps")):
            row[f"actual_{name}"] = float(actual[index, component])
            row[f"direct_{name}"] = float(direct[index, component])
            row[f"extrapolated_{name}"] = float(extrapolated[index, component])
        rows.append(row)
    return rows


def future_risk_rows(
    data: Dict[str, np.ndarray],
    horizons: List[int],
    forecast_window_steps: int,
    decision_threshold: float = 0.5,
) -> List[Dict[str, object]]:
    if "pred_future_congestion_risk_logits" not in data:
        return []
    rows: List[Dict[str, object]] = []
    metric_names = ("future_charge_queue_risk",)
    for horizon in horizons:
        index = horizon - 1
        logits = data["pred_future_congestion_risk_logits"][:, index]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        targets = data["target_future_congestion_risk"][:, index]
        masks = data["target_future_congestion_risk_mask"][:, index] > 0.0
        for component, metric_name in enumerate(metric_names):
            valid = masks[:, component]
            actual = targets[valid, component] > 0.5
            score = probabilities[valid, component]
            positives = int(np.sum(actual))
            negatives = int(actual.size - positives)
            predicted_at_half = score >= 0.5
            true_positive_at_half = int(np.sum(actual & predicted_at_half))
            true_negative_at_half = int(np.sum(~actual & ~predicted_at_half))
            precision_at_half = true_positive_at_half / max(
                int(np.sum(predicted_at_half)), 1
            )
            recall_at_half = true_positive_at_half / max(positives, 1)
            specificity_at_half = true_negative_at_half / max(negatives, 1)
            f1_at_half = (
                2.0
                * precision_at_half
                * recall_at_half
                / max(precision_at_half + recall_at_half, 1.0e-12)
            )
            predicted = score >= decision_threshold
            true_positive = int(np.sum(actual & predicted))
            true_negative = int(np.sum(~actual & ~predicted))
            precision = true_positive / max(int(np.sum(predicted)), 1)
            recall = true_positive / max(positives, 1)
            specificity = true_negative / max(negatives, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
            rows.append(
                {
                    "rollout_horizon_steps": horizon,
                    "forecast_window_steps": forecast_window_steps,
                    "metric": metric_name,
                    "sample_count": int(actual.size),
                    "event_count": positives,
                    "event_prevalence": float(np.mean(actual)),
                    "predicted_probability_mean": float(np.mean(score)),
                    "precision_at_0_5": float(precision_at_half),
                    "recall_at_0_5": float(recall_at_half),
                    "specificity_at_0_5": float(specificity_at_half),
                    "f1_at_0_5": float(f1_at_half),
                    "decision_threshold": float(decision_threshold),
                    "precision_at_threshold": float(precision),
                    "recall_at_threshold": float(recall),
                    "specificity_at_threshold": float(specificity),
                    "f1_at_threshold": float(f1),
                    "brier_score": float(np.mean((score - actual.astype(float)) ** 2)),
                    "roc_auc": binary_roc_auc(actual, score),
                    "average_precision": binary_average_precision(actual, score),
                }
            )
    return rows


def make_future_risk_figure(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    charge_rows = [row for row in rows if row["metric"] == "future_charge_queue_risk"]
    horizons = [int(row["rollout_horizon_steps"]) for row in charge_rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].plot(horizons, [row["roc_auc"] for row in charge_rows], marker="o")
    axes[0].plot(
        horizons, [row["average_precision"] for row in charge_rows], marker="s"
    )
    axes[0].set_title("Future charge-queue discrimination")
    axes[0].set_xlabel("Open-loop rollout horizon (steps)")
    axes[0].set_ylabel("Score")
    axes[0].legend(["ROC AUC", "Average precision"])
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        horizons, [row["precision_at_threshold"] for row in charge_rows], marker="o"
    )
    axes[1].plot(
        horizons, [row["recall_at_threshold"] for row in charge_rows], marker="s"
    )
    axes[1].plot(horizons, [row["f1_at_threshold"] for row in charge_rows], marker="^")
    axes[1].set_title("Fixed-threshold early warning")
    axes[1].set_xlabel("Open-loop rollout horizon (steps)")
    axes[1].set_ylabel("Score")
    axes[1].legend(["Precision", "Recall", "F1"])
    axes[1].grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_figure(summary: List[Dict[str, object]], per_kpi: List[Dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    horizons = [int(row["horizon_steps"]) for row in summary]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for key, label in [
        ("agent_state_mae", "AGV state"),
        ("node_state_mae", "Node state"),
        ("global_state_mae", "Global state"),
    ]:
        axes[0].plot(horizons, [float(row[key]) for row in summary], marker="o", label=label)
    axes[0].set_xlabel("Open-loop prediction horizon (steps)")
    axes[0].set_ylabel("Mean absolute error (normalized state)")
    axes[0].set_title("State prediction error growth")
    axes[0].legend(frameon=False)

    for metric in [
        "delta_time_sec",
        "delta_energy_wh",
        "blocked_delta",
        "charge_queue_blocked_agent_steps",
    ]:
        rows = [row for row in per_kpi if row["metric"] == metric]
        if not rows:
            continue
        axes[1].plot(
            [int(row["horizon_steps"]) for row in rows],
            [float(row["mae"]) for row in rows],
            marker="o",
            label=metric,
        )
    axes[1].set_xlabel("Open-loop prediction horizon (steps)")
    axes[1].set_ylabel("Mean absolute error (physical units)")
    axes[1].set_title("Physics-output prediction error growth")
    axes[1].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.future_risk_threshold < 1.0:
        raise ValueError("future-risk-threshold must be in (0, 1)")
    horizons = parse_horizons(args.horizons)
    max_horizon = max(horizons)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto" or (device == "cuda" and not torch.cuda.is_available()):
        device = "cpu"
    checkpoint_header = torch.load(
        Path(args.model_path), map_location="cpu", weights_only=False
    )
    model_version = checkpoint_header.get("model_version")
    if model_version == V14_MODEL_VERSION:
        policy = SimpleNamespace(
            model=load_multistep_world_model_v14(args.model_path, device=device)
        )
    elif model_version == V13_MODEL_VERSION:
        policy = load_multistep_world_model_policy_v13(args.model_path, device=device)
    elif model_version == V12_MODEL_VERSION:
        policy = load_multistep_world_model_policy_v12(args.model_path, device=device)
    elif model_version == V11_MODEL_VERSION:
        policy = load_multistep_world_model_policy_v11(args.model_path, device=device)
    elif model_version == V10_MODEL_VERSION:
        policy = load_multistep_world_model_policy_v10(args.model_path, device=device)
    elif model_version == V9_MODEL_VERSION:
        policy = load_multistep_world_model_policy(args.model_path, device=device)
    else:
        raise ValueError(f"Unsupported multi-step world-model version: {model_version}")
    transitions, transition_source = load_or_collect_diagnostic_transitions(args)
    future_risk_horizon = None
    future_terminal_horizon = None
    terminal_scales = None
    if model_version in {V13_MODEL_VERSION, V14_MODEL_VERSION}:
        future_risk_horizon = int(checkpoint_header["future_risk_horizon"])
        transitions = annotate_future_congestion_risk(
            transitions, horizon=future_risk_horizon
        )
    if model_version == V14_MODEL_VERSION:
        future_terminal_horizon = int(checkpoint_header["future_terminal_horizon"])
        terminal_scales = np.asarray(
            checkpoint_header["future_terminal_scale"], dtype=np.float32
        )
        transitions = annotate_future_terminal_kpis(
            transitions, horizon=future_terminal_horizon
        )
    sequences = build_sequence_samples(
        transitions, horizon=max_horizon, stride=args.sequence_stride
    )
    loader = DataLoader(
        MultiStepSequenceDataset(sequences), batch_size=args.batch_size, shuffle=False
    )
    data = collect_predictions(policy, loader, device)
    summary, per_kpi = diagnostic_rows(data, horizons, args.agv_count)
    future_rows = future_risk_rows(
        data,
        horizons,
        forecast_window_steps=int(future_risk_horizon or 0),
        decision_threshold=args.future_risk_threshold,
    )
    terminal_rows = future_terminal_rows(
        data,
        horizons,
        agv_count=args.agv_count,
        forecast_window_steps=int(future_terminal_horizon or 0),
        terminal_scales=(
            terminal_scales
            if terminal_scales is not None
            else np.ones(len(FUTURE_TERMINAL_KPI_NAMES), dtype=np.float32)
        ),
    )
    terminal_predictions = future_terminal_prediction_table(
        data,
        horizons,
        agv_count=args.agv_count,
        forecast_window_steps=int(future_terminal_horizon or 0),
    )

    write_csv(output_dir / "multistep_state_error_by_horizon.csv", summary)
    write_csv(output_dir / "multistep_kpi_error_by_horizon.csv", per_kpi)
    write_csv(output_dir / "future_congestion_risk_by_horizon.csv", future_rows)
    write_csv(output_dir / "future_terminal_kpi_by_horizon.csv", terminal_rows)
    write_csv(output_dir / "future_terminal_paired_predictions.csv", terminal_predictions)
    make_figure(summary, per_kpi, output_dir / "multistep_open_loop_diagnostics.png")
    make_future_risk_figure(
        future_rows, output_dir / "future_congestion_risk_diagnostics.png"
    )
    manifest = vars(args) | {
        "model_path": str(Path(args.model_path).resolve()),
        "diagnostic_horizons": horizons,
        "transition_count": len(transitions),
        "trajectory_source": transition_source,
        "sequence_count": len(sequences),
        "validation_protocol": "fresh_seed_open_loop_without_teacher_forcing",
        "model_version": model_version,
        "future_risk_horizon": future_risk_horizon,
        "future_risk_threshold": args.future_risk_threshold,
        "future_terminal_horizon": future_terminal_horizon,
        "future_terminal_scales": (
            terminal_scales.tolist() if terminal_scales is not None else None
        ),
    }
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Multi-step diagnostics saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
