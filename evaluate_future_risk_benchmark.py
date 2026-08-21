from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from agv_case_env import AGV_A_Charge_Env
from diagnose_world_model_multistep import binary_average_precision, binary_roc_auc
from future_risk_baselines import (
    RiskDataset,
    build_risk_samples,
    heuristic_charge_risk_score,
    load_baseline_model,
    move_batch as move_risk_batch,
)
from physics_graph_world_model import collect_world_model_transitions
from physics_graph_world_model_multistep import (
    MultiStepSequenceDataset,
    build_sequence_samples,
)
from physics_graph_world_model_multistep_v13 import (
    annotate_future_congestion_risk,
    load_multistep_world_model_policy_v13,
)
from train_world_model_multistep import move_batch as move_world_model_batch


ROOT = Path(__file__).resolve().parent
MODEL_SEEDS = (42, 43, 44)
BASELINE_ARCHITECTURES = ("mlp", "gru", "gnn")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fair held-out benchmark for future charge-queue prediction."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--sequence-stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=25313)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classification_summary(actual: np.ndarray, score: np.ndarray):
    return {
        "roc_auc": binary_roc_auc(actual, score),
        "average_precision": binary_average_precision(actual, score),
        "brier_score": float(np.mean((score - actual.astype(float)) ** 2)),
        "event_prevalence": float(np.mean(actual)),
        "sample_count": int(actual.size),
        "event_count": int(np.sum(actual)),
    }


def baseline_probabilities(model, samples, batch_size: int, device: str):
    loader = DataLoader(
        RiskDataset(samples), batch_size=batch_size, shuffle=False
    )
    values = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_risk_batch(batch, device)
            values.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(values)


def v13_probabilities(policy, sequences, batch_size: int, device: str):
    loader = DataLoader(
        MultiStepSequenceDataset(sequences), batch_size=batch_size, shuffle=False
    )
    values = []
    policy.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_world_model_batch(batch, device)
            output = policy.model.rollout(batch, teacher_forcing_ratio=0.0)
            values.append(
                torch.sigmoid(output["pred_future_congestion_risk_logits"][:, 0, 0])
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(values)


def paired_episode_bootstrap(
    actual: np.ndarray,
    episode_ids: np.ndarray,
    full_score: np.ndarray,
    baseline_score: np.ndarray,
    replicates: int,
    seed: int,
):
    unique_episodes = np.unique(episode_ids)
    episode_indices = {
        episode: np.flatnonzero(episode_ids == episode) for episode in unique_episodes
    }
    rng = np.random.default_rng(seed)
    auc_deltas = []
    ap_deltas = []
    for _ in range(replicates):
        sampled = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)
        indices = np.concatenate([episode_indices[episode] for episode in sampled])
        y = actual[indices]
        if np.all(y) or not np.any(y):
            continue
        auc_deltas.append(
            binary_roc_auc(y, full_score[indices])
            - binary_roc_auc(y, baseline_score[indices])
        )
        ap_deltas.append(
            binary_average_precision(y, full_score[indices])
            - binary_average_precision(y, baseline_score[indices])
        )
    if not auc_deltas:
        raise RuntimeError("Episode bootstrap produced no identifiable replicate")

    def summarize(values):
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)),
            "probability_nonpositive": float(np.mean(array <= 0.0)),
        }

    return summarize(auc_deltas), summarize(ap_deltas), len(auc_deltas)


def make_figure(summary_rows, bootstrap_rows, path: Path):
    import matplotlib.pyplot as plt

    labels = [row["architecture"] for row in summary_rows]
    auc = [row["roc_auc"] for row in summary_rows]
    ap = [row["average_precision"] for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, auc, width=0.36, label="ROC AUC")
    axes[0].bar(x + 0.18, ap, width=0.36, label="Average precision")
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Future charge-queue prediction")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    comparisons = [row for row in bootstrap_rows if row["metric"] == "roc_auc"]
    bx = np.arange(len(comparisons))
    means = np.asarray([row["delta_mean"] for row in comparisons])
    low = np.asarray([row["ci_low"] for row in comparisons])
    high = np.asarray([row["ci_high"] for row in comparisons])
    axes[1].errorbar(
        bx,
        means,
        yerr=np.vstack([means - low, high - means]),
        fmt="o",
        capsize=4,
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xticks(
        bx, [row["baseline_architecture"] for row in comparisons], rotation=20
    )
    axes[1].set_title("V13 minus baseline (episode bootstrap)")
    axes[1].set_ylabel("ROC AUC difference")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but unavailable")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def env_factory(seed: int) -> AGV_A_Charge_Env:
        return AGV_A_Charge_Env(
            agv_count=3,
            env_variant="full",
            reward_mode="hybrid",
            scenario="rush",
            dispatch_rule="dt_aware",
            capacity_mode="stress",
            max_steps=args.max_steps,
            seed=seed,
        )

    transitions = collect_world_model_transitions(
        env_factory=env_factory,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        exploration_rate=0.35,
    )
    transitions = annotate_future_congestion_risk(transitions, horizon=80)
    samples = build_risk_samples(
        transitions, stride=args.sequence_stride, rollout_horizon=10
    )
    all_sequences = build_sequence_samples(
        transitions, horizon=10, stride=args.sequence_stride
    )
    sequence_by_key = {
        (int(row["episode_id"]), int(row["start_transition_id"])): row
        for row in all_sequences
    }
    aligned_sequences = [
        sequence_by_key[(int(sample["episode_id"]), int(sample["transition_id"]))]
        for sample in samples
    ]
    actual = np.asarray([float(sample["target"]) > 0.5 for sample in samples])
    episode_ids = np.asarray([int(sample["episode_id"]) for sample in samples])

    scores: Dict[str, np.ndarray] = {
        "Rule": np.asarray([heuristic_charge_risk_score(sample) for sample in samples])
    }
    method_metadata: Dict[str, Dict[str, object]] = {
        "Rule": {"architecture": "Rule", "model_seed": "deterministic"}
    }
    for architecture in BASELINE_ARCHITECTURES:
        for model_seed in MODEL_SEEDS:
            name = f"{architecture.upper()}-s{model_seed}"
            checkpoint_path = (
                ROOT
                / "world_model_runs"
                / "future_risk_baselines_v1"
                / f"{architecture}_seed{model_seed}"
                / "model.pt"
            )
            model, checkpoint = load_baseline_model(
                checkpoint_path, samples[0], args.device
            )
            scores[name] = baseline_probabilities(
                model, samples, args.batch_size, args.device
            )
            method_metadata[name] = {
                "architecture": architecture.upper(),
                "model_seed": model_seed,
                "parameters": checkpoint["trainable_parameters"],
            }
    for model_seed in MODEL_SEEDS:
        name = f"V13-s{model_seed}"
        model_path = (
            ROOT
            / "world_model_runs"
            / f"pi_gwm_multistep_v13_multiscale_v2_seed{model_seed}"
            / "physics_graph_world_model_multistep.pt"
        )
        policy = load_multistep_world_model_policy_v13(model_path, device=args.device)
        scores[name] = v13_probabilities(
            policy, aligned_sequences, args.batch_size, args.device
        )
        method_metadata[name] = {
            "architecture": "V13",
            "model_seed": model_seed,
            "parameters": sum(parameter.numel() for parameter in policy.model.parameters()),
        }

    method_rows = []
    prediction_rows = []
    episode_rows = []
    for name, score in scores.items():
        method_rows.append({"method": name, **method_metadata[name], **classification_summary(actual, score)})
        for index in range(len(actual)):
            prediction_rows.append(
                {
                    "method": name,
                    "episode_id": int(episode_ids[index]),
                    "sample_index": index,
                    "actual": int(actual[index]),
                    "score": float(score[index]),
                }
            )
        for episode in np.unique(episode_ids):
            mask = episode_ids == episode
            if np.any(actual[mask]) and not np.all(actual[mask]):
                episode_rows.append(
                    {
                        "method": name,
                        "episode_id": int(episode),
                        **classification_summary(actual[mask], score[mask]),
                    }
                )

    ensembles: Dict[str, np.ndarray] = {"Rule": scores["Rule"]}
    for architecture in BASELINE_ARCHITECTURES:
        ensembles[architecture.upper()] = np.mean(
            [scores[f"{architecture.upper()}-s{seed}"] for seed in MODEL_SEEDS], axis=0
        )
    ensembles["V13"] = np.mean(
        [scores[f"V13-s{seed}"] for seed in MODEL_SEEDS], axis=0
    )
    architecture_rows = [
        {"architecture": name, **classification_summary(actual, score)}
        for name, score in ensembles.items()
    ]
    bootstrap_rows = []
    for index, (baseline, baseline_score) in enumerate(ensembles.items()):
        if baseline == "V13":
            continue
        auc_stats, ap_stats, valid_replicates = paired_episode_bootstrap(
            actual,
            episode_ids,
            ensembles["V13"],
            baseline_score,
            args.bootstrap_replicates,
            args.seed + index,
        )
        for metric, stats in (("roc_auc", auc_stats), ("average_precision", ap_stats)):
            bootstrap_rows.append(
                {
                    "full_architecture": "V13",
                    "baseline_architecture": baseline,
                    "metric": metric,
                    "delta_mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "probability_nonpositive": stats["probability_nonpositive"],
                    "valid_replicates": valid_replicates,
                }
            )

    write_csv(output_dir / "method_seed_summary.csv", method_rows)
    write_csv(output_dir / "architecture_ensemble_summary.csv", architecture_rows)
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "paired_bootstrap_v13_vs_baselines.csv", bootstrap_rows)
    write_csv(output_dir / "sample_predictions.csv", prediction_rows)
    make_figure(
        architecture_rows,
        bootstrap_rows,
        output_dir / "future_risk_baseline_benchmark.png",
    )
    manifest = vars(args) | {
        "protocol": "same_trajectory_same_label_paired_architecture_benchmark",
        "future_risk_horizon": 80,
        "history_length": 20,
        "methods": list(ensembles),
        "sample_count": len(samples),
        "event_count": int(np.sum(actual)),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Future-risk benchmark saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
