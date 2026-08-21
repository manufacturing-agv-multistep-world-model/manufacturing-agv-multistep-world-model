from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    CounterfactualCollectionConfig,
    collect_counterfactual_samples,
    summarize_counterfactual_samples,
)
from physics_graph_world_model import WorldModelTransitionDataset
from physics_graph_world_model_counterfactual_v141 import (
    load_counterfactual_model_v141,
)


PROTOCOL = "v144_preregistered_independent_action_ranking_confirmation_v1"
UTILITY_WEIGHTS = np.asarray([-1.0, 1.0, -1.0], dtype=np.float64)


def _install_numpy_pickle_compatibility() -> None:
    """Read NumPy 2 caches on older NumPy installations."""

    if "numpy._core" not in sys.modules:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules["numpy._core"] = numpy_core
        sys.modules["numpy._core.multiarray"] = numpy_multiarray
        sys.modules["numpy._core.numeric"] = numpy_numeric


def _select_device(name: str, require_cuda: bool) -> str:
    available = torch.cuda.is_available()
    if require_cuda and not available:
        raise RuntimeError("CUDA was required but is unavailable")
    if name == "auto":
        return "cuda" if available else "cpu"
    if name == "cuda" and not available:
        raise RuntimeError("CUDA was selected but is unavailable")
    return name


def _predict(
    model: torch.nn.Module,
    samples: Sequence[Dict[str, np.ndarray]],
    batch_size: int,
    device: str,
) -> np.ndarray:
    loader = DataLoader(
        WorldModelTransitionDataset(list(samples)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions.append(
                model.forward_counterfactual(batch)["counterfactual_delta"]
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confirm V14.1 candidate-action ranking on untouched trajectories."
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-cache", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--behavior-steps", type=int, default=4000)
    parser.add_argument("--warmup-steps", type=int, default=1200)
    parser.add_argument("--sample-stride", type=int, default=80)
    parser.add_argument("--candidates-per-state", type=int, default=3)
    parser.add_argument("--max-rollout-steps", type=int, default=500)
    parser.add_argument("--data-seed", type=int, default=15400)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=15499)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _load_or_collect(args: argparse.Namespace) -> Tuple[List[Dict[str, np.ndarray]], str]:
    config = CounterfactualCollectionConfig(
        episodes=args.episodes,
        behavior_steps=args.behavior_steps,
        warmup_steps=args.warmup_steps,
        sample_stride=args.sample_stride,
        candidates_per_state=args.candidates_per_state,
        horizons_sec=COUNTERFACTUAL_HORIZONS_SEC,
        max_rollout_steps=args.max_rollout_steps,
        seed=args.data_seed,
    )
    signature = {
        "schema": PROTOCOL,
        "config": config.__dict__,
        "training_data_disjoint": True,
        "model_parameters_frozen": True,
        "primary_endpoint": "terminal_equal_normalized_utility_ranking_regret",
    }
    if args.diagnostic_cache.is_file():
        _install_numpy_pickle_compatibility()
        with gzip.open(args.diagnostic_cache, "rb") as stream:
            payload = pickle.load(stream)
        if payload.get("signature") != signature:
            raise ValueError("V14.4 cache does not match the frozen protocol")
        return payload["samples"], "cache"
    samples = collect_counterfactual_samples(config)
    args.diagnostic_cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.diagnostic_cache, "wb", compresslevel=4) as stream:
        pickle.dump(
            {"signature": signature, "samples": samples},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return samples, "collected"


def _ranking_rows(
    samples: Sequence[Dict[str, np.ndarray]],
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    scale: np.ndarray,
) -> List[Dict[str, float]]:
    terminal = len(COUNTERFACTUAL_HORIZONS_SEC) - 1
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, sample in enumerate(samples):
        if bool(np.all(mask[index, terminal])):
            key = (int(sample["episode_id"]), int(sample["state_id"]))
            groups.setdefault(key, []).append(index)

    rows: List[Dict[str, float]] = []
    terminal_scale = np.maximum(np.asarray(scale[terminal], dtype=np.float64), 1.0e-9)
    for (episode_id, state_id), indices in groups.items():
        truth = np.sum(
            target[indices, terminal] / terminal_scale * UTILITY_WEIGHTS,
            axis=1,
        )
        estimate = np.sum(
            prediction[indices, terminal] / terminal_scale * UTILITY_WEIGHTS,
            axis=1,
        )
        truth = np.concatenate([[0.0], truth])
        estimate = np.concatenate([[0.0], estimate])
        true_best = int(np.argmax(truth))
        predicted_best = int(np.argmax(estimate))
        model_regret = float(truth[true_best] - truth[predicted_best])
        baseline_regret = float(truth[true_best] - truth[0])
        rows.append(
            {
                "episode_id": float(episode_id),
                "state_id": float(state_id),
                "candidate_count": float(len(truth)),
                "model_regret": model_regret,
                "baseline_regret": baseline_regret,
                "regret_gain": baseline_regret - model_regret,
                "top1_agreement": float(predicted_best == true_best),
                "random_top1": 1.0 / float(len(truth)),
            }
        )
    return rows


def _summarize_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    model_regret = float(np.mean([row["model_regret"] for row in rows]))
    baseline_regret = float(np.mean([row["baseline_regret"] for row in rows]))
    return {
        "decision_states": float(len(rows)),
        "model_mean_regret": model_regret,
        "baseline_mean_regret": baseline_regret,
        "regret_reduction": (
            1.0 - model_regret / baseline_regret
            if baseline_regret > 1.0e-12
            else float("nan")
        ),
        "top1_agreement": float(np.mean([row["top1_agreement"] for row in rows])),
        "random_top1": float(np.mean([row["random_top1"] for row in rows])),
    }


def _bootstrap(
    rows: Sequence[Dict[str, float]], replicates: int, seed: int
) -> Dict[str, float]:
    episode_ids = sorted({int(row["episode_id"]) for row in rows})
    by_episode = {
        episode: [row for row in rows if int(row["episode_id"]) == episode]
        for episode in episode_ids
    }
    rng = np.random.default_rng(seed)
    reductions = []
    for _ in range(replicates):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        selected = [row for episode in sampled for row in by_episode[int(episode)]]
        reductions.append(_summarize_rows(selected)["regret_reduction"])
    values = np.asarray(reductions, dtype=np.float64)
    return {
        "replicates": float(replicates),
        "mean": float(np.nanmean(values)),
        "ci_low": float(np.nanquantile(values, 0.025)),
        "ci_high": float(np.nanquantile(values, 0.975)),
        "p_reduction_le_zero": float(np.nanmean(values <= 0.0)),
    }


def _episode_summary(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    output = []
    for episode in sorted({int(row["episode_id"]) for row in rows}):
        subset = [row for row in rows if int(row["episode_id"]) == episode]
        output.append({"episode_id": episode, **_summarize_rows(subset)})
    return output


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    ranking = audit["ensemble_ranking"]
    bootstrap = audit["episode_bootstrap"]
    lines = [
        "# V14.4 preregistered independent action-ranking confirmation",
        "",
        (
            f"Untouched data seed: {audit['data_seed']}; trajectories: "
            f"{audit['episodes']}; paired candidate samples: {audit['samples']}."
        ),
        "",
        "The primary endpoint is candidate-action ranking at the 720-second physical horizon. "
        "Energy, completed tasks, and charge-queue time receive equal weights after frozen "
        "training-scale normalization. The unchanged DT-aware action is the zero-effect baseline.",
        "",
        f"- Eligible decision states: {int(ranking['decision_states'])}",
        f"- Model mean regret: {ranking['model_mean_regret']:.5f}",
        f"- Baseline mean regret: {ranking['baseline_mean_regret']:.5f}",
        f"- Regret reduction: {ranking['regret_reduction']:.3f}",
        f"- Top-1 agreement: {ranking['top1_agreement']:.3f}",
        f"- Random-choice top-1 expectation: {ranking['random_top1']:.3f}",
        (
            f"- Episode-bootstrap 95% CI: [{bootstrap['ci_low']:.3f}, "
            f"{bootstrap['ci_high']:.3f}]"
        ),
        "",
        "## Episode-level stability",
        "",
        "| Episode | States | Regret reduction | Top-1 |",
        "|---:|---:|---:|---:|",
    ]
    for row in audit["episode_results"]:
        lines.append(
            f"| {row['episode_id']} | {int(row['decision_states'])} "
            f"| {row['regret_reduction']:+.3f} | {row['top1_agreement']:.3f} |"
        )
    lines.extend(["", "## Preregistered continuation criteria", ""])
    lines.extend(
        f"- [{'x' if item['passed'] else ' '}] {item['criterion']}"
        for item in audit["criteria"]
    )
    lines.extend(
        [
            "",
            "Proceed to shadow decision evaluation: "
            f"**{'YES' if audit['passed'] else 'NO'}**.",
            "",
            "Passing supports action-ranking validity only. It is not a closed-loop "
            "throughput, energy, or safety claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose(args: argparse.Namespace) -> Path:
    if len(args.checkpoint) != 3:
        raise ValueError("V14.4 requires exactly three frozen model seeds")
    if args.episodes != 12 or args.data_seed != 15400:
        raise ValueError("V14.4 formal confirmation is frozen at 12 episodes and seed 15400")
    if args.bootstrap_replicates < 2000:
        raise ValueError("V14.4 requires at least 2000 trajectory-bootstrap replicates")
    for checkpoint in args.checkpoint:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    device = _select_device(args.device, args.require_cuda)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, cache_source = _load_or_collect(args)
    target = np.stack([sample["target_delta"] for sample in samples])
    mask = np.stack([sample["target_mask"] for sample in samples]) > 0.0
    predictions = []
    scales = []
    for checkpoint in args.checkpoint:
        model = load_counterfactual_model_v141(checkpoint, device=device)
        predictions.append(_predict(model, samples, args.batch_size, device))
        scales.append(model.counterfactual_scale.detach().cpu().numpy())
    prediction_array = np.stack(predictions)
    ensemble_prediction = np.mean(prediction_array, axis=0)
    ensemble_scale = np.mean(np.stack(scales), axis=0)
    ensemble_rows = _ranking_rows(
        samples, ensemble_prediction, target, mask, ensemble_scale
    )
    ensemble_ranking = _summarize_rows(ensemble_rows)
    model_rankings = [
        _summarize_rows(_ranking_rows(samples, pred, target, mask, scale))
        for pred, scale in zip(predictions, scales)
    ]
    episode_results = _episode_summary(ensemble_rows)
    bootstrap = _bootstrap(
        ensemble_rows, args.bootstrap_replicates, args.bootstrap_seed
    )
    dataset_summary = summarize_counterfactual_samples(samples)
    nonzero = np.asarray(dataset_summary["target_nonzero_rate"], dtype=np.float64)
    positive_episodes = sum(row["regret_reduction"] > 0.0 for row in episode_results)
    criteria = [
        {
            "criterion": "Exactly 12 untouched complete trajectories are evaluated",
            "passed": len(episode_results) == 12,
        },
        {
            "criterion": "At least 1000 paired candidate samples are available",
            "passed": len(samples) >= 1000,
        },
        {
            "criterion": "At least 350 complete terminal decision states are ranked",
            "passed": ensemble_ranking["decision_states"] >= 350,
        },
        {
            "criterion": "Every target component has at least 85% valid physical-time coverage",
            "passed": bool(
                np.all(np.asarray(dataset_summary["valid_target_rate"]) >= 0.85)
            ),
        },
        {
            "criterion": "Terminal energy and task effects each have at least 10% nonzero coverage",
            "passed": bool(nonzero[-1, 0] >= 0.10 and nonzero[-1, 1] >= 0.10),
        },
        {
            "criterion": "Ensemble terminal ranking regret is reduced by at least 15%",
            "passed": ensemble_ranking["regret_reduction"] >= 0.15,
        },
        {
            "criterion": "Trajectory-bootstrap 95% interval for regret reduction is above zero",
            "passed": bootstrap["ci_low"] > 0.0,
        },
        {
            "criterion": "At least 9 of 12 trajectories have positive regret reduction",
            "passed": positive_episodes >= 9,
        },
        {
            "criterion": "Top-1 agreement exceeds random choice by at least 8 percentage points",
            "passed": (
                ensemble_ranking["top1_agreement"]
                - ensemble_ranking["random_top1"]
                >= 0.08
            ),
        },
        {
            "criterion": "At least two of three frozen model seeds independently reduce regret",
            "passed": sum(row["regret_reduction"] > 0.0 for row in model_rankings) >= 2,
        },
    ]
    audit: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "preregistered_confirmation",
        "data_seed": args.data_seed,
        "episodes": args.episodes,
        "samples": len(samples),
        "cache_source": cache_source,
        "checkpoints": [str(path) for path in args.checkpoint],
        "utility_weights": {
            "energy_wh": -1.0,
            "completed_tasks": 1.0,
            "charge_queue_time_sec": -1.0,
        },
        "dataset_summary": dataset_summary,
        "ensemble_ranking": ensemble_ranking,
        "individual_model_rankings": model_rankings,
        "episode_results": episode_results,
        "episode_bootstrap": bootstrap,
        "positive_episode_count": positive_episodes,
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
    }
    audit_path = args.output_dir / "V144_RANKING_CONFIRMATION_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown_path = args.output_dir / "V144_RANKING_CONFIRMATION_AUDIT.md"
    _write_markdown(markdown_path, audit)
    print(markdown_path.read_text(encoding="utf-8"))
    return args.output_dir


if __name__ == "__main__":
    diagnose(build_parser().parse_args())
