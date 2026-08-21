from __future__ import annotations

import argparse
import csv
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
    COUNTERFACTUAL_METRIC_NAMES,
    CounterfactualCollectionConfig,
    collect_counterfactual_samples,
    summarize_counterfactual_samples,
)
from physics_graph_world_model import WorldModelTransitionDataset
from physics_graph_world_model_counterfactual_v141 import (
    load_counterfactual_model_v141,
)


DIAGNOSTIC_SCHEMA = "v141_independent_paired_counterfactual_confirmation_v1"


def _install_numpy_pickle_compatibility() -> None:
    """Read NumPy 2 caches on NumPy 1 installations without changing payloads."""

    if "numpy._core" not in sys.modules:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules["numpy._core"] = numpy_core
        sys.modules["numpy._core.multiarray"] = numpy_multiarray
        sys.modules["numpy._core.numeric"] = numpy_numeric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently confirm V14.1 paired counterfactual predictions."
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-cache", type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--behavior-steps", type=int, default=4000)
    parser.add_argument("--warmup-steps", type=int, default=1200)
    parser.add_argument("--sample-stride", type=int, default=80)
    parser.add_argument("--candidates-per-state", type=int, default=3)
    parser.add_argument("--max-rollout-steps", type=int, default=500)
    parser.add_argument("--data-seed", type=int, default=15100)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=15199)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _select_device(name: str, require_cuda: bool) -> str:
    available = torch.cuda.is_available()
    if require_cuda and not available:
        raise RuntimeError("CUDA was required but is unavailable")
    if name == "auto":
        return "cuda" if available else "cpu"
    if name == "cuda" and not available:
        raise RuntimeError("CUDA was selected but is unavailable")
    return name


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
        "schema": DIAGNOSTIC_SCHEMA,
        "config": config.__dict__,
        "training_data_disjoint": True,
    }
    cache = args.diagnostic_cache
    if cache is not None and cache.is_file():
        _install_numpy_pickle_compatibility()
        with gzip.open(cache, "rb") as stream:
            payload = pickle.load(stream)
        if payload.get("signature") != signature:
            raise ValueError("Diagnostic cache does not match the frozen protocol")
        return payload["samples"], "cache"
    samples = collect_counterfactual_samples(config)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wb", compresslevel=4) as stream:
            pickle.dump(
                {"signature": signature, "samples": samples},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return samples, "collected"


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
                .cpu()
                .numpy()
            )
    return np.concatenate(predictions)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])


def _component_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    scale: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []
    for horizon_index, horizon in enumerate(COUNTERFACTUAL_HORIZONS_SEC):
        for metric_index, metric in enumerate(COUNTERFACTUAL_METRIC_NAMES):
            valid = mask[:, horizon_index, metric_index]
            truth = target[valid, horizon_index, metric_index]
            estimate = prediction[valid, horizon_index, metric_index]
            error = np.abs(estimate - truth)
            zero_error = np.abs(truth)
            material = np.abs(truth) >= 0.25 * scale[horizon_index, metric_index]
            rows.append(
                {
                    "horizon_sec": float(horizon),
                    "metric": metric,
                    "samples": int(truth.size),
                    "nonzero_rate": float(np.mean(np.abs(truth) > 1.0e-6)),
                    "mae": float(np.mean(error)),
                    "normalized_mae": float(
                        np.mean(error) / scale[horizon_index, metric_index]
                    ),
                    "zero_mae": float(np.mean(zero_error)),
                    "mae_gain_over_zero": float(np.mean(zero_error - error)),
                    "spearman": _spearman(estimate, truth),
                    "material_samples": int(np.sum(material)),
                    "material_sign_accuracy": float(
                        np.mean(np.sign(estimate[material]) == np.sign(truth[material]))
                    )
                    if np.any(material)
                    else float("nan"),
                }
            )
    return rows


def _episode_bootstrap_gain(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    episode_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(episode_ids)
    rows = []
    for horizon_index, horizon in enumerate(COUNTERFACTUAL_HORIZONS_SEC):
        for metric_index, metric in enumerate(COUNTERFACTUAL_METRIC_NAMES):
            gains = []
            for _ in range(replicates):
                sampled = rng.choice(unique, size=unique.size, replace=True)
                indices = np.concatenate(
                    [np.flatnonzero(episode_ids == episode) for episode in sampled]
                )
                valid = mask[indices, horizon_index, metric_index]
                truth = target[indices, horizon_index, metric_index][valid]
                estimate = prediction[indices, horizon_index, metric_index][valid]
                gains.append(float(np.mean(np.abs(truth) - np.abs(estimate - truth))))
            rows.append(
                {
                    "horizon_sec": float(horizon),
                    "metric": metric,
                    "mean_gain": float(np.mean(gains)),
                    "ci_low": float(np.quantile(gains, 0.025)),
                    "ci_high": float(np.quantile(gains, 0.975)),
                    "p_gain_le_zero": float(np.mean(np.asarray(gains) <= 0.0)),
                }
            )
    return rows


def _ranking_metrics(
    samples: Sequence[Dict[str, np.ndarray]],
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    scale: np.ndarray,
) -> Dict[str, float]:
    horizon_index = len(COUNTERFACTUAL_HORIZONS_SEC) - 1
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, sample in enumerate(samples):
        key = (int(sample["episode_id"]), int(sample["state_id"]))
        if bool(np.all(mask[index, horizon_index])):
            groups.setdefault(key, []).append(index)
    regrets = []
    baseline_regrets = []
    agreements = []
    weights = np.asarray([-1.0, 1.0, -1.0], dtype=np.float64)
    terminal_scale = scale[horizon_index]
    for indices in groups.values():
        true_scores = np.sum(
            target[indices, horizon_index] / terminal_scale * weights, axis=1
        )
        predicted_scores = np.sum(
            prediction[indices, horizon_index] / terminal_scale * weights, axis=1
        )
        true_with_baseline = np.concatenate([[0.0], true_scores])
        predicted_with_baseline = np.concatenate([[0.0], predicted_scores])
        true_best = int(np.argmax(true_with_baseline))
        predicted_best = int(np.argmax(predicted_with_baseline))
        regrets.append(
            float(true_with_baseline[true_best] - true_with_baseline[predicted_best])
        )
        baseline_regrets.append(float(true_with_baseline[true_best]))
        agreements.append(float(true_best == predicted_best))
    return {
        "decision_states": float(len(groups)),
        "top1_agreement": float(np.mean(agreements)),
        "mean_regret": float(np.mean(regrets)),
        "baseline_mean_regret": float(np.mean(baseline_regrets)),
        "regret_reduction": float(
            1.0 - np.mean(regrets) / np.mean(baseline_regrets)
        )
        if np.mean(baseline_regrets) > 1.0e-12
        else float("nan"),
    }


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    lines = [
        "# V14.1 paired counterfactual independent confirmation",
        "",
        (
            f"Fresh data seed: {audit['data_seed']}; episodes: {audit['episodes']}; "
            f"paired samples: {audit['samples']}."
        ),
        "",
        "| Horizon (s) | Metric | Nonzero | MAE | Zero MAE | Gain | Spearman | Sign accuracy |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["ensemble_components"]:
        lines.append(
            f"| {row['horizon_sec']:.0f} | {row['metric']} | {row['nonzero_rate']:.3f} "
            f"| {row['mae']:.4f} | {row['zero_mae']:.4f} "
            f"| {row['mae_gain_over_zero']:+.4f} | {row['spearman']:.3f} "
            f"| {row['material_sign_accuracy']:.3f} |"
        )
    ranking = audit["ranking"]
    lines.extend(
        [
            "",
            "## Action-ranking audit",
            "",
            f"- Decision states: {int(ranking['decision_states'])}",
            f"- Top-1 agreement: {ranking['top1_agreement']:.3f}",
            f"- Model mean regret: {ranking['mean_regret']:.4f}",
            f"- Baseline mean regret: {ranking['baseline_mean_regret']:.4f}",
            f"- Regret reduction: {ranking['regret_reduction']:.3f}",
            "",
            "## Preregistered continuation criteria",
            "",
        ]
    )
    lines.extend(
        f"- [{'x' if item['passed'] else ' '}] {item['criterion']}"
        for item in audit["criteria"]
    )
    lines.extend(
        [
            "",
            "Proceed to shadow control evaluation: "
            f"**{'YES' if audit['passed'] else 'NO'}**.",
            "",
            "Passing is prediction and ranking evidence, not a closed-loop performance claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose(args: argparse.Namespace) -> Path:
    if len(args.checkpoint) != 3:
        raise ValueError("The formal audit requires exactly three model seeds")
    if args.bootstrap_replicates < 200:
        raise ValueError("At least 200 bootstrap replicates are required")
    for checkpoint in args.checkpoint:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    device = _select_device(args.device, args.require_cuda)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, cache_source = _load_or_collect(args)
    target = np.stack([sample["target_delta"] for sample in samples])
    mask = np.stack([sample["target_mask"] for sample in samples]) > 0.0
    episode_ids = np.asarray([int(sample["episode_id"]) for sample in samples])
    predictions = []
    scales = []
    for checkpoint in args.checkpoint:
        model = load_counterfactual_model_v141(checkpoint, device=device)
        predictions.append(_predict(model, samples, args.batch_size, device))
        scales.append(model.counterfactual_scale.detach().cpu().numpy())
    prediction_array = np.stack(predictions)
    ensemble = np.mean(prediction_array, axis=0)
    scale = np.mean(np.stack(scales), axis=0)
    individual_components = [
        _component_metrics(prediction, target, mask, model_scale)
        for prediction, model_scale in zip(predictions, scales)
    ]
    ensemble_components = _component_metrics(ensemble, target, mask, scale)
    bootstrap = _episode_bootstrap_gain(
        ensemble,
        target,
        mask,
        episode_ids,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    ranking = _ranking_metrics(samples, ensemble, target, mask, scale)
    dataset_summary = summarize_counterfactual_samples(samples)
    terminal = {
        row["metric"]: row
        for row in ensemble_components
        if row["horizon_sec"] == COUNTERFACTUAL_HORIZONS_SEC[-1]
    }
    terminal_bootstrap = {
        row["metric"]: row
        for row in bootstrap
        if row["horizon_sec"] == COUNTERFACTUAL_HORIZONS_SEC[-1]
    }
    criteria = [
        {
            "criterion": "At least 500 independent paired samples",
            "passed": len(samples) >= 500,
        },
        {
            "criterion": "Every target component has at least 85% valid physical-time coverage",
            "passed": bool(
                np.all(np.asarray(dataset_summary["valid_target_rate"]) >= 0.85)
            ),
        },
        {
            "criterion": "Terminal energy and task effects are identifiable (nonzero rate >= 10%)",
            "passed": all(
                terminal[name]["nonzero_rate"] >= 0.10
                for name in ("energy_wh", "completed_tasks")
            ),
        },
        {
            "criterion": "Terminal queue effects are identifiable (nonzero rate >= 2%)",
            "passed": terminal["charge_queue_time_sec"]["nonzero_rate"] >= 0.02,
        },
        {
            "criterion": "Ensemble terminal MAE beats the zero-effect baseline for all metrics",
            "passed": all(row["mae_gain_over_zero"] > 0.0 for row in terminal.values()),
        },
        {
            "criterion": "At least two terminal bootstrap intervals favor V14.1 over zero",
            "passed": sum(row["ci_low"] > 0.0 for row in terminal_bootstrap.values()) >= 2,
        },
        {
            "criterion": "Material terminal-effect sign accuracy is at least 65% for energy and tasks",
            "passed": all(
                terminal[name]["material_sign_accuracy"] >= 0.65
                for name in ("energy_wh", "completed_tasks")
            ),
        },
        {
            "criterion": "Counterfactual action ranking reduces mean regret by at least 10%",
            "passed": ranking["regret_reduction"] >= 0.10,
        },
    ]
    audit = {
        "protocol": DIAGNOSTIC_SCHEMA,
        "data_seed": args.data_seed,
        "episodes": args.episodes,
        "samples": len(samples),
        "cache_source": cache_source,
        "checkpoints": [str(path) for path in args.checkpoint],
        "dataset_summary": dataset_summary,
        "individual_components": individual_components,
        "ensemble_components": ensemble_components,
        "bootstrap_gain": bootstrap,
        "ranking": ranking,
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
    }
    (args.output_dir / "counterfactual_confirmation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "counterfactual_component_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ensemble_components[0]))
        writer.writeheader()
        writer.writerows(ensemble_components)
    _write_markdown(args.output_dir / "COUNTERFACTUAL_CONFIRMATION_AUDIT.md", audit)
    print((args.output_dir / "COUNTERFACTUAL_CONFIRMATION_AUDIT.md").read_text(encoding="utf-8"))
    return args.output_dir


if __name__ == "__main__":
    diagnose(build_parser().parse_args())
