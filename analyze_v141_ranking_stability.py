from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from physics_graph_world_model import WorldModelTransitionDataset
from physics_graph_world_model_counterfactual_v141 import (
    load_counterfactual_model_v141,
)


def _install_numpy_pickle_compatibility() -> None:
    if "numpy._core" not in sys.modules:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules["numpy._core"] = numpy_core
        sys.modules["numpy._core.multiarray"] = numpy_multiarray
        sys.modules["numpy._core.numeric"] = numpy_numeric


def _predict(model: torch.nn.Module, samples: Sequence[dict]) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(
            WorldModelTransitionDataset(list(samples)), batch_size=256
        ):
            outputs.append(
                model.forward_counterfactual(batch)["counterfactual_delta"].numpy()
            )
    return np.concatenate(outputs)


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    _install_numpy_pickle_compatibility()
    with gzip.open(args.cache, "rb") as stream:
        samples = pickle.load(stream)["samples"]
    predictions = []
    scales = []
    for checkpoint in args.checkpoint:
        model = load_counterfactual_model_v141(checkpoint).eval()
        predictions.append(_predict(model, samples))
        scales.append(model.counterfactual_scale.numpy())
    prediction = np.mean(np.stack(predictions), axis=0)
    scale = np.mean(np.stack(scales), axis=0)
    target = np.stack([sample["target_delta"] for sample in samples])
    mask = np.stack([sample["target_mask"] for sample in samples]) > 0.0
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, sample in enumerate(samples):
        if bool(np.all(mask[index, -1])):
            key = (int(sample["episode_id"]), int(sample["state_id"]))
            groups.setdefault(key, []).append(index)

    utility_weights = np.asarray([-1.0, 1.0, -1.0])
    rows = []
    for (episode_id, _), indices in groups.items():
        true_scores = np.sum(
            target[indices, -1] / scale[-1] * utility_weights, axis=1
        )
        predicted_scores = np.sum(
            prediction[indices, -1] / scale[-1] * utility_weights, axis=1
        )
        true_values = np.concatenate([[0.0], true_scores])
        predicted_values = np.concatenate([[0.0], predicted_scores])
        true_best = int(np.argmax(true_values))
        predicted_best = int(np.argmax(predicted_values))
        rows.append(
            {
                "episode_id": episode_id,
                "regret": float(true_values[true_best] - true_values[predicted_best]),
                "baseline_regret": float(true_values[true_best]),
                "predicted_baseline": int(predicted_best == 0),
                "true_baseline": int(true_best == 0),
                "top1_agreement": int(predicted_best == true_best),
            }
        )
    episode_ids = sorted({row["episode_id"] for row in rows})
    episode_rows = []
    for episode_id in episode_ids:
        subset = [row for row in rows if row["episode_id"] == episode_id]
        regret = float(np.mean([row["regret"] for row in subset]))
        baseline_regret = float(np.mean([row["baseline_regret"] for row in subset]))
        episode_rows.append(
            {
                "episode_id": episode_id,
                "states": len(subset),
                "regret": regret,
                "baseline_regret": baseline_regret,
                "regret_reduction": (
                    1.0 - regret / baseline_regret
                    if baseline_regret > 1.0e-12
                    else float("nan")
                ),
            }
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    reductions = []
    for _ in range(args.bootstrap_replicates):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        subset = [row for episode in sampled for row in rows if row["episode_id"] == episode]
        regret = float(np.mean([row["regret"] for row in subset]))
        baseline_regret = float(np.mean([row["baseline_regret"] for row in subset]))
        reductions.append(
            1.0 - regret / baseline_regret if baseline_regret > 1.0e-12 else np.nan
        )
    reductions_array = np.asarray(reductions)
    result = {
        "status": "post_hoc_stability_audit_not_preregistered_evidence",
        "states": len(rows),
        "predicted_baseline_rate": float(
            np.mean([row["predicted_baseline"] for row in rows])
        ),
        "true_baseline_rate": float(
            np.mean([row["true_baseline"] for row in rows])
        ),
        "top1_agreement": float(
            np.mean([row["top1_agreement"] for row in rows])
        ),
        "episode_results": episode_rows,
        "regret_reduction_bootstrap_mean": float(np.nanmean(reductions_array)),
        "regret_reduction_ci95": np.nanquantile(
            reductions_array, [0.025, 0.975]
        ).tolist(),
        "p_regret_reduction_le_zero": float(
            np.nanmean(reductions_array <= 0.0)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=15200)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
