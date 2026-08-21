from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from counterfactual_rollout_v141 import COUNTERFACTUAL_METRIC_NAMES
from physics_graph_world_model_counterfactual_v142 import MATERIAL_EFFECT_FRACTION
from train_counterfactual_world_model_v141 import (
    _grouped_split,
    _install_numpy_pickle_compatibility,
)


def _features(samples: Sequence[dict]) -> np.ndarray:
    rows = []
    for sample in samples:
        baseline = np.asarray(sample["baseline_actions"], dtype=np.int64)
        candidate = np.asarray(sample["candidate_actions"], dtype=np.int64)
        baseline_one_hot = np.eye(4, dtype=np.float32)[baseline].reshape(-1)
        candidate_one_hot = np.eye(4, dtype=np.float32)[candidate].reshape(-1)
        rows.append(
            np.concatenate(
                [
                    np.asarray(sample["agent_features"]).reshape(-1),
                    np.asarray(sample["node_features"]).reshape(-1),
                    np.asarray(sample["global_features"]).reshape(-1),
                    baseline_one_hot,
                    candidate_one_hot,
                    candidate_one_hot - baseline_one_hot,
                    np.asarray(
                        sample.get("counterfactual_aux_features", []),
                        dtype=np.float32,
                    ).reshape(-1),
                ]
            )
        )
    return np.stack(rows).astype(np.float32)


def _metrics(prediction: np.ndarray, target: np.ndarray, scale: float) -> Dict[str, float]:
    error = np.abs(prediction - target)
    zero_error = np.abs(target)
    material = np.abs(target) >= MATERIAL_EFFECT_FRACTION * scale
    return {
        "mae": float(np.mean(error)),
        "zero_mae": float(np.mean(zero_error)),
        "mae_gain_over_zero": float(np.mean(zero_error - error)),
        "material_rate": float(np.mean(material)),
        "material_sign_accuracy": (
            float(np.mean(np.sign(prediction[material]) == np.sign(target[material])))
            if np.any(material)
            else float("nan")
        ),
    }


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    _install_numpy_pickle_compatibility()
    with gzip.open(args.cache, "rb") as stream:
        samples = pickle.load(stream)["samples"]
    train, valid, train_ids, valid_ids = _grouped_split(
        samples, args.validation_fraction, args.split_seed
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scales = np.asarray(checkpoint["counterfactual_scale"], dtype=np.float32)
    train_x = _features(train)
    valid_x = _features(valid)
    standardizer = StandardScaler().fit(train_x)
    train_scaled = standardizer.transform(train_x)
    valid_scaled = standardizer.transform(valid_x)
    train_targets = np.stack([sample["target_delta"] for sample in train])
    valid_targets = np.stack([sample["target_delta"] for sample in valid])
    train_masks = np.stack([sample["target_mask"] for sample in train]) > 0.0
    valid_masks = np.stack([sample["target_mask"] for sample in valid]) > 0.0
    rows = []
    for horizon in range(train_targets.shape[1]):
        for metric in range(train_targets.shape[2]):
            train_valid = train_masks[:, horizon, metric]
            test_valid = valid_masks[:, horizon, metric]
            y_train = train_targets[train_valid, horizon, metric]
            y_test = valid_targets[test_valid, horizon, metric]
            ridge = Ridge(alpha=10.0, solver="lsqr").fit(
                train_scaled[train_valid], y_train
            )
            ridge_prediction = ridge.predict(valid_scaled[test_valid])
            trees = ExtraTreesRegressor(
                n_estimators=args.trees,
                min_samples_leaf=3,
                max_features=0.7,
                random_state=args.seed,
                n_jobs=args.jobs,
            ).fit(train_x[train_valid], y_train)
            tree_prediction = trees.predict(valid_x[test_valid])
            material_train = np.abs(y_train) >= (
                MATERIAL_EFFECT_FRACTION * scales[horizon, metric]
            )
            if np.unique(material_train).size == 2:
                classifier = ExtraTreesClassifier(
                    n_estimators=args.trees,
                    min_samples_leaf=3,
                    max_features=0.7,
                    class_weight="balanced",
                    random_state=args.seed,
                    n_jobs=args.jobs,
                ).fit(train_x[train_valid], material_train)
                probability = classifier.predict_proba(valid_x[test_valid])[:, 1]
                hurdle_prediction = np.where(
                    probability >= 0.5, tree_prediction, 0.0
                )
            else:
                hurdle_prediction = tree_prediction
            for name, prediction in (
                ("ridge", ridge_prediction),
                ("extra_trees", tree_prediction),
                ("hurdle_extra_trees", hurdle_prediction),
            ):
                rows.append(
                    {
                        "horizon_sec": [120.0, 360.0, 720.0][horizon],
                        "metric": COUNTERFACTUAL_METRIC_NAMES[metric],
                        "model": name,
                        **_metrics(
                            prediction,
                            y_test,
                            float(scales[horizon, metric]),
                        ),
                    }
                )
    result = {
        "status": "development_learnability_audit_training_seed_only",
        "training_episode_ids": train_ids,
        "validation_episode_ids": valid_ids,
        "feature_count": int(train_x.shape[1]),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    terminal = [row for row in rows if row["horizon_sec"] == 720.0]
    print(json.dumps(terminal, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=14100)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=14200)
    parser.add_argument("--jobs", type=int, default=4)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
