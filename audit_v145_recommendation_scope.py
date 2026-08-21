from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from counterfactual_rollout_v141 import COUNTERFACTUAL_HORIZONS_SEC
from diagnose_counterfactual_ranking_v144 import (
    UTILITY_WEIGHTS,
    _install_numpy_pickle_compatibility,
    _predict,
    _select_device,
)
from diagnose_counterfactual_shadow_v145 import FROZEN_UTILITY_MARGIN
from physics_graph_world_model_counterfactual_v141 import load_counterfactual_model_v141


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit V14.5 recommendation joint-action scope.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(args: argparse.Namespace) -> None:
    if len(args.checkpoint) != 3:
        raise ValueError("Exactly three frozen V14.1 checkpoints are required")
    _install_numpy_pickle_compatibility()
    with gzip.open(args.cache, "rb") as stream:
        samples = pickle.load(stream)["samples"]
    device = _select_device(args.device, False)
    predictions = []
    scales = []
    for checkpoint in args.checkpoint:
        model = load_counterfactual_model_v141(checkpoint, device=device)
        predictions.append(_predict(model, samples, args.batch_size, device))
        scales.append(model.counterfactual_scale.detach().cpu().numpy())
    prediction = np.stack(predictions)
    target = np.stack([sample["target_delta"] for sample in samples])
    mask = np.stack([sample["target_mask"] for sample in samples]) > 0.0
    terminal = len(COUNTERFACTUAL_HORIZONS_SEC) - 1
    scale = np.maximum(np.mean(np.stack(scales), axis=0)[terminal], 1.0e-9)
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, sample in enumerate(samples):
        if bool(np.all(mask[index, terminal])):
            key = (int(sample["episode_id"]), int(sample["state_id"]))
            groups.setdefault(key, []).append(index)

    rows = []
    for indices in groups.values():
        utilities = np.sum(
            prediction[:, indices, terminal] / scale * UTILITY_WEIGHTS,
            axis=2,
        )
        choices = np.argmax(
            np.concatenate([np.zeros((3, 1)), utilities], axis=1), axis=1
        )
        if not np.all(choices == choices[0]) or choices[0] == 0:
            continue
        selected = int(choices[0]) - 1
        gain = float(np.mean(utilities[:, selected]))
        if gain < FROZEN_UTILITY_MARGIN:
            continue
        index = indices[selected]
        true_gain = float(np.sum(target[index, terminal] / scale * UTILITY_WEIGHTS))
        rows.append(
            {
                "hamming_distance": int(samples[index]["action_hamming_distance"]),
                "deadlock_active": bool(samples[index]["global_features"][3] > 0.5),
                "all_wait_action": bool(
                    np.all(np.asarray(samples[index]["candidate_actions"]) == 0)
                ),
                "maximum_wait_fraction": float(
                    np.max(samples[index]["agent_features"][:, 5])
                ),
                "predicted_gain": gain,
                "minimum_member_predicted_gain": float(
                    np.min(utilities[:, selected])
                ),
                "true_gain": true_gain,
                "beneficial": bool(true_gain > 0.0),
            }
        )
    distribution = {
        str(distance): {
            "recommendations": sum(row["hamming_distance"] == distance for row in rows),
            "beneficial": sum(
                row["hamming_distance"] == distance and row["beneficial"] for row in rows
            ),
        }
        for distance in sorted({row["hamming_distance"] for row in rows})
    }
    threshold_audit = {}
    for threshold in (0.15, 0.20, 0.25, 0.30, 0.40):
        selected = [row for row in rows if row["predicted_gain"] >= threshold]
        threshold_audit[f"{threshold:.2f}"] = {
            "recommendations": len(selected),
            "beneficial_precision": (
                float(np.mean([row["beneficial"] for row in selected]))
                if selected
                else None
            ),
            "mean_true_gain": (
                float(np.mean([row["true_gain"] for row in selected]))
                if selected
                else None
            ),
        }
    result = {
        "source": "frozen_v145_shadow_confirmation_cache",
        "recommendations": len(rows),
        "frozen_utility_margin": FROZEN_UTILITY_MARGIN,
        "joint_action_hamming_distribution": distribution,
        "single_agv_recommendation_fraction": (
            sum(row["hamming_distance"] == 1 for row in rows) / len(rows) if rows else 0.0
        ),
        "single_agv_beneficial_precision": (
            np.mean(
                [row["beneficial"] for row in rows if row["hamming_distance"] == 1]
            )
            if any(row["hamming_distance"] == 1 for row in rows)
            else None
        ),
        "deadlock_active_recommendations": sum(row["deadlock_active"] for row in rows),
        "deadlock_inactive_recommendations": sum(not row["deadlock_active"] for row in rows),
        "deadlock_inactive_beneficial_precision": (
            np.mean([row["beneficial"] for row in rows if not row["deadlock_active"]])
            if any(not row["deadlock_active"] for row in rows)
            else None
        ),
        "zero_wait_recommendations": sum(
            row["maximum_wait_fraction"] <= 1.0e-9 for row in rows
        ),
        "all_wait_recommendations": sum(row["all_wait_action"] for row in rows),
        "all_wait_beneficial_precision": (
            np.mean([row["beneficial"] for row in rows if row["all_wait_action"]])
            if any(row["all_wait_action"] for row in rows)
            else None
        ),
        "deadlock_inactive_non_all_wait_recommendations": sum(
            not row["deadlock_active"] and not row["all_wait_action"] for row in rows
        ),
        "deadlock_inactive_non_all_wait_beneficial_precision": (
            np.mean(
                [
                    row["beneficial"]
                    for row in rows
                    if not row["deadlock_active"] and not row["all_wait_action"]
                ]
            )
            if any(not row["deadlock_active"] and not row["all_wait_action"] for row in rows)
            else None
        ),
        "execution_threshold_audit": threshold_audit,
        "ensemble_lower_bound_audit": {
            "minimum_member_margin": FROZEN_UTILITY_MARGIN,
            "recommendations": sum(
                row["minimum_member_predicted_gain"] >= FROZEN_UTILITY_MARGIN
                for row in rows
            ),
            "beneficial_precision": (
                float(
                    np.mean(
                        [
                            row["beneficial"]
                            for row in rows
                            if row["minimum_member_predicted_gain"]
                            >= FROZEN_UTILITY_MARGIN
                        ]
                    )
                )
                if any(
                    row["minimum_member_predicted_gain"] >= FROZEN_UTILITY_MARGIN
                    for row in rows
                )
                else None
            ),
            "mean_true_gain": (
                float(
                    np.mean(
                        [
                            row["true_gain"]
                            for row in rows
                            if row["minimum_member_predicted_gain"]
                            >= FROZEN_UTILITY_MARGIN
                        ]
                    )
                )
                if any(
                    row["minimum_member_predicted_gain"] >= FROZEN_UTILITY_MARGIN
                    for row in rows
                )
                else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
