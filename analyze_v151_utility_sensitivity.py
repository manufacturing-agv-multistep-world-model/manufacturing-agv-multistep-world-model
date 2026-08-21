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

from compare_paired_vs_absolute_v151 import _exact_two_sided_sign_test
from counterfactual_rollout_v141 import COUNTERFACTUAL_HORIZONS_SEC
from diagnose_counterfactual_ranking_v144 import _predict, _select_device, _summarize_rows
from physics_graph_world_model_absolute_v151 import load_absolute_model_v151
from physics_graph_world_model_counterfactual_v141 import load_counterfactual_model_v141


PROTOCOL = "v151_postconfirmation_fixed_utility_sensitivity_v1"
WEIGHT_SETS = {
    "equal": (1.0, 1.0, 1.0),
    "energy_priority": (2.0, 1.0, 1.0),
    "throughput_priority": (1.0, 2.0, 1.0),
    "queue_priority": (1.0, 1.0, 2.0),
}


def _install_numpy_pickle_compatibility() -> None:
    if "numpy._core" not in sys.modules:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules["numpy._core"] = numpy_core
        sys.modules["numpy._core.multiarray"] = numpy_multiarray
        sys.modules["numpy._core.numeric"] = numpy_numeric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen V15.1 utility sensitivity analysis.")
    parser.add_argument("--paired-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--absolute-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--confirmation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=18599)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _ranking_rows(
    samples: Sequence[Dict[str, np.ndarray]],
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    scale: np.ndarray,
    positive_weights: Sequence[float],
) -> List[Dict[str, float]]:
    terminal = len(COUNTERFACTUAL_HORIZONS_SEC) - 1
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, sample in enumerate(samples):
        if bool(np.all(mask[index, terminal])):
            key = (int(sample["episode_id"]), int(sample["state_id"]))
            groups.setdefault(key, []).append(index)
    signed_weights = np.asarray(
        [-positive_weights[0], positive_weights[1], -positive_weights[2]],
        dtype=np.float64,
    )
    terminal_scale = np.maximum(np.asarray(scale[terminal], dtype=np.float64), 1.0e-9)
    rows = []
    for (episode_id, state_id), indices in groups.items():
        truth = np.sum(target[indices, terminal] / terminal_scale * signed_weights, axis=1)
        estimate = np.sum(prediction[indices, terminal] / terminal_scale * signed_weights, axis=1)
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
                "model_regret": model_regret,
                "baseline_regret": baseline_regret,
                "top1_agreement": float(predicted_best == true_best),
                "random_top1": 1.0 / float(len(truth)),
            }
        )
    return rows


def _episode_differences(
    direct_rows: Sequence[Dict[str, float]], absolute_rows: Sequence[Dict[str, float]]
) -> List[Dict[str, float]]:
    direct = {(int(r["episode_id"]), int(r["state_id"])): r for r in direct_rows}
    absolute = {(int(r["episode_id"]), int(r["state_id"])): r for r in absolute_rows}
    if direct.keys() != absolute.keys():
        raise RuntimeError("Utility variants do not contain identical decision states")
    output = []
    for episode in sorted({key[0] for key in direct}):
        keys = [key for key in direct if key[0] == episode]
        output.append(
            {
                "episode_id": episode,
                "decision_states": len(keys),
                "direct_regret": float(np.mean([direct[key]["model_regret"] for key in keys])),
                "absolute_regret": float(np.mean([absolute[key]["model_regret"] for key in keys])),
                "absolute_minus_direct_regret": float(
                    np.mean(
                        [absolute[key]["model_regret"] - direct[key]["model_regret"] for key in keys]
                    )
                ),
            }
        )
    return output


def _bootstrap(
    direct_rows: Sequence[Dict[str, float]],
    absolute_rows: Sequence[Dict[str, float]],
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    episode_ids = sorted({int(row["episode_id"]) for row in direct_rows})
    direct_by_episode = {
        episode: [row for row in direct_rows if int(row["episode_id"]) == episode]
        for episode in episode_ids
    }
    absolute_by_episode = {
        episode: [row for row in absolute_rows if int(row["episode_id"]) == episode]
        for episode in episode_ids
    }
    rng = np.random.default_rng(seed)
    differences = []
    direct_reductions = []
    for _ in range(replicates):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        direct_selected = [row for episode in sampled for row in direct_by_episode[int(episode)]]
        absolute_selected = [row for episode in sampled for row in absolute_by_episode[int(episode)]]
        direct_summary = _summarize_rows(direct_selected)
        absolute_summary = _summarize_rows(absolute_selected)
        differences.append(
            absolute_summary["model_mean_regret"] - direct_summary["model_mean_regret"]
        )
        direct_reductions.append(direct_summary["regret_reduction"])
    differences = np.asarray(differences, dtype=np.float64)
    reductions = np.asarray(direct_reductions, dtype=np.float64)
    return {
        "absolute_minus_direct_ci_low": float(np.quantile(differences, 0.025)),
        "absolute_minus_direct_ci_high": float(np.quantile(differences, 0.975)),
        "direct_reduction_ci_low": float(np.quantile(reductions, 0.025)),
        "direct_reduction_ci_high": float(np.quantile(reductions, 0.975)),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> Path:
    if len(args.paired_checkpoint) != 3 or len(args.absolute_checkpoint) != 3:
        raise ValueError("Exactly three checkpoints are required for each formulation")
    if not args.confirmation_cache.is_file():
        raise FileNotFoundError(args.confirmation_cache)
    device = _select_device(args.device, args.require_cuda)
    _install_numpy_pickle_compatibility()
    with gzip.open(args.confirmation_cache, "rb") as stream:
        payload = pickle.load(stream)
    samples = payload["samples"]
    target = np.stack([sample["target_delta"] for sample in samples])
    mask = np.stack([sample["target_mask"] for sample in samples])
    paired_models = [load_counterfactual_model_v141(path, device=device) for path in args.paired_checkpoint]
    absolute_models = [load_absolute_model_v151(path, device=device) for path in args.absolute_checkpoint]
    direct_prediction = np.mean(
        np.stack([_predict(model, samples, args.batch_size, device) for model in paired_models]), axis=0
    )
    absolute_prediction = np.mean(
        np.stack([_predict(model, samples, args.batch_size, device) for model in absolute_models]), axis=0
    )
    scale = paired_models[0].counterfactual_scale.detach().cpu().numpy()
    summary_rows = []
    episode_rows = []
    detailed = {}
    for index, (name, weights) in enumerate(WEIGHT_SETS.items()):
        direct_rows = _ranking_rows(samples, direct_prediction, target, mask, scale, weights)
        absolute_rows = _ranking_rows(samples, absolute_prediction, target, mask, scale, weights)
        direct_summary = _summarize_rows(direct_rows)
        absolute_summary = _summarize_rows(absolute_rows)
        episodes = _episode_differences(direct_rows, absolute_rows)
        bootstrap = _bootstrap(
            direct_rows,
            absolute_rows,
            args.bootstrap_replicates,
            args.bootstrap_seed + index,
        )
        sign = _exact_two_sided_sign_test(
            [row["absolute_minus_direct_regret"] for row in episodes]
        )
        difference = absolute_summary["model_mean_regret"] - direct_summary["model_mean_regret"]
        summary_rows.append(
            {
                "weight_set": name,
                "energy_weight": weights[0],
                "throughput_weight": weights[1],
                "queue_weight": weights[2],
                "direct_mean_regret": direct_summary["model_mean_regret"],
                "direct_regret_reduction": direct_summary["regret_reduction"],
                "direct_top1": direct_summary["top1_agreement"],
                "absolute_mean_regret": absolute_summary["model_mean_regret"],
                "absolute_regret_reduction": absolute_summary["regret_reduction"],
                "absolute_top1": absolute_summary["top1_agreement"],
                "absolute_minus_direct_regret": difference,
                "difference_ci_low": bootstrap["absolute_minus_direct_ci_low"],
                "difference_ci_high": bootstrap["absolute_minus_direct_ci_high"],
                "positive_episodes": sign["positive_episodes"],
                "sign_test_p": sign["p_value"],
            }
        )
        for row in episodes:
            episode_rows.append({"weight_set": name, **row})
        detailed[name] = {
            "weights": weights,
            "direct": direct_summary,
            "absolute": absolute_summary,
            "bootstrap": bootstrap,
            "sign_test": sign,
        }
    directional_robustness = all(
        row["absolute_minus_direct_regret"] > 0.0 and row["direct_regret_reduction"] > 0.0
        for row in summary_rows
    )
    audit = {
        "protocol": PROTOCOL,
        "postconfirmation_sensitivity_not_primary_inference": True,
        "confirmation_cache": str(args.confirmation_cache),
        "samples": len(samples),
        "episodes": len({int(sample["episode_id"]) for sample in samples}),
        "weight_sets": WEIGHT_SETS,
        "results": detailed,
        "directional_robustness": directional_robustness,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "utility_sensitivity_summary.csv", summary_rows)
    _write_csv(args.output_dir / "utility_sensitivity_by_episode.csv", episode_rows)
    (args.output_dir / "V151_UTILITY_SENSITIVITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# V15.1 fixed utility-weight sensitivity",
        "",
        "This is a post-confirmation robustness analysis and does not replace the frozen primary test.",
        "",
        "| Weight set | Direct regret | Direct reduction | Direct top-1 | Absolute regret | Absolute - direct | 95% CI | Positive episodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['weight_set']} | {row['direct_mean_regret']:.4f} "
            f"| {row['direct_regret_reduction']:.3f} | {row['direct_top1']:.3f} "
            f"| {row['absolute_mean_regret']:.4f} | {row['absolute_minus_direct_regret']:+.4f} "
            f"| [{row['difference_ci_low']:+.4f}, {row['difference_ci_high']:+.4f}] "
            f"| {int(row['positive_episodes'])}/12 |"
        )
    lines.extend(
        [
            "",
            f"Point-estimate directional robustness across all frozen weights: **{'YES' if directional_robustness else 'NO'}**.",
            "",
            "Statistical significance is not claimed unless the corresponding interval excludes zero.",
        ]
    )
    report = args.output_dir / "V151_UTILITY_SENSITIVITY_AUDIT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return args.output_dir


if __name__ == "__main__":
    analyze(build_parser().parse_args())
