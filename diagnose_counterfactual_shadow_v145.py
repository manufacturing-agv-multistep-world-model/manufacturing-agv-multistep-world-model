from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    CounterfactualCollectionConfig,
    collect_counterfactual_samples_parallel,
    summarize_counterfactual_samples,
)
from diagnose_counterfactual_ranking_v144 import (
    UTILITY_WEIGHTS,
    _install_numpy_pickle_compatibility,
    _predict,
    _select_device,
)
from physics_graph_world_model_counterfactual_v141 import (
    load_counterfactual_model_v141,
)


PROTOCOL = "v145_preregistered_nonacting_shadow_recommender_confirmation_parallel_v2"
FROZEN_UTILITY_MARGIN = 0.15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confirm a frozen unanimous non-acting shadow recommender."
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
    parser.add_argument("--data-seed", type=int, default=15600)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=15699)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--parallel-episodes", type=int, default=1)
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
        "training_and_v144_disjoint": True,
        "model_parameters_frozen": True,
        "recommendations_executed": False,
        "unanimous_model_choice_required": True,
        "frozen_normalized_utility_margin": FROZEN_UTILITY_MARGIN,
        "trajectory_rng_scheme": "seedsequence_base_episode_v145",
    }
    if args.diagnostic_cache.is_file():
        _install_numpy_pickle_compatibility()
        with gzip.open(args.diagnostic_cache, "rb") as stream:
            payload = pickle.load(stream)
        if payload.get("signature") != signature:
            raise ValueError("V14.5 cache does not match the frozen shadow protocol")
        return payload["samples"], "cache"
    samples = collect_counterfactual_samples_parallel(
        config, parallel_episodes=args.parallel_episodes
    )
    args.diagnostic_cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.diagnostic_cache, "wb", compresslevel=4) as stream:
        pickle.dump(
            {"signature": signature, "samples": samples},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return samples, "collected"


def _shadow_rows(
    samples: Sequence[Dict[str, np.ndarray]],
    predictions: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    scales: np.ndarray,
) -> List[Dict[str, float]]:
    terminal = len(COUNTERFACTUAL_HORIZONS_SEC) - 1
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, sample in enumerate(samples):
        if bool(np.all(mask[index, terminal])):
            key = (int(sample["episode_id"]), int(sample["state_id"]))
            groups.setdefault(key, []).append(index)

    scale = np.maximum(np.mean(scales, axis=0)[terminal], 1.0e-9)
    rows: List[Dict[str, float]] = []
    for (episode_id, state_id), indices in groups.items():
        model_utilities = np.sum(
            predictions[:, indices, terminal] / scale * UTILITY_WEIGHTS,
            axis=2,
        )
        choices = np.argmax(
            np.concatenate(
                [np.zeros((model_utilities.shape[0], 1)), model_utilities],
                axis=1,
            ),
            axis=1,
        )
        unanimous = bool(np.all(choices == choices[0]))
        selected = int(choices[0]) - 1 if unanimous and choices[0] > 0 else -1
        predicted_gain = (
            float(np.mean(model_utilities[:, selected])) if selected >= 0 else 0.0
        )
        recommended = bool(
            unanimous
            and selected >= 0
            and predicted_gain >= FROZEN_UTILITY_MARGIN
        )
        true_gain = (
            float(
                np.sum(
                    target[indices[selected], terminal] / scale * UTILITY_WEIGHTS
                )
            )
            if recommended
            else 0.0
        )
        rows.append(
            {
                "episode_id": float(episode_id),
                "state_id": float(state_id),
                "candidate_count": float(len(indices) + 1),
                "unanimous": float(unanimous),
                "recommended": float(recommended),
                "predicted_gain": predicted_gain,
                "true_gain": true_gain,
                "beneficial": float(recommended and true_gain > 0.0),
            }
        )
    return rows


def _summary(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    recommendations = [row for row in rows if row["recommended"] > 0.5]
    gains = np.asarray([row["true_gain"] for row in recommendations])
    return {
        "eligible_states": float(len(rows)),
        "recommendations": float(len(recommendations)),
        "coverage": float(len(recommendations) / len(rows)) if rows else 0.0,
        "benefit_precision": float(np.mean(gains > 0.0)) if gains.size else float("nan"),
        "false_recommendation_rate": (
            float(np.mean(gains <= 0.0)) if gains.size else float("nan")
        ),
        "mean_true_gain": float(np.mean(gains)) if gains.size else float("nan"),
        "median_true_gain": float(np.median(gains)) if gains.size else float("nan"),
        "mean_predicted_gain": (
            float(np.mean([row["predicted_gain"] for row in recommendations]))
            if recommendations
            else float("nan")
        ),
    }


def _episode_summary(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    result = []
    for episode in sorted({int(row["episode_id"]) for row in rows}):
        subset = [row for row in rows if int(row["episode_id"]) == episode]
        result.append({"episode_id": episode, **_summary(subset)})
    return result


def _trajectory_bootstrap(
    rows: Sequence[Dict[str, float]], replicates: int, seed: int
) -> Dict[str, float]:
    episodes = sorted({int(row["episode_id"]) for row in rows})
    grouped = {
        episode: [row for row in rows if int(row["episode_id"]) == episode]
        for episode in episodes
    }
    rng = np.random.default_rng(seed)
    gains = []
    precisions = []
    coverages = []
    for _ in range(replicates):
        sampled = rng.choice(episodes, size=len(episodes), replace=True)
        selected = [row for episode in sampled for row in grouped[int(episode)]]
        summary = _summary(selected)
        gains.append(summary["mean_true_gain"])
        precisions.append(summary["benefit_precision"])
        coverages.append(summary["coverage"])
    gain_values = np.asarray(gains, dtype=np.float64)
    precision_values = np.asarray(precisions, dtype=np.float64)
    coverage_values = np.asarray(coverages, dtype=np.float64)
    return {
        "replicates": float(replicates),
        "mean_gain_ci95": np.nanquantile(gain_values, [0.025, 0.975]).tolist(),
        "precision_ci95": np.nanquantile(
            precision_values, [0.025, 0.975]
        ).tolist(),
        "coverage_ci95": np.nanquantile(
            coverage_values, [0.025, 0.975]
        ).tolist(),
        "p_mean_gain_le_zero": float(np.nanmean(gain_values <= 0.0)),
    }


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    summary = audit["shadow_summary"]
    bootstrap = audit["trajectory_bootstrap"]
    lines = [
        "# V14.5 preregistered non-acting shadow-recommender confirmation",
        "",
        (
            f"Untouched data seed: {audit['data_seed']}; trajectories: "
            f"{audit['episodes']}; paired candidate samples: {audit['samples']}."
        ),
        "",
        "The baseline DT-aware action was always executed. Recommendations were logged "
        "only when all three frozen models selected the same non-baseline candidate and "
        f"the ensemble normalized utility margin was at least {FROZEN_UTILITY_MARGIN:.2f}.",
        "",
        f"- Eligible decision states: {int(summary['eligible_states'])}",
        f"- Shadow recommendations: {int(summary['recommendations'])}",
        f"- Recommendation coverage: {summary['coverage']:.3f}",
        f"- Beneficial-recommendation precision: {summary['benefit_precision']:.3f}",
        f"- Mean true normalized gain: {summary['mean_true_gain']:.4f}",
        f"- Median true normalized gain: {summary['median_true_gain']:.4f}",
        (
            "- Trajectory-bootstrap mean-gain 95% CI: "
            f"[{bootstrap['mean_gain_ci95'][0]:.4f}, "
            f"{bootstrap['mean_gain_ci95'][1]:.4f}]"
        ),
        "",
        "## Trajectory-level stability",
        "",
        "| Episode | Recommendations | Coverage | Precision | Mean true gain |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in audit["episode_results"]:
        lines.append(
            f"| {row['episode_id']} | {int(row['recommendations'])} "
            f"| {row['coverage']:.3f} | {row['benefit_precision']:.3f} "
            f"| {row['mean_true_gain']:+.4f} |"
        )
    lines.extend(["", "## Preregistered continuation criteria", ""])
    lines.extend(
        f"- [{'x' if item['passed'] else ' '}] {item['criterion']}"
        for item in audit["criteria"]
    )
    lines.extend(
        [
            "",
            "Proceed to a separately preregistered guarded closed-loop pilot: "
            f"**{'YES' if audit['passed'] else 'NO'}**.",
            "",
            "Passing establishes shadow-recommendation reliability only. No recommendation "
            "was executed, so this is not a closed-loop system-performance claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose(args: argparse.Namespace) -> Path:
    if len(args.checkpoint) != 3:
        raise ValueError("V14.5 requires exactly three frozen model seeds")
    if args.episodes != 12 or args.data_seed != 15600:
        raise ValueError("V14.5 formal confirmation is frozen at 12 episodes and seed 15600")
    if args.bootstrap_replicates < 2000:
        raise ValueError("V14.5 requires at least 2000 trajectory-bootstrap replicates")
    if not 1 <= args.parallel_episodes <= args.episodes:
        raise ValueError("Parallel episodes must be between 1 and the episode count")
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
    rows = _shadow_rows(
        samples,
        np.stack(predictions),
        target,
        mask,
        np.stack(scales),
    )
    summary = _summary(rows)
    episode_results = _episode_summary(rows)
    bootstrap = _trajectory_bootstrap(
        rows, args.bootstrap_replicates, args.bootstrap_seed
    )
    dataset_summary = summarize_counterfactual_samples(samples)
    positive_episodes = sum(
        row["recommendations"] > 0 and row["mean_true_gain"] > 0.0
        for row in episode_results
    )
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
            "criterion": "At least 350 complete terminal decision states are eligible",
            "passed": summary["eligible_states"] >= 350,
        },
        {
            "criterion": "Every target component has at least 85% valid physical-time coverage",
            "passed": bool(
                np.all(np.asarray(dataset_summary["valid_target_rate"]) >= 0.85)
            ),
        },
        {
            "criterion": "At least 80 shadow recommendations are issued",
            "passed": summary["recommendations"] >= 80,
        },
        {
            "criterion": "Recommendation coverage is between 15% and 45%",
            "passed": 0.15 <= summary["coverage"] <= 0.45,
        },
        {
            "criterion": "At least 70% of recommendations have positive true utility",
            "passed": summary["benefit_precision"] >= 0.70,
        },
        {
            "criterion": "Mean true normalized recommendation gain is positive",
            "passed": summary["mean_true_gain"] > 0.0,
        },
        {
            "criterion": "Trajectory-bootstrap 95% mean-gain interval is above zero",
            "passed": bootstrap["mean_gain_ci95"][0] > 0.0,
        },
        {
            "criterion": "At least 9 of 12 trajectories have positive mean recommendation gain",
            "passed": positive_episodes >= 9,
        },
    ]
    audit: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "preregistered_nonacting_shadow_confirmation",
        "data_seed": args.data_seed,
        "episodes": args.episodes,
        "samples": len(samples),
        "cache_source": cache_source,
        "checkpoints": [str(path) for path in args.checkpoint],
        "recommendations_executed": False,
        "unanimous_model_choice_required": True,
        "frozen_normalized_utility_margin": FROZEN_UTILITY_MARGIN,
        "dataset_summary": dataset_summary,
        "shadow_summary": summary,
        "episode_results": episode_results,
        "trajectory_bootstrap": bootstrap,
        "positive_episode_count": positive_episodes,
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
    }
    json_path = args.output_dir / "V145_SHADOW_CONFIRMATION_AUDIT.json"
    json_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown_path = args.output_dir / "V145_SHADOW_CONFIRMATION_AUDIT.md"
    _write_markdown(markdown_path, audit)
    print(markdown_path.read_text(encoding="utf-8"))
    return args.output_dir


if __name__ == "__main__":
    diagnose(build_parser().parse_args())
