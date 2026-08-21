from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    CounterfactualCollectionConfig,
    collect_counterfactual_samples_parallel,
    summarize_counterfactual_samples,
)
from diagnose_counterfactual_ranking_v144 import (
    _predict,
    _ranking_rows,
    _select_device,
    _summarize_rows,
)
from physics_graph_world_model_absolute_v151 import load_absolute_model_v151
from physics_graph_world_model_counterfactual_v141 import (
    load_counterfactual_model_v141,
)


PROTOCOL = "v151_preregistered_paired_delta_vs_absolute_outcome_v1"
FROZEN_SETTINGS = {
    "episodes": 12,
    "behavior_steps": 4000,
    "warmup_steps": 1200,
    "sample_stride": 80,
    "candidates_per_state": 3,
    "max_rollout_steps": 500,
    "data_seed": 18400,
    "bootstrap_replicates": 5000,
    "bootstrap_seed": 18499,
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
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct paired-delta supervision with absolute-outcome "
            "prediction followed by subtraction."
        )
    )
    parser.add_argument("--paired-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--absolute-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--parallel-episodes", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _load_or_collect(
    cache: Path, parallel_episodes: int,
) -> Tuple[List[Dict[str, np.ndarray]], str, Dict[str, Any]]:
    settings = FROZEN_SETTINGS
    config = CounterfactualCollectionConfig(
        episodes=settings["episodes"],
        behavior_steps=settings["behavior_steps"],
        warmup_steps=settings["warmup_steps"],
        sample_stride=settings["sample_stride"],
        candidates_per_state=settings["candidates_per_state"],
        horizons_sec=COUNTERFACTUAL_HORIZONS_SEC,
        max_rollout_steps=settings["max_rollout_steps"],
        seed=settings["data_seed"],
    )
    signature = {
        "schema": PROTOCOL,
        "config": config.__dict__,
        "training_data_disjoint": True,
        "model_parameters_frozen": True,
        "primary_endpoint": "paired_terminal_ranking_regret_difference",
        "comparison": "direct_delta_supervision_vs_absolute_then_difference",
    }
    if cache.is_file():
        _install_numpy_pickle_compatibility()
        with gzip.open(cache, "rb") as stream:
            payload = pickle.load(stream)
        if payload.get("signature") != signature:
            raise ValueError("V15.1 diagnostic cache does not match the frozen protocol")
        return payload["samples"], "cache", signature
    samples = collect_counterfactual_samples_parallel(
        config,
        parallel_episodes=max(1, min(int(parallel_episodes), config.episodes)),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wb", compresslevel=4) as stream:
        pickle.dump(
            {"signature": signature, "samples": samples},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return samples, "collected", signature


def _paired_rows(
    direct_rows: Sequence[Dict[str, float]],
    absolute_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    direct_by_key = {
        (int(row["episode_id"]), int(row["state_id"])): row
        for row in direct_rows
    }
    absolute_by_key = {
        (int(row["episode_id"]), int(row["state_id"])): row
        for row in absolute_rows
    }
    if direct_by_key.keys() != absolute_by_key.keys():
        raise RuntimeError("The formulations did not rank identical decision states")
    output = []
    for episode_id, state_id in sorted(direct_by_key):
        direct = direct_by_key[(episode_id, state_id)]
        absolute = absolute_by_key[(episode_id, state_id)]
        output.append(
            {
                "episode_id": float(episode_id),
                "state_id": float(state_id),
                "direct_regret": direct["model_regret"],
                "absolute_regret": absolute["model_regret"],
                "absolute_minus_direct_regret": (
                    absolute["model_regret"] - direct["model_regret"]
                ),
                "direct_top1": direct["top1_agreement"],
                "absolute_top1": absolute["top1_agreement"],
                "baseline_regret": direct["baseline_regret"],
            }
        )
    return output


def _episode_comparison(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    output = []
    for episode_id in sorted({int(row["episode_id"]) for row in rows}):
        subset = [row for row in rows if int(row["episode_id"]) == episode_id]
        output.append(
            {
                "episode_id": episode_id,
                "decision_states": len(subset),
                "direct_mean_regret": float(
                    np.mean([row["direct_regret"] for row in subset])
                ),
                "absolute_mean_regret": float(
                    np.mean([row["absolute_regret"] for row in subset])
                ),
                "absolute_minus_direct_regret": float(
                    np.mean(
                        [row["absolute_minus_direct_regret"] for row in subset]
                    )
                ),
                "direct_top1": float(
                    np.mean([row["direct_top1"] for row in subset])
                ),
                "absolute_top1": float(
                    np.mean([row["absolute_top1"] for row in subset])
                ),
            }
        )
    return output


def _paired_bootstrap(
    rows: Sequence[Dict[str, float]], replicates: int, seed: int
) -> Dict[str, float]:
    episode_ids = sorted({int(row["episode_id"]) for row in rows})
    by_episode = {
        episode: [row for row in rows if int(row["episode_id"]) == episode]
        for episode in episode_ids
    }
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        selected = [row for episode in sampled for row in by_episode[int(episode)]]
        differences.append(
            float(
                np.mean(
                    [row["absolute_minus_direct_regret"] for row in selected]
                )
            )
        )
    values = np.asarray(differences, dtype=np.float64)
    return {
        "replicates": replicates,
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_absolute_minus_direct_le_zero": float(np.mean(values <= 0.0)),
    }


def _exact_two_sided_sign_test(differences: Sequence[float]) -> Dict[str, float]:
    nonzero = [float(value) for value in differences if abs(float(value)) > 1.0e-12]
    positives = sum(value > 0.0 for value in nonzero)
    n = len(nonzero)
    if n == 0:
        return {"nonzero_episodes": 0, "positive_episodes": 0, "p_value": 1.0}
    extreme = max(positives, n - positives)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / (2.0**n)
    return {
        "nonzero_episodes": n,
        "positive_episodes": positives,
        "p_value": min(1.0, 2.0 * tail),
    }


def _write_rows(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    direct = audit["direct_delta_ranking"]
    absolute = audit["absolute_outcome_ranking"]
    paired = audit["paired_comparison"]
    bootstrap = audit["paired_episode_bootstrap"]
    sign = audit["exact_episode_sign_test"]
    lines = [
        "# V15.1 paired-formulation confirmation",
        "",
        (
            f"Fresh data seed: {audit['data_seed']}; complete trajectories: "
            f"{audit['episodes']}; paired candidates: {audit['samples']}."
        ),
        "",
        "Both methods use the same frozen V13 physics-graph backbone, identical "
        "56,457-parameter action-value heads, the same training pairs and the same "
        "three initialization seeds. The only scientific change is the target: direct "
        "paired effects versus two absolute branch outcomes followed by subtraction.",
        "",
        "| Formulation | Mean regret | Regret reduction vs unchanged action | Top-1 |",
        "|---|---:|---:|---:|",
        (
            f"| Direct paired-effect supervision | {direct['model_mean_regret']:.5f} "
            f"| {direct['regret_reduction']:.3f} | {direct['top1_agreement']:.3f} |"
        ),
        (
            f"| Absolute outcomes then difference | {absolute['model_mean_regret']:.5f} "
            f"| {absolute['regret_reduction']:.3f} | {absolute['top1_agreement']:.3f} |"
        ),
        (
            f"| Unchanged DT-aware action | {direct['baseline_mean_regret']:.5f} "
            "| 0.000 | - |"
        ),
        "",
        (
            "Mean paired absolute-minus-direct regret: "
            f"**{paired['mean_difference']:+.5f}**."
        ),
        (
            f"Trajectory-bootstrap 95% CI: [{bootstrap['ci_low']:+.5f}, "
            f"{bootstrap['ci_high']:+.5f}]."
        ),
        (
            f"Exact two-sided episode sign test: {sign['positive_episodes']}/"
            f"{sign['nonzero_episodes']} nonzero trajectories favor direct paired "
            f"supervision, p={sign['p_value']:.6f}."
        ),
        "",
        "## Trajectory-level comparison",
        "",
        "| Episode | States | Direct regret | Absolute regret | Absolute - direct | Direct top-1 | Absolute top-1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["episode_results"]:
        lines.append(
            f"| {row['episode_id']} | {row['decision_states']} "
            f"| {row['direct_mean_regret']:.5f} "
            f"| {row['absolute_mean_regret']:.5f} "
            f"| {row['absolute_minus_direct_regret']:+.5f} "
            f"| {row['direct_top1']:.3f} | {row['absolute_top1']:.3f} |"
        )
    lines.extend(["", "## Protocol integrity", ""])
    lines.extend(
        f"- [{'x' if item['passed'] else ' '}] {item['criterion']}"
        for item in audit["integrity_criteria"]
    )
    lines.extend(["", "## Frozen scientific-support criteria", ""])
    lines.extend(
        f"- [{'x' if item['passed'] else ' '}] {item['criterion']}"
        for item in audit["scientific_criteria"]
    )
    lines.extend(
        [
            "",
            f"Protocol integrity: **{'PASS' if audit['integrity_passed'] else 'FAIL'}**.",
            (
                "Evidence supports the paired-effect formulation contribution: "
                f"**{'YES' if audit['scientific_support'] else 'NO'}**."
            ),
            "",
            "A negative frozen result remains reportable and must not trigger seed, "
            "weight or threshold replacement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(args: argparse.Namespace) -> Path:
    if len(args.paired_checkpoint) != 3 or len(args.absolute_checkpoint) != 3:
        raise ValueError("Exactly three checkpoints are required for each formulation")
    device = _select_device(args.device, args.require_cuda)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.parallel_episodes < 1:
        raise ValueError("parallel_episodes must be positive")
    samples, cache_source, signature = _load_or_collect(
        args.diagnostic_cache, args.parallel_episodes
    )
    target = np.stack([sample["target_delta"] for sample in samples])
    mask = np.stack([sample["target_mask"] for sample in samples])

    paired_models = [
        load_counterfactual_model_v141(path, device=device)
        for path in args.paired_checkpoint
    ]
    absolute_models = [
        load_absolute_model_v151(path, device=device)
        for path in args.absolute_checkpoint
    ]
    paired_predictions = np.stack(
        [_predict(model, samples, args.batch_size, device) for model in paired_models]
    )
    absolute_predictions = np.stack(
        [_predict(model, samples, args.batch_size, device) for model in absolute_models]
    )
    paired_prediction = np.mean(paired_predictions, axis=0)
    absolute_prediction = np.mean(absolute_predictions, axis=0)
    evaluation_scale = paired_models[0].counterfactual_scale.detach().cpu().numpy()
    if not all(
        np.allclose(
            evaluation_scale,
            model.counterfactual_scale.detach().cpu().numpy(),
        )
        for model in paired_models[1:]
    ):
        raise RuntimeError("Paired-model training scales differ across seeds")

    direct_rows = _ranking_rows(
        samples, paired_prediction, target, mask, evaluation_scale
    )
    absolute_rows = _ranking_rows(
        samples, absolute_prediction, target, mask, evaluation_scale
    )
    paired_rows = _paired_rows(direct_rows, absolute_rows)
    episode_results = _episode_comparison(paired_rows)
    bootstrap = _paired_bootstrap(
        paired_rows,
        FROZEN_SETTINGS["bootstrap_replicates"],
        FROZEN_SETTINGS["bootstrap_seed"],
    )
    sign_test = _exact_two_sided_sign_test(
        [row["absolute_minus_direct_regret"] for row in episode_results]
    )
    direct_summary = _summarize_rows(direct_rows)
    absolute_summary = _summarize_rows(absolute_rows)
    direct_head_parameters = [
        sum(parameter.numel() for parameter in model.counterfactual_value_head.parameters())
        for model in paired_models
    ]
    absolute_head_parameters = [
        sum(parameter.numel() for parameter in model.counterfactual_value_head.parameters())
        for model in absolute_models
    ]
    mean_difference = float(
        np.mean([row["absolute_minus_direct_regret"] for row in paired_rows])
    )
    positive_episodes = sum(
        row["absolute_minus_direct_regret"] > 0.0 for row in episode_results
    )
    integrity_criteria = [
        {
            "criterion": "The frozen 12 complete confirmation trajectories are present",
            "passed": len({int(row["episode_id"]) for row in paired_rows}) == 12,
        },
        {
            "criterion": "Both methods rank identical state-action candidates",
            "passed": len(direct_rows) == len(absolute_rows) == len(paired_rows),
        },
        {
            "criterion": "All six heads have exactly 56,457 trainable-stage parameters",
            "passed": all(
                count == 56_457
                for count in direct_head_parameters + absolute_head_parameters
            ),
        },
        {
            "criterion": "The evaluation utility uses one common paired-training scale",
            "passed": bool(np.isfinite(evaluation_scale).all())
            and bool(np.all(evaluation_scale > 0.0)),
        },
        {
            "criterion": "All paired regret values are finite",
            "passed": all(
                np.isfinite(row["absolute_minus_direct_regret"])
                for row in paired_rows
            ),
        },
    ]
    scientific_criteria = [
        {
            "criterion": "Direct paired supervision has lower mean ranking regret",
            "passed": mean_difference > 0.0,
        },
        {
            "criterion": "The trajectory-bootstrap 95% interval is above zero",
            "passed": bootstrap["ci_low"] > 0.0,
        },
        {
            "criterion": "At least 9 of 12 trajectories favor direct paired supervision",
            "passed": positive_episodes >= 9,
        },
        {
            "criterion": "Direct paired top-1 agreement is not lower",
            "passed": direct_summary["top1_agreement"]
            >= absolute_summary["top1_agreement"],
        },
    ]
    audit = {
        "protocol": PROTOCOL,
        "data_seed": FROZEN_SETTINGS["data_seed"],
        "episodes": FROZEN_SETTINGS["episodes"],
        "samples": len(samples),
        "cache_source": cache_source,
        "cache_signature": signature,
        "device": device,
        "paired_checkpoints": [str(path) for path in args.paired_checkpoint],
        "absolute_checkpoints": [str(path) for path in args.absolute_checkpoint],
        "paired_head_parameters": direct_head_parameters,
        "absolute_head_parameters": absolute_head_parameters,
        "common_evaluation_scale": evaluation_scale.tolist(),
        "dataset_summary": summarize_counterfactual_samples(samples),
        "direct_delta_ranking": direct_summary,
        "absolute_outcome_ranking": absolute_summary,
        "paired_comparison": {
            "mean_difference": mean_difference,
            "positive_episodes": positive_episodes,
        },
        "paired_episode_bootstrap": bootstrap,
        "exact_episode_sign_test": sign_test,
        "episode_results": episode_results,
        "integrity_criteria": integrity_criteria,
        "scientific_criteria": scientific_criteria,
        "integrity_passed": all(item["passed"] for item in integrity_criteria),
        "scientific_support": all(item["passed"] for item in scientific_criteria),
    }
    (args.output_dir / "V151_PAIRED_FORMULATION_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_rows(args.output_dir / "paired_state_ranking_rows.csv", paired_rows)
    _write_rows(args.output_dir / "episode_comparison.csv", episode_results)
    _write_markdown(
        args.output_dir / "V151_PAIRED_FORMULATION_AUDIT.md", audit
    )
    if not audit["integrity_passed"]:
        raise RuntimeError("V15.1 protocol-integrity checks failed")
    print(
        (args.output_dir / "V151_PAIRED_FORMULATION_AUDIT.md").read_text(
            encoding="utf-8"
        )
    )
    return args.output_dir


if __name__ == "__main__":
    compare(build_parser().parse_args())
