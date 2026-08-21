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

from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    CounterfactualCollectionConfig,
    collect_counterfactual_samples,
    summarize_counterfactual_samples,
)
from diagnose_counterfactual_ranking_v144 import (
    _predict,
    _ranking_rows,
    _select_device,
    _summarize_rows,
)
from flat_counterfactual_baseline_v150 import (
    EXPECTED_TRAINABLE_PARAMETERS,
    load_flat_counterfactual_baseline_v150,
)
from physics_graph_world_model_counterfactual_v141 import (
    load_counterfactual_model_v141,
)


PROTOCOL = "v150_frozen_paired_graph_vs_flat_ranking_v1"
FROZEN_SETTINGS = {
    "development": {
        "episodes": 3,
        "behavior_steps": 1800,
        "warmup_steps": 600,
        "sample_stride": 120,
        "candidates_per_state": 3,
        "max_rollout_steps": 500,
        "data_seed": 16400,
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 16499,
    },
    "confirmation": {
        "episodes": 12,
        "behavior_steps": 4000,
        "warmup_steps": 1200,
        "sample_stride": 80,
        "candidates_per_state": 3,
        "max_rollout_steps": 500,
        "data_seed": 17400,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 17499,
    },
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
        description="Compare frozen V14.1 and parameter-matched flat ensembles."
    )
    parser.add_argument("--phase", choices=tuple(FROZEN_SETTINGS), required=True)
    parser.add_argument("--graph-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--flat-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _load_or_collect(
    phase: str, cache: Path
) -> Tuple[List[Dict[str, np.ndarray]], str, Dict[str, Any]]:
    settings = FROZEN_SETTINGS[phase]
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
        "phase": phase,
        "config": config.__dict__,
        "training_data_disjoint": True,
        "model_parameters_frozen": True,
        "primary_endpoint": "paired_terminal_ranking_regret_difference",
    }
    if cache.is_file():
        _install_numpy_pickle_compatibility()
        with gzip.open(cache, "rb") as stream:
            payload = pickle.load(stream)
        if payload.get("signature") != signature:
            raise ValueError("V15.0 diagnostic cache does not match the frozen protocol")
        return payload["samples"], "cache", signature
    samples = collect_counterfactual_samples(config)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wb", compresslevel=4) as stream:
        pickle.dump(
            {"signature": signature, "samples": samples},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return samples, "collected", signature


def _paired_rows(
    graph_rows: Sequence[Dict[str, float]],
    flat_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    graph_by_key = {
        (int(row["episode_id"]), int(row["state_id"])): row for row in graph_rows
    }
    flat_by_key = {
        (int(row["episode_id"]), int(row["state_id"])): row for row in flat_rows
    }
    if graph_by_key.keys() != flat_by_key.keys():
        raise RuntimeError("Graph and flat models did not rank identical decision states")
    output = []
    for episode_id, state_id in sorted(graph_by_key):
        graph = graph_by_key[(episode_id, state_id)]
        flat = flat_by_key[(episode_id, state_id)]
        output.append(
            {
                "episode_id": float(episode_id),
                "state_id": float(state_id),
                "graph_regret": graph["model_regret"],
                "flat_regret": flat["model_regret"],
                "flat_minus_graph_regret": (
                    flat["model_regret"] - graph["model_regret"]
                ),
                "graph_top1": graph["top1_agreement"],
                "flat_top1": flat["top1_agreement"],
                "baseline_regret": graph["baseline_regret"],
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
                "graph_mean_regret": float(np.mean([r["graph_regret"] for r in subset])),
                "flat_mean_regret": float(np.mean([r["flat_regret"] for r in subset])),
                "flat_minus_graph_regret": float(
                    np.mean([r["flat_minus_graph_regret"] for r in subset])
                ),
                "graph_top1": float(np.mean([r["graph_top1"] for r in subset])),
                "flat_top1": float(np.mean([r["flat_top1"] for r in subset])),
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
            float(np.mean([row["flat_minus_graph_regret"] for row in selected]))
        )
    values = np.asarray(differences, dtype=np.float64)
    return {
        "replicates": replicates,
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_flat_minus_graph_le_zero": float(np.mean(values <= 0.0)),
    }


def _write_rows(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    graph = audit["graph_ranking"]
    flat = audit["flat_ranking"]
    paired = audit["paired_comparison"]
    bootstrap = audit["paired_episode_bootstrap"]
    lines = [
        "# V15.0 parameter-budget-matched architecture comparison",
        "",
        f"Phase: **{audit['phase']}**; data seed: {audit['data_seed']}; "
        f"complete trajectories: {audit['episodes']}; paired candidates: {audit['samples']}.",
        "",
        "Both learned methods use three-seed ensembles and the same paired candidate states. "
        "The flat MLP has exactly the same 56,457 trainable parameters as the V14.1 "
        "counterfactual head but receives no adjacency matrix or static physical features.",
        "",
        "| Method | Mean regret | Regret reduction vs unchanged action | Top-1 |",
        "|---|---:|---:|---:|",
        f"| Full V14.1 physics-graph model | {graph['model_mean_regret']:.5f} "
        f"| {graph['regret_reduction']:.3f} | {graph['top1_agreement']:.3f} |",
        f"| Parameter-matched flat MLP | {flat['model_mean_regret']:.5f} "
        f"| {flat['regret_reduction']:.3f} | {flat['top1_agreement']:.3f} |",
        f"| Unchanged DT-aware action | {graph['baseline_mean_regret']:.5f} | 0.000 | - |",
        "",
        f"Mean paired flat-minus-graph regret: **{paired['mean_difference']:+.5f}**.",
        f"Trajectory-bootstrap 95% CI: [{bootstrap['ci_low']:+.5f}, "
        f"{bootstrap['ci_high']:+.5f}].",
        "",
        "## Trajectory-level comparison",
        "",
        "| Episode | States | Graph regret | Flat regret | Flat - graph | Graph top-1 | Flat top-1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["episode_results"]:
        lines.append(
            f"| {row['episode_id']} | {row['decision_states']} "
            f"| {row['graph_mean_regret']:.5f} | {row['flat_mean_regret']:.5f} "
            f"| {row['flat_minus_graph_regret']:+.5f} "
            f"| {row['graph_top1']:.3f} | {row['flat_top1']:.3f} |"
        )
    lines.extend(["", "## Protocol integrity", ""])
    lines.extend(
        f"- [{'x' if item['passed'] else ' '}] {item['criterion']}"
        for item in audit["integrity_criteria"]
    )
    criteria_title = (
        "Frozen scientific-support criteria"
        if audit["phase"] == "confirmation"
        else "Directional development diagnostics (not formal evidence)"
    )
    lines.extend(["", f"## {criteria_title}", ""])
    support = audit["scientific_support"]
    support_label = (
        "NOT EVALUATED (development only)"
        if support is None
        else ("YES" if support else "NO")
    )
    lines.extend(
        f"- [{'x' if item['passed'] else ' '}] {item['criterion']}"
        for item in audit["scientific_criteria"]
    )
    lines.extend(
        [
            "",
            "Protocol integrity: "
            f"**{'PASS' if audit['integrity_passed'] else 'FAIL'}**.",
            "Evidence supports an incremental graph/physics representation contribution: "
            f"**{support_label}**.",
            "",
            "A negative comparison remains a valid frozen result and must not trigger seed or threshold replacement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(args: argparse.Namespace) -> Path:
    if len(args.graph_checkpoint) != 3 or len(args.flat_checkpoint) != 3:
        raise ValueError("V15.0 requires exactly three graph and three flat checkpoints")
    for checkpoint in [*args.graph_checkpoint, *args.flat_checkpoint]:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    settings = FROZEN_SETTINGS[args.phase]
    device = _select_device(args.device, args.require_cuda)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, cache_source, signature = _load_or_collect(
        args.phase, args.diagnostic_cache
    )
    target = np.stack([sample["target_delta"] for sample in samples])
    mask = np.stack([sample["target_mask"] for sample in samples]) > 0.0

    graph_predictions = []
    graph_scales = []
    graph_parameter_counts = []
    for checkpoint in args.graph_checkpoint:
        model = load_counterfactual_model_v141(checkpoint, device=device)
        graph_predictions.append(_predict(model, samples, args.batch_size, device))
        graph_scales.append(model.counterfactual_scale.detach().cpu().numpy())
        graph_parameter_counts.append(
            sum(p.numel() for p in model.counterfactual_value_head.parameters())
        )
    flat_predictions = []
    flat_scales = []
    flat_parameter_counts = []
    for checkpoint in args.flat_checkpoint:
        model, stored = load_flat_counterfactual_baseline_v150(
            checkpoint, device=device
        )
        flat_predictions.append(_predict(model, samples, args.batch_size, device))
        flat_scales.append(model.counterfactual_scale.detach().cpu().numpy())
        flat_parameter_counts.append(int(stored["trainable_parameters"]))

    all_scales = [*graph_scales, *flat_scales]
    scales_equal = all(
        np.allclose(all_scales[0], scale, rtol=0.0, atol=1.0e-7)
        for scale in all_scales[1:]
    )
    if not scales_equal:
        raise RuntimeError("Graph and flat models do not use identical training scales")
    scale = all_scales[0]
    graph_prediction = np.mean(np.stack(graph_predictions), axis=0)
    flat_prediction = np.mean(np.stack(flat_predictions), axis=0)
    graph_rows = _ranking_rows(samples, graph_prediction, target, mask, scale)
    flat_rows = _ranking_rows(samples, flat_prediction, target, mask, scale)
    paired_rows = _paired_rows(graph_rows, flat_rows)
    episode_results = _episode_comparison(paired_rows)
    bootstrap = _paired_bootstrap(
        paired_rows,
        settings["bootstrap_replicates"],
        settings["bootstrap_seed"],
    )
    graph_summary = _summarize_rows(graph_rows)
    flat_summary = _summarize_rows(flat_rows)
    mean_difference = float(
        np.mean([row["flat_minus_graph_regret"] for row in paired_rows])
    )
    positive_episodes = sum(
        row["flat_minus_graph_regret"] > 0.0 for row in episode_results
    )
    dataset_summary = summarize_counterfactual_samples(samples)
    expected_min_samples = 1000 if args.phase == "confirmation" else 30
    expected_min_states = 350 if args.phase == "confirmation" else 10
    integrity_criteria = [
        {
            "criterion": "The frozen number of complete trajectories is present",
            "passed": len(episode_results) == settings["episodes"],
        },
        {
            "criterion": "The frozen minimum paired candidate sample count is met",
            "passed": len(samples) >= expected_min_samples,
        },
        {
            "criterion": "The frozen minimum complete decision-state count is met",
            "passed": len(paired_rows) >= expected_min_states,
        },
        {
            "criterion": "Graph and flat methods rank identical state-action candidates",
            "passed": len(graph_rows) == len(flat_rows) == len(paired_rows),
        },
        {
            "criterion": "All six frozen models use identical train-only target scales",
            "passed": scales_equal,
        },
        {
            "criterion": "All graph heads and flat models have exactly 56,457 trainable parameters",
            "passed": all(
                count == EXPECTED_TRAINABLE_PARAMETERS
                for count in [*graph_parameter_counts, *flat_parameter_counts]
            ),
        },
        {
            "criterion": "Every target component has at least 85% physical-time coverage",
            "passed": bool(
                np.all(np.asarray(dataset_summary["valid_target_rate"]) >= 0.85)
            ),
        },
    ]
    scientific_criteria = [
        {
            "criterion": "Full V14.1 has lower mean regret than the flat MLP",
            "passed": mean_difference > 0.0,
        },
        {
            "criterion": "The paired trajectory-bootstrap 95% interval is above zero",
            "passed": bootstrap["ci_low"] > 0.0,
        },
        {
            "criterion": (
                "At least 9 of 12 confirmation trajectories favor Full V14.1"
                if args.phase == "confirmation"
                else "At least one of three development trajectories favors Full V14.1"
            ),
            "passed": (
                positive_episodes >= 9
                if args.phase == "confirmation"
                else positive_episodes >= 1
            ),
        },
        {
            "criterion": "Full V14.1 top-1 agreement is not below the flat MLP",
            "passed": (
                graph_summary["top1_agreement"] >= flat_summary["top1_agreement"]
            ),
        },
    ]
    audit = {
        "protocol": PROTOCOL,
        "phase": args.phase,
        "status": "implementation_only"
        if args.phase == "development"
        else "frozen_independent_confirmation",
        "data_seed": settings["data_seed"],
        "episodes": settings["episodes"],
        "samples": len(samples),
        "cache_source": cache_source,
        "cache_signature": signature,
        "graph_checkpoints": [str(path) for path in args.graph_checkpoint],
        "flat_checkpoints": [str(path) for path in args.flat_checkpoint],
        "graph_head_parameter_counts": graph_parameter_counts,
        "flat_parameter_counts": flat_parameter_counts,
        "same_training_scale": scales_equal,
        "dataset_summary": dataset_summary,
        "graph_ranking": graph_summary,
        "flat_ranking": flat_summary,
        "paired_comparison": {
            "mean_difference": mean_difference,
            "positive_episode_count": positive_episodes,
        },
        "episode_results": episode_results,
        "paired_episode_bootstrap": bootstrap,
        "integrity_criteria": integrity_criteria,
        "integrity_passed": all(item["passed"] for item in integrity_criteria),
        "scientific_criteria": scientific_criteria,
        "scientific_support": (
            all(item["passed"] for item in scientific_criteria)
            if args.phase == "confirmation"
            else None
        ),
    }
    _write_rows(args.output_dir / "paired_state_ranking_rows.csv", paired_rows)
    (args.output_dir / "V150_ARCHITECTURE_COMPARISON_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = args.output_dir / "V150_ARCHITECTURE_COMPARISON_AUDIT.md"
    _write_markdown(markdown, audit)
    print(markdown.read_text(encoding="utf-8"))
    if not audit["integrity_passed"]:
        raise RuntimeError("V15.0 protocol-integrity checks failed")
    return args.output_dir


if __name__ == "__main__":
    compare(build_parser().parse_args())
