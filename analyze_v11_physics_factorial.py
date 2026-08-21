from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
EXPECTED_TRANSITION_SCHEMA = (
    "assignment_visible_congestion_independent_arrival_streams_v4"
)
CONDITIONS = ("Full", "No physics loss", "No physical features", "Data-only graph")
METRICS = ("delta_time_sec", "delta_energy_wh", "blocked_delta")
PRIMARY_HORIZONS = (5, 10)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


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


def bootstrap_delta(
    left: Dict[int, float],
    right: Dict[int, float],
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    episodes = np.asarray(sorted(set(left) & set(right)), dtype=np.int64)
    if len(episodes) != len(left) or len(episodes) != len(right):
        raise ValueError("Paired bootstrap episode sets do not match")
    differences = np.asarray([left[int(e)] - right[int(e)] for e in episodes])
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.integers(0, len(episodes), size=len(episodes))
        values[index] = float(np.mean(differences[sampled]))
    return {
        "delta_mean": float(np.mean(differences)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_nonnegative": float(np.mean(values >= 0.0)),
        "episode_count": int(len(episodes)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the V11 physics factorial.")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=27414)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    if not evaluation_dir.is_absolute():
        evaluation_dir = ROOT / evaluation_dir
    manifest = json.loads(
        (evaluation_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    rows = read_csv(evaluation_dir / "paired_physical_predictions.csv")
    position_rows = read_csv(evaluation_dir / "paired_position_predictions.csv")
    checkpoint_rows = read_csv(evaluation_dir / "checkpoint_audit.csv")

    raw_errors: Dict[Tuple[str, int, int, int, str], List[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["condition"],
            int(row["model_seed"]),
            int(row["episode_id"]),
            int(row["horizon_steps"]),
            row["metric"],
        )
        raw_errors[key].append(float(row["absolute_error"]))
    raw_position: Dict[Tuple[str, int, int, int], List[float]] = defaultdict(list)
    for row in position_rows:
        key = (
            row["condition"],
            int(row["model_seed"]),
            int(row["episode_id"]),
            int(row["horizon_steps"]),
        )
        raw_position[key].append(float(row["position_node_accuracy"]))

    seed_rows = []
    condition_episode: Dict[Tuple[str, int, int, str], float] = {}
    episodes = sorted({key[2] for key in raw_errors})
    model_seeds = sorted({key[1] for key in raw_errors})
    horizons = sorted({key[3] for key in raw_errors})
    for condition in CONDITIONS:
        for model_seed in model_seeds:
            for horizon in horizons:
                for metric in METRICS:
                    values = [
                        value
                        for key, samples in raw_errors.items()
                        if key[0] == condition
                        and key[1] == model_seed
                        and key[3] == horizon
                        and key[4] == metric
                        for value in samples
                    ]
                    seed_rows.append(
                        {
                            "condition": condition,
                            "model_seed": model_seed,
                            "horizon_steps": horizon,
                            "metric": metric,
                            "mae": float(np.mean(values)),
                            "sample_count": len(values),
                        }
                    )
        for episode in episodes:
            for horizon in horizons:
                for metric in METRICS:
                    values = [
                        value
                        for seed in model_seeds
                        for value in raw_errors[(condition, seed, episode, horizon, metric)]
                    ]
                    if not values:
                        raise RuntimeError("Missing condition/episode prediction cell")
                    condition_episode[(condition, episode, horizon, metric)] = float(
                        np.mean(values)
                    )

    position_episode: Dict[Tuple[str, int, int], float] = {}
    for condition in CONDITIONS:
        for episode in episodes:
            for horizon in horizons:
                values = [
                    value
                    for seed in model_seeds
                    for value in raw_position[(condition, seed, episode, horizon)]
                ]
                position_episode[(condition, episode, horizon)] = float(np.mean(values))

    denominators = {
        (horizon, metric): float(
            np.mean(
                [
                    condition_episode[("Data-only graph", episode, horizon, metric)]
                    for episode in episodes
                ]
            )
        )
        for horizon in PRIMARY_HORIZONS
        for metric in METRICS
    }
    composite: Dict[str, Dict[int, float]] = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for episode in episodes:
            composite[condition][episode] = float(
                np.mean(
                    [
                        condition_episode[(condition, episode, horizon, metric)]
                        / max(denominators[(horizon, metric)], 1.0e-12)
                        for horizon in PRIMARY_HORIZONS
                        for metric in METRICS
                    ]
                )
            )

    feature_on = {
        episode: 0.5
        * (composite["Full"][episode] + composite["No physics loss"][episode])
        for episode in episodes
    }
    feature_off = {
        episode: 0.5
        * (
            composite["No physical features"][episode]
            + composite["Data-only graph"][episode]
        )
        for episode in episodes
    }
    loss_on = {
        episode: 0.5
        * (composite["Full"][episode] + composite["No physical features"][episode])
        for episode in episodes
    }
    loss_off = {
        episode: 0.5
        * (composite["No physics loss"][episode] + composite["Data-only graph"][episode])
        for episode in episodes
    }

    bootstrap_rows = []
    comparisons = [
        ("Full minus Data-only graph composite", composite["Full"], composite["Data-only graph"]),
        ("Physical-feature main effect (on minus off)", feature_on, feature_off),
        ("Physics-loss main effect (on minus off)", loss_on, loss_off),
    ]
    for index, (label, left, right) in enumerate(comparisons):
        bootstrap_rows.append(
            {
                "comparison": label,
                "horizon_steps": "5_and_10",
                "metric": "normalized_composite",
                **bootstrap_delta(
                    left,
                    right,
                    args.bootstrap_replicates,
                    args.bootstrap_seed + index,
                ),
            }
        )
    h10_metric_changes = {}
    for index, metric in enumerate(METRICS):
        full = {
            episode: condition_episode[("Full", episode, 10, metric)]
            for episode in episodes
        }
        data_only = {
            episode: condition_episode[("Data-only graph", episode, 10, metric)]
            for episode in episodes
        }
        result = bootstrap_delta(
            full,
            data_only,
            args.bootstrap_replicates,
            args.bootstrap_seed + 10 + index,
        )
        baseline_mean = float(np.mean(list(data_only.values())))
        result["relative_change"] = result["delta_mean"] / max(baseline_mean, 1.0e-12)
        h10_metric_changes[metric] = result
        bootstrap_rows.append(
            {
                "comparison": "Full minus Data-only graph",
                "horizon_steps": 10,
                "metric": metric,
                **result,
            }
        )
    full_position = {
        episode: position_episode[("Full", episode, 10)] for episode in episodes
    }
    data_position = {
        episode: position_episode[("Data-only graph", episode, 10)]
        for episode in episodes
    }
    position_result = bootstrap_delta(
        full_position,
        data_position,
        args.bootstrap_replicates,
        args.bootstrap_seed + 20,
    )
    bootstrap_rows.append(
        {
            "comparison": "Full minus Data-only graph",
            "horizon_steps": 10,
            "metric": "position_node_accuracy",
            **position_result,
        }
    )

    condition_rows = []
    for condition in CONDITIONS:
        for horizon in horizons:
            for metric in METRICS:
                values = [
                    condition_episode[(condition, episode, horizon, metric)]
                    for episode in episodes
                ]
                condition_rows.append(
                    {
                        "condition": condition,
                        "horizon_steps": horizon,
                        "metric": metric,
                        "mae": float(np.mean(values)),
                        "episode_sd": float(np.std(values, ddof=1)),
                        "episode_count": len(values),
                    }
                )
        condition_rows.append(
            {
                "condition": condition,
                "horizon_steps": "5_and_10",
                "metric": "normalized_composite",
                "mae": float(np.mean(list(composite[condition].values()))),
                "episode_sd": float(np.std(list(composite[condition].values()), ddof=1)),
                "episode_count": len(episodes),
            }
        )

    full_vs_data = bootstrap_rows[0]
    feature_effect = bootstrap_rows[1]
    loss_effect = bootstrap_rows[2]
    improved_metrics = sum(
        h10_metric_changes[metric]["delta_mean"] < 0.0 for metric in METRICS
    )
    no_large_degradation = all(
        h10_metric_changes[metric]["relative_change"] <= 0.10 for metric in METRICS
    )
    integrity = (
        int(manifest["test_seed"]) == 27413
        and int(manifest["episodes"]) == 20
        and manifest.get("transition_schema_version") == EXPECTED_TRANSITION_SCHEMA
        and len(checkpoint_rows) == 12
        and len(episodes) == 20
        and len({int(row["parameter_count"]) for row in checkpoint_rows}) == 1
        and {row["transition_schema_version"] for row in checkpoint_rows}
        == {EXPECTED_TRANSITION_SCHEMA}
        and {int(row["data_seed"]) for row in checkpoint_rows} == {4200}
        and {int(row["split_seed"]) for row in checkpoint_rows} == {4200}
    )
    criteria = [
        ("Protocol integrity and all 12 equal-capacity checkpoints", integrity),
        (
            "Full V11 composite error is lower than Data-only graph with 95% CI below zero",
            float(full_vs_data["ci_high"]) < 0.0,
        ),
        (
            "At least two of three H10 physical metrics improve and none degrades by more than 10%",
            improved_metrics >= 2 and no_large_degradation,
        ),
        (
            "Physical-feature main effect lowers composite error",
            float(feature_effect["delta_mean"]) < 0.0,
        ),
        (
            "Physics-loss main effect lowers composite error",
            float(loss_effect["delta_mean"]) < 0.0,
        ),
        (
            "Full V11 H10 position accuracy is no more than 2 percentage points below Data-only graph",
            float(position_result["delta_mean"]) >= -0.02,
        ),
    ]
    passed = all(value for _, value in criteria)

    write_csv(evaluation_dir / "seed_level_summary.csv", seed_rows)
    write_csv(evaluation_dir / "condition_summary.csv", condition_rows)
    write_csv(evaluation_dir / "paired_episode_bootstrap.csv", bootstrap_rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    composite_means = [float(np.mean(list(composite[name].values()))) for name in CONDITIONS]
    composite_sem = [
        float(np.std(list(composite[name].values()), ddof=1) / np.sqrt(len(episodes)))
        for name in CONDITIONS
    ]
    x = np.arange(len(CONDITIONS))
    axes[0].bar(x, composite_means, yerr=np.asarray(composite_sem) * 1.96, capsize=4)
    axes[0].set_xticks(x, CONDITIONS, rotation=18, ha="right")
    axes[0].set_ylabel("Normalized multistep MAE")
    axes[0].set_title("Physics-factorial open-loop prediction")
    axes[0].grid(axis="y", alpha=0.25)
    relative = [100.0 * h10_metric_changes[m]["relative_change"] for m in METRICS]
    axes[1].bar(np.arange(len(METRICS)), relative)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].axhline(10.0, color="red", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(np.arange(len(METRICS)), METRICS, rotation=18, ha="right")
    axes[1].set_ylabel("Full vs data-only MAE change (%)")
    axes[1].set_title("H10 physical-output change")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(evaluation_dir / "v11_physics_factorial_audit.png", dpi=300)
    plt.close(fig)

    lines = [
        "# V11 physics-factorial independent audit",
        "",
        f"Test seed: {manifest['test_seed']}; complete episodes: {len(episodes)}; "
        f"model checkpoints: {len(checkpoint_rows)}.",
        "",
        "| Condition | Normalized H5/H10 composite MAE |",
        "|---|---:|",
    ]
    for condition, value in zip(CONDITIONS, composite_means):
        lines.append(f"| {condition} | {value:.4f} |")
    lines.extend(
        [
            "",
            "## Paired episode bootstrap",
            "",
            "| Comparison | Metric | Mean delta | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in bootstrap_rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | "
            f"{float(row['delta_mean']):+.4f} | "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] |"
        )
    lines.extend(["", "## Preregistered criteria", ""])
    for label, value in criteria:
        lines.append(f"- [{'x' if value else ' '}] {label}")
    lines.extend(
        [
            "",
            f"Physics-factorial evidence package passed: **{'YES' if passed else 'NO'}**.",
            "",
            "Training losses are not compared across cells because the physics-loss term differs. "
            "All conclusions use common held-out physical-unit metrics.",
        ]
    )
    report = Path(args.report)
    if not report.is_absolute():
        report = ROOT / report
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
