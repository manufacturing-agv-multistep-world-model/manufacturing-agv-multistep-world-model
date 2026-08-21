from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from diagnose_world_model_multistep import (
    binary_average_precision,
    spearman_correlation,
)


ROOT = Path(__file__).resolve().parent
MODEL_SEEDS = (42, 43, 44)
DIRECT_SOURCE = "v14_direct_terminal_head"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit V14 dual-timescale fresh-seed open-loop diagnostics."
    )
    parser.add_argument("--results-root", default="experiment_results")
    parser.add_argument(
        "--report", default="experiment_results/v14_dual_timescale_open_loop_audit.md"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=314159)
    return parser


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing V14 diagnostic file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def one_row(
    rows: Sequence[Dict[str, str]],
    *,
    source: str,
    metric: str,
    horizon: int = 1,
) -> Dict[str, str]:
    matches = [
        row
        for row in rows
        if row["source"] == source
        and row["metric"] == metric
        and int(row["rollout_horizon_steps"]) == horizon
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for source={source}, metric={metric}, H={horizon}"
        )
    return matches[0]


def coefficient_of_variation(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.std(array) / max(abs(float(np.mean(array))), 1.0e-12))


def paired_metrics(rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    actual_energy = np.asarray([float(row["actual_energy_wh"]) for row in rows])
    actual_tasks = np.asarray([float(row["actual_completed_tasks"]) for row in rows])
    actual_queue = np.asarray([float(row["actual_charge_queue_steps"]) for row in rows])
    valid_efficiency = actual_tasks > 0.0
    actual_eer = actual_energy[valid_efficiency] / actual_tasks[valid_efficiency]
    values: Dict[str, float] = {}
    for source in ("direct", "extrapolated"):
        predicted_energy = np.asarray(
            [float(row[f"{source}_energy_wh"]) for row in rows]
        )
        predicted_tasks = np.asarray(
            [float(row[f"{source}_completed_tasks"]) for row in rows]
        )
        predicted_queue = np.asarray(
            [float(row[f"{source}_charge_queue_steps"]) for row in rows]
        )
        predicted_eer = predicted_energy[valid_efficiency] / np.maximum(
            predicted_tasks[valid_efficiency], 0.25
        )
        values[f"{source}_eer_mae"] = float(
            np.mean(np.abs(predicted_eer - actual_eer))
        )
        values[f"{source}_eer_spearman"] = spearman_correlation(
            actual_eer, predicted_eer
        )
        values[f"{source}_queue_ap"] = binary_average_precision(
            actual_queue > 0.0, predicted_queue
        )
    values["eer_mae_delta"] = (
        values["direct_eer_mae"] - values["extrapolated_eer_mae"]
    )
    values["queue_ap_delta"] = (
        values["direct_queue_ap"] - values["extrapolated_queue_ap"]
    )
    return values


def episode_bootstrap(
    rows: Sequence[Dict[str, str]],
    samples: int,
    seed: int,
) -> Dict[str, tuple[float, float]]:
    if samples < 100:
        raise ValueError("At least 100 episode-bootstrap samples are required")
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_id"]), []).append(row)
    episode_ids = np.asarray(sorted(grouped), dtype=np.int64)
    if episode_ids.size < 2:
        raise ValueError("Episode bootstrap requires at least two test episodes")
    rng = np.random.default_rng(seed)
    distributions = {"eer_mae_delta": [], "queue_ap_delta": []}
    for _ in range(samples):
        sampled_ids = rng.choice(episode_ids, size=episode_ids.size, replace=True)
        sampled_rows = [row for episode_id in sampled_ids for row in grouped[int(episode_id)]]
        metrics = paired_metrics(sampled_rows)
        for key in distributions:
            if math.isfinite(metrics[key]):
                distributions[key].append(metrics[key])
    return {
        key: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for key, values in distributions.items()
        if values
    }


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = ROOT / results_root

    summaries: Dict[int, Dict[str, float]] = {}
    intervals: Dict[int, Dict[str, tuple[float, float]]] = {}
    for model_seed in MODEL_SEEDS:
        directory = results_root / f"world_model_multistep_v14_dual_timescale_v1_seed{model_seed}"
        metric_rows = read_rows(directory / "future_terminal_kpi_by_horizon.csv")
        paired_rows = read_rows(directory / "future_terminal_paired_predictions.csv")
        energy = one_row(
            metric_rows,
            source=DIRECT_SOURCE,
            metric="future_cumulative_energy_wh",
        )
        tasks = one_row(
            metric_rows,
            source=DIRECT_SOURCE,
            metric="future_cumulative_completed_tasks",
        )
        efficiency = one_row(
            metric_rows,
            source=DIRECT_SOURCE,
            metric="future_energy_per_completed_task_wh",
        )
        queue = one_row(
            metric_rows,
            source=DIRECT_SOURCE,
            metric="future_charge_queue_event",
        )
        paired = paired_metrics(paired_rows)
        summaries[model_seed] = {
            "energy_nmae": float(energy["normalized_mae"]),
            "energy_spearman": float(energy["spearman"]),
            "task_nmae": float(tasks["normalized_mae"]),
            "task_spearman": float(tasks["spearman"]),
            "eer_spearman": float(efficiency["spearman"]),
            "queue_prevalence": float(queue["event_prevalence"]),
            "queue_auc": float(queue["roc_auc"]),
            "queue_ap": float(queue["average_precision"]),
            **paired,
        }
        intervals[model_seed] = episode_bootstrap(
            paired_rows,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + model_seed,
        )

    criteria: List[tuple[str, bool]] = []
    criteria.append(
        (
            "All model seeds have terminal-energy normalized MAE <= 0.25",
            all(row["energy_nmae"] <= 0.25 for row in summaries.values()),
        )
    )
    criteria.append(
        (
            "All model seeds have terminal-throughput normalized MAE <= 0.35",
            all(row["task_nmae"] <= 0.35 for row in summaries.values()),
        )
    )
    criteria.append(
        (
            "All model seeds have energy and throughput Spearman >= 0.50",
            all(
                row["energy_spearman"] >= 0.50 and row["task_spearman"] >= 0.50
                for row in summaries.values()
            ),
        )
    )
    criteria.append(
        (
            "All model seeds have future EER Spearman >= 0.50",
            all(row["eer_spearman"] >= 0.50 for row in summaries.values()),
        )
    )
    criteria.append(
        (
            "Each test set has identifiable charge-queue prevalence (2%-20%)",
            all(
                0.02 <= row["queue_prevalence"] <= 0.20
                for row in summaries.values()
            ),
        )
    )
    criteria.append(
        (
            "All model seeds have terminal queue AUC >= 0.80 and AP lift >= 0.15",
            all(
                row["queue_auc"] >= 0.80
                and row["queue_ap"] - row["queue_prevalence"] >= 0.15
                for row in summaries.values()
            ),
        )
    )
    criteria.append(
        (
            "Across-seed energy and throughput normalized-MAE CV <= 10%",
            coefficient_of_variation(
                [row["energy_nmae"] for row in summaries.values()]
            )
            <= 0.10
            and coefficient_of_variation(
                [row["task_nmae"] for row in summaries.values()]
            )
            <= 0.10,
        )
    )
    supported_improvement = 0
    significant_degradation = 0
    for model_seed in MODEL_SEEDS:
        eer_interval = intervals[model_seed]["eer_mae_delta"]
        queue_interval = intervals[model_seed]["queue_ap_delta"]
        if eer_interval[1] < 0.0 or queue_interval[0] > 0.0:
            supported_improvement += 1
        if eer_interval[0] > 0.0 and queue_interval[1] < 0.0:
            significant_degradation += 1
    criteria.append(
        (
            "At least two seeds improve EER error or queue AP over short-rollout extrapolation, with no seed worse on both",
            supported_improvement >= 2 and significant_degradation == 0,
        )
    )
    proceed = all(passed for _, passed in criteria)

    lines = [
        "# V14 dual-timescale world-model independent open-loop audit",
        "",
        "The direct 80-step terminal head is compared on identical fresh trajectories "
        "against linear extrapolation of the frozen V13 10-step rollout. Episode-level "
        "bootstrap intervals preserve within-trajectory dependence.",
        "",
        "| Model seed | Energy NMAE | Energy rho | Task NMAE | Task rho | EER rho | Queue prevalence | Queue AUC | Queue AP | Delta EER MAE | Delta queue AP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_seed in MODEL_SEEDS:
        row = summaries[model_seed]
        lines.append(
            f"| {model_seed} | {row['energy_nmae']:.4f} | {row['energy_spearman']:.4f} | "
            f"{row['task_nmae']:.4f} | {row['task_spearman']:.4f} | "
            f"{row['eer_spearman']:.4f} | {row['queue_prevalence']:.4f} | "
            f"{row['queue_auc']:.4f} | {row['queue_ap']:.4f} | "
            f"{row['eer_mae_delta']:+.4f} | {row['queue_ap_delta']:+.4f} |"
        )
    lines.extend(["", "## Episode-bootstrap differences: V14 minus extrapolated V13", ""])
    for model_seed in MODEL_SEEDS:
        eer_interval = intervals[model_seed]["eer_mae_delta"]
        queue_interval = intervals[model_seed]["queue_ap_delta"]
        lines.append(
            f"- Seed {model_seed}: EER MAE 95% CI [{eer_interval[0]:+.4f}, "
            f"{eer_interval[1]:+.4f}]; queue AP 95% CI "
            f"[{queue_interval[0]:+.4f}, {queue_interval[1]:+.4f}]."
        )
    lines.extend(["", "## Preregistered continuation criteria", ""])
    for label, passed in criteria:
        lines.append(f"- [{'x' if passed else ' '}] {label}")
    lines.extend(
        [
            "",
            f"Proceed to shadow/counterfactual evaluation: **{'YES' if proceed else 'NO'}**.",
            "",
            "Passing does not constitute a closed-loop performance claim.",
        ]
    )
    report = Path(args.report)
    if not report.is_absolute():
        report = ROOT / report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
