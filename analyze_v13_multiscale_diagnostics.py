from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
MODEL_SEEDS = (42, 43, 44)
ROLLOUT_HORIZONS = (1, 5, 10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit preregistered V13 multi-timescale open-loop diagnostics."
    )
    parser.add_argument("--results-root", default="experiment_results")
    parser.add_argument(
        "--report", default="experiment_results/v13_multiscale_open_loop_audit.md"
    )
    return parser


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing V13 diagnostic file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def coefficient_of_variation(values: List[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / max(abs(mean), 1.0e-12)


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    selected: Dict[tuple[int, int], Dict[str, str]] = {}
    for seed in MODEL_SEEDS:
        path = (
            results_root
            / f"world_model_multistep_v13_multiscale_v2_seed{seed}"
            / "future_congestion_risk_by_horizon.csv"
        )
        rows = read_rows(path)
        for horizon in ROLLOUT_HORIZONS:
            matches = [
                row
                for row in rows
                if row["metric"] == "future_charge_queue_risk"
                and int(row["rollout_horizon_steps"]) == horizon
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one charge-risk row for seed={seed}, horizon={horizon}"
                )
            selected[(seed, horizon)] = matches[0]

    criteria: List[tuple[str, bool]] = []
    h1_rows = [selected[(seed, 1)] for seed in MODEL_SEEDS]
    criteria.append(
        (
            "Each fresh-seed test set has identifiable charge events (2%-95% prevalence)",
            all(0.02 <= float(row["event_prevalence"]) <= 0.95 for row in h1_rows),
        )
    )
    for horizon, minimum_auc in ((1, 0.80), (5, 0.75), (10, 0.70)):
        criteria.append(
            (
                f"All model seeds achieve charge-risk ROC AUC >= {minimum_auc:.2f} at rollout H{horizon}",
                all(
                    float(selected[(seed, horizon)]["roc_auc"]) >= minimum_auc
                    for seed in MODEL_SEEDS
                ),
            )
        )
    criteria.append(
        (
            "All model seeds exceed the H1 prevalence baseline in average precision by >= 0.15",
            all(
                float(row["average_precision"]) - float(row["event_prevalence"])
                >= 0.15
                for row in h1_rows
            ),
        )
    )
    criteria.append(
        (
            "At least two model seeds achieve H1 recall >= 0.70 at the fixed 0.5 threshold",
            sum(float(row["recall_at_0_5"]) >= 0.70 for row in h1_rows) >= 2,
        )
    )
    criteria.append(
        (
            "At least two model seeds achieve H1 precision >= 0.50 at the fixed 0.5 threshold",
            sum(float(row["precision_at_0_5"]) >= 0.50 for row in h1_rows) >= 2,
        )
    )
    criteria.append(
        (
            "All model seeds have H1 Brier score <= 0.25",
            all(float(row["brier_score"]) <= 0.25 for row in h1_rows),
        )
    )
    h1_auc = [float(row["roc_auc"]) for row in h1_rows]
    criteria.append(
        (
            "Across-seed H1 ROC AUC coefficient of variation <= 10%",
            coefficient_of_variation(h1_auc) <= 0.10,
        )
    )
    proceed = all(passed for _, passed in criteria)

    lines = [
        "# V13 multi-timescale world-model open-loop audit",
        "",
        "The 80-step future charge-queue head is evaluated on fresh trajectories. "
        "Passing permits shadow/counterfactual evaluation only; it does not grant control authority.",
        "",
        "| Model seed | Rollout H | Prevalence | ROC AUC | Average precision | Precision@0.5 | Recall@0.5 | F1@0.5 | Brier |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in MODEL_SEEDS:
        for horizon in ROLLOUT_HORIZONS:
            row = selected[(seed, horizon)]
            lines.append(
                f"| {seed} | {horizon} | {float(row['event_prevalence']):.4f} | "
                f"{float(row['roc_auc']):.4f} | {float(row['average_precision']):.4f} | "
                f"{float(row['precision_at_0_5']):.4f} | {float(row['recall_at_0_5']):.4f} | "
                f"{float(row['f1_at_0_5']):.4f} | {float(row['brier_score']):.4f} |"
            )
    lines.extend(["", "## Preregistered continuation criteria", ""])
    for label, passed in criteria:
        lines.append(f"- [{'x' if passed else ' '}] {label}")
    lines.extend(
        [
            "",
            f"Proceed to shadow/counterfactual evaluation: **{'YES' if proceed else 'NO'}**.",
            "",
            "No closed-loop performance claim is permitted at this stage.",
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
