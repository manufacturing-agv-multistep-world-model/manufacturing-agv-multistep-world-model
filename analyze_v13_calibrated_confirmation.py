from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
MODEL_SEEDS = (42, 43, 44)
ROLLOUT_HORIZONS = (1, 5, 10)
CONFIRMATION_SEED = 23313
CONFIRMATION_EPISODES = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the preregistered calibrated V13 confirmation."
    )
    parser.add_argument("--results-root", default="experiment_results")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--report", required=True)
    return parser


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing confirmation result: {path}")
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
    calibration_path = Path(args.calibration)
    if not calibration_path.is_absolute():
        calibration_path = ROOT / calibration_path
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    selected: Dict[tuple[int, int], Dict[str, str]] = {}
    integrity_checks: List[tuple[str, bool]] = []
    for seed in MODEL_SEEDS:
        result_dir = (
            results_root / f"world_model_multistep_v13_v2_confirm_model_seed{seed}"
        )
        rows = read_csv(result_dir / "future_congestion_risk_by_horizon.csv")
        manifest = json.loads(
            (result_dir / "diagnostic_manifest.json").read_text(encoding="utf-8")
        )
        frozen_threshold = float(calibration["models"][str(seed)]["threshold"])
        integrity_checks.extend(
            [
                (
                    f"Model {seed} uses confirmation seed {CONFIRMATION_SEED}",
                    int(manifest["seed"]) == CONFIRMATION_SEED,
                ),
                (
                    f"Model {seed} uses {CONFIRMATION_EPISODES} confirmation episodes",
                    int(manifest["episodes"]) == CONFIRMATION_EPISODES,
                ),
                (
                    f"Model {seed} uses its frozen calibration threshold",
                    abs(float(manifest["future_risk_threshold"]) - frozen_threshold)
                    <= 1.0e-12,
                ),
            ]
        )
        for horizon in ROLLOUT_HORIZONS:
            matches = [
                row
                for row in rows
                if row["metric"] == "future_charge_queue_risk"
                and int(row["rollout_horizon_steps"]) == horizon
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one charge-risk row for model={seed}, horizon={horizon}"
                )
            selected[(seed, horizon)] = matches[0]

    criteria: List[tuple[str, bool]] = []
    h1_rows = [selected[(seed, 1)] for seed in MODEL_SEEDS]
    criteria.append(
        (
            "Each confirmation set has 2%-20% charge-risk prevalence",
            all(0.02 <= float(row["event_prevalence"]) <= 0.20 for row in h1_rows),
        )
    )
    for horizon, minimum_auc in ((1, 0.80), (5, 0.75), (10, 0.70)):
        criteria.append(
            (
                f"All models achieve ROC AUC >= {minimum_auc:.2f} at rollout H{horizon}",
                all(
                    float(selected[(seed, horizon)]["roc_auc"]) >= minimum_auc
                    for seed in MODEL_SEEDS
                ),
            )
        )
    criteria.append(
        (
            "All models exceed the H1 prevalence baseline in average precision by >= 0.15",
            all(
                float(row["average_precision"]) - float(row["event_prevalence"])
                >= 0.15
                for row in h1_rows
            ),
        )
    )
    criteria.append(
        (
            "All models achieve H1 precision >= 0.50 at their frozen thresholds",
            all(float(row["precision_at_threshold"]) >= 0.50 for row in h1_rows),
        )
    )
    recalls = [float(row["recall_at_threshold"]) for row in h1_rows]
    criteria.append(
        (
            "At least two models achieve H1 recall >= 0.65 and none is below 0.55",
            sum(value >= 0.65 for value in recalls) >= 2 and min(recalls) >= 0.55,
        )
    )
    criteria.append(
        (
            "All models have H1 Brier score <= 0.25",
            all(float(row["brier_score"]) <= 0.25 for row in h1_rows),
        )
    )
    h1_auc = [float(row["roc_auc"]) for row in h1_rows]
    criteria.append(
        (
            "Across-model H1 ROC AUC coefficient of variation <= 10%",
            coefficient_of_variation(h1_auc) <= 0.10,
        )
    )
    integrity_passed = all(passed for _, passed in integrity_checks)
    proceed = integrity_passed and all(passed for _, passed in criteria)

    lines = [
        "# V13-v2 calibrated independent confirmation audit",
        "",
        "Thresholds were selected on calibration seed 22313 and frozen before "
        "evaluation on confirmation seed 23313. Model parameters were unchanged.",
        "",
        "| Model seed | Rollout H | Threshold | Prevalence | ROC AUC | Average precision | Precision | Recall | F1 | Brier |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in MODEL_SEEDS:
        for horizon in ROLLOUT_HORIZONS:
            row = selected[(seed, horizon)]
            lines.append(
                f"| {seed} | {horizon} | {float(row['decision_threshold']):.3f} | "
                f"{float(row['event_prevalence']):.4f} | {float(row['roc_auc']):.4f} | "
                f"{float(row['average_precision']):.4f} | "
                f"{float(row['precision_at_threshold']):.4f} | "
                f"{float(row['recall_at_threshold']):.4f} | "
                f"{float(row['f1_at_threshold']):.4f} | {float(row['brier_score']):.4f} |"
            )
    lines.extend(["", "## Protocol-integrity checks", ""])
    for label, passed in integrity_checks:
        lines.append(f"- [{'x' if passed else ' '}] {label}")
    lines.extend(["", "## Preregistered continuation criteria", ""])
    for label, passed in criteria:
        lines.append(f"- [{'x' if passed else ' '}] {label}")
    lines.extend(
        [
            "",
            f"Proceed to shadow/counterfactual evaluation: **{'YES' if proceed else 'NO'}**.",
            "",
            "Passing does not constitute a closed-loop control-performance claim.",
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
