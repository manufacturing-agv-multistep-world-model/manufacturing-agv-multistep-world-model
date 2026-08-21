from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINES = ("Rule", "MLP", "GRU", "GNN")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def cv(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / max(abs(mean), 1.0e-12)


def build_parser():
    parser = argparse.ArgumentParser(description="Audit the future-risk benchmark.")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = ROOT / benchmark_dir
    architecture_rows = {
        row["architecture"]: row
        for row in read_csv(benchmark_dir / "architecture_ensemble_summary.csv")
    }
    method_rows = read_csv(benchmark_dir / "method_seed_summary.csv")
    bootstrap_rows = read_csv(
        benchmark_dir / "paired_bootstrap_v13_vs_baselines.csv"
    )
    manifest = json.loads(
        (benchmark_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    full = architecture_rows["V13"]
    criteria = []
    prevalence = float(full["event_prevalence"])
    criteria.append(("Test prevalence is identifiable (2%-20%)", 0.02 <= prevalence <= 0.20))
    criteria.append(("V13 ensemble ROC AUC is at least 0.80", float(full["roc_auc"]) >= 0.80))
    criteria.append(
        (
            "V13 ensemble average precision exceeds prevalence by at least 0.15",
            float(full["average_precision"]) - prevalence >= 0.15,
        )
    )
    criteria.append(
        (
            "V13 has the highest point-estimate ROC AUC",
            all(
                float(full["roc_auc"]) > float(architecture_rows[name]["roc_auc"])
                for name in BASELINES
            ),
        )
    )
    criteria.append(
        (
            "V13 has the highest point-estimate average precision",
            all(
                float(full["average_precision"])
                > float(architecture_rows[name]["average_precision"])
                for name in BASELINES
            ),
        )
    )
    learned_ci_checks = []
    for baseline in ("MLP", "GRU", "GNN"):
        for metric in ("roc_auc", "average_precision"):
            row = next(
                item
                for item in bootstrap_rows
                if item["baseline_architecture"] == baseline
                and item["metric"] == metric
            )
            learned_ci_checks.append(float(row["ci_low"]) > 0.0)
    criteria.append(
        (
            "Paired episode-bootstrap 95% intervals favor V13 over every learned baseline for AUC and AP",
            all(learned_ci_checks),
        )
    )
    v13_auc = [
        float(row["roc_auc"])
        for row in method_rows
        if row["architecture"] == "V13"
    ]
    criteria.append(
        ("V13 initialization-level ROC AUC CV is at most 5%", cv(v13_auc) <= 0.05)
    )
    passed = all(value for _, value in criteria)

    lines = [
        "# Future charge-risk architecture benchmark audit",
        "",
        f"Fresh test seed: {manifest['seed']}; episodes: {manifest['episodes']}; "
        f"eligible samples: {manifest['sample_count']}.",
        "",
        "| Architecture | ROC AUC | Average precision | Brier score | Prevalence |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("Rule", "MLP", "GRU", "GNN", "V13"):
        row = architecture_rows[name]
        lines.append(
            f"| {name} | {float(row['roc_auc']):.4f} | "
            f"{float(row['average_precision']):.4f} | "
            f"{float(row['brier_score']):.4f} | {float(row['event_prevalence']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired episode-bootstrap: V13 minus baseline",
            "",
            "| Baseline | Metric | Mean difference | 95% CI | P(delta <= 0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap_rows:
        lines.append(
            f"| {row['baseline_architecture']} | {row['metric']} | "
            f"{float(row['delta_mean']):+.4f} | "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] | "
            f"{float(row['probability_nonpositive']):.4f} |"
        )
    lines.extend(["", "## Preregistered criteria", ""])
    for label, value in criteria:
        lines.append(f"- [{'x' if value else ' '}] {label}")
    lines.extend(
        [
            "",
            f"Architecture-evidence package passed: **{'YES' if passed else 'NO'}**.",
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
