from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_SEEDS = (42, 43, 44)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_row(rows: list[dict[str, str]], metric: str, horizon: int) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["metric"] == metric and int(row["horizon_steps"]) == horizon
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {metric} at H={horizon}; found {len(matches)}")
    return matches[0]


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    summaries: list[dict[str, float | int | bool]] = []
    manifests = []
    for seed in MODEL_SEEDS:
        directory = args.results_root / f"world_model_multistep_v12_charge_seed{seed}"
        rows = read_csv(directory / "multistep_kpi_error_by_horizon.csv")
        manifest = json.loads((directory / "diagnostic_manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        charge_h1 = metric_row(rows, "charge_queue_blocked_agent_steps", 1)
        charge_h5 = metric_row(rows, "charge_queue_blocked_agent_steps", 5)
        charge_h10 = metric_row(rows, "charge_queue_blocked_agent_steps", 10)
        route_h10 = metric_row(rows, "route_blocked_agent_steps", 10)
        energy_h5 = metric_row(rows, "delta_energy_wh", 5)
        throughput_h10 = metric_row(rows, "throughput_delta", 10)
        gates = {
            "charge_h1_f1": value(charge_h1, "event_f1_at_0_5") >= 0.85,
            "charge_h5_f1": value(charge_h5, "event_f1_at_0_5") >= 0.60,
            "charge_h10_f1": value(charge_h10, "event_f1_at_0_5") >= 0.60,
            "charge_h5_r2": value(charge_h5, "r2") >= 0.20,
            "route_h10_f1": value(route_h10, "event_f1_at_0_5") >= 0.85,
            "energy_h5_r2": value(energy_h5, "r2") >= 0.40,
            "throughput_h10_mae": value(throughput_h10, "mae") <= 0.06,
        }
        summaries.append(
            {
                "seed": seed,
                "charge_h1_f1": value(charge_h1, "event_f1_at_0_5"),
                "charge_h5_f1": value(charge_h5, "event_f1_at_0_5"),
                "charge_h10_f1": value(charge_h10, "event_f1_at_0_5"),
                "charge_h5_r2": value(charge_h5, "r2"),
                "charge_h10_r2": value(charge_h10, "r2"),
                "route_h10_f1": value(route_h10, "event_f1_at_0_5"),
                "energy_h5_r2": value(energy_h5, "r2"),
                "throughput_h10_mae": value(throughput_h10, "mae"),
                "passed": all(gates.values()),
            }
        )

    model_versions = {manifest["model_version"] for manifest in manifests}
    evaluation_seeds = {int(manifest["seed"]) for manifest in manifests}
    protocols = {manifest["validation_protocol"] for manifest in manifests}
    protocol_valid = (
        model_versions == {"pi_gwm_multistep_v12_charge_aware"}
        and len(evaluation_seeds) == 1
        and protocols == {"fresh_seed_open_loop_without_teacher_forcing"}
    )
    passed_models = sum(bool(row["passed"]) for row in summaries)
    proceed = protocol_valid and passed_models >= 2

    lines = [
        "# V12 charge-aware world-model open-loop diagnostic audit",
        "",
        "This is a development-stage continuation gate on fresh trajectories without teacher forcing.",
        "Thresholds are reported as development criteria, not as retrospectively claimed preregistration.",
        "",
        "| Model seed | Charge F1 H1 | Charge F1 H5 | Charge F1 H10 | Charge R2 H5 | Charge R2 H10 | Route F1 H10 | Energy R2 H5 | Throughput MAE H10 | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['seed']} | {row['charge_h1_f1']:.3f} | {row['charge_h5_f1']:.3f} | "
            f"{row['charge_h10_f1']:.3f} | {row['charge_h5_r2']:.3f} | "
            f"{row['charge_h10_r2']:.3f} | {row['route_h10_f1']:.3f} | "
            f"{row['energy_h5_r2']:.3f} | {row['throughput_h10_mae']:.3f} | "
            f"{'YES' if row['passed'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Development continuation criteria",
            "",
            "- Charge-queue event F1 >= 0.85 at H1.",
            "- Charge-queue event F1 >= 0.60 at H5 and H10.",
            "- Charge-queue magnitude R2 >= 0.20 at H5.",
            "- Route-blocking event F1 >= 0.85 at H10.",
            "- Energy R2 >= 0.40 at H5.",
            "- Throughput MAE <= 0.06 SKU/step at H10.",
            "- At least two independently initialized models must pass every criterion.",
            "",
            f"- Protocol integrity: **{'PASS' if protocol_valid else 'FAIL'}**.",
            f"- Independently initialized models passing all gates: **{passed_models}/3**.",
            f"- Proceed to preregistered one-hour control pilot: **{'YES' if proceed else 'NO'}**.",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
