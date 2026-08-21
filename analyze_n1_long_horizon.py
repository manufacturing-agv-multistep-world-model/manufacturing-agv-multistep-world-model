from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analyze_multistep_decision_attribution import (
    METHODS,
    as_float,
    executed_model_traces_differ,
    load_csv,
    proposal_difference_rate,
)
from analyze_n1_confirmation import (
    DATA_ONLY,
    FULL,
    PI_ONLY,
    aggregate_mean_relative_change,
    checkpoint_hashes,
    paired_bootstrap,
    write_csv,
)


FROZEN_ENV_SEEDS = list(range(36001, 36011))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit frozen N1 long-horizon stability")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--confirmation-dir", required=True)
    parser.add_argument("--hours", type=float, choices=[4.0, 8.0], required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    return parser


def main(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir)
    confirmation_dir = Path(args.confirmation_dir)
    summary_rows = load_csv(result_dir / "summary.csv")
    trace_rows = load_csv(result_dir / "trace.csv")
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    confirmation_manifest = json.loads(
        (confirmation_dir / "run_manifest.json").read_text(encoding="utf-8")
    )

    expected_config = {
        "phase": "confirmation",
        "hours": args.hours,
        "env_seeds": FROZEN_ENV_SEEDS,
        "model_seeds": [42, 43, 44],
        "control_mode": "ensemble",
        "minimum_ensemble_agreement": 2,
        "scenario": "rush",
        "capacity_mode": "baseline",
        "planning_horizon": 3,
        "beam_width": 8,
        "risk_gate": 0.75,
        "override_mode": "evidence_gated",
    }
    checks: dict[str, bool] = {
        f"Frozen configuration: {key}": manifest.get(key) == expected
        for key, expected in expected_config.items()
    }
    frozen_fields = (
        "model_seeds",
        "control_mode",
        "minimum_ensemble_agreement",
        "scenario",
        "capacity_mode",
        "planning_horizon",
        "beam_width",
        "risk_gate",
        "override_mode",
        "transition_schema_version",
    )
    checks["Long-horizon control settings match independent confirmation"] = all(
        manifest.get(field) == confirmation_manifest.get(field)
        for field in frozen_fields
    )
    checks["Long-horizon checkpoints match independent confirmation"] = (
        checkpoint_hashes(manifest) == checkpoint_hashes(confirmation_manifest)
    )
    checks["Long-horizon seeds are disjoint from independent confirmation"] = not (
        set(manifest.get("env_seeds", []))
        & set(confirmation_manifest.get("env_seeds", []))
    )

    indexed: dict[tuple[int, str], dict[str, str]] = {}
    duplicate = False
    for row in summary_rows:
        key = (int(float(row["seed"])), row["method"])
        duplicate = duplicate or key in indexed
        indexed[key] = row
    expected_keys = {
        (seed, method) for seed in FROZEN_ENV_SEEDS for method in METHODS
    }
    checks["Exactly one run exists for every frozen seed and method"] = (
        not duplicate and set(indexed) == expected_keys
    )
    if set(indexed) != expected_keys:
        raise ValueError("Long-horizon result matrix is incomplete or unexpected")
    checks["Every run reaches the fixed physical horizon"] = all(
        as_float(row, "fixed_time_reached") == 1.0 for row in summary_rows
    )
    checks["Fixed-horizon overshoot is at most 1%"] = all(
        as_float(row, "fixed_time_overshoot_sec")
        <= 0.01 * as_float(row, "fixed_time_target_sec")
        for row in summary_rows
    )
    checks["Every run has positive throughput and finite EER"] = all(
        as_float(row, "throughput") > 0.0
        and math.isfinite(as_float(row, "energy_efficiency_wh_per_sku"))
        for row in summary_rows
    )
    signatures: dict[int, set[str]] = defaultdict(set)
    for row in summary_rows:
        signatures[int(float(row["seed"]))].add(row["paired_arrival_signature"])
    checks["All methods receive identical exogenous task streams per seed"] = all(
        len(items) == 1 for items in signatures.values()
    )
    checks["Learned planners remain behaviorally active"] = proposal_difference_rate(trace_rows) > 0.0
    checks["Full and data-only controllers execute different traces"] = (
        executed_model_traces_differ(trace_rows)
    )

    metric_columns = {
        "UPH": "uph",
        "EER": "energy_efficiency_wh_per_sku",
        "FDE": "fleet_distribution_entropy",
        "Empty": "empty_running_ratio",
        "Wait": "avg_task_wait_time",
        "Conflicts": "conflict_count",
        "Blocking": "blocking_onset_count",
        "ChargeQueue": "charge_queue_time_sec",
        "OutOfBattery": "out_of_battery_rate",
        "Timeout": "timeout_rate",
        "DecisionP95": "p95_decision_compute_sec",
        "Override": "mean_world_model_override_accepted",
    }
    values: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (seed, method), row in indexed.items():
        for metric, column in metric_columns.items():
            values[method][metric][seed] = as_float(row, column)

    bootstrap_specs = (
        ("Full vs PI-only", FULL, PI_ONLY, "UPH"),
        ("Full vs PI-only", FULL, PI_ONLY, "EER"),
        ("Full vs PI-only", FULL, PI_ONLY, "Wait"),
        ("Full vs data-only", FULL, DATA_ONLY, "EER"),
    )
    bootstrap_rows: list[dict[str, Any]] = []
    for index, (comparison, left, right, metric) in enumerate(bootstrap_specs):
        bootstrap_rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "scale": "relative_change",
                **paired_bootstrap(
                    values[left][metric],
                    values[right][metric],
                    args.bootstrap_replicates,
                    args.bootstrap_seed + index,
                    relative=True,
                ),
            }
        )
    by_bootstrap = {
        (row["comparison"], row["metric"]): row for row in bootstrap_rows
    }
    aggregate_uph = aggregate_mean_relative_change(values[FULL]["UPH"], values[PI_ONLY]["UPH"])
    aggregate_eer_pi = aggregate_mean_relative_change(values[FULL]["EER"], values[PI_ONLY]["EER"])
    aggregate_eer_data = aggregate_mean_relative_change(values[FULL]["EER"], values[DATA_ONLY]["EER"])
    aggregate_wait = aggregate_mean_relative_change(values[FULL]["Wait"], values[PI_ONLY]["Wait"])
    full_p95 = mean(values[FULL]["DecisionP95"].values())
    full_override = mean(values[FULL]["Override"].values())

    stability_checks = {
        "Full aggregate UPH remains at least 95% of PI-only": aggregate_uph >= -0.05,
        "Full aggregate EER remains below PI-only": aggregate_eer_pi < 0.0,
        "Full aggregate EER remains below data-only": aggregate_eer_data < 0.0,
        "Full-vs-PI EER 95% upper bound excludes degradation above 1%": by_bootstrap[("Full vs PI-only", "EER")]["ci_high"] < 0.01,
        "Full aggregate waiting time is no more than 110% of PI-only": aggregate_wait <= 0.10,
        "Full conflicts are no greater than PI-only": mean(values[FULL]["Conflicts"].values()) <= mean(values[PI_ONLY]["Conflicts"].values()),
        "Full blocking onsets are no greater than PI-only": mean(values[FULL]["Blocking"].values()) <= mean(values[PI_ONLY]["Blocking"].values()),
        "Full out-of-battery rate is no greater than PI-only": mean(values[FULL]["OutOfBattery"].values()) <= mean(values[PI_ONLY]["OutOfBattery"].values()),
        "Full timeout rate is no greater than PI-only": mean(values[FULL]["Timeout"].values()) <= mean(values[PI_ONLY]["Timeout"].values()),
        "Full P95 decision time remains at most 2 seconds": full_p95 <= 2.0,
        "Full accepted override remains active and bounded (1%-50%)": 0.01 <= full_override <= 0.50,
    }
    checks.update(stability_checks)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["method"]].append(row)
    lines = [
        f"# N1 frozen {args.hours:g} h long-horizon stability audit",
        "",
        "The long-horizon experiment uses the independently confirmed checkpoints and frozen bounded evidence gate on ten new paired environment seeds.",
        "",
        "| Method | UPH | EER | FDE | Empty ratio | Wait (s) | Charge queue (s) | Conflicts | Blocking | OOB | Timeout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        rows = grouped[method]
        lines.append(
            f"| {method} | {mean(as_float(row, 'uph') for row in rows):.3f} | "
            f"{mean(as_float(row, 'energy_efficiency_wh_per_sku') for row in rows):.3f} | "
            f"{mean(as_float(row, 'fleet_distribution_entropy') for row in rows):.4f} | "
            f"{mean(as_float(row, 'empty_running_ratio') for row in rows):.4f} | "
            f"{mean(as_float(row, 'avg_task_wait_time') for row in rows):.2f} | "
            f"{mean(as_float(row, 'charge_queue_time_sec') for row in rows):.2f} | "
            f"{mean(as_float(row, 'conflict_count') for row in rows):.3f} | "
            f"{mean(as_float(row, 'blocking_onset_count') for row in rows):.3f} | "
            f"{mean(as_float(row, 'out_of_battery_rate') for row in rows):.4f} | "
            f"{mean(as_float(row, 'timeout_rate') for row in rows):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired seed bootstrap",
            "",
            "| Comparison | Metric | Mean relative change | 95% CI | P(delta >= 0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap_rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['delta_mean']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['probability_nonnegative']:.4f} |"
        )
    lines.extend(["", "## Frozen protocol and stability checks", ""])
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
    passed = all(checks.values())
    lines.extend(
        [
            "",
            f"Long-horizon stability passed: **{'YES' if passed else 'NO'}**.",
            f"Proceed to the next horizon: **{'YES' if passed else 'NO'}**.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (result_dir / "long_horizon_audit.md").write_text(report, encoding="utf-8")
    write_csv(result_dir / "paired_bootstrap.csv", bootstrap_rows)
    (result_dir / "long_horizon_status.json").write_text(
        json.dumps({"hours": args.hours, "passed": passed, "checks": checks}, indent=2),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main(build_parser().parse_args())
