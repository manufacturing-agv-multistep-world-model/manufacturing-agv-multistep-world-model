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


DEVELOPMENT_SEEDS = list(range(37001, 37006))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit energy-neutral 4 h development")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--one-hour-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=40117)
    return parser


def main(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir)
    one_hour_dir = Path(args.one_hour_dir)
    summary_rows = load_csv(result_dir / "summary.csv")
    trace_rows = load_csv(result_dir / "trace.csv")
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    one_hour_manifest = json.loads(
        (one_hour_dir / "run_manifest.json").read_text(encoding="utf-8")
    )

    expected = {
        "phase": "development",
        "hours": 4.0,
        "env_seeds": DEVELOPMENT_SEEDS,
        "model_seeds": [42, 43, 44],
        "control_mode": "ensemble",
        "minimum_ensemble_agreement": 2,
        "scenario": "rush",
        "capacity_mode": "baseline",
        "planning_horizon": 3,
        "beam_width": 8,
        "risk_gate": 0.75,
        "override_mode": "energy_neutral_gated",
    }
    checks: dict[str, bool] = {
        f"Frozen development configuration: {key}": manifest.get(key) == value
        for key, value in expected.items()
    }
    comparable = (
        "env_seeds",
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
    checks["Four-hour settings match the one-hour development version"] = all(
        manifest.get(key) == one_hour_manifest.get(key) for key in comparable
    )
    checks["Four-hour checkpoints match one-hour development"] = (
        checkpoint_hashes(manifest) == checkpoint_hashes(one_hour_manifest)
    )

    indexed: dict[tuple[int, str], dict[str, str]] = {}
    duplicate = False
    for row in summary_rows:
        key = (int(float(row["seed"])), row["method"])
        duplicate = duplicate or key in indexed
        indexed[key] = row
    expected_keys = {
        (seed, method) for seed in DEVELOPMENT_SEEDS for method in METHODS
    }
    checks["Exactly one run exists for each seed and method"] = (
        not duplicate and set(indexed) == expected_keys
    )
    if set(indexed) != expected_keys:
        raise ValueError("Incomplete 4 h development result matrix")
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
    checks["Learned planner proposals remain distinct"] = proposal_difference_rate(trace_rows) > 0.0
    checks["Full and data-only execution traces differ"] = executed_model_traces_differ(trace_rows)

    columns = {
        "UPH": "uph",
        "EER": "energy_efficiency_wh_per_sku",
        "Wait": "avg_task_wait_time",
        "Conflicts": "conflict_count",
        "Blocking": "blocking_onset_count",
        "OOB": "out_of_battery_rate",
        "Timeout": "timeout_rate",
        "P95": "p95_decision_compute_sec",
        "Override": "mean_world_model_override_accepted",
    }
    values: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (seed, method), row in indexed.items():
        for metric, column in columns.items():
            values[method][metric][seed] = as_float(row, column)

    specs = (
        ("Full vs PI-only", FULL, PI_ONLY, "UPH"),
        ("Full vs PI-only", FULL, PI_ONLY, "EER"),
        ("Full vs PI-only", FULL, PI_ONLY, "Wait"),
        ("Full vs data-only", FULL, DATA_ONLY, "EER"),
    )
    bootstrap_rows: list[dict[str, Any]] = []
    for index, (comparison, left, right, metric) in enumerate(specs):
        bootstrap_rows.append(
            {
                "comparison": comparison,
                "metric": metric,
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
    uph_change = aggregate_mean_relative_change(values[FULL]["UPH"], values[PI_ONLY]["UPH"])
    eer_pi_change = aggregate_mean_relative_change(values[FULL]["EER"], values[PI_ONLY]["EER"])
    eer_data_change = aggregate_mean_relative_change(values[FULL]["EER"], values[DATA_ONLY]["EER"])
    wait_change = aggregate_mean_relative_change(values[FULL]["Wait"], values[PI_ONLY]["Wait"])
    checks.update(
        {
            "Full aggregate UPH is at least 95% of PI-only": uph_change >= -0.05,
            "Full aggregate EER is below PI-only": eer_pi_change < 0.0,
            "Full aggregate EER is below data-only": eer_data_change < 0.0,
            "Full-vs-PI EER interval excludes degradation above 1%": by_bootstrap[("Full vs PI-only", "EER")]["ci_high"] < 0.01,
            "Full aggregate waiting is no more than 110% of PI-only": wait_change <= 0.10,
            "Full conflicts are no greater than PI-only": mean(values[FULL]["Conflicts"].values()) <= mean(values[PI_ONLY]["Conflicts"].values()),
            "Full blocking is no greater than PI-only": mean(values[FULL]["Blocking"].values()) <= mean(values[PI_ONLY]["Blocking"].values()),
            "Full OOB and timeout are no greater than PI-only": mean(values[FULL]["OOB"].values()) <= mean(values[PI_ONLY]["OOB"].values()) and mean(values[FULL]["Timeout"].values()) <= mean(values[PI_ONLY]["Timeout"].values()),
            "Full P95 decision time is at most 2 seconds": mean(values[FULL]["P95"].values()) <= 2.0,
            "Full override remains active and bounded (1%-50%)": 0.01 <= mean(values[FULL]["Override"].values()) <= 0.50,
        }
    )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["method"]].append(row)
    lines = [
        "# N1 energy-neutral gate 4 h development audit",
        "",
        "| Method | UPH | EER | Wait (s) | Conflicts | Blocking | OOB | Timeout | P95 decision | Override |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        rows = grouped[method]
        lines.append(
            f"| {method} | {mean(as_float(row, 'uph') for row in rows):.3f} | "
            f"{mean(as_float(row, 'energy_efficiency_wh_per_sku') for row in rows):.3f} | "
            f"{mean(as_float(row, 'avg_task_wait_time') for row in rows):.2f} | "
            f"{mean(as_float(row, 'conflict_count') for row in rows):.3f} | "
            f"{mean(as_float(row, 'blocking_onset_count') for row in rows):.3f} | "
            f"{mean(as_float(row, 'out_of_battery_rate') for row in rows):.4f} | "
            f"{mean(as_float(row, 'timeout_rate') for row in rows):.4f} | "
            f"{mean(as_float(row, 'p95_decision_compute_sec') for row in rows):.5f} | "
            f"{mean(as_float(row, 'mean_world_model_override_accepted') for row in rows):.4f} |"
        )
    lines.extend(["", "## Paired seed bootstrap", "", "| Comparison | Metric | Mean relative change | 95% CI |", "|---|---|---:|---:|"])
    for row in bootstrap_rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['delta_mean']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        )
    lines.extend(["", "## Development continuation checks", ""])
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
    passed = all(checks.values())
    lines.extend(["", f"Proceed to a new frozen confirmation design: **{'YES' if passed else 'NO'}**."])
    report = "\n".join(lines) + "\n"
    (result_dir / "energy_neutral_4h_development_audit.md").write_text(report, encoding="utf-8")
    write_csv(result_dir / "paired_bootstrap.csv", bootstrap_rows)
    (result_dir / "development_status.json").write_text(
        json.dumps({"passed": passed, "checks": checks}, indent=2), encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main(build_parser().parse_args())
