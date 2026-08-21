from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from analyze_multistep_decision_attribution import (
    METHODS,
    as_float,
    executed_model_traces_differ,
    load_csv,
    proposal_difference_rate,
)


FULL = "Full V11 physics-graph MPC"
PI_ONLY = "PI-only guard"
DATA_ONLY = "Data-only graph MPC"
FROZEN_ENV_SEEDS = list(range(35001, 35016))
FROZEN_MODEL_SEEDS = [42, 43, 44]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the frozen N1 confirmation run")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--development-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=38117)
    parser.add_argument("--report-name", default="confirmation_audit.md")
    return parser


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(
    left: dict[int, float],
    right: dict[int, float],
    replicates: int,
    seed: int,
    *,
    relative: bool,
) -> dict[str, float]:
    seeds = np.asarray(sorted(set(left) & set(right)), dtype=np.int64)
    if len(seeds) != len(left) or len(seeds) != len(right) or len(seeds) == 0:
        raise ValueError("Paired seed sets do not match")
    left_values = np.asarray([left[int(item)] for item in seeds], dtype=float)
    right_values = np.asarray([right[int(item)] for item in seeds], dtype=float)
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("Nonfinite paired metric")
    if relative:
        if np.any(np.abs(right_values) <= 1.0e-12):
            raise ValueError("Relative paired metric has a zero denominator")
        differences = (left_values - right_values) / right_values
    else:
        differences = left_values - right_values
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(seeds), size=(replicates, len(seeds)))
    values = np.mean(differences[sampled], axis=1)
    return {
        "delta_mean": float(np.mean(differences)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_nonnegative": float(np.mean(values >= 0.0)),
        "seed_count": int(len(seeds)),
    }


def aggregate_mean_relative_change(
    left: dict[int, float], right: dict[int, float]
) -> float:
    if set(left) != set(right) or not left:
        raise ValueError("Paired seed sets do not match")
    left_mean = mean(left.values())
    right_mean = mean(right.values())
    if abs(right_mean) <= 1.0e-12:
        raise ValueError("Aggregate relative metric has a zero denominator")
    return (left_mean - right_mean) / right_mean


def checkpoint_hashes(manifest: dict[str, Any]) -> dict[tuple[str, int], str]:
    return {
        (row["condition"], int(row["model_seed"])): row["sha256"]
        for row in manifest["checkpoint_audit"]
    }


def main(args: argparse.Namespace) -> None:
    if args.bootstrap_replicates < 1000:
        raise ValueError("At least 1000 bootstrap replicates are required")
    result_dir = Path(args.result_dir)
    development_dir = Path(args.development_dir)
    summary_rows = load_csv(result_dir / "summary.csv")
    trace_rows = load_csv(result_dir / "trace.csv")
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    development_manifest = json.loads(
        (development_dir / "run_manifest.json").read_text(encoding="utf-8")
    )

    frozen_config = {
        "phase": "confirmation",
        "hours": 1.0,
        "env_seeds": FROZEN_ENV_SEEDS,
        "model_seeds": FROZEN_MODEL_SEEDS,
        "control_mode": "ensemble",
        "minimum_ensemble_agreement": 2,
        "scenario": "rush",
        "capacity_mode": "baseline",
        "planning_horizon": 3,
        "beam_width": 8,
        "risk_gate": 0.75,
        "override_mode": "evidence_gated",
    }
    config_checks = {
        f"Frozen configuration: {key}": manifest.get(key) == expected
        for key, expected in frozen_config.items()
    }
    comparable_fields = (
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
    config_checks["Confirmation and development use identical frozen control settings"] = all(
        manifest.get(field) == development_manifest.get(field)
        for field in comparable_fields
    )
    config_checks["Confirmation checkpoints exactly match development checkpoints"] = (
        checkpoint_hashes(manifest) == checkpoint_hashes(development_manifest)
    )
    config_checks["Confirmation seeds are disjoint from development seeds"] = not (
        set(manifest.get("env_seeds", []))
        & set(development_manifest.get("env_seeds", []))
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
    integrity_checks: dict[str, bool] = {
        "Exactly one complete run exists for every frozen seed and method": (
            not duplicate and set(indexed) == expected_keys
        ),
        "Every run reaches the fixed physical horizon": all(
            as_float(row, "fixed_time_reached") == 1.0 for row in summary_rows
        ),
        "Fixed-horizon overshoot is at most 1%": all(
            as_float(row, "fixed_time_overshoot_sec")
            <= 0.01 * as_float(row, "fixed_time_target_sec")
            for row in summary_rows
        ),
        "Every run completes at least one task with finite EER": all(
            as_float(row, "throughput") > 0.0
            and math.isfinite(as_float(row, "energy_efficiency_wh_per_sku"))
            for row in summary_rows
        ),
    }
    arrival_signatures: dict[int, set[str]] = defaultdict(set)
    for row in summary_rows:
        arrival_signatures[int(float(row["seed"]))].add(row["paired_arrival_signature"])
    integrity_checks["All methods receive identical exogenous task streams per seed"] = all(
        len(values) == 1 for values in arrival_signatures.values()
    )
    integrity_checks["Learned planners propose actions different from the baseline"] = (
        proposal_difference_rate(trace_rows) > 0.0
    )
    integrity_checks["Full and data-only controllers execute different action traces"] = (
        executed_model_traces_differ(trace_rows)
    )

    metric_columns = {
        "UPH": "uph",
        "EER": "energy_efficiency_wh_per_sku",
        "Wait": "avg_task_wait_time",
        "Conflicts": "conflict_count",
        "Blocking": "blocking_onset_count",
        "DecisionP95": "p95_decision_compute_sec",
        "AcceptedOverride": "mean_world_model_override_accepted",
    }
    values: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (seed, method), row in indexed.items():
        for label, column in metric_columns.items():
            values[method][label][seed] = as_float(row, column)

    bootstrap_specs = (
        ("Full vs PI-only", FULL, PI_ONLY, "UPH", True),
        ("Full vs PI-only", FULL, PI_ONLY, "EER", True),
        ("Full vs PI-only", FULL, PI_ONLY, "Wait", True),
        ("Full vs data-only", FULL, DATA_ONLY, "EER", True),
    )
    bootstrap_rows: list[dict[str, Any]] = []
    for index, (comparison, left, right, metric, relative) in enumerate(bootstrap_specs):
        bootstrap_rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "scale": "relative_change" if relative else "absolute_difference",
                **paired_bootstrap(
                    values[left][metric],
                    values[right][metric],
                    args.bootstrap_replicates,
                    args.bootstrap_seed + index,
                    relative=relative,
                ),
            }
        )
    by_bootstrap = {
        (row["comparison"], row["metric"]): row for row in bootstrap_rows
    }
    full_uph = mean(values[FULL]["UPH"].values())
    pi_uph = mean(values[PI_ONLY]["UPH"].values())
    full_wait = mean(values[FULL]["Wait"].values())
    pi_wait = mean(values[PI_ONLY]["Wait"].values())
    full_p95 = mean(values[FULL]["DecisionP95"].values())
    full_override = mean(values[FULL]["AcceptedOverride"].values())
    aggregate_eer_change = aggregate_mean_relative_change(
        values[FULL]["EER"], values[PI_ONLY]["EER"]
    )
    full_eer_pi = by_bootstrap[("Full vs PI-only", "EER")]
    full_eer_data = by_bootstrap[("Full vs data-only", "EER")]

    continuation_checks = {
        "Full mean UPH is at least 95% of PI-only": full_uph >= 0.95 * pi_uph,
        "Full aggregate mean EER is at least 1% below PI-only": aggregate_eer_change
        <= -0.01,
        "Full-vs-PI EER 95% interval is entirely below zero": full_eer_pi["ci_high"] < 0.0,
        "Full-vs-data-only EER 95% interval is entirely below zero": full_eer_data["ci_high"] < 0.0,
        "Full mean waiting time is no more than 105% of PI-only": full_wait <= 1.05 * pi_wait,
        "Full mean P95 decision time is at most 2 seconds": full_p95 <= 2.0,
        "Full accepted override remains active and bounded (1%-50%)": 0.01 <= full_override <= 0.50,
        "Full conflicts are no greater than PI-only": mean(values[FULL]["Conflicts"].values())
        <= mean(values[PI_ONLY]["Conflicts"].values()),
        "Full blocking onsets are no greater than PI-only": mean(values[FULL]["Blocking"].values())
        <= mean(values[PI_ONLY]["Blocking"].values()),
    }

    per_seed_rows = []
    for seed in FROZEN_ENV_SEEDS:
        per_seed_rows.append(
            {
                "seed": seed,
                "full_uph": values[FULL]["UPH"][seed],
                "pi_uph": values[PI_ONLY]["UPH"][seed],
                "full_vs_pi_uph_relative": (
                    values[FULL]["UPH"][seed] - values[PI_ONLY]["UPH"][seed]
                )
                / values[PI_ONLY]["UPH"][seed],
                "full_eer": values[FULL]["EER"][seed],
                "pi_eer": values[PI_ONLY]["EER"][seed],
                "data_only_eer": values[DATA_ONLY]["EER"][seed],
                "full_vs_pi_eer_relative": (
                    values[FULL]["EER"][seed] - values[PI_ONLY]["EER"][seed]
                )
                / values[PI_ONLY]["EER"][seed],
                "full_vs_data_eer_relative": (
                    values[FULL]["EER"][seed] - values[DATA_ONLY]["EER"][seed]
                )
                / values[DATA_ONLY]["EER"][seed],
                "full_wait_sec": values[FULL]["Wait"][seed],
                "pi_wait_sec": values[PI_ONLY]["Wait"][seed],
                "full_p95_decision_sec": values[FULL]["DecisionP95"][seed],
                "full_override_rate": values[FULL]["AcceptedOverride"][seed],
            }
        )
    write_csv(result_dir / "paired_seed_metrics.csv", per_seed_rows)
    write_csv(result_dir / "paired_bootstrap.csv", bootstrap_rows)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["method"]].append(row)
    lines = [
        "# N1 frozen independent confirmation audit",
        "",
        "The bounded evidence gate, model checkpoints, control settings, and 15 environment seeds were frozen before this run.",
        "Analysis version: v2 protocol-implementation correction. The 1% minimum-effect check uses the preregistered ratio of aggregate method means; paired seed ratios remain the basis of the bootstrap interval.",
        "",
        "| Method | UPH | EER (Wh/SKU) | Wait (s) | Conflicts | Blocking | P95 decision (s) | Accepted override |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        rows = grouped[method]
        lines.append(
            f"| {method} | {mean(as_float(row, 'uph') for row in rows):.3f} | "
            f"{mean(as_float(row, 'energy_efficiency_wh_per_sku') for row in rows):.3f} | "
            f"{mean(as_float(row, 'avg_task_wait_time') for row in rows):.2f} | "
            f"{mean(as_float(row, 'conflict_count') for row in rows):.3f} | "
            f"{mean(as_float(row, 'blocking_onset_count') for row in rows):.3f} | "
            f"{mean(as_float(row, 'p95_decision_compute_sec') for row in rows):.5f} | "
            f"{mean(as_float(row, 'mean_world_model_override_accepted') for row in rows):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired seed bootstrap",
            "",
            "Relative change is (Full - comparator) / comparator; negative EER and waiting values are favorable.",
            f"The preregistered aggregate-mean EER change for Full vs PI-only is {aggregate_eer_change:+.4f}.",
            "",
            "| Comparison | Metric | Mean relative change | 95% CI | P(delta >= 0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap_rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['delta_mean']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['probability_nonnegative']:.4f} |"
        )
    all_checks = {**config_checks, **integrity_checks, **continuation_checks}
    lines.extend(["", "## Frozen protocol and continuation checks", ""])
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}" for name, passed in all_checks.items()
    )
    passed = all(all_checks.values())
    lines.extend(
        [
            "",
            f"Independent confirmation passed: **{'YES' if passed else 'NO'}**.",
            f"Proceed to 4 h / 8 h system-level evaluation: **{'YES' if passed else 'NO'}**.",
            "",
            "A failed criterion must be reported as a limitation; the confirmation seed set must not be rerun with retuned thresholds.",
        ]
    )
    report = "\n".join(lines) + "\n"
    report_path = result_dir / args.report_name
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main(build_parser().parse_args())
