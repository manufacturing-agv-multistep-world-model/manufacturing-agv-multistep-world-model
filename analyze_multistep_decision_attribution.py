from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METHODS = (
    "DT-aware",
    "PI-only guard",
    "Data-only graph MPC",
    "Full V11 physics-graph MPC",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit N1 decision-attribution results")
    parser.add_argument("--result-dir", required=True)
    return parser


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def proposal_difference_rate(trace_rows: list[dict[str, str]]) -> float:
    eligible = [row for row in trace_rows if row["checkpoint_condition"] in {"full", "data_only"}]
    if not eligible:
        return 0.0
    changed = 0
    for row in eligible:
        planned = row.get("world_model_raw_planned_actions", "")
        baseline = row.get("world_model_baseline_actions", "")
        changed += bool(planned and baseline and planned != baseline)
    return changed / len(eligible)


def executed_model_traces_differ(trace_rows: list[dict[str, str]]) -> bool:
    actions: dict[str, list[str]] = defaultdict(list)
    for row in trace_rows:
        condition = row["checkpoint_condition"]
        if condition in {"full", "data_only"}:
            actions[condition].append(row.get("executed_actions", ""))
    return actions["full"] != actions["data_only"]


def main(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir)
    summary_rows = load_csv(result_dir / "summary.csv")
    trace_rows = load_csv(result_dir / "trace.csv")
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))

    model_multiplier = (
        1 if manifest.get("control_mode", "single") == "ensemble" else len(manifest["model_seeds"])
    )
    expected = 2 * len(manifest["env_seeds"]) + 2 * len(
        manifest["env_seeds"]
    ) * model_multiplier
    checks: dict[str, bool] = {
        "Expected number of complete method runs": len(summary_rows) == expected,
        "Every run reaches the fixed physical horizon": all(
            as_float(row, "fixed_time_reached") == 1.0 for row in summary_rows
        ),
        "All decision times are finite and nonnegative": all(
            math.isfinite(as_float(row, "mean_decision_compute_sec"))
            and 0.0 <= as_float(row, "mean_decision_compute_sec") < 1.0e6
            for row in summary_rows
        ),
        "Every method completes at least one task in every run": all(
            as_float(row, "throughput") > 0.0 for row in summary_rows
        ),
        "Every reported EER is finite": all(
            math.isfinite(as_float(row, "energy_efficiency_wh_per_sku"))
            for row in summary_rows
        ),
    }
    by_env: dict[int, set[str]] = defaultdict(set)
    for row in summary_rows:
        by_env[int(float(row["seed"]))].add(row["paired_arrival_signature"])
    checks["All paired methods receive identical exogenous task streams"] = all(
        len(signatures) == 1 for signatures in by_env.values()
    )

    checkpoint_rows = manifest["checkpoint_audit"]
    by_model: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in checkpoint_rows:
        by_model[int(row["model_seed"])][row["condition"]] = row
    checks["Full and data-only checkpoints have equal parameterization"] = all(
        pair["full"]["parameter_count"] == pair["data_only"]["parameter_count"]
        and pair["full"]["state_shape_signature"]
        == pair["data_only"]["state_shape_signature"]
        and pair["full"]["data_seed"] == pair["data_only"]["data_seed"]
        for pair in by_model.values()
    )

    proposal_rate = proposal_difference_rate(trace_rows)
    accepted_rows = [
        row for row in summary_rows if row["checkpoint_condition"] in {"full", "data_only"}
    ]
    accepted_rate = mean(
        as_float(row, "mean_world_model_override_accepted") for row in accepted_rows
    )
    checks["Learned planners propose at least one action different from their baseline"] = (
        proposal_rate > 0.0
    )
    checks["Full and data-only learned controllers execute different action traces"] = (
        executed_model_traces_differ(trace_rows)
    )

    metric_columns = {
        "UPH": "uph",
        "EER (Wh/SKU)": "energy_efficiency_wh_per_sku",
        "Empty ratio": "empty_running_ratio",
        "Wait (s)": "avg_task_wait_time",
        "Conflicts": "conflict_count",
        "Blocking onsets": "blocking_onset_count",
        "Decision mean (s)": "mean_decision_compute_sec",
        "Decision p95 (s)": "p95_decision_compute_sec",
        "Accepted override rate": "mean_world_model_override_accepted",
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["method"]].append(row)
    means = {
        method: {
            label: mean(as_float(row, column) for row in grouped[method])
            for label, column in metric_columns.items()
        }
        for method in METHODS
    }
    full = means["Full V11 physics-graph MPC"]
    pi_only = means["PI-only guard"]
    if manifest["phase"] == "development":
        checks.update(
            {
                "Development Full V11 UPH is at least 90% of PI-only": full["UPH"]
                >= 0.90 * pi_only["UPH"],
                "Development Full V11 EER is no more than 110% of PI-only": full[
                    "EER (Wh/SKU)"
                ]
                <= 1.10 * pi_only["EER (Wh/SKU)"],
                "Development Full V11 accepted override rate is active but below 90%": 0.01
                <= full["Accepted override rate"]
                <= 0.90,
                "Development Full V11 P95 decision time is at most 2 seconds": full[
                    "Decision p95 (s)"
                ]
                <= 2.0,
            }
        )

    lines = [
        "# N1 multistep decision-attribution audit",
        "",
        f"Phase: {manifest['phase']}; physical horizon: {manifest['hours']} h; "
        f"environment seeds: {manifest['env_seeds']}; model seeds: {manifest['model_seeds']}; "
        f"control mode: {manifest.get('control_mode', 'single')}.",
        "",
        "| Method | UPH | EER | Empty ratio | Wait (s) | Conflicts | Blocking | Mean decision (s) | P95 decision (s) | Accepted override |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = means[method]
        lines.append(
            f"| {method} | {values['UPH']:.3f} | {values['EER (Wh/SKU)']:.3f} | "
            f"{values['Empty ratio']:.4f} | {values['Wait (s)']:.2f} | "
            f"{values['Conflicts']:.3f} | {values['Blocking onsets']:.3f} | "
            f"{values['Decision mean (s)']:.5f} | {values['Decision p95 (s)']:.5f} | "
            f"{values['Accepted override rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Learned-planner proposal difference rate: {proposal_rate:.4f}.",
            f"Mean accepted learned override rate: {accepted_rate:.4f}.",
            f"Maximum fixed-horizon overshoot: {max(as_float(row, 'fixed_time_overshoot_sec') for row in summary_rows):.3f} s.",
            "",
            "## Protocol-integrity checks",
            "",
        ]
    )
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
    passed = all(checks.values())
    next_stage = "development run" if manifest["phase"] == "smoke" else "frozen confirmation design"
    lines.extend(
        [
            "",
            f"Proceed to the {next_stage}: **{'YES' if passed else 'NO'}**.",
            "",
            "Smoke/development results are implementation evidence only and must not be used as final performance claims.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (result_dir / "attribution_audit.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main(build_parser().parse_args())
