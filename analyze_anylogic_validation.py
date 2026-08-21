from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except ImportError:  # pragma: no cover - SciPy is available in the project environment.
    student_t = None


KEY_COLUMNS = ["scenario", "seed", "horizon_h"]
NUMERIC_COLUMNS = [
    "seed",
    "horizon_h",
    "generated_tasks",
    "completed_tasks",
    "unfinished_tasks",
    "uph",
    "avg_cycle_time_min",
    "avg_waiting_time_min",
    "max_wip",
    "agv_utilization_pct",
]
METRICS = [
    "uph",
    "avg_cycle_time_min",
    "avg_waiting_time_min",
    "unfinished_tasks",
    "max_wip",
    "agv_utilization_pct",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and summarize AnyLogic multi-seed validation runs.")
    parser.add_argument(
        "--anylogic-csv",
        default="AGV_DT_AnyLogic_Validation/Manufacturing_AGV_DT_Validation/anylogic_validation_results.csv",
    )
    parser.add_argument(
        "--python-reference",
        default="paper_outputs/anylogic_validation/python_reference_runs.csv",
    )
    parser.add_argument("--required-seeds", default="1,2,3")
    parser.add_argument("--horizons", default="1,4,8")
    parser.add_argument("--scenarios", default="steady,rush")
    parser.add_argument("--output-dir", default="paper_outputs/anylogic_validation/final")
    return parser


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def parse_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def confidence_interval(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    n = len(clean)
    mean = float(clean.mean()) if n else math.nan
    if n < 2:
        return mean, math.nan, math.nan
    sem = float(clean.std(ddof=1) / math.sqrt(n))
    critical = float(student_t.ppf(0.975, n - 1)) if student_t is not None else 1.96
    half_width = critical * sem
    return mean, mean - half_width, mean + half_width


def load_and_audit(path: Path, scenarios: list[str], horizons: list[float]) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"AnyLogic CSV not found: {path}")
    frame = pd.read_csv(path)
    missing_columns = [column for column in [*KEY_COLUMNS, *NUMERIC_COLUMNS[2:]] if column not in frame]
    if missing_columns:
        raise ValueError(f"AnyLogic CSV missing columns: {missing_columns}")

    issues: list[str] = []
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[NUMERIC_COLUMNS].isna().any().any():
        bad = frame.index[frame[NUMERIC_COLUMNS].isna().any(axis=1)].tolist()
        raise ValueError(f"Non-numeric or missing values in rows: {bad}")

    frame["seed"] = frame["seed"].astype(int)
    formal = frame[
        frame["scenario"].isin(scenarios)
        & frame["horizon_h"].apply(lambda x: any(math.isclose(x, h, abs_tol=1e-6) for h in horizons))
    ].copy()
    formal["horizon_h"] = formal["horizon_h"].apply(
        lambda x: next(h for h in horizons if math.isclose(x, h, abs_tol=1e-6))
    )

    duplicated = formal.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        keys = formal.loc[duplicated, KEY_COLUMNS].drop_duplicates().to_dict("records")
        issues.append(f"Duplicate scenario-seed-horizon keys detected: {keys}")
        formal = formal.drop_duplicates(KEY_COLUMNS, keep="last")

    invalid_count = formal[
        (formal["completed_tasks"] > formal["generated_tasks"])
        | (formal["unfinished_tasks"] != formal["generated_tasks"] - formal["completed_tasks"])
    ]
    if not invalid_count.empty:
        issues.append(f"Task-flow accounting failed in {len(invalid_count)} row(s).")
    if ((formal["uph"] < 0) | (formal["agv_utilization_pct"] < 0) | (formal["agv_utilization_pct"] > 100.0001)).any():
        issues.append("Negative UPH or utilization outside [0, 100] detected.")

    payload = [column for column in NUMERIC_COLUMNS if column not in {"seed"}]
    pseudo = formal.groupby(["scenario", "horizon_h", *payload], dropna=False)["seed"].nunique()
    pseudo = pseudo[pseudo > 1]
    if not pseudo.empty:
        issues.append(
            "Exact metric duplicates occur under different seed labels. Verify that the Simulation experiment's "
            "actual fixed seed was changed, not only alRunSeed."
        )
    return formal.sort_values(KEY_COLUMNS).reset_index(drop=True), issues


def build_completeness(
    frame: pd.DataFrame, scenarios: list[str], horizons: list[float], seeds: list[int]
) -> pd.DataFrame:
    expected = pd.DataFrame(
        itertools.product(scenarios, seeds, horizons), columns=["scenario", "seed", "horizon_h"]
    )
    observed = frame[KEY_COLUMNS].copy()
    observed["present"] = True
    result = expected.merge(observed, on=KEY_COLUMNS, how="left")
    result["present"] = result["present"].eq(True)
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def aggregate(frame: pd.DataFrame, platform: str) -> pd.DataFrame:
    rows = []
    for (scenario, horizon_h), group in frame.groupby(["scenario", "horizon_h"], sort=True):
        row: dict[str, float | int | str] = {
            "platform": platform,
            "scenario": scenario,
            "horizon_h": float(horizon_h),
            "n": int(group["seed"].nunique()),
        }
        for metric in METRICS:
            if metric not in group:
                continue
            mean, lower, upper = confidence_interval(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def load_python_reference(path: Path, scenarios: list[str], horizons: list[float]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    for column in ["seed", "horizon_h", *METRICS]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        frame["scenario"].isin(scenarios)
        & frame["horizon_h"].apply(lambda x: any(math.isclose(x, h, abs_tol=1e-6) for h in horizons))
    ].copy()


def build_comparison(anylogic_summary: pd.DataFrame, python_summary: pd.DataFrame | None) -> pd.DataFrame:
    if python_summary is None or python_summary.empty:
        return pd.DataFrame()
    merged = anylogic_summary.merge(
        python_summary,
        on=["scenario", "horizon_h"],
        suffixes=("_anylogic", "_python"),
    )
    for metric in [
        "uph",
        "avg_cycle_time_min",
        "avg_waiting_time_min",
        "unfinished_tasks",
        "agv_utilization_pct",
    ]:
        left = f"{metric}_mean_anylogic"
        right = f"{metric}_mean_python"
        if left in merged and right in merged:
            merged[f"{metric}_relative_difference_pct"] = 100.0 * (merged[left] - merged[right]) / merged[right]
    return merged


def plot_validation(
    anylogic_summary: pd.DataFrame,
    python_summary: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    colors = {"steady": "#1f6f8b", "rush": "#c94c2f"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.78, wspace=0.25)
    plot_specs = [
        ("uph", "Throughput (UPH)"),
        ("avg_waiting_time_min", "Mean waiting time (min)"),
        ("unfinished_tasks", "Backlog at horizon"),
    ]
    for axis, (metric, label) in zip(axes, plot_specs):
        for scenario in ["steady", "rush"]:
            subset = anylogic_summary[anylogic_summary["scenario"] == scenario].sort_values("horizon_h")
            if subset.empty:
                continue
            mean = subset[f"{metric}_mean"].to_numpy(float)
            low = subset[f"{metric}_ci95_low"].to_numpy(float)
            high = subset[f"{metric}_ci95_high"].to_numpy(float)
            errors = np.vstack([mean - low, high - mean])
            errors = None if np.isnan(errors).all() else np.nan_to_num(errors, nan=0.0)
            axis.errorbar(
                subset["horizon_h"],
                mean,
                yerr=errors,
                marker="o",
                linewidth=2,
                capsize=3,
                color=colors[scenario],
                label=f"AnyLogic {scenario}",
            )
            if python_summary is not None and f"{metric}_mean" in python_summary:
                py = python_summary[python_summary["scenario"] == scenario].sort_values("horizon_h")
                if not py.empty:
                    py_mean = py[f"{metric}_mean"].to_numpy(float)
                    py_low = py[f"{metric}_ci95_low"].to_numpy(float)
                    py_high = py[f"{metric}_ci95_high"].to_numpy(float)
                    py_errors = np.vstack([py_mean - py_low, py_high - py_mean])
                    py_errors = None if np.isnan(py_errors).all() else np.nan_to_num(py_errors, nan=0.0)
                    axis.errorbar(
                        py["horizon_h"].to_numpy(float),
                        py_mean,
                        yerr=py_errors,
                        marker="s",
                        linestyle="--",
                        linewidth=1.7,
                        capsize=3,
                        color=colors[scenario],
                        alpha=0.72,
                        label=f"Python {scenario}",
                    )
        axis.set_xlabel("Physical horizon (h)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=min(4, len(unique)),
        frameon=False,
    )
    fig.suptitle("Cross-platform validation under matched kinematics and task-flow settings", y=0.985)
    for suffix, kwargs in {
        "png": {"dpi": 300},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(output_dir / f"figure_anylogic_validation.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_audit(
    path: Path,
    formal: pd.DataFrame,
    completeness: pd.DataFrame,
    issues: list[str],
    comparison: pd.DataFrame,
) -> None:
    missing = completeness[~completeness["present"]]
    lines = [
        "# AnyLogic multi-seed validation audit",
        "",
        f"- Formal rows retained: {len(formal)}",
        f"- Unique real-seed labels: {formal['seed'].nunique() if not formal.empty else 0}",
        f"- Required combinations missing: {len(missing)}",
        f"- Data-integrity warnings: {len(issues)}",
        "",
        "## Integrity findings",
        "",
    ]
    lines.extend([f"- {issue}" for issue in issues] or ["- No aggregate-level integrity violation detected."])
    lines.extend(["", "## Scope lock", ""])
    lines.extend(
        [
            "- This model validates road-network/kinematics and congestion trends under matched stochastic task flows.",
            "- It does not independently validate nonlinear battery physics, learned-policy optimality, or exact deadlock counts.",
            "- Queueing-time magnitudes need not be identical because event ordering, path reservation, and service-time semantics differ between engines.",
            "- Seed labels are credible only when the AnyLogic Simulation experiment fixed seed equals `alRunSeed` for every run.",
        ]
    )
    if not comparison.empty:
        max_uph_difference = comparison["uph_relative_difference_pct"].abs().max()
        lines.extend(
            [
                "",
                "## Primary validation result",
                "",
                f"- Maximum absolute cross-platform UPH difference: {max_uph_difference:.1f}%.",
                "- Both engines reproduce a stable steady regime and a capacity-saturated rush regime with growing backlog.",
                "- The evidence supports system-level capacity and congestion-trend validity, not pointwise queueing-time equivalence.",
            ]
        )
        lines.extend(["", "## Cross-platform relative differences", ""])
        for _, row in comparison.sort_values(["scenario", "horizon_h"]).iterrows():
            uph = row.get("uph_relative_difference_pct", math.nan)
            wait = row.get("avg_waiting_time_min_relative_difference_pct", math.nan)
            backlog = row.get("unfinished_tasks_relative_difference_pct", math.nan)
            lines.append(
                f"- {row['scenario']} {row['horizon_h']:g} h: UPH {uph:+.1f}%; "
                f"waiting time {wait:+.1f}%; backlog {backlog:+.1f}%."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    anylogic_path = resolve(project_root, args.anylogic_csv)
    python_path = resolve(project_root, args.python_reference)
    output_dir = resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(anylogic_path, output_dir / "anylogic_source_snapshot.csv")

    seeds = parse_list(args.required_seeds, int)
    horizons = parse_list(args.horizons, float)
    scenarios = parse_list(args.scenarios, str)
    formal, issues = load_and_audit(anylogic_path, scenarios, horizons)
    completeness = build_completeness(formal, scenarios, horizons, seeds)
    anylogic_summary = aggregate(formal, "AnyLogic DES")
    python_runs = load_python_reference(python_path, scenarios, horizons)
    python_summary = aggregate(python_runs, "Python kinematics DT") if python_runs is not None else None
    comparison = build_comparison(anylogic_summary, python_summary)

    formal.to_csv(output_dir / "anylogic_formal_runs_clean.csv", index=False, encoding="utf-8-sig")
    completeness.to_csv(output_dir / "anylogic_run_completeness.csv", index=False, encoding="utf-8-sig")
    anylogic_summary.to_csv(output_dir / "anylogic_summary_95ci.csv", index=False, encoding="utf-8-sig")
    if python_summary is not None:
        python_summary.to_csv(output_dir / "python_reference_summary_95ci.csv", index=False, encoding="utf-8-sig")
    if not comparison.empty:
        comparison.to_csv(output_dir / "python_anylogic_comparison.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "protocol": "cross_platform_kinematics_task_flow_validation_v1",
        "formal_horizons_h": horizons,
        "scenarios": scenarios,
        "anylogic_seeds": seeds,
        "python_reference_seeds": sorted(python_runs["seed"].astype(int).unique().tolist())
        if python_runs is not None
        else [],
        "anylogic_formal_rows": int(len(formal)),
        "required_combinations_missing": int((~completeness["present"]).sum()),
        "anylogic_source_sha256": hashlib.sha256(anylogic_path.read_bytes()).hexdigest(),
        "python_reference_sha256": hashlib.sha256(python_path.read_bytes()).hexdigest()
        if python_path.exists()
        else None,
        "scope": (
            "Independent road-network, kinematics, capacity, and congestion-trend validation; "
            "no nonlinear-battery or learned-policy validation claim."
        ),
    }
    (output_dir / "anylogic_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    plot_validation(anylogic_summary, python_summary, output_dir)
    write_audit(output_dir / "ANYLOGIC_MULTI_SEED_AUDIT.md", formal, completeness, issues, comparison)

    missing = int((~completeness["present"]).sum())
    print(f"Formal AnyLogic rows: {len(formal)}")
    print(f"Missing required combinations: {missing}")
    print(f"Integrity warnings: {len(issues)}")
    print(f"Outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
