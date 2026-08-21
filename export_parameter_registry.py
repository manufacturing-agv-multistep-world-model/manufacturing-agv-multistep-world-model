from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from jms_parameter_registry import (
    GRAPH_MAPPO_DEFAULTS,
    MPC_UTILITY_WEIGHTS,
    WORLD_MODEL_DEFAULTS,
    parameter_table_rows,
)


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export JMS-interpretable parameter tables.")
    parser.add_argument("--output-dir", default="docs/generated_parameter_registry")
    return parser


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dict_rows(category: str, values: Dict[str, Any], unit: str = "") -> List[Dict[str, Any]]:
    return [
        {
            "category": category,
            "name": key,
            "value": value,
            "unit": unit,
            "rationale": "Algorithm default registered for reproducible JMS experiments.",
        }
        for key, value in values.items()
    ]


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# JMS-Interpretable Parameter Registry",
        "",
        "All default parameters used by the paper-facing simulator and learning scripts are centralized here.",
        "Sensitivity values are listed where the parameter is part of the planned robustness analysis.",
        "",
        "| category | name | value | unit | rationale | source | sensitivity values |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('category', '')} | "
            f"{row.get('name', '')} | "
            f"{row.get('value', '')} | "
            f"{row.get('unit', '')} | "
            f"{row.get('rationale', '')} | "
            f"{row.get('source', '')} | "
            f"{row.get('sensitivity_values', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    env_rows = parameter_table_rows()
    algorithm_rows = (
        _dict_rows("graph_mappo_training", GRAPH_MAPPO_DEFAULTS)
        + _dict_rows("world_model_training", WORLD_MODEL_DEFAULTS)
        + _dict_rows("mpc_utility", MPC_UTILITY_WEIGHTS, unit="utility weight")
    )
    write_csv(output_dir / "environment_parameter_registry.csv", env_rows)
    write_csv(output_dir / "algorithm_parameter_registry.csv", algorithm_rows)
    write_markdown(output_dir / "environment_parameter_registry.md", env_rows)
    write_csv(output_dir / "all_parameter_registry.csv", env_rows + algorithm_rows)
    print(f"Parameter registry exported to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
