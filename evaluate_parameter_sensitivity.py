from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from run_experiments import (
    ROOT,
    load_graph_mappo_model,
    load_world_model,
    parse_fixed_time_hours,
    run_episode,
    write_csv,
    write_markdown_summary,
)


SENSITIVITY_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "speed": {
        "config_key": "speed_max_mps",
        "label": "Speed limit",
        "unit": "m/s",
        "levels": [
            ("-20%", 0.96, 0.8),
            ("-10%", 1.08, 0.9),
            ("baseline", 1.20, 1.0),
            ("+10%", 1.32, 1.1),
            ("+20%", 1.44, 1.2),
        ],
    },
    "acceleration": {
        "config_key": "acceleration_mps2",
        "label": "Acceleration",
        "unit": "m/s^2",
        "levels": [
            ("-20%", 0.40, 0.8),
            ("-10%", 0.45, 0.9),
            ("baseline", 0.50, 1.0),
            ("+10%", 0.55, 1.1),
            ("+20%", 0.60, 1.2),
        ],
    },
    "loaded_energy": {
        "config_key": "loaded_energy_factor",
        "label": "Loaded-energy factor",
        "unit": "ratio",
        "levels": [
            ("-20%", 1.240, 0.8),
            ("-10%", 1.395, 0.9),
            ("baseline", 1.550, 1.0),
            ("+10%", 1.705, 1.1),
            ("+20%", 1.860, 1.2),
        ],
    },
    "charge_rate": {
        "config_key": "charge_soc_per_min",
        "label": "Charging rate",
        "unit": "%SOC/min",
        "levels": [
            ("-20%", 1.60, 0.8),
            ("-10%", 1.80, 0.9),
            ("baseline", 2.00, 1.0),
            ("+10%", 2.20, 1.1),
            ("+20%", 2.40, 1.2),
        ],
    },
    "arrival_rate": {
        "config_key": "arrival_rate_multiplier",
        "label": "Arrival-rate multiplier",
        "unit": "x",
        "levels": [
            ("-20%", 0.80, 0.8),
            ("-10%", 0.90, 0.9),
            ("baseline", 1.00, 1.0),
            ("+10%", 1.10, 1.1),
            ("+20%", 1.20, 1.2),
        ],
    },
}


METHODS = [
    ("Nearest", "nearest", "heuristic"),
    ("DT-aware", "dt_aware", "heuristic"),
    ("PI-GWM-MPC-G", "dt_aware", "world_model_guarded"),
    ("PI-GWM-GMAPPO", "dt_aware", "graph_mappo"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-physical-time parameter sensitivity experiments for the JMS AGV DT study."
    )
    parser.add_argument("--model-path", default="world_model_runs/pi_gwm_interpretable_v2_energy_calibrated/physics_graph_world_model.pt")
    parser.add_argument("--graph-policy-path", default="route_a_graph_mappo_runs/gmappo_interpretable_v3_energy_guarded/graph_mappo_policy.pt")
    parser.add_argument("--output-dir", default="experiment_results/parameter_sensitivity_f1_f5_v1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fixed-time-hours", default="8")
    parser.add_argument(
        "--capacity-mode",
        choices=["baseline", "stress"],
        default="baseline",
        help="Use baseline for the primary JMS evidence chain; stress is a separate robustness scenario.",
    )
    parser.add_argument(
        "--parameters",
        default="speed,acceleration,loaded_energy,charge_rate,arrival_rate",
        help="Comma-separated parameter keys to evaluate.",
    )
    parser.add_argument(
        "--methods",
        default="Nearest,DT-aware,PI-GWM-MPC-G,PI-GWM-GMAPPO",
        help="Comma-separated method labels to evaluate.",
    )
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke-test matrix.")
    parser.add_argument("--write-trace", action="store_true", help="Write trace.csv and attention_samples.csv.")
    return parser


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    horizons = parse_fixed_time_hours(args.fixed_time_hours)
    parameter_keys = _parse_list(args.parameters)
    method_names = set(_parse_list(args.methods))
    unknown = [key for key in parameter_keys if key not in SENSITIVITY_PARAMETERS]
    if unknown:
        raise ValueError(f"Unknown sensitivity parameters: {unknown}")
    selected_methods = [method for method in METHODS if method[0] in method_names]
    if not selected_methods:
        raise ValueError("No valid methods selected for sensitivity analysis.")

    if args.quick:
        horizons = horizons[:1]
        parameter_keys = parameter_keys[:1]
        selected_methods = selected_methods[:2]

    specs: List[Dict[str, Any]] = []
    for hours in horizons:
        for parameter_key in parameter_keys:
            meta = SENSITIVITY_PARAMETERS[parameter_key]
            levels = meta["levels"]
            if args.quick:
                levels = levels[:2]
            for level_label, value, multiplier in levels:
                for method_label, dispatch_rule, policy_override in selected_methods:
                    specs.append(
                        {
                            "experiment": "F_parameter_sensitivity_fixed_time",
                            "method": method_label,
                            "env_variant": "full",
                            "execution_env_variant": "full",
                            "policy_variant": "full",
                            "reward_mode": "hybrid",
                            "scenario": "rush",
                            "dispatch_rule": dispatch_rule,
                            "capacity_mode": args.capacity_mode,
                            "agv_count": 3,
                            "policy_override": policy_override,
                            "fixed_time_target_h": float(hours),
                            "fixed_time_target_sec": float(hours) * 3600.0,
                            "max_released_jobs": None,
                            "sensitivity_parameter": parameter_key,
                            "sensitivity_label": meta["label"],
                            "sensitivity_unit": meta["unit"],
                            "sensitivity_level": level_label,
                            "sensitivity_value": float(value),
                            "sensitivity_multiplier": float(multiplier),
                            # Change exactly one physical factor per sensitivity arm.
                            "config_overrides": {meta["config_key"]: float(value)},
                        }
                    )
    return specs


def _needs_policy(specs: Iterable[Dict[str, Any]], policy_name: str) -> bool:
    return any(spec.get("policy_override") == policy_name for spec in specs)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)

    models: Dict[str, Any] = {}
    if _needs_policy(specs, "world_model_guarded") or _needs_policy(specs, "world_model"):
        models["world_model"] = load_world_model(str(_resolve(args.model_path)))
    if _needs_policy(specs, "graph_mappo"):
        models["graph_mappo"] = load_graph_mappo_model(str(_resolve(args.graph_policy_path)))

    summaries: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []
    total_runs = len(specs) * args.episodes
    run_index = 0

    for spec in specs:
        for episode in range(args.episodes):
            seed = args.seed + episode
            run_index += 1
            summary, trace, attention = run_episode(
                spec=spec,
                episode_id=episode,
                seed=seed,
                max_steps=args.max_steps,
                policy="heuristic",
                models=models,
            )
            summaries.append(summary)
            if args.write_trace:
                traces.extend(trace)
                attention_rows.extend(attention)
            print(
                f"[{run_index}/{total_runs}] {spec['sensitivity_parameter']}={spec['sensitivity_level']} "
                f"({spec['sensitivity_value']} {spec['sensitivity_unit']}) | {spec['method']} | "
                f"{spec['fixed_time_target_h']:.1f}h | seed={seed} | "
                f"UPH={float(summary['uph']):.2f} | throughput={float(summary['throughput']):.0f}"
            )

    write_csv(output_dir / "summary.csv", summaries)
    if args.write_trace:
        write_csv(output_dir / "trace.csv", traces)
        write_csv(output_dir / "attention_samples.csv", attention_rows)
    write_markdown_summary(output_dir / "summary.md", summaries)
    print(f"Parameter sensitivity data saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
