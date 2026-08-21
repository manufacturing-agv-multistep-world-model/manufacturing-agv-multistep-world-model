from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from agv_case_env import AGV_A_Charge_Env
from run_experiments import heuristic_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the kinematics-only Python reference matched to the AnyLogic DES."
    )
    parser.add_argument("--seeds", default="41001,41002,41003,41004,41005,41006,41007,41008,41009,41010")
    parser.add_argument("--horizons", default="1,4,8")
    parser.add_argument("--scenarios", default="steady,rush")
    parser.add_argument("--output", default="paper_outputs/anylogic_validation/python_reference_runs.csv")
    return parser


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_reference(seed: int, horizon_h: float, scenario: str) -> dict[str, float | int | str]:
    target_sec = horizon_h * 3600.0
    max_steps = int(math.ceil(target_sec / 2.0)) + 200
    env = AGV_A_Charge_Env(
        agv_count=3,
        env_variant="kinematics",
        reward_mode="hybrid",
        scenario=scenario,
        dispatch_rule="nearest",
        capacity_mode="baseline",
        arrival_process="poisson",
        initial_backlog_per_template=0,
        max_steps=max_steps,
        seed=seed,
    )
    observation, _ = env.reset(seed=seed)
    del observation

    while env.metrics.total_time_sec < target_sec:
        action = heuristic_action(env)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    summary = env.summary()
    reached = float(summary["real_time_sec"]) >= target_sec
    if not reached:
        raise RuntimeError(
            f"Reference run stopped early: scenario={scenario}, seed={seed}, "
            f"horizon={horizon_h} h, elapsed={summary['real_time_sec']} s"
        )

    return {
        "platform": "Python kinematics DT",
        "scenario": scenario,
        "seed": seed,
        "horizon_h": horizon_h,
        "dispatch_rule": "nearest",
        "capacity_mode": "baseline",
        "released_tasks": int(float(summary["released_jobs"])),
        "completed_tasks": int(float(summary["throughput"])),
        "unfinished_tasks": int(float(summary["active_jobs"])),
        "uph": float(summary["uph"]),
        "avg_cycle_time_min": float(summary["avg_task_cycle_time"]) / 60.0,
        "avg_waiting_time_min": float(summary["avg_task_wait_time"]) / 60.0,
        "agv_utilization_pct": 100.0 * float(summary["agv_utilization"]),
        "elapsed_hours": float(summary["real_time_sec"]) / 3600.0,
    }


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_ints(args.seeds)
    horizons = parse_floats(args.horizons)
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    if not seeds or not horizons or not scenarios:
        raise ValueError("Seeds, horizons, and scenarios must all be non-empty.")
    if any(item not in {"steady", "rush"} for item in scenarios):
        raise ValueError("Scenarios must be steady and/or rush.")

    rows = []
    total = len(seeds) * len(horizons) * len(scenarios)
    for index, (scenario, horizon_h, seed) in enumerate(
        (
            (scenario, horizon_h, seed)
            for scenario in scenarios
            for horizon_h in horizons
            for seed in seeds
        ),
        start=1,
    ):
        row = run_reference(seed, horizon_h, scenario)
        rows.append(row)
        print(
            f"[{index:02d}/{total:02d}] {scenario} {horizon_h:g} h seed={seed}: "
            f"UPH={row['uph']:.3f}"
        )

    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} matched Python reference runs to {output.resolve()}")


if __name__ == "__main__":
    main()
