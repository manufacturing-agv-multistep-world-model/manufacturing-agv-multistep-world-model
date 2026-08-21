from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from agv_case_env import AGV_A_Charge_Env
from run_experiments import heuristic_action


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained or heuristic AGV scheduling policy.")
    parser.add_argument("--model-path", default="models/agv_ppo_gat_full_dt_marl_hybrid_stress_steady_3agv_seed42.zip")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--agv-count", type=int, default=3)
    parser.add_argument("--env-variant", choices=["ideal", "kinematics", "full"], default="full")
    parser.add_argument("--reward-mode", choices=["individual", "global", "hybrid"], default="hybrid")
    parser.add_argument("--scenario", choices=["steady", "rush"], default="rush")
    parser.add_argument("--dispatch-rule", choices=["fcfs", "nearest", "priority", "dt_marl"], default="dt_marl")
    parser.add_argument("--capacity-mode", choices=["baseline", "stress"], default="stress")
    parser.add_argument("--output", default="experiment_results/simulation_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def maybe_load_model(model_path: str):
    if not os.path.exists(model_path):
        print("No trained model found. Falling back to heuristic policy for environment validation.")
        return None
    from stable_baselines3 import PPO

    print(f"Loading trained model from {model_path}")
    return PPO.load(model_path)


def run_drive_diagnostic(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = AGV_A_Charge_Env(
        agv_count=args.agv_count,
        env_variant=args.env_variant,
        reward_mode=args.reward_mode,
        scenario=args.scenario,
        dispatch_rule=args.dispatch_rule,
        capacity_mode=args.capacity_mode,
        max_steps=args.steps,
        seed=args.seed,
    )
    model = maybe_load_model(args.model_path)
    obs, _ = env.reset(seed=args.seed)
    rows = []
    total_reward = 0.0

    print("Starting AGV digital-twin evaluation...")
    for step in range(args.steps):
        if model is None:
            action = heuristic_action(env)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        metrics = info["metrics"]
        rows.append(
            {
                "step": step,
                "reward": reward,
                "episode_reward": total_reward,
                "real_time_sec": metrics["real_time_sec"],
                "uph": metrics["uph"],
                "throughput": metrics["throughput"],
                "drt": metrics["deadlock_resolution_time"],
                "eer_wh_per_sku": metrics["energy_efficiency_wh_per_sku"],
                "fde": metrics["fleet_distribution_entropy"],
                "empty_running_ratio": metrics["empty_running_ratio"],
                "agv_utilization": metrics["agv_utilization"],
                "avg_task_wait_time": metrics["avg_task_wait_time"],
                "deadlock_count": metrics["deadlock_count"],
                "conflict_count": metrics["conflict_count"],
                "blocked_count": metrics["blocked_count"],
                "positions": "|".join(str(p) for p in info["positions"]),
                "position_names": "|".join(info["position_names"]),
                "phases": "|".join(info["phases"]),
                "batteries": "|".join(f"{b:.2f}" for b in info["batteries"]),
            }
        )

        if step % 50 == 0:
            print(
                f"step={step} throughput={metrics['throughput']:.0f} "
                f"UPH={metrics['uph']:.2f} FDE={metrics['fleet_distribution_entropy']:.3f}"
            )

        if terminated or truncated:
            break

    if rows:
        with output_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = env.summary()
    print("Evaluation complete.")
    print(
        f"UPH={summary['uph']:.2f}, DRT={summary['deadlock_resolution_time']:.2f}s, "
        f"EER={summary['energy_efficiency_wh_per_sku']:.2f}Wh/SKU, "
        f"FDE={summary['fleet_distribution_entropy']:.3f}, "
        f"Empty={summary['empty_running_ratio']:.3f}"
    )
    print(f"Trace saved to {output_path.resolve()}")


if __name__ == "__main__":
    run_drive_diagnostic(build_parser().parse_args())
