from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from agv_case_env import AGV_A_Charge_Env
from physics_graph_world_model import (
    PhysicsInformedGraphWorldModel,
    WorldModelMetadata,
    WorldModelTransitionDataset,
    collect_world_model_transitions,
    save_world_model,
    world_model_loss,
)
from jms_parameter_registry import WORLD_MODEL_DEFAULTS


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a physics-informed graph world model for AGV digital-twin dispatch."
    )
    parser.add_argument("--episodes", type=int, default=WORLD_MODEL_DEFAULTS["episodes"])
    parser.add_argument("--max-steps", type=int, default=WORLD_MODEL_DEFAULTS["max_steps"])
    parser.add_argument("--epochs", type=int, default=WORLD_MODEL_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=WORLD_MODEL_DEFAULTS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=WORLD_MODEL_DEFAULTS["learning_rate"])
    parser.add_argument("--physics-weight", type=float, default=WORLD_MODEL_DEFAULTS["physics_weight"])
    parser.add_argument("--exploration-rate", type=float, default=WORLD_MODEL_DEFAULTS["exploration_rate"])
    parser.add_argument("--agv-count", type=int, default=3)
    parser.add_argument("--env-variant", choices=["ideal", "kinematics", "full"], default="full")
    parser.add_argument("--reward-mode", choices=["individual", "global", "hybrid"], default="hybrid")
    parser.add_argument("--scenario", choices=["steady", "rush"], default="rush")
    parser.add_argument("--dispatch-rule", choices=["fcfs", "nearest", "priority", "dt_aware", "dt_marl"], default="dt_aware")
    parser.add_argument("--capacity-mode", choices=["baseline", "stress"], default="stress")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="world_model_runs/pi_gwm")
    return parser


def select_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return name


def make_env_factory(args: argparse.Namespace):
    def factory(seed: int) -> AGV_A_Charge_Env:
        return AGV_A_Charge_Env(
            agv_count=args.agv_count,
            env_variant=args.env_variant,
            reward_mode=args.reward_mode,
            scenario=args.scenario,
            dispatch_rule=args.dispatch_rule,
            capacity_mode=args.capacity_mode,
            max_steps=args.max_steps,
            seed=seed,
        )

    return factory


def split_samples(samples: List[Dict[str, np.ndarray]], seed: int, train_ratio: float = 0.82):
    """Split complete trajectories so adjacent transitions cannot leak across sets."""

    rng = np.random.default_rng(seed)
    episode_ids = np.asarray(
        sorted({int(np.asarray(sample["episode_id"]).item()) for sample in samples}),
        dtype=np.int64,
    )
    if len(episode_ids) < 2:
        raise ValueError("Episode-group validation requires transitions from at least two episodes.")
    rng.shuffle(episode_ids)
    split = max(1, min(len(episode_ids) - 1, int(len(episode_ids) * train_ratio)))
    train_ids = set(int(value) for value in episode_ids[:split])
    valid_ids = set(int(value) for value in episode_ids[split:])
    train = [
        sample
        for sample in samples
        if int(np.asarray(sample["episode_id"]).item()) in train_ids
    ]
    valid = [
        sample
        for sample in samples
        if int(np.asarray(sample["episode_id"]).item()) in valid_ids
    ]
    return train, valid


def metadata_from_sample(sample: Dict[str, np.ndarray], hidden_dim: int = 96) -> WorldModelMetadata:
    return WorldModelMetadata(
        agv_count=int(sample["agent_features"].shape[0]),
        node_count=int(sample["node_features"].shape[0]),
        agent_dim=int(sample["agent_features"].shape[1]),
        node_dim=int(sample["node_features"].shape[1]),
        global_dim=int(sample["global_features"].shape[0]),
        hidden_dim=hidden_dim,
    )


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "actions":
            moved[key] = value.long().to(device)
        else:
            moved[key] = value.float().to(device)
    return moved


def evaluate(model, loader, device: str, agv_count: int, physics_weight: float) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch)
            _, parts = world_model_loss(output, batch, agv_count=agv_count, physics_weight=physics_weight)
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def write_history(path: Path, history: List[Dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Collecting world-model transitions from the high-fidelity DT...")
    samples = collect_world_model_transitions(
        env_factory=make_env_factory(args),
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        exploration_rate=args.exploration_rate,
    )
    print(f"Collected {len(samples)} transitions.")

    train_samples, valid_samples = split_samples(samples, seed=args.seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        WorldModelTransitionDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    valid_loader = DataLoader(WorldModelTransitionDataset(valid_samples), batch_size=args.batch_size, shuffle=False)

    metadata = metadata_from_sample(samples[0])
    model = PhysicsInformedGraphWorldModel(metadata).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history: List[Dict[str, float]] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            output = model(batch)
            loss, parts = world_model_loss(
                output,
                batch,
                agv_count=args.agv_count,
                physics_weight=args.physics_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1

        train_parts = {f"train_{key}": value / max(batches, 1) for key, value in totals.items()}
        valid_parts = {f"valid_{key}": value for key, value in evaluate(
            model,
            valid_loader,
            device=device,
            agv_count=args.agv_count,
            physics_weight=args.physics_weight,
        ).items()}
        row = {"epoch": float(epoch), **train_parts, **valid_parts}
        history.append(row)
        if row["valid_loss"] < best_valid_loss:
            best_valid_loss = float(row["valid_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(
                f"epoch {epoch:03d} | "
                f"train_loss={row['train_loss']:.5f} | valid_loss={row['valid_loss']:.5f} | "
                f"valid_physics={row['valid_physics_loss']:.5f}"
            )

    if best_state is None:
        raise RuntimeError("World-model training produced no valid checkpoint")
    model.load_state_dict(best_state)

    model_path = output_dir / "physics_graph_world_model.pt"
    args_dict = vars(args) | {
        "parameter_profile": "jms_interpretable_v2_energy_calibrated",
        "world_model_defaults": dict(WORLD_MODEL_DEFAULTS),
        "validation_split": "episode_grouped",
        "random_seed_control": "numpy_torch_dataloader",
        "selected_epoch": best_epoch,
        "selected_valid_loss": best_valid_loss,
    }
    save_world_model(model_path, model, metadata, history, args_dict)
    write_history(output_dir / "training_history.csv", history)
    (output_dir / "training_args.json").write_text(
        json.dumps(args_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_summary.txt").write_text(
        "\n".join(
            [
                f"samples={len(samples)}",
                f"train_samples={len(train_samples)}",
                f"valid_samples={len(valid_samples)}",
                f"train_episodes={len({int(np.asarray(sample['episode_id']).item()) for sample in train_samples})}",
                f"valid_episodes={len({int(np.asarray(sample['episode_id']).item()) for sample in valid_samples})}",
                "validation_split=episode_grouped",
                f"device={device}",
                f"selected_epoch={best_epoch}",
                f"selected_valid_loss={best_valid_loss}",
                f"model_path={model_path}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"World model saved to {model_path.resolve()}")
    return model_path


if __name__ == "__main__":
    train(build_parser().parse_args())
