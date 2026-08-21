from __future__ import annotations

import argparse
import csv
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from physics_graph_world_model_multistep_v13 import annotate_future_congestion_risk
from train_world_model_multistep import split_v12_congestion_stratified


ROOT = Path(__file__).resolve().parent
ARCHITECTURES = ("mlp", "gru", "gnn")
ACTION_COUNT = 4
HISTORY_LENGTH = 20


def state_vector(row: Dict[str, np.ndarray]) -> np.ndarray:
    actions = np.asarray(row["actions"], dtype=np.int64)
    action_one_hot = np.eye(ACTION_COUNT, dtype=np.float32)[actions].reshape(-1)
    return np.concatenate(
        [
            np.asarray(row["agent_features"], dtype=np.float32).reshape(-1),
            np.asarray(row["global_features"], dtype=np.float32).reshape(-1),
            action_one_hot,
        ]
    ).astype(np.float32)


def build_risk_samples(
    transitions: Sequence[Dict[str, np.ndarray]],
    stride: int,
    rollout_horizon: int = 10,
    history_length: int = HISTORY_LENGTH,
) -> List[Dict[str, np.ndarray]]:
    """Build aligned current-state samples with padded retrospective histories."""

    episodes: Dict[int, List[Dict[str, np.ndarray]]] = {}
    for row in transitions:
        episode = int(np.asarray(row["episode_id"]).item())
        episodes.setdefault(episode, []).append(row)
    samples: List[Dict[str, np.ndarray]] = []
    for episode, rows in sorted(episodes.items()):
        rows.sort(key=lambda row: int(np.asarray(row["transition_id"]).item()))
        for start in range(0, len(rows) - rollout_horizon + 1, stride):
            row = rows[start]
            if float(row["future_congestion_risk_mask"][0]) <= 0.0:
                continue
            history_start = max(0, start - history_length + 1)
            history_rows = rows[history_start : start + 1]
            history = [state_vector(candidate) for candidate in history_rows]
            if len(history) < history_length:
                history = [history[0]] * (history_length - len(history)) + history
            samples.append(
                {
                    "episode_id": np.asarray(episode, dtype=np.int64),
                    "transition_id": np.asarray(
                        int(np.asarray(row["transition_id"]).item()), dtype=np.int64
                    ),
                    "state_vector": state_vector(row),
                    "history": np.stack(history).astype(np.float32),
                    "agent_features": np.asarray(
                        row["agent_features"], dtype=np.float32
                    ),
                    "node_features": np.asarray(row["node_features"], dtype=np.float32),
                    "adjacency_matrix": np.asarray(
                        row["adjacency_matrix"], dtype=np.float32
                    ),
                    "global_features": np.asarray(
                        row["global_features"], dtype=np.float32
                    ),
                    "actions": np.asarray(row["actions"], dtype=np.int64),
                    "target": np.asarray(row["future_congestion_risk"][0], dtype=np.float32),
                }
            )
    if not samples:
        raise ValueError("No eligible future-risk samples were constructed")
    return samples


class RiskDataset(Dataset):
    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.tensors = {
            key: torch.as_tensor(np.stack([sample[key] for sample in samples]))
            for key in samples[0]
        }

    def __len__(self) -> int:
        return int(self.tensors["target"].shape[0])

    def __getitem__(self, index: int):
        return {key: value[index] for key, value in self.tensors.items()}


class MLPRiskPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        return self.network(batch["state_vector"].float()).squeeze(-1)


class GRURiskPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        _, hidden = self.gru(batch["history"].float())
        return self.head(hidden[-1]).squeeze(-1)


class DynamicGNNRiskPredictor(nn.Module):
    """Dynamic topology model without engineered physical node/edge attributes."""

    def __init__(
        self,
        agent_dim: int,
        node_dim: int,
        global_dim: int,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_dim + ACTION_COUNT, hidden_dim), nn.SiLU()
        )
        self.node_self = nn.Linear(node_dim, hidden_dim)
        self.node_neighbor = nn.Linear(node_dim, hidden_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        actions = F.one_hot(batch["actions"].long(), num_classes=ACTION_COUNT).float()
        agent_tokens = self.agent_encoder(
            torch.cat([batch["agent_features"].float(), actions], dim=-1)
        )
        nodes = batch["node_features"].float()
        adjacency = batch["adjacency_matrix"].float()
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_features = torch.matmul(adjacency, nodes) / degree
        node_tokens = F.silu(
            self.node_self(nodes) + self.node_neighbor(neighbor_features)
        )
        global_token = self.global_encoder(batch["global_features"].float())
        context = torch.cat(
            [agent_tokens.mean(dim=1), node_tokens.mean(dim=1), global_token], dim=-1
        )
        return self.head(context).squeeze(-1)


def make_model(architecture: str, sample: Dict[str, np.ndarray], hidden_dim: int):
    if architecture == "mlp":
        return MLPRiskPredictor(int(sample["state_vector"].shape[0]), hidden_dim)
    if architecture == "gru":
        return GRURiskPredictor(int(sample["state_vector"].shape[0]), hidden_dim)
    if architecture == "gnn":
        return DynamicGNNRiskPredictor(
            int(sample["agent_features"].shape[-1]),
            int(sample["node_features"].shape[-1]),
            int(sample["global_features"].shape[-1]),
            hidden_dim,
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def load_baseline_model(
    checkpoint_path: str | Path,
    sample: Dict[str, np.ndarray],
    device: str = "cpu",
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_model(
        checkpoint["architecture"], sample, int(checkpoint["hidden_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def heuristic_charge_risk_score(sample: Dict[str, np.ndarray]) -> float:
    """Interpretable score from multi-vehicle SOC, charger occupancy, and demand."""

    batteries = np.sort(np.asarray(sample["agent_features"], dtype=float)[:, 1])
    second_lowest_soc = float(batteries[min(1, len(batteries) - 1)])
    nodes = np.asarray(sample["node_features"], dtype=float)
    charge_nodes = nodes[:, 3] > 0.5
    charger_occupancy = (
        float(np.max(nodes[charge_nodes, 0])) if np.any(charge_nodes) else 0.0
    )
    scenario_pressure = float(sample["global_features"][6])
    return float(
        np.clip(
            0.65 * (1.0 - second_lowest_soc)
            + 0.20 * charger_occupancy
            + 0.15 * scenario_pressure,
            0.0,
            1.0,
        )
    )


def move_batch(batch, device: str):
    return {
        key: value.long().to(device) if key in {"episode_id", "transition_id", "actions"}
        else value.float().to(device)
        for key, value in batch.items()
    }


def evaluate(model, loader, device: str, positive_weight: torch.Tensor):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["target"], pos_weight=positive_weight
            )
            total += float(loss.detach().cpu()) * len(logits)
            count += len(logits)
    return total / max(count, 1)


def write_history(path: Path, rows: List[Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(
    architecture: str,
    seed: int,
    train_samples,
    valid_samples,
    output_dir: Path,
    args,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(architecture, train_samples[0], args.hidden_dim).to(args.device)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    targets = np.asarray([float(sample["target"]) for sample in train_samples])
    positive_weight_value = min(
        max(float(np.sum(targets <= 0.0)) / max(float(np.sum(targets > 0.0)), 1.0), 1.0),
        20.0,
    )
    positive_weight = torch.as_tensor([positive_weight_value], device=args.device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        RiskDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=args.device == "cuda",
    )
    valid_loader = DataLoader(
        RiskDataset(valid_samples),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.device == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = move_batch(batch, args.device)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["target"], pos_weight=positive_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(logits)
            count += len(logits)
        train_loss = total / max(count, 1)
        valid_loss = evaluate(model, valid_loader, args.device, positive_weight)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Baseline training produced no checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "seed": seed,
            "state_dict": best_state,
            "hidden_dim": args.hidden_dim,
            "positive_weight": positive_weight_value,
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "trainable_parameters": trainable,
            "sample_shapes": {
                key: list(value.shape) for key, value in train_samples[0].items()
            },
            "protocol": vars(args),
        },
        output_dir / "model.pt",
    )
    write_history(output_dir / "training_history.csv", history)
    print(
        f"{architecture} seed={seed}: epoch={best_epoch}, "
        f"valid_loss={best_loss:.6f}, parameters={trainable:,}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Train future charge-risk baselines.")
    parser.add_argument("--transition-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--architectures", default="mlp,gru,gnn")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--future-risk-horizon", type=int, default=80)
    parser.add_argument("--sequence-stride", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=4200)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def main():
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA baseline training requested but unavailable")
    cache_path = Path(args.transition_cache)
    with gzip.open(cache_path, "rb") as stream:
        transitions = pickle.load(stream)["transitions"]
    transitions = annotate_future_congestion_risk(
        transitions, horizon=args.future_risk_horizon
    )
    train_rows, valid_rows = split_v12_congestion_stratified(
        transitions, seed=args.split_seed
    )
    train_samples = build_risk_samples(
        train_rows, stride=args.sequence_stride, rollout_horizon=10
    )
    valid_samples = build_risk_samples(
        valid_rows, stride=args.sequence_stride, rollout_horizon=10
    )
    print(
        f"Risk baseline data: train={len(train_samples)}, valid={len(valid_samples)}, "
        f"train_prevalence={np.mean([s['target'] for s in train_samples]):.4f}, "
        f"valid_prevalence={np.mean([s['target'] for s in valid_samples]):.4f}"
    )
    output_root = Path(args.output_root)
    architectures = [item.strip() for item in args.architectures.split(",")]
    seeds = [int(item.strip()) for item in args.seeds.split(",")]
    for architecture in architectures:
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}")
        for seed in seeds:
            train_one(
                architecture,
                seed,
                train_samples,
                valid_samples,
                output_root / f"{architecture}_seed{seed}",
                args,
            )


if __name__ == "__main__":
    main()
