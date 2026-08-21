from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from counterfactual_rollout_v141 import (
    COUNTERFACTUAL_HORIZONS_SEC,
    COUNTERFACTUAL_METRIC_NAMES,
)


MODEL_VERSION = "flat_mlp_counterfactual_v150_trainable_budget_matched_v1"
EXPECTED_INPUT_DIM = 192
EXPECTED_TRAINABLE_PARAMETERS = 56_457


class FlatCounterfactualBaselineV150(nn.Module):
    """Nongraph paired-action baseline with the V14.1 head parameter budget."""

    def __init__(
        self,
        agv_count: int,
        node_count: int,
        agent_dim: int,
        node_dim: int,
        global_dim: int,
        action_dim: int = 4,
        counterfactual_scale: torch.Tensor | np.ndarray | None = None,
        counterfactual_event_weight: torch.Tensor | np.ndarray | None = None,
        counterfactual_horizons_sec: Sequence[float] = COUNTERFACTUAL_HORIZONS_SEC,
    ) -> None:
        super().__init__()
        self.agv_count = int(agv_count)
        self.node_count = int(node_count)
        self.agent_dim = int(agent_dim)
        self.node_dim = int(node_dim)
        self.global_dim = int(global_dim)
        self.action_dim = int(action_dim)
        self.counterfactual_horizons_sec = tuple(
            float(value) for value in counterfactual_horizons_sec
        )
        self.input_dim = (
            self.agv_count * self.agent_dim
            + self.node_count * self.node_dim
            + self.global_dim
            + self.agv_count * self.action_dim
        )
        if self.input_dim != EXPECTED_INPUT_DIM:
            raise ValueError(
                f"V15.0 frozen architecture requires 192 inputs, got {self.input_dim}"
            )
        output_shape = (
            len(self.counterfactual_horizons_sec),
            len(COUNTERFACTUAL_METRIC_NAMES),
        )
        scales = (
            torch.ones(output_shape, dtype=torch.float32)
            if counterfactual_scale is None
            else torch.as_tensor(counterfactual_scale, dtype=torch.float32)
        )
        event_weights = (
            torch.ones(output_shape, dtype=torch.float32)
            if counterfactual_event_weight is None
            else torch.as_tensor(counterfactual_event_weight, dtype=torch.float32)
        )
        if tuple(scales.shape) != output_shape or torch.any(scales <= 0.0):
            raise ValueError("Counterfactual scales must match the frozen output shape")
        if tuple(event_weights.shape) != output_shape or torch.any(event_weights < 1.0):
            raise ValueError("Counterfactual event weights must match the output shape")
        self.register_buffer("counterfactual_scale", scales.clone())
        self.register_buffer("counterfactual_event_weight", event_weights.clone())
        self.action_value_head = nn.Sequential(
            nn.Linear(EXPECTED_INPUT_DIM, 192),
            nn.SiLU(),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, int(np.prod(output_shape))),
        )
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeError(
                f"Frozen V15.0 parameter budget changed: {parameter_count}"
            )

    def _flat_state_action(
        self, batch: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> torch.Tensor:
        batch_size = actions.shape[0]
        action_one_hot = F.one_hot(
            actions.long(), num_classes=self.action_dim
        ).float()
        features = torch.cat(
            [
                batch["agent_features"].float().reshape(batch_size, -1),
                batch["node_features"].float().reshape(batch_size, -1),
                batch["global_features"].float().reshape(batch_size, -1),
                action_one_hot.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        if features.shape[-1] != EXPECTED_INPUT_DIM:
            raise ValueError("Unexpected flattened state-action width")
        return features

    def _action_value(
        self, batch: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> torch.Tensor:
        values = self.action_value_head(self._flat_state_action(batch, actions))
        return values.reshape(
            -1,
            len(self.counterfactual_horizons_sec),
            len(COUNTERFACTUAL_METRIC_NAMES),
        )

    def forward_counterfactual(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        candidate = self._action_value(batch, batch["candidate_actions"])
        baseline = self._action_value(batch, batch["baseline_actions"])
        normalized_delta = candidate - baseline
        return {
            "counterfactual_delta": normalized_delta * self.counterfactual_scale,
            "normalized_counterfactual_delta": normalized_delta,
            "counterfactual_scale": self.counterfactual_scale,
            "counterfactual_event_weight": self.counterfactual_event_weight,
        }


def dimensions_from_sample(sample: Dict[str, np.ndarray]) -> Dict[str, int]:
    agent = np.asarray(sample["agent_features"])
    node = np.asarray(sample["node_features"])
    global_features = np.asarray(sample["global_features"])
    return {
        "agv_count": int(agent.shape[0]),
        "node_count": int(node.shape[0]),
        "agent_dim": int(agent.shape[1]),
        "node_dim": int(node.shape[1]),
        "global_dim": int(global_features.shape[0]),
        "action_dim": 4,
    }


def save_flat_counterfactual_baseline_v150(
    path: str | Path,
    model: FlatCounterfactualBaselineV150,
    history: Sequence[Dict[str, float]],
    args: Dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dimensions = {
        name: getattr(model, name)
        for name in (
            "agv_count",
            "node_count",
            "agent_dim",
            "node_dim",
            "global_dim",
            "action_dim",
        )
    }
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "state_dict": model.state_dict(),
            "dimensions": dimensions,
            "history": list(history),
            "args": args,
            "counterfactual_horizons_sec": model.counterfactual_horizons_sec,
            "counterfactual_scale": model.counterfactual_scale.detach().cpu(),
            "counterfactual_event_weight": (
                model.counterfactual_event_weight.detach().cpu()
            ),
            "uses_adjacency": False,
            "uses_static_physical_features": False,
            "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        },
        destination,
    )


def load_flat_counterfactual_baseline_v150(
    path: str | Path, device: str = "cpu"
) -> Tuple[FlatCounterfactualBaselineV150, Dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Checkpoint is not a {MODEL_VERSION} model")
    model = FlatCounterfactualBaselineV150(
        **checkpoint["dimensions"],
        counterfactual_scale=checkpoint["counterfactual_scale"],
        counterfactual_event_weight=checkpoint["counterfactual_event_weight"],
        counterfactual_horizons_sec=checkpoint["counterfactual_horizons_sec"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint

