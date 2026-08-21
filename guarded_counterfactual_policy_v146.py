from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from counterfactual_rollout_v141 import _different_candidates
from diagnose_counterfactual_ranking_v144 import UTILITY_WEIGHTS
from industrial_safety_guard import (
    apply_industrial_safety_guard,
    required_safety_actions,
)
from physics_graph_world_model import baseline_dt_aware_action


FROZEN_UTILITY_MARGIN = 0.15
TERMINAL_HORIZON_INDEX = 2


@dataclass(frozen=True)
class V146AuthorityLimits:
    utility_margin: float = FROZEN_UTILITY_MARGIN
    cooldown_sec: float = 60.0
    maximum_overrides: int = 12
    maximum_action_hamming_distance: int = 3


class GuardedDTBaselinePolicy:
    """Run the DT-aware fallback through the same hard safety layer as V14.6."""

    def __init__(self) -> None:
        self.last_plan: Dict[str, Any] = {}

    def predict_guarded(self, env: Any) -> np.ndarray:
        action = baseline_dt_aware_action(env)
        self.last_plan = {
            "baseline_action": action.tolist(),
            "raw_planned_action": action.tolist(),
            "executed_planned_action": action.tolist(),
            "override_accepted": False,
            "override_evidence": "baseline",
            "override_mode": "v146_guarded_dt_baseline",
            "ensemble_size": 0,
            "v146_rejection_reason": "baseline_method",
        }
        return action


class GuardedCounterfactualPolicyV146:
    """Bounded execution policy using three frozen V14.1 counterfactual models."""

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        limits: V146AuthorityLimits | None = None,
        policy_label: str = "v146_bounded_unanimous",
    ) -> None:
        if len(models) != 3:
            raise ValueError("V14.6 requires exactly three frozen V14.1 models")
        self.models = list(models)
        self.limits = limits or V146AuthorityLimits()
        self.policy_label = str(policy_label)
        if self.limits.utility_margin != FROZEN_UTILITY_MARGIN:
            raise ValueError("V14.6 development must retain the frozen V14.5 margin")
        self._devices = [next(model.parameters()).device for model in self.models]
        self._terminal_scale = np.maximum(
            np.mean(
                np.stack(
                    [model.counterfactual_scale.detach().cpu().numpy() for model in self.models]
                ),
                axis=0,
            )[TERMINAL_HORIZON_INDEX],
            1.0e-9,
        )
        self.override_count = 0
        self.last_override_time_sec = -np.inf
        self.deadlock_count_at_last_override: float | None = None
        self.permanent_fallback = False
        self.last_plan: Dict[str, Any] = {}
        for model in self.models:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _state_batch(
        obs: Dict[str, np.ndarray],
        baseline: np.ndarray,
        candidates: Sequence[np.ndarray],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        count = len(candidates)
        batch: Dict[str, torch.Tensor] = {}
        for key in ("agent_features", "node_features", "adjacency_matrix", "global_features"):
            value = torch.as_tensor(obs[key], dtype=torch.float32, device=device)
            batch[key] = value.unsqueeze(0).expand(count, *value.shape)
        batch["baseline_actions"] = torch.as_tensor(
            np.repeat(baseline[None, :], count, axis=0),
            dtype=torch.long,
            device=device,
        )
        batch["candidate_actions"] = torch.as_tensor(
            np.stack(candidates), dtype=torch.long, device=device
        )
        return batch

    def _predict(self, env: Any, baseline: np.ndarray, candidates: Sequence[np.ndarray]) -> np.ndarray:
        obs = env._get_obs()
        predictions: List[np.ndarray] = []
        with torch.inference_mode():
            for model, device in zip(self.models, self._devices):
                output = model.forward_counterfactual(
                    self._state_batch(obs, baseline, candidates, device)
                )["counterfactual_delta"]
                predictions.append(output.detach().cpu().numpy())
        return np.stack(predictions)

    def _fallback(
        self,
        baseline: np.ndarray,
        reason: str,
        *,
        candidate_count: int = 0,
        unanimous: bool = False,
        predicted_gain: float = 0.0,
        member_actions: Sequence[np.ndarray] = (),
    ) -> np.ndarray:
        self.last_plan = {
            "baseline_action": baseline.tolist(),
            "raw_planned_action": baseline.tolist(),
            "executed_planned_action": baseline.tolist(),
            "override_accepted": False,
            "override_evidence": "baseline",
            "override_mode": self.policy_label,
            "ensemble_size": len(self.models),
            "ensemble_agreement_count": len(self.models) if unanimous else 0,
            "ensemble_agreement_fraction": 1.0 if unanimous else 0.0,
            "ensemble_member_actions": [action.tolist() for action in member_actions],
            "v146_shadow_recommended": bool(unanimous and predicted_gain >= self.limits.utility_margin),
            "v146_predicted_normalized_gain": float(predicted_gain),
            "v146_candidate_count": int(candidate_count),
            "v146_rejection_reason": reason,
            "v146_override_count": int(self.override_count),
            "v146_permanent_fallback": bool(self.permanent_fallback),
        }
        return baseline

    def predict_guarded(self, env: Any) -> np.ndarray:
        baseline = np.asarray(baseline_dt_aware_action(env), dtype=np.int64)
        summary = env.summary()
        current_time = float(summary["real_time_sec"])
        current_deadlocks = float(summary["deadlock_count"])

        if (
            self.deadlock_count_at_last_override is not None
            and current_deadlocks > self.deadlock_count_at_last_override
        ):
            self.permanent_fallback = True
        if summary["out_of_battery_rate"] > 0.0 or summary["timeout_rate"] > 0.0:
            self.permanent_fallback = True
        if self.permanent_fallback:
            return self._fallback(baseline, "automatic_safety_fallback")
        if bool(getattr(env, "deadlock_active", False)):
            return self._fallback(baseline, "deadlock_recovery_active")
        if np.any(required_safety_actions(env) >= 0):
            return self._fallback(baseline, "hard_safety_constraint")

        candidates = [
            np.asarray(action, dtype=np.int64)
            for action in _different_candidates(env)
            if np.count_nonzero(np.asarray(action, dtype=np.int64) != baseline)
            <= self.limits.maximum_action_hamming_distance
        ]
        if not candidates:
            return self._fallback(baseline, "no_bounded_candidate")

        prediction = self._predict(env, baseline, candidates)
        utilities = np.sum(
            prediction[:, :, TERMINAL_HORIZON_INDEX, :]
            / self._terminal_scale[None, None, :]
            * UTILITY_WEIGHTS[None, None, :],
            axis=2,
        )
        choices = np.argmax(
            np.concatenate([np.zeros((len(self.models), 1)), utilities], axis=1),
            axis=1,
        )
        member_actions = [
            baseline if choice == 0 else candidates[int(choice) - 1] for choice in choices
        ]
        unanimous = bool(np.all(choices == choices[0]))
        if not unanimous or choices[0] == 0:
            return self._fallback(
                baseline,
                "no_unanimous_nonbaseline_choice",
                candidate_count=len(candidates),
                unanimous=unanimous,
                member_actions=member_actions,
            )

        selected_index = int(choices[0]) - 1
        candidate = candidates[selected_index]
        predicted_gain = float(np.mean(utilities[:, selected_index]))
        if predicted_gain < self.limits.utility_margin:
            return self._fallback(
                baseline,
                "below_frozen_utility_margin",
                candidate_count=len(candidates),
                unanimous=True,
                predicted_gain=predicted_gain,
                member_actions=member_actions,
            )
        if not np.array_equal(apply_industrial_safety_guard(env, candidate), candidate):
            return self._fallback(
                baseline,
                "candidate_changed_by_safety_guard",
                candidate_count=len(candidates),
                unanimous=True,
                predicted_gain=predicted_gain,
                member_actions=member_actions,
            )
        if current_time - self.last_override_time_sec < self.limits.cooldown_sec:
            return self._fallback(
                baseline,
                "physical_time_cooldown",
                candidate_count=len(candidates),
                unanimous=True,
                predicted_gain=predicted_gain,
                member_actions=member_actions,
            )
        if self.override_count >= self.limits.maximum_overrides:
            return self._fallback(
                baseline,
                "override_budget_exhausted",
                candidate_count=len(candidates),
                unanimous=True,
                predicted_gain=predicted_gain,
                member_actions=member_actions,
            )

        mean_delta = np.mean(prediction[:, selected_index, TERMINAL_HORIZON_INDEX], axis=0)
        self.override_count += 1
        self.last_override_time_sec = current_time
        self.deadlock_count_at_last_override = current_deadlocks
        self.last_plan = {
            "baseline_action": baseline.tolist(),
            "raw_planned_action": candidate.tolist(),
            "executed_planned_action": candidate.tolist(),
            "override_accepted": True,
            "override_evidence": "accept_ensemble_agreement",
            "override_mode": self.policy_label,
            "ensemble_size": len(self.models),
            "ensemble_agreement_count": len(self.models),
            "ensemble_agreement_fraction": 1.0,
            "ensemble_member_actions": [action.tolist() for action in member_actions],
            "predicted_energy_reduction_wh": float(-mean_delta[0]),
            "predicted_throughput_delta": float(mean_delta[1]),
            "predicted_charge_queue_reduction_agent_steps": float(-mean_delta[2]),
            "v146_shadow_recommended": True,
            "v146_predicted_normalized_gain": predicted_gain,
            "v146_candidate_count": len(candidates),
            "v146_rejection_reason": "accepted",
            "v146_override_count": self.override_count,
            "v146_permanent_fallback": False,
        }
        return candidate
