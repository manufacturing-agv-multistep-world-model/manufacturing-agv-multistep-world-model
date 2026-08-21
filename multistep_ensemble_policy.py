from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

import numpy as np


class EnsembleAgreementPolicy:
    """Execute a learned action only when independently trained planners agree."""

    def __init__(
        self,
        members: list[Any],
        minimum_agreement: int = 2,
        decision_mode: str = "agreement_only",
        minimum_throughput_delta: float = 0.0,
        maximum_risk_time_increase_sec: float = 2.0,
    ):
        if len(members) < 2:
            raise ValueError("An ensemble requires at least two member policies")
        if minimum_agreement < 2 or minimum_agreement > len(members):
            raise ValueError("minimum_agreement must be between 2 and the ensemble size")
        if decision_mode not in {
            "agreement_only",
            "bounded_evidence",
            "energy_neutral_bounded_evidence",
        }:
            raise ValueError(
                "decision_mode must be 'agreement_only', 'bounded_evidence', "
                "or 'energy_neutral_bounded_evidence'"
            )
        if maximum_risk_time_increase_sec < 0.0:
            raise ValueError("maximum_risk_time_increase_sec must be nonnegative")
        self.members = list(members)
        self.minimum_agreement = int(minimum_agreement)
        self.decision_mode = decision_mode
        self.minimum_throughput_delta = float(minimum_throughput_delta)
        self.maximum_risk_time_increase_sec = float(maximum_risk_time_increase_sec)
        self.last_plan: dict[str, Any] = {}

    def predict_guarded(self, env: Any) -> np.ndarray:
        actions = [
            np.asarray(member.predict_guarded(env), dtype=np.int64)
            for member in self.members
        ]
        plans = [dict(member.last_plan) for member in self.members]
        action_tuples = [tuple(int(value) for value in action) for action in actions]
        counts = Counter(action_tuples)
        majority_action, agreement_count = counts.most_common(1)[0]
        baseline = np.asarray(plans[0]["baseline_action"], dtype=np.int64)
        candidate = np.asarray(majority_action, dtype=np.int64)
        candidate_deviates = not np.array_equal(candidate, baseline)
        agreeing_indices = [
            index for index, action in enumerate(action_tuples) if action == majority_action
        ]

        def average_plan_value(key: str) -> float:
            values = [float(plans[index].get(key, 0.0)) for index in agreeing_indices]
            return mean(values) if values else 0.0

        agreement_passed = agreement_count >= self.minimum_agreement
        evidence_label = "accept_ensemble_agreement"
        if self.decision_mode in {
            "bounded_evidence",
            "energy_neutral_bounded_evidence",
        }:
            physical_safe = not any(
                bool(plans[index].get("physical_gate_applied", False))
                for index in agreeing_indices
            )
            throughput_delta = average_plan_value("predicted_throughput_delta")
            time_increase = average_plan_value("predicted_time_increase_sec")
            risk_threshold = average_plan_value("risk_gate_threshold")
            predicted_energy_reduction = average_plan_value(
                "predicted_energy_reduction_wh"
            )
            risk_energy_passed = (
                self.decision_mode != "energy_neutral_bounded_evidence"
                or predicted_energy_reduction >= 0.0
            )
            energy_threshold = average_plan_value("energy_gate_threshold_wh")
            charge_threshold = average_plan_value(
                "charge_queue_gate_threshold_agent_steps"
            )
            risk_evidence = (
                risk_threshold > 0.0
                and average_plan_value("predicted_risk_reduction") >= risk_threshold
                and throughput_delta >= self.minimum_throughput_delta
                and time_increase <= self.maximum_risk_time_increase_sec
                and risk_energy_passed
            )
            energy_evidence = (
                energy_threshold > 0.0
                and average_plan_value("predicted_energy_reduction_wh")
                >= energy_threshold
                and throughput_delta >= self.minimum_throughput_delta
                and time_increase <= 0.0
            )
            charge_evidence = (
                charge_threshold > 0.0
                and average_plan_value(
                    "predicted_charge_queue_reduction_agent_steps"
                )
                >= charge_threshold
                and throughput_delta >= self.minimum_throughput_delta
                and time_increase <= self.maximum_risk_time_increase_sec
            )
            evidence_passed = risk_evidence or energy_evidence or charge_evidence
            accepted = agreement_passed and candidate_deviates and physical_safe and evidence_passed
            if risk_evidence:
                evidence_label = (
                    "accept_energy_neutral_risk"
                    if self.decision_mode == "energy_neutral_bounded_evidence"
                    else "accept_bounded_risk"
                )
            elif energy_evidence:
                evidence_label = "accept_strict_energy"
            elif charge_evidence:
                evidence_label = "accept_bounded_charge"
            else:
                evidence_label = "reject_insufficient_bounded_evidence"
        else:
            accepted = agreement_passed and candidate_deviates

        selected = candidate if accepted else baseline.copy()
        deviates = not np.array_equal(selected, baseline)

        self.last_plan = {
            "raw_planned_action": list(majority_action),
            "executed_planned_action": selected.tolist(),
            "baseline_action": baseline.tolist(),
            "predicted_risk_reduction": average_plan_value("predicted_risk_reduction"),
            "predicted_energy_reduction_wh": average_plan_value(
                "predicted_energy_reduction_wh"
            ),
            "predicted_throughput_delta": average_plan_value(
                "predicted_throughput_delta"
            ),
            "predicted_time_increase_sec": average_plan_value(
                "predicted_time_increase_sec"
            ),
            "predicted_route_blocking_reduction_agent_steps": average_plan_value(
                "predicted_route_blocking_reduction_agent_steps"
            ),
            "predicted_charge_queue_reduction_agent_steps": average_plan_value(
                "predicted_charge_queue_reduction_agent_steps"
            ),
            "risk_gate_threshold": average_plan_value("risk_gate_threshold"),
            "energy_gate_threshold_wh": average_plan_value(
                "energy_gate_threshold_wh"
            ),
            "charge_queue_gate_threshold_agent_steps": average_plan_value(
                "charge_queue_gate_threshold_agent_steps"
            ),
            "risk_gate_applied": False,
            "energy_gate_applied": False,
            "physical_gate_applied": any(
                bool(plan.get("physical_gate_applied", False)) for plan in plans
            ),
            "override_accepted": bool(accepted and deviates),
            "override_evidence": evidence_label
            if accepted and deviates
            else (
                "ensemble_baseline_or_disagreement"
                if not agreement_passed
                else evidence_label
            ),
            "override_mode": f"ensemble_{self.decision_mode}",
            "minimum_throughput_delta": self.minimum_throughput_delta,
            "maximum_risk_time_increase_sec": self.maximum_risk_time_increase_sec,
            "minimum_risk_energy_reduction_wh": (
                0.0
                if self.decision_mode == "energy_neutral_bounded_evidence"
                else float("-inf")
            ),
            "analytical_future_conflict_agent_count": average_plan_value(
                "analytical_future_conflict_agent_count"
            ),
            "operational_energy_action": any(
                bool(plan.get("operational_energy_action", False)) for plan in plans
            ),
            "analytical_charge_staggering": any(
                bool(plan.get("analytical_charge_staggering", False)) for plan in plans
            ),
            "dedicated_charge_gate_required": any(
                bool(plan.get("dedicated_charge_gate_required", False)) for plan in plans
            ),
            "analytical_charge_pressure_agent_count": average_plan_value(
                "analytical_charge_pressure_agent_count"
            ),
            "available_charge_slots": average_plan_value("available_charge_slots"),
            "anti_stagnation_applied": any(
                bool(plan.get("anti_stagnation_applied", False)) for plan in plans
            ),
            "unsafe_candidate_filter_applied": any(
                bool(plan.get("unsafe_candidate_filter_applied", False)) for plan in plans
            ),
            "candidate_count": average_plan_value("candidate_count"),
            "ensemble_size": len(self.members),
            "ensemble_agreement_count": agreement_count,
            "ensemble_agreement_fraction": agreement_count / len(self.members),
            "ensemble_member_actions": [list(action) for action in action_tuples],
        }
        return selected
