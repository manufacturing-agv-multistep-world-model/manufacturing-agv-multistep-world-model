from __future__ import annotations

import unittest

from evaluate_parameter_sensitivity import build_parser, build_specs


class ParameterSensitivityDesignTests(unittest.TestCase):
    def build_dt_specs(self):
        args = build_parser().parse_args(
            [
                "--fixed-time-hours",
                "8",
                "--capacity-mode",
                "baseline",
                "--parameters",
                "speed,acceleration,loaded_energy,charge_rate,arrival_rate",
                "--methods",
                "DT-aware",
            ]
        )
        return build_specs(args)

    def test_each_parameter_has_symmetric_local_levels(self) -> None:
        specs = self.build_dt_specs()
        for parameter in ("speed", "acceleration", "loaded_energy", "charge_rate", "arrival_rate"):
            levels = [
                spec["sensitivity_multiplier"]
                for spec in specs
                if spec["sensitivity_parameter"] == parameter
            ]
            self.assertEqual([0.8, 0.9, 1.0, 1.1, 1.2], levels)

    def test_each_arm_changes_exactly_one_config_factor(self) -> None:
        for spec in self.build_dt_specs():
            self.assertEqual(1, len(spec["config_overrides"]))
            if spec["sensitivity_parameter"] == "speed":
                self.assertEqual({"speed_max_mps"}, set(spec["config_overrides"]))

    def test_charge_rate_is_centered_on_two_percent_soc_per_minute(self) -> None:
        charge_specs = [
            spec
            for spec in self.build_dt_specs()
            if spec["sensitivity_parameter"] == "charge_rate"
        ]
        self.assertEqual(
            [1.6, 1.8, 2.0, 2.2, 2.4],
            [spec["sensitivity_value"] for spec in charge_specs],
        )


if __name__ == "__main__":
    unittest.main()
