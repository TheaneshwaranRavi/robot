"""Portable tests for the simulation embodiment config extensions."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from sim.config import ConfigError, load_config, summarize, validate_config


ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = ROOT / "config" / "embodiment.sim.json"
EXAMPLE_CONFIG = ROOT / "config" / "embodiment.example.json"


class SimulationConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(SIM_CONFIG.read_text())

    def test_sim_config_is_valid(self) -> None:
        load_config(SIM_CONFIG)

    def test_example_config_remains_valid(self) -> None:
        load_config(EXAMPLE_CONFIG)

    def test_rejects_bad_target(self) -> None:
        self.config["target"] = "space"
        with self.assertRaisesRegex(ConfigError, "target"):
            validate_config(self.config)

    def test_rejects_joint_limit_with_wrong_length(self) -> None:
        for bound in ("low", "high"):
            with self.subTest(bound=bound):
                config = copy.deepcopy(self.config)
                limits = config["joint_limits"]["arm_joint_positions"]
                limits[bound] = limits[bound][:-1]
                with self.assertRaisesRegex(ConfigError, "must list 6 numbers"):
                    validate_config(config)

    def test_rejects_non_finite_joint_limit(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                config = copy.deepcopy(self.config)
                config["joint_limits"]["arm_joint_positions"]["low"][0] = value
                with self.assertRaisesRegex(ConfigError, "finite numbers"):
                    validate_config(config)

    def test_rejects_joint_limit_where_low_is_not_less_than_high(self) -> None:
        for low_offset in (0.0, 0.1):
            with self.subTest(low_offset=low_offset):
                config = copy.deepcopy(self.config)
                limits = config["joint_limits"]["arm_joint_positions"]
                limits["low"][0] = limits["high"][0] + low_offset
                with self.assertRaisesRegex(ConfigError, "low must be less than high"):
                    validate_config(config)

    def test_rejects_joint_limits_for_unknown_state(self) -> None:
        self.config["joint_limits"]["unknown_state"] = {"low": [0.0], "high": [1.0]}
        with self.assertRaisesRegex(ConfigError, "not an observation.state name"):
            validate_config(self.config)

    def test_summary_includes_simulation_only_warning(self) -> None:
        self.assertIn(
            "Target: simulation only; never use with hardware",
            summarize(load_config(SIM_CONFIG)),
        )


if __name__ == "__main__":
    unittest.main()
