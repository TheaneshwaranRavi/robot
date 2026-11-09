"""Portable tests for policy and control-loop safety checks."""
from __future__ import annotations

import unittest

from sim.safety import (
    ActionLimits,
    SafetyStop,
    Watchdog,
    clamp_targets,
    require_remote_ready,
    require_target,
)


class ActionLimitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = ActionLimits(
            names=("joint", "gripper"),
            low=(-1.0, 0.0),
            high=(1.0, 1.0),
        )

    def test_clamp_clips_values_to_each_dimension(self) -> None:
        self.assertEqual(self.limits.clamp([-2.0, 2.0]), [-1.0, 1.0])

    def test_clamp_rejects_non_finite_values(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(SafetyStop, "non-finite value in action"):
                    self.limits.clamp([value, 0.5])

    def test_clamp_rejects_wrong_length(self) -> None:
        for action in ([0.0], [0.0, 0.0, 0.0]):
            with self.subTest(action=action):
                with self.assertRaisesRegex(SafetyStop, "expected 2"):
                    self.limits.clamp(action)

    def test_clamp_targets_applies_tracking_window_and_joint_limits(self) -> None:
        actual = clamp_targets(
            targets=[-100.0, 100.0, 0.3, -0.3],
            measured=[-0.95, 0.95, 0.0, 0.0],
            low=[-1.0] * 4,
            high=[1.0] * 4,
            max_tracking_error=0.2,
        )
        self.assertEqual(actual, [-1.0, 1.0, 0.2, -0.2])


class WatchdogTest(unittest.TestCase):
    def test_policy_deadline(self) -> None:
        watchdog = Watchdog(policy_timeout_s=0.2, step_period_s=0.1)
        watchdog.check_policy_time(0.2)
        with self.assertRaisesRegex(SafetyStop, "policy took.*deadline"):
            watchdog.check_policy_time(0.2001)

    def test_realtime_requires_consecutive_deadline_misses(self) -> None:
        watchdog = Watchdog(
            policy_timeout_s=1.0,
            step_period_s=0.1,
            max_misses=3,
            realtime=True,
        )

        watchdog.check_step_time(0.11)
        watchdog.check_step_time(0.11)
        self.assertEqual(watchdog.consecutive_misses, 2)
        watchdog.check_step_time(0.1)
        self.assertEqual(watchdog.consecutive_misses, 0)

        watchdog.check_step_time(0.11)
        watchdog.check_step_time(0.11)
        with self.assertRaisesRegex(SafetyStop, "3 times in a row"):
            watchdog.check_step_time(0.11)
        self.assertEqual(watchdog.consecutive_misses, 3)
        self.assertEqual(watchdog.total_misses, 5)

    def test_offline_mode_ignores_step_overruns(self) -> None:
        watchdog = Watchdog(
            policy_timeout_s=1.0,
            step_period_s=0.01,
            max_misses=1,
            realtime=False,
        )
        for elapsed_s in (1.0, 10.0, 100.0):
            watchdog.check_step_time(elapsed_s)
        self.assertEqual(watchdog.consecutive_misses, 0)
        self.assertEqual(watchdog.total_misses, 0)


class ConfigurationSafetyTest(unittest.TestCase):
    def test_require_remote_ready_refuses_placeholder_configs(self) -> None:
        for config in ({}, {"placeholder": True}):
            with self.subTest(config=config):
                with self.assertRaisesRegex(SafetyStop, "placeholder"):
                    require_remote_ready(config)
        require_remote_ready({"placeholder": False})

    def test_require_target_refuses_mismatches(self) -> None:
        require_target({"target": "sim"}, "sim")
        require_target({}, "hardware")

        for config, expected in (({"target": "hardware"}, "sim"), ({"target": "sim"}, "hardware")):
            with self.subTest(config=config, expected=expected):
                with self.assertRaisesRegex(SafetyStop, "config target"):
                    require_target(config, expected)


if __name__ == "__main__":
    unittest.main()
