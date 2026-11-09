"""Unit tests for Phase 3 contracts and strict configuration."""
from __future__ import annotations

import copy
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from phase3.config import ConfigError, load_config, validate_config
from phase3.contracts import (
    JOINT_NAMES,
    CameraFrame,
    CellEvent,
    InspectionResult,
    PartObservation,
    RobotCommand,
    RobotState,
    SafetyResult,
)


def joints(gripper: float = 0.5) -> dict[str, float]:
    return {name: gripper if name == "gripper" else 0.0 for name in JOINT_NAMES}


def frame(**changes) -> CameraFrame:
    values = {
        "frame_id": "frame-1",
        "timestamp_s": 1.0,
        "received_at_s": 1.1,
        "rgb": np.zeros((8, 10, 3), dtype=np.uint8),
        "camera_id": "camera-1",
        "calibration_id": "cal-1",
    }
    values.update(changes)
    return CameraFrame(**values)


def observation(**changes) -> PartObservation:
    values = {
        "part_id": "part-1",
        "source_frame_id": "frame-1",
        "timestamp_s": 1.1,
        "roi_xyxy": (1, 1, 9, 7),
        "centroid_px": (5, 4),
        "angle_rad": 0.1,
        "pick_pixel": (5, 4),
        "confidence": 0.9,
        "mask": np.ones((8, 10), dtype=np.bool_),
    }
    values.update(changes)
    return PartObservation(**values)


def inspection(**changes) -> InspectionResult:
    values = {
        "part_id": "part-1",
        "source_frame_id": "frame-1",
        "timestamp_s": 1.2,
        "score": 0.2,
        "threshold": 0.5,
        "decision": "GOOD",
        "backend_id": "fake-detector",
        "model_id": "model-1",
        "latency_s": 0.01,
    }
    values.update(changes)
    return InspectionResult(**values)


def command(**changes) -> RobotCommand:
    values = {
        "command_id": "command-1",
        "issued_at_s": 2.0,
        "deadline_s": 3.0,
        "joint_targets": joints(),
        "mode": "PICK",
        "part_id": "part-1",
    }
    values.update(changes)
    return RobotCommand(**values)


class CameraFrameTest(unittest.TestCase):
    def test_valid_frame_is_frozen_and_owns_read_only_pixels(self) -> None:
        pixels = np.zeros((4, 5, 3), dtype=np.uint8)
        item = frame(rgb=pixels)
        pixels[0, 0, 0] = 99
        self.assertEqual(item.rgb[0, 0, 0], 0)
        self.assertFalse(item.rgb.flags.writeable)
        with self.assertRaises(ValueError):
            item.rgb[0, 0, 0] = 1
        with self.assertRaises(FrozenInstanceError):
            item.frame_id = "changed"

    def test_rejects_bad_image_and_timestamps(self) -> None:
        bad = (
            {"rgb": np.zeros((4, 5), dtype=np.uint8)},
            {"rgb": np.zeros((4, 5, 3), dtype=np.float32)},
            {"rgb": np.zeros((4, 5, 4), dtype=np.uint8)},
            {"timestamp_s": -0.1},
            {"received_at_s": 0.9},
            {"frame_id": " "},
            {"fresh": 1},
        )
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                frame(**changes)


class PartObservationTest(unittest.TestCase):
    def test_valid_observation_copies_mask(self) -> None:
        mask = np.ones((3, 3), dtype=np.bool_)
        item = observation(mask=mask)
        mask[0, 0] = False
        self.assertTrue(item.mask[0, 0])
        self.assertFalse(item.mask.flags.writeable)

    def test_rejects_bad_geometry_score_and_mask(self) -> None:
        bad = (
            {"roi_xyxy": (1, 1, 1, 4)},
            {"centroid_px": (100, 4)},
            {"pick_pixel": (5,)},
            {"angle_rad": float("nan")},
            {"confidence": 1.1},
            {"mask": np.ones((2, 2), dtype=np.uint8)},
            {"source_frame_id": ""},
        )
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                observation(**changes)


class InspectionResultTest(unittest.TestCase):
    def test_accepts_all_decisions(self) -> None:
        for decision in ("GOOD", "DEFECTIVE", "UNCERTAIN"):
            with self.subTest(decision=decision):
                self.assertEqual(inspection(decision=decision).decision, decision)

    def test_rejects_bad_decision_scores_and_good_error(self) -> None:
        bad = (
            {"decision": "MAYBE"},
            {"decision": []},
            {"score": -0.1},
            {"score": float("inf")},
            {"threshold": 1.1},
            {"latency_s": -0.1},
            {"model_id": ""},
            {"error": "inference failed"},
        )
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                inspection(**changes)

    def test_error_may_only_be_non_good(self) -> None:
        result = inspection(decision="UNCERTAIN", error="inference failed")
        self.assertEqual(result.decision, "UNCERTAIN")


class RobotContractTest(unittest.TestCase):
    def test_state_has_exact_named_joints_and_normalized_gripper(self) -> None:
        source = joints(0.25)
        state = RobotState(1.0, source, "IDLE", "fake")
        source["shoulder_pan"] = 1.0
        self.assertEqual(state.joint_positions["shoulder_pan"], 0.0)
        self.assertEqual(state.gripper, 0.25)
        with self.assertRaises(TypeError):
            state.joint_positions["shoulder_pan"] = 1.0

    def test_state_rejects_unknown_nonfinite_and_bad_gripper(self) -> None:
        unknown = joints()
        unknown["joint_7"] = unknown.pop("wrist_roll")
        nonfinite = joints()
        nonfinite["elbow_flex"] = float("nan")
        for values in (unknown, nonfinite, joints(-0.1), joints(1.1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RobotState(1.0, values, "IDLE", "fake")
        with self.assertRaises(ValueError):
            RobotState(-1.0, joints(), "IDLE", "fake")

    def test_command_rejects_empty_id_inverted_time_and_bad_joints(self) -> None:
        bad = (
            {"command_id": ""},
            {"issued_at_s": -1.0},
            {"deadline_s": 1.0},
            {"mode": ""},
            {"joint_targets": {name: 0.0 for name in JOINT_NAMES[:-1]}},
        )
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                command(**changes)


class SafetyResultTest(unittest.TestCase):
    def test_accept_or_stop_reason_are_mutually_exclusive(self) -> None:
        cmd = command()
        accepted = SafetyResult(cmd.command_id, 2.1, True, "ARMED", command=cmd)
        stopped = SafetyResult(cmd.command_id, 2.1, False, "TRIPPED", stop_reason="limit")
        self.assertTrue(accepted.accepted)
        self.assertEqual(stopped.reason, "limit")
        with self.assertRaises(ValueError):
            SafetyResult(cmd.command_id, 2.1, True, "ARMED")
        with self.assertRaises(ValueError):
            SafetyResult(cmd.command_id, 2.1, False, "TRIPPED")
        with self.assertRaises(ValueError):
            SafetyResult("other", 2.1, True, "ARMED", command=cmd)

    def test_rejects_bad_interlock_clamps_and_inverted_time(self) -> None:
        cmd = command()
        with self.assertRaises(ValueError):
            SafetyResult(cmd.command_id, 2.1, True, "UNKNOWN", command=cmd)
        with self.assertRaises(ValueError):
            SafetyResult(cmd.command_id, 1.9, True, "ARMED", command=cmd)
        with self.assertRaises(ValueError):
            SafetyResult(cmd.command_id, 3.1, True, "ARMED", command=cmd)
        with self.assertRaises(ValueError):
            SafetyResult(cmd.command_id, 2.1, True, "ARMED", command=cmd, applied_clamps=("",))
        with self.assertRaises(ValueError):
            SafetyResult(cmd.command_id, 2.1, True, [], command=cmd)


class CellEventTest(unittest.TestCase):
    def test_payload_is_recursively_frozen(self) -> None:
        payload = {"decision": "GOOD", "values": [1, 2.0], "nested": {"ok": True}}
        event = CellEvent(1, 1.0, 2.0, "cycle-1", "cell/result", "controller", payload, "part-1")
        payload["decision"] = "REJECT"
        self.assertEqual(event.payload["decision"], "GOOD")
        self.assertEqual(event.payload["values"], (1, 2.0))
        with self.assertRaises(TypeError):
            event.payload["decision"] = "REJECT"

    def test_rejects_bad_schema_ids_time_and_payload(self) -> None:
        valid = dict(
            schema_version=1,
            sim_time_s=1.0,
            wall_time_s=2.0,
            cycle_id="cycle-1",
            topic="cell/result",
            producer="controller",
            payload={},
        )
        bad = (
            {"schema_version": 0},
            {"sim_time_s": -1.0},
            {"wall_time_s": float("inf")},
            {"cycle_id": ""},
            {"topic": ""},
            {"payload": []},
            {"payload": {"bad": float("nan")}},
            {"part_id": ""},
        )
        for changes in bad:
            values = {**valid, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CellEvent(**values)


class Phase3ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_sim_config_has_exact_joints_limits_and_unbounded_wait(self) -> None:
        self.assertEqual(tuple(self.config["joint_names"]), JOINT_NAMES)
        self.assertEqual(self.config["joint_limits"]["shoulder_pan"], {"low": -1.91986, "high": 1.91986})
        self.assertEqual(self.config["joint_limits"]["gripper"], {"low": -0.17453, "high": 1.74533})
        self.assertIsNone(self.config["timeouts_s"]["WAIT_PART"])

    def test_target_interlock_rejects_sim_for_hardware(self) -> None:
        with self.assertRaisesRegex(ConfigError, "target"):
            load_config(expected_target="hardware")

    def test_rejects_unknown_keys_and_wrong_joint_names(self) -> None:
        unknown = copy.deepcopy(self.config)
        unknown["extra"] = True
        with self.assertRaisesRegex(ConfigError, "unknown"):
            validate_config(unknown)
        bad_names = copy.deepcopy(self.config)
        bad_names["joint_names"][-1] = "claw"
        with self.assertRaisesRegex(ConfigError, "joint_names"):
            validate_config(bad_names)
        mixed_key = copy.deepcopy(self.config)
        mixed_key[7] = mixed_key.pop("bins")
        with self.assertRaisesRegex(ConfigError, "unknown"):
            validate_config(mixed_key)

    def test_rejects_non_integer_schema_and_unknown_nested_key(self) -> None:
        schema = copy.deepcopy(self.config)
        schema["schema_version"] = 1.0
        with self.assertRaisesRegex(ConfigError, "schema_version"):
            validate_config(schema)
        nested = copy.deepcopy(self.config)
        nested["joint_limits"]["gripper"]["units"] = "rad"
        with self.assertRaisesRegex(ConfigError, "unknown"):
            validate_config(nested)

    def test_rejects_wrong_or_inverted_limits(self) -> None:
        wrong = copy.deepcopy(self.config)
        wrong["joint_limits"]["elbow_flex"]["high"] = 1.6
        with self.assertRaisesRegex(ConfigError, "MJCF"):
            validate_config(wrong)
        inverted = copy.deepcopy(self.config)
        inverted["joint_limits"]["elbow_flex"] = {"low": 2.0, "high": 1.0}
        with self.assertRaisesRegex(ConfigError, "less than"):
            validate_config(inverted)

    def test_rejects_bad_timeout_step_and_bins(self) -> None:
        timeout = copy.deepcopy(self.config)
        timeout["timeouts_s"]["CAPTURE"] = 0.0
        with self.assertRaisesRegex(ConfigError, "greater than zero"):
            validate_config(timeout)
        step = copy.deepcopy(self.config)
        step["max_joint_step_rad"] = float("nan")
        with self.assertRaisesRegex(ConfigError, "finite"):
            validate_config(step)
        bins = copy.deepcopy(self.config)
        bins["bins"]["reject"] = bins["bins"]["good"]
        with self.assertRaisesRegex(ConfigError, "differ"):
            validate_config(bins)


if __name__ == "__main__":
    unittest.main()
