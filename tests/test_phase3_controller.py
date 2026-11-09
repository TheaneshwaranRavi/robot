"""Fake-driven tests for the pure Phase 3 cell controller."""
from __future__ import annotations

import copy
import unittest
from typing import Any

from phase3.config import load_config
from phase3.contracts import RobotCommand
from phase3.controller import Controller, State
from phase3.fakes import FakeCamera, FakeDetector, FakeLocaliser, FakeRobot, neutral_joints


class Harness:
    """Translate semantic controller outputs into deterministic fake responses."""

    def __init__(self, decision: str, *, detector_raises: bool = False, drop_attempts=()) -> None:
        self.controller = Controller(load_config())
        self.camera = FakeCamera(seed=11)
        self.localiser = FakeLocaliser(seed=12)
        self.detector = FakeDetector([decision], seed=13, raise_on_calls={1} if detector_raises else ())
        self.robot = FakeRobot(seed=14, drop_on_pick_attempts=drop_attempts)
        self.camera.open()
        self.robot.connect()
        self.t = 0.0
        self.outputs: list[tuple[str, Any]] = []
        self.commands: list[str] = []
        self._out = self.controller.update(self.t)

    def close(self) -> None:
        self.camera.close()
        self.robot.close()

    def _robot_command(self, topic: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        action = topic.removeprefix("robot/")
        mode = action.upper()
        targets = neutral_joints(gripper=0.0 if action == "pick" else 1.0)
        command = RobotCommand(
            command_id=payload["command_id"],
            issued_at_s=self.t,
            deadline_s=self.t + 1.0,
            joint_targets=targets,
            mode=mode,
            part_id=payload.get("part_id"),
        )
        result = self.robot.command(command)
        self.commands.append(topic)
        return "robot/done", {"action": action, "ok": result.accepted, "reason": result.reason or ""}

    def _respond(self, outputs: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        inbox: list[tuple[str, Any]] = []
        for topic, payload in outputs:
            self.outputs.append((topic, payload))
            if topic == "robot/home":
                result = self.robot.home(self.t)
                inbox.append(("robot/done", {"action": "home", "ok": result.accepted, "reason": result.reason or ""}))
            elif topic == "camera/capture":
                inbox.append(("camera/frame", self.camera.capture()))
            elif topic == "localiser/localise":
                item = self.localiser.localise(payload["frame"], part_id=payload["part_id"])
                inbox.append(("localiser/result", item))
            elif topic == "detector/inspect":
                try:
                    item = self.detector.inspect(payload["frame"], payload["observation"])
                except Exception as error:
                    inbox.append(("detector/error", {"reason": str(error)}))
                else:
                    inbox.append(("detector/result", item))
            elif topic in ("robot/pick", "robot/place_good", "robot/place_reject"):
                inbox.append(self._robot_command(topic, payload))
            elif topic == "robot/verify_grasp":
                state = self.robot.read_state()
                inbox.append(("robot/grasp", {"ok": state.holding_part, "reason": "part not held"}))
            elif topic == "robot/verify_drop":
                self.robot.read_state()
                ok = self.robot.placement_verified(payload["bin"])
                inbox.append(("robot/drop", {"ok": ok, "reason": "part did not reach the selected bin"}))
            elif topic == "robot/hold":
                self.commands.append(topic)
                if payload.get("recovery"):
                    self.robot.clear_fault()
                    result = self.robot.hold(self.t)
                    inbox.append(("robot/done", {"action": "hold", "ok": result.accepted, "reason": result.reason or ""}))
        return inbox

    def run_part(self, part_id: str = "part-1") -> Controller:
        sent_part = False
        for _ in range(100):
            inbox = self._respond(self._out)
            if self.controller.state is State.WAIT_PART and not sent_part:
                inbox.append(("part/available", {"part_id": part_id}))
                sent_part = True
            if sent_part and self.controller.state in (State.WAIT_PART, State.SAFE_STOP):
                if sum(self.controller.counts.values()) > 0:
                    return self.controller
            self.t += 0.1
            self._out = self.controller.update(self.t, inbox)
        self.fail("controller cycle did not settle")

    def fail(self, message: str) -> None:
        raise AssertionError(message)


class ControllerCycleTest(unittest.TestCase):
    def run_case(self, decision: str, **kwargs) -> Harness:
        harness = Harness(decision, **kwargs)
        self.addCleanup(harness.close)
        harness.run_part()
        return harness

    def test_good_part_completes_good_cycle(self) -> None:
        harness = self.run_case("GOOD")
        controller = harness.controller
        self.assertEqual(controller.counts, {"good": 1, "rejected": 0, "failed": 0})
        self.assertIn("robot/place_good", harness.commands)
        self.assertNotIn("robot/place_reject", harness.commands)
        expected = [
            State.BOOT,
            State.HOME,
            State.WAIT_PART,
            State.CAPTURE,
            State.LOCALISE,
            State.INSPECT,
            State.DECIDE,
            State.PICK,
            State.VERIFY_GRASP,
            State.PLACE_GOOD,
            State.VERIFY_DROP,
            State.RECORD,
            State.WAIT_PART,
        ]
        self.assertEqual(controller.state_history, expected)

    def test_defective_part_goes_to_reject(self) -> None:
        harness = self.run_case("DEFECTIVE")
        self.assertEqual(harness.controller.counts, {"good": 0, "rejected": 1, "failed": 0})
        self.assertIn("robot/place_reject", harness.commands)

    def test_uncertain_part_goes_to_reject(self) -> None:
        harness = self.run_case("UNCERTAIN")
        self.assertEqual(harness.controller.rejected_parts, 1)
        self.assertIn("robot/place_reject", harness.commands)
        self.assertNotIn("robot/place_good", harness.commands)

    def test_detector_exception_goes_to_reject_never_good(self) -> None:
        harness = self.run_case("GOOD", detector_raises=True)
        self.assertEqual(harness.controller.rejected_parts, 1)
        self.assertIn("robot/place_reject", harness.commands)
        self.assertNotIn("robot/place_good", harness.commands)
        records = [payload for topic, payload in harness.outputs if topic == "record/part"]
        self.assertIn("scripted detector failure", records[-1]["inspection_error"])

    def test_one_grasp_retry_then_success(self) -> None:
        harness = self.run_case("GOOD", drop_attempts={1})
        self.assertEqual(harness.controller.good_parts, 1)
        self.assertEqual(harness.commands.count("robot/pick"), 2)
        self.assertIn(State.RECOVER, harness.controller.state_history)

    def test_repeated_grasp_failure_safe_stops_with_reason(self) -> None:
        harness = self.run_case("GOOD", drop_attempts={1, 2})
        controller = harness.controller
        self.assertEqual(controller.state, State.SAFE_STOP)
        self.assertEqual(controller.failed_parts, 1)
        self.assertIn("grasp", controller.stop_reason)
        self.assertIn("robot/hold", harness.commands)


class ControllerTimeoutTest(unittest.TestCase):
    def test_stage_timeout_retries_once_then_safe_stops(self) -> None:
        config = copy.deepcopy(load_config())
        config["timeouts_s"]["CAPTURE"] = 0.2
        controller = Controller(config)
        controller.update(0.0)
        controller.update(0.1, [("robot/done", {"action": "home", "ok": True, "reason": ""})])
        controller.update(0.2, [("part/available", {"part_id": "part-timeout"})])
        first = controller.update(0.41)
        self.assertEqual(controller.state, State.RECOVER)
        self.assertTrue(any(topic == "robot/hold" and payload["recovery"] for topic, payload in first))
        controller.update(0.5, [("robot/done", {"action": "hold", "ok": True, "reason": ""})])
        final = controller.update(0.71)
        self.assertEqual(controller.state, State.SAFE_STOP)
        self.assertEqual(controller.failed_parts, 1)
        self.assertIn("CAPTURE timeout", controller.stop_reason)
        self.assertTrue(any(topic == "cell/safe_stop" for topic, _ in final))

    def test_wait_part_has_no_timeout_and_time_is_monotonic(self) -> None:
        controller = Controller(load_config())
        controller.update(0.0)
        controller.update(0.1, [("robot/done", {"action": "home", "ok": True, "reason": ""})])
        controller.update(10_000.0)
        self.assertEqual(controller.state, State.WAIT_PART)
        with self.assertRaises(ValueError):
            controller.update(9_999.0)

    def test_inspect_timeout_is_uncertain_and_routes_to_reject(self) -> None:
        controller = Controller(load_config())
        camera = FakeCamera(seed=20)
        localiser = FakeLocaliser(seed=21)
        camera.open()
        self.addCleanup(camera.close)
        controller.update(0.0)
        controller.update(0.1, [("robot/done", {"action": "home", "ok": True, "reason": ""})])
        controller.update(0.2, [("part/available", {"part_id": "part-inspect-timeout"})])
        item = camera.capture()
        controller.update(0.3, [("camera/frame", item)])
        located = localiser.localise(item, part_id="part-inspect-timeout")
        controller.update(0.4, [("localiser/result", located)])
        out = controller.update(2.41)
        self.assertEqual(controller.state, State.PICK)
        self.assertEqual(controller.target_bin, "REJECT")
        self.assertNotIn(State.RECOVER, controller.state_history)
        self.assertTrue(any(topic == "robot/pick" for topic, _ in out))

    def test_malformed_detector_result_routes_to_reject(self) -> None:
        controller = Controller(load_config())
        camera = FakeCamera(seed=22)
        localiser = FakeLocaliser(seed=23)
        camera.open()
        self.addCleanup(camera.close)
        controller.update(0.0)
        controller.update(0.1, [("robot/done", {"action": "home", "ok": True, "reason": ""})])
        controller.update(0.2, [("part/available", {"part_id": "part-malformed"})])
        item = camera.capture()
        controller.update(0.3, [("camera/frame", item)])
        located = localiser.localise(item, part_id="part-malformed")
        controller.update(0.4, [("localiser/result", located)])
        out = controller.update(0.5, [("detector/result", {"bad": "shape"})])
        self.assertEqual(controller.state, State.PICK)
        self.assertEqual(controller.target_bin, "REJECT")
        self.assertFalse(any(topic == "robot/place_good" for topic, _ in out))


class FakeBackendTest(unittest.TestCase):
    def test_camera_is_seed_deterministic(self) -> None:
        first, second = FakeCamera(seed=7), FakeCamera(seed=7)
        first.open()
        second.open()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        self.assertTrue((first.capture().rgb == second.capture().rgb).all())

    def test_robot_can_take_a_fixed_number_of_steps_and_fault(self) -> None:
        robot = FakeRobot(seed=8, move_steps=2)
        robot.connect()
        self.addCleanup(robot.close)
        cmd = RobotCommand("move-1", 0.0, 3.0, neutral_joints(gripper=0.0), "PICK", "part-1")
        self.assertTrue(robot.command(cmd).accepted)
        self.assertEqual(robot.read_state().motion_state, "MOVING")
        final = robot.read_state()
        self.assertEqual(final.motion_state, "IDLE")
        self.assertTrue(final.holding_part)
        robot.fault_next("motor fault")
        faulted = robot.command(RobotCommand("move-2", 2.0, 3.0, neutral_joints(), "HOME"))
        self.assertFalse(faulted.accepted)
        self.assertEqual(faulted.stop_reason, "motor fault")


if __name__ == "__main__":
    unittest.main()
