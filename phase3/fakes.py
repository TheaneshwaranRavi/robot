"""Deterministic fake camera, perception, and robot backends."""
from __future__ import annotations

from numbers import Integral
from typing import Iterable, Sequence

import numpy as np

from .contracts import (
    ARM_JOINT_NAMES,
    JOINT_NAMES,
    CameraFrame,
    InspectionResult,
    PartObservation,
    RobotCommand,
    RobotState,
    SafetyResult,
)
from .interfaces import CameraInterface, Detector, RobotInterface, Target


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("seed must be an integer")
    return int(value)


def _calls(values: Iterable[int], name: str) -> frozenset[int]:
    result = frozenset(int(value) for value in values)
    if any(value < 1 for value in result):
        raise ValueError(f"{name} call numbers are one-based")
    return result


def neutral_joints(*, gripper: float = 1.0) -> dict[str, float]:
    """Return a named zero arm pose with a normalized gripper value."""
    return {**{joint: 0.0 for joint in ARM_JOINT_NAMES}, "gripper": float(gripper)}


class FakeCamera(CameraInterface):
    """Seeded RGB source with deterministic frame IDs and timestamps."""

    def __init__(
        self,
        *,
        seed: int,
        frames: Sequence[np.ndarray] | None = None,
        image_shape: tuple[int, int] = (16, 16),
        start_time_s: float = 0.0,
        period_s: float = 1.0 / 30.0,
        camera_id: str = "fake-camera",
        calibration_id: str = "fake-calibration",
    ) -> None:
        self.seed = _seed(seed)
        if len(image_shape) != 2 or any(not isinstance(v, int) or v <= 0 for v in image_shape):
            raise ValueError("image_shape must contain two positive integers")
        if not np.isfinite(start_time_s) or start_time_s < 0.0:
            raise ValueError("start_time_s must be finite and non-negative")
        if not np.isfinite(period_s) or period_s <= 0.0:
            raise ValueError("period_s must be finite and positive")
        self._rng = np.random.default_rng(self.seed)
        self._frames = tuple(np.array(frame, copy=True) for frame in frames) if frames is not None else None
        if self._frames is not None and not self._frames:
            raise ValueError("frames cannot be empty")
        self.image_shape = image_shape
        self.start_time_s = float(start_time_s)
        self.period_s = float(period_s)
        self.camera_id = camera_id
        self.calibration_id = calibration_id
        self._opened = False
        self._index = 0

    @property
    def target(self) -> Target:
        return "fake"

    def open(self) -> None:
        self._opened = True

    def capture(self) -> CameraFrame:
        if not self._opened:
            raise RuntimeError("fake camera is not open")
        if self._frames is None:
            height, width = self.image_shape
            rgb = self._rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        else:
            rgb = self._frames[self._index % len(self._frames)]
        timestamp = self.start_time_s + self._index * self.period_s
        frame = CameraFrame(
            frame_id=f"fake-frame-{self._index:06d}",
            timestamp_s=timestamp,
            received_at_s=timestamp,
            rgb=rgb,
            camera_id=self.camera_id,
            calibration_id=self.calibration_id,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        self._opened = False


class FakeLocaliser:
    """Return a centered planar observation for each fake frame."""

    def __init__(self, *, seed: int, raise_on_calls: Iterable[int] = ()) -> None:
        self.seed = _seed(seed)
        self.raise_on_calls = _calls(raise_on_calls, "raise_on_calls")
        self.calls = 0

    def localise(self, frame: CameraFrame, *, part_id: str = "part-1") -> PartObservation:
        self.calls += 1
        if self.calls in self.raise_on_calls:
            raise RuntimeError(f"scripted localiser failure on call {self.calls}")
        height, width, _ = frame.rgb.shape
        x0, y0 = (1.0 if width > 2 else 0.0), (1.0 if height > 2 else 0.0)
        x1, y1 = float(width - 1), float(height - 1)
        if x1 <= x0:
            x0, x1 = 0.0, float(width)
        if y1 <= y0:
            y0, y1 = 0.0, float(height)
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        mask = np.ones((height, width), dtype=np.bool_)
        return PartObservation(
            part_id=part_id,
            source_frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            roi_xyxy=(x0, y0, x1, y1),
            centroid_px=center,
            angle_rad=0.0,
            pick_pixel=center,
            confidence=1.0,
            mask=mask,
        )


class FakeDetector(Detector):
    """Return scripted decisions or raise on selected one-based calls."""

    def __init__(
        self,
        decisions: Sequence[str | InspectionResult],
        *,
        seed: int,
        raise_on_calls: Iterable[int] = (),
    ) -> None:
        self.seed = _seed(seed)
        if not decisions:
            raise ValueError("decisions cannot be empty")
        self.decisions = tuple(decisions)
        self.raise_on_calls = _calls(raise_on_calls, "raise_on_calls")
        self.calls = 0

    @property
    def target(self) -> Target:
        return "fake"

    def inspect(self, frame: CameraFrame, observation: PartObservation) -> InspectionResult:
        self.calls += 1
        if self.calls in self.raise_on_calls:
            raise RuntimeError(f"scripted detector failure on call {self.calls}")
        if observation.source_frame_id != frame.frame_id:
            raise ValueError("observation does not belong to frame")
        scripted = self.decisions[min(self.calls - 1, len(self.decisions) - 1)]
        if isinstance(scripted, InspectionResult):
            return scripted
        if scripted not in ("GOOD", "DEFECTIVE", "UNCERTAIN"):
            raise ValueError(f"unknown scripted decision {scripted!r}")
        score = {"GOOD": 0.1, "UNCERTAIN": 0.5, "DEFECTIVE": 0.9}[scripted]
        return InspectionResult(
            part_id=observation.part_id,
            source_frame_id=frame.frame_id,
            timestamp_s=max(frame.timestamp_s, observation.timestamp_s),
            score=score,
            threshold=0.5,
            decision=scripted,
            backend_id="fake-detector",
            model_id="scripted",
            latency_s=0.0,
        )


class FakeRobot(RobotInterface):
    """Named-joint robot that finishes immediately or after deterministic reads."""

    def __init__(
        self,
        *,
        seed: int,
        move_steps: int = 0,
        drop_on_pick_attempts: Iterable[int] = (),
    ) -> None:
        self.seed = _seed(seed)
        if isinstance(move_steps, bool) or not isinstance(move_steps, Integral) or move_steps < 0:
            raise ValueError("move_steps must be a non-negative integer")
        self.move_steps = int(move_steps)
        self.drop_on_pick_attempts = _calls(drop_on_pick_attempts, "drop_on_pick_attempts")
        self._connected = False
        self._timestamp_s = 0.0
        self._joints = neutral_joints()
        self._motion_state = "IDLE"
        self._fault: str | None = None
        self._fault_next: str | None = None
        self._drop_next = False
        self._holding_part = False
        self._pick_attempts = 0
        self._pending: tuple[RobotCommand, bool] | None = None
        self._remaining_steps = 0
        self._command_number = 0
        self._placements: list[str] = []
        self._last_placement_ok = False
        self._last_placement: str | None = None

    @property
    def target(self) -> Target:
        return "fake"

    @property
    def holding_part(self) -> bool:
        return self._holding_part

    @property
    def placements(self) -> tuple[str, ...]:
        return tuple(self._placements)

    @property
    def last_placement_ok(self) -> bool:
        return self._last_placement_ok

    def placement_verified(self, bin_name: str) -> bool:
        return self._last_placement_ok and self._last_placement == bin_name

    def connect(self) -> None:
        self._connected = True

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("fake robot is not connected")

    def read_state(self) -> RobotState:
        self._require_connected()
        if self._pending is not None:
            self._timestamp_s += 1.0
            self._remaining_steps -= 1
            if self._remaining_steps <= 0:
                command, drop = self._pending
                self._pending = None
                self._complete(command, drop)
        return RobotState(
            timestamp_s=self._timestamp_s,
            joint_positions=self._joints,
            motion_state=self._motion_state,
            backend_id="fake-robot",
            fault=self._fault,
            holding_part=self._holding_part,
        )

    def command(self, command: RobotCommand) -> SafetyResult:
        self._require_connected()
        if not isinstance(command, RobotCommand):
            raise ValueError("command must be a RobotCommand")
        self._timestamp_s = max(self._timestamp_s, command.issued_at_s)
        if self._pending is not None:
            return self._rejected(command, "another command is still moving")
        if self._fault is not None:
            return self._rejected(command, self._fault)
        if self._fault_next is not None:
            self._fault, self._fault_next = self._fault_next, None
            self._motion_state = "FAULT"
            return self._rejected(command, self._fault)

        drop = False
        if command.mode.upper() == "PICK":
            self._pick_attempts += 1
            drop = self._drop_next or self._pick_attempts in self.drop_on_pick_attempts
            self._drop_next = False
        if self.move_steps == 0:
            self._complete(command, drop)
        else:
            self._pending = (command, drop)
            self._remaining_steps = self.move_steps
            self._motion_state = "MOVING"
        return SafetyResult(
            source_command_id=command.command_id,
            timestamp_s=self._timestamp_s,
            accepted=True,
            interlock_state="ARMED",
            command=command,
        )

    def _complete(self, command: RobotCommand, drop: bool) -> None:
        self._joints = dict(command.joint_targets)
        mode = command.mode.upper()
        if mode == "PICK":
            self._holding_part = not drop
            self._last_placement_ok = False
            self._last_placement = None
        elif mode in ("PLACE_GOOD", "PLACE_REJECT"):
            bin_name = mode.removeprefix("PLACE_")
            self._last_placement_ok = self._holding_part
            self._last_placement = bin_name if self._last_placement_ok else None
            if self._last_placement_ok:
                self._placements.append(bin_name)
            self._holding_part = False
        self._motion_state = "IDLE"

    def _rejected(self, command: RobotCommand, reason: str) -> SafetyResult:
        return SafetyResult(
            source_command_id=command.command_id,
            timestamp_s=self._timestamp_s,
            accepted=False,
            interlock_state="TRIPPED" if self._fault else "ARMED",
            stop_reason=reason,
        )

    def _local_command(self, mode: str, timestamp_s: float, targets: dict[str, float]) -> RobotCommand:
        self._command_number += 1
        return RobotCommand(
            command_id=f"fake-{mode.lower()}-{self._command_number}",
            issued_at_s=timestamp_s,
            deadline_s=timestamp_s + 1.0,
            joint_targets=targets,
            mode=mode,
        )

    def hold(self, timestamp_s: float) -> SafetyResult:
        return self.command(self._local_command("HOLD", timestamp_s, dict(self._joints)))

    def home(self, timestamp_s: float) -> SafetyResult:
        return self.command(self._local_command("HOME", timestamp_s, neutral_joints()))

    def fault_next(self, reason: str = "scripted robot fault") -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be non-empty")
        self._fault_next = reason

    def clear_fault(self) -> None:
        self._fault = None
        self._motion_state = "IDLE"

    def drop_next_part(self) -> None:
        self._drop_next = True

    def drop_part(self) -> None:
        """Drop a currently held part before placement verification."""
        self._require_connected()
        self._holding_part = False
        self._last_placement_ok = False
        self._last_placement = None

    def close(self) -> None:
        self._connected = False
        self._pending = None
        self._motion_state = "IDLE"


__all__ = ["FakeCamera", "FakeLocaliser", "FakeDetector", "FakeRobot", "neutral_joints"]
