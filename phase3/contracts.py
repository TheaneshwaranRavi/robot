"""Validated, immutable contracts shared by Phase 3 backends."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT_NAME = "gripper"
JOINT_NAMES = ARM_JOINT_NAMES + (GRIPPER_JOINT_NAME,)
DECISIONS = frozenset({"GOOD", "DEFECTIVE", "UNCERTAIN"})
INTERLOCK_STATES = frozenset({"ARMED", "DISARMED", "TRIPPED"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _number(
    value: Any,
    name: str,
    *,
    nonnegative: bool = False,
    unit_interval: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    if unit_interval and not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _numeric_tuple(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    try:
        items = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain {length} finite numbers") from error
    if len(items) != length:
        raise ValueError(f"{name} must contain {length} finite numbers")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(items))


def _readonly_array(value: Any, *, dtype: np.dtype, ndim: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype).name}")
    if value.ndim != ndim or any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}-D array")
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _joint_values(value: Any, name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping keyed by joint name")
    keys = set(value)
    expected = set(JOINT_NAMES)
    if keys != expected:
        missing = sorted(expected - keys, key=str)
        unknown = sorted(keys - expected, key=str)
        raise ValueError(f"{name} has missing joints {missing} and unknown joints {unknown}")
    result = {joint: _number(value[joint], f"{name}.{joint}") for joint in JOINT_NAMES}
    if not 0.0 <= result[GRIPPER_JOINT_NAME] <= 1.0:
        raise ValueError(f"{name}.gripper must be normalized between 0 and 1")
    return MappingProxyType(result)


def _freeze_payload(value: Any, name: str = "payload") -> Any:
    """Copy JSON-like data and reject values that cannot be recorded safely."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        return _number(value, name)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = _text(key, f"{name} key")
            frozen[clean_key] = _freeze_payload(item, f"{name}.{clean_key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item, f"{name}[{index}]") for index, item in enumerate(value))
    raise ValueError(f"{name} must contain only JSON-compatible values")


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One calibrated RGB capture."""

    frame_id: str
    timestamp_s: float
    rgb: np.ndarray = field(repr=False, compare=False)
    camera_id: str
    calibration_id: str
    fresh: bool = True
    received_at_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _text(self.frame_id, "frame_id"))
        timestamp = _number(self.timestamp_s, "timestamp_s", nonnegative=True)
        object.__setattr__(self, "timestamp_s", timestamp)
        rgb = _readonly_array(self.rgb, dtype=np.dtype(np.uint8), ndim=3, name="rgb")
        if rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "camera_id", _text(self.camera_id, "camera_id"))
        object.__setattr__(self, "calibration_id", _text(self.calibration_id, "calibration_id"))
        if not isinstance(self.fresh, bool):
            raise ValueError("fresh must be a bool")
        if self.received_at_s is not None:
            received = _number(self.received_at_s, "received_at_s", nonnegative=True)
            if received < timestamp:
                raise ValueError("received_at_s cannot be before timestamp_s")
            object.__setattr__(self, "received_at_s", received)

    @property
    def timestamp(self) -> float:
        return self.timestamp_s


@dataclass(frozen=True, slots=True)
class PartObservation:
    """Image-space part location and planar pick pose."""

    part_id: str
    source_frame_id: str
    timestamp_s: float
    roi_xyxy: tuple[float, float, float, float]
    centroid_px: tuple[float, float]
    angle_rad: float
    pick_pixel: tuple[float, float]
    confidence: float
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "part_id", _text(self.part_id, "part_id"))
        object.__setattr__(self, "source_frame_id", _text(self.source_frame_id, "source_frame_id"))
        object.__setattr__(self, "timestamp_s", _number(self.timestamp_s, "timestamp_s", nonnegative=True))
        roi = _numeric_tuple(self.roi_xyxy, 4, "roi_xyxy")
        x0, y0, x1, y1 = roi
        if x0 < 0.0 or y0 < 0.0 or x1 <= x0 or y1 <= y0:
            raise ValueError("roi_xyxy must be non-negative and ordered as x0, y0, x1, y1")
        object.__setattr__(self, "roi_xyxy", roi)
        centroid = _numeric_tuple(self.centroid_px, 2, "centroid_px")
        pick = _numeric_tuple(self.pick_pixel, 2, "pick_pixel")
        if any(value < 0.0 for value in (*centroid, *pick)):
            raise ValueError("centroid_px and pick_pixel must be non-negative")
        if not (x0 <= centroid[0] <= x1 and y0 <= centroid[1] <= y1):
            raise ValueError("centroid_px must lie inside roi_xyxy")
        if not (x0 <= pick[0] <= x1 and y0 <= pick[1] <= y1):
            raise ValueError("pick_pixel must lie inside roi_xyxy")
        object.__setattr__(self, "centroid_px", centroid)
        object.__setattr__(self, "pick_pixel", pick)
        object.__setattr__(self, "angle_rad", _number(self.angle_rad, "angle_rad"))
        object.__setattr__(self, "confidence", _number(self.confidence, "confidence", unit_interval=True))
        if self.mask is not None:
            object.__setattr__(
                self,
                "mask",
                _readonly_array(self.mask, dtype=np.dtype(np.bool_), ndim=2, name="mask"),
            )

    @property
    def roi(self) -> tuple[float, float, float, float]:
        return self.roi_xyxy

    @property
    def centroid(self) -> tuple[float, float]:
        return self.centroid_px

    @property
    def angle(self) -> float:
        return self.angle_rad


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """Normalized anomaly score and binary sorting decision."""

    part_id: str
    source_frame_id: str
    timestamp_s: float
    score: float
    threshold: float
    decision: str
    backend_id: str
    model_id: str
    latency_s: float
    heatmap_ref: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "part_id", _text(self.part_id, "part_id"))
        object.__setattr__(self, "source_frame_id", _text(self.source_frame_id, "source_frame_id"))
        object.__setattr__(self, "timestamp_s", _number(self.timestamp_s, "timestamp_s", nonnegative=True))
        object.__setattr__(self, "score", _number(self.score, "score", unit_interval=True))
        object.__setattr__(self, "threshold", _number(self.threshold, "threshold", unit_interval=True))
        if not isinstance(self.decision, str) or self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        object.__setattr__(self, "backend_id", _text(self.backend_id, "backend_id"))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(self, "latency_s", _number(self.latency_s, "latency_s", nonnegative=True))
        object.__setattr__(self, "heatmap_ref", _optional_text(self.heatmap_ref, "heatmap_ref"))
        error = _optional_text(self.error, "error")
        if error is not None and self.decision == "GOOD":
            raise ValueError("a result with an internal error cannot decide GOOD")
        object.__setattr__(self, "error", error)


@dataclass(frozen=True, slots=True)
class RobotState:
    """Measured SO-101 state; arm values are radians and gripper is normalized."""

    timestamp_s: float
    joint_positions: Mapping[str, float]
    motion_state: str
    backend_id: str
    fault: str | None = None
    holding_part: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _number(self.timestamp_s, "timestamp_s", nonnegative=True))
        object.__setattr__(self, "joint_positions", _joint_values(self.joint_positions, "joint_positions"))
        object.__setattr__(self, "motion_state", _text(self.motion_state, "motion_state"))
        object.__setattr__(self, "backend_id", _text(self.backend_id, "backend_id"))
        object.__setattr__(self, "fault", _optional_text(self.fault, "fault"))
        if not isinstance(self.holding_part, bool):
            raise ValueError("holding_part must be a bool")

    @property
    def joints(self) -> Mapping[str, float]:
        return self.joint_positions

    @property
    def gripper(self) -> float:
        return self.joint_positions[GRIPPER_JOINT_NAME]


@dataclass(frozen=True, slots=True)
class RobotCommand:
    """Deadline-bound named joint-position command."""

    command_id: str
    issued_at_s: float
    deadline_s: float
    joint_targets: Mapping[str, float]
    mode: str
    part_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _text(self.command_id, "command_id"))
        issued = _number(self.issued_at_s, "issued_at_s", nonnegative=True)
        deadline = _number(self.deadline_s, "deadline_s", nonnegative=True)
        if deadline < issued:
            raise ValueError("deadline_s cannot be before issued_at_s")
        object.__setattr__(self, "issued_at_s", issued)
        object.__setattr__(self, "deadline_s", deadline)
        object.__setattr__(self, "joint_targets", _joint_values(self.joint_targets, "joint_targets"))
        object.__setattr__(self, "mode", _text(self.mode, "mode"))
        object.__setattr__(self, "part_id", _optional_text(self.part_id, "part_id"))

    @property
    def joints(self) -> Mapping[str, float]:
        return self.joint_targets

    @property
    def gripper(self) -> float:
        return self.joint_targets[GRIPPER_JOINT_NAME]


@dataclass(frozen=True, slots=True)
class SafetyResult:
    """Accepted command or a recorded stop reason from the safety boundary."""

    source_command_id: str
    timestamp_s: float
    accepted: bool
    interlock_state: str
    command: RobotCommand | None = None
    stop_reason: str | None = None
    applied_clamps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_command_id", _text(self.source_command_id, "source_command_id"))
        timestamp = _number(self.timestamp_s, "timestamp_s", nonnegative=True)
        object.__setattr__(self, "timestamp_s", timestamp)
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a bool")
        if not isinstance(self.interlock_state, str) or self.interlock_state not in INTERLOCK_STATES:
            raise ValueError(f"interlock_state must be one of {sorted(INTERLOCK_STATES)}")
        reason = _optional_text(self.stop_reason, "stop_reason")
        if self.accepted:
            if not isinstance(self.command, RobotCommand) or reason is not None:
                raise ValueError("an accepted result needs a command and no stop_reason")
            if self.command.command_id != self.source_command_id:
                raise ValueError("command_id does not match source_command_id")
            if timestamp < self.command.issued_at_s:
                raise ValueError("timestamp_s cannot be before the command was issued")
            if timestamp > self.command.deadline_s:
                raise ValueError("an expired command cannot be accepted")
        elif self.command is not None or reason is None:
            raise ValueError("a rejected result needs a stop_reason and no command")
        object.__setattr__(self, "stop_reason", reason)
        if isinstance(self.applied_clamps, (str, bytes)):
            raise ValueError("applied_clamps must be a sequence of labels")
        try:
            clamps = tuple(_text(item, "applied_clamps item") for item in self.applied_clamps)
        except TypeError as error:
            raise ValueError("applied_clamps must be a sequence of labels") from error
        object.__setattr__(self, "applied_clamps", clamps)

    @property
    def reason(self) -> str | None:
        return self.stop_reason


@dataclass(frozen=True, slots=True)
class CellEvent:
    """Versioned, recordable cell event."""

    schema_version: int
    sim_time_s: float
    wall_time_s: float
    cycle_id: str
    topic: str
    producer: str
    payload: Mapping[str, Any]
    part_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, Integral):
            raise ValueError("schema_version must be a positive integer")
        if int(self.schema_version) < 1:
            raise ValueError("schema_version must be a positive integer")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "sim_time_s", _number(self.sim_time_s, "sim_time_s", nonnegative=True))
        object.__setattr__(self, "wall_time_s", _number(self.wall_time_s, "wall_time_s", nonnegative=True))
        object.__setattr__(self, "cycle_id", _text(self.cycle_id, "cycle_id"))
        object.__setattr__(self, "topic", _text(self.topic, "topic"))
        object.__setattr__(self, "producer", _text(self.producer, "producer"))
        object.__setattr__(self, "part_id", _optional_text(self.part_id, "part_id"))
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        object.__setattr__(self, "payload", _freeze_payload(self.payload))

    @property
    def sim_time(self) -> float:
        return self.sim_time_s

    @property
    def wall_time(self) -> float:
        return self.wall_time_s


__all__ = [
    "ARM_JOINT_NAMES",
    "GRIPPER_JOINT_NAME",
    "JOINT_NAMES",
    "DECISIONS",
    "INTERLOCK_STATES",
    "CameraFrame",
    "PartObservation",
    "InspectionResult",
    "RobotState",
    "RobotCommand",
    "SafetyResult",
    "CellEvent",
]
