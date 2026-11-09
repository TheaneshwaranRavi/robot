"""Backend-neutral Phase 3 interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from .contracts import CameraFrame, InspectionResult, PartObservation, RobotCommand, RobotState, SafetyResult


Target = Literal["sim", "hardware", "fake"]


class CameraInterface(ABC):
    """Capture calibrated RGB frames."""

    @property
    @abstractmethod
    def target(self) -> Target:
        ...

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def capture(self) -> CameraFrame:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class Detector(ABC):
    """Inspect one localized part; internal errors must never produce GOOD."""

    @property
    @abstractmethod
    def target(self) -> Target:
        ...

    @abstractmethod
    def inspect(self, frame: CameraFrame, observation: PartObservation) -> InspectionResult:
        """Return a result or raise; callers route exceptions to UNCERTAIN/REJECT."""
        ...


class RobotInterface(ABC):
    """Command named SO-101 joint positions through a safety boundary."""

    @property
    @abstractmethod
    def target(self) -> Target:
        ...

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def read_state(self) -> RobotState:
        ...

    @abstractmethod
    def command(self, command: RobotCommand) -> SafetyResult:
        ...

    @abstractmethod
    def hold(self, timestamp_s: float) -> SafetyResult:
        ...

    @abstractmethod
    def home(self, timestamp_s: float) -> SafetyResult:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


__all__ = ["Target", "CameraInterface", "Detector", "RobotInterface"]
