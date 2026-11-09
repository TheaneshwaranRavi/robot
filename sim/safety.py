"""Safety layer between any policy and the robot (simulated or, later, real).

Pure Python on purpose: no numpy or MuJoCo, so these checks are tested in the light client image.
Every action passes through `ActionLimits.clamp`, and the control loop stops through `SafetyStop`.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


class SafetyStop(RuntimeError):
    """Raised to halt the control loop. `reason` is recorded in the episode summary."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def require_remote_ready(config: dict[str, Any]) -> None:
    """Refuse to send observations to a remote policy unless the embodiment is real.

    A placeholder config describes no actual robot, so its observations and actions are
    meaningless to a policy server and must never be exchanged with one.
    """
    if config.get("placeholder", True):
        raise SafetyStop(
            "refusing remote policy: embodiment config is a placeholder; "
            "use a real embodiment such as config/embodiment.sim.json"
        )


def require_target(config: dict[str, Any], expected: str) -> None:
    """Stop a sim config from driving hardware, and a hardware config from driving the sim."""
    target = config.get("target", "hardware")
    if target != expected:
        raise SafetyStop(f"config target is {target!r}, but this runner needs {expected!r}")


def check_finite(values: Iterable[float], label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise SafetyStop(f"non-finite value in {label}[{index}]: {value!r}")


@dataclass(frozen=True)
class ActionLimits:
    """Per-dimension bounds for one flat action vector, built from the config's action list."""

    names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ActionLimits":
        names: list[str] = []
        low: list[float] = []
        high: list[float] = []
        for action in config["action"]:
            for i in range(action["dim"]):
                names.append(f"{action['name']}[{i}]")
                low.append(float(action["low"]))
                high.append(float(action["high"]))
        return cls(tuple(names), tuple(low), tuple(high))

    @property
    def dim(self) -> int:
        return len(self.low)

    def clamp(self, action: Sequence[float]) -> list[float]:
        """Return the action clipped into bounds. Non-finite input stops the loop, never clips."""
        if len(action) != self.dim:
            raise SafetyStop(f"action has {len(action)} values, expected {self.dim}")
        check_finite(action, "action")
        return [min(max(float(v), lo), hi) for v, lo, hi in zip(action, self.low, self.high)]


def clamp_targets(
    targets: Sequence[float],
    measured: Sequence[float],
    low: Sequence[float],
    high: Sequence[float],
    max_tracking_error: float,
) -> list[float]:
    """Keep joint targets inside joint limits and within `max_tracking_error` of the measured
    position, so a commanded target can never run far ahead of where the joint actually is."""
    check_finite(targets, "joint_targets")
    check_finite(measured, "joint_positions")
    result = []
    for target, q, lo, hi in zip(targets, measured, low, high):
        target = min(max(target, q - max_tracking_error), q + max_tracking_error)
        result.append(min(max(target, lo), hi))
    return result


class Watchdog:
    """Deadline checks for policy calls and (in real-time mode) for each control step."""

    def __init__(self, policy_timeout_s: float, step_period_s: float, max_misses: int = 3,
                 realtime: bool = False, clock=time.monotonic) -> None:
        self.policy_timeout_s = policy_timeout_s
        self.step_period_s = step_period_s
        self.max_misses = max_misses
        self.realtime = realtime
        self.clock = clock
        self.consecutive_misses = 0
        self.total_misses = 0

    def check_policy_time(self, elapsed_s: float) -> None:
        if elapsed_s > self.policy_timeout_s:
            raise SafetyStop(
                f"policy took {elapsed_s * 1000:.0f} ms, deadline {self.policy_timeout_s * 1000:.0f} ms"
            )

    def check_step_time(self, elapsed_s: float) -> None:
        """Only enforced in real-time mode; offline runs may step faster or slower than real time."""
        if not self.realtime:
            return
        if elapsed_s > self.step_period_s:
            self.consecutive_misses += 1
            self.total_misses += 1
            if self.consecutive_misses >= self.max_misses:
                raise SafetyStop(
                    f"missed the {self.step_period_s * 1000:.1f} ms control deadline "
                    f"{self.consecutive_misses} times in a row"
                )
        else:
            self.consecutive_misses = 0
