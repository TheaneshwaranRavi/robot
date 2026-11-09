"""Strict loader for the Phase 3 SO-101 configuration."""
from __future__ import annotations

import copy
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

from .contracts import JOINT_NAMES


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "phase3.sim.json"
CONTROLLER_STATES = (
    "BOOT",
    "HOME",
    "WAIT_PART",
    "CAPTURE",
    "LOCALISE",
    "INSPECT",
    "DECIDE",
    "PICK",
    "VERIFY_GRASP",
    "PLACE_GOOD",
    "PLACE_REJECT",
    "VERIFY_DROP",
    "RECORD",
    "RECOVER",
    "SAFE_STOP",
)
SO101_JOINT_LIMITS_RAD = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.17453, 1.74533),
}


class ConfigError(ValueError):
    """The Phase 3 config is missing, unsafe, or inconsistent."""


def _keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ConfigError(
            f"{label} keys mismatch: missing {sorted(expected - actual, key=str)}, "
            f"unknown {sorted(actual - expected, key=str)}"
        )


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{label} must be a finite number")
    if positive and result <= 0.0:
        raise ConfigError(f"{label} must be greater than zero")
    return result


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def validate_config(config: Mapping[str, Any], expected_target: str | None = None) -> dict[str, Any]:
    """Validate and return a defensive copy of a Phase 3 config."""
    if not isinstance(config, Mapping):
        raise ConfigError("config must be an object")
    _keys(
        config,
        {"schema_version", "target", "joint_names", "joint_limits", "max_joint_step_rad", "timeouts_s", "bins"},
        "config",
    )
    if (
        isinstance(config["schema_version"], bool)
        or not isinstance(config["schema_version"], Integral)
        or config["schema_version"] != 1
    ):
        raise ConfigError("schema_version must be 1")
    target = config["target"]
    if target not in ("sim", "hardware"):
        raise ConfigError("target must be 'sim' or 'hardware'")
    if expected_target is not None:
        if expected_target not in ("sim", "hardware"):
            raise ConfigError("expected_target must be 'sim' or 'hardware'")
        if target != expected_target:
            raise ConfigError(f"config target is {target!r}, but {expected_target!r} was requested")

    names = config["joint_names"]
    if not isinstance(names, list) or tuple(names) != JOINT_NAMES:
        raise ConfigError(f"joint_names must be exactly {list(JOINT_NAMES)} in that order")
    limits = config["joint_limits"]
    if not isinstance(limits, Mapping):
        raise ConfigError("joint_limits must be an object")
    _keys(limits, set(JOINT_NAMES), "joint_limits")
    normalized_limits: dict[str, dict[str, float]] = {}
    for joint in JOINT_NAMES:
        entry = limits[joint]
        if not isinstance(entry, Mapping):
            raise ConfigError(f"joint_limits.{joint} must be an object")
        _keys(entry, {"low", "high"}, f"joint_limits.{joint}")
        low = _finite(entry["low"], f"joint_limits.{joint}.low")
        high = _finite(entry["high"], f"joint_limits.{joint}.high")
        if low >= high:
            raise ConfigError(f"joint_limits.{joint}.low must be less than high")
        expected_low, expected_high = SO101_JOINT_LIMITS_RAD[joint]
        if not (math.isclose(low, expected_low, abs_tol=1e-9) and math.isclose(high, expected_high, abs_tol=1e-9)):
            raise ConfigError(f"joint_limits.{joint} must match the SO-101 MJCF range")
        normalized_limits[joint] = {"low": low, "high": high}

    max_step = _finite(config["max_joint_step_rad"], "max_joint_step_rad", positive=True)
    timeouts = config["timeouts_s"]
    if not isinstance(timeouts, Mapping):
        raise ConfigError("timeouts_s must be an object")
    _keys(timeouts, set(CONTROLLER_STATES), "timeouts_s")
    normalized_timeouts: dict[str, float | None] = {}
    for state in CONTROLLER_STATES:
        value = timeouts[state]
        if state in ("WAIT_PART", "SAFE_STOP"):
            if value is not None:
                raise ConfigError(f"timeouts_s.{state} must be null")
            normalized_timeouts[state] = None
        else:
            normalized_timeouts[state] = _finite(value, f"timeouts_s.{state}", positive=True)

    bins = config["bins"]
    if not isinstance(bins, Mapping):
        raise ConfigError("bins must be an object")
    _keys(bins, {"good", "reject"}, "bins")
    good = _nonempty(bins["good"], "bins.good")
    reject = _nonempty(bins["reject"], "bins.reject")
    if good == reject:
        raise ConfigError("GOOD and REJECT bin names must differ")

    return {
        "schema_version": 1,
        "target": target,
        "joint_names": list(JOINT_NAMES),
        "joint_limits": normalized_limits,
        "max_joint_step_rad": max_step,
        "timeouts_s": normalized_timeouts,
        "bins": {"good": good, "reject": reject},
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_config(
    path: str | Path = CONFIG_PATH,
    *,
    expected_target: str | None = "sim",
) -> dict[str, Any]:
    """Read JSON, reject duplicate/non-finite data, and validate the target interlock."""
    try:
        raw = json.loads(
            Path(path).read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ConfigError(f"invalid number {value}")),
        )
    except ConfigError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot load Phase 3 config: {error}") from error
    return copy.deepcopy(validate_config(raw, expected_target=expected_target))


__all__ = [
    "CONFIG_PATH",
    "CONTROLLER_STATES",
    "SO101_JOINT_LIMITS_RAD",
    "ConfigError",
    "load_config",
    "validate_config",
]
