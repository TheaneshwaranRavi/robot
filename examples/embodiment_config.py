"""Load and validate a robot embodiment config using only the standard library."""
import json
import math
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer, got {value!r}")
    return value


def _require_name(entry: dict[str, Any], field: str, seen: set[str]) -> str:
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{field}.name must be a non-empty string")
    if name in seen:
        raise ConfigError(f"duplicate name {name!r} in {field}")
    seen.add(name)
    return name


def _require_list(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty list")
    if not all(isinstance(entry, dict) for entry in value):
        raise ConfigError(f"{field} entries must be objects")
    return value


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigError("config must be a JSON object")

    if not isinstance(config.get("embodiment_tag"), str) or not config["embodiment_tag"]:
        raise ConfigError("embodiment_tag must be a non-empty string")
    if not isinstance(config.get("placeholder"), bool):
        raise ConfigError("placeholder must be true or false")
    _require_positive_int(config.get("control_hz"), "control_hz")
    _require_positive_int(config.get("action_horizon"), "action_horizon")

    observation = config.get("observation")
    if not isinstance(observation, dict):
        raise ConfigError("observation must be an object")

    seen: set[str] = set()
    for camera in _require_list(observation.get("video"), "observation.video"):
        _require_name(camera, "observation.video", seen)
        for key in ("height", "width", "channels"):
            _require_positive_int(camera.get(key), f"observation.video.{key}")

    for state in _require_list(observation.get("state"), "observation.state"):
        _require_name(state, "observation.state", seen)
        _require_positive_int(state.get("dim"), "observation.state.dim")

    if not isinstance(observation.get("language"), str) or not observation["language"]:
        raise ConfigError("observation.language must be a non-empty string")

    action_names: set[str] = set()
    for action in _require_list(config.get("action"), "action"):
        name = _require_name(action, "action", action_names)
        _require_positive_int(action.get("dim"), "action.dim")
        low, high = action.get("low"), action.get("high")
        if not all(
            isinstance(bound, (int, float)) and not isinstance(bound, bool) and math.isfinite(bound)
            for bound in (low, high)
        ):
            raise ConfigError(f"action {name!r} needs finite numeric low and high limits")
        if low >= high:
            raise ConfigError(f"action {name!r} low must be less than high")

    return config


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    return validate_config(config)


def summarize(config: dict[str, Any]) -> list[str]:
    observation = config["observation"]
    cameras = ", ".join(
        f"{c['name']} {c['height']}x{c['width']}x{c['channels']}" for c in observation["video"]
    )
    state_dim = sum(s["dim"] for s in observation["state"])
    action_dim = sum(a["dim"] for a in config["action"])
    lines = [
        f"Embodiment: {config['embodiment_tag']} at {config['control_hz']} Hz",
        f"Cameras: {cameras}",
        f"State dim: {state_dim}",
        f"Action dim: {action_dim}, horizon {config['action_horizon']}",
    ]
    if config["placeholder"]:
        lines.append("WARNING: placeholder embodiment; not valid for hardware")
    return lines
