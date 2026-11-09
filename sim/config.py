"""Shared access to the embodiment config validator in examples/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = ROOT / "config" / "embodiment.sim.json"

if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from embodiment_config import ConfigError, load_config, summarize, validate_config  # noqa: E402

__all__ = ["ROOT", "SIM_CONFIG", "ConfigError", "load_config", "summarize", "validate_config"]
