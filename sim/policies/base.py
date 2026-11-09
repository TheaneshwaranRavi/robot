from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Policy(ABC):
    """Maps an observation dict (keyed by embodiment-config names) to an action chunk.

    The returned array has shape (action_horizon, action_dim) and is *unclamped*: the control
    loop clamps every row before it reaches the robot, whatever the backend.
    """

    name = "policy"

    def reset(self, task: str) -> None:
        self.task = task

    @abstractmethod
    def act(self, obs: dict[str, Any]) -> np.ndarray:
        ...

    def close(self) -> None:
        pass
