"""Scripted pick-and-place expert.

It is *privileged*: it reads the cube pose straight from the simulator, which a learned policy
could not. It still emits the same joint-delta actions a learned policy would, by planning each
chunk with damped-least-squares IK on a kinematic copy of the arm. Its episodes can therefore
serve as demonstrations later.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any

import mujoco
import numpy as np

from ..env import SimEnv
from .base import Policy


class Phase(Enum):
    APPROACH = "approach"  # above the cube, gripper open
    DESCEND = "descend"    # fingers around the cube
    CLOSE = "close"        # squeeze
    LIFT = "lift"          # raise and hold


HOVER = 0.12        # m above the grasp point
LIFT = 0.15         # m above the grasp point
GRASP_Z = 0.005     # tcp height above the cube centre when grasping
CLOSE_CALLS = 2     # policy calls spent closing (2 × execute_k control steps)
REACHED = 0.008     # m position error that counts as arrived


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Tool frame pointing down: tool z → world -z, tool x (finger axis) → world -x.
TOOL_DOWN = np.diag([-1.0, 1.0, -1.0])


class ScriptedPickPolicy(Policy):
    name = "scripted"

    def __init__(self, env: SimEnv, horizon: int, max_delta: float, damping: float = 0.05) -> None:
        self.env = env
        self.horizon = horizon
        self.max_delta = max_delta
        self.damping = damping
        self.plan_data = mujoco.MjData(env.model)
        self.phase = Phase.APPROACH
        self.close_calls = 0

    def reset(self, task: str) -> None:
        super().reset(task)
        self.phase = Phase.APPROACH
        self.close_calls = 0

    # -- goal selection ----------------------------------------------------------------------

    def _goal(self) -> tuple[np.ndarray, np.ndarray, float]:
        cube, yaw = self.env.cube_pose()
        # The cube is square: align the fingers with the nearest face (yaw mod 90°).
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
        rotation = _rot_z(yaw) @ TOOL_DOWN
        grasp = cube + np.array([0.0, 0.0, GRASP_Z])
        if self.phase is Phase.APPROACH:
            return grasp + [0, 0, HOVER], rotation, 0.0
        if self.phase is Phase.DESCEND:
            return grasp, rotation, 0.0
        if self.phase is Phase.CLOSE:
            return grasp, rotation, 1.0
        lift_point = np.array([cube[0], cube[1], self.env.cube_rest_z + GRASP_Z + LIFT])
        return lift_point, rotation, 1.0

    def _advance_phase(self) -> None:
        """Phase changes are decided from the *real* simulator state at the start of each call."""
        goal, _, _ = self._goal()
        error = np.linalg.norm(goal - self.env.tcp_pos())
        if self.phase is Phase.APPROACH and error < REACHED:
            self.phase = Phase.DESCEND
        elif self.phase is Phase.DESCEND and error < REACHED:
            self.phase = Phase.CLOSE
        elif self.phase is Phase.CLOSE:
            self.close_calls += 1
            if self.close_calls > CLOSE_CALLS:
                self.phase = Phase.LIFT

    # -- IK planning ---------------------------------------------------------------------------

    def _ik_delta(self, goal_pos: np.ndarray, goal_rot: np.ndarray) -> np.ndarray:
        env, d = self.env, self.plan_data
        site = env.tcp_site
        pos = d.site_xpos[site]
        rot = d.site_xmat[site].reshape(3, 3)
        pos_err = goal_pos - pos
        rot_err = 0.5 * sum(np.cross(rot[:, i], goal_rot[:, i]) for i in range(3))
        error = np.concatenate([pos_err, 0.5 * rot_err])
        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, d, jacp, jacr, site)
        jac = np.vstack([jacp, jacr])[:, env.arm_dofadr]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + self.damping ** 2 * np.eye(6), error)
        largest = np.max(np.abs(delta))
        if largest > self.max_delta:
            delta *= self.max_delta / largest
        return delta

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        self._advance_phase()
        goal_pos, goal_rot, gripper = self._goal()
        env, d = self.env, self.plan_data
        d.qpos[:] = env.data.qpos
        q = np.asarray(obs["arm_joint_positions"], dtype=np.float64).copy()
        chunk = np.zeros((self.horizon, 7))
        for i in range(self.horizon):
            d.qpos[env.arm_qadr] = q
            mujoco.mj_fwdPosition(env.model, d)
            delta = self._ik_delta(goal_pos, goal_rot)
            q = np.clip(q + delta, env.joint_low, env.joint_high)
            chunk[i, :6] = delta
            chunk[i, 6] = gripper
        return chunk
