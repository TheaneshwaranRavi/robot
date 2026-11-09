"""Inverse kinematics for the SO-101 tool site (phase 3).

Damped least squares on the 5 arm joints for the ``gripperframe`` site: position plus a
"tool pointing down" constraint. The site's x axis runs along the fingers (it points along +x
in the zero pose), so a top-down grasp asks for site x axis = world -z.

The solver works on a scratch MjData so it never disturbs the simulation state.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import ARM_JOINTS, SortCell

DOWN = np.array([0.0, 0.0, -1.0])
LIMIT_MARGIN = 0.03  # rad kept away from every joint limit


@dataclass(frozen=True)
class IKResult:
    q: dict[str, float]
    position_error: float      # m
    axis_error: float          # rad between tool axis and the requested direction
    converged: bool


def solve_ik(cell: SortCell, target: np.ndarray, seed: dict[str, float] | None = None,
             tool_axis: np.ndarray | None = DOWN, iterations: int = 300, tol_pos: float = 1e-3,
             tol_axis: float = 0.02, damping: float = 1e-3, axis_weight: float = 0.15) -> IKResult:
    """Solve for arm joints placing the tool site at ``target`` (world frame, metres)."""
    model = cell.model
    data = mujoco.MjData(model)
    data.qpos[:] = cell.data.qpos
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
    qadr = np.array([model.jnt_qposadr[j] for j in joint_ids])
    dadr = np.array([model.jnt_dofadr[j] for j in joint_ids])
    lo = model.jnt_range[joint_ids, 0] + LIMIT_MARGIN
    hi = model.jnt_range[joint_ids, 1] - LIMIT_MARGIN
    if seed:
        data.qpos[qadr] = [seed.get(n, data.qpos[a]) for n, a in zip(ARM_JOINTS, qadr)]
    data.qpos[qadr] = np.clip(data.qpos[qadr], lo, hi)
    target = np.asarray(target, dtype=float)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pos_err = axis_err = np.inf
    for _ in range(iterations):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        site_pos = data.site_xpos[cell.tool_site]
        x_axis = data.site_xmat[cell.tool_site].reshape(3, 3)[:, 0]
        e_pos = target - site_pos
        pos_err = float(np.linalg.norm(e_pos))
        if tool_axis is not None:
            e_rot = np.cross(x_axis, tool_axis)  # rotation that turns x_axis toward tool_axis
            axis_err = float(np.arccos(np.clip(x_axis @ tool_axis, -1.0, 1.0)))
        else:
            e_rot, axis_err = np.zeros(3), 0.0
        if pos_err < tol_pos and axis_err < tol_axis:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, cell.tool_site)
        J = jacp[:, dadr]
        err = e_pos
        if tool_axis is not None:
            J = np.vstack([J, axis_weight * jacr[:, dadr]])
            err = np.concatenate([e_pos, axis_weight * e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(J.shape[0]), err)
        data.qpos[qadr] = np.clip(data.qpos[qadr] + np.clip(dq, -0.2, 0.2), lo, hi)
    q = {n: float(data.qpos[a]) for n, a in zip(ARM_JOINTS, qadr)}
    return IKResult(q, pos_err, axis_err, bool(pos_err < tol_pos * 3 and axis_err < tol_axis * 3))


__all__ = ["DOWN", "IKResult", "solve_ik"]
