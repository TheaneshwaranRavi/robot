"""Offline programming of the cell's arm: compute joint teach points with inverse kinematics.

Usage:  .venv/bin/python -m cosim.tools.teach [--layout PATH] [--dry-run]

Frame convention (must match world_sim)
---------------------------------------
The arm is the MJCF ``sim/assets/arm_table.xml`` (bodies/joints/actuators only). Its MJCF origin
is placed at ``layout.arm.base`` in the world and rotated by ``layout.arm.mount_yaw`` about +z:

    p_world = base + Rz(mount_yaw) @ p_arm        (arm x at yaw 0 -> world rotated by mount_yaw)
    p_arm   = Rz(-mount_yaw) @ (p_world - base)

IK is solved in the arm's own frame (the MJCF as loaded), with world targets converted through the
transform above. With mount_yaw = -pi/2 the arm's +x axis points along world -y, toward the AMR at
the LOAD station. Tool convention: the tool z axis (approach direction) points world-down and the
finger axis is the arm -x direction when j1 = 0 (same ``TOOL_DOWN`` as sim/policies/scripted.py).

Teach points (all joint angles in rad):
  home        kept as already stored in layout.json (the arm's rest pose; not solved)
  above_load  TCP up to 0.25 m above the AMR deck centre at the LOAD station, tool pointing down.
              At the layout's 0.55 m stand-off the arm cannot reach 0.25 m (the wrist would be beyond
              full extension), so the highest reachable height with a 2 cm reach margin is used.
  place_load  TCP 0.05 m above the deck centre
The deck centre is (LOAD.x, LOAD.y, amr.deck_height).

Every point is verified with forward kinematics (<= 1 cm) and against the joint limits. The result
is written into the ``arm.teach_points`` block of layout.json only (the rest of the file is kept
byte-for-byte).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..protocol import LAYOUT_PATH, load_layout

ARM_XML = Path(__file__).resolve().parents[2] / "sim" / "assets" / "arm_table.xml"
JOINTS = ["j1_base_yaw", "j2_shoulder", "j3_elbow", "j4_forearm_roll", "j5_wrist_pitch", "j6_tool_roll"]
TOOL_DOWN = np.diag([-1.0, 1.0, -1.0])  # tool z -> -z, tool x -> -x
ABOVE_HEIGHT = 0.25
PLACE_HEIGHT = 0.05
REACH_MARGIN = 0.02  # m kept free from full arm extension (avoids the singularity)
POS_TOL = 0.01  # m, acceptance of the FK check


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ArmKinematics:
    """MuJoCo kinematic copy of the arm mounted at ``layout.arm`` (base position + mount_yaw)."""

    def __init__(self, layout: dict[str, Any], xml_path: Path = ARM_XML) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        arm = layout["arm"]
        self.base = np.asarray(arm["base"], dtype=float)
        self.mount_yaw = float(arm["mount_yaw"])
        names = arm.get("joint_names", JOINTS)
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        if min(ids) < 0:
            raise ValueError(f"arm joint missing in {xml_path}")
        self.qadr = np.array([self.model.jnt_qposadr[i] for i in ids])
        self.dofadr = np.array([self.model.jnt_dofadr[i] for i in ids])
        self.low = np.array([self.model.jnt_range[i, 0] for i in ids])
        self.high = np.array([self.model.jnt_range[i, 1] for i in ids])
        self.tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    # -- frame transforms ----------------------------------------------------------------------
    def world_to_arm(self, p_world: np.ndarray) -> np.ndarray:
        return rot_z(-self.mount_yaw) @ (np.asarray(p_world, dtype=float) - self.base)

    def arm_to_world(self, p_arm: np.ndarray) -> np.ndarray:
        return self.base + rot_z(self.mount_yaw) @ np.asarray(p_arm, dtype=float)

    # -- kinematics ----------------------------------------------------------------------------
    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)

    def fk_arm(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position and orientation matrix in the arm frame."""
        self._set(np.asarray(q, dtype=float))
        return self.data.site_xpos[self.tcp].copy(), self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    def fk_world(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = self.fk_arm(q)
        return self.arm_to_world(pos), rot_z(self.mount_yaw) @ rot

    def planar_seed(self, target_world: np.ndarray, reach_margin: float = 0.0) -> np.ndarray | None:
        """Closed-form tool-down solution (j4 = j6 = 0), used to seed the DLS solver.

        The arm is a planar 2-link chain (shoulder -> elbow -> wrist pitch) plus a straight tool; with
        the tool pointing down q2 + q3 + q5 = pi. Returns None if the wrist point is out of reach
        (distance shoulder-wrist > L2 + L3 - reach_margin) or a joint limit would be violated.
        """
        m, d = self.model, self.data
        self._set(np.zeros(6))
        body = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()  # noqa: E731
        shoulder, elbow, wrist = body("link2"), body("link3"), body("link5")
        l2, l3 = np.linalg.norm(elbow - shoulder), np.linalg.norm(wrist - elbow)
        tool = d.site_xpos[self.tcp][2] - wrist[2]  # wrist-pitch axis to TCP along the tool axis
        p = self.world_to_arm(target_world)
        j1 = math.atan2(p[1], p[0])
        radial, height = math.hypot(p[0], p[1]), p[2] + tool - shoulder[2]
        dist = math.hypot(radial, height)
        if dist > l2 + l3 - reach_margin or dist < abs(l2 - l3) + 1e-6:
            return None
        bend = math.acos(max(-1.0, min(1.0, (dist ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3))))
        direction = math.atan2(radial, height)  # angle from +z toward the radial axis
        inner = math.atan2(l3 * math.sin(bend), l2 + l3 * math.cos(bend))
        q2 = direction - inner
        q3 = bend
        q5 = math.pi - q2 - q3
        q = np.array([j1, q2, q3, 0.0, q5, 0.0])
        if np.any(q < self.low) or np.any(q > self.high):
            return None
        return q

    def solve(self, target_world: np.ndarray, q0: np.ndarray, rot_world: np.ndarray | None = None,
              iterations: int = 800, damping: float = 0.05) -> np.ndarray:
        """Damped-least-squares IK (as in sim/policies/scripted.py) to a world position, tool down."""
        goal_pos = self.world_to_arm(target_world)
        if rot_world is None:
            # tool down; finger axis follows the base yaw so that j4 = j6 = 0 is a valid solution
            goal_rot = rot_z(math.atan2(goal_pos[1], goal_pos[0])) @ TOOL_DOWN
        else:
            goal_rot = rot_z(-self.mount_yaw) @ rot_world
        q = np.clip(np.asarray(q0, dtype=float), self.low, self.high)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iterations):
            self._set(q)
            pos = self.data.site_xpos[self.tcp]
            rot = self.data.site_xmat[self.tcp].reshape(3, 3)
            rot_err = 0.5 * sum(np.cross(rot[:, i], goal_rot[:, i]) for i in range(3))
            error = np.concatenate([goal_pos - pos, 0.5 * rot_err])
            if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp)
            jac = np.vstack([jacp, jacr])[:, self.dofadr]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(6), error)
            largest = np.max(np.abs(delta))
            if largest > 0.1:
                delta *= 0.1 / largest
            q = np.clip(q + delta, self.low, self.high)
        return q

    def verify(self, q: np.ndarray, target_world: np.ndarray) -> dict[str, float]:
        """FK check: position error (m), tool-down alignment error (rad), joint-limit violation (rad)."""
        pos, rot = self.fk_world(q)
        tool_z = rot[:, 2]
        tilt = math.acos(max(-1.0, min(1.0, -tool_z[2])))
        q = np.asarray(q, dtype=float)
        violation = float(max(np.max(self.low - q), np.max(q - self.high), 0.0))
        return {"pos_err": float(np.linalg.norm(pos - target_world)), "tilt": tilt, "limit_violation": violation}


def deck_centre(layout: dict[str, Any]) -> np.ndarray:
    x, y, _ = layout["stations"]["LOAD"]
    return np.array([x, y, layout["amr"]["deck_height"]])


def reachable_height(kin: ArmKinematics, deck: np.ndarray, wanted: float) -> float:
    """Highest TCP height above the deck (<= wanted) that is reachable tool-down with a reach margin.

    With the LOAD stand-off of 0.55 m the wrist is close to the arm's full extension, so the
    requested 0.25 m is not reachable (see module docstring / report); step down until it is.
    """
    height = wanted
    while height > PLACE_HEIGHT + 0.02:
        if kin.planar_seed(deck + [0, 0, height], reach_margin=REACH_MARGIN) is not None:
            return round(height, 3)
        height -= 0.005
    raise RuntimeError("no reachable above_load height")


def compute_teach_points(layout: dict[str, Any]) -> dict[str, Any]:
    """Return {name: joint list} for home/above_load/place_load plus a verification report."""
    kin = ArmKinematics(layout)
    home = np.array(layout["arm"]["teach_points"]["home"], dtype=float)
    deck = deck_centre(layout)
    targets = {"above_load": deck + [0, 0, reachable_height(kin, deck, ABOVE_HEIGHT)],
               "place_load": deck + [0, 0, PLACE_HEIGHT]}
    points: dict[str, list[float]] = {"home": [round(float(v), 6) for v in home]}
    report: dict[str, dict[str, float]] = {}
    seed = home
    for name, target in targets.items():
        start = kin.planar_seed(target)
        q = kin.solve(target, seed if start is None else start)
        q = np.round(q, 6)
        check = kin.verify(q, target)
        if check["pos_err"] > POS_TOL or check["tilt"] > 0.05 or check["limit_violation"] > 0.0:
            raise RuntimeError(f"teach point {name} failed verification: {check}")
        points[name] = [float(v) + 0.0 for v in q]
        report[name] = check
        seed = q
    return {"points": points, "report": report, "targets": {k: v.tolist() for k, v in targets.items()}}


def format_block(points: dict[str, list[float]]) -> str:
    comment = "Filled in by cosim/tools/teach.py (offline programming). Joint angles in rad."
    lines = [f'      "_comment": {json.dumps(comment)}']
    for name in ("home", "above_load", "place_load"):
        lines.append(f'      "{name}": {json.dumps(points[name])}')
    return '"teach_points": {\n' + ",\n".join(lines) + "\n    }"


def write_layout(path: Path, points: dict[str, list[float]]) -> None:
    """Replace only the ``arm.teach_points`` block of the layout file, leaving the rest verbatim."""
    text = path.read_text()
    pattern = re.compile(r'"teach_points"\s*:\s*\{[^{}]*\}')
    if len(pattern.findall(text)) != 1:
        raise RuntimeError("expected exactly one teach_points block in layout")
    new = pattern.sub(lambda _m: format_block(points), text)
    json.loads(new)  # must stay valid JSON
    path.write_text(new)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute arm teach points with IK and store them in layout.json")
    parser.add_argument("--layout", default=str(LAYOUT_PATH))
    parser.add_argument("--dry-run", action="store_true", help="print the result, do not write the layout")
    args = parser.parse_args(argv)
    layout = load_layout(args.layout)
    result = compute_teach_points(layout)
    for name, q in result["points"].items():
        info = result["report"].get(name)
        extra = (f"  target(world)={np.round(result['targets'][name], 3).tolist()} "
                 f"pos_err={info['pos_err'] * 1000:.3f} mm tilt={math.degrees(info['tilt']):.3f} deg") if info else ""
        print(f"{name:11s} {q}{extra}")
    if not args.dry_run:
        write_layout(Path(args.layout), result["points"])
        print(f"wrote arm.teach_points to {args.layout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
