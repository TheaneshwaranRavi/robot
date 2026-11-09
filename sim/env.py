"""MuJoCo tabletop environment whose observations and actions follow the embodiment config."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .safety import SafetyStop, clamp_targets

MODEL_PATH = Path(__file__).resolve().parent / "assets" / "arm_table.xml"

ARM_JOINTS = (
    "j1_base_yaw", "j2_shoulder", "j3_elbow", "j4_forearm_roll", "j5_wrist_pitch", "j6_tool_roll",
)
FINGER_JOINT = "finger_left"
# Tool pointing straight down, about 0.27 m above the table in front of the robot.
HOME_QPOS = (0.0, 0.343, 1.302, 0.0, 1.497, 0.0)
CUBE_HALF = 0.025
TASK = "pick up the red cube"


class SimEnv:
    """Owns the MuJoCo model, physics substeps and offscreen rendering.

    The arm is position-controlled. Each action carries joint *deltas*: they are added to the
    current joint targets, which are then clamped to the joint limits and to a maximum distance
    from the measured joint positions (see `safety.clamp_targets`).
    """

    def __init__(self, config: dict[str, Any], max_tracking_error: float = 0.25) -> None:
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.control_hz = config["control_hz"]
        self.substeps = max(1, round(1.0 / (self.control_hz * self.model.opt.timestep)))
        # Make substeps × timestep exactly one control period.
        self.model.opt.timestep = 1.0 / (self.control_hz * self.substeps)
        self.max_tracking_error = max_tracking_error

        m = self.model
        self.arm_joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dofadr = np.array([m.jnt_dofadr[j] for j in self.arm_joint_ids])
        finger = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, FINGER_JOINT)
        self.finger_qadr = m.jnt_qposadr[finger]
        self.finger_range = m.jnt_range[finger].copy()
        self.cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_qadr = m.jnt_qposadr[m.body_jntadr[self.cube_body]]
        self.tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        self._check_config_matches_model()
        self.joint_low = list(self.model.jnt_range[self.arm_joint_ids, 0])
        self.joint_high = list(self.model.jnt_range[self.arm_joint_ids, 1])

        cameras = config["observation"]["video"]
        self.cameras = [(c["name"], c["height"], c["width"]) for c in cameras]
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.targets = np.array(HOME_QPOS)
        self.cube_rest_z = CUBE_HALF
        self.task = TASK

    # -- setup -------------------------------------------------------------------------------

    def _check_config_matches_model(self) -> None:
        """The config is the contract with the policy; the model must agree with it exactly."""
        limits = self.config.get("joint_limits", {}).get("arm_joint_positions")
        if limits is None:
            raise SafetyStop("sim config needs joint_limits.arm_joint_positions")
        model_range = self.model.jnt_range[self.arm_joint_ids]
        if not (np.allclose(model_range[:, 0], limits["low"]) and np.allclose(model_range[:, 1], limits["high"])):
            raise SafetyStop("joint_limits in the config do not match the MJCF joint ranges")
        state_dims = {s["name"]: s["dim"] for s in self.config["observation"]["state"]}
        action_dims = {a["name"]: a["dim"] for a in self.config["action"]}
        expected_state = {"arm_joint_positions": 6, "gripper_position": 1}
        expected_action = {"arm_joint_delta": 6, "gripper_command": 1}
        if state_dims != expected_state or action_dims != expected_action:
            raise SafetyStop(
                f"config spaces {state_dims} / {action_dims} do not match this robot "
                f"{expected_state} / {expected_action}"
            )

    def reset(self, seed: int = 0) -> dict[str, Any]:
        """Home the arm, open the gripper and place the cube at a seeded random pose."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = HOME_QPOS
        self.targets = np.array(HOME_QPOS)
        x = rng.uniform(0.38, 0.50)
        y = rng.uniform(-0.10, 0.10)
        yaw = rng.uniform(-0.4, 0.4)
        self.data.qpos[self.cube_qadr:self.cube_qadr + 3] = (x, y, CUBE_HALF)
        self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.data.ctrl[:6] = self.targets
        self.data.ctrl[6] = self.finger_range[0]
        mujoco.mj_forward(self.model, self.data)
        # Let the cube settle onto the table before the episode starts.
        for _ in range(self.substeps * 5):
            mujoco.mj_step(self.model, self.data)
        self.cube_rest_z = float(self.data.xpos[self.cube_body][2])
        return self.observe()

    # -- observation / action ----------------------------------------------------------------

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    def gripper_position(self) -> float:
        lo, hi = self.finger_range
        return float((self.data.qpos[self.finger_qadr] - lo) / (hi - lo))

    def render(self, camera: str, height: int, width: int) -> np.ndarray:
        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height, width)
        renderer = self._renderers[key]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def observe(self, render: bool = True) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "arm_joint_positions": self.arm_qpos().astype(np.float32),
            "gripper_position": np.array([self.gripper_position()], dtype=np.float32),
            "task_description": self.task,
        }
        for name, height, width in self.cameras:
            obs[name] = (
                self.render(name, height, width) if render
                else np.zeros((height, width, 3), dtype=np.uint8)
            )
        return obs

    def apply(self, action: list[float]) -> None:
        """Apply one clamped 7-D action: 6 joint deltas, then the gripper command (0 open, 1 closed)."""
        measured = self.arm_qpos()
        self.targets = np.array(clamp_targets(
            list(self.targets + np.asarray(action[:6])), list(measured),
            self.joint_low, self.joint_high, self.max_tracking_error,
        ))
        lo, hi = self.finger_range
        self.data.ctrl[:6] = self.targets
        self.data.ctrl[6] = lo + float(action[6]) * (hi - lo)

    def step(self) -> None:
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qacc))):
            raise SafetyStop("physics became unstable (non-finite state)")

    # -- task state (privileged; used by the scripted expert and success check) --------------

    def cube_pose(self) -> tuple[np.ndarray, float]:
        pos = self.data.xpos[self.cube_body].copy()
        q = self.data.xquat[self.cube_body]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return pos, yaw

    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    def cube_lifted(self, height: float = 0.05) -> bool:
        cube, _ = self.cube_pose()
        held = np.linalg.norm(cube - self.tcp_pos()) < 0.04
        return bool(held and cube[2] > self.cube_rest_z + height)

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
