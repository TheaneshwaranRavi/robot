"""SO-101 sorting cell in MuJoCo (phase 3, milestone 1).

Builds the cell programmatically with MjSpec on top of the official Menagerie SO-101 model:
work table, inspection pad, GOOD and REJECT bins, one free part, an overhead inspection camera
and lights. The robot base sits at the world origin on the table top (z = 0); the arm reaches
along +x.

    from phase3.scene import build_cell, ARM_JOINTS
    cell = build_cell()            # SortCell with model, data and named indices
    rgb = cell.render("overhead")  # uint8 HxWx3

Fetch the robot model first:  python -m phase3.tools.fetch_so101
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "so101"
MODEL_PATH = ASSET_DIR / "so101.xml"

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
ALL_JOINTS = ARM_JOINTS + (GRIPPER_JOINT,)
TOOL_SITE = "gripperframe"
MATTE = "cell_matte"

# Cell layout (metres, robot base frame = world frame).
TABLE_SIZE = (0.70, 0.70)            # full x, y extent of the table top
TABLE_CENTER = (0.18, 0.0)
INSPECT_PAD = (0.22, 0.0)            # where parts are presented to the camera and picked
BIN_POSITIONS = {"GOOD": (0.12, 0.22), "REJECT": (0.12, -0.22)}
BIN_INNER = 0.10                     # inner width of a square bin
BIN_WALL_H = 0.04
PART_SIZE = (0.04, 0.03, 0.015)      # nominal plate: full x, y, z extent (graspable across y)
PART_RGBA = (0.30, 0.52, 0.80, 1.0)  # nominal anodised-blue finish
HOLE_X = (-0.011, 0.011)             # nominal hole centres along the plate's x axis
HOLE_RADIUS = 0.004
HOLE_RGBA = (0.04, 0.04, 0.05, 1.0)
N_SCRATCH_SLOTS = 3                  # hidden scratch geoms the defect generator may switch on
OVERHEAD_CAM = {"pos": (0.22, 0.0, 0.55), "fovy": 45.0}
INSPECT_CAM = {"pos": (0.22, 0.0, 0.30), "fovy": 30.0}  # ~0.6 mm/pixel at 256 x 256
# Folded "camera clear" pose: no arm link reaches past x = 0.07 m, links stay >= 6 cm above the
# table and there are no self-contacts (found by a grid search in the milestone-1 spike).
HOME_POSE = {"shoulder_pan": 0.0, "shoulder_lift": -1.70, "elbow_flex": -1.65, "wrist_flex": 0.20,
             "wrist_roll": 0.0, "gripper": 0.0}
# Highest tool-down (fingers vertical) height reachable over each station, from the spike:
# inspection pad ~0.09 m, bins ~0.07 m. Transit moves above that must not require tool-down.
TOOL_DOWN_MAX_Z = {"pad": 0.09, "bin": 0.07}


@dataclass
class SortCell:
    """Compiled cell plus the indices callers need; no control logic lives here."""

    model: mujoco.MjModel
    data: mujoco.MjData
    joint_qpos: dict[str, int]
    actuator: dict[str, int]
    tool_site: int
    part_body: int
    part_qpos: int
    _renderers: dict[tuple[int, int], mujoco.Renderer] = field(default_factory=dict, repr=False)

    def joint_positions(self) -> dict[str, float]:
        return {name: float(self.data.qpos[i]) for name, i in self.joint_qpos.items()}

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        for name, value in targets.items():
            self.data.ctrl[self.actuator[name]] = value

    def set_pose(self, q: dict[str, float]) -> None:
        """Teleport joints (and their position targets) to ``q``; for setup, not control."""
        for name, value in q.items():
            self.data.qpos[self.joint_qpos[name]] = value
            self.data.ctrl[self.actuator[name]] = value
        mujoco.mj_forward(self.model, self.data)

    def set_part_pose(self, xy: tuple[float, float], yaw: float = 0.0) -> None:
        """Place the part flat on the table at (x, y) with a yaw, at rest."""
        a = self.part_qpos
        z = PART_SIZE[2] / 2 + 1e-4
        self.data.qpos[a:a + 3] = (xy[0], xy[1], z)
        self.data.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        dof = self.model.jnt_dofadr[self.model.body_jntadr[self.part_body]]
        self.data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def tool_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tool_site].copy()

    def step(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def _renderer(self, height: int, width: int, segmentation: bool) -> mujoco.Renderer:
        key = (height, width, segmentation)
        if key not in self._renderers:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            if segmentation:
                renderer.enable_segmentation_rendering()
            self._renderers[key] = renderer
        return self._renderers[key]

    def render(self, camera: str = "overhead", width: int = 640, height: int = 480) -> np.ndarray:
        renderer = self._renderer(height, width, False)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def render_geom_ids(self, camera: str = "overhead", width: int = 640, height: int = 480) -> np.ndarray:
        """Per-pixel geom id (int32, -1 = background). Invisible (alpha 0) geoms are not drawn."""
        renderer = self._renderer(height, width, True)
        renderer.update_scene(self.data, camera=camera)
        seg = renderer.render()
        ids = seg[..., 0].astype(np.int32)
        ids[seg[..., 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1
        return ids

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


def _box(parent, name: str, size, pos, rgba, contact: bool = True, **kw):
    return parent.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in size],
                           pos=list(pos), rgba=list(rgba), contype=1 if contact else 0,
                           conaffinity=1 if contact else 0, material=MATTE, **kw)


def build_spec(model_path: Path = MODEL_PATH) -> mujoco.MjSpec:
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} missing; run: python -m phase3.tools.fetch_so101")
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.modelname = "so101_sort_cell"
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    world = spec.worldbody
    # Matte material for everything we add: MuJoCo's default specular term washes colours out
    # under the inspection camera, which would hide colour defects. Geom rgba still sets colour.
    spec.add_material(name=MATTE, specular=0.05, shininess=0.1, reflectance=0.0)
    spec.visual.headlight.diffuse = [0.2, 0.2, 0.2]
    spec.visual.headlight.ambient = [0.12, 0.12, 0.12]

    world.add_light(name="key", pos=[0.3, -0.4, 1.2], dir=[-0.2, 0.3, -1.0], diffuse=[0.35, 0.35, 0.35],
                    castshadow=True)
    world.add_light(name="fill", pos=[-0.3, 0.4, 1.0], dir=[0.3, -0.3, -1.0], diffuse=[0.15, 0.15, 0.15],
                    castshadow=False)

    # Table top surface at z = 0 (the robot base stands on it).
    tx, ty = TABLE_SIZE
    _box(world, "table", (tx, ty, 0.04), (TABLE_CENTER[0], TABLE_CENTER[1], -0.02), (0.55, 0.52, 0.48, 1))
    _box(world, "inspect_pad", (0.08, 0.08, 0.001), (INSPECT_PAD[0], INSPECT_PAD[1], 0.0005),
         (0.18, 0.19, 0.21, 1), contact=False)

    colours = {"GOOD": (0.20, 0.65, 0.30, 1), "REJECT": (0.80, 0.22, 0.20, 1)}
    for name, (bx, by) in BIN_POSITIONS.items():
        bin_body = world.add_body(name=f"bin_{name}", pos=[bx, by, 0.0])
        w, h, t = BIN_INNER, BIN_WALL_H, 0.004
        _box(bin_body, f"bin_{name}_floor", (w + 2 * t, w + 2 * t, 0.002), (0, 0, 0.001), colours[name])
        for i, (dx, dy, sx, sy) in enumerate(((1, 0, t, w + 2 * t), (-1, 0, t, w + 2 * t),
                                              (0, 1, w, t), (0, -1, w, t))):
            _box(bin_body, f"bin_{name}_wall{i}", (sx, sy, h),
                 (dx * (w + t) / 2, dy * (w + t) / 2, h / 2), colours[name])

    # Part family "plate": a rectangular plate with two drilled holes along its long (x) axis.
    # Holes and surface scratches are visual-only geoms on the top face so the defect generator
    # can show/hide/resize them per sample; the plate box is the only collision geom.
    part = world.add_body(name="part", pos=[INSPECT_PAD[0], INSPECT_PAD[1], PART_SIZE[2] / 2])
    part.add_freejoint(name="part_free")
    _box(part, "part_geom", PART_SIZE, (0, 0, 0), PART_RGBA, mass=0.02, friction=[1.2, 0.01, 0.001])
    top = PART_SIZE[2] / 2
    for i, hx in enumerate(HOLE_X):
        part.add_geom(name=f"part_hole_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[HOLE_RADIUS, 0.0004, 0], pos=[hx, 0, top], rgba=list(HOLE_RGBA),
                      contype=0, conaffinity=0, mass=0, material=MATTE)
    for i in range(N_SCRATCH_SLOTS):
        part.add_geom(name=f"part_scratch_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.008, 0.0004, 0.0003], pos=[0, 0, top + 0.0001], rgba=[0.1, 0.1, 0.1, 0],
                      contype=0, conaffinity=0, mass=0, material=MATTE)

    cx, cy, cz = OVERHEAD_CAM["pos"]
    # Close-up inspection camera straight above the pad (same orientation as the overhead one).
    ix, iy, iz = INSPECT_CAM["pos"]
    world.add_camera(name="inspect", pos=[ix, iy, iz], quat=[0.7071068, 0.0, 0.0, 0.7071068],
                     fovy=INSPECT_CAM["fovy"])
    # Looking straight down; image x along world -y, image up along world +x.
    world.add_camera(name="overhead", pos=[cx, cy, cz], quat=[0.7071068, 0.0, 0.0, 0.7071068],
                     fovy=OVERHEAD_CAM["fovy"])
    world.add_camera(name="cell_view", pos=[0.75, -0.55, 0.55],
                     xyaxes=[0.6, 0.8, 0.0, -0.45, 0.34, 0.83], fovy=50.0)
    return spec


def build_cell(model_path: Path = MODEL_PATH) -> SortCell:
    model = build_spec(model_path).compile()
    data = mujoco.MjData(model)
    joint_qpos = {}
    for name in ALL_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"SO-101 model has no joint {name!r}")
        joint_qpos[name] = int(model.jnt_qposadr[jid])
    actuator = {}
    for name in ALL_JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0 or model.actuator_trnid[aid, 0] != mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name):
            raise ValueError(f"actuator {name!r} missing or not driving joint {name!r}")
        actuator[name] = int(aid)
    part_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part")
    part_qpos = int(model.jnt_qposadr[model.body_jntadr[part_body]])
    cell = SortCell(model, data, joint_qpos, actuator,
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE), part_body, part_qpos)
    cell.set_pose(HOME_POSE)
    return cell


__all__ = ["ALL_JOINTS", "ARM_JOINTS", "BIN_POSITIONS", "GRIPPER_JOINT", "HOLE_RADIUS", "HOLE_X", "HOME_POSE", "INSPECT_CAM", "INSPECT_PAD", "N_SCRATCH_SLOTS", "PART_RGBA",
           "PART_SIZE", "SortCell", "TOOL_DOWN_MAX_Z", "build_cell", "build_spec"]
