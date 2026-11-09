"""Phase 3 milestone 1: SO-101 cell builds, maps joints exactly, stays finite, renders, and reaches."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "phase3" / "assets" / "so101" / "so101.xml"
HAVE = importlib.util.find_spec("mujoco") is not None and MODEL.exists()


@unittest.skipUnless(HAVE, "needs mujoco and the SO-101 model (python -m phase3.tools.fetch_so101)")
class SO101CellTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        import numpy as np
        from phase3 import scene
        cls.mj, cls.np, cls.scene = mujoco, np, scene
        cls.cell = scene.build_cell()

    @classmethod
    def tearDownClass(cls):
        cls.cell.close()

    def test_exact_joint_and_actuator_mapping(self):
        m = self.cell.model
        joints = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == self.mj.mjtJoint.mjJNT_HINGE]
        self.assertEqual(tuple(joints), self.scene.ALL_JOINTS)
        for name in self.scene.ALL_JOINTS:
            aid = self.cell.actuator[name]
            self.assertEqual(m.actuator(aid).name, name)
            self.assertEqual(m.joint(m.actuator_trnid[aid, 0]).name, name)

    def test_home_is_camera_clear_and_collision_free(self):
        m, d = self.cell.model, self.cell.data
        self.cell.set_pose(self.scene.HOME_POSE)
        for name in ("lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"):
            pos = d.xpos[self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_BODY, name)]
            self.assertLess(pos[0], 0.10, name)
            self.assertGreater(pos[2], 0.05, name)
        part = self.cell.part_body
        robot_contacts = [i for i in range(d.ncon)
                          if part not in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])
                          and 0 not in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])]
        self.assertEqual(robot_contacts, [])

    def test_physics_finite_and_part_rests(self):
        self.cell.set_pose(self.scene.HOME_POSE)
        self.cell.set_part_pose(self.scene.INSPECT_PAD, yaw=0.2)
        self.cell.step(2.0)
        d = self.cell.data
        self.assertTrue(self.np.isfinite(d.qpos).all() and self.np.isfinite(d.qvel).all())
        part = d.qpos[self.cell.part_qpos:self.cell.part_qpos + 3]
        self.assertAlmostEqual(part[2], self.scene.PART_SIZE[2] / 2, delta=0.002)
        for name, value in self.scene.HOME_POSE.items():
            self.assertAlmostEqual(self.cell.joint_positions()[name], value, delta=0.02, msg=name)

    def test_overhead_render_sees_the_part(self):
        self.cell.set_pose(self.scene.HOME_POSE)
        self.cell.set_part_pose(self.scene.INSPECT_PAD)
        img = self.cell.render("overhead", 320, 240)
        self.assertEqual(img.shape, (240, 320, 3))
        self.assertEqual(img.dtype, self.np.uint8)
        centre = img[110:130, 150:170].astype(float).mean(axis=(0, 1))
        self.assertGreater(centre[2], centre[0] + 40, "blue part should be at the image centre")

    def test_tool_down_reach_for_pick_and_bins(self):
        from phase3.kinematics import solve_ik
        np = self.np
        rng = np.random.default_rng(0)
        targets = {"pick": (*self.scene.INSPECT_PAD, 0.015),
                   "above_pad": (*self.scene.INSPECT_PAD, self.scene.TOOL_DOWN_MAX_Z["pad"] - 0.01)}
        for name, (x, y) in self.scene.BIN_POSITIONS.items():
            targets[f"above_{name}"] = (x, y, self.scene.TOOL_DOWN_MAX_Z["bin"] - 0.01)
        for label, target in targets.items():
            ok = False
            for _ in range(30):
                seed = {"shoulder_pan": -float(np.arctan2(target[1], target[0])),
                        "shoulder_lift": rng.uniform(-1.6, 1.6), "elbow_flex": rng.uniform(-1.6, 1.6),
                        "wrist_flex": rng.uniform(-1.6, 1.6)}
                if solve_ik(self.cell, np.array(target), seed=seed).converged:
                    ok = True
                    break
            self.assertTrue(ok, f"no tool-down IK solution for {label} {target}")


if __name__ == "__main__":
    unittest.main()
