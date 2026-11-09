"""Contract tests for the factory-cell MuJoCo world simulator."""
from __future__ import annotations

import math
import unittest

try:
    import mujoco
    import numpy as np
except (ImportError, OSError):
    HAS_SIM_DEPENDENCIES = False
else:
    HAS_SIM_DEPENDENCIES = True

if HAS_SIM_DEPENDENCIES:
    from cosim.protocol import load_layout
    from cosim.world_sim import PHYSICS_DT, WORLD_TOPICS, WorldSim


@unittest.skipUnless(HAS_SIM_DEPENDENCIES, "requires importable numpy and mujoco")
class WorldSimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = load_layout()
        self.world = WorldSim(self.layout, seed=7)

    def tearDown(self) -> None:
        self.world.close()

    def test_scene_builds_from_layout_and_reuses_only_the_arm(self) -> None:
        model = self.world.model
        self.assertAlmostEqual(model.opt.timestep, PHYSICS_DT)
        for item in (*self.layout["walls"], *self.layout["obstacles"]):
            with self.subTest(geom=item["name"]):
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, item["name"]), 0
                )
        for camera in ("overview", "arm_cam"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera), 0
            )
        for name in self.layout["arm"]["joint_names"]:
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name), 0
            )
        for actuator in ("a1", "a2", "a3", "a4", "a5", "a6"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator), 0
            )
        for body in ("amr", "crate", "base"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body), 0
            )

        self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table"), -1)
        self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube"), -1)
        self.assertEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera"), -1
        )
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        np.testing.assert_allclose(self.world.data.xpos[base_id], self.layout["arm"]["base"])
        rotation = self.world.data.xmat[base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        self.assertAlmostEqual(yaw, self.layout["arm"]["mount_yaw"])

    def test_lidar_hits_north_wall_from_start_pose(self) -> None:
        scan = self.world.scan(noise=False)
        self.assertEqual(len(scan["ranges"]), self.layout["lidar"]["beams"])
        forward_index = len(scan["ranges"]) // 2
        expected = 8.0 - self.layout["amr"]["start"][1]
        self.assertAlmostEqual(scan["ranges"][forward_index], expected, delta=0.05)

    def test_cmd_vel_moves_forward_and_odom_drifts_slightly(self) -> None:
        start = np.asarray(self.layout["amr"]["start"], dtype=np.float64)
        for _ in range(20):
            self.world.set_cmd_vel(0.5, 0.0)
            self.world.step_for(0.1)

        truth = self.world.truth_payload()
        odom = self.world.odom_payload()
        truth_distance = math.hypot(truth["x"] - start[0], truth["y"] - start[1])
        odom_distance = math.hypot(odom["x"] - start[0], odom["y"] - start[1])
        self.assertAlmostEqual(truth_distance, 1.0, delta=0.03)
        self.assertNotAlmostEqual(odom_distance, truth_distance, delta=0.001)
        self.assertLess(abs(odom_distance - truth_distance), truth_distance * 0.06)
        self.assertAlmostEqual(
            odom_distance / truth_distance,
            self.layout["odom"]["v_scale_error"],
            delta=0.02,
        )

    def test_arm_follows_clipped_joint_command(self) -> None:
        target = np.array([0.2, 0.1, 0.7, 0.15, 1.1, -0.2])
        self.world.handle_message("arm/joint_cmd", {"t": 0.0, "q": target.tolist()})
        self.world.step_for(1.5)
        np.testing.assert_allclose(self.world.arm_state_payload()["q"], target, atol=0.02)

        self.world.set_joint_targets([99.0, -99.0, 99.0, -99.0, 99.0, -99.0])
        np.testing.assert_allclose(
            self.world.arm_targets,
            np.array([
                self.world.arm_high[0],
                self.world.arm_low[1],
                self.world.arm_high[2],
                self.world.arm_low[3],
                self.world.arm_high[4],
                self.world.arm_low[5],
            ]),
        )

    def test_crate_load_follows_amr_and_unload_stays_on_drop_table(self) -> None:
        load_x, load_y, load_yaw = self.layout["stations"]["LOAD"]
        self.world.set_amr_pose(load_x, load_y, load_yaw)
        self.world.handle_message(
            "cell/event", {"t": self.world.time, "event": "load", "station": "LOAD"}
        )
        self.assertEqual(self.world.crate_payload()["state"], "on_amr")
        expected_z = self.layout["amr"]["deck_height"] + self.layout["crate"]["size"][2] / 2
        np.testing.assert_allclose(self.world.crate_position(), [load_x, load_y, expected_z])

        self.world.set_amr_pose(load_x + 1.0, load_y + 0.5, 0.4)
        np.testing.assert_allclose(
            self.world.crate_position()[:2], [load_x + 1.0, load_y + 0.5]
        )
        self.world.handle_event("unload", "DROP")
        drop_table = next(
            obstacle for obstacle in self.layout["obstacles"] if obstacle["name"] == "drop_table"
        )
        expected_drop = [
            *drop_table["center"],
            drop_table["h"] + self.layout["crate"]["size"][2] / 2,
        ]
        np.testing.assert_allclose(self.world.crate_position(), expected_drop)
        self.assertEqual(self.world.crate_payload()["state"], "on_drop_table")

        self.world.set_amr_pose(4.0, 4.0, 0.0)
        self.world.step()
        np.testing.assert_allclose(self.world.crate_position(), expected_drop)

    def test_cmd_vel_is_clipped_and_watchdog_stops_amr(self) -> None:
        self.world.set_cmd_vel(99.0, -99.0)
        self.assertEqual(
            self.world.commanded_velocity,
            (self.layout["amr"]["max_v"], -self.layout["amr"]["max_w"]),
        )
        self.world.set_cmd_vel(0.4, 0.0)
        start = self.world.amr_pose().copy()
        self.world.step_for(0.49)
        self.assertGreater(np.linalg.norm(self.world.amr_pose()[:2] - start[:2]), 0.15)
        self.world.step_for(0.04)
        self.assertEqual(self.world.commanded_velocity, (0.0, 0.0))
        stopped = self.world.amr_pose().copy()
        self.world.step_for(0.2)
        np.testing.assert_allclose(self.world.amr_pose(), stopped, atol=1e-8)

    def test_publish_all_emits_every_world_topic(self) -> None:
        class FakeBus:
            def __init__(self) -> None:
                self.messages = []

            def publish(self, topic, payload) -> None:
                self.messages.append((topic, payload))

        bus = FakeBus()
        self.world.publish_all(bus, rtf=0.0)
        self.assertEqual({topic for topic, _ in bus.messages}, WORLD_TOPICS)
        for topic, payload in bus.messages:
            with self.subTest(topic=topic):
                self.assertEqual(payload["t"], self.world.time)
        self.world.handle_message("sys/shutdown", {"t": self.world.time})
        self.assertTrue(self.world.shutdown_requested)


if __name__ == "__main__":
    unittest.main()
