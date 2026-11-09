"""Navigator tests with synthetic data only (no MuJoCo, no bus, no other processes)."""
from __future__ import annotations

import importlib.util
import itertools
import math
import unittest

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_BUS = all(importlib.util.find_spec(m) for m in ("zmq", "msgpack"))

if HAVE_NUMPY:
    import numpy as np

    from cosim.nav_astar import astar, plan_to_pose
    from cosim.nav_grid import Grid, build_navmap, distance_field, inflate
    from cosim.nav_pf import ParticleFilter, PFConfig, compose, relative_motion
    from cosim.nav_pursuit import FollowerConfig, PathFollower, simulate_unicycle, wrap
    from cosim.nav_scan import beam_angles, forward_min, raycast

LAYOUT_PATH = None
if HAVE_NUMPY:
    from pathlib import Path
    import json
    LAYOUT_PATH = Path(__file__).resolve().parents[1] / "cosim" / "layout.json"


def load_layout():
    return json.loads(LAYOUT_PATH.read_text())


def lidar_angles(layout):
    lid = layout["lidar"]
    n = lid["beams"]
    return -lid["fov"] / 2, lid["fov"] / (n - 1), n


_NAVMAP = None


def shared_navmap():
    global _NAVMAP
    if _NAVMAP is None:
        _NAVMAP = build_navmap(load_layout())
    return _NAVMAP


def rect_cells(grid, center, size):
    (cx, cy), (sx, sy) = center, size
    xs = np.linspace(cx - sx / 2 + 0.01, cx + sx / 2 - 0.01, 7)
    ys = np.linspace(cy - sy / 2 + 0.01, cy + sy / 2 - 0.01, 7)
    X, Y = np.meshgrid(xs, ys)
    return grid.world_to_cell(X.ravel(), Y.ravel())


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class GridTest(unittest.TestCase):
    def setUp(self):
        self.layout = load_layout()
        self.m = shared_navmap()

    def test_grid_marks_walls_and_obstacles(self):
        g = self.m.grid
        self.assertAlmostEqual(g.res, 0.05)
        for item in self.layout["walls"] + self.layout["obstacles"]:
            ix, iy = rect_cells(g, item["center"], item["size"])
            self.assertTrue(g.occ[iy, ix].all(), item["name"])
        # open floor, stations and the start pose are free and not lethal
        for name, pose in self.layout["stations"].items():
            if name.startswith("_"):
                continue
            x, y = pose[0], pose[1]
            ix, iy = g.world_to_cell(x, y)
            self.assertFalse(g.occ[iy, ix], name)
            self.assertTrue(np.isfinite(self.m.cost[iy, ix]), name)
        ix, iy = g.world_to_cell(6.5, 4.0)
        self.assertFalse(self.m.inflated[iy, ix])

    def test_inflation_radius(self):
        fx, fy = self.layout["amr"]["footprint"]
        self.assertAlmostEqual(self.m.inflation_radius, math.hypot(fx / 2, fy / 2) + 0.05)
        g = self.m.grid
        # rack_2 spans y 1.7..2.3; points closer than the radius to its face are inflated, farther ones not
        r = self.m.inflation_radius
        for dy, expect in ((r - 0.06, True), (r + 0.06, False)):
            ix, iy = g.world_to_cell(5.0, 2.3 + dy)
            self.assertEqual(bool(self.m.inflated[iy, ix]), expect, dy)

    def test_distance_field_close_to_euclidean(self):
        rng = np.random.default_rng(0)
        occ = rng.random((40, 50)) < 0.03
        d = distance_field(occ, 1.0, max_dist=60.0)
        oy, ox = np.nonzero(occ)
        yy, xx = np.mgrid[0:40, 0:50]
        exact = np.sqrt((xx[..., None] - ox) ** 2 + (yy[..., None] - oy) ** 2).min(axis=-1)
        self.assertTrue(np.all(d >= exact - 1e-9))
        self.assertLess(float(np.max(d - exact)), 0.03 * exact.max() + 0.1)

    def test_inflate_disk(self):
        occ = np.zeros((21, 21), dtype=bool)
        occ[10, 10] = True
        out = inflate(occ, 0.25, 0.05)  # radius 5 cells
        self.assertTrue(out[10, 15] and out[15, 10] and out[13, 14])
        self.assertFalse(out[10, 16] or out[14, 14])


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.layout = load_layout()
        self.m = shared_navmap()

    def test_astar_small_grid(self):
        cost = np.ones((10, 10))
        cost[1:10, 5] = np.inf           # wall with a gap at the top row
        path = astar(cost, (1, 8), (8, 8))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (1, 8))
        self.assertEqual(path[-1], (8, 8))
        self.assertIn((5, 0), path)
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            self.assertLessEqual(max(abs(x1 - x0), abs(y1 - y0)), 1)
        cost[0, 5] = np.inf
        self.assertIsNone(astar(cost, (1, 8), (8, 8)))

    def test_all_station_pairs_and_racks_avoided(self):
        g, st = self.m.grid, self.layout["stations"]
        names = [k for k in st if not k.startswith("_")]
        racks = [o for o in self.layout["obstacles"] if o["name"].startswith("rack")]
        for a, b in itertools.permutations(names, 2):
            with self.subTest(a=a, b=b):
                res = plan_to_pose(g, self.m.cost, st[a][:2], st[b])
                self.assertIsNotNone(res)
                pts, _ = res
                np.testing.assert_allclose(pts[0], st[a][:2], atol=1e-9)
                np.testing.assert_allclose(pts[-1], st[b][:2], atol=1e-9)
                self.assertLessEqual(float(np.max(np.hypot(*np.diff(pts, axis=0).T))), 0.0501)
                ix, iy = g.world_to_cell(pts[:, 0], pts[:, 1])
                self.assertTrue(np.all(np.isfinite(self.m.cost[iy, ix])), "path enters a lethal cell")
                # away from the endpoints the path stays outside the contract's inflated grid
                far = (np.hypot(pts[:, 0] - st[a][0], pts[:, 1] - st[a][1]) > 0.6) & \
                      (np.hypot(pts[:, 0] - st[b][0], pts[:, 1] - st[b][1]) > 0.6)
                self.assertFalse(self.m.inflated[iy[far], ix[far]].any())
                for rack in racks:
                    (cx, cy), (sx, sy) = rack["center"], rack["size"]
                    dx = np.maximum(np.abs(pts[:, 0] - cx) - sx / 2, 0)
                    dy = np.maximum(np.abs(pts[:, 1] - cy) - sy / 2, 0)
                    self.assertGreater(float(np.hypot(dx, dy).min()), self.m.inflation_radius - 0.05,
                                       rack["name"])

    def test_docking_approach_is_straight_along_goal_yaw(self):
        st = self.layout["stations"]
        pts, dock = plan_to_pose(self.m.grid, self.m.cost, st["HOME"][:2], st["LOAD"])
        self.assertIsNotNone(dock)
        tail = pts[dock:]
        heading = np.arctan2(np.diff(tail[:, 1]), np.diff(tail[:, 0]))
        self.assertTrue(np.all(np.abs(wrap_arr(heading - st["LOAD"][2])) < 1e-6))
        self.assertAlmostEqual(float(np.hypot(*(tail[-1] - tail[0]))), 0.5, places=6)


def wrap_arr(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_follower(path, start, goal_pose, dock=None, steps=2000, dt=0.05):
    f = PathFollower(FollowerConfig(max_v=0.6, max_w=1.2))
    f.set_path(path, goal_pose, dock)
    pose = start
    traj, cmds = [pose], []
    for _ in range(steps):
        v, w = f.step(pose)
        cmds.append((v, w))
        if f.phase == "done":
            break
        pose = simulate_unicycle(pose, v, w, dt)
        traj.append(pose)
    return f, pose, np.array(traj), np.array(cmds)


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class PursuitTest(unittest.TestCase):
    def assert_arrived(self, f, pose, goal):
        self.assertEqual(f.phase, "done")
        self.assertLess(math.hypot(pose[0] - goal[0], pose[1] - goal[1]), 0.05)
        self.assertLess(abs(wrap(pose[2] - goal[2])), math.radians(3))

    def test_straight_path_converges(self):
        path = np.stack([np.linspace(0, 4, 81), np.zeros(81)], axis=1)
        goal = (4.0, 0.0, math.pi / 2)
        f, pose, traj, cmds = run_follower(path, (0.0, 0.3, 0.4), goal)
        self.assert_arrived(f, pose, goal)
        late = traj[traj[:, 0] > 1.5]
        self.assertLess(float(np.abs(late[:, 1]).max()), 0.03)  # cross-track error after convergence
        self.assertLessEqual(float(np.abs(cmds[:, 0]).max()), 0.6 + 1e-9)
        self.assertLessEqual(float(np.abs(cmds[:, 1]).max()), 1.2 + 1e-9)

    def test_curved_path_converges(self):
        th = np.linspace(-math.pi / 2, math.pi / 2, 160)
        path = np.stack([1.5 * np.cos(th), 1.5 + 1.5 * np.sin(th)], axis=1)  # half circle r=1.5
        goal = (path[-1, 0], path[-1, 1], math.pi)
        f, pose, traj, cmds = run_follower(path, (0.1, -0.1, 0.0), goal)
        self.assert_arrived(f, pose, goal)
        radial = np.abs(np.hypot(traj[:, 0], traj[:, 1] - 1.5) - 1.5)
        self.assertLess(float(radial[len(radial) // 4:].max()), 0.08)

    def test_docking_path_arrives_aligned(self):
        layout = load_layout()
        m = shared_navmap()
        st = layout["stations"]
        pts, dock = plan_to_pose(m.grid, m.cost, st["HOME"][:2], st["LOAD"])
        f, pose, traj, _ = run_follower(pts, tuple(st["HOME"]), tuple(st["LOAD"]), dock, steps=4000)
        self.assert_arrived(f, pose, st["LOAD"])
        # on the final 0.3 m the AMR already points along the goal yaw (no big turn near the pedestal)
        near = traj[np.hypot(traj[:, 0] - st["LOAD"][0], traj[:, 1] - st["LOAD"][1]) < 0.3]
        self.assertLess(float(np.abs(wrap_arr(near[:, 2] - st["LOAD"][2])).max()), math.radians(10))


def noisy_odom_step(odom, v, w, dt, rng, scale_v=1.03, scale_w=0.97):
    return simulate_unicycle(odom, v * scale_v + rng.normal(0, 0.02), w * scale_w + rng.normal(0, 0.03), dt)


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class ParticleFilterTest(unittest.TestCase):
    def test_pose_algebra(self):
        p0, p1 = (1.0, 2.0, 0.7), (1.4, 2.9, -0.4)
        q = compose(p0, relative_motion(p0, p1))
        np.testing.assert_allclose(q, p1, atol=1e-12)

    def test_tracks_10m_trajectory(self):
        layout = load_layout()
        m = shared_navmap()
        amin, ainc, n = lidar_angles(layout)
        angles = beam_angles(amin, ainc, n)
        rng = np.random.default_rng(7)
        pf = ParticleFilter(m.grid, m.dist, layout["lidar"]["range_max"], PFConfig(), seed=3)
        truth = tuple(layout["amr"]["start"])
        pf.init(truth)
        odom = truth
        last_odom = odom
        # ~10 m: 5.4 m north up the east aisle, turn left in place, 4.6 m west along y = 6.6
        # (between rack_1 and the north wall) with gentle S-curves
        legs = [(0.5, 0.0, 10.8), (0.0, 0.8, math.pi / 2 / 0.8), (0.5, 0.04, 3.0), (0.5, -0.08, 3.0),
                (0.5, 0.04, 3.2)]
        dt, t, dist = 0.02, 0.0, 0.0
        errs = []
        for v, w, dur in legs:
            for _ in range(int(round(dur / dt))):
                truth = simulate_unicycle(truth, v, w, dt)
                odom = noisy_odom_step(odom, v, w, dt, rng)
                dist += abs(v) * dt
                t += dt
                if int(round(t / dt)) % 5 == 0:   # 10 Hz scans
                    pf.predict(relative_motion(last_odom, odom))
                    last_odom = odom
                    ranges = raycast(m.grid, truth, angles, layout["lidar"]["range_max"])
                    ranges = ranges + rng.normal(0, layout["lidar"]["noise_std"], n)
                    pf.correct(ranges, angles)
                    (x, y, yaw), _, _ = pf.estimate()
                    errs.append(math.hypot(x - truth[0], y - truth[1]))
        self.assertGreater(dist, 9.95)
        (x, y, yaw), std_xy, std_yaw = pf.estimate()
        odom_err = math.hypot(odom[0] - truth[0], odom[1] - truth[1])
        final = math.hypot(x - truth[0], y - truth[1])
        self.assertLess(final, 0.10, f"final error {final:.3f} m (odom alone {odom_err:.3f} m)")
        self.assertLess(abs(wrap(yaw - truth[2])), math.radians(3))
        self.assertLess(max(errs), 0.15)
        self.assertGreater(odom_err, final)  # the filter beats dead reckoning
        self.assertLess(std_xy, 0.1)


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class ScanTest(unittest.TestCase):
    def test_raycast_and_forward_sector(self):
        occ = np.zeros((40, 40), dtype=bool)
        occ[:, 30] = True                                  # wall at x = 1.5 .. 1.55
        g = Grid(occ=occ, origin=(0.0, 0.0), res=0.05)
        angles = np.array([0.0, math.radians(20), math.pi / 2])
        r = raycast(g, (0.5, 1.0, 0.0), angles, 8.0)
        self.assertAlmostEqual(float(r[0]), 1.0, delta=0.015)
        self.assertAlmostEqual(float(r[1]), 1.0 / math.cos(math.radians(20)), delta=0.02)
        self.assertAlmostEqual(forward_min(r, angles), float(r[0]))
        self.assertTrue(math.isinf(forward_min(r[2:], angles[2:])))


# --------------------------------------------------------------------------------------------
# closed loop: NavigatorCore + kinematic AMR + grid-raycast lidar + drifting odometry
# --------------------------------------------------------------------------------------------
class KinematicWorld:
    def __init__(self, layout, grid, seed=0):
        self.layout, self.grid = layout, grid
        self.rng = np.random.default_rng(seed)
        self.truth = tuple(layout["amr"]["start"])
        self.odom = self.truth
        self.amin, self.ainc, self.n = lidar_angles(layout)
        self.angles = beam_angles(self.amin, self.ainc, self.n)
        self.t, self.v, self.w, self.step = 0.0, 0.0, 0.0, 0
        self.extra = None  # optional extra occupancy (unknown obstacle)

    def messages(self):
        msgs = [("sim/clock", {"t": self.t, "rtf": 0.0}),
                ("amr/odom", {"t": self.t, "x": self.odom[0], "y": self.odom[1], "yaw": self.odom[2],
                              "v": self.v, "w": self.w})]
        if self.step % 5 == 0:
            grid = self.grid if self.extra is None else Grid(self.grid.occ | self.extra, self.grid.origin, self.grid.res)
            r = raycast(grid, self.truth, self.angles, 8.0) + self.rng.normal(0, 0.01, self.n)
            msgs.append(("amr/scan", {"t": self.t, "angle_min": self.amin, "angle_inc": self.ainc,
                                      "range_max": 8.0, "ranges": r.tolist()}))
        return msgs

    def advance(self, dt=0.02):
        self.truth = simulate_unicycle(self.truth, self.v, self.w, dt)
        self.odom = noisy_odom_step(self.odom, self.v, self.w, dt, self.rng)
        self.t = round(self.t + dt, 6)
        self.step += 1


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class NavigatorCoreTest(unittest.TestCase):
    def setUp(self):
        from cosim.navigator import NavigatorCore
        self.layout = load_layout()
        self.core = NavigatorCore(self.layout, seed=1)
        self.world = KinematicWorld(self.layout, self.core.map.grid, seed=2)

    def run_until(self, goal_msg=None, t_max=90.0, stop_on=("arrived", "failed")):
        out_log = []
        first = True
        while self.world.t < t_max:
            msgs = self.world.messages()
            if first and goal_msg:
                msgs.append(("amr/goal", dict(goal_msg, t=self.world.t)))
                first = False
            out = self.core.process(msgs)
            out_log.extend(out)
            for topic, p in out:
                if topic == "amr/cmd_vel":
                    self.world.v, self.world.w = p["v"], p["w"]
                if topic == "amr/status" and goal_msg and p["goal_id"] == goal_msg["goal_id"] \
                        and p["state"] in stop_on:
                    return p, out_log
            self.world.advance()
        return None, out_log

    def test_mission_home_to_load(self):
        st = self.layout["stations"]["LOAD"]
        goal = {"goal_id": 7, "station": "LOAD", "x": st[0], "y": st[1], "yaw": st[2]}
        status, log = self.run_until(goal)
        self.assertIsNotNone(status, "no terminal status")
        self.assertEqual(status["state"], "arrived", status)
        tx, ty, tyaw = self.world.truth
        self.assertLess(math.hypot(tx - st[0], ty - st[1]), 0.05)
        self.assertLess(abs(wrap(tyaw - st[2])), math.radians(3))
        topics = [tp for tp, _ in log]
        self.assertIn("amr/path", topics)
        states = [p["state"] for tp, p in log if tp == "amr/status" and p["goal_id"] == 7]
        self.assertEqual(states[0], "navigating")
        cmds = np.array([[p["v"], p["w"]] for tp, p in log if tp == "amr/cmd_vel"])
        self.assertLessEqual(float(np.abs(cmds[:, 0]).max()), self.layout["amr"]["max_v"] + 1e-9)
        self.assertLessEqual(float(np.abs(cmds[:, 1]).max()), self.layout["amr"]["max_w"] + 1e-9)
        poses = [p for tp, p in log if tp == "amr/pose"]
        rate = len(poses) / self.world.t
        self.assertTrue(9.0 <= rate <= 11.0, rate)
        self.assertTrue({"t", "x", "y", "yaw", "std_xy", "std_yaw"} <= set(poses[-1]))

    def test_emergency_stop_and_unreachable_goal(self):
        # an unknown box 0.22 m in front of the lidar: no forward command while it is in the
        # forward +-30 deg sector closer than 0.25 m
        g = self.core.map.grid
        extra = np.zeros(g.occ.shape, dtype=bool)
        x0, y0, _ = self.layout["amr"]["start"]
        ix, iy = g.world_to_cell(np.array([x0 - 0.3, x0 + 0.3]), np.array([y0 + 0.22, y0 + 0.4]))
        extra[iy[0]:iy[1] + 1, ix[0]:ix[1] + 1] = True
        self.world.extra = extra
        st = self.layout["stations"]["DROP"]
        goal = {"goal_id": 3, "station": "DROP", "x": st[0], "y": st[1], "yaw": st[2]}
        self.core.process(self.world.messages() + [("amr/goal", dict(goal, t=0.0))])
        checked = 0
        while self.world.t < 3.0:
            out = self.core.process(self.world.messages())
            close = forward_min(*self.core.scan, math.radians(30)) < 0.25
            for topic, p in out:
                if topic == "amr/cmd_vel":
                    self.world.v, self.world.w = p["v"], p["w"]
                    if close:
                        checked += 1
                        self.assertLessEqual(p["v"], 0.0)
            self.world.advance()
        self.assertGreater(checked, 5)

        # a goal inside a rack is reported as failed
        self.world.extra = None
        status, _ = self.run_until({"goal_id": 4, "station": "X", "x": 5.0, "y": 6.0, "yaw": 0.0}, t_max=4.0)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["reason"], "no_path")


@unittest.skipUnless(HAVE_NUMPY and HAVE_BUS, "needs numpy, zmq, msgpack")
class CliTest(unittest.TestCase):
    def test_cli_parses_and_module_imports(self):
        import cosim.navigator as nav
        self.assertTrue(callable(nav.main))
        self.assertNotIn("amr/truth", nav.SUBSCRIBE)


if __name__ == "__main__":
    unittest.main()
