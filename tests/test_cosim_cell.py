"""Tests for the cell: teach points (IK), trajectories, mission state machine, dashboard."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import mujoco  # noqa: F401
    import numpy as np
    from cosim.protocol import load_layout
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False

try:
    import PIL  # noqa: F401
    HAVE_PIL = True
except ImportError:  # pragma: no cover
    HAVE_PIL = False


@unittest.skipUnless(HAVE, "mujoco/numpy not installed")
class TeachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from cosim.tools import teach
        cls.teach = teach
        cls.layout = load_layout()
        cls.kin = teach.ArmKinematics(cls.layout)
        cls.result = teach.compute_teach_points(cls.layout)

    def test_points_reach_targets_within_1cm(self) -> None:
        for name in ("above_load", "place_load"):
            target = np.array(self.result["targets"][name])
            q = np.array(self.result["points"][name])
            pos, rot = self.kin.fk_world(q)
            self.assertLess(np.linalg.norm(pos - target), 0.01, name)
            self.assertLess(self.kin.verify(q, target)["tilt"], 0.05, name)  # tool points down
            self.assertTrue(np.all(q >= self.kin.low) and np.all(q <= self.kin.high), name)

    def test_targets_are_above_deck_centre(self) -> None:
        deck = self.teach.deck_centre(self.layout)
        self.assertAlmostEqual(deck[2], self.layout["amr"]["deck_height"])
        t = self.result["targets"]
        self.assertTrue(np.allclose(t["place_load"], deck + [0, 0, 0.05]))
        self.assertTrue(np.allclose(t["above_load"][:2], deck[:2]))
        self.assertGreater(t["above_load"][2] - deck[2], 0.10)
        self.assertLessEqual(t["above_load"][2] - deck[2], 0.25 + 1e-9)

    def test_mount_transform_roundtrip(self) -> None:
        p = np.array([2.3, 3.9, 0.5])
        self.assertTrue(np.allclose(self.kin.arm_to_world(self.kin.world_to_arm(p)), p))
        # yaw 0 of the arm points along world -y (mount_yaw = -pi/2)
        self.assertTrue(np.allclose(self.kin.arm_to_world([1, 0, 0]) - self.kin.base, [0, -1, 0], atol=1e-9))

    def test_layout_teach_points_present_and_match(self) -> None:
        stored = self.layout["arm"]["teach_points"]
        for name in ("home", "above_load", "place_load"):
            self.assertEqual(len(stored[name]), 6)
        self.assertTrue(np.allclose(stored["above_load"], self.result["points"]["above_load"], atol=1e-5))
        self.assertTrue(np.allclose(stored["place_load"], self.result["points"]["place_load"], atol=1e-5))

    def test_write_layout_replaces_only_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            shutil.copy(ROOT / "cosim" / "layout.json", path)
            before = json.loads(path.read_text())
            self.teach.write_layout(path, self.result["points"])
            after = json.loads(path.read_text())
            before["arm"].pop("teach_points")
            after["arm"].pop("teach_points")
            self.assertEqual(before, after)


@unittest.skipUnless(HAVE, "mujoco/numpy not installed")
class TrajectoryTests(unittest.TestCase):
    def test_speed_limit_and_waypoints(self) -> None:
        from cosim.cell import Trapezoid
        q0 = np.array([0.0, 0.0, 0.6, 0.0, 1.2, 0.0])
        q1 = np.array([0.0, 1.0, 0.5, 0.2, 1.6, -0.3])
        traj = Trapezoid(q0, q1)
        self.assertTrue(np.allclose(traj.at(0.0), q0))
        self.assertTrue(np.allclose(traj.at(traj.duration), q1))
        self.assertTrue(np.allclose(traj.at(traj.duration + 5), q1))
        dt = 1.0 / 30.0
        ts = np.arange(0.0, traj.duration + dt, dt)
        qs = np.array([traj.at(t) for t in ts])
        speed = np.max(np.abs(np.diff(qs, axis=0)), axis=1) / dt
        self.assertLessEqual(speed.max(), 1.0)
        self.assertGreater(speed.max(), 0.5)
        self.assertTrue(np.all(np.diff(np.array([traj.progress(t) for t in ts])) >= -1e-12))  # monotonic

    def test_short_move_is_triangular_and_zero_move(self) -> None:
        from cosim.cell import Trapezoid
        short = Trapezoid(np.zeros(6), np.array([0.05, 0, 0, 0, 0, 0]))
        self.assertLess(short.vpeak, 0.8)
        self.assertAlmostEqual(short.at(short.duration)[0], 0.05)
        zero = Trapezoid(np.zeros(6), np.zeros(6))
        self.assertEqual(zero.duration, 0.0)


@unittest.skipUnless(HAVE, "mujoco/numpy not installed")
class MissionTests(unittest.TestCase):
    def setUp(self) -> None:
        from cosim.cell import Mission
        self.layout = load_layout()
        self.mission_cls = Mission

    def run_mission(self, cycles=1, fail_leg=None, never_arrive_leg=None, leg_timeout=120.0):
        """Fake bus: the 'navigator' acks each goal and reports arrived 5 s later."""
        m = self.mission_cls(self.layout, cycles, leg_timeout=leg_timeout)
        out_all, pending, leg = [], [], 0
        t = 0.0
        goals = []
        while t < 2000.0 and not m.done:
            inbox = [msg for at, msg in pending if at <= t]
            pending = [(at, msg) for at, msg in pending if at > t]
            out = m.update(t, inbox)
            for topic, payload in out:
                out_all.append((t, topic, payload))
                if topic == "amr/goal":
                    new_goal = not goals or goals[-1]["goal_id"] != payload["goal_id"]
                    leg += new_goal
                    goals.append(payload)
                    if leg == never_arrive_leg:
                        continue
                    state = "failed" if leg == fail_leg else "arrived"
                    gid = payload["goal_id"]
                    pending.append((t + 0.2, ("amr/status", {"t": t, "goal_id": gid, "state": "navigating", "reason": ""})))
                    pending.append((t + 5.0, ("amr/status", {"t": t, "goal_id": gid, "state": state, "reason": "blocked" if state == "failed" else ""})))
            t += 1.0 / 30.0
        return m, out_all, goals

    def test_full_cycle_order(self) -> None:
        m, out, goals = self.run_mission(1)
        self.assertTrue(m.done and not m.failed)
        self.assertEqual([g["station"] for g in goals], ["LOAD", "DROP", "HOME"])
        self.assertEqual([g["goal_id"] for g in goals], [1, 2, 3])  # acked in time: no re-sends
        order = [(topic, payload.get("station") or payload.get("state")) for _, topic, payload in out
                 if topic in ("amr/goal", "cell/event")]
        self.assertEqual(order, [("amr/goal", "LOAD"), ("cell/event", "LOAD"), ("amr/goal", "DROP"),
                                 ("cell/event", "DROP"), ("amr/goal", "HOME")])
        events = [p["event"] for _, topic, p in out if topic == "cell/event"]
        self.assertEqual(events, ["load", "unload"])
        self.assertEqual(len(m.cycle_times), 1)
        self.assertEqual(m.status_payload(0.0)["state"], "done")

    def test_arm_stream_rate_limits_and_waypoints(self) -> None:
        m, out, _ = self.run_mission(1)
        cmds = [(t, p["q"]) for t, topic, p in out if topic == "arm/joint_cmd"]
        times = np.array([t for t, _ in cmds])
        self.assertAlmostEqual(float(np.median(np.diff(times))), 1 / 30, places=3)
        qs = np.array([q for _, q in cmds])
        speed = np.max(np.abs(np.diff(qs, axis=0)), axis=1) / np.diff(times)
        self.assertLessEqual(speed.max(), 1.0 + 1e-6)
        tp = self.layout["arm"]["teach_points"]
        for name in ("above_load", "place_load"):
            self.assertLess(np.min(np.max(np.abs(qs - np.array(tp[name])), axis=1)), 1e-6, name)
        self.assertTrue(np.allclose(qs[0], tp["home"]))
        self.assertTrue(np.allclose(qs[-1], tp["home"]))
        # the load event is sent while the tool is at place_load
        t_load = next(t for t, topic, p in out if topic == "cell/event" and p["event"] == "load")
        q_at = qs[np.argmin(np.abs(times - t_load))]
        self.assertLess(np.max(np.abs(q_at - np.array(tp["place_load"]))), 1e-6)

    def test_multiple_cycles(self) -> None:
        m, out, goals = self.run_mission(2)
        self.assertTrue(m.done and not m.failed)
        self.assertEqual([g["station"] for g in goals], ["LOAD", "DROP", "HOME"] * 2)
        self.assertEqual(len(m.cycle_times), 2)

    def test_failed_status_aborts(self) -> None:
        m, out, goals = self.run_mission(1, fail_leg=2)
        self.assertTrue(m.failed and m.done)
        self.assertEqual({g["goal_id"] for g in goals}, {1, 2})
        self.assertIn("blocked", m.message)
        self.assertFalse(any(p.get("event") == "unload" for _, topic, p in out if topic == "cell/event"))

    def test_leg_timeout_aborts_and_goal_is_resent(self) -> None:
        m, out, goals = self.run_mission(1, never_arrive_leg=1, leg_timeout=20.0)
        self.assertTrue(m.failed)
        self.assertIn("timeout", m.message)
        self.assertGreater(len(goals), 3)  # re-sent until acknowledged
        self.assertEqual({g["goal_id"] for g in goals}, {1})

    def test_ignores_stale_goal_status(self) -> None:
        m = self.mission_cls(self.layout, 1)
        m.update(0.0, [])
        m.update(0.1, [("amr/status", {"t": 0.1, "goal_id": 99, "state": "arrived", "reason": ""})])
        self.assertEqual(m.state, "to_load")


@unittest.skipUnless(HAVE and HAVE_PIL, "numpy/Pillow not installed")
class DashboardTests(unittest.TestCase):
    def test_renders_frame_from_synthetic_messages(self) -> None:
        import math
        from cosim.dashboard import H, W, Dashboard
        layout = load_layout()
        d = Dashboard(layout)
        empty = d.render()
        self.assertEqual(empty.size, (W, H))
        msgs = [
            ("sim/clock", {"t": 12.5, "rtf": 1.0}),
            ("amr/truth", {"t": 12.5, "x": 6.0, "y": 3.0, "yaw": 1.0}),
            ("amr/pose", {"t": 12.5, "x": 6.05, "y": 3.02, "yaw": 0.98, "std_xy": 0.1, "std_yaw": 0.05}),
            ("amr/path", {"t": 12.5, "goal_id": 1, "points": [[6.0, 3.0], [4.0, 3.5], [2.0, 3.45]]}),
            ("amr/scan", {"t": 12.5, "angle_min": -2.356, "angle_inc": 4.712 / 270, "range_max": 8.0,
                          "ranges": [2.0 + 0.5 * math.sin(i / 10) for i in range(271)]}),
            ("arm/state", {"t": 12.5, "q": [0.0, 0.5, 0.6, 0.0, 1.2, 0.0]}),
            ("cell/crate", {"t": 12.5, "state": "on_amr", "x": 6.0, "y": 3.0}),
            ("amr/status", {"t": 12.5, "goal_id": 1, "state": "navigating", "reason": ""}),
            ("cell/status", {"t": 12.5, "state": "to_load", "cycle": 1, "cycles": 1, "message": "AMR -> LOAD"}),
        ]
        for topic, payload in msgs:
            d.ingest(topic, payload)
        frame = d.render()
        self.assertEqual(frame.size, (960, 540))
        self.assertGreater(np.abs(np.asarray(frame, dtype=int) - np.asarray(empty, dtype=int)).sum(), 10000)
        # true pose colour (green) appears at the true pose position
        px, py = d.px(6.0, 3.0)
        r, g, b = frame.getpixel((int(px / 2), int(py / 2)))
        self.assertGreater(g, r)


if __name__ == "__main__":
    unittest.main()
