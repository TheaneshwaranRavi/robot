"""AMR navigator: particle-filter localisation + A* planning + pure-pursuit control.

    python -m cosim.navigator [--layout cosim/layout.json] [--host 127.0.0.1]

Subscribes sim/clock, amr/odom, amr/scan, amr/goal, sys/shutdown (never amr/truth).
Publishes amr/pose (10 Hz sim time), amr/path, amr/cmd_vel (20 Hz while navigating) and
amr/status (on every transition, repeated at 1 Hz so late subscribers catch up).

All algorithms live in cosim/nav_*.py (numpy only). `NavigatorCore` is the bus-free state
machine: feed it (topic, payload) messages, it returns the messages to publish, so it can be
tested without ZeroMQ or other processes.
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .nav_astar import plan_to_pose
from .nav_grid import add_obstacle_points, build_navmap
from .nav_pf import ParticleFilter, PFConfig, compose, relative_motion
from .nav_pursuit import FollowerConfig, PathFollower, wrap
from .nav_scan import beam_angles, forward_min, scan_points

SUBSCRIBE = ("sim/clock", "amr/odom", "amr/scan", "amr/goal", "sys/shutdown")

CONTROL_PERIOD = 0.05     # s sim time (20 Hz)
POSE_PERIOD = 0.1         # s sim time (contract: 10 Hz)
STATUS_PERIOD = 1.0       # s sim time, status heartbeat
REPLAN_PERIOD = 2.0       # s sim time (contract)
ESTOP_RANGE = 0.25        # m, forward ±30° sector (contract)
ESTOP_SECTOR = math.radians(30.0)
ESTOP_REPLAN_AFTER = 2.0  # s stopped by the e-stop before replanning around unknown obstacles
BLOCKED_FAIL_AFTER = 20.0 # s blocked (e-stop or no dynamic path) before reporting failure
UNKNOWN_MIN_DIST = 0.25   # m from the static map for a scan hit to count as an unknown obstacle
GOAL_IGNORE_RADIUS = 0.5  # m: unknown hits this close to the goal don't block planning (crate/fixtures)
LOST_STD = 0.5            # m, localisation considered lost above this std_xy
ACC_V = 0.8               # m/s^2 command slew limit
ACC_W = 3.0               # rad/s^2


class NavigatorCore:
    def __init__(self, layout: dict[str, Any], seed: int | None = None, n_particles: int = 500) -> None:
        self.layout = layout
        amr = layout["amr"]
        self.max_v, self.max_w = float(amr["max_v"]), float(amr["max_w"])
        self.stations = {k: v for k, v in layout["stations"].items() if not k.startswith("_")}
        self.map = build_navmap(layout)
        self.range_max = float(layout["lidar"]["range_max"])
        self.pf = ParticleFilter(self.map.grid, self.map.dist, self.range_max,
                                 PFConfig(n_particles=n_particles), seed=seed)
        self.start = tuple(float(v) for v in amr["start"])
        self.pf.init(self.start)
        self.follower = PathFollower(FollowerConfig(max_v=self.max_v, max_w=self.max_w))

        self.t = 0.0
        self.odom_buf: deque = deque(maxlen=400)       # (t, x, y, yaw) in the odom frame
        self.pf_odom_ref = None                        # odom pose at the last PF prediction
        self.est = (self.start, 0.0, 0.0)              # (pose, std_xy, std_yaw) at pf_odom_ref
        self.scan = None                               # (ranges, angles) latest scan, robot frame
        self.scan_t = -1.0

        self.goal: dict[str, Any] | None = None
        self.state = "idle"
        self.reason = ""
        self.path: np.ndarray | None = None
        self.last_plan_t = -1e9
        self.blocked_since: float | None = None
        self.estop = False
        self.estop_since: float | None = None
        self.cmd = (0.0, 0.0)
        self.last_control_t = -1e9
        self.last_pose_t = -1e9
        self.last_status_t = -1e9
        self.shutdown = False
        self.stats = {"pf_updates": 0, "plans": 0}

    # ---------------------------------------------------------------- pose / localisation
    def odom_pose(self):
        if not self.odom_buf:
            return None
        _, x, y, yaw = self.odom_buf[-1]
        return (x, y, yaw)

    def _odom_at(self, t: float):
        buf = self.odom_buf
        if not buf:
            return None
        if t >= buf[-1][0]:
            return buf[-1][1:]
        if t <= buf[0][0]:
            return buf[0][1:]
        for i in range(len(buf) - 1, 0, -1):
            t0, t1 = buf[i - 1][0], buf[i][0]
            if t0 <= t <= t1:
                a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                p0, p1 = buf[i - 1], buf[i]
                return (p0[1] + a * (p1[1] - p0[1]), p0[2] + a * (p1[2] - p0[2]),
                        float(wrap(p0[3] + a * wrap(p1[3] - p0[3]))))
        return buf[-1][1:]

    def pose(self):
        """Current map-frame pose: last PF estimate composed with the odometry since then."""
        est_pose = self.est[0]
        cur = self.odom_pose()
        if cur is None or self.pf_odom_ref is None:
            return est_pose
        return compose(est_pose, relative_motion(self.pf_odom_ref, cur))

    def _on_scan(self, msg: dict[str, Any]) -> None:
        ranges = np.asarray([self.range_max if r is None else r for r in msg["ranges"]], dtype=float)
        ranges = np.where(np.isfinite(ranges), ranges, self.range_max)
        angles = beam_angles(float(msg["angle_min"]), float(msg["angle_inc"]), len(ranges))
        self.scan = (ranges, angles)
        self.scan_t = float(msg["t"])
        odom = self._odom_at(self.scan_t)
        if odom is None:
            return
        if self.pf_odom_ref is None:
            # odom frame == map frame at the start pose, so the first odom reading is a pose in map
            self.pf.init(odom)
            self.pf_odom_ref = odom
        else:
            self.pf.predict(relative_motion(self.pf_odom_ref, odom))
            self.pf_odom_ref = odom
        if self.pf.correct(ranges, angles):
            self.stats["pf_updates"] += 1
        self.est = self.pf.estimate()

    # ---------------------------------------------------------------- planning
    def _unknown_points(self, pose) -> np.ndarray:
        if self.scan is None:
            return np.zeros((0, 2))
        pts = scan_points(pose, self.scan[0], self.scan[1], self.range_max)
        if not len(pts):
            return pts
        g = self.map.grid
        ix, iy = g.world_to_cell(pts[:, 0], pts[:, 1])
        ok = g.in_bounds(ix, iy)
        d = np.zeros(len(pts))
        d[ok] = self.map.dist[iy[ok], ix[ok]]
        keep = ok & (d > UNKNOWN_MIN_DIST)
        if self.goal is not None:
            keep &= np.hypot(pts[:, 0] - self.goal["x"], pts[:, 1] - self.goal["y"]) > GOAL_IGNORE_RADIUS
        return pts[keep]

    def _plan(self, use_unknown: bool) -> bool:
        pose = self.pose()
        cost = self.map.cost
        if use_unknown:
            cost = add_obstacle_points(self.map, self._unknown_points(pose))
        goal = (self.goal["x"], self.goal["y"], self.goal["yaw"])
        res = plan_to_pose(self.map.grid, cost, pose[:2], goal)
        self.last_plan_t = self.t
        self.stats["plans"] += 1
        if res is None:
            return False
        pts, dock = res
        self.path = pts
        self.follower.set_path(pts, goal, dock)
        return True

    def _path_blocked(self, pose) -> bool:
        if self.path is None:
            return False
        unk = self._unknown_points(pose)
        if not len(unk):
            return False
        prog = self.follower.s.progress
        ahead = self.follower.s.legs[0][prog:prog + 40] if self.follower.s.legs else self.path[:40]
        if not len(ahead):
            return False
        d = np.hypot(ahead[:, None, 0] - unk[None, :, 0], ahead[:, None, 1] - unk[None, :, 1])
        return bool(d.min() < self.map.lethal_radius + 0.05)

    # ---------------------------------------------------------------- messages out
    def _status_msg(self) -> tuple[str, dict]:
        gid = self.goal["goal_id"] if self.goal else -1
        return "amr/status", {"t": self.t, "goal_id": gid, "state": self.state, "reason": self.reason}

    def _path_msg(self) -> tuple[str, dict]:
        pts = self.path[::2] if self.path is not None else np.zeros((0, 2))
        if self.path is not None and len(self.path) and (len(pts) == 0 or not np.allclose(pts[-1], self.path[-1])):
            pts = np.vstack([pts, self.path[-1:]])
        return "amr/path", {"t": self.t, "goal_id": self.goal["goal_id"] if self.goal else -1,
                            "points": [[round(float(x), 3), round(float(y), 3)] for x, y in pts]}

    def _set_state(self, state: str, reason: str, out: list) -> None:
        self.state, self.reason = state, reason
        out.append(self._status_msg())
        self.last_status_t = self.t
        print(f"[navigator] t={self.t:7.2f} goal={self.goal and self.goal['goal_id']} -> {state} {reason}",
              flush=True)

    def _stop(self, out: list) -> None:
        self.follower.stop()
        self.cmd = (0.0, 0.0)
        out.append(("amr/cmd_vel", {"t": self.t, "v": 0.0, "w": 0.0}))

    # ---------------------------------------------------------------- goal handling
    def _on_goal(self, msg: dict[str, Any], out: list) -> None:
        gid = msg.get("goal_id")
        if self.goal is not None and gid == self.goal["goal_id"]:
            out.append(self._status_msg())  # duplicate dispatch: just re-announce
            return
        if self.goal is not None and self.state == "navigating":
            self._set_state("failed", "preempted", out)
        station = msg.get("station")
        ref = self.stations.get(station, [None, None, None])
        x = msg.get("x", ref[0])
        y = msg.get("y", ref[1])
        yaw = msg.get("yaw", ref[2])
        self.goal = {"goal_id": gid, "station": station, "x": x, "y": y, "yaw": yaw}
        self.blocked_since = None
        self.estop_since = None
        if x is None or y is None or yaw is None:
            self._set_state("failed", f"unknown station {station!r}", out)
            return
        if not self._plan(use_unknown=True) and not self._plan(use_unknown=False):
            self.path = None
            self._set_state("failed", "no_path", out)
            self._stop(out)
            return
        out.append(self._path_msg())
        self._set_state("navigating", "", out)

    # ---------------------------------------------------------------- control
    def _control(self, dt: float, out: list) -> None:
        if self.state != "navigating":
            return
        pose = self.pose()
        if self.est[1] > LOST_STD:
            self._set_state("failed", f"localization_lost (std_xy={self.est[1]:.2f} m)", out)
            self._stop(out)
            return

        # replanning: periodic, when the path ahead is blocked, or after a sustained e-stop
        if self.follower.replannable():
            blocked = self._path_blocked(pose) and self.t - self.last_plan_t >= 0.5
            estop_long = self.estop_since is not None and self.t - self.estop_since >= ESTOP_REPLAN_AFTER \
                and self.t - self.last_plan_t >= 1.0
            if blocked or estop_long or self.t - self.last_plan_t >= REPLAN_PERIOD:
                if self._plan(use_unknown=True):
                    self.blocked_since = None
                    out.append(self._path_msg())
                elif self._plan(use_unknown=False) is False:
                    self._set_state("failed", "no_path", out)
                    self._stop(out)
                    return
                else:
                    # statically reachable but blocked by unknown obstacles right now: wait
                    self.follower.stop()
                    if self.blocked_since is None:
                        self.blocked_since = self.t
                    out.append(self._path_msg())
            if self.follower.phase == "idle":  # waiting for an unknown obstacle to clear
                if self.blocked_since is not None and self.t - self.blocked_since > BLOCKED_FAIL_AFTER:
                    self._set_state("failed", "path_blocked", out)
                    self._stop(out)
                    return
                if self.t - self.last_plan_t >= 0.5 and self._plan(use_unknown=True):
                    self.blocked_since = None
                    out.append(self._path_msg())
                else:
                    self._send_cmd(0.0, 0.0, dt, out)
                    return

        v, w = self.follower.step(pose)
        if self.follower.phase == "done":
            self._stop(out)
            self._set_state("arrived", "", out)
            return

        # emergency stop: obstacle in the forward ±30° sector closer than 0.25 m
        self.estop = False
        if v > 0.0 and self.scan is not None and \
                forward_min(self.scan[0], self.scan[1], ESTOP_SECTOR) < ESTOP_RANGE:
            v, self.estop = 0.0, True
            if self.estop_since is None:
                self.estop_since = self.t
                print(f"[navigator] t={self.t:7.2f} emergency stop", flush=True)
            if self.t - self.estop_since > BLOCKED_FAIL_AFTER:
                self._set_state("failed", "emergency_stop_timeout", out)
                self._stop(out)
                return
            self._send_cmd(0.0, w, dt, out, slew=False)
            return
        self.estop_since = None
        self._send_cmd(v, w, dt, out)

    def _send_cmd(self, v: float, w: float, dt: float, out: list, slew: bool = True) -> None:
        v = float(np.clip(v, -self.max_v, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))
        if slew and dt > 0:
            pv, pw = self.cmd
            v = float(np.clip(v, pv - ACC_V * dt, pv + ACC_V * dt))
            w = float(np.clip(w, pw - ACC_W * dt, pw + ACC_W * dt))
        self.cmd = (v, w)
        out.append(("amr/cmd_vel", {"t": self.t, "v": v, "w": w}))

    # ---------------------------------------------------------------- entry points
    def process(self, messages: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        """Handle a batch of messages. Only the newest scan of the batch is used (if the
        navigator falls behind at --rtf 0 it skips stale scans instead of lagging further)."""
        out: list = []
        latest_scan = None
        for topic, msg in messages:
            t = msg.get("t")
            if isinstance(t, (int, float)):
                self.t = max(self.t, float(t))
            if topic == "sys/shutdown":
                self.shutdown = True
            elif topic == "amr/odom":
                self.odom_buf.append((float(msg["t"]), float(msg["x"]), float(msg["y"]), float(msg["yaw"])))
            elif topic == "amr/scan":
                latest_scan = msg
            elif topic == "amr/goal":
                if latest_scan is not None:
                    self._on_scan(latest_scan)
                    latest_scan = None
                self._on_goal(msg, out)
        if latest_scan is not None:
            self._on_scan(latest_scan)
        if self.shutdown:
            self._stop(out)
            return out
        if self.t - self.last_control_t >= CONTROL_PERIOD - 1e-9:
            dt = min(0.2, self.t - self.last_control_t) if self.last_control_t > -1e8 else CONTROL_PERIOD
            self.last_control_t = self.t
            self._control(dt, out)
        if self.t - self.last_pose_t >= POSE_PERIOD - 1e-9:
            self.last_pose_t = self.t
            (x, y, yaw), sxy, syaw = self.pose(), self.est[1], self.est[2]
            out.append(("amr/pose", {"t": self.t, "x": x, "y": y, "yaw": yaw, "std_xy": sxy, "std_yaw": syaw}))
        if self.t - self.last_status_t >= STATUS_PERIOD:
            out.append(self._status_msg())
            self.last_status_t = self.t
        return out


def main(argv: list[str] | None = None) -> int:
    from .protocol import LAYOUT_PATH, Bus, load_layout

    parser = argparse.ArgumentParser(description="AMR navigator (particle filter + A* + pure pursuit)")
    parser.add_argument("--layout", type=Path, default=LAYOUT_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    core = NavigatorCore(load_layout(args.layout), seed=args.seed, n_particles=args.particles)
    bus = Bus("navigator", SUBSCRIBE, host=args.host)
    stop = {"flag": False}

    def on_signal(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print(f"[navigator] up: grid {core.map.grid.occ.shape[1]}x{core.map.grid.occ.shape[0]} @ "
          f"{core.map.grid.res} m, inflation {core.map.inflation_radius:.3f} m", flush=True)
    try:
        while not stop["flag"] and not core.shutdown:
            msgs = bus.poll(50)
            if not msgs:
                continue
            for topic, payload in core.process(msgs):
                bus.publish(topic, payload)
    finally:
        try:
            bus.publish("amr/cmd_vel", {"t": core.t, "v": 0.0, "w": 0.0})
        finally:
            bus.close()
    print(f"[navigator] exit: {core.stats}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
