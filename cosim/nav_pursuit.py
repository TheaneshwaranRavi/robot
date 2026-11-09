"""Pure-pursuit path follower with a docking approach and final in-place rotation (numpy only).

A path is one or two legs. A docking path (see nav_astar.plan_to_pose) is
  leg 0: A* path to a pre-goal point behind the goal (passed through at moderate speed),
  then an in-place turn to face along the goal yaw,
  leg 1: the straight final approach to the goal.
A direct path is a single final leg. The final leg slows down near the goal; when the goal is
within the position tolerance the AMR turns in place to the goal yaw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class FollowerConfig:
    max_v: float = 0.6
    max_w: float = 1.2
    lookahead: float = 0.45        # m, follow leg
    lookahead_final: float = 0.25  # m, final approach
    pos_tol: float = 0.03          # m (contract tolerance is 5 cm; margin for estimate error)
    yaw_tol: float = math.radians(1.5)  # contract: 3 deg
    leg_tol: float = 0.04          # m, switch from leg 0 to the turn at the pre-goal
    rotate_in_place: float = math.radians(60.0)  # heading error above which we turn first
    slow_gain: float = 0.8         # v <= slow_gain * distance-to-goal (+ v_min) on the final leg
    v_min: float = 0.04
    w_gain: float = 2.5
    w_min: float = 0.12            # minimum turn rate while correcting heading
    reverse_max: float = 0.08      # m/s, small reverse allowed only to fix a tiny overshoot


@dataclass
class FollowerState:
    phase: str = "idle"            # idle | follow | align | approach | rotate | done
    progress: int = 0
    legs: list = field(default_factory=list)
    leg: int = 0
    goal: tuple = (0.0, 0.0, 0.0)


class PathFollower:
    def __init__(self, cfg: FollowerConfig | None = None) -> None:
        self.cfg = cfg or FollowerConfig()
        self.s = FollowerState()

    # -- path management -------------------------------------------------------------------
    def set_path(self, points: np.ndarray, goal_pose, dock_index: int | None = None) -> None:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if dock_index is not None and 0 < dock_index < len(pts) - 1:
            legs = [pts[:dock_index + 1], pts[dock_index:]]
        else:
            legs = [pts]
        self.s = FollowerState(phase="follow" if len(legs) == 2 else "approach",
                               progress=0, legs=legs, leg=0, goal=tuple(goal_pose))

    @property
    def phase(self) -> str:
        return self.s.phase

    def replannable(self) -> bool:
        """Replanning only makes sense while following the A* leg (the final leg is a short
        straight segment that was checked collision-free when it was planned)."""
        return self.s.phase == "follow"

    def stop(self) -> None:
        self.s = FollowerState()

    # -- control ----------------------------------------------------------------------------
    def step(self, pose) -> tuple[float, float]:
        """One control step. pose = (x, y, yaw) in the map frame. Returns (v, w)."""
        c, s = self.cfg, self.s
        x, y, yaw = pose
        gx, gy, gyaw = s.goal
        if s.phase in ("idle", "done"):
            return 0.0, 0.0
        if s.phase == "follow":
            pts = s.legs[0]
            end = pts[-1]
            d_end = math.hypot(end[0] - x, end[1] - y)
            behind = math.cos(yaw) * (end[0] - x) + math.sin(yaw) * (end[1] - y) < 0.0
            if d_end < c.leg_tol or (behind and d_end < 0.15 and s.progress >= len(pts) - 2):
                s.phase, s.leg, s.progress = "align", 1, 0
            else:
                v, w = self._pursue(pose, pts, c.lookahead, final=False)
                return v, w
        if s.phase == "align":
            leg = s.legs[1]
            heading = math.atan2(leg[-1][1] - leg[0][1], leg[-1][0] - leg[0][0])
            err = wrap(heading - yaw)
            if abs(err) < math.radians(4.0):
                s.phase = "approach"
            else:
                return 0.0, self._turn(err)
        if s.phase == "approach":
            pts = s.legs[s.leg]
            if math.hypot(gx - x, gy - y) < c.pos_tol:
                s.phase = "rotate"
            else:
                return self._pursue(pose, pts, c.lookahead_final, final=True)
        if s.phase == "rotate":
            dist = math.hypot(gx - x, gy - y)
            if dist > 3.0 * c.pos_tol:          # drifted off while turning: re-approach directly
                s.legs, s.leg, s.progress = [np.array([[x, y], [gx, gy]])], 0, 0
                s.phase = "approach"
                return self._pursue(pose, s.legs[0], c.lookahead_final, final=True)
            err = wrap(gyaw - yaw)
            if abs(err) < c.yaw_tol:
                s.phase = "done"
                return 0.0, 0.0
            return 0.0, self._turn(err)
        return 0.0, 0.0

    def _turn(self, err: float) -> float:
        c = self.cfg
        w = c.w_gain * err
        w = math.copysign(max(abs(w), c.w_min), err)
        return float(np.clip(w, -c.max_w, c.max_w))

    def _pursue(self, pose, pts: np.ndarray, lookahead: float, final: bool) -> tuple[float, float]:
        c, s = self.cfg, self.s
        x, y, yaw = pose
        # advance progress to the closest point in a window ahead (never backwards)
        hi = min(len(pts), s.progress + 60)
        d2 = (pts[s.progress:hi, 0] - x) ** 2 + (pts[s.progress:hi, 1] - y) ** 2
        s.progress += int(np.argmin(d2))
        # lookahead point: first point at least `lookahead` away beyond the closest point
        rest = pts[s.progress:]
        dist = np.hypot(rest[:, 0] - x, rest[:, 1] - y)
        far = np.nonzero(dist >= lookahead)[0]
        tx, ty = rest[far[0]] if len(far) else rest[-1]
        end = pts[-1]
        d_end = math.hypot(end[0] - x, end[1] - y)
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        lx = cy_ * (tx - x) + sy_ * (ty - y)
        ly = -sy_ * (tx - x) + cy_ * (ty - y)
        ld2 = max(lx * lx + ly * ly, 1e-6)
        alpha = math.atan2(ly, lx)

        if final and d_end < 0.25 and lx < 0.0 and abs(ly) < 0.05:
            # tiny overshoot of the goal: back up slowly instead of looping around
            v = -min(c.reverse_max, c.slow_gain * d_end + c.v_min)
            k = 2.0 * ly / ld2
            return v, float(np.clip(v * k, -c.max_w, c.max_w))
        if abs(alpha) > c.rotate_in_place:
            return 0.0, self._turn(alpha)

        k = 2.0 * ly / ld2  # curvature of the arc through the lookahead point
        v = c.max_v
        if final:
            v = min(v, c.slow_gain * d_end + c.v_min)
        else:
            v = min(v, 0.6 * d_end + 0.12)  # ease into the pre-goal turn
        v *= max(0.25, math.cos(alpha))       # slow down when not pointing at the path
        if abs(k) * v > c.max_w:
            v = c.max_w / abs(k)
        return float(v), float(np.clip(v * k, -c.max_w, c.max_w))


def simulate_unicycle(pose, v: float, w: float, dt: float):
    """Exact unicycle integration over dt (used by tests and the kinematic sim)."""
    x, y, th = pose
    if abs(w) < 1e-9:
        return x + v * dt * math.cos(th), y + v * dt * math.sin(th), th
    r = v / w
    th2 = th + w * dt
    return x + r * (math.sin(th2) - math.sin(th)), y - r * (math.cos(th2) - math.cos(th)), wrap(th2)
