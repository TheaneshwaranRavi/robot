"""Cell controller ("PLC" / arm program): the mission state machine of the factory cell.

For each cycle:  AMR -> LOAD, arm pick-and-place sequence (home -> above_load -> place_load ->
[event load] -> above_load -> home), AMR -> DROP, [event unload], AMR -> HOME.

All timing uses sim time ``t`` from ``sim/clock``. ``Mission`` is a pure state machine
(``update(t, messages) -> outgoing messages``) so it can be tested with a fake bus.

Run:  python -m cosim.cell [--layout PATH] [--host 127.0.0.1] [--cycles 1] [--shutdown]
Exit code 0 when all cycles finished, 1 on failure/abort.
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from typing import Any

import numpy as np

from .protocol import LAYOUT_PATH, Bus, load_layout

ARM_RATE_HZ = 30.0
JOINT_SPEED = 0.8       # rad/s, peak per joint (limit is 1.0)
JOINT_ACCEL = 1.5       # rad/s^2
LEG_TIMEOUT = 120.0     # s sim time per AMR leg
DWELL = 0.5             # s pause around material-handling events
GOAL_RESEND = 2.0       # s sim time between goal re-sends until the navigator acknowledges

Message = tuple[str, dict[str, Any]]


class Trapezoid:
    """Joint-space move with a shared trapezoidal (or triangular) speed profile on the largest delta."""

    def __init__(self, q0, q1, vmax: float = JOINT_SPEED, amax: float = JOINT_ACCEL) -> None:
        self.q0 = np.asarray(q0, dtype=float)
        self.q1 = np.asarray(q1, dtype=float)
        self.dist = float(np.max(np.abs(self.q1 - self.q0)))
        if self.dist < 1e-9:
            self.t_acc = self.vpeak = self.duration = 0.0
            return
        if self.dist >= vmax ** 2 / amax:  # reaches cruise speed
            self.vpeak = vmax
            self.t_acc = vmax / amax
            self.duration = self.dist / vmax + vmax / amax
        else:                              # triangular
            self.vpeak = math.sqrt(self.dist * amax)
            self.t_acc = self.vpeak / amax
            self.duration = 2.0 * self.t_acc
        self.amax = amax

    def progress(self, tau: float) -> float:
        """Distance covered along the profile after tau seconds (0..dist)."""
        if self.duration <= 0.0:
            return 0.0
        tau = min(max(tau, 0.0), self.duration)
        if tau < self.t_acc:
            return 0.5 * self.amax * tau ** 2
        if tau <= self.duration - self.t_acc:
            return 0.5 * self.amax * self.t_acc ** 2 + self.vpeak * (tau - self.t_acc)
        rest = self.duration - tau
        return self.dist - 0.5 * self.amax * rest ** 2

    def at(self, tau: float) -> np.ndarray:
        if self.duration <= 0.0:
            return self.q1.copy()
        return self.q0 + (self.q1 - self.q0) * (self.progress(tau) / self.dist)


class Mission:
    """Mission state machine. Feed it sim time + received messages; it returns messages to publish.

    States: init -> to_load -> arm -> to_drop -> unload -> to_home -> (next cycle: to_load | done);
    ``failed`` on a failed AMR status or a leg timeout.
    """

    def __init__(self, layout: dict[str, Any], cycles: int = 1, leg_timeout: float = LEG_TIMEOUT) -> None:
        points = layout["arm"]["teach_points"]
        self.teach = {k: np.asarray(points[k], dtype=float) for k in ("home", "above_load", "place_load")}
        self.stations = layout["stations"]
        self.cycles = cycles
        self.leg_timeout = leg_timeout
        self.state = "init"
        self.cycle = 0
        self.message = ""
        self.done = False
        self.failed = False
        self.cycle_times: list[float] = []
        self.q_cmd = self.teach["home"].copy()
        self._goal_id = 0
        self._station = ""
        self._goal_ack = False
        self._goal_state = ""
        self._goal_reason = ""
        self._goal_sent_t = 0.0
        self._leg_start = 0.0
        self._cycle_start = 0.0
        self._last_cmd_t = -1e9
        self._last_key: tuple | None = None
        self._actions: list[tuple] = []
        self._traj: Trapezoid | None = None
        self._action_start = 0.0
        self._dwell_until: float | None = None

    # -- helpers -------------------------------------------------------------------------------
    def _set_state(self, state: str, message: str = "") -> None:
        self.state, self.message = state, message

    def _dispatch(self, t: float, station: str, state: str, out: list[Message]) -> None:
        self._goal_id += 1
        self._goal_ack = False
        self._goal_state = self._goal_reason = ""
        self._goal_sent_t = self._leg_start = t
        self._station = station
        self._set_state(state, f"AMR -> {station}")
        out.append(self._goal_msg(t))

    def _goal_msg(self, t: float) -> Message:
        x, y, yaw = self.stations[self._station]
        return "amr/goal", {"t": t, "goal_id": self._goal_id, "station": self._station,
                            "x": x, "y": y, "yaw": yaw}

    def _fail(self, reason: str) -> None:
        self.failed = self.done = True
        self._set_state("failed", reason)

    def _start_actions(self, state: str, message: str, actions: list[tuple]) -> None:
        self._actions, self._traj, self._dwell_until = list(actions), None, None
        self._set_state(state, message)

    def _run_actions(self, t: float, out: list[Message]) -> bool:
        """Advance the arm action list. True once every action has finished."""
        while True:
            if self._traj is not None:
                tau = t - self._action_start
                self.q_cmd = self._traj.at(tau)
                if tau < self._traj.duration:
                    return False
                self.q_cmd, self._traj = self._traj.q1.copy(), None
            if self._dwell_until is not None:
                if t < self._dwell_until:
                    return False
                self._dwell_until = None
            if not self._actions:
                return True
            action = self._actions.pop(0)
            if action[0] == "move":
                self._traj, self._action_start = Trapezoid(self.q_cmd, action[1]), t
            elif action[0] == "dwell":
                self._dwell_until = t + action[1]
            elif action[0] == "event":
                out.append(("cell/event", {"t": t, "event": action[1], "station": action[2]}))

    # -- main update ---------------------------------------------------------------------------
    def update(self, t: float, messages: list[Message]) -> list[Message]:
        out: list[Message] = []
        for topic, payload in messages:
            if topic == "amr/status" and self._goal_id and payload.get("goal_id") == self._goal_id:
                self._goal_state = payload.get("state", "")
                self._goal_reason = payload.get("reason", "")
                if self._goal_state in ("navigating", "arrived", "failed"):
                    self._goal_ack = True
        if not self.done:
            self._step(t, out)
        if t - self._last_cmd_t >= 1.0 / ARM_RATE_HZ - 1e-9:
            self._last_cmd_t = t
            out.append(("arm/joint_cmd", {"t": t, "q": [float(v) for v in self.q_cmd]}))
        key = (self.state, self.cycle, self.message)
        # Publish on every change, and repeat once per sim second: a subscriber that connects late
        # (ZeroMQ slow joiner) would otherwise miss the current state until the next change.
        if key != self._last_key or t - getattr(self, "_last_status_t", -1e9) >= 1.0:
            self._last_key = key
            self._last_status_t = t
            out.append(("cell/status", self.status_payload(t)))
        return out

    def _step(self, t: float, out: list[Message]) -> None:
        if self.state == "init":
            self.cycle, self._cycle_start = 1, t
            self._dispatch(t, "LOAD", "to_load", out)
        elif self.state in ("to_load", "to_drop", "to_home"):
            if self._goal_state == "failed":
                self._fail(f"AMR leg to {self._station} failed: {self._goal_reason}")
            elif self._goal_state == "arrived":
                self._arrived(t, out)
            elif t - self._leg_start > self.leg_timeout:
                self._fail(f"timeout ({self.leg_timeout:.0f} s sim) on AMR leg to {self._station}")
            elif not self._goal_ack and t - self._goal_sent_t >= GOAL_RESEND:
                self._goal_sent_t = t  # navigator may have missed it (late subscriber): re-send same id
                out.append(self._goal_msg(t))
        elif self.state == "arm":
            if self._run_actions(t, out):
                self._dispatch(t, "DROP", "to_drop", out)
        elif self.state == "unload":
            if self._run_actions(t, out):
                self._dispatch(t, "HOME", "to_home", out)

    def _arrived(self, t: float, out: list[Message]) -> None:
        tp = self.teach
        if self.state == "to_load":
            self._start_actions("arm", "arm: home > above_load > place_load > load > above_load > home", [
                ("move", tp["above_load"]), ("move", tp["place_load"]), ("dwell", DWELL),
                ("event", "load", "LOAD"), ("dwell", DWELL), ("move", tp["above_load"]), ("move", tp["home"])])
        elif self.state == "to_drop":
            self._start_actions("unload", "unload crate at DROP", [
                ("dwell", DWELL), ("event", "unload", "DROP"), ("dwell", DWELL)])
        else:  # back home: cycle complete
            self.cycle_times.append(t - self._cycle_start)
            if self.cycle >= self.cycles:
                self.done = True
                self._set_state("done", f"all {self.cycles} cycle(s) finished")
            else:
                self.cycle, self._cycle_start = self.cycle + 1, t
                self._dispatch(t, "LOAD", "to_load", out)

    def status_payload(self, t: float) -> dict[str, Any]:
        return {"t": t, "state": self.state, "cycle": self.cycle, "message": self.message,
                "cycles": self.cycles, "cycle_times": [round(c, 2) for c in self.cycle_times]}


def run(layout_path: str, host: str, cycles: int, shutdown: bool, wall_timeout: float | None) -> int:
    layout = load_layout(layout_path)
    mission = Mission(layout, cycles)
    bus = Bus("cell", ["sim/clock", "amr/status", "sys/shutdown"], host=host)
    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    started = time.monotonic()
    t = 0.0
    have_clock = False
    try:
        while not stop["flag"] and not mission.done:
            if wall_timeout and time.monotonic() - started > wall_timeout:
                mission._fail("wall-clock timeout")
                break
            incoming = bus.poll(20)
            batch = []
            for topic, payload in incoming:
                if topic == "sys/shutdown":
                    stop["flag"] = True
                elif topic == "sim/clock":
                    t, have_clock = max(t, payload["t"]), True
                else:
                    batch.append((topic, payload))
            if stop["flag"] or not have_clock:
                continue
            for topic, payload in mission.update(t, batch):
                bus.publish(topic, payload)
        if mission.done:
            for topic, payload in mission.update(t, []):  # flush final status
                bus.publish(topic, payload)
            time.sleep(0.3)
            if shutdown:
                bus.publish("sys/shutdown", {"t": t})
                time.sleep(0.2)
        else:
            mission._fail("aborted before completion")
            bus.publish("cell/status", mission.status_payload(t))
            time.sleep(0.2)
    finally:
        bus.close()
    ok = mission.done and not mission.failed
    print(f"cell: {'OK' if ok else 'FAILED'} state={mission.state} cycle_times={mission.cycle_times} {mission.message}",
          flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Factory-cell mission controller")
    parser.add_argument("--layout", default=str(LAYOUT_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--shutdown", action="store_true", help="publish sys/shutdown when finished")
    parser.add_argument("--wall-timeout", type=float, default=None, help="abort after this many wall seconds")
    args = parser.parse_args(argv)
    return run(args.layout, args.host, args.cycles, args.shutdown, args.wall_timeout)


if __name__ == "__main__":
    sys.exit(main())
