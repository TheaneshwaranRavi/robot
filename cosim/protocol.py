"""Shared message bus for the factory-cell co-simulation.

Three processes exchange state over ZeroMQ PUB/SUB, like Visual Components, Isaac Sim and Nav2
exchanging joint values, sensor data and AMR poses:

    world_sim  (physics + sensors)  binds tcp://*:5601
    navigator  (localization + nav) binds tcp://*:5602
    cell       (arm program, PLC)   binds tcp://*:5603

Every message is a two-frame multipart: [topic (utf-8 bytes), msgpack(payload dict)].
Every payload has "t": simulation time in seconds, taken from the latest sim/clock.
world_sim owns the clock; nobody else uses wall time for control decisions.

Topic catalogue (publisher → payload keys):

  world_sim
    sim/clock      {t, rtf}                              every physics publish tick (≥ 50 Hz sim time)
    amr/scan       {t, angle_min, angle_inc, range_max, ranges: [float]}   lidar, layout.lidar.rate_hz
                   angle_min/angle_inc are in the AMR frame (0 = forward, CCW positive); inf → range_max
    amr/odom       {t, x, y, yaw, v, w}                  wheel odometry in the odom frame (drifts), 50 Hz
                   odom frame starts equal to the map frame at the AMR start pose
    amr/truth      {t, x, y, yaw}                        ground truth in the map frame — EVALUATION/DASHBOARD ONLY,
                                                         the navigator must never subscribe to it
    arm/state      {t, q: [6]}                           measured arm joint angles
    cell/crate     {t, state: "on_load_station"|"on_amr"|"on_drop_table", x, y}

  navigator
    amr/cmd_vel    {t, v, w}                             velocity command, clipped to layout.amr.max_v/max_w
    amr/pose       {t, x, y, yaw, std_xy, std_yaw}       localized pose in the map frame (particle filter)
    amr/path       {t, goal_id, points: [[x, y], ...]}   current global plan
    amr/status     {t, goal_id, state: "idle"|"navigating"|"arrived"|"failed", reason}

  cell
    arm/joint_cmd  {t, q: [6]}                           streamed joint targets (30 Hz sim time)
    amr/goal       {t, goal_id, station, x, y, yaw}      dispatch the AMR to a station (goal_id unique int)
    cell/event     {t, event: "load"|"unload", station}  material handling; world_sim moves the crate
    cell/status    {t, state, cycle, message}            human-readable progress
    sys/shutdown   {t}                                   everyone exits cleanly
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import msgpack
import zmq

LAYOUT_PATH = Path(__file__).resolve().parent / "layout.json"

PORTS = {"world": 5601, "navigator": 5602, "cell": 5603}


def load_layout(path: str | Path = LAYOUT_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


class Bus:
    """One PUB socket bound on this process's port, one SUB socket connected to the others."""

    def __init__(self, name: str, subscribe: Iterable[str], host: str = "127.0.0.1",
                 ports: dict[str, int] | None = None) -> None:
        ports = ports or PORTS
        self.name = name
        self.context = zmq.Context.instance()
        self.pub = self.context.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.setsockopt(zmq.SNDHWM, 1000)
        self.pub.bind(f"tcp://*:{ports[name]}")
        self.sub = self.context.socket(zmq.SUB)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.setsockopt(zmq.RCVHWM, 1000)
        for peer, port in ports.items():
            if peer != name:
                self.sub.connect(f"tcp://{host}:{port}")
        for topic in subscribe:
            self.sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.pub.send_multipart([topic.encode(), msgpack.packb(payload, use_bin_type=True)])

    def poll(self, timeout_ms: int = 0) -> list[tuple[str, dict[str, Any]]]:
        """Return all queued messages (waits up to timeout_ms for the first one)."""
        messages = []
        events = dict(self.poller.poll(timeout_ms))
        while self.sub in events:
            topic, payload = self.sub.recv_multipart()
            messages.append((topic.decode(), msgpack.unpackb(payload, raw=False)))
            events = dict(self.poller.poll(0))
        return messages

    def close(self) -> None:
        self.pub.close(0)
        self.sub.close(0)
