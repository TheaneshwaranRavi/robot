"""MuJoCo physics and sensors for the factory-cell co-simulation.

Run standalone (headless by default):

    python -m cosim.world_sim --rtf 0 --duration 5 --record outputs/world_smoke.mp4
"""
from __future__ import annotations

import argparse
import math
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from .protocol import LAYOUT_PATH, Bus, load_layout
from .world_model import build_world_xml


PHYSICS_DT = 0.002
STATE_RATE_HZ = 50.0
RECORD_RATE_HZ = 10.0
COMMAND_TIMEOUT_S = 0.5
WORLD_TOPICS = {
    "sim/clock",
    "amr/scan",
    "amr/odom",
    "amr/truth",
    "arm/state",
    "cell/crate",
}


def _object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"generated world has no {kind.name} named {name!r}")
    return object_id


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class WorldSim:
    """Physics-only world core; networking and wall-clock pacing live in :func:`run`."""

    def __init__(self, layout: dict[str, Any], seed: int = 0) -> None:
        self.layout = layout
        self.model = mujoco.MjModel.from_xml_string(build_world_xml(layout))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.shutdown_requested = False

        self.amr_body_id = _object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "amr")
        self.amr_joint_ids = np.array([
            _object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("amr_x", "amr_y", "amr_yaw")
        ])
        self.amr_qadr = self.model.jnt_qposadr[self.amr_joint_ids]
        self.amr_dofadr = self.model.jnt_dofadr[self.amr_joint_ids]

        self.arm_joint_ids = np.array([
            _object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in layout["arm"]["joint_names"]
        ])
        self.arm_qadr = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_actuator_ids = np.array([
            _object_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a{index}")
            for index in range(1, 7)
        ])
        self.arm_low = self.model.jnt_range[self.arm_joint_ids, 0].copy()
        self.arm_high = self.model.jnt_range[self.arm_joint_ids, 1].copy()

        self.crate_body_id = _object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "crate")
        self.crate_mocap_id = int(self.model.body_mocapid[self.crate_body_id])
        if self.crate_mocap_id < 0:
            raise ValueError("generated crate body is not a mocap body")

        start = np.asarray(layout["amr"]["start"], dtype=np.float64)
        self.data.qpos[self.amr_qadr] = start
        home = np.asarray(layout["arm"]["teach_points"]["home"], dtype=np.float64)
        if home.shape != (6,):
            raise ValueError("arm.teach_points.home must contain six joint angles")
        home = np.clip(home, self.arm_low, self.arm_high)
        self.data.qpos[self.arm_qadr] = home
        self.data.ctrl[self.arm_actuator_ids] = home
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        if gripper_id >= 0:
            self.data.ctrl[gripper_id] = 0.0

        self._requested_v = 0.0
        self._requested_w = 0.0
        self._last_command_time: float | None = None
        self._arm_targets = home.copy()

        self._odom_pose = start.copy()
        self._odom_v = 0.0
        self._odom_w = 0.0
        self._odom_period = 1.0 / float(layout["odom"]["rate_hz"])
        self._last_odom_time = 0.0
        self._next_odom_time = self._odom_period

        self.crate_state = "on_load_station"
        crate_height = float(layout["crate"]["size"][2])
        load_x, load_y, _ = layout["stations"]["LOAD"]
        self._crate_fixed_position = np.array(
            [load_x, load_y, float(layout["amr"]["deck_height"]) + crate_height / 2],
            dtype=np.float64,
        )
        self._crate_fixed_yaw = 0.0
        self._renderers: dict[tuple[str, int, int], mujoco.Renderer] = {}
        self._sync_crate_pose()
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def arm_targets(self) -> np.ndarray:
        return self._arm_targets.copy()

    @property
    def commanded_velocity(self) -> tuple[float, float]:
        if self._command_timed_out():
            return 0.0, 0.0
        return self._requested_v, self._requested_w

    def _command_timed_out(self) -> bool:
        return (
            self._last_command_time is None
            or self.time - self._last_command_time >= COMMAND_TIMEOUT_S
        )

    def amr_pose(self) -> np.ndarray:
        pose = np.asarray(self.data.qpos[self.amr_qadr], dtype=np.float64).copy()
        pose[2] = _wrap_angle(float(pose[2]))
        return pose

    def set_amr_pose(self, x: float, y: float, yaw: float) -> None:
        """Set the planar pose directly, useful for station events and deterministic tests."""
        self.data.qpos[self.amr_qadr] = (float(x), float(y), float(yaw))
        self.data.qvel[self.amr_dofadr] = 0.0
        self._sync_crate_pose()
        mujoco.mj_forward(self.model, self.data)

    def set_cmd_vel(self, v: float, w: float) -> None:
        v, w = float(v), float(w)
        if not (math.isfinite(v) and math.isfinite(w)):
            v, w = 0.0, 0.0
        self._requested_v = float(np.clip(v, -self.layout["amr"]["max_v"], self.layout["amr"]["max_v"]))
        self._requested_w = float(np.clip(w, -self.layout["amr"]["max_w"], self.layout["amr"]["max_w"]))
        self._last_command_time = self.time

    def set_joint_targets(self, q: Any) -> None:
        targets = np.asarray(q, dtype=np.float64)
        if targets.shape != (6,) or not np.all(np.isfinite(targets)):
            return
        targets = np.clip(targets, self.arm_low, self.arm_high)
        self._arm_targets = targets.copy()
        self.data.ctrl[self.arm_actuator_ids] = targets

    def _set_crate_pose(self, position: np.ndarray, yaw: float) -> None:
        self.data.mocap_pos[self.crate_mocap_id] = position
        self.data.mocap_quat[self.crate_mocap_id] = (
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        )

    def _sync_crate_pose(self) -> None:
        if self.crate_state == "on_amr":
            x, y, yaw = self.amr_pose()
            z = float(self.layout["amr"]["deck_height"]) + float(self.layout["crate"]["size"][2]) / 2
            self._set_crate_pose(np.array([x, y, z], dtype=np.float64), float(yaw))
        else:
            self._set_crate_pose(self._crate_fixed_position, self._crate_fixed_yaw)

    def crate_position(self) -> np.ndarray:
        return np.asarray(self.data.mocap_pos[self.crate_mocap_id], dtype=np.float64).copy()

    def handle_event(self, event: str, station: str | None = None) -> None:
        del station  # The event owner validates mission sequencing; the world performs the transfer.
        if event == "load":
            self.crate_state = "on_amr"
            self._sync_crate_pose()
            mujoco.mj_forward(self.model, self.data)
        elif event == "unload":
            drop_table = next(
                obstacle for obstacle in self.layout["obstacles"] if obstacle["name"] == "drop_table"
            )
            crate_height = float(self.layout["crate"]["size"][2])
            self._crate_fixed_position = np.array(
                [*drop_table["center"], float(drop_table["h"]) + crate_height / 2],
                dtype=np.float64,
            )
            self._crate_fixed_yaw = 0.0
            self.crate_state = "on_drop_table"
            self._sync_crate_pose()
            mujoco.mj_forward(self.model, self.data)

    def handle_message(self, topic: str, payload: dict[str, Any]) -> None:
        if topic == "arm/joint_cmd":
            self.set_joint_targets(payload.get("q"))
        elif topic == "amr/cmd_vel":
            self.set_cmd_vel(payload.get("v", 0.0), payload.get("w", 0.0))
        elif topic == "cell/event":
            self.handle_event(str(payload.get("event", "")), payload.get("station"))
        elif topic == "sys/shutdown":
            self.shutdown_requested = True

    process_message = handle_message

    def _apply_amr_velocity(self) -> None:
        v, w = self.commanded_velocity
        yaw = float(self.data.qpos[self.amr_qadr[2]])
        self.data.qvel[self.amr_dofadr] = (
            v * math.cos(yaw),
            v * math.sin(yaw),
            w,
        )

    def _update_odom(self) -> None:
        epsilon = PHYSICS_DT / 10
        while self.time + epsilon >= self._next_odom_time:
            dt = self._next_odom_time - self._last_odom_time
            truth_yaw = float(self.data.qpos[self.amr_qadr[2]])
            vx, vy, actual_w = self.data.qvel[self.amr_dofadr]
            actual_v = float(vx * math.cos(truth_yaw) + vy * math.sin(truth_yaw))
            odom = self.layout["odom"]
            measured_v = actual_v * float(odom["v_scale_error"]) + self.rng.normal(
                0.0, float(odom["v_noise_std"])
            )
            measured_w = float(actual_w) * float(odom["w_scale_error"]) + self.rng.normal(
                0.0, float(odom["w_noise_std"])
            )
            self._odom_pose[0] += measured_v * math.cos(self._odom_pose[2]) * dt
            self._odom_pose[1] += measured_v * math.sin(self._odom_pose[2]) * dt
            self._odom_pose[2] = _wrap_angle(self._odom_pose[2] + measured_w * dt)
            self._odom_v = float(measured_v)
            self._odom_w = float(measured_w)
            self._last_odom_time = self._next_odom_time
            self._next_odom_time += self._odom_period

    def step(self, steps: int = 1) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(steps):
            self._apply_amr_velocity()
            mujoco.mj_step(self.model, self.data)
            self._sync_crate_pose()
            mujoco.mj_forward(self.model, self.data)
            self._update_odom()

    def step_for(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        steps = int(math.ceil(float(seconds) / PHYSICS_DT - 1e-12))
        self.step(steps)

    def scan(self, noise: bool = True) -> dict[str, Any]:
        lidar = self.layout["lidar"]
        beam_count = int(lidar["beams"])
        fov = float(lidar["fov"])
        range_max = float(lidar["range_max"])
        angles = np.linspace(-fov / 2, fov / 2, beam_count, dtype=np.float64)
        x, y, yaw = self.amr_pose()
        world_angles = angles + yaw
        directions = np.column_stack(
            (np.cos(world_angles), np.sin(world_angles), np.zeros(beam_count))
        ).astype(np.float64, copy=False)
        origin = np.array([x, y, float(lidar["height"])], dtype=np.float64)
        geom_ids = np.empty(beam_count, dtype=np.int32)
        distances = np.empty(beam_count, dtype=np.float64)
        mujoco.mj_multiRay(
            self.model,
            self.data,
            origin,
            np.ascontiguousarray(directions.reshape(-1)),
            None,
            True,
            self.amr_body_id,
            geom_ids,
            distances,
            None,
            beam_count,
            range_max,
        )
        distances = np.where(distances < 0, range_max, distances)
        if noise and float(lidar["noise_std"]) > 0:
            distances += self.rng.normal(0.0, float(lidar["noise_std"]), beam_count)
        distances = np.clip(distances, 0.0, range_max)
        return {
            "t": self.time,
            "angle_min": float(angles[0]),
            "angle_inc": float(angles[1] - angles[0]) if beam_count > 1 else 0.0,
            "range_max": range_max,
            "ranges": [float(value) for value in distances],
        }

    lidar_scan = scan

    def clock_payload(self, rtf: float) -> dict[str, Any]:
        return {"t": self.time, "rtf": float(rtf)}

    def truth_payload(self) -> dict[str, Any]:
        x, y, yaw = self.amr_pose()
        return {"t": self.time, "x": float(x), "y": float(y), "yaw": float(yaw)}

    def odom_payload(self) -> dict[str, Any]:
        x, y, yaw = self._odom_pose
        return {
            "t": self.time,
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
            "v": self._odom_v,
            "w": self._odom_w,
        }

    def arm_state_payload(self) -> dict[str, Any]:
        q = np.asarray(self.data.qpos[self.arm_qadr], dtype=np.float64)
        return {"t": self.time, "q": [float(value) for value in q]}

    arm_payload = arm_state_payload

    def crate_payload(self) -> dict[str, Any]:
        x, y, _ = self.crate_position()
        return {
            "t": self.time,
            "state": self.crate_state,
            "x": float(x),
            "y": float(y),
        }

    def topic_payloads(self, rtf: float = 1.0, include_scan: bool = True) -> dict[str, dict[str, Any]]:
        payloads = {
            "sim/clock": self.clock_payload(rtf),
            "amr/odom": self.odom_payload(),
            "amr/truth": self.truth_payload(),
            "arm/state": self.arm_state_payload(),
            "cell/crate": self.crate_payload(),
        }
        if include_scan:
            payloads["amr/scan"] = self.scan()
        return payloads

    def publish_all(self, bus: Any, rtf: float = 1.0) -> None:
        for topic, payload in self.topic_payloads(rtf=rtf, include_scan=True).items():
            bus.publish(topic, payload)

    def render(self, camera: str = "overview", width: int = 960, height: int = 540) -> np.ndarray:
        key = (camera, int(height), int(width))
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self._renderers[key]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


class _FFmpegRecorder:
    def __init__(self, path: Path, width: int = 960, height: int = 540, fps: int = 10) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("--record requires ffmpeg on PATH")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            [
                executable,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.frames = 0

    def add(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"recording frame has shape {frame.shape}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input is closed")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError as error:
            detail = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"ffmpeg stopped while recording: {detail}") from error
        self.frames += 1

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("ffmpeg did not finish within 30 seconds")
        detail = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}: {detail}")


def _publish_state(world: WorldSim, bus: Any, rtf: float) -> None:
    for topic, payload in world.topic_payloads(rtf=rtf, include_scan=False).items():
        bus.publish(topic, payload)


def run(
    world: WorldSim,
    bus: Any,
    *,
    rtf: float,
    duration: float | None,
    recorder: _FFmpegRecorder | None = None,
    stop_requested: Callable[[], bool] = lambda: False,
) -> float:
    """Run the process loop and return elapsed simulation time."""
    start_sim = world.time
    end_sim = None if duration is None else start_sim + duration
    wall_start = time.monotonic()
    state_period = 1.0 / STATE_RATE_HZ
    scan_period = 1.0 / float(world.layout["lidar"]["rate_hz"])
    frame_period = 1.0 / RECORD_RATE_HZ
    next_state = world.time + state_period
    next_scan = world.time + scan_period
    next_frame = world.time + frame_period
    epsilon = PHYSICS_DT / 10

    world.publish_all(bus, rtf=rtf)
    if recorder is not None:
        recorder.add(world.render("overview", recorder.width, recorder.height))

    while not world.shutdown_requested and not stop_requested():
        if end_sim is not None and world.time + epsilon >= end_sim:
            break
        for topic, payload in bus.poll(0):
            world.handle_message(topic, payload)
        if world.shutdown_requested or stop_requested():
            break

        world.step()
        if world.time + epsilon >= next_state:
            _publish_state(world, bus, rtf)
            while next_state <= world.time + epsilon:
                next_state += state_period
        if world.time + epsilon >= next_scan:
            bus.publish("amr/scan", world.scan())
            while next_scan <= world.time + epsilon:
                next_scan += scan_period
        if recorder is not None and world.time + epsilon >= next_frame:
            if end_sim is None or world.time + epsilon < end_sim:
                recorder.add(world.render("overview", recorder.width, recorder.height))
            while next_frame <= world.time + epsilon:
                next_frame += frame_period

        if rtf > 0:
            deadline = wall_start + (world.time - start_sim) / rtf
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    return world.time - start_sim


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the factory-cell MuJoCo world simulator.")
    parser.add_argument("--layout", type=Path, default=LAYOUT_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rtf", type=float, default=1.0, help="real-time factor; 0 runs unpaced")
    parser.add_argument("--duration", type=float, help="stop after this many simulated seconds")
    parser.add_argument("--record", type=Path, help="write the overview camera to an MP4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="use offscreen rendering (the default; accepted for explicit launch scripts)",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.rtf) or args.rtf < 0:
        parser.error("--rtf must be finite and non-negative")
    if args.duration is not None and (
        not math.isfinite(args.duration) or args.duration < 0
    ):
        parser.error("--duration must be finite and non-negative")

    stopped = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stopped
        stopped = True

    previous_handlers: dict[signal.Signals, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

    world: WorldSim | None = None
    bus: Bus | None = None
    recorder: _FFmpegRecorder | None = None
    try:
        world = WorldSim(load_layout(args.layout), seed=args.seed)
        bus = Bus(
            "world",
            subscribe=("arm/joint_cmd", "amr/cmd_vel", "cell/event", "sys/shutdown"),
            host=args.host,
        )
        if args.record is not None:
            recorder = _FFmpegRecorder(args.record)
        elapsed = run(
            world,
            bus,
            rtf=args.rtf,
            duration=args.duration,
            recorder=recorder,
            stop_requested=lambda: stopped,
        )
    finally:
        close_error: BaseException | None = None
        if recorder is not None:
            try:
                recorder.close()
            except BaseException as error:
                close_error = error
        if world is not None:
            world.close()
        if bus is not None:
            bus.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if close_error is not None:
            raise close_error

    destination = f"; video {args.record}" if args.record is not None else ""
    print(f"world_sim stopped cleanly at t={world.time:.3f} s ({elapsed:.3f} s simulated){destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
