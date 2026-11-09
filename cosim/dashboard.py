"""Top-down "AMR position synchronised back" dashboard.

Subscribes to every topic on the bus and renders a 960x540 plan view: map (walls, racks, fences,
stations), planned path, lidar points, true pose (green), estimated pose (blue) with its
uncertainty, crate, arm base/orientation, plus a side panel with sim time, mission state, pose
numbers, arm joint bars and a legend. No GUI needed: frames go to an mp4 (``--record``, 10 fps of
*sim* time, Pillow drawing + ffmpeg) and/or a final PNG (``--snapshot``).

Run:  python -m cosim.dashboard [--layout PATH] [--host 127.0.0.1] [--record out.mp4] [--snapshot out.png]
Stops on sys/shutdown, SIGINT/SIGTERM, or ~1 s of sim time after cell/status reports done/failed.
"""
from __future__ import annotations

import argparse
import math
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import msgpack
import zmq
from PIL import Image, ImageDraw, ImageFont

from .protocol import LAYOUT_PATH, PORTS, load_layout

W, H = 960, 540
SS = 2                      # supersampling factor for smooth edges
MAP_X, MAP_Y, MAP_PX = 10, 46, 60.0   # map origin (top-left, px) and pixels per metre -> 720 x 480
PANEL_X = 742
JOINT_RANGES = [(-2.6, 2.6), (-1.5, 1.8), (-0.2, 2.6), (-2.6, 2.6), (-0.2, 2.4), (-3.0, 3.0)]  # arm_table.xml

BG = (244, 245, 247)
INK = (31, 36, 48)
MUTED = (110, 118, 132)
FLOOR = (252, 252, 253)
GRID = (230, 233, 238)
WALL = (58, 64, 78)
RACK = (176, 182, 194)
FENCE = (232, 150, 40)
TRUE_C = (30, 160, 70)
EST_C = (35, 99, 235)
PATH_C = (240, 120, 20)
LIDAR_C = (220, 40, 60)
CRATE_C = (150, 98, 52)
STATION_C = (120, 90, 200)
PANEL_BG = (255, 255, 255)

_FONT_CACHE: dict[tuple[str, int], ImageFont.ImageFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    key = ("b" if bold else "r", size)
    if key not in _FONT_CACHE:
        candidates = ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                      else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]
        loaded = None
        for path in candidates:
            try:
                loaded = ImageFont.truetype(path, size * SS, index=1 if (bold and path.endswith(".ttc")) else 0)
                break
            except OSError:
                continue
        _FONT_CACHE[key] = loaded or ImageFont.load_default(size * SS)
    return _FONT_CACHE[key]


def wrap_deg(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Dashboard:
    """Holds the latest message per topic and draws frames from it."""

    def __init__(self, layout: dict[str, Any]) -> None:
        self.layout = layout
        self.msg: dict[str, dict[str, Any]] = {}
        self.trail: deque[tuple[float, float]] = deque(maxlen=600)
        self.t = 0.0

    def ingest(self, topic: str, payload: dict[str, Any]) -> None:
        self.msg[topic] = payload
        if topic == "sim/clock":
            self.t = max(self.t, payload.get("t", 0.0))
        elif topic == "amr/truth":
            self.trail.append((payload["x"], payload["y"]))

    # -- coordinate helpers (drawing happens at SS x resolution) -------------------------------
    @staticmethod
    def px(x: float, y: float) -> tuple[float, float]:
        return (MAP_X + x * MAP_PX) * SS, (MAP_Y + (8.0 - y) * MAP_PX) * SS

    def _rect_poly(self, cx, cy, yaw, sx, sy) -> list[tuple[float, float]]:
        c, s = math.cos(yaw), math.sin(yaw)
        return [self.px(cx + c * dx - s * dy, cy + s * dx + c * dy)
                for dx, dy in ((sx / 2, sy / 2), (-sx / 2, sy / 2), (-sx / 2, -sy / 2), (sx / 2, -sy / 2))]

    # -- rendering -----------------------------------------------------------------------------
    def render(self) -> Image.Image:
        img = Image.new("RGB", (W * SS, H * SS), BG)
        d = ImageDraw.Draw(img, "RGBA")
        self._header(d)
        self._map(d)
        self._panel(d)
        return img.resize((W, H), Image.LANCZOS)

    def _text(self, d, xy, text, size=11, fill=INK, bold=False, anchor="la") -> None:
        d.text((xy[0] * SS, xy[1] * SS), text, font=font(size, bold), fill=fill, anchor=anchor)

    def _header(self, d) -> None:
        d.rectangle((0, 0, W * SS, 36 * SS), fill=(24, 30, 44))
        self._text(d, (12, 18), "AMR position synchronised back", 15, (255, 255, 255), True, "lm")
        self._text(d, (W - 12, 18), f"sim time  {self.t:7.1f} s", 15, (255, 255, 255), True, "rm")

    def _map(self, d) -> None:
        lay = self.layout
        fw, fh = lay["floor"]["size"]
        x0, y0 = self.px(0, fh)
        x1, y1 = self.px(fw, 0)
        d.rectangle((x0, y0, x1, y1), fill=FLOOR)
        for gx in range(int(fw) + 1):
            d.line((*self.px(gx, 0), *self.px(gx, fh)), fill=GRID, width=SS)
            if gx < fw:
                self._text(d, (self.px(gx, 0)[0] / SS + 3, self.px(0, 0)[1] / SS - 12), f"{gx}", 9, MUTED)
        for gy in range(int(fh) + 1):
            d.line((*self.px(0, gy), *self.px(fw, gy)), fill=GRID, width=SS)
            if 0 < gy < fh:
                self._text(d, (MAP_X + 3, self.px(0, gy)[1] / SS - 11), f"{gy}", 9, MUTED)
        for wall in lay["walls"]:
            d.polygon(self._rect_poly(*wall["center"], 0.0, *wall["size"]), fill=WALL)
        for obs in lay["obstacles"]:
            name = obs["name"]
            if "fence" in name:
                color, outline = (*FENCE, 230), None
            elif "rack" in name:
                color, outline = RACK, WALL
            else:
                color, outline = (205, 208, 216), MUTED
            poly = self._rect_poly(*obs["center"], 0.0, *obs["size"])
            d.polygon(poly, fill=color, outline=outline)
            if "rack" in name or name in ("drop_table",):
                cx, cy = self.px(*obs["center"])
                self._text(d, (cx / SS, cy / SS), name.replace("_", " "), 9, INK, False, "mm")
        self._arm(d)
        self._stations(d)
        self._crate(d)
        # path
        path = self.msg.get("amr/path")
        if path and len(path.get("points", [])) > 1:
            pts = [self.px(x, y) for x, y in path["points"]]
            d.line(pts, fill=PATH_C, width=3 * SS, joint="curve")
            gx, gy = pts[-1]
            r = 5 * SS
            d.ellipse((gx - r, gy - r, gx + r, gy + r), outline=PATH_C, width=2 * SS)
        # trail of true poses
        if len(self.trail) > 1:
            d.line([self.px(x, y) for x, y in self.trail], fill=(*TRUE_C, 110), width=2 * SS, joint="curve")
        self._lidar(d)
        self._robots(d)
        d.rectangle((x0, y0, x1, y1), outline=WALL, width=SS)

    def _arm(self, d) -> None:
        arm = self.layout["arm"]
        bx, by, _ = arm["base"]
        cx, cy = self.px(bx, by)
        reach = 0.6 * MAP_PX * SS
        d.ellipse((cx - reach, cy - reach, cx + reach, cy + reach), outline=(*STATION_C, 90), width=SS)
        q = self.msg.get("arm/state", {}).get("q") or arm["teach_points"].get("home", [0.0] * 6)
        yaw = arm["mount_yaw"] + q[0]
        # plan-view reach of the arm: shoulder angle shortens the horizontal extent
        extent = 0.12 + 0.46 * abs(math.sin(q[1] if len(q) > 1 else 0.0) + math.sin((q[1] + q[2]) if len(q) > 2 else 0.0)) / 1.5
        ex, ey = self.px(bx + extent * math.cos(yaw), by + extent * math.sin(yaw))
        d.line((cx, cy, ex, ey), fill=STATION_C, width=5 * SS)
        r = 7 * SS
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=STATION_C, outline=(255, 255, 255), width=SS)
        self._text(d, (cx / SS + 12, cy / SS - 14), "arm", 9, STATION_C, True, "lm")

    def _stations(self, d) -> None:
        fp = self.layout["amr"]["footprint"]
        for name, (x, y, yaw) in ((k, v) for k, v in self.layout["stations"].items() if not k.startswith("_")):
            poly = self._rect_poly(x, y, yaw, *fp)
            d.line(poly + [poly[0]], fill=(*STATION_C, 150), width=SS)
            tx, ty = self.px(x, y)
            self._text(d, (tx / SS, ty / SS + 28), name, 10, STATION_C, True, "mm")

    def _crate(self, d) -> None:
        crate = self.msg.get("cell/crate")
        if not crate or "x" not in crate:
            return
        sx, sy = self.layout["crate"]["size"][:2]
        amr = self.msg.get("amr/truth")
        yaw = amr["yaw"] if amr and crate.get("state") == "on_amr" else 0.0
        poly = self._rect_poly(crate["x"], crate["y"], yaw, sx, sy)
        d.polygon(poly, fill=CRATE_C, outline=(90, 55, 25))

    def _lidar(self, d) -> None:
        scan = self.msg.get("amr/scan")
        pose = self.msg.get("amr/truth") or self.msg.get("amr/pose")
        if not scan or not pose:
            return
        rmax = scan["range_max"]
        r = 1.6 * SS
        for i, rng in enumerate(scan["ranges"]):
            if not math.isfinite(rng) or rng >= rmax - 0.02:
                continue
            a = pose["yaw"] + scan["angle_min"] + i * scan["angle_inc"]
            wx, wy = pose["x"] + rng * math.cos(a), pose["y"] + rng * math.sin(a)
            if not (-0.2 <= wx <= self.layout["floor"]["size"][0] + 0.2 and -0.2 <= wy <= self.layout["floor"]["size"][1] + 0.2):
                continue
            px, py = self.px(wx, wy)
            d.ellipse((px - r, py - r, px + r, py + r), fill=(*LIDAR_C, 200))

    def _robot_shape(self, d, pose, color, fill_alpha: int) -> None:
        fp = self.layout["amr"]["footprint"]
        poly = self._rect_poly(pose["x"], pose["y"], pose["yaw"], *fp)
        d.polygon(poly, fill=(*color, fill_alpha), outline=color)
        d.line(poly + [poly[0]], fill=color, width=2 * SS)
        cx, cy = self.px(pose["x"], pose["y"])
        tip = self.px(pose["x"] + 0.38 * math.cos(pose["yaw"]), pose["y"] + 0.38 * math.sin(pose["yaw"]))
        d.line((cx, cy, *tip), fill=color, width=3 * SS)
        r = 3 * SS
        d.ellipse((tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r), fill=color)

    def _robots(self, d) -> None:
        est = self.msg.get("amr/pose")
        truth = self.msg.get("amr/truth")
        if est:
            rad = max(2 * est.get("std_xy", 0.0), 0.04) * MAP_PX * SS
            cx, cy = self.px(est["x"], est["y"])
            d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=(*EST_C, 40), outline=(*EST_C, 160))
            self._robot_shape(d, est, EST_C, 25)
        if truth:
            self._robot_shape(d, truth, TRUE_C, 70)

    def _panel(self, d) -> None:
        x, y0 = PANEL_X, 46
        d.rounded_rectangle(((x - 6) * SS, y0 * SS, (W - 8) * SS, (MAP_Y + 480) * SS), radius=6 * SS, fill=PANEL_BG, outline=GRID)
        tx = x + 4
        cell = self.msg.get("cell/status", {})
        status = self.msg.get("amr/status", {})
        y = y0 + 8
        self._text(d, (tx, y), "MISSION", 9, MUTED, True)
        state = cell.get("state", "waiting")
        color = {"done": TRUE_C, "failed": LIDAR_C}.get(state, INK)
        self._text(d, (tx, y + 12), state.upper().replace("_", " "), 15, color, True)
        cyc = f"cycle {cell.get('cycle', 0)}/{cell.get('cycles', '-')}"
        self._text(d, (tx, y + 32), cyc, 10, MUTED)
        msg = cell.get("message", "")
        for i, line in enumerate(self._wrap(msg, 26)[:2]):
            self._text(d, (tx, y + 45 + 12 * i), line, 9, MUTED)
        y += 74
        astate = status.get("state", "-")
        reason = f" ({status['reason']})" if status.get("reason") else ""
        self._text(d, (tx, y), f"AMR nav: {astate}{reason}"[:34], 10, INK, True)
        crate = self.msg.get("cell/crate", {}).get("state", "-").replace("_", " ")
        self._text(d, (tx, y + 14), f"crate: {crate}", 10, INK)
        y += 36

        truth, est = self.msg.get("amr/truth"), self.msg.get("amr/pose")
        self._text(d, (tx, y), "POSE (map frame)", 9, MUTED, True)
        self._text(d, (tx, y + 13), self._pose_line("true", truth), 10, TRUE_C, True)
        self._text(d, (tx, y + 27), self._pose_line("est ", est), 10, EST_C, True)
        if truth and est:
            err = math.hypot(truth["x"] - est["x"], truth["y"] - est["y"])
            dyaw = math.degrees(abs(wrap_deg(truth["yaw"] - est["yaw"])))
            self._text(d, (tx, y + 41), f"error {err * 100:5.1f} cm  {dyaw:4.1f} deg", 10, INK)
        y += 62

        self._text(d, (tx, y), "ARM JOINTS", 9, MUTED, True)
        q = self.msg.get("arm/state", {}).get("q", [])
        bar_x0, bar_w = tx + 22, 120
        for i, (lo, hi) in enumerate(JOINT_RANGES):
            by = y + 14 + i * 14
            self._text(d, (tx, by - 1), f"j{i + 1}", 9, MUTED)
            d.rounded_rectangle((bar_x0 * SS, by * SS + 2 * SS, (bar_x0 + bar_w) * SS, (by + 8) * SS), radius=2 * SS,
                                fill=(232, 235, 240))
            if i < len(q):
                frac = max(0.0, min(1.0, (q[i] - lo) / (hi - lo)))
                zero = (0 - lo) / (hi - lo)
                a, b = sorted((zero, frac))
                d.rectangle(((bar_x0 + a * bar_w) * SS, by * SS + 2 * SS, (bar_x0 + b * bar_w) * SS, (by + 8) * SS),
                            fill=STATION_C)
                self._text(d, (bar_x0 + bar_w + 5, by - 1), f"{q[i]:+.2f}", 9, INK)
        y += 14 + 6 * 14 + 8

        self._text(d, (tx, y), "LEGEND", 9, MUTED, True)
        items = [("line", TRUE_C, "true pose (ground truth)"), ("line", EST_C, "estimated pose + 2σ"),
                 ("line", PATH_C, "planned path"), ("dot", LIDAR_C, "lidar hits"),
                 ("box", CRATE_C, "crate"), ("box", STATION_C, "stations / arm")]
        for i, (kind, col, label) in enumerate(items):
            ly = y + 15 + i * 13
            cx = tx + 6
            if kind == "line":
                d.line((cx * SS - 6 * SS, (ly + 5) * SS, cx * SS + 8 * SS, (ly + 5) * SS), fill=col, width=3 * SS)
            elif kind == "dot":
                d.ellipse((cx * SS - 2 * SS, (ly + 3) * SS, cx * SS + 4 * SS, (ly + 9) * SS), fill=col)
            else:
                d.rectangle((cx * SS - 5 * SS, (ly + 1) * SS, cx * SS + 8 * SS, (ly + 9) * SS), fill=col)
            self._text(d, (tx + 20, ly), label, 9, INK)

    @staticmethod
    def _pose_line(label: str, pose: dict[str, Any] | None) -> str:
        if not pose:
            return f"{label}  --"
        return f"{label}  {pose['x']:5.2f} {pose['y']:5.2f}  {math.degrees(pose['yaw']):6.1f}°"

    @staticmethod
    def _wrap(text: str, width: int) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            if len(line) + len(word) + 1 > width and line:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        return lines + ([line] if line else [])


class Recorder:
    """Pipes RGB frames to ffmpeg (libx264 mp4)."""

    def __init__(self, path: str, fps: int = 10) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE)

    def write(self, img: Image.Image) -> None:
        self.proc.stdin.write(img.convert("RGB").tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=60)


def run(args: argparse.Namespace) -> int:
    dash = Dashboard(load_layout(args.layout))
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, 5000)
    for port in PORTS.values():
        sub.connect(f"tcp://{args.host}:{port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    recorder = Recorder(args.record, args.fps) if args.record else None
    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    period = 1.0 / args.fps
    next_frame: float | None = None
    end_t: float | None = None
    started = time.monotonic()
    frames = 0
    try:
        while not stop["flag"]:
            if args.wall_timeout and time.monotonic() - started > args.wall_timeout:
                break
            if not dict(poller.poll(50)):
                continue
            # Drain everything queued before rendering. ZeroMQ fair-queues between publishers, so
            # rendering mid-drain lets the busy world_sim stream (clock, truth) fall far behind the
            # navigator and cell streams, and the view shows stale ground truth.
            while True:
                try:
                    topic, raw = sub.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                topic_s, payload = topic.decode(), msgpack.unpackb(raw, raw=False)
                if topic_s == "sys/shutdown":
                    stop["flag"] = True
                    break
                dash.ingest(topic_s, payload)
                if topic_s == "cell/status" and payload.get("state") in ("done", "failed") and end_t is None:
                    end_t = dash.t + 1.0
            if next_frame is None and "sim/clock" in dash.msg:
                next_frame = dash.t
            if next_frame is not None and dash.t >= next_frame:
                # One render per drain; repeat it for every frame slot that has passed, so the video
                # keeps one frame per 1/fps of sim time and stays in step with world_sim's recording.
                due = int((dash.t - next_frame) / period) + 1
                if recorder:
                    image = dash.render()
                    for _ in range(due):
                        recorder.write(image)
                    frames += due
                next_frame += due * period
            if end_t is not None and dash.t >= end_t:
                break
        final = dash.render()
        if args.snapshot:
            final.save(args.snapshot)
    finally:
        if recorder:
            recorder.close()
        sub.close(0)
    print(f"dashboard: {frames} frames, t={dash.t:.1f}s", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Top-down co-simulation dashboard / recorder")
    parser.add_argument("--layout", default=str(LAYOUT_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--record", default=None, help="write an mp4 (960x540, --fps of sim time)")
    parser.add_argument("--snapshot", default=None, help="write the last frame as PNG on exit")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--wall-timeout", type=float, default=None)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
