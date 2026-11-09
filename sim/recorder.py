"""Episode video and summary writer. Recording problems never fail a run."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


class Recorder:
    def __init__(self, out_dir: Path, fps: int, enabled: bool = True) -> None:
        self.out_dir = out_dir
        self.fps = fps
        self.enabled = enabled
        self.frames: list[np.ndarray] = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, frame: np.ndarray) -> None:
        if self.enabled:
            self.frames.append(np.ascontiguousarray(frame, dtype=np.uint8))

    def save_video(self) -> Path | None:
        if not self.enabled or not self.frames:
            return None
        if shutil.which("ffmpeg"):
            path = self.out_dir / "episode.mp4"
            if self._write_mp4(path):
                return path
        print("warning: ffmpeg unavailable or failed; writing a GIF instead", file=sys.stderr)
        return self._write_gif(self.out_dir / "episode.gif")

    def _write_mp4(self, path: Path) -> bool:
        height, width, _ = self.frames[0].shape
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
        try:
            subprocess.run(command, input=b"".join(f.tobytes() for f in self.frames), check=True)
            return True
        except (OSError, subprocess.CalledProcessError):
            return False

    def _write_gif(self, path: Path) -> Path | None:
        try:
            from PIL import Image
        except ImportError:
            print("warning: Pillow missing; no video written", file=sys.stderr)
            return None
        images = [Image.fromarray(f) for f in self.frames[::2]]
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=int(2000 / self.fps), loop=0)
        return path

    def save_summary(self, summary: dict[str, Any]) -> Path:
        path = self.out_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        return path
