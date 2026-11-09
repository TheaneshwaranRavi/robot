"""Start the whole factory-cell co-simulation and run a mission.

    python -m cosim.launch                       # 1 cycle at real time, no recording
    python -m cosim.launch --cycles 2 --rtf 0 --record outputs/cosim
        → outputs/cosim/overview.mp4, dashboard.mp4 and demo.mp4 (side by side)

Processes start in dependency order (world → navigator → dashboard → cell). The mission ends when
the cell program exits; its exit code is the launcher's. Every child is then stopped with SIGTERM
(SIGKILL after a grace period), so nothing is left running.
"""
from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def spawn(module: str, args: list[str], log_dir: Path | None) -> subprocess.Popen:
    command = [sys.executable, "-m", f"cosim.{module}", *args]
    log = open(log_dir / f"{module}.log", "w") if log_dir else subprocess.DEVNULL
    return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def stop(processes: dict[str, subprocess.Popen], grace_s: float = 5.0) -> None:
    for proc in processes.values():
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    for proc in processes.values():
        try:
            proc.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def side_by_side(left: Path, right: Path, out: Path) -> bool:
    if not (shutil.which("ffmpeg") and left.exists() and right.exists()):
        return False
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
        "-filter_complex", "[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack=inputs=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart", str(out),
    ]
    return subprocess.run(command).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the factory-cell co-simulation.")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--rtf", type=float, default=1.0, help="real-time factor; 0 = as fast as possible")
    parser.add_argument("--record", type=Path, help="directory for videos and logs")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0, help="wall-clock limit in seconds")
    args = parser.parse_args(argv)

    log_dir = args.record
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    processes: dict[str, subprocess.Popen] = {}
    try:
        world_args = ["--rtf", str(args.rtf)]
        if args.record:
            world_args += ["--record", str(args.record / "overview.mp4")]
        processes["world_sim"] = spawn("world_sim", world_args, log_dir)
        time.sleep(1.0)  # let PUB sockets bind before subscribers connect
        processes["navigator"] = spawn("navigator", [], log_dir)
        if not args.no_dashboard:
            dash_args = ["--record", str(args.record / "dashboard.mp4")] if args.record else []
            processes["dashboard"] = spawn("dashboard", dash_args, log_dir)
        time.sleep(1.5)  # ZeroMQ slow-joiner: subscriptions must be live before the first goal
        processes["cell"] = spawn("cell", ["--cycles", str(args.cycles)], log_dir)

        started = time.monotonic()
        while True:
            code = processes["cell"].poll()
            if code is not None:
                break
            for name, proc in processes.items():
                if name != "cell" and proc.poll() is not None:
                    print(f"{name} exited early with code {proc.returncode}", file=sys.stderr)
                    return 3
            if time.monotonic() - started > args.timeout:
                print(f"mission did not finish within {args.timeout:.0f} s", file=sys.stderr)
                return 4
            time.sleep(0.2)
    finally:
        stop(processes)

    print(f"cell program finished with exit code {code}")
    if args.record:
        demo = args.record / "demo.mp4"
        if side_by_side(args.record / "overview.mp4", args.record / "dashboard.mp4", demo):
            print(f"demo video: {demo}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
