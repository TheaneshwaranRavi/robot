"""Closed-loop runner: observe → policy → clamp → apply → step → record.

    python -m sim.run                      # scripted pick-and-place, writes outputs/<run>/episode.mp4
    python -m sim.run --policy groot       # remote GR00T PolicyServer (GROOT_HOST/GROOT_PORT)

Exit codes: 0 task succeeded, 1 task not completed, 2 config error, 3 safety stop.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import ROOT, SIM_CONFIG, ConfigError, load_config, summarize
from .safety import ActionLimits, SafetyStop, Watchdog, check_finite, require_target

EXIT_SUCCESS, EXIT_FAILED, EXIT_CONFIG, EXIT_SAFETY = 0, 1, 2, 3
SUCCESS_STREAK = 15  # control steps the cube must stay lifted (0.5 s at 30 Hz)


def run_episode(env, policy, limits: ActionLimits, watchdog: Watchdog, *, seed: int, steps: int,
                execute_k: int, recorder=None, video_size: tuple[int, int] | None = None,
                render_obs: bool = True, realtime: bool = False,
                hold_steps: int = 0) -> dict[str, Any]:
    """Run one episode and return its summary. Never raises SafetyStop: it is recorded instead.

    After success the loop keeps going for `hold_steps` (still policy-driven) so videos show the
    cube being held, then stops."""
    obs = env.reset(seed) if render_obs else (env.reset(seed), env.observe(render=False))[1]
    policy.reset(env.task)
    period = 1.0 / env.control_hz
    camera = env.cameras[0][0]
    summary: dict[str, Any] = {
        "seed": seed, "policy": policy.name, "task": env.task, "steps": 0, "success": False,
        "success_step": None, "stop_reason": None, "policy_calls": 0,
    }
    chunk: np.ndarray | None = None
    index = 0
    streak = 0
    started = time.monotonic()
    try:
        for step in range(steps):
            tick = time.monotonic()
            if chunk is None or index >= min(execute_k, len(chunk)):
                call_start = time.monotonic()
                chunk = np.asarray(policy.act(obs), dtype=np.float64)
                watchdog.check_policy_time(time.monotonic() - call_start)
                if chunk.ndim != 2 or chunk.shape[1] != limits.dim or len(chunk) == 0:
                    raise SafetyStop(f"policy returned shape {chunk.shape}, expected (T, {limits.dim})")
                summary["policy_calls"] += 1
                index = 0
            action = limits.clamp(list(chunk[index]))
            index += 1
            env.apply(action)
            env.step()
            obs = env.observe(render=render_obs)
            check_finite(obs["arm_joint_positions"], "arm_joint_positions")
            if recorder is not None:
                recorder.add(env.render(camera, *video_size) if video_size else obs[camera])
            summary["steps"] = step + 1
            if summary["success"]:
                if step + 1 - summary["success_step"] >= hold_steps:
                    break
            else:
                streak = streak + 1 if env.cube_lifted() else 0
                if streak >= SUCCESS_STREAK:
                    summary["success"] = True
                    summary["success_step"] = step + 1
                    if hold_steps == 0:
                        break
            elapsed = time.monotonic() - tick
            watchdog.check_step_time(elapsed)
            if realtime and elapsed < period:
                time.sleep(period - elapsed)
    except SafetyStop as stop:
        # Holding still is the safe state: targets are left where they are and stepping stops.
        summary["stop_reason"] = stop.reason
    cube, _ = env.cube_pose()
    summary.update({
        "wall_time_s": round(time.monotonic() - started, 3),
        "sim_time_s": round(summary["steps"] / env.control_hz, 3),
        "cube_height_m": round(float(cube[2] - env.cube_rest_z), 4),
        "deadline_misses": watchdog.total_misses,
    })
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the virtual robot in closed loop.")
    parser.add_argument("--policy", choices=["scripted", "groot"], default="scripted")
    parser.add_argument("--config", default=str(SIM_CONFIG))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300, help="control steps per episode")
    parser.add_argument("--execute-k", type=int, default=8,
                        help="actions executed from each chunk before asking the policy again")
    parser.add_argument("--realtime", action="store_true",
                        help="pace at control_hz and stop after 3 missed deadlines in a row")
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    parser.add_argument("--hold-steps", type=int, default=45,
                        help="control steps to keep running after success (shows the hold)")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--video-size", default="480x640", help="HxW of the recorded video")
    parser.add_argument("--host", default=os.getenv("GROOT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GROOT_PORT", "5555")))
    parser.add_argument("--timeout-ms", type=int, default=int(os.getenv("GROOT_TIMEOUT_MS", "5000")))
    parser.add_argument("--api-token", default=os.getenv("GROOT_API_TOKEN") or None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        require_target(config, "sim")
    except (ConfigError, SafetyStop) as error:
        print(f"Invalid config: {error}", file=sys.stderr)
        return EXIT_CONFIG
    for line in summarize(config):
        print(line)
    if not 1 <= args.execute_k <= config["action_horizon"]:
        print(f"--execute-k must be between 1 and {config['action_horizon']}", file=sys.stderr)
        return EXIT_CONFIG
    height, width = (int(v) for v in args.video_size.lower().split("x"))

    from .env import SimEnv  # MuJoCo is imported only once the config is known to be good

    env = SimEnv(config)
    limits = ActionLimits.from_config(config)
    try:
        if args.policy == "groot":
            from .policies.groot import GrootPolicy
            policy = GrootPolicy(config, args.host, args.port, args.timeout_ms, args.api_token)
            policy_timeout = args.timeout_ms / 1000 + 0.5
        else:
            from .policies.scripted import ScriptedPickPolicy
            policy = ScriptedPickPolicy(env, config["action_horizon"], max(limits.high[:6]))
            policy_timeout = 1.0
    except SafetyStop as stop:
        print(f"Safety stop: {stop.reason}", file=sys.stderr)
        return EXIT_SAFETY

    from .recorder import Recorder

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            out_dir = Path(args.out) / f"{run_id}-{args.policy}-seed{seed}"
            recorder = Recorder(out_dir, fps=config["control_hz"], enabled=not args.no_record)
            watchdog = Watchdog(policy_timeout, 1.0 / config["control_hz"], realtime=args.realtime)
            summary = run_episode(
                env, policy, limits, watchdog, seed=seed, steps=args.steps,
                execute_k=args.execute_k, recorder=recorder, video_size=(height, width),
                realtime=args.realtime, hold_steps=args.hold_steps,
            )
            video = recorder.save_video()
            summary["video"] = str(video) if video else None
            recorder.save_summary(summary)
            results.append(summary)
            outcome = "SUCCESS" if summary["success"] else (
                f"SAFETY STOP: {summary['stop_reason']}" if summary["stop_reason"] else "not completed")
            print(f"seed {seed}: {outcome} after {summary['steps']} steps "
                  f"({summary['sim_time_s']} s sim, {summary['wall_time_s']} s wall); {out_dir}")
    finally:
        policy.close()
        env.close()

    if any(r["stop_reason"] for r in results):
        return EXIT_SAFETY
    return EXIT_SUCCESS if all(r["success"] for r in results) else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
