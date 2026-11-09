# Factory-cell co-simulation — build contract

An open, CPU-only (MacBook) version of a "Visual Components + Isaac Sim + Nav2" closed loop:

```
 cell (arm program / PLC)  ──arm/joint_cmd──▶  world_sim (MuJoCo physics + sensors)
        ▲   │ amr/goal, cell/event                   │ amr/scan, amr/odom, arm/state, sim/clock
        │   ▼                                        ▼
        └── amr/status, amr/pose ◀──────────  navigator (particle filter + A* + path follower)
                                       amr/cmd_vel ─────▶ world_sim
```

Read first: `cosim/protocol.py` (message bus + topic catalogue — the single source of truth for messages),
`cosim/layout.json` (cell geometry, stations, sensor parameters), `sim/assets/arm_table.xml` (the arm MJCF to reuse).
Python 3.12 venv: `.venv/bin/python` (has mujoco 4.0, numpy, pyzmq, msgpack, Pillow; ffmpeg on PATH).

## Rules for every builder
- Own ONLY your files (listed below). Do not edit `protocol.py`, `layout.json`, other builders' files, or anything outside `cosim/` and your test file. If the contract is wrong or missing something, write it in your final report instead of changing it (one exception: the cell builder fills `arm.teach_points` in layout.json via its teach tool).
- Each process is runnable as `python -m cosim.<module>` with `--layout`, `--host`, and stops cleanly on `sys/shutdown` or SIGINT/SIGTERM.
- All timing uses sim time `t` from messages (world_sim's clock), never wall time, so everything scales with `--rtf`.
- The navigator never subscribes to `amr/truth`.
- Unit tests must not need other processes running; they must pass with `.venv/bin/python -m unittest discover -s tests -v` in well under a minute, and must skip cleanly (`unittest.skipUnless`) if mujoco/numpy are missing.
- No commits, no pushes. Finish with a short report: what you built, how you tested it, known limitations, and any contract problems.

## world_sim — `cosim/world_sim.py` (+ `cosim/world_model.py` optional), test `tests/test_cosim_world.py`
- Build the MuJoCo scene programmatically from layout.json: floor, walls/obstacles as static boxes, the arm from `sim/assets/arm_table.xml` mounted at `arm.base` with `arm.mount_yaw` (reuse its bodies/joints/actuators; drop its table/cube/camera), the AMR (box footprint, deck at `deck_height`; planar x/y/yaw joints driven by velocity actuators or direct qvel from the differential-drive command), the crate, a station marker for LOAD/DROP, and two cameras: `overview` (oblique, whole cell) and `arm_cam` (looking at the arm and LOAD station).
- Physics at 2 ms. Subscribe `arm/joint_cmd` (position targets, clipped to joint ranges), `amr/cmd_vel` (clip to max_v/max_w; zero it if no command for 0.5 s sim time — watchdog), `cell/event` (load: crate placed on the AMR deck and carried along — weld/kinematic attach is fine; unload at DROP: crate placed on drop_table), `sys/shutdown`.
- Publish per protocol: `sim/clock`, `amr/scan` (2D lidar via `mujoco.mj_ray`/`mj_multiRay` at lidar.height, excluding the AMR's own geoms, Gaussian noise), `amr/odom` (integrate the AMR's actual velocity with noise and the scale errors in layout.odom so it drifts realistically), `amr/truth`, `arm/state`, `cell/crate`.
- `--rtf` real-time factor (default 1.0; `--rtf 0` = as fast as possible), `--record <path.mp4>` renders the `overview` camera at 10 fps sim time (960x540) with ffmpeg, `--headless` default.
- Tests: scene builds from layout; lidar returns expected range to a known wall from the start pose (±5 cm); cmd_vel moves the AMR forward and odom drifts but stays within a few % ; arm follows a joint_cmd; crate load/unload; cmd_vel watchdog stops the AMR.

## navigator — `cosim/navigator.py` (+ helpers `cosim/nav_*.py`), test `tests/test_cosim_navigator.py`
- Occupancy grid from layout.json (5 cm cells; walls + obstacles), inflated by the AMR's half-diagonal + 5 cm margin.
- Localization: particle filter (AMCL-like) in the map frame using `amr/odom` deltas for motion + `amr/scan` likelihood-field model (subsample ~60 beams); start from `amr.start` with small spread. Publish `amr/pose` at 10 Hz.
- Planning: A* (8-connected) on the inflated grid from current pose to the goal station; publish `amr/path`. Replan if blocked or every 2 s.
- Control: pure-pursuit path follower → `amr/cmd_vel` (≤ max_v/max_w, slow down near goal), final in-place rotation to the goal yaw; arrival tolerance 5 cm / 3°; emergency stop if any lidar beam in the forward ±30° sector < 0.25 m. Publish `amr/status` transitions (navigating → arrived/failed with reason).
- Tests with synthetic data only (no MuJoCo needed): grid marks obstacles; A* finds a path between all station pairs and avoids racks; pure pursuit on a straight and a curved path converges in a kinematic sim; particle filter with a ray-cast-on-grid scan simulator and noisy odom tracks a 10 m trajectory with < 10 cm final error.

## cell — `cosim/cell.py` (+ `cosim/tools/teach.py`), test `tests/test_cosim_cell.py`
- "Offline programming": `cosim/tools/teach.py` uses MuJoCo + damped-least-squares IK (see `sim/policies/scripted.py`) on the world_sim arm geometry, or on `sim/assets/arm_table.xml` with the mount transform applied, to compute joint teach points and writes them into `layout.json` → `arm.teach_points`: `home`, `above_load` (tool 0.25 m above the AMR deck centre at the LOAD station, pointing down), `place_load` (tool 0.05 m above the deck).
- `cell.py`: a mission state machine driven by sim time: for each of `--cycles` (default 1): dispatch AMR to LOAD → wait `arrived` → arm motion home→above_load→place_load→(event load)→above_load→home with joint-space interpolation and trapezoidal timing (≤ 1 rad/s per joint), streaming `arm/joint_cmd` at 30 Hz sim time → dispatch to DROP → wait arrived → event unload → dispatch HOME → wait arrived. Abort on `failed` status or per-leg timeout (120 s sim). Publish `cell/status` with cycle times. Exit code 0 when all cycles finish, 1 on failure.
- Also `cosim/dashboard.py`: subscribes to everything, renders a top-down 2D view (map, racks, stations, planned path, true pose green, estimated pose blue with uncertainty, lidar points, crate state, arm joint bars, mission state) — the "AMR position synchronized back" view. `--record <path.mp4>` at 10 fps sim time (960x540) using Pillow drawing + ffmpeg; no GUI required.
- Tests: teach points reach the targets within 1 cm (IK forward check); trajectory interpolation respects joint speed limit and hits waypoints; state machine transitions with a fake bus (feed status messages, check goals/events emitted in order); dashboard renders a frame from synthetic messages.
