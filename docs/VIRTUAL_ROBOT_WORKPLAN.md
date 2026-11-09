# Virtual Robot Workplan

Status: **implemented — see README 'Virtual robot'**
Prepared: 2026-11-09
Branch: `TheaneshwaranRavi/virtual-robot-sim` (Orca workspace `virtual-robot-sim`)

## 1. Goal

Give this repository a simulated robot that runs on an Apple Silicon Mac with no NVIDIA GPU and no hardware. That makes the embodiment, the control loop and the safety checks real and testable now. Today they exist only as placeholders. The same loop must later drive a real GR00T N1.7 `PolicyServer` on a Linux GPU host without code changes.

### 1.1 Functional requirements

| ID | Requirement |
| --- | --- |
| F1 | A tabletop scene: 6-DoF arm, parallel-jaw gripper, table, one graspable cube, one fixed camera. |
| F2 | A non-placeholder embodiment config (`config/embodiment.sim.json`) that describes that scene's actual observation and action spaces and passes `examples/embodiment_config.py`. |
| F3 | A closed-loop runner, `python -m sim.run`, that runs observe → policy → clamp → apply → step physics → record. |
| F4 | Two policy backends behind one interface: `scripted` (default, offline pick-and-place) and `groot` (remote `PolicyServer`, opt-in by flag). |
| F5 | Writes a video of each episode plus a JSON episode summary to `outputs/` (gitignored). |
| F6 | Runs headless in Docker (`docker compose run --rm sim`) and natively on macOS. |

### 1.2 Non-functional requirements

| Area | Target |
| --- | --- |
| Hardware | M-series MacBook Air, CPU only. Docker images are `linux/arm64`; nothing may need CUDA. |
| Speed | One 10 s episode (300 control steps at 30 Hz, 224×224 camera) renders in under 60 s on CPU. Tests finish in under 2 min. |
| Determinism | The same seed gives the same trajectory and the same success result. The tests depend on this. |
| Safety | Every action is clamped to config limits. Non-finite values or missed deadlines stop the loop. A placeholder config can never reach the `groot` backend. |
| Footprint | No large binary assets, and no asset with an unclear licence, committed. The client image stays as small as it is today. |

### 1.3 Constraints and assumptions

- Python 3.12 and the existing Docker-first workflow. No host virtualenv is required, although native macOS runs are supported for viewing.
- MuJoCo (Apache-2.0) is the physics engine. Isaac Sim and Isaac Lab cannot run on this Mac (see `docs/NVIDIA_PLAN.md`), so this simulator is a stand-in for local development, not a replacement for Isaac Lab training.
- Resolved: the GR00T N1.7 `PolicyServer` `get_action` request and response format was checked against NVIDIA/Isaac-GR00T commit `7aa0bca` and is written down in [`docs/GROOT_PROTOCOL.md`](GROOT_PROTOCOL.md). The `groot` backend is still tested only against an in-process fake server, not a live GR00T server.

## 2. High-level design

```
                 config/embodiment.sim.json
                            │  (validated once at start-up)
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ sim.run  (control loop @ control_hz)                                 │
│                                                                      │
│   ┌──────────┐ obs dict  ┌──────────────┐ action chunk ┌───────────┐ │
│   │ SimEnv   │──────────▶│ Policy       │─────────────▶│ Safety    │ │
│   │ (MuJoCo) │           │  scripted ▸  │  (H×7)       │ clamp +   │ │
│   │          │◀──────────│  groot    ▸──┼──ZeroMQ──▶ PolicyServer │ │
│   └────┬─────┘  clamped  └──────────────┘  (remote,    │ watchdog  │ │
│        │        action                     opt-in)     └───────────┘ │
│        ▼                                                             │
│   ┌──────────┐                                                       │
│   │ Recorder │──▶ outputs/<run-id>/episode.mp4|gif, summary.json     │
│   └──────────┘                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Components

| Module | Responsibility |
| --- | --- |
| `sim/assets/arm_table.xml` | Hand-written MJCF: 6 revolute joints with capsule/box geometry, position actuators, 2 slide-joint fingers, table, cube and `front_camera`. No meshes. |
| `sim/env.py` — `SimEnv` | Loads the model, `reset(seed)`, `observe() -> dict`, `apply(action)`, `step()`. It owns physics substeps (timestep 2 ms, so about 17 substeps per 30 Hz control step) and offscreen rendering. |
| `sim/policies/base.py` — `Policy` | `reset(task: str)` and `act(obs) -> np.ndarray[H, 7]`. One interface for both backends. |
| `sim/policies/scripted.py` | A waypoint state machine (pre-grasp → descend → close → lift → hold). Joint deltas come from damped-least-squares IK on MuJoCo's site Jacobian, so it emits the same `arm_joint_delta` actions a learned policy would. |
| `sim/policies/groot.py` | Packs observations into the GR00T request, sends it over its own small ZeroMQ REQ client (`examples/groot_ping.py` is unchanged), validates the response's shape and dtype. |
| `sim/safety.py` | `clamp(action, limits)`, a `Watchdog` (finite check plus deadline), joint-position soft limits. Has no MuJoCo dependency, so it is unit-testable in the light client image too. |
| `sim/recorder.py` | Frames → mp4 via ffmpeg when available, otherwise GIF via Pillow. Writes `summary.json` (seed, steps, success, stop reason, timing). |
| `sim/run.py` | CLI: `--policy scripted|groot`, `--config`, `--seed`, `--steps`, `--realtime`, `--out`, `--no-record`. Exit codes: 0 = success, 1 = task failed, 2 = config error, 3 = safety stop. |

### 2.2 Interfaces (contracts)

Observation dict, keyed by config names:

| Key | Shape / dtype | Source |
| --- | --- | --- |
| `front_camera` | `(224, 224, 3) uint8` | offscreen render of `front_camera` |
| `arm_joint_positions` | `(6,) float32`, rad | `qpos` of the 6 arm joints |
| `gripper_position` | `(1,) float32`, 0 = open, 1 = closed | finger slide `qpos`, normalised |
| `task_description` | `str` | CLI, default `"pick up the red cube"` |

Action chunk: `(action_horizon=16, 7) float32` = 6 × `arm_joint_delta` (±0.05 rad per step) followed by 1 × `gripper_command` (0..1). The runner executes the first `execute_k` steps of each chunk (default 8, receding horizon), then asks the policy again.

Mapping to GR00T: the `groot` backend adapts these keys to the server's `video`, `state` and `language` groups with batch and time axes. The adapter lives in one function (`build_observation`), so a format change touches one place. The confirmed format is in [`docs/GROOT_PROTOCOL.md`](GROOT_PROTOCOL.md).

### 2.3 Config changes (extend, never loosen)

`config/embodiment.sim.json` keeps the example's shapes and limits, but sets `"placeholder": false` and adds:

- `"target": "sim"`: a new **optional** field. The validator accepts `"sim"` or `"hardware"` and rejects anything else. A missing value means `"hardware"`, so the existing config keeps its behaviour.
- `"joint_limits"`: per-joint position bounds that the safety layer enforces in addition to the per-step delta limits.

Two interlocks follow from this:

1. The `groot` backend refuses any config with `placeholder: true`.
2. Any future hardware runner refuses `target: "sim"`, so a sim config can't drive a real robot by mistake.

### 2.4 Packaging

| Item | Decision |
| --- | --- |
| Images | Keep the light client `Dockerfile` free of simulation dependencies; the only change is that it now also copies `sim/`, so the stdlib-only sim tests (config + safety) run there. Add `Dockerfile.sim` on the same `python:3.12-slim` base, plus `libosmesa6` and `ffmpeg` apt packages. |
| Dependencies | `requirements.txt` stays client-only. A new `requirements-sim.txt` pins `mujoco`, `numpy` and `Pillow`, includes `-r requirements.txt`, and uses exact versions only. |
| Rendering | `MUJOCO_GL=osmesa` in Docker (software, no GPU). Native macOS uses MuJoCo's default (CGL). |
| Compose | Adds `sim` (runner, mounts `./outputs`) and `sim-tests`. The existing `groot-client` and `tests` services are unchanged. |

## 3. Deep dive

### 3.1 Control loop and timing

```
reset(seed) → for t in range(steps):
    if chunk exhausted: chunk = clamp(policy.act(obs))      # watchdog: policy deadline
    a = chunk[i]; watchdog.check_finite(a)
    target_q = targets + a[:6]; gripper = a[6]                # deltas add to the current targets
    target_q = clamp(target_q, joint_limits, within 0.25 rad of measured q)
    env.apply(target_q, gripper); env.step(substeps)          # watchdog: step deadline
    obs = env.observe(); recorder.add(obs.front_camera)
    if success(): keep going for --hold-steps more steps, then break
```

Joint deltas are added to the current joint *targets*, not to the measured position, so a slow joint does not lose commanded motion. The new targets are then clamped to the joint limits and to within 0.25 rad of the measured position.

- **Offline mode (default):** physics runs as fast as the CPU allows. The deadline applies only to `policy.act` (default 1 s scripted, `GROOT_TIMEOUT_MS` for remote).
- **`--realtime`:** each control step must finish within `1/control_hz`. Three consecutive misses trigger a safety stop. This is what future hardware will need.
- **Success:** the cube's z is more than 5 cm above its rest height for 15 consecutive steps (0.5 s) and is between the fingers.

### 3.2 Error handling

| Failure | Behaviour |
| --- | --- |
| Invalid or placeholder config for `groot` | Exit 2 before any network I/O. |
| Policy timeout, ZeroMQ `Again`, or malformed reply | No retry inside an episode. Hold the last safe position (zero delta, gripper unchanged) and stop with exit 3, recording the reason. A silent retry would hide latency problems a real robot cannot tolerate. |
| Non-finite observation or action | Stop immediately, exit 3. |
| MuJoCo instability (warnings, NaN `qacc`) | Treated as non-finite. Stop, exit 3. |
| ffmpeg missing | Fall back to GIF and log a warning. Recording never fails the run. |

### 3.3 Observability

Each run writes `outputs/<timestamp>-<seed>/summary.json` and the video, and prints a one-line summary. There is no metrics stack. It isn't needed for a single-machine dev tool (see §6).

## 4. Testing strategy

| Layer | Test | Image |
| --- | --- | --- |
| Unit | Clamp keeps out-of-range, NaN and inf values in bounds; NaN triggers the watchdog | client (`tests`; the client image now copies `sim/` so these run there) |
| Unit | Validator accepts `embodiment.sim.json`, rejects a bad `target`, and the existing config still validates | client |
| Unit | `groot` backend refuses a placeholder config | client |
| Integration | Scripted policy, seeds 0–4: cube lifted within 300 steps, deterministic across two runs | sim (`sim-tests`) |
| Integration | `groot` backend against the in-process fake server returns a correctly shaped chunk; a fake server that sleeps triggers a timeout stop | sim |
| Smoke | `docker compose run --rm sim --steps 300` exits 0 and writes a video | sim |

Every existing test in `tests/test_portable.py` must keep passing unchanged.

## 5. Phased work plan

Each phase ends with a commit and a green test run. Effort is rough, for one agent.

| Phase | Deliverable | Acceptance check | Effort |
| --- | --- | --- | --- |
| **0. Setup** | `Dockerfile.sim`, `requirements-sim.txt`, `sim` + `sim-tests` services, `outputs/` gitignored | `docker compose build sim` succeeds on arm64; `python -c "import mujoco"` works with `MUJOCO_GL=osmesa` and renders one frame | S |
| **1. Scene** | `arm_table.xml`, `SimEnv` (reset/observe/apply/step/render) | Model loads; 1,000 zero-action steps stay stable; observation shapes match the config | M |
| **2. Config + safety** | `embodiment.sim.json`, validator `target` and `joint_limits` extension, `sim/safety.py` | Unit tests in §4 pass in the **client** image | S |
| **3. Loop + scripted policy** | `Policy` interface, IK-based scripted pick-and-place, `run.py`, recorder | Seeds 0–4 lift the cube; video written; determinism test passes | M–L |
| **4. GR00T backend** | Confirm the N1.7 `get_action` format from the pinned upstream source and record it in `docs/`; adapter + fake-server tests | Fake-server integration tests pass; placeholder refusal tested. **No live GPU run is required for this phase.** | M |
| **5. Docs + demo** | README "Virtual robot" section; one demo video; Orca workspace comment updated | A fresh clone can follow the README end to end | S |

Critical path: 0 → 1 → 3. Phase 2 can be done alongside Phase 1, and Phase 4 alongside Phase 3 once the `Policy` interface lands.

## 6. Trade-offs

| Decision | Chosen | Alternative | Why |
| --- | --- | --- | --- |
| Simulator | MuJoCo | PyBullet; Isaac Sim; Newton web demos | Native arm64 wheels, deterministic, Apache-2.0, small. PyBullet is unmaintained-ish and less accurate in contact. Isaac Sim doesn't run on a Mac. Cost: physics and rendering differ from Isaac Lab, so tuned behaviour won't transfer one-to-one. |
| Robot model | Hand-written primitive MJCF | MuJoCo Menagerie (Franka, UR5e) | No meshes, no per-model licence review, full control of joint limits. Cost: less realistic. Menagerie can be swapped in later behind `SimEnv`. |
| Action space | Joint deltas (as in the example config) | End-effector pose deltas | Matches the existing config and GR00T's state/action style. The scripted policy pays an IK cost instead. |
| Headless rendering | OSMesa (software) | EGL | EGL needs a GPU driver, which Docker on a Mac doesn't have. OSMesa is slower, which is fine at 224×224. |
| Images | Separate `Dockerfile.sim` | One image for everything | Keeps the GR00T client small and its dependency promise ("portable client deps only") intact. |
| Remote failure policy | Stop on first failure | Retry with backoff | Mirrors what real hardware needs. Retries would mask latency. |

## 7. Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| `mujoco` + OSMesa render problems on `linux/arm64` | Medium | Phase 0 proves rendering before anything else. Fallback: render on native macOS, and in Docker run tests with rendering off (`--no-record`). |
| GR00T N1.7 request format differs from the assumptions | High | Phase 4 starts by reading the pinned upstream source. The adapter is isolated in one function. |
| Scripted grasp is flaky across seeds | Medium | Keep cube placement randomisation small; tune friction and finger force; assert on 5 fixed seeds, not random ones. |
| Sim config mistaken for hardware config | Low | `target: "sim"` interlock (§2.3). |

## 8. What to revisit as this grows

- **Data collection:** record episodes in LeRobot format so scripted demos can fine-tune GR00T on the GPU host.
- **Scene variety:** several objects, randomised textures and lighting, a wrist camera. Each needs config and test updates.
- **Isaac Lab port:** reuse the same embodiment config and `Policy` interface. Only `SimEnv` changes.
- **Real hardware:** a `HardwareEnv` with the same interface, plus e-stop, calibration and controller watchdogs, which this plan deliberately leaves out.

## 9. Open questions and the defaults taken

1. **Which robot?** A generic hand-written 6-DoF arm with a parallel gripper, no Menagerie model.
2. **First task?** Pick-and-place of one cube (lift and hold).
3. **GR00T backend in this pass?** Yes. It is built and tested against an in-process fake server only; no live GR00T server has been used.
