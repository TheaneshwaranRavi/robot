# Phase 3 Workplan — See It, Detect It, Act on It

Status: **proposed**
Prepared: 2026-11-09
Scope: CPU-only MuJoCo development on Apple Silicon, with a direct path to the real SO-101.

## 1. Goal

Build a MuJoCo cell where a camera sees parts on a table or indexed conveyor, a vision system decides whether each part is good or defective, and an SO-101 arm sorts it into a GOOD or REJECT bin. The same controller, detector contract, safety boundary, and recording format must work with simulated and real backends. This is also the rehearsal system for the Intel Physical AI Challenge in Zurich, 4–6 December 2026.

### 1.1 Functional requirements

| ID | Requirement |
| --- | --- |
| F1 | Simulate an SO-101 with five arm joints and one gripper joint, a fixed camera, parts, a work surface, and GOOD and REJECT bins. |
| F2 | Generate nominal parts and labelled defects: missing hole, crack or scratch texture, wrong colour, and deformed size. |
| F3 | Randomise lighting, part pose, camera pose, materials, background, exposure, blur, and sensor noise from a recorded seed. |
| F4 | Export RGB images and ground truth for binary quality, defect type, defect mask or region, pose, and randomisation settings. |
| F5 | Provide one camera interface with a MuJoCo backend now and a real-camera backend later. |
| F6 | Provide a classical-CV detector and an Anomalib detector behind one inspection interface. |
| F7 | Train PatchCore or PaDiM on good images only, export it to OpenVINO IR, and run CPU inference. |
| F8 | Localise the part separately from anomaly scoring and transform its pick point into the robot base frame. |
| F9 | Run a cell-controller state machine that captures, inspects, decides, picks, places, verifies, and recovers. |
| F10 | Provide one robot interface with a MuJoCo backend now and a LeRobot SO-101 backend later. |
| F11 | Route every command through clamps, freshness checks, watchdogs, and sim/hardware interlocks. |
| F12 | Record synchronized video, decisions, anomaly outputs, robot state, safety events, metrics, and summaries; show them in a headless dashboard and support replay. |
| F13 | Optionally let a RAX-style coding agent write, test, rewrite, and evaluate sorting-skill code in simulation only. |

### 1.2 Non-functional requirements

| Area | Target |
| --- | --- |
| Hardware | Apple Silicon MacBook, CPU only. Docker images remain `linux/arm64`. No CUDA dependency. |
| Runtime | Python 3.12 and MuJoCo 4.0.0 for this repository. |
| Transfer | Perception inputs and robot commands use backend-neutral contracts. Hardware-specific mapping stays in adapters. |
| Reproducibility | A seed plus a versioned config reproduces geometry, defects, randomisation, and scenario order. |
| Safety | Invalid, stale, non-finite, late, or out-of-bounds commands cause a hold or latched stop. Generated code cannot bypass safety. |
| Measurement | Report measured results. Accuracy and latency targets are set after the first baselines; no benchmark is assumed here. |
| Operability | One launcher starts a demo, records artifacts, and exits with a useful status code. A replay mode works without hardware. |
| Footprint | Do not commit large generated datasets or model binaries. Keep manifests, checksums, licences, and small fixtures. |

### 1.3 Constraints and assumptions

- Use MuJoCo Menagerie `robotstudio_so101/so101.xml` and `scene.xml`. They are Apache-2.0 and require MuJoCo 3.1.3 or newer.
- That model derives from TheRobotStudio `SO-ARM100/Simulation/SO101/so101_new_calib.xml`. A camera variant also exists as `so101_new_calib_camera.xml`.
- The six hinge joints are `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and `gripper`. The first five move the arm.
- The model uses STL meshes and position actuators with `kp` near 998. That gain is a simulation property, not a hardware command value.
- The older Menagerie `trs_so_arm100` model is SO-100. It must not be substituted for SO-101.
- Anomalib 2.7.0 installs with `pip install anomalib`; `anomalib[openvino]` adds OpenVINO support. PatchCore and PaDiM can train on good images only and export to IR.
- Anomalib training on macOS arm64 is **TO VERIFY** with a milestone-1 smoke test.
- OpenVINO 2026.4.1 has macOS arm64 CPython 3.12 wheels. OpenVINO and Anomalib availability inside the Linux arm64 container is **TO VERIFY**.
- The challenge target includes an SO-101, cameras, an Intel Core Ultra laptop, and OpenVINO. Intel Physical AI Studio 0.1.0 preview is Apache-2.0, integrates LeRobot, and lists SO-101.
- Intel Robotics AI Suite targets x86 Linux with ROS 2 Humble. It is not required for the MuJoCo vertical slice.
- The exact LeRobot SO-101 API, control rate, calibration format, and event-supplied camera are **TO VERIFY**.
- Start with a stationary table or index-and-stop conveyor. Continuous conveyor tracking is deferred.

## 2. High-level design

```
            seeded scene and part generator ── labels + manifest ──▶ dataset
                         │ RGB
                         ▼
                  CameraInterface
             ┌───────────┴───────────┐
             │ MuJoCoCamera          │ RealCamera (later; TO VERIFY)
             └───────────┬───────────┘
                         ▼
             localisation + shared preprocessing
             ┌───────────┴───────────┐
             │ Classical CV          │ Anomalib → OpenVINO CPU
             └───────────┬───────────┘
                         ▼
             cell controller state machine ◀── optional RAX candidate (sim only)
                         │
                         ▼
                     SortSkill
                         ▼
         SafetySupervisor: validate → clamp → watchdog → interlock
                         ▼
                  RobotInterface
             ┌───────────┴───────────┐
             │ MuJoCo SO-101         │ LeRobot SO-101 (later)
             └───────────┬───────────┘
                         ▼
                  GOOD / REJECT bin

 all events ───────────────────────────────▶ recorder + dashboard + evaluator
```

### 2.1 Components

The exact Phase 3 file layout is **TO VERIFY** until implementation. The paths below are proposed names, not claims about current files.

| Proposed module | Responsibility |
| --- | --- |
| `phase3/scene.py`, `phase3/parts.py` | Load the cell, own sim time and rendering, build parts, and return ground-truth defect metadata. |
| `phase3/dataset.py` | Run seeded scenes, stream images to disk, and write split manifests without buffering a whole run. |
| `phase3/camera.py` | Define `CameraInterface`; implement MuJoCo capture and later adapt the real camera. |
| `phase3/localise.py` | Find the part ROI and planar pose. Keep privileged MuJoCo truth out of scored runs. |
| `phase3/detectors.py` | Define `Detector`; implement classical CV and Anomalib/OpenVINO backends. |
| `phase3/robot.py` | Define `RobotInterface`; implement MuJoCo and later LeRobot SO-101 adapters. |
| `phase3/sort_skill.py` | Generate conservative approach, grasp, lift, place, retreat, and home commands. |
| `phase3/controller.py` | Pure state machine in the style of `cosim/cell.py`. Use sim time for simulated transitions. |
| `phase3/safety.py` | Extend the ideas in `sim/safety.py` for SO-101 commands, frames, and hardware arming. |
| `phase3/recorder.py` | Stream videos and JSONL events. Write summaries, provenance, and evaluation-ready logs. |
| `phase3/dashboard.py` | Show the camera, heatmap, score, decision, state, joints, latency, throughput, and confusion matrix. |
| `phase3/rax/` | Optional sandbox, tools, held-out layouts, and evaluator for generated skill code. |

Reuse the small interfaces in `sim/policies/base.py`, the safety primitives in `sim/safety.py`, the recording approach in `sim/recorder.py`, and the pure mission logic and dashboard patterns in `cosim/`.

### 2.2 Interfaces (contracts)

| Contract | Minimum fields and behaviour |
| --- | --- |
| `CameraFrame` | `frame_id`, capture timestamp, RGB `uint8`, camera ID, calibration ID, and freshness status. |
| `CameraInterface` | `open`, `capture`, and `close`. Backend settings are reported in run metadata. |
| `PartObservation` | ROI or mask, image centroid, planar angle, pick pixel, localisation confidence, and source frame ID. |
| `InspectionResult` | anomaly score, threshold, `GOOD|DEFECTIVE|UNCERTAIN`, heatmap reference, backend, model ID, and latency. |
| `RobotState` | five named arm positions, gripper state, timestamp, motion state, fault, and backend ID. |
| `RobotCommand` | five named position targets, normalized gripper target, command ID, deadline, and mode. |
| `RobotInterface` | `connect`, `read_state`, `command`, `hold`, `home`, and `close`. Exact LeRobot calls are **TO VERIFY**. |
| `SafetyResult` | accepted command or stop reason, applied clamps, interlock state, and source command ID. |
| `CellEvent` | schema version, sim and wall timestamps, cycle and part IDs, topic, payload, and producer. |

Frames are named `camera`, `table`, `robot_base`, and `part`. Units are metres, radians, and seconds. A versioned calibration artifact defines transforms. The real calibration target and library are **TO VERIFY**.

### 2.3 Config and safety changes (extend, never loosen)

- Add a separate Phase 3 config. Do not reuse the existing six-arm-joint embodiment dimensions. Validate all six SO-101 hinge names against the MJCF.
- Keep `target: sim|hardware` and reject a sim config in the hardware backend.
- Require explicit hardware arming after backend, calibration, joint direction, limits, and stop input pass startup checks.
- Clamp joint position, per-step change, speed, gripper command, tracking error, and workspace. Values come from reviewed config.
- Mark hardware limits, signs, zero offsets, safe poses, and timing budgets **TO VERIFY** until calibrated.
- Reject stale frames and states. Reject non-finite values. Hold on communication loss. Latch repeated deadline failures.
- Put the safety supervisor after every controller, including scripted, learned, manual, and RAX-generated skills.
- Keep a physical emergency stop independent of software. Exact event hardware and wiring are **TO VERIFY**.

### 2.4 Packaging and artifacts

- Proposed files `requirements-phase3.txt` and `Dockerfile.phase3` isolate vision dependencies from the light client image.
- Pin Python packages and record the OS, architecture, CPU, package versions, model hash, dataset hash, and config hash per run.
- Keep generated datasets and OpenVINO IR outside Git. Commit only tiny fixtures and manifests. IR filenames and the Anomalib export command are **TO VERIFY**.
- Do not add ROS 2 to the MuJoCo path. Add a bridge only if the event integration proves it necessary.

## 3. Deep dive

### 3.1 Synthetic data and defect labels

Use one parametric, graspable part family first. Its exact challenge geometry and tolerances are **TO VERIFY**. Build a nominal part with required hole features. A missing-hole defect fills a required hole. A crack or scratch is a generated texture and mask. Wrong colour changes only the controlled material range. Deformed size changes one or more dimensions while preserving a valid MuJoCo body.

Each sample stores seed, split, binary label, defect type, severity parameters, part pose, bounding box, mask, camera calibration, randomisation values, and generator version. Good images have `defect_type: none`. Defect type is evaluation ground truth; the deployed decision remains good versus defective. Randomise light, camera pose, part pose, surfaces, background, exposure, blur, and noise. Do not encode labels in filenames, backgrounds, or scenario order.

Split by disjoint seeds and randomisation groups before generation. Train uses good images only. Validation contains good and defective images for threshold selection. The held-out test set is never used to choose preprocessing, model, or threshold. Add a domain-shift set with unseen but plausible settings.

### 3.2 Perception pipeline

One preprocessor supplies the same colour order, crop, resize, and normalization to both backends. Save its config with the model. Localisation finds the part and pick pose before anomaly scoring. The classical baseline uses good references, colour statistics, contour dimensions, hole evidence, and texture residuals. Its thresholds come from validation data.

Start with PatchCore. Compare PaDiM if export, memory, or CPU latency is unsuitable. Train only on good images. Export to OpenVINO IR and compare native and IR scores on fixed fixtures. Set export tolerance and the decision threshold from validation evidence. Default `UNCERTAIN` to REJECT. Never convert inference errors into GOOD.

### 3.3 Camera-to-robot transform and grasp

Use a fixed overhead or oblique camera. For a planar surface, map the pick pixel to the table plane with calibrated intrinsics and extrinsics. Simulation may derive the transform from MuJoCo, but the controller receives the same calibration artifact as hardware. Use a top-down grasp, fixed bin poses, and conservative clearance. Use SO-101-aware IK or a calibrated joint-space skill; do not copy the existing six-DoF IK unchanged.

### 3.4 Cell state machine and failure policy

```
BOOT → HOME → WAIT_PART → CAPTURE → LOCALISE → INSPECT → DECIDE
                                                        │
                       ┌──────── GOOD ──────────────────┤
                       │                                └── DEFECTIVE/UNCERTAIN
                       ▼                                             ▼
                  PICK → VERIFY_GRASP → PLACE_GOOD or PLACE_REJECT
                                            │
                                            ▼
                                      VERIFY_DROP → RECORD → WAIT_PART

Any state → RECOVER for one bounded retry → SAFE_STOP on repeated or unsafe failure
```

The controller follows `cosim/cell.py`: pure state, explicit inputs, emitted commands, status on change, and timeouts based on sim time in simulation. Every command has an ID and result. No-part waits. Stale image, bad calibration, malformed inference, unreachable pick, collision risk, dropped part, and missed deadline have explicit transitions and recorded reasons.

### 3.5 Sim-to-real boundary

`MuJoCoRobot` and `LeRobotRobot` implement the same contract. Names, units, ordering, gripper normalization, limits, signs, zeros, and timestamps are mapped in the adapter. `MuJoCoCamera` and `RealCamera` return the same `CameraFrame`. A replay camera supports regression tests and an offline demo.

### 3.6 Optional RAX-style agent track

RAX (Robot Agent eXploration, arXiv 2610.22496) uses a frozen coding agent such as Codex CLI or Claude Code. It calls robot tools through MCP or Bash, then writes, tests, rewrites, and compiles `controller.py:main(ctx)` before held-out evaluation. The Apache-2.0 RAX repository has MuJoCo paths through DexJoCo and LIBERO; an SO-101 backend would be our work.

Keep this track optional and simulation-only. Give it a narrow `ctx` API, read-only observations, bounded time and files, and no direct actuator handle. Run generated code in a subprocess. Safety remains out of process or otherwise impossible for the candidate to replace. Evaluate on unseen layouts and seeds. Promotion to real hardware is out of scope for the hackathon.

### 3.7 Observability and metrics

Reuse the Phase 1 episode video plus `summary.json` idea and the Phase 2 latest-message dashboard. Stream events as JSONL so long runs do not remain in memory. A side-by-side demo shows camera and heatmap beside robot video, decision, controller state, and metrics.

| Metric | Definition |
| --- | --- |
| Precision | Defective true positives divided by all defective predictions. Report per defect and aggregate. |
| Recall | Detected defective parts divided by all defective parts. Keep missed defects visible. |
| F1 | Harmonic mean of precision and recall at the chosen threshold. |
| AUROC | Threshold-independent ranking of good versus defective anomaly scores on the held-out set. |
| Sort accuracy | Correct-bin outcomes divided by parts presented. A missed or failed placement is incorrect; also report failure causes. |
| Picks per minute | Successful placements divided by active cell runtime in minutes. State whether setup and recovery time are included. |
| CPU inference latency | OpenVINO call time after separate warm-up. Report distribution and at least median and tail latency with hardware metadata. |

No numeric target is declared before baseline measurement. Milestone 3 records baselines. The team then sets demo gates and records their rationale.

## 4. Testing strategy

| Layer | Test |
| --- | --- |
| Schema and unit | Reject bad fields, joints, timestamps, shapes, finite checks, or calibration IDs; verify valid defect geometry, labels, masks, and seeded output. |
| Unit | Classical and OpenVINO detectors obey the same contract and never return GOOD on an inference error. |
| Unit | Safety clamps commands, enforces tracking and workspace bounds, catches timeouts, and refuses the wrong target. |
| Unit | State-machine tables cover every state, timeout, recovery, and terminal fault with a fake clock and fake backends. |
| Model | Native Anomalib and exported IR scores are compared on fixed fixtures within a recorded project tolerance. |
| Contract | MuJoCo, replay, fake hardware, and later LeRobot backends pass the same robot and camera suites. |
| Integration | Seeded good and each defect type flow from capture through the expected bin with privileged truth disabled. |
| Fault injection | Exercise stale frames, camera loss, inference failure, unreachable parts, command loss, joint lag, drop, and watchdog expiry. |
| Regression | Replay immutable event logs and images; compare decisions, transitions, artifacts, and metrics. |
| Soak | Run repeated mixed parts and confirm clean shutdown, bounded artifact growth, no unsafe command, and reproducible summaries. |
| Hardware later | Low-speed dry run, home, one pick, one GOOD place, one REJECT place, stop input, and communication-loss hold. |

The existing simulation and co-simulation tests must keep passing unchanged.

## 5. Phased work plan

Each milestone ends with a runnable artifact, a recorded check, and green relevant tests. Effort is relative, not a schedule.

| Milestone | Deliverable | Acceptance criteria | Effort |
| --- | --- | --- | --- |
| **0. Contracts** | Proposed package skeleton, schemas, configs, backend fakes, licence manifest, and test fixtures | Interfaces reject bad data; fake camera and robot complete one state-machine cycle | S |
| **1. Platform spike** | Menagerie SO-101 scene, camera render, dependency image or native setup, Anomalib/OpenVINO smoke | Exact joint mapping is asserted; physics stays finite; one good-image train/export/infer path runs on macOS arm64 or its failure and fallback are recorded; Docker arm64 status is resolved | M |
| **2. Synthetic data** | Part generator, four defect families, randomisation, manifest, deterministic splits, and dataset CLI | Every label matches rendered ground truth; same seed reproduces samples; train has only good images; held-out splits are isolated | M |
| **3. Perception** | Localiser, classical baseline, PatchCore and optional PaDiM experiment, OpenVINO export, evaluator | Both backends run the same held-out evaluator; IR parity is recorded; required metrics and CPU latency are emitted without invented targets | M–L |
| **4. Safe sorting** | SO-101 sort skill, state machine, MuJoCo robot backend, safety supervisor, and fault recovery | Reference good and defect scenarios reach the correct bins; injected faults hold or stop with the expected reason | L |
| **5. Closed-loop demo** | Launcher, streaming recorder, dashboard, replay, mixed-part run, and summary report | One command runs a seeded session; video, events, metrics, provenance, and exit status agree | M |
| **6. Hardware rehearsal** | LeRobot and real-camera adapter skeletons, calibration workflow, fake-device tests, and bring-up runbook | Contract tests pass without hardware; all unknown hardware mappings remain blocked by explicit **TO VERIFY** checks | M |
| **7. Hackathon pack** | Cached dependencies, known-good artifacts, calibration assets, demo modes, backup recording, and operator checklist | A clean machine can run sim and replay offline; the team rehearses live, sim, and recorded fallback paths | S–M |
| **8. Optional RAX** | Restricted tools, SO-101 sim context, candidate runner, held-out evaluator, and safety tests | Generated code cannot reach actuators except through safety; held-out results are recorded; no hardware deployment exists | L |

Critical path: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7. Milestone 8 starts only after milestone 4 makes the safety boundary testable.

## 6. 40-hour hackathon playbook

### 6.1 Bring pre-built

- A known-good SO-101 MuJoCo scene, backend contract tests, scripted pick/place, and safety supervisor.
- A seeded generator, small legal-to-carry dataset, classical baseline, Anomalib candidate and IR, plus exact preprocessing and threshold selection.
- Evaluation, replay, recorder, and dashboard tools with known-good fixtures.
- Pinned dependency files, cached wheels or images, model and dataset checksums, and offline documentation.
- Camera and robot calibration flows, a printed calibration target, fixed mounts, cables, powered hub, tools, and spares.
- A fake hardware backend, recorded camera input, a simulated demo, and a final backup video.
- A one-page operator checklist for arm, home, workspace, stop, reset, and safe shutdown.
- The event-provided object dimensions, camera, laptop OS, network, power, and allowed pre-built assets are **TO VERIFY** against final rules.
- The required Physical AI Studio integration point is **TO VERIFY**. Keep the OpenVINO sorting loop independent of the preview framework.

### 6.2 Work blocks

| Hours | Outcome |
| --- | --- |
| 0–4 | Inventory hardware and rules. Run the offline golden path. Test the stop. Freeze the smallest demo and assign owners. |
| 4–10 | Bring up robot and camera adapters. Verify signs, limits, home, gripper, workspace, and calibration. Make one low-speed scripted pick. |
| 10–16 | Capture real good and defect validation images. Match preprocessing. Run classical and IR models. Select model and threshold from evidence. |
| 16–24 | Integrate capture, inspect, decide, pick, and both bins. Handle no part, uncertain result, stale frame, miss, and drop. |
| 24–30 | Measure metrics. Tune lighting, waypoints, calibration, and threshold. Inject faults. Run repeated cycles and preserve the known-good artifacts. |
| 30–36 | Finish dashboard and story. Rehearse live, sim, and replay modes. Record backup video. Freeze feature work. |
| 36–40 | Run the final soak and reset procedure. Repair only demo blockers. Polish the pitch and keep contingency time. |

### 6.3 Demo script

1. Show the workspace check, target interlock, and stop input.
2. Place a good part. Show the frame, score, threshold, GOOD decision, and GOOD-bin motion.
3. Place a visibly defective part. Show the heatmap or evidence, REJECT decision, and REJECT-bin motion.
4. Show one different defect type to prove the decision is not tied to colour alone.
5. Point to sort accuracy, picks per minute, and CPU inference latency on the dashboard.
6. Trigger or replay one stale-frame or uncertain case. Show the safe outcome and recorded reason.
7. Explain the backend swap. If hardware fails, fall back in order: live simulation, deterministic replay, then recorded video.

### 6.4 Team split

| Role | Three-person team | Four-person team |
| --- | --- | --- |
| Perception and data | Generator, capture, Anomalib, OpenVINO, thresholds, metrics | Same |
| Robotics and safety | SO-101, calibration, grasp, bins, limits, stop, recovery | Same |
| Integration and demo | Controller, interfaces, recorder, dashboard, pitch, fallback | Split into platform/integration and demo/product |
| Safety operator | One named person for every live run; all members know stop and reset | Same |

## 7. Trade-offs

| Decision | Chosen | Cost or alternative |
| --- | --- | --- |
| Robot model | Menagerie SO-101 | Mesh asset and licence tracking are required; the current primitive arm is simpler but wrong. |
| Camera | Fixed overhead or oblique | Easier calibration than a wrist camera, but occlusion may be worse. |
| Material flow | Stationary or index-and-stop | Lower throughput than continuous tracking, but safer and easier to transfer. |
| Detector | PatchCore first, PaDiM comparison | Selection follows measured export, memory, accuracy, and latency. |
| Baseline | Classical CV beside anomaly detection | Extra code, but it is explainable and protects the demo from model failure. |
| Motion | Scripted top-down grasp | Less general than learned manipulation, but testable and safe for this task. |
| Topology | In-process vertical slice first | Phase 2 ZeroMQ separation is added only where isolation or replay earns the complexity. |
| Uncertainty | Route to REJECT | Raises false rejects, but does not pass an uncertain defect as good. |
| RAX | Optional and simulation-only | Gives less spectacle on hardware, but preserves the safety and time budget. |

## 8. Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Synthetic-to-real gap | High | Randomise relevant factors, add real validation images, keep preprocessing identical, and test held-out shifts. |
| Tiny texture defects disappear at camera resolution | Medium | Validate visible defect scale on event hardware; exact camera and minimum defect size are **TO VERIFY**. |
| Good-only model flags harmless variation | Medium | Calibrate on varied good images, report false rejects, compare the baseline, and keep an uncertain policy. |
| OpenVINO export differs from native inference | Medium | Prove export in milestone 1 and run parity fixtures before integration. |
| macOS arm64 Anomalib training fails | Medium | Time-box the smoke test; use a **TO VERIFY** approved CPU training host or ship a pre-exported model. |
| Camera calibration moves | Medium | Rigid mount, versioned calibration, startup check, and quick recalibration flow. |
| Sim and real joint mappings differ | High | Named joints, adapter tests, low-speed calibration, explicit signs and zeros, and no shared unchecked arrays. |
| Pick, drop, or jam failure | Medium | Conservative waypoints, grasp/drop verification, one bounded recovery, then safe stop. |
| Event API or hardware changes | Medium | Keep adapters narrow, cache offline fallbacks, and confirm final rules and inventory. |
| RAX code is unsafe or overfits | High | Sim only, restricted tools, immutable safety layer, resource limits, and held-out evaluation. |

## 9. What to revisit as this grows

- Continuous conveyor tracking and motion compensation.
- Multiple part families, multi-label defects, severity grading, depth, or multi-camera localisation.
- Wrist camera and active inspection for hidden surfaces.
- Real good-image adaptation, synthetic-to-real training, and automated drift checks.
- LeRobot dataset recording and imitation learning through Physical AI Studio.
- Phase 2 multi-process messaging for isolation or remote execution; ROS 2 and Intel Robotics AI Suite only for an x86 deployment that needs them.
- Hardware promotion rules for agent-written skills. RAX output remains simulation-only until a separate safety case exists.

## 10. Open questions and defaults taken

| Question | Default |
| --- | --- |
| Which part? | One parametric plate-like, top-graspable family. Exact challenge geometry is **TO VERIFY**. |
| Which camera? | One fixed overhead or oblique RGB camera. Device, lens, resolution, and SDK are **TO VERIFY**. |
| Table or conveyor? | Stationary table or indexed stop for the first vertical slice. |
| Which learned model? | PatchCore first. Compare PaDiM when the baseline pipeline works. |
| How is the threshold chosen? | Validation data and an explicit error-cost policy. Never the held-out test set. |
| What happens when uncertain? | REJECT in the two-bin demo. A real production policy would need owner approval. |
| How is the part localised? | Image-derived planar pose. MuJoCo truth is for labels and evaluation only. |
| How does the arm move? | Named joint-position commands, scripted top-down grasp, fixed bin poses, and bounded recovery. |
| Which real backend? | LeRobot SO-101 adapter. Exact calls, rate, and calibration schema are **TO VERIFY**. |
| Where does training run? | Apple Silicon CPU if milestone 1 passes. The fallback environment is **TO VERIFY**. |
| Must Physical AI Studio be in the live path? | No by default. Its event requirement and integration point are **TO VERIFY**. |
| What are the success gates? | Set and record them after classical and OpenVINO baselines. Do not invent numbers. |

## 11. Progress log

### 2026-11-09: milestones 0 and 1

Milestone 0 (contracts):
- **Contracts:** `phase3/contracts.py` holds typed, validated dataclasses for camera frames, part observations, inspection results, robot state and commands, safety results and cell events.
- **Interfaces and fakes:** `phase3/interfaces.py` defines the camera, detector and robot interfaces. `phase3/fakes.py` provides deterministic test versions of each.
- **Controller:** `phase3/controller.py` is a pure state machine covering every state in section 3.4.
- **Config:** `phase3/config/phase3.sim.json` and `phase3/config.py` give strict validation.
- **Tests:** 37 tests.

Milestone 1 (platform spike):
- **SO-101 model:** `python -m phase3.tools.fetch_so101` fetches Menagerie `robotstudio_so101` at a pinned commit into a gitignored folder, with a SHA-256 manifest.
- **Cell:** `phase3/scene.py` builds the cell with MjSpec: table, inspection pad, GOOD and REJECT bins, a free part, an overhead camera and a cell camera. `tests/test_phase3_scene.py` checks the exact joint and actuator mapping, finite physics, the render, and reach.
- **Home pose:** a folded "camera clear" pose was found by grid search. No arm link reaches past x = 0.07 m and there are no self-contacts.
- **Tool-down reach (finding):** with the fingers vertical, the arm reaches about 0.09 m above the pad and about 0.07 m above the bins. Approach and transit above those heights must not require a vertical tool. `phase3/kinematics.py` gives damped-least-squares IK with a tool-down constraint.
- **Anomalib + OpenVINO on macOS arm64 (resolves the milestone-1 TO VERIFY):** `.venv-vision/bin/python -m phase3.tools.anomalib_smoke` ran the whole chain: render, train PatchCore (ResNet-18, CPU), export to OpenVINO, CPU inference.
  - Timings: training on 40 good crops took about 40 s, and the export took about 7 s.
  - This is a tiny, easy smoke set, so its scores are not a result.
- **Export quirk (finding):** Anomalib 2.7's export fixes the IR batch at 2 (`[2,3,?,?]`), and reshaping it to batch 1 fails in an Interpolate node. The smoke test therefore pads one image to batch 2. This roughly doubles the inference cost; measured CPU latency was about 250 ms median, about 810 ms p95. Milestone 3 must fix the export (a batch-1 trace or a direct `openvino.convert_model`) before latency is reported.
- **Dependencies:** `requirements-phase3.txt` pins the vision stack. numpy is left to the resolver because Anomalib selects 2.4.x, while the sim environment pins 2.5.4.

Still open:
- The vision stack inside the Linux arm64 Docker image is not tested yet.
- A real-camera or hardware backend is not started (milestone 6).
