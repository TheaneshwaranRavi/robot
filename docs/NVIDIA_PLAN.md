# NVIDIA Robotics Integration Plan

Status: **proposal — awaiting approval**  
Prepared: 2026-10-25  
Announcement reviewed: NVIDIA GTC Berlin, 2026-10-21

## 1. Repository Assessment

This repository is currently empty. It contains only Git metadata:

- Branch: `main`, with no commits.
- Remote: `https://github.com/TheaneshwaranRavi/robot.git`, with no remote refs.
- Source code, README, manifests, lockfiles, tests, documentation, robot models, and hardware configuration: none.
- Language/framework: not yet selected.
- Robot type and hardware assumptions: not documented and therefore unknown.

The NVIDIA integration is consequently a greenfield design. It cannot yet be tailored to a specific robot embodiment, controller, sensors, middleware, or safety architecture.

## 2. Official Release Research

### Isaac Lab 3.0

**Purpose.** An open-source robot-learning framework for simulation, reinforcement learning, imitation learning, training, evaluation, and policy deployment. Version 3.0 introduces a backend-neutral task API, kit-less execution with Newton, optional standalone PhysX/RTX libraries, and optional full Isaac Sim integration.

**Canonical resources.** [NVIDIA Isaac Lab 3.0 documentation](https://docs.nvidia.com/isaaclab/3.0/), [installation requirements](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation), and the canonical NVIDIA-maintained repository at [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab). The repository is under NVIDIA's `isaac-sim` GitHub organization rather than `github.com/NVIDIA`.

**Requirements.** Python 3.12; Ubuntu 22.04+ x86_64, Linux aarch64, or Windows 11 x86_64; NVIDIA GPU for native training; default PyTorch 2.11/CUDA 13.0 stack; Turing or newer GPU (compute capability 7.5+); Linux driver 580.65.06+ or Windows driver 580.88+. Full Isaac Sim 6.2 use is substantially heavier and NVIDIA recommends at least 32 GB RAM, 16 GB VRAM, and a supported RTX GPU for x86_64 workstations. A separate CUDA Toolkit is not required for the default wheel-based Newton/Warp installation.

**License.** BSD-3-Clause for Isaac Lab source. Optional Isaac Sim packages and containers also carry NVIDIA/Omniverse license terms and an EULA.

**MacBook Air.** Native Isaac Lab training is unsupported on macOS and Apple GPUs. The browser demo gallery can run simplified Newton simulations and pretrained policy playback locally through CPU/WebAssembly. Source editing, configuration authoring, result inspection, and remote orchestration can also be done locally. Real training, CUDA simulation, RTX rendering, and Isaac Sim require a supported Windows/Linux NVIDIA system; Linux is the recommended target.

### Isaac OS

**Purpose.** NVIDIA describes Isaac OS as an open-source runtime framework built on Holoscan that connects AI, sensors, controls, and robot hardware for deterministic real-time execution from simulation to physical robots.

**Official publication status.** The [October 21 announcement](https://nvidianews.nvidia.com/news/nvidia-advances-isaac-platform-for-general-purpose-robotics-with-new-reference-humanoid-frameworks-simulation-pipelines-and-data) is currently the only authoritative Isaac OS product description located. As of 2026-10-25, no public Isaac OS GitHub repository, dedicated installation guide, release artifact, version matrix, or license file could be found in NVIDIA's public GitHub organizations or NVIDIA documentation. NVIDIA's [Isaac ROS FAQ](https://developer.nvidia.com/isaac/ros) also describes Isaac OS as a commercially licensed Holoscan-based runtime, which conflicts with the newsroom's “open source runtime framework” wording. This must be resolved before adopting or redistributing it.

**Known upstream requirements, not Isaac OS requirements.** Holoscan 5.x supports Linux with Python 3.12+ for its Python API. Current packaged Holoscan installation paths target Ubuntu 22.04/24.04, CUDA 12 or 13, x86_64 NVIDIA dGPU systems, and NVIDIA aarch64 platforms such as Jetson/IGX/DGX Spark. These are useful planning bounds but must not be treated as a published Isaac OS compatibility matrix.

**License.** Unknown for Isaac OS until NVIDIA publishes the repository or product license. Holoscan SDK is Apache-2.0, but that does not establish the license of Isaac OS as a whole.

**MacBook Air.** Isaac OS cannot currently be installed from an official public release. Holoscan's supported runtime is Linux/NVIDIA hardware, so native macOS execution should not be planned. Keep any future Isaac OS process behind a network boundary and run it on the robot's Jetson/IGX computer or another Linux NVIDIA edge host.

### Isaac GR00T N1.7

**Purpose.** An open vision-language-action foundation model for generalized robot manipulation. It consumes language, camera images, and robot state and produces action chunks. The repository includes fine-tuning, evaluation, policy serving, and TensorRT deployment workflows.

**Canonical resources.** [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), including its [hardware recommendations](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md).

**Requirements.** Ubuntu 22.04+ for discrete-GPU systems; Python 3.12; CUDA 12.6+ (the current default dGPU setup documents CUDA 12.8); PyTorch 2.7+; `uv`; Git LFS; FFmpeg 4–7; and Hugging Face access to the gated Cosmos-Reason2-2B backbone. Inference requires one NVIDIA GPU with at least 16 GB VRAM. Fine-tuning requires at least 40 GB VRAM, with H100/L40-class hardware recommended. Jetson Thor/Orin and DGX Spark have separate CUDA/JetPack profiles.

**License.** Repository code is Apache-2.0. Model weights use the NVIDIA Open Model License, so code and weights must be reviewed separately before redistribution or production use. The gated backbone has its own access terms.

**MacBook Air.** The supported package explicitly targets Linux x86_64/aarch64 CUDA environments. Full installation, inference, fine-tuning, TensorRT export, and deployment do not run natively on Apple Silicon. The Mac can prepare datasets/configuration, inspect results, and call a remote GR00T policy server over the network. Inference and training require Linux plus an NVIDIA GPU or a suitable cloud instance.

### SimReady Central Library

**Purpose.** An alpha-stage library and discovery index for more than 2,000 simulation-ready OpenUSD assets. Assets include collision geometry, physical properties, articulation metadata, semantics, materials, and profile-specific validation evidence for robotics and factory simulation.

**Canonical resources.** NVIDIA's [GTC announcement](https://nvidianews.nvidia.com/news/nvidia-advances-isaac-platform-for-general-purpose-robotics-with-new-reference-humanoid-frameworks-simulation-pipelines-and-data), the official NVIDIA [SimReady Central dataset](https://huggingface.co/datasets/nvidia/simready-central-library), and [SimReady documentation](https://docs.omniverse.nvidia.com/simready/latest/).

**Requirements.** There is no GPU, CUDA, OS, or Python requirement merely to browse metadata or download an asset. Assets are OpenUSD packages and their runtime requirements come from the chosen consumer, normally Isaac Sim/Isaac Lab or another OpenUSD-compatible tool. Hugging Face authentication and acceptance of access conditions may be required. Reproducible use should pin each asset dataset and revision.

**License.** The central index dataset is marked Apache-2.0. Contributors retain ownership and control each asset package's license. Every selected asset's own license must therefore be recorded and checked; the index license does not automatically cover all indexed assets.

**MacBook Air.** Asset discovery, metadata handling, license review, and ordinary file download can run locally. Basic OpenUSD authoring may be possible with non-NVIDIA tooling. Isaac Sim rendering, PhysX/RTX validation, and Isaac Lab training using those assets require a supported NVIDIA system.

## 3. Recommended Architecture

Because no robot implementation exists yet, adopt a split workstation/runtime architecture rather than coupling the repository directly to CUDA packages:

1. **Portable project layer:** Python 3.12 schemas, robot/task configuration, dataset metadata, network client interfaces, tests, and documentation. This layer should work on macOS and Linux without CUDA.
2. **Isaac Lab adapter:** Simulation environments, robot assets, training configuration, policy evaluation, and export. Run on a pinned Linux NVIDIA GPU environment.
3. **GR00T adapter:** Dataset conversion, embodiment configuration, remote policy serving, and evaluation. Keep the model server separate from robot control and run it on Linux/NVIDIA hardware.
4. **Asset manifest:** Pin SimReady asset URLs, revisions, profiles, checksums, and per-asset licenses rather than committing large USD assets.
5. **Runtime adapter:** Defer Isaac OS integration until NVIDIA publishes installable artifacts and clear licensing. Design a narrow runtime interface now so Isaac OS, Isaac ROS 2, or a custom controller can implement it later.

Isaac Lab is the best first integration because it can define and test the robot/task before physical hardware is finalized. GR00T should be added only after the robot embodiment, observation/action spaces, demonstration data, and safety envelope are known. Isaac OS is not yet actionable from public artifacts.

## 4. Proposed Implementation After Approval

### Phase A — Minimal, non-breaking scaffold

1. Add a concise README section describing the local/remote split and supported hosts.
2. Add a small Python 3.12 project/environment file containing only portable dependencies; keep NVIDIA/CUDA packages in an opt-in GPU dependency group or separate lock/environment file.
3. Add one example script that validates configuration and queries a remote policy endpoint or emits a dry-run request. It must run on the Mac without an NVIDIA GPU and must not command physical hardware.
4. Add sample configuration with explicit observation/action shapes and placeholder embodiment fields, clearly marked for replacement once robot details are supplied.
5. Add a smoke test for the portable example.

### Phase B — GPU simulation host

1. Provision Ubuntu 22.04/24.04 with a supported NVIDIA GPU and current driver.
2. Pin Isaac Lab `v3.0.0`, Python 3.12, and the documented CUDA/PyTorch pair.
3. Generate an external Isaac Lab project rather than vendoring Isaac Lab.
4. Import or create the robot's URDF/MJCF/USD model after hardware details are known.
5. Select SimReady assets and record exact revisions and licenses in an asset manifest.
6. Run a deterministic smoke environment before starting training.

### Phase C — GR00T evaluation

1. Define the robot embodiment and relative end-effector action mapping.
2. Convert demonstration data to the GR00T/LeRobot format and validate it before training.
3. Start with remote inference on a 16 GB+ GPU; measure latency against the robot control period.
4. Fine-tune only after the open-loop baseline is understood; use a 40 GB+ GPU.
5. Put the policy behind a versioned network API and a safety supervisor. Never connect raw model output directly to actuators.

### Phase D — Physical deployment

1. Choose the actual edge target (for example Jetson AGX Thor, Jetson Orin, or an x86_64 NVIDIA computer).
2. Implement hardware drivers and hard real-time control separately from the learned policy service.
3. Add command limits, watchdogs, emergency stop behavior, stale-data rejection, and staged simulation/hardware-in-the-loop tests.
4. Re-evaluate Isaac OS only after its public repository, requirements, release cadence, and license are available. Use Isaac ROS 2 or a small custom transport if deployment cannot wait.

## 5. Requirements Before Repository-Specific Work

The following information is required before building a real robot integration:

- Robot type, manufacturer/model, degrees of freedom, and kinematics file.
- Controller protocol and control frequency.
- Sensors and their data formats/rates.
- On-robot compute and operating system.
- Simulation task and success criteria.
- Available demonstration/training data.
- Safety constraints and physical test procedure.
- Budget or available cloud/on-premises NVIDIA GPU hardware.

## 6. Risks

- **Empty repository:** there is no existing application to integrate with; any implementation establishes the project's architecture.
- **Isaac OS availability/licensing:** announcement language and the Isaac ROS FAQ are inconsistent, and no public installable release was found.
- **Hardware mismatch:** Apple Silicon cannot execute the supported CUDA workloads, making remote Linux/NVIDIA infrastructure mandatory.
- **Version churn:** Isaac Lab, Isaac Sim, CUDA, PyTorch, GR00T, and JetPack versions are tightly coupled; all GPU environments must be pinned.
- **Model and asset licensing:** GR00T code, model weights, gated backbone, SimReady index, and individual assets have distinct terms.
- **Embodiment gap:** GR00T is not a drop-in “brain”; observations, action spaces, calibration, data, evaluation, and safety integration are robot-specific.
- **Sim-to-real gap:** policies that work in simulation may fail on hardware due to dynamics, latency, sensing, calibration, or unmodeled contacts.
- **Safety:** learned policies must not bypass deterministic limits, watchdogs, emergency stops, or validated low-level controllers.
- **Cost:** useful simulation, inference, and especially fine-tuning may require substantial GPU memory, storage, and cloud spend.

## 7. Approval Gate

No integration code should be added until this plan is approved and the intended robot/hardware details are supplied. After approval, implement only Phase A first, validate it on the Mac, and commit the documentation/scaffold and example as small, clearly described commits.
