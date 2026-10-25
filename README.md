# Robot

Minimal Docker-first integration scaffold for NVIDIA Isaac GR00T N1.7. The repository currently has no robot embodiment or hardware implementation, so this first integration safely verifies connectivity to GR00T's official ZeroMQ `PolicyServer` without sending observations or actuator commands.

## What Runs Where

| Environment | Supported work |
| --- | --- |
| Apple Silicon Mac | Edit configuration, build and run the CPU-only client container, inspect assets/results, and connect to a remote policy server |
| Linux + NVIDIA GPU | Run Isaac GR00T inference/fine-tuning, Isaac Lab training, Isaac Sim, and later robot deployment |

Docker provides the project environment; no host Python virtual environment is required.

## Local Setup

Requirements:

- Docker Desktop with Linux containers
- A reachable GR00T N1.7 policy server for the live connectivity check

Create local configuration:

```bash
cp .env.example .env
```

Build the deterministic client image and verify its configuration without a server:

```bash
docker compose build
docker compose run --rm groot-client --dry-run
```

The default `GROOT_HOST=host.docker.internal` reaches a server exposed by the host. To use a remote Linux GPU machine, set `GROOT_HOST` in `.env` to its private hostname or IP and restrict TCP port `5555` to trusted clients.

Run the live health check:

```bash
docker compose run --rm groot-client
```

Expected output starts with `GR00T PolicyServer is healthy`.

## GPU Host: GR00T N1.7

Run the server on a supported Ubuntu/NVIDIA machine using the official [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) instructions. GR00T N1.7 requires Python 3.12 and an NVIDIA GPU with at least 16 GB VRAM for inference. Fine-tuning requires at least 40 GB VRAM. NVIDIA provides platform-specific Dockerfiles under `scripts/deployment/`; use those rather than extending this lightweight client image with CUDA packages.

After installing GR00T and obtaining model access, start its official server:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

`NEW_EMBODIMENT` is only a scaffold value. Do not request actions or connect a robot until its observation schema, action space, controller limits, calibration, watchdogs, and emergency-stop behavior are defined.

## Isaac Lab and SimReady

Isaac Lab 3.0 belongs on the Linux GPU host and should be kept in its own NVIDIA-supported container/environment. SimReady assets should be referenced through a future manifest containing exact dataset revisions, checksums, profiles, and per-asset licenses; large USD assets should not be committed here.

See [the NVIDIA integration plan](docs/NVIDIA_PLAN.md) for requirements, licensing notes, architecture, and phased implementation.
