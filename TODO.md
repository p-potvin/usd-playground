# Project: USD Digital Twin Playground

## Phase 0: Infrastructure & Setup

- [x] Integrate `vaultwares-agentciation` framework
- [x] Add NVIDIA Cosmos submodules (`cosmos-reason2`, `cosmos-transfer2.5`)
- [x] Refactor pipeline to event-driven orchestrator (`run_pipeline_demo.py`)

## Phase 1: Capture & Reconstruction

- [x] Capture test video (simulated with `test_input.mp4`)
- [x] Extract frames using ffmpeg
- [x] Run COLMAP SfM for camera pose estimation
- [x] Train Gaussian Splat model (gsplat/3DGRUT)
- [x] Export to PLY

## Phase 2: USD Conversion & Authoring

- [x] Convert PLY to OpenUSD (26.03 schema)
- [x] Compose scene in USD (add lights, floor)
- [x] Validate USD structure

## Phase 3: Isaac Sim Integration

- [x] Load USD scene into Isaac Sim
- [x] Add navigation cameras
- [x] (Optional) Import robot (URDF -> USD)
- [x] Generate synthetic data with Replicator

## Phase 4: Cosmos Augmentation

- [x] Scene annotation with Cosmos Reason 2
- [x] Domain transfer with Cosmos Transfer 2.5
