# Project: USD Digital Twin Playground

## Phase 0: Infrastructure & Setup

- [x] Integrate `vaultwares-agentciation` framework
- [x] Add NVIDIA Cosmos submodules (`cosmos-reason2`, `cosmos-transfer2.5`)
- [x] Refactor pipeline to event-driven orchestrator (`run_pipeline_demo.py`)

## Phase 1: Capture & Reconstruction

- [x] Capture test video (simulated with `test_input.mp4`)
- [ ] Extract frames using ffmpeg
- [ ] Run COLMAP SfM for camera pose estimation
- [ ] Train Gaussian Splat model (gsplat/3DGRUT)
- [ ] Export to PLY

## Phase 2: USD Conversion & Authoring

- [ ] Convert PLY to OpenUSD (26.03 schema)
- [ ] Compose scene in USD (add lights, floor)
- [ ] Validate USD structure

## Phase 3: Isaac Sim Integration

- [ ] Load USD scene into Isaac Sim
- [ ] Add navigation cameras
- [ ] (Optional) Import robot (URDF -> USD)
- [ ] Generate synthetic data with Replicator

## Phase 4: Cosmos Augmentation

- [ ] Scene annotation with Cosmos Reason 2
- [ ] Domain transfer with Cosmos Transfer 2.5
