# Project: USD Digital Twin Playground

## Phase 1: Capture & Reconstruction
- [ ] Capture test video (3-5 mins, 4K)
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
