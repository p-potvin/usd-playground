<!-- v1.0.0 -->
# DA3-Standard Pipeline Benchmark — First Successful End-to-End Run

> **Sat, 12 Jul 2026 03:07 AM EDT** — First end-to-end success of the `da3-standard` preset on Hugging Face Jobs (L4 GPU). This document is formatted for both human and agent consumption.

---

## Summary

| Metric | Value |
|---|---|
| **Preset** | `da3-standard` |
| **GPU** | L4 (22GB) — Hugging Face Jobs |
| **Total wall time** | ~35 min (02:27 → 03:02) |
| **Total GPU cost** | **$0.36** |
| **Cost per frame** | **$0.00072/frame** |
| **Time per frame** | **4.2 sec/frame** (end-to-end) |
| **Gaussian count** | **1,839,412** |
| **Job ID** | `local-run-20260713-022745` |

---

## Input

| Property | Value |
|---|---|
| Source video | `backyard_134s_sunny.mp4` |
| Video duration | 134.28 seconds |
| Video file size | 251 MB |
| Sampling rate | 7 fps |
| Frames extracted | 500 |
| Frame resolution | 1920×1080 (Full HD) |
| Frame format | JPEG |
| Frame file size | ~0.7 MB each |
| Total frames size | 239.7 MB |
| Frames for DA3 SfM | 80 (subsampled from 500) |
| Subsampling ratio | 80/500 = 16% |

---

## Job A — DA3 SfM (Split Job, `--sfm-only`)

| Metric | Value |
|---|---|
| DA3 model | `depth-anything/DA3-LARGE-1.1` |
| DA3 inference time | 19.8 seconds |
| Frames processed | 80 |
| **Throughput** | **4 frames/sec** |
| Job A total wall time | 47 seconds (incl. scheduling + download) |
| Job A cost | **~$0.01** |
| **Cost per SfM frame** | **$0.000125/frame** |
| **Time per SfM frame** | **0.25 sec/frame** |
| Output | `processed_min.zip` (8 MB) |
| Output contents | `transforms.json` + `sparse_pc.ply` |

---

## Job B — Nerfstudio Splatfacto Training (`--train-only`)

| Metric | Value |
|---|---|
| Training iterations | 15,000 |
| Training time | 1,372.2 seconds (22.9 min) |
| Export time | 70.7 seconds (1.2 min) |
| Job B total wall time | 1,561 seconds (26.0 min, incl. scheduling + download) |
| Job B cost | **~$0.35** |
| **Training throughput** | **10.9 iter/sec** (655 iter/min) |
| **Cost per iteration** | **$0.0000233/iter** |
| **Cost per training frame** | **$0.0007/frame** |
| **Time per training frame** | **2.74 sec/frame** |
| Image cache mode | `cpu` (`pipeline.datamanager.cache-images=cpu`) |
| Checkpoint interval | Every 1,000 iterations |
| Cull alpha threshold | 0.05 |
| Downscale factor | 1 (full resolution) |

### Training Args

```json
[
  "--max-num-iterations", "15000",
  "--vis", "tensorboard",
  "--viewer.quit-on-train-completion", "True",
  "--pipeline.datamanager.cache-images", "cpu",
  "--steps-per-save", "1000",
  "--pipeline.model.cull-alpha-thresh", "0.05"
]
```

---

## Output Files

| File | Size | Description |
|---|---|---|
| `splat.ply` | 435 MB | Full-attribute 3DGS PLY (from `ns-export`) |
| `cloud.splat` | **56.1 MB** | Packed `.splat` for web viewer (**7.75x compression**) |
| `cloud.ply` | 414 MB | Converted cloud PLY |
| `cloud_preview.ply` | 2.9 MB | 200k point preview for viewport |
| `cloud.usda` | 1,395 MB | USD stage (points + primvars) |
| `model.zip` | 1,825 MB | Training checkpoint archive |
| `processed_min.zip` | 8 MB | transforms.json + sparse_pc.ply |
| `frames.zip` | 240 MB | Input frames archive |

### File Locations

```
D:\vaultwares-studio-jobs\data\jobs\local-run-20260713-022745\
├── frames\                                    # 500 JPEG frames (1920×1080)
├── frames.zip                                 # 240 MB archive
├── reconstruction\
│   ├── cloud.splat                            # ← 56 MB viewer file
│   ├── cloud.ply                              # 414 MB
│   ├── cloud_preview.ply                      # 2.9 MB preview
│   ├── cloud.usda                             # 1.4 GB USD stage
│   ├── gsplat_export\
│   │   └── splat.ply                          # 435 MB raw export
│   ├── remote_out\
│   │   ├── model.zip                          # 1.8 GB checkpoint
│   │   └── processed_min.zip                  # 8 MB SfM output
│   └── summary.json                           # Job metadata
└── reconstruction_sfm\
    └── remote_out\
        ├── processed_min.zip                  # 8 MB (Job A output)
        └── summary.json                       # SfM metadata
```

---

## Post-Processing

| Step | Result |
|---|---|
| Gravity alignment | Rotated **59.4°** to bring scene up to +Y |
| Skewness | +1.07 (not flipped) |
| Camera staging | 7 default cameras generated |
| Splat packing | 435 MB PLY → 56.1 MB .splat (7.75x compression) |

---

## Cost Breakdown

| Component | Cost | % of Total |
|---|---|---|
| Job A (DA3 SfM) | $0.01 | 2.8% |
| Job B (Training + Export) | $0.35 | 97.2% |
| **Total** | **$0.36** | 100% |

### Cost Scaling Estimates

| Frames | Est. Cost | Est. Time |
|---|---|---|
| 100 | $0.07 | ~7 min |
| 250 | $0.18 | ~18 min |
| 500 | $0.36 | ~35 min |
| 1,000 | $0.72 | ~70 min |
| 2,000 | $1.44 | ~140 min |

> *SfM cost is negligible (~$0.01 flat). Training scales linearly with frame count at $0.0007/frame.*

---

## Key Configuration

| Setting | Value | Why |
|---|---|---|
| DA3 model | DA3-LARGE-1.1 | Balance of speed/quality |
| Max SfM frames | 80 | Prevents OOM on 22GB L4 (quadratic attention) |
| Downscale factor | 1 | Full resolution training |
| Max iterations | 15,000 | Sufficient convergence for outdoor scenes |
| `TORCHDYNAMO_DISABLE` | 1 | Prevents torch.compile inductor import crash |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces CUDA fragmentation |
| xformers | **Removed** | Caused torch version mismatch (2.1→2.13) |
| `applied_transform` | **Removed** from transforms.json | 4x4 matrix caused `3x4 @ 5x4` shape error |
| Intrinsics scaling | DA3 internal → original 1920×1080 | DA3 resizes internally; nerfstudio loads full-res |
| moviepy deps | `--no-deps` install | Prevents numpy downgrade (moviepy wants numpy≤1.20) |

---

## Bugs Fixed During This Run

| # | Bug | Root Cause | Fix |
|---|---|---|---|
| 1 | CUDA OOM on DA3 inference | 500 frames → quadratic attention | Subsample to 80 frames |
| 2 | `ImportError: mm_configs` | xformers upgraded torch 2.1→2.13 | Remove xformers, `--no-deps` on all DA3 deps |
| 3 | `ModuleNotFoundError: proglog` | moviepy installed with `--no-deps` | Add `proglog imageio_ffmpeg` to `--no-deps` list |
| 4 | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x4 and 5x4)` | `applied_transform` was 4x4 identity in transforms.json | Remove `applied_transform` (not needed for DA3) |
| 5 | `AssertionError: image size (1920,1080) != camera params (504,280)` | DA3 intrinsics at internal resolution, images at full res | Scale intrinsics by `orig_w/da_w`, `orig_h/da_h` |
| 6 | Depth map path mismatch | `transforms.json` referenced `depths/{stem}.npy` but saved as `frame_{i:05d}.npy` | Remove `depth_file_path` (splatfacto doesn't use depths) |
| 7 | `TORCHDYNAMO_DISABLE` only in train-only path | Full mode didn't set it | Added to both training paths |

---

## Environment

| Component | Version |
|---|---|
| Base image | `ghcr.io/nerfstudio-project/nerfstudio:latest` |
| PyTorch | 2.1.2+cu118 |
| CUDA | 11.8.0 |
| Python | 3.10 |
| DA3 | `git+https://github.com/ByteDance-Seed/Depth-Anything-3.git` (main) |
| DA3 model | `depth-anything/DA3-LARGE-1.1` |
| HF Space | `hf.co/spaces/clopeux/vw-studio-da3` |
| GPU flavor | `l4x1` ($0.80/h) |

---

## Agent-Readable Summary

```yaml
preset: da3-standard
job_id: local-run-20260713-022745
date: 2026-07-12
status: SUCCESS
gpu: L4 (22GB)
total_cost_usd: 0.36
total_wall_time_min: 35
frames_total: 500
frames_sfm: 80
frame_resolution: "1920x1080"
da3_model: "depth-anything/DA3-LARGE-1.1"
da3_inference_s: 19.8
da3_throughput_fps: 4.0
da3_cost_usd: 0.01
train_iterations: 15000
train_time_s: 1372.2
train_throughput_iter_per_s: 10.9
train_cost_usd: 0.35
export_time_s: 70.7
gaussian_count: 1839412
splat_ply_mb: 435
cloud_splat_mb: 56.1
compression_ratio: 7.75
gravity_rotation_deg: 59.4
cost_per_frame_usd: 0.00072
time_per_frame_s: 4.2
```
