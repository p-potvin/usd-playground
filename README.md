# usd-playground
Baseline project for generating **Digital Twins** using [OpenUSD](https://openusd.org),
[NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/), and
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim) — with real-life footage as
the primary data source.

## Repository Contents

| File | Description |
|------|-------------|
| [`REPORT.md`](REPORT.md) | Detailed research report: technologies, hardware, libraries, Isaac Sim integration (local & cloud), camera navigation, robotic systems, and Cosmos models |
| [`requirements.txt`](requirements.txt) | Python dependencies for the full pipeline (OpenUSD, gsplat, Nerfstudio, Cosmos, Isaac Sim) |
| [`TECHNICAL_SPECS.md`](TECHNICAL_SPECS.md) | Technical specifications: package versions, download/install sizes, CUDA requirements, storage estimates |

## Quick Start

```bash
# Create and activate a Python 3.11 environment
conda create -n usd-twin python=3.11 -y
conda activate usd-twin

# Install COLMAP and ffmpeg (needed by Nerfstudio)
conda install -c conda-forge colmap ffmpeg -y

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all Python dependencies
pip install -r requirements.txt

# For NVIDIA Omniverse / Isaac Sim (optional, requires NVIDIA developer account)
pip install isaacsim --extra-index-url https://pypi.nvidia.com
```

## Pipeline Overview

```
Video capture  →  COLMAP + gsplat / 3DGRUT  →  PLY export
    →  OpenUSD 26.03 conversion  →  USD scene
    →  Isaac Sim loading  →  Camera navigation + Cosmos augmentation
```

See [`REPORT.md`](REPORT.md) for the complete workflow, hardware feasibility analysis,
and integration guides.

## Hardware Requirements

- **Minimum:** NVIDIA GPU with 12 GB VRAM + RT cores (e.g. RTX 3080), 32 GB RAM, 250 GB free SSD
- **Recommended:** RTX 4080/4090 (16–24 GB), 64 GB RAM, 500 GB NVMe SSD
- **Cloud alternative:** AWS `g6e.2xlarge` (L40S, 48 GB) — see [`REPORT.md §4.2`](REPORT.md#42-cloud-gpus)

## Key Technologies

- **3D Reconstruction:** [gsplat](https://github.com/nerfstudio-project/gsplat) + [3DGRUT](https://github.com/nv-tlabs/3dgrut) (3DGUT / 3DGRT)
- **Scene Format:** [OpenUSD](https://openusd.org) 26.03+ with native Gaussian Splat schema
- **Simulation:** [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim) 4.5 / 5.0
- **World Models:** [NVIDIA Cosmos](https://github.com/nvidia-cosmos) Reason 2 + Transfer 2.5
- **Capture Preprocessing:** [Nerfstudio](https://nerf.studio) + [COLMAP](https://colmap.github.io)
