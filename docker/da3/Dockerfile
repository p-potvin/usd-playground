# vw-studio-da3 — DA3 (Depth Anything 3) execution image.
# rebuild: 2026-07-12
#
# Based on the nerfstudio image (for splatfacto training in --train-only mode),
# with DA3 + the specific gsplat commit needed for DA3's Gaussian head.
#
# Built server-side on Hugging Face:
#   .venv\Scripts\python.exe tools\push_da3_space.py
# Jobs run it via image="hf.co/spaces/<user>/vw-studio-da3".

FROM ghcr.io/nerfstudio-project/nerfstudio:latest

USER root
ENV PYTHONUNBUFFERED=1

# TF32 auto-enable for Ampere+ GPUs (same as prod worker).
RUN python3 -c "import site; \
        print(' '.join(site.getsitepackages() + [site.getusersitepackages()]))" \
        > /tmp/sitedirs.txt \
    && for d in $(cat /tmp/sitedirs.txt) \
                /usr/local/lib/python3.10/dist-packages \
                /usr/lib/python3/dist-packages \
                /usr/lib/python3.10/dist-packages; do \
            if [ -d "$d" ]; then \
                printf '%s\n' \
                    'try:' \
                    '    import torch' \
                    "    torch.set_float32_matmul_precision('high')" \
                    'except Exception:' \
                    '    pass' \
                    > "$d/sitecustomize.py" \
                && echo "[Dockerfile] wrote sitecustomize.py to $d"; \
            fi; \
        done

# CuDNN runtime kernel selection. The nerfstudio base is CUDA 11.8 / torch
# 2.1.2+cu118, so this must be the cu11 build — the cu12 package that used to
# be here never matched the runtime it was loaded into.
RUN python3 -m pip install --no-cache-dir nvidia-cuda-nvrtc-cu11 \
    || pip3 install --no-cache-dir nvidia-cuda-nvrtc-cu11 \
    || echo "[Dockerfile] nvrtc package install skipped"

# nvcc, for building gsplat's CUDA extension.
# nerfstudio's own Dockerfile ends its multi-stage build on
# nvidia/cuda:11.8.0-runtime-ubuntu22.04, which ships no compiler — so every
# source build of a CUDA extension in this image silently failed. The CUDA apt
# repo is already configured in the base, so pull the 11.8 compiler + cudart
# headers to match. Version-exact: a cu12 nvcc against torch cu118 will not link.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-11-8 cuda-cudart-dev-11-8 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=/usr/local/cuda-11.8/bin:${PATH}
# No GPU is present at build time, so the arch list can't be probed — name the
# flavors we actually run on: T4 (7.5), A100 (8.0), A10G (8.6), L4 (8.9).
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
RUN nvcc --version

# wheel registers the bdist_wheel setuptools command that gsplat's setup.py
# needs even for --no-build-isolation metadata prep; without it the build
# fails with "invalid command 'bdist_wheel'" before nvcc is ever invoked.
# ninja is optional but turns gsplat's build from a slow serial distutils
# compile into a parallel one — worth the ~2MB.
RUN python3 -m pip install --no-cache-dir wheel ninja

# Alias python3 → python for job commands.
RUN ln -sf "$(command -v python3)" /usr/local/bin/python \
    && (python3 -m pip install --no-cache-dir \
            "huggingface_hub>=0.26" \
            "plyfile>=1.0.3" \
            "trimesh>=4.3.0" \
        || pip3 install --no-cache-dir \
            "huggingface_hub>=0.26" \
            "plyfile>=1.0.3" \
            "trimesh>=4.3.0")

# Depth Anything 3 — pose + depth + direct 3DGS prediction.
# DA3 uses hatchling + hatch-vcs as build backend, so we need git and those
# packages available. We install with --no-deps to avoid pulling conflicting
# versions of torch, numpy, etc. — the nerfstudio base image already has them.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir hatchling>=1.25 hatch-vcs>=0.4
RUN python3 -m pip install --no-cache-dir --no-deps \
        "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
# Install DA3 runtime deps that aren't already in the nerfstudio image.
# All with --no-deps to prevent pip from upgrading torch/numpy/etc.
# moviepy needs: proglog, imageio_ffmpeg, decorator, imageio, tqdm
# (all already in nerfstudio image except proglog and imageio_ffmpeg)
#
# e3nn needs opt_einsum_fx + scipy at import time (not just declared, actually
# imported by e3nn.util.jit at module load). --no-deps drops both silently,
# and DA3's rotate_sh() (depth_anything_3/utils/sh_helpers.py) wraps
# `from e3nn.o3 import matrix_to_angles, wigner_D` in a bare try/except that
# swallows the resulting ImportError and warns "Dependency 'e3nn' not found"
# instead of failing — so a broken e3nn install doesn't surface until
# rotate_sh() is actually called mid-inference (--gs-only / infer_gs=True
# only), producing a *NameError* three function calls later instead of the
# real ImportError. Found 2026-07-30 via a $0.03 real GPU job that got all
# the way to DA3 inference before dying here.
RUN python3 -m pip install --no-cache-dir --no-deps einops omegaconf safetensors \
        pillow_heif evo e3nn "moviepy==1.0.3" proglog imageio_ffmpeg
RUN python3 -m pip install --no-cache-dir opt_einsum_fx scipy
# Prove e3nn's rotation path (the one --gs-only actually needs) imports clean.
RUN python3 -c "from e3nn.o3 import matrix_to_angles, wigner_D; print('[Dockerfile] e3nn.o3 OK')"

# The Gaussian head needs a specific gsplat commit (not the nerfstudio-bundled
# one). This pin matches DA3's own `gs` extra in its pyproject.toml.
# Deliberately NOT guarded by `|| echo`: the previous fallback turned a missing
# compiler into a silently broken --gs-only mode that only surfaced at runtime.
# If this can't build, the image build should fail here and say so.
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"
# Prove the extension actually imports before we ship the image.
RUN python3 -c "import gsplat; print('[Dockerfile] gsplat OK:', gsplat.__version__)"

# Stage bootstrap + entrypoints.
COPY vw_stage.py /opt/vw/vw_stage.py
COPY da3_entrypoint.py /opt/vw/da3_entrypoint.py
COPY recon_entrypoint.py /opt/vw/recon_entrypoint.py
COPY render_entrypoint.py /opt/vw/render_entrypoint.py
# Gaussian merge utilities (used by --merge-splat incremental mode).
COPY gaussian_merge.py /opt/vw/gaussian_merge.py
COPY splat_io.py /opt/vw/splat_io.py

WORKDIR /workspace
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCHDYNAMO_DISABLE=1
CMD ["python", "/opt/vw/vw_stage.py"]
