# vw-studio-recon-lab (MASt3R variant) -- GPU-only SfM alternative to COLMAP.
#
# What this image is for:
#   Job A (SfM stage) using MASt3R-SfM instead of COLMAP. Handles low-texture /
#   low-parallax captures where COLMAP starves. Emits processed_min.zip with a
#   nerfstudio-format transforms.json + sparse_pc.ply, so the existing prod
#   worker's --train-only path (vw-studio-worker) consumes it unchanged for
#   splatfacto / 3DGRUT training.
#
# What this image is NOT for:
#   Training. Nerfstudio + splatfacto stay on the prod worker Space. This image
#   only has the SfM half. Two-Space split keeps each build small and focused.
#
# Build server-side on Hugging Face via tools/push_lab_space.py.
# Space must be PUBLIC (HF Jobs can't pull from private Space registries).

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

USER root
ENV PYTHONUNBUFFERED=1
ENV VW_LAB=1
ENV VW_LAB_ENGINE=mast3r
# Non-interactive apt for HF's build environment.
ENV DEBIAN_FRONTEND=noninteractive

# System deps needed for ASMK's cython build + OpenCV runtime + git submodules.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential python3-dev cython3 \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone MASt3R with the dust3r submodule (recursive is REQUIRED -- dust3r ships
# as a submodule and the code imports from it).
WORKDIR /opt
RUN git clone --recursive https://github.com/naver/mast3r.git

WORKDIR /opt/mast3r
# MASt3R's own requirements + dust3r's (they're distinct).
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r dust3r/requirements.txt

# ASMK for retrieval-based pair selection. Needed when we have more than a few
# dozen frames -- otherwise MASt3R would try N^2 pairs and OOM. Cython build.
# faiss-cpu is a HARD PREREQ: asmk's setup.py fails at install-time with
# "faiss package not installed" if it isn't already importable. faiss-cpu is
# fine here -- MASt3R retrieval on 3000 frames is not the bottleneck.
RUN pip install --no-cache-dir cython faiss-cpu \
    && git clone https://github.com/jenicek/asmk.git /opt/asmk \
    && cd /opt/asmk/cython && cython *.pyx && cd .. \
    && pip install --no-cache-dir .

# HF Hub + PLY writer for our entrypoint.
RUN pip install --no-cache-dir \
        "huggingface_hub>=0.26" \
        "plyfile>=1.0.3" \
        "trimesh>=4.3.0"

# Pre-download the pretrained model repo so the first HF Job doesn't burn its
# wall clock pulling ~2.75 GB from HF Hub. The official naver repo ships
# `model.safetensors` (not `.pth` -- that's the camenduru community mirror
# only), so we use snapshot_download to grab everything: config.json,
# model.safetensors, README. MASt3R's AsymmetricMASt3R.from_pretrained reads
# config.json and picks the right weight format automatically.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')"

# Retrieval model + codebook for MASt3R-SfM's retrieval-based pair selection.
# Naver hosts these on their CDN, NOT on HF Hub (the *_trainingfree HF repo
# doesn't exist). Filenames must match what MASt3R's code expects, and both
# files live next to the main checkpoint under MASt3R's internal cache path.
RUN mkdir -p /root/.cache/torch/hub/checkpoints \
    && wget -q -O /root/.cache/torch/hub/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
        https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
    && wget -q -O /root/.cache/torch/hub/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl \
        https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl \
    || echo "[Dockerfile] retrieval weights pre-fetch skipped -- will download on first run"

# Compile RoPE CUDA kernels for a real speedup on Job A. HF Space builders are
# CPU-only, so torch.cuda can't introspect visible arches; we pin the list to
# the GPUs we might rent (Ampere L4=8.6, Ada 4090/L40S=8.9, Hopper H100=9.0).
# +PTX suffix on the last entry lets JIT fill in for any newer arch. Without
# this the build hit IndexError from an empty arch list and fell back to
# pure-python RoPE, which is ~2-3x slower on MASt3R's transformer forward pass.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
RUN cd /opt/mast3r/dust3r/croco/models/curope \
    && (python setup.py build_ext --inplace \
        || echo "[Dockerfile] curope build skipped -- falling back to pure python RoPE")

# vw stage bootstrap + our MASt3R entrypoint. push_lab_space.py uploads these.
COPY vw_stage.py /opt/vw/vw_stage.py
COPY mast3r_entrypoint.py /opt/vw/mast3r_entrypoint.py
# Also keep the COLMAP entrypoint around so we can still fire an old-style lab
# job against this image if we want to A/B compare (works only if we install
# COLMAP separately -- currently the image ships MASt3R only).
COPY recon_entrypoint.py /opt/vw/recon_entrypoint.py

# Make MASt3R importable from anywhere. dust3r submodule lives under mast3r/.
ENV PYTHONPATH=/opt/mast3r:/opt/mast3r/dust3r:${PYTHONPATH}

WORKDIR /workspace
CMD ["python", "/opt/vw/vw_stage.py"]
