"""Reconstruction quality presets.

Each preset picks splatfacto training parameters AND the HF Jobs flavor that
fits them, plus a cost estimate for the consent dialog. LOCAL_DEBUG preserves
the historical 250-iteration quick path used when reconstruction runs on the
local machine (heavy local training stays opt-in; the PC hosts the
VaultWares API).

Flag spelling drifts between nerfstudio releases — the worker entrypoint
probes ``ns-train splatfacto --help`` and drops unknown flags rather than
failing the job.

Split-job presets (split_jobs=True) run COLMAP on a cheap cpu-upgrade
instance first, then hand the processed_min.zip to a GPU job for training
only. This eliminates paying L4 rates ($0.80/hr) for CPU-only COLMAP work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .runners import CostEstimate, RATE_TABLE_SOURCE, estimate_cost


class SfmMethod(str, Enum):
    COLMAP = "colmap"
    MAST3R = "mast3r"
    DA3 = "da3"


def _flavor_label(flavor: str | list[str]) -> str:
    """Display form of a flavor or fallback list, e.g. 't4-small|a10g-small'."""
    return flavor if isinstance(flavor, str) else "|".join(flavor)


@dataclass(frozen=True)
class QualityPreset:
    key: str
    label: str
    iterations: int
    downscale_factor: int
    gaussian_cap: int
    flavor: str
    est_minutes: float
    extra_train_args: tuple[str, ...] = field(default_factory=tuple)
    # Checkpoint cadence for ns-train --steps-per-save. Production default
    # is 1000; lab presets can widen this (fewer ckpts, smaller model.zip)
    # or shrink it (more snapshots through cull boundaries).
    steps_per_save: int = 1000
    # Split-job optimization: run COLMAP on cpu-upgrade, training on `flavor`.
    # When True, sfm_flavor and sfm_est_minutes define the first job; the
    # existing flavor/est_minutes apply only to the training job.
    split_jobs: bool = False
    # A single flavor, or a fallback list tried in order when the first
    # candidate won't leave HF Jobs' SCHEDULING state within a timeout (spot
    # GPU availability, esp. l4x1, fluctuates). See HfJobsStageRunner.run.
    sfm_flavor: str | list[str] = "cpu-upgrade"
    sfm_est_minutes: float = 0.0  # 0 = not split
    # Lab-mode toggles: skip ffmpeg sharpness pruning and hard-cap the post-
    # extract frame count. Used by the cpu-upgrade experiment Space to send
    # raw, unpruned dense frame sets to COLMAP. 0 = no cap (production default).
    unrestricted_frames: bool = False
    frame_cap: int = 0
    # Explicit SfM job timeout in seconds; 0 = default (max(3600, sfm_est*60*4)).
    # Needed when sfm_est_minutes can't be trusted as an upper bound (lab runs).
    sfm_timeout_seconds: int = 0
    # Image used by the SfM (Job A) leg. Empty string = use the runner's default
    # worker_image. Lab presets point this at the experimental Space so prod is
    # never touched.
    sfm_image_override: str = ""
    # SfM engine: colmap (default), mast3r, or da3 (Depth Anything 3).
    # Controls which entrypoint the SfM leg invokes.
    sfm_method: SfmMethod = SfmMethod.COLMAP
    # DA3-specific: model ID on HuggingFace (depth-anything/DA3-LARGE-1.1 etc.)
    da3_model: str = ""
    # DA3-specific: when True, the SfM leg outputs a direct 3DGS PLY (gs_ply)
    # and skips splatfacto training entirely. Only works with DA3-GIANT or
    # DA3NESTED models that have the Gaussian head.
    da3_direct_gs: bool = False
    # Frames handed to DA3 inference (evenly subsampled). The entrypoint's own
    # default of 80 was tuned for pose+depth on a 22GB L4; infer_gs=True stacks
    # a Gaussian head on top of the already-quadratic multi-view attention and
    # needs materially more. Lowered for the direct-3DGS presets.
    da3_max_sfm_frames: int = 80

    def train_args(self) -> list[str]:
        """splatfacto arguments shared by local and remote execution."""
        args = [
            "--max-num-iterations", str(self.iterations),
            "--vis", "none",
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.datamanager.cache-images", "cpu",
            # Save a checkpoint every 1000 steps instead of only the final
            # one — lets us extract splat snapshots at any step retroactively
            # via ns-export --load-step N. Stitched-video case at step 2900
            # had 1.1M GSs that disappeared after the cull at 3400; only the
            # final checkpoint was kept, so the high-density snapshot was
            # unrecoverable. Each .ckpt is ~150-300 MB; 15 checkpoints in
            # standard run ≈ ~3 GB extra in model.zip. Worth it.
            "--steps-per-save", str(self.steps_per_save),
        ]
        args.extend(self.extra_train_args)
        return args

    def cost(self) -> CostEstimate:
        """Combined cost estimate. For split presets this sums both jobs."""
        if self.split_jobs and self.sfm_est_minutes > 0:
            sfm = self.sfm_cost()
            train = estimate_cost(self.flavor, self.est_minutes)
            return CostEstimate(
                flavor=f"{_flavor_label(self.sfm_flavor)}+{self.flavor}",
                est_minutes=self.sfm_est_minutes + self.est_minutes,
                rate_usd_per_hour=0.0,  # mixed flavors; use est_usd directly
                est_usd=round(sfm.est_usd + train.est_usd, 2),
                source=RATE_TABLE_SOURCE,
            )
        return estimate_cost(self.flavor, self.est_minutes)

    def sfm_cost(self) -> CostEstimate:
        # sfm_flavor may be a fallback list — quote against the priciest
        # candidate so the confirm dialog never undercounts what could run.
        candidates = self.sfm_flavor if isinstance(self.sfm_flavor, list) else [self.sfm_flavor]
        return max(
            (estimate_cost(f, self.sfm_est_minutes) for f in candidates),
            key=lambda e: e.est_usd,
        )

    def train_cost(self) -> CostEstimate:
        return estimate_cost(self.flavor, self.est_minutes)


PRESETS: dict[str, QualityPreset] = {
    "draft": QualityPreset(
        key="draft",
        label="Draft (fast, low cost)",
        iterations=7_000,
        downscale_factor=4,
        gaussian_cap=300_000,
        flavor="l4x1",
        est_minutes=15,
        extra_train_args=("--pipeline.model.stop-split-at", "5000"),
    ),
    "standard": QualityPreset(
        key="standard",
        label="Standard",
        iterations=15_000,
        # No training downscale — splatfacto sees the full-res images so the
        # output gaussians keep the detail SfM already extracted at full res.
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=25,  # training-only time (was 60 combined before split)
        extra_train_args=("--pipeline.model.cull-alpha-thresh", "0.05"),
        split_jobs=True,
        sfm_flavor="cpu-upgrade",
        sfm_est_minutes=35,  # ns-process-data sequential + mapper on cpu-upgrade
    ),
    # Refine an existing splat: launcher passes --refine-from, worker resumes
    # from the base model.zip checkpoint via ns-train --load-dir.
    #
    # CPU/GPU breakdown (1500-frame joint set, measured):
    #   ~90 min  feature_extractor + vocab_tree matching + image_registrator  (CPU only)
    #   ~15 min  5k splatfacto resume iters                                  (GPU)
    #   ~15 min  boot / IO / model pull
    # Split path: cpu-upgrade handles SfM (~90 min @ $0.10/hr = $0.15),
    # l4x1 handles training-only (~20 min @ $0.80/hr = $0.27) = ~$0.42 total
    # vs $1.60 for the naive single-job path.
    "refine": QualityPreset(
        key="refine",
        label="Refine (resume from base checkpoint)",
        iterations=5_000,
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=20,  # training-only time (was 120 combined before split)
        extra_train_args=("--pipeline.model.cull-alpha-thresh", "0.05"),
        split_jobs=True,
        sfm_flavor="cpu-upgrade",
        sfm_est_minutes=90,  # refine COLMAP: feature_extractor + vocab_tree + image_registrator
    ),
    "high": QualityPreset(
        key="high",
        label="High (slow, best quality)",
        iterations=30_000,
        downscale_factor=2,
        gaussian_cap=1_500_000,
        flavor="a10g-large",
        est_minutes=75,
        extra_train_args=("--pipeline.model.rasterize-mode", "antialiased"),
    ),
    # Experimental: cpu-upgrade Space, dense unpruned frame set, hard-cap 3000.
    # Pairs with docker/lab/Dockerfile pushed via tools/push_lab_space.py.
    # Job B (GPU training) is intentionally left pointing at l4x1; queue_lab_recon.py
    # fires Job A only — the GPU half is iterated separately in the HF console.
    "lab-cpu-3000": QualityPreset(
        key="lab-cpu-3000",
        label="Lab: cpu-upgrade SfM, 3000 unpruned frames",
        iterations=15_000,
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=25,
        split_jobs=True,
        sfm_flavor="cpu-upgrade",
        sfm_est_minutes=240,  # 4h labelled budget; sfm_timeout_seconds bumps the actual cap to 12h
        unrestricted_frames=True,
        frame_cap=3000,
        sfm_timeout_seconds=21_600,  # 6h ceiling (HF account caps 12h+ as 500)
        sfm_image_override="hf.co/spaces/{owner}/vw-studio-recon-lab",
        # Wider checkpoint cadence than prod (1500 vs 1000) — gets the 3000-step
        # snapshot near the splatfacto cull boundary while keeping model.zip
        # in the 6-9 GB range on a 15k-iter run.
        steps_per_save=1500,
    ),
    # Experimental: MASt3R-SfM instead of COLMAP. Runs on l4x1 (SfM needs GPU
    # for MASt3R; there is no CPU-only path). Lab image is docker/lab/Dockerfile
    # rebuilt around pytorch+cuda12.1 with the naver/mast3r install baked in.
    # Rationale: COLMAP starves on low-texture / grass / sky / low-parallax
    # captures. MASt3R's matcher is learned and handles those cases. Produces
    # a nerfstudio-format processed_min.zip (transforms.json + sparse_pc.ply)
    # that the prod worker's --train-only path consumes unchanged.
    "lab-mast3r-sfm": QualityPreset(
        key="lab-mast3r-sfm",
        label="Lab: MASt3R-SfM on l4x1, training on l4x1",
        iterations=15_000,
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=25,
        split_jobs=True,
        sfm_flavor="l4x1",  # GPU-required for MASt3R inference
        sfm_est_minutes=45,  # ~45 min expected for 3000 frames w/ retrieval-20
        unrestricted_frames=True,
        frame_cap=3000,
        sfm_timeout_seconds=10_800,  # 3h ceiling -- MASt3R is much faster than COLMAP
        sfm_image_override="hf.co/spaces/{owner}/vw-studio-recon-lab",
        steps_per_save=1500,
    ),
    # --- DA3 (Depth Anything 3) presets ---
    # DA3 replaces COLMAP/MASt3R for SfM: a single model predicts camera poses,
    # multi-view depth, and (with the Giant model) direct 3D Gaussians.
    #
    # da3-draft: DA3-GIANT direct 3DGS output — no splatfacto training at all.
    # Produces a viewable splat in ~5 min on an L4. Quality is lower than
    # trained splatfacto but sufficient for preview / layout planning.
    #
    # Points at vw-studio-da3-gs, NOT the proven vw-studio-da3 image. The
    # --gs-only path was never actually exercised before 2026-07-29 — the
    # gsplat CUDA extension silently failed to build (nerfstudio's base image
    # ships no nvcc) and the Dockerfile swallowed the failure with `|| echo`.
    # vw-studio-da3-gs adds the CUDA 11.8 devel toolchain and drops that
    # fallback so a broken build fails loud instead of shipping a dead mode.
    # Repoint at vw-studio-da3 once this preset has a verified green run.
    "da3-draft": QualityPreset(
        key="da3-draft",
        label="DA3 Draft (direct 3DGS, no training)",
        iterations=0,  # no splatfacto training
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=0,  # single-job, SfM only
        split_jobs=False,
        sfm_method=SfmMethod.DA3,
        da3_model="depth-anything/DA3-GIANT-1.1",
        da3_direct_gs=True,
        # 24GB-class cards only. t4-small (16GB) was tried first on 2026-07-30
        # on the reasoning that DA3-GIANT's 5.4GB of weights don't need an L4 —
        # wrong axis: peak memory is activations, not weights, and the Gaussian
        # head OOM'd at 13.90GB allocated with 80 frames. Both entries below are
        # 24GB; a10g-small leads because l4x1 spot capacity is the least
        # reliable of the two (20+ min in SCHEDULING, same day).
        sfm_flavor=["a10g-small", "l4x1"],
        # Halved from the 80 that the pose+depth path uses. This is a draft
        # preview preset — 40 views is plenty — and it buys real headroom
        # against the quadratic attention on top of the Gaussian head.
        da3_max_sfm_frames=40,
        sfm_est_minutes=5,
        sfm_timeout_seconds=1_800,  # 30 min ceiling
        sfm_image_override="hf.co/spaces/{owner}/vw-studio-da3-gs",
    ),
    # da3-standard: DA3-LARGE for pose+depth → splatfacto training.
    # DA3-LARGE (0.35B) is fast and has no GS head, but its pose+depth are
    # sufficient to seed splatfacto. Split: DA3 SfM on l4x1 (~10 min),
    # splatfacto training on l4x1 (~20 min).
    "da3-standard": QualityPreset(
        key="da3-standard",
        label="DA3 Standard (DA3 SfM + splatfacto)",
        iterations=15_000,
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=20,
        extra_train_args=("--pipeline.model.cull-alpha-thresh", "0.05"),
        split_jobs=True,
        sfm_method=SfmMethod.DA3,
        da3_model="depth-anything/DA3-LARGE-1.1",
        da3_direct_gs=False,
        sfm_flavor="l4x1",
        sfm_est_minutes=10,
        sfm_timeout_seconds=3_600,
        sfm_image_override="hf.co/spaces/{owner}/vw-studio-da3",
    ),
    # da3-high: DA3-LARGE SfM + longer splatfacto training on a bigger GPU.
    "da3-high": QualityPreset(
        key="da3-high",
        label="DA3 High (DA3 SfM + extended training)",
        iterations=30_000,
        downscale_factor=2,
        gaussian_cap=1_500_000,
        flavor="a10g-large",
        est_minutes=60,
        extra_train_args=("--pipeline.model.rasterize-mode", "antialiased"),
        split_jobs=True,
        sfm_method=SfmMethod.DA3,
        da3_model="depth-anything/DA3-LARGE-1.1",
        da3_direct_gs=False,
        sfm_flavor="l4x1",
        sfm_est_minutes=10,
        sfm_timeout_seconds=3_600,
        sfm_image_override="hf.co/spaces/{owner}/vw-studio-da3",
    ),
    # da3-incremental: Add new footage to an existing splat without retraining.
    # DA3-GIANT runs on the new frames only, produces a direct 3DGS PLY, then
    # the entrypoint merges it with the base splat.ply via ICP alignment +
    # voxel dedup + dynamic culling. Single job, ~5 min on L4. No checkpoint
    # needed — works from just the base splat.ply (no model.zip required).
    #
    # Also gs-only, so also on vw-studio-da3-gs — see da3-draft's comment.
    "da3-incremental": QualityPreset(
        key="da3-incremental",
        label="DA3 Incremental (merge new footage into existing splat)",
        iterations=0,
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=0,
        split_jobs=False,
        sfm_method=SfmMethod.DA3,
        da3_model="depth-anything/DA3-GIANT-1.1",
        da3_direct_gs=True,
        # Same DA3-GIANT + Gaussian head as da3-draft — same 24GB floor and
        # same frame cap. See da3-draft above for the OOM that established it.
        sfm_flavor=["a10g-small", "l4x1"],
        da3_max_sfm_frames=40,
        sfm_est_minutes=5,
        sfm_timeout_seconds=1_800,
        sfm_image_override="hf.co/spaces/{owner}/vw-studio-da3-gs",
    ),
    # Historical local quick path: a smoke-level run that proves the toolchain
    # without tying up the local GPU. Used when no remote runner is configured.
    "local-debug": QualityPreset(
        key="local-debug",
        label="Local debug (250 iterations)",
        iterations=250,
        downscale_factor=4,
        gaussian_cap=100_000,
        flavor="cpu-basic",  # unused locally; kept for completeness
        est_minutes=0,
    ),
}

DEFAULT_PRESET_KEY = "standard"


def get_preset(key: str | None) -> QualityPreset:
    return PRESETS.get((key or "").lower(), PRESETS[DEFAULT_PRESET_KEY])
