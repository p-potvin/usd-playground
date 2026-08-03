<!-- v1.2.0 -->
# DA3-Streaming Integration — Spec

> **Fri, 1 Aug 2026** — Plan for replacing the 80-frame `--max-sfm-frames` cap
> with DA3-Streaming's sliding-window chunking. Written against the **actual**
> upstream source, not the README. Several things differ from the config we
> sketched; those are called out explicitly under [Config](#config-corrected).
>
> **Updated Sun, 3 Aug 2026 — phases 1 and 2 are done, both free.** Streaming
> runs end-to-end on the local RTX 3060, and the pose converter is written and
> **validated against a known-good `da3-standard` run**. See
> [Phase 1 findings](#phase-1-findings-measured-not-predicted) and
> [Phase 2 findings](#phase-2-findings-pose-convention-validated).
> Cost so far: **$0.00**.

---

## Why

`da3-standard` currently subsamples 500 extracted frames down to **80** before
DA3 sees them. Everything past that is discarded. The cap exists because DA3's
multi-view attention is quadratic and OOMs above ~100 frames on a 22 GB L4.

DA3-Streaming removes the cap by processing overlapping windows and stitching
them with least-squares alignment plus loop closure. Upstream reports 8.51 FPS on
an A100 across 11,373 frames.

**Second benefit, and arguably the bigger one:** it natively resolves seams
between separate captures. That is the "final pass that fixes the gaps between
runs" we were going to hand-roll, and it also makes multi-video scenes tractable.

---

## What it actually is

**It replaces the SfM stage. It does not replace `--gs-only`.**

Outputs are poses and a point cloud, not gaussians:

| File | Content |
|---|---|
| `camera_poses.txt` | flattened 4×4 **C2W** matrices, one per line |
| `intrinsic.txt` | `fx, fy, cx, cy` per frame |
| `pcd/combined_pcd.ply` | merged point cloud across all chunks |
| `loop_closures.txt` | detected loop pairs |
| `camera_poses.ply` | trajectory visualisation |
| `results_output/` | per-frame depth + confidence `.npz` |

So this feeds **splatfacto** — it upgrades `da3-standard`, the path that already
produces the quality we want. `da3-draft` stays the 76-second preview and is
untouched by this work.

---

## Integration surface

Five distinct pieces of work. None is hard; the count is the risk.

### 1. Vendoring — `da3_streaming/` is not pip-installed

`pyproject.toml` declares `packages = ["src/depth_anything_3"]`. The streaming
code lives in a top-level `da3_streaming/` directory that pip never sees:

```
da3_streaming/
├── da3_streaming.py       # CLI entrypoint
├── npz_output_process.py
├── configs/               # base_config.yaml
├── fastloop/              # loop closure
├── loop_utils/
├── scripts/
└── requirements.txt       # its own deps
```

Our image installs DA3 with `pip install git+...`, so **streaming will not be
present**. The Dockerfile needs to clone and `COPY` that tree to `/opt/vw/da3_streaming`.

### 2. Extra dependencies

`da3_streaming/requirements.txt` adds seven packages, of which three are
non-trivial in our image:

| Package | Status | Note |
|---|---|---|
| `faiss-gpu` | ✅ **deferrable** | Needed only for loop closure (phase 5). `faiss-cpu` satisfies the module-scope import for phases 1–4 — verified locally |
| `numba` | ✅ | installed clean; JIT warm-up is opt-in via config |
| `pypose` | ✅ | installed clean, no torch conflict |
| `pandas`, `prettytable`, `einops`, `safetensors` | ✅ | trivial or already present |

**Plus 7 undeclared packages and ~11 transitive ones** — see
[Phase 1 findings](#phase-1-findings-measured-not-predicted). Install every DA3
dep with `--no-deps`: `e3nn` declares `torch>=2.2.0` and will silently upgrade
torch 2.1.2+cu118 to 2.13.0+**cpu**, killing CUDA. That is benchmark-doc bug #2
recurring; it bit again during phase 1.

### 3. Weights must be on local disk

The config points at **local files**, not HF model IDs:

```yaml
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'
```

Our entrypoint uses `DepthAnything3.from_pretrained(model_id)` and lets HF cache
it. Streaming needs the files materialised. Two options: bake them into the
image (+5.4 GB layer, but no per-job download) or `hf_hub_download` at job start
into `./weights/`. **Prefer the latter** — the current image already pulls the
model per job in ~30 s at 279 MB/s, and baking 5.4 GB into every layer pull is
worse.

`dino_salad.ckpt` (DINO-SALAD place recognition, for loop closure) is a
**separate model we do not currently fetch at all**. Another argument for
starting with `loop_enable: False`.

### 4. Entrypoint mode

New `--stream-sfm` mode in `da3_entrypoint.py`, parallel to the existing
`--sfm-only`:

```
frames.zip
  └─► resize to target resolution        (see Config)
  └─► python /opt/vw/da3_streaming/da3_streaming.py
        --image_dir <resized> --config <generated> --output_dir <work>
  └─► convert outputs -> transforms.json + sparse_pc.ply
  └─► filter (quantile + AABB, reuse splat_filter)
  └─► processed_min.zip                  (unchanged contract downstream)
```

The output contract stays `processed_min.zip`, so **Job B / splatfacto needs no
changes at all** — this drops in behind the existing split-job plumbing.

### 5. Pose converter ✅ built and validated

`vaultwares_studio/streaming_convert.py`. The existing `da3_to_transforms()`
takes DA3's `(N,3,4)` **w2c** extrinsics and inverts them; streaming already
emits **C2W**, so this module does *not* invert, and applies the OpenCV→OpenGL
flip. Both choices were the ones in question, and both are now confirmed
empirically — see [Phase 2 findings](#phase-2-findings-pose-convention-validated).

---

## Config (corrected)

The config we sketched does not match the real schema. Corrections:

| Sketched | Reality |
|---|---|
| `chunk_size: 65` (top level) | key is `Model.chunk_size`; **default is 120** |
| `overlap: 22` | key is `Model.overlap`; **default is 60** (50%) |
| `input_width: 672` | ❌ **no such key** — resolution is set by pre-resizing input images |
| `input_height: 378` | ❌ **no such key** — same |
| `precision: "bfloat16"` | ❌ **no such key** in `base_config.yaml` |

Resolution being controlled by pre-resizing is consistent with our workflow
step 1, so that part holds — it just isn't a config knob.

### The VRAM problem with 65 @ 672×378

Benchmarks (24 GB card):

| Chunk | Overlap | Resolution | Peak VRAM |
|---|---|---|---|
| 30 | 15 | ~504×378 | ~11.5 GB* |
| 30 | 15 | ~504×378 | ~18.7 GB |
| 60 | 30 | ~504×378 | ~21.2 GB |

<sub>*the 11.5 GB row is at ~504×154, not ~504×378</sub>

60 @ 504×378 already sits at **88% of a 24 GB card**. The proposed 65 @ 672×378
is larger on both axes. At patch-14:

- 504×378 → 36×27 = **972** patches/frame → 60 × 972 = **58,320** tokens
- 672×378 → 48×27 = **1,296** patches/frame → 65 × 1,296 = **84,240** tokens

That is **1.44× the tokens**. Linear scaling lands ~30 GB; attention is
quadratic in the worst case. Either way it does not fit in 24 GB.

**Starting config** — benchmarked numbers only, upstream's 50% overlap:

```yaml
Model:
  chunk_size: 80          # 0.99x the benchmarked token count at 504x280
  overlap: 40             # upstream's 50% ratio; do not tune until it works
  loop_enable: False      # phase 1: avoid faiss-gpu + SALAD weights
  align_lib: 'triton'     # triton 2.1.0 already present via torch 2.1.2
  align_method: 'sim3'
  depth_threshold: 15.0   # upstream's own depth truncation
  Pointcloud_Save:
    sample_ratio: 0.015   # 1.5% — check this is enough to seed splatfacto
```

Input frames pre-resized to **504×280** (36×20 patches) — aspect-correct for
16:9, and cheaper per frame than the KITTI-shaped benchmark resolution.

Flavor: `a10g-small` (24 GB) to match the benchmark. If we want 672×378 later,
move to `l40sx1` (48 GB) rather than squeezing.

### Two upstream settings that overlap with our filter work

- `depth_threshold: 15.0` — upstream's own depth truncation, i.e. step 1 of our
  filter chain, already built in.
- `Pointcloud_Save.conf_threshold_coef: 0.75` — confidence filtering at
  `mean(conf) * 0.75`.

So the sanitisation we need on top is mostly the **AABB mask**, which
`splat_filter.sanitize_splat` already does. Measure before adding SOR: the
quantile+AABB pass collapsed the draft's extent 400× on its own, and SOR is
O(n log n) on millions of points.

---

## Phasing

Each phase ends somewhere useful, so we can stop if the value isn't there.

**Phase 1 — does it run at all?** ✅ **DONE, $0.00** *(local RTX 3060, 2026-08-03)*
Ran 60 frames end-to-end. All blockers found and recorded above. No GPU spend.

**Phase 2 — are the poses right?** ✅ **DONE, $0.00** *(local, 2026-08-03)*
Converter written (`vaultwares_studio/streaming_convert.py`) and validated
against the known-good run. Both conventions confirmed. See below.

**Phase 3 — full 500 frames.** *(~$0.20)*
All 500 through streaming, then `processed_min.zip`. Compare pose count and point
cloud coverage against the 80-frame baseline. **This is where the actual value
gets proven** — if 500 posed frames don't beat 80, stop here.

**Phase 4 — train on it.** *(~$0.35)*
Feed Job B unchanged. Compare against the 1.84M-gaussian benchmark. Only worth
running if phase 3 shows materially better coverage.

**Phase 5 — loop closure.** *(cost TBD)*
Add `faiss-gpu` + SALAD weights, `loop_enable: True`. This is what makes
multi-video seam-free, so it matters — but it is also the piece most likely to
fight the image, which is why it is last.

Cumulative through phase 4: **~$0.75**.

---

## Phase 1 findings (measured, not predicted)

Ran on the local RTX 3060, 60 frames @ 504×154, chunk 30 / overlap 15,
`loop_enable: False`. It took **five attempts** to get a clean run; every failure
is a real integration blocker that would otherwise have been found on paid GPU
time.

### Undeclared dependencies

`da3_streaming/requirements.txt` lists 7 packages. It is missing **7 more**:

```
matplotlib  rich  scipy  scikit-learn  tqdm  trimesh  pyyaml
```

Plus everything `depth_anything_3` itself needs when installed `--no-deps`:
`huggingface_hub`, `opencv-python`, `imageio`, `omegaconf`, `plyfile`, `addict`,
`pycolmap`, `moviepy` + `proglog` + `imageio_ffmpeg` + `decorator`.

### `loop_utils/salad` is a git submodule

`.gitmodules` points it at `serizba/salad`. A plain clone (or `--depth 1` without
`--recurse-submodules`) leaves the directory **empty** and the import fails.
Vendoring must use:

```bash
git clone --recurse-submodules https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

### Two config flags do not guard their imports

This answers open question 1, and adds a second instance of the same pattern:

| Config | Expectation | Reality |
|---|---|---|
| `loop_enable: False` | avoids faiss | ❌ `da3_streaming.py:34` imports `LoopDetector` at module scope, which does `import faiss` at module scope |
| `align_lib: 'torch'` | avoids triton | ❌ `sim3utils.py:23` imports `alignment_triton` unconditionally, which does `import triton` at module scope |

**Both must be importable regardless of config.** Consequences:

- **faiss**: `faiss-cpu` satisfies the import and has Windows wheels. Since loop
  closure is disabled we never call it, so `faiss-gpu` is *not* needed for
  phases 1–4. This substantially de-risks the container: `faiss-gpu` was the
  dependency most likely to sink this, and it turns out to be deferrable.
- **triton**: the Linux container already has triton 2.1.0 via torch 2.1.2, so
  **the container is unaffected**. Windows has no triton wheel, so local runs
  need an import-only shim (`da3-streaming-run/shims/triton/`). Only `jit` and
  `language.constexpr` are needed — 4 decorators and some annotations.

`sim3solve` (C++ SIM3 optimiser) is genuinely optional: it prints
`Sim3solve of C++ Version failed, Will using Python Version.` and continues.

### Resolution: the benchmark numbers are KITTI-shaped

The benchmark rows are 504×154 (3.27:1) and 504×378 (1.33:1). Our footage is
1920×1080 (**1.778**). 504×154 matches KITTI's aspect — those numbers were chosen
for KITTI, not 16:9 video. **504×378 would squash our footage.**

At patch-14, aspect-correct options against the benchmark's token count
(chunk 60 × 972 patches = 58,320 tokens @ 21.2 GB of 24 GB):

| Config | Patches/frame | Aspect | Tokens | vs benchmark |
|---|---|---|---|---|
| 60 @ 504×378 *(benchmarked)* | 972 | 1.333 ❌ | 58,320 | 1.00× |
| 60 @ 672×378 *(sketched)* | 1296 | 1.778 ✅ | 77,760 | **1.33× — won't fit** |
| 60 @ 504×280 | 720 | 1.800 ✅ | 43,200 | 0.74× |
| **80 @ 504×280** | 720 | 1.800 ✅ | 57,600 | **0.99×** |

**504×280 is the recommendation** — 1.25% off 16:9, and cheaper than the
benchmark. At chunk 80 it fits 33% more frames per window than the benchmarked
config *at the same memory*, with correct geometry.

### Streaming's output needs no aggressive filtering

Merged point cloud from the smoke run vs the `da3-draft` direct-3DGS output:

| | `da3-draft` | DA3-Streaming |
|---|---|---|
| radius p95 | 1.4 | 0.72 |
| radius max | **581** | **1.04** |
| max / p95 | **416×** | **1.44×** |

`depth_threshold: 15.0` and `conf_threshold_coef: 0.75` already remove the tail.
So `splat_filter`'s AABB mask is **not needed on this path** — a good result, and
it means SOR almost certainly isn't either.

### Output sanity

60 frames → 60 poses + 60 intrinsic rows. All 4×4 matrices have bottom row
`[0,0,0,1]` and rotations with `det = 1.0`. Trajectory is smooth (median step
0.0175, max 0.0638 — no jumps). 55,897 coloured points, all finite.

---

## Phase 2 findings: pose convention validated

`vaultwares_studio/streaming_convert.py` converts streaming's flat output into
`transforms.json` + `sparse_pc.ply`. Validated by running streaming on **exactly
the 80 frames** the known-good `da3-standard` run used, then Umeyama-aligning
(similarity: scale + rotation + translation) the two trajectories.

The OpenCV→OpenGL flip negates the Y and Z rotation columns but leaves the
translation column untouched, which makes the two checks independent:
**position** validates C2W-vs-W2C, **orientation** validates the flip.

| Check | Correct | Counterfactual |
|---|---|---|
| Orientation (median) | **9.07°** | **178.91°** without the flip |
| Position RMSE | **11.9%** of extent | **26.0%** wrongly inverting to W2C |

The 178.91° is decisive: a near-perfect inversion, exactly what dropping the flip
predicts. Both conventions are confirmed:

- **Do not invert.** Streaming already emits camera-to-world.
- **Do apply** `diag(1, -1, -1, 1)` on the right.

Both failure modes are pinned by `tests/test_streaming_convert.py` (16 tests) and
both were mutation-tested — reintroducing either makes the suite fail.

### The residual error is tail drift, not a converter bug

Orientation error against the reference, by position in the sequence:

| Frames | Median | Max |
|---|---|---|
| 0–15 | 12.96° | 56.78° |
| 16–31 | **5.11°** | 16.10° |
| 32–47 | **6.61°** | 12.86° |
| 48–63 | **5.22°** | 18.92° |
| **64–79** | **36.42°** | 113.32° |

The well-conditioned middle chunks agree to **5–7°** — two independent DA3
configurations at different resolutions landing within a few degrees of each
other. The error is concentrated in the final chunk: accumulated SIM3 drift with
`loop_enable: False`.

Two mitigating factors for real runs, and one warning:

- These 80 frames are **sparsely sampled** (~1.6 s apart), which is adversarial
  for a method that assumes sequential video with small baselines. Production
  runs feed consecutive frames.
- **Loop closure is designed for exactly this** and was disabled. This moves
  phase 5 from optional to likely-required for long sequences.
- ⚠️ Do not read the 11.9% / 9.07° as an accuracy figure. The reference is
  another DA3 estimate, not ground truth. These numbers validate *conventions*;
  they say nothing about absolute accuracy.

---

## Open questions

1. ~~Does `loop_enable: False` remove the faiss dependency?~~ **Answered: no**,
   but `faiss-cpu` satisfies it and `faiss-gpu` is deferrable to phase 5.
2. **Is `sample_ratio: 0.015` enough** to seed splatfacto? 60 frames gave 55,897
   points. Extrapolating to 500 frames is ~465k, which is comparable to the
   ~500k `sparse_pc.ply` the current path produces — so probably fine, but worth
   confirming at full scale.
3. **Scale is still arbitrary.** Streaming does not make depth metric. Real-world
   units need an external reference (known object size, GPS baseline, IMU) and
   are out of scope here.
4. ~~Pose convention~~ **Validated** — see phase 2 findings.
5. **Does loop closure fix the tail drift?** The last chunk degrades badly with
   `loop_enable: False` on sparsely-sampled frames. Loop closure exists for
   exactly this, which raises the priority of phase 5 from "nice to have" to
   "probably required for long sequences".

---

## Agent-Readable Summary

```yaml
plan: da3-streaming-integration
date: 2026-08-01
status: PHASE_2_COMPLETE
replaces: "da3_entrypoint --max-sfm-frames 80 cap on the SfM path"
does_not_replace: "--gs-only / da3-draft (that path stays as-is)"
feeds: splatfacto via unchanged processed_min.zip contract

upstream_facts:
  streaming_is_pip_installed: false
  streaming_path_in_repo: da3_streaming/
  cli: "da3_streaming.py --image_dir X --config Y --output_dir Z"
  outputs: [camera_poses.txt, intrinsic.txt, "pcd/combined_pcd.ply", loop_closures.txt, camera_poses.ply]
  pose_convention: "C2W 4x4 flattened (existing da3_to_transforms expects w2c — do not invert)"
  extra_deps: [faiss-gpu, pandas, prettytable, einops, safetensors, numba, pypose]
  weights_are_local_paths: true
  salad_weights_needed_for_loop: true

config_corrections:
  input_width: "NO SUCH KEY — resolution set by pre-resizing frames"
  input_height: "NO SUCH KEY"
  precision: "NO SUCH KEY in base_config.yaml"
  chunk_size: "nested under Model:, upstream default 120 not 65"
  overlap: "nested under Model:, upstream default 60 (50%) not 22 (33%)"

vram_analysis:
  benchmarked_max: {chunk: 60, res: 504x378, peak_gb: 21.2, card_gb: 24}
  proposed_config: {chunk: 65, res: 672x378}
  token_ratio_vs_benchmark: 1.44
  verdict: "will not fit 24GB; start from benchmarked 60/30 @ 504x378"

starting_config: {chunk_size: 80, overlap: 40, loop_enable: false, align_lib: triton, resolution: 504x280, flavor: a10g-small}
highest_risk: "RESOLVED — pose convention validated 2026-08-03; new top risk is end-of-sequence chunk drift without loop closure"
phases_cost_usd: {phase1: 0.00, phase2: 0.00, phase3: 0.20, phase4: 0.35}
phases_done: [1, 2]
```
