<!-- v1.0.0 -->
# DA3 Direct-3DGS (`--gs-only`) — Why It Never Ran, and What It Took to Fix

> **Thu, 30 Jul 2026 5:05 PM EDT** — Investigation into why the `da3-draft` and
> `da3-incremental` presets had never produced output, through to the first
> green run. Companion to `da3-standard-benchmark-20260713.md`, which covers the
> pose+depth path. Formatted for both human and agent consumption.

---

## Summary

**Status: RESOLVED.** `da3-draft` now completes end-to-end — 4,503,479 gaussians
in 76 s for ~$0.02.

It had **never run successfully a single time** since it was written. Not
"crashed repeatedly" — never executed. Five independent failures were stacked on
top of each other, each hidden by the one above it, each only reachable after
fixing the previous one.

| Layer | Failure | Hidden by |
|---|---|---|
| 1 | gsplat CUDA extension never built | `\|\| echo "...skipped"` in the Dockerfile |
| 2 | No `nvcc` in the base image | Layer 1's silent fallback |
| 2b | No `wheel` / `ninja` → `invalid command 'bdist_wheel'` | Only surfaced once layer 2 was fixed |
| 3 | `e3nn` unimportable → `NameError` mid-inference | DA3's own bare `try/except` |
| 4 | CUDA OOM in the Gaussian head | Only reachable once layer 3 was fixed |
| 5 | Exporter renders a preview video and dies on `fps=None` | Treated as fatal despite the PLY already being written |

`da3-standard` and `da3-high` were never affected — they use DA3 for pose+depth
only and never touch the Gaussian head, so none of these paths execute.

**Total cost to find and fix all of it: ~$0.09** across five GPU jobs, the
longest of which ran 152 seconds.

### The shape of this bug

Every layer here failed *silently or misleadingly*. A Dockerfile `|| echo`, a
base image quietly lacking a compiler, a dependency's bare `except` converting an
`ImportError` into a `NameError` three frames away, and finally a completed 4.5M
gaussian run being discarded because a preview clip we never asked for couldn't
find an fps. None of it was reachable by reading code — each layer only became
visible by running the thing and reading what actually came back.

---

## Layer 1 — The silent Dockerfile fallback

`docker/da3/Dockerfile` installed gsplat like this:

```dockerfile
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf..." \
    || echo "[Dockerfile] gsplat custom commit build skipped (nvcc not available). --gs-only mode will not work."
```

The `|| echo` turned a hard build failure into a successful image build that
silently lacked the one component `--gs-only` depends on. The image reported
healthy; the failure only appeared at job runtime, at $0.80/hr.

Worse, the build logs were useless for diagnosing it: every interesting layer
showed `CACHED`, so the echo never even reappeared in later builds.

**Fix:** remove the `|| echo` entirely and add a post-install assertion.

```dockerfile
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf..."
RUN python3 -c "import gsplat; print('[Dockerfile] gsplat OK:', gsplat.__version__)"
```

> **Rule of thumb this established:** an optional-dependency fallback in a
> Dockerfile is only acceptable if something downstream *checks*. Otherwise it
> converts a build-time error into a runtime one and moves the cost from free to
> billed.

---

## Layer 2 — nerfstudio's base image ships no compiler

Confirmed from nerfstudio's own Dockerfile: it is a multi-stage build whose
**final** stage is

```
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime
```

`-runtime`, not `-devel`. The builder stage has `nvcc` (it compiles COLMAP,
GLOMAP, tiny-cuda-nn) but the shipped image deliberately drops it to save size.
So *every* source build of a CUDA extension in this image was always going to
fail.

**Fix** — pull the CUDA 11.8 compiler from the apt repo already configured in
the base, version-matched to torch 2.1.2+cu118:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-11-8 cuda-cudart-dev-11-8 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=/usr/local/cuda-11.8/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
RUN nvcc --version
```

`TORCH_CUDA_ARCH_LIST` is required because no GPU is present at build time, so
the arch list can't be probed. The values cover the flavors we actually run on:
T4 (7.5), A100 (8.0), A10G (8.6), L4 (8.9).

Also fixed in passing: the image installed `nvidia-cuda-nvrtc-**cu12**` on a
CUDA **11.8** runtime. Corrected to `-cu11`.

### 2b — missing `wheel` and `ninja`

With `nvcc` present, the build got further and hit:

```
error: invalid command 'bdist_wheel'
```

`wheel` registers the `bdist_wheel` setuptools command, which gsplat's
`setup.py` needs even for `--no-build-isolation` metadata prep. `ninja` is
optional but turns a serial distutils compile into a parallel one.

```dockerfile
RUN python3 -m pip install --no-cache-dir wheel ninja
```

---

## Layer 3 — `e3nn` swallowed by DA3's own `try/except`

With gsplat finally built and verified (`gsplat OK: 1.5.2`), the first real GPU
run got all the way through model load and into inference, then died:

```
[da3-recon] Running DA3 inference on 80 images
[INFO ] Processed Images Done taking 0.534s. Shape: torch.Size([80, 3, 280, 504])
[INFO ] Selecting reference view using strategy: saddle_balanced
Traceback (most recent call last):
  ...
  File ".../depth_anything_3/model/gs_adapter.py", line 157, in forward
    gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
  File ".../depth_anything_3/utils/sh_helpers.py", line 76, in rotate_sh
    alpha, beta, gamma = matrix_to_angles(permuted_rotations_so3)
NameError: name 'matrix_to_angles' is not defined
```

A `NameError` for a function that plainly *is* imported at the top of
`sh_helpers.py`:

```python
from e3nn.o3 import matrix_to_angles, wigner_D
```

…except that import sits inside a bare `try/except` in DA3's own source. When it
fails, DA3 logs a warning and continues:

```
[WARN ] Dependency 'e3nn' not found. Required for rotating the camera space SH coeff
```

That warning appeared **200 log lines before** the crash, right after model load,
and reads like a benign optional-feature notice. The names are never bound, and
the real `ImportError` surfaces three call frames later as a `NameError` in an
unrelated-looking function — only on the `infer_gs=True` path.

### Root cause of the root cause

We install DA3's deps with `--no-deps` (deliberately — DA3 declares `xformers`
and `torch>=2` and would otherwise drag torch 2.1 → 2.13, which is
[benchmark doc bug #2](da3-standard-benchmark-20260713.md)). But `e3nn` needs two
packages *at import time*, not merely declared:

| Package | Needed by |
|---|---|
| `opt_einsum_fx` | `e3nn/util/jit.py` line 12, on module load |
| `scipy` | `e3nn` runtime; absent from the nerfstudio image |

`--no-deps` dropped both, so `import e3nn.o3` raised `ModuleNotFoundError:
No module named 'opt_einsum_fx'`, which DA3 caught and hid.

**Fix** — install the two transitive deps explicitly (with deps, they're small
and pull nothing dangerous), then assert the exact import DA3 needs:

```dockerfile
RUN python3 -m pip install --no-cache-dir opt_einsum_fx scipy
RUN python3 -c "from e3nn.o3 import matrix_to_angles, wigner_D; print('[Dockerfile] e3nn.o3 OK')"
```

Reproduced locally in seconds once the failure mode was understood — the local
venv showed the identical `ModuleNotFoundError`, confirming it wasn't
container-specific.

---

## Layer 4 — CUDA OOM in the Gaussian head

With e3nn fixed, inference ran deep into the Gaussian head and then:

```
torch.cuda.OutOfMemoryError: Tried to allocate 552.00 MiB.
GPU 0 has a total capacty of 14.74 GiB of which 558.19 MiB is free.
  depth_anything_3/model/gsdpt.py line 109, in _forward_impl
    fused = fused + self.images_merger(images)
```

Self-inflicted. `t4-small` (16 GB) had been chosen on the reasoning that
DA3-GIANT's 5.4 GB of weights don't need an L4's 22 GB — **wrong axis**. Peak
memory is activations, not weights, and `infer_gs=True` stacks a Gaussian head
on top of multi-view attention that the entrypoint's own comment already
describes as OOM-prone above ~100 frames on 22 GB.

Two changes, since either alone was marginal:

| Change | Rationale |
|---|---|
| `sfm_flavor=["a10g-small", "l4x1"]` | 24 GB floor; t4 dropped entirely |
| `da3_max_sfm_frames=40` | Halved from the pose+depth default |

`--max-sfm-frames` already existed in the entrypoint but **`reconstruction.py`
never passed it**, so every run silently used the default 80 that was tuned for
SfM *without* the Gaussian head. It is now a preset field. `da3-standard` keeps
80 — the change is scoped to the direct-3DGS presets.

---

## Layer 5 — a preview video killing a successful run

40 frames on 24 GB cleared the OOM, inference completed, and the job still
reported ERROR:

```
File ".../depth_anything_3/utils/export/gs.py", line 147, in export_to_gs_video
  clip.write_videofile(
...
TypeError: must be real number, not NoneType
```

Buried in the same log: `[vw-stage] outputs uploaded`. DA3's exporter writes the
PLY **and then** renders a preview video; the video step calls moviepy with an
fps we never supply. The 306 MB / 4.5M gaussian PLY was already on disk. The run
had succeeded and was being thrown away over a clip nobody wanted.

**Fix:** treat the export as non-fatal when a PLY landed.

```python
try:
    prediction = da3_inference(..., export_dir=out_dir, export_format="gs_ply")
except Exception as exc:
    if not list(out_dir.rglob("*.ply")):
        return fail(out_dir, "da3_inference_failed", str(exc))
    log(f"DA3 export raised after writing the PLY, continuing: {exc}")
```

> Note: in the *installed* DA3 build (the layer is cached from before upstream
> `main` moved), `export_format="gs_ply"` reaches `export_to_gs_video` as well as
> `export_to_gs_ply`. Not worth further archaeology — tolerating the exporter is
> more robust than depending on its dispatch, and removes moviepy from the
> critical path.

---

## The green run

| Metric | Value |
|---|---|
| Job ID | `6a6bb9c3b36a6516e96a32f0` |
| Flavor | `a10g-small` |
| **Wall time** | **76 s** |
| DA3 inference | 73.7 s |
| **Cost** | **~$0.02** |
| Frames | 500 extracted → 40 for DA3 |
| **Gaussians** | **4,503,479** |
| Gravity alignment | 25.1° to +Y |

Artifacts: `splat.ply` 306 MB, `cloud.ply` 252 MB, `cloud.splat` 144 MB,
`cloud_preview.ply` 3 MB, `cloud.usda` 990 MB.

### Quality, honestly

Structurally correct and clearly the right scene — paths, beds and foliage all
read — but **soft**, nothing like the crisp geometry of trained splatfacto. That
is inherent, not a bug: this is a feed-forward prediction with zero per-scene
optimisation.

| | `da3-standard` | `da3-draft` |
|---|---|---|
| Gaussians | 1.84M | 4.5M |
| Wall time | ~35 min | ~76 s |
| Cost | $0.36 | ~$0.02 |
| Quality | sharp | soft, structurally correct |

Which is exactly the tradeoff the preset advertises. Use it for preview and
layout planning; use `da3-standard` when the geometry has to hold up.

### Two data-quality issues worth fixing

1. **`opacity` max is `inf`.** Harmless through a sigmoid when rendering, but any
   `mean`/`sum` over opacity yields `nan` — which `gaussian_merge.py`'s quality
   scoring does. Clamp before trusting `da3-incremental`.
2. **Outlier tail.** 96% of gaussians sit within radius 1.58 while the full bbox
   is ±600, and skewness came out at 32.7 (vs 1.07 on the trained run). This
   defeats the viewer's auto-framing — the screenshots for this doc had to be
   framed off the interquartile core. A percentile clip in
   `convert_splat_outputs` would fix framing for every DA3 draft, and would
   probably improve the gravity-alignment estimate too.

---

## Side quest: local Windows GPU testing (abandoned)

The intent was to rehearse the container toolchain on the local RTX 3060 before
spending on HF. It got 90% of the way and then hit a wall worth recording so
nobody retries it blindly.

Isolated env at `D:\3D Reconstruction\da3-gsplat\.venv` — plain `venv` via the
official `py install` manager (**not** uv, deliberately, to keep it out of the
shared uv Python pool), Python 3.11.9, `torch==2.1.2+cu118`, `numpy==1.26.4`
(torch 2.1 predates the numpy 2 ABI).

| Attempt | Result |
|---|---|
| `nvidia-cuda-nvcc-cu11` from PyPI | Windows wheel ships `ptxas.exe` but **no `nvcc.exe`** |
| System CUDA 12.0 + torch cu118 | torch's `_check_cuda_version` hard-errors on major mismatch |
| Real CUDA 11.8 toolkit, side-by-side with 12.0 | `nvcc` works; `cl.exe` found via `vcvarsall.bat` |
| gsplat build with CUDA 11.8 + MSVC 14.51 | **Blocked** (below) |

The blocker is not CUDA's:

```
yvals_core.h(911): error: static assertion failed with
  "error STL1002: Unexpected compiler version, expected CUDA 13.2 or newer."
```

That is **Microsoft's own STL** refusing CUDA 11.8 with this MSVC generation
(14.51, VS 18). `nvcc`'s complementary check —

```
host_config.h(153): fatal error C1189: #error: -- unsupported Microsoft Visual
Studio version! Only the versions between 2017 and 2022 (inclusive) are supported!
```

— *can* be bypassed with `NVCC_APPEND_FLAGS=-allow-unsupported-compiler`, but
that flag does nothing for Microsoft's `static_assert`. Unblocking this requires
installing an older MSVC toolset (v143) side-by-side via the VS Installer.

**Verdict:** not worth it. The container build already proves the toolchain, and
gsplat compiles there in ~25 min unattended. Local iteration would only pay off
if we were changing gsplat itself.

The CUDA 11.8 install *did* succeed and is still there
(`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`), side-by-side with
12.0 — harmless, and useful if the MSVC toolset is ever added. One gotcha: the
silent installer fails (exit 127 / no-op) when handed a long component list;
installing in small batches works.

---

## Side quest 2: GPU scheduling capacity

After the fixes landed, the rerun couldn't get a GPU at all:

```
stage: SCHEDULING   msg: Waiting for requested hardware to become available
```

`l4x1` sat there 20+ minutes; `t4-small` and `a10g-small` each refused to
schedule within 120 s. All three cancelled cleanly, no stuck jobs, no quota
issue — genuine shared-pool capacity pressure.

**ZeroGPU is not an option.** Verified against `huggingface_hub`'s actual
`JobHardware` enum — there is no `zerogpu` member. It's a Spaces/Gradio
allocation model (`@spaces.GPU`), not a Jobs flavor. The `zerogpu` entry in our
`FLAVOR_RATES_USD_PER_HOUR` table is aspirational and has never been valid for
Jobs. Valid GPU flavors:

```
t4-small  t4-medium  l4x1  l4x4  l40sx1/x4/x8
a10g-small  a10g-large  a10g-largex2/x4
a100-large  a100x4  a100x8
h200  h200x2/x4/x8   rtx-pro-6000  rtx-pro-6000x2/x4/x8
```

**Fix:** `HfJobsStageRunner.run` now accepts a *list* for `ctx.params["flavor"]`
and tries candidates in order, cancelling any that doesn't leave `SCHEDULING`
within `flavor_scheduling_timeout_seconds` (default 120 s). Cost confirmation
quotes the **priciest** candidate so the dialog never undercounts. `da3-draft`
and `da3-incremental` now use `["t4-small", "a10g-small"]` — DA3-GIANT is 5.4 GB
and does not need L4's 22 GB.

Covered by `tests/test_flavor_fallback.py` (6 tests, fully faked hub).

> ⚠️ **Unverified:** the fallback has never completed a *successful* run — it has
> only been exercised on the all-candidates-exhausted path. The success branch
> (leaves SCHEDULING → poll loop → download outputs) is covered by unit tests but
> not yet by a real job.

---

## Current state

| Item | Status |
|---|---|
| `clopeux/vw-studio-da3-gs` image | ✅ Builds; `gsplat OK: 1.5.2` + `e3nn.o3 OK` |
| `da3-draft` end-to-end | ✅ Green — 4.5M gaussians, 76 s, ~$0.02 |
| Flavor fallback success path | ✅ Verified by the green run (`a10g-small`) |
| `da3-incremental` | ⏳ Untested — shares the fixed `--gs-only` path, but the ICP merge itself has never run |
| `da3-standard` / `da3-high` | ✅ Unaffected, still on proven `vw-studio-da3` |
| Local Windows gsplat build | ❌ Abandoned (MSVC/CUDA STL assert) |

`da3-draft` and `da3-incremental` still point at `vw-studio-da3-gs` via
`sfm_image_override`. Now that a green run exists, **repoint them at
`vw-studio-da3`** — the two Spaces differ only in the nvcc/wheel/e3nn additions,
all of which are safe for the pose+depth presets too. Left as a deliberate
follow-up rather than done blind, since it means rebuilding the image the proven
`da3-standard` path depends on.

To re-run: `.venv\Scripts\python.exe tools\retry_da3_draft_recon.py local-run-20260730-105436`
(frames already extracted; reruns the reconstruction stage only).

---

## Agent-Readable Summary

```yaml
investigation: da3-direct-3dgs-gs-only
date: 2026-07-30
status: RESOLVED
preset_affected: [da3-draft, da3-incremental]
preset_unaffected: [da3-standard, da3-high]
never_ran_before: true
total_debug_cost_usd: 0.09
jobs_burned: 5

green_run:
  job_id: 6a6bb9c3b36a6516e96a32f0
  flavor: a10g-small
  wall_time_s: 76
  da3_inference_s: 73.7
  cost_usd: 0.02
  gaussians: 4503479
  frames_extracted: 500
  frames_to_da3: 40
  gravity_rotation_deg: 25.1
  artifacts: [splat.ply, cloud.ply, cloud.splat, cloud_preview.ply, cloud.usda]

root_causes:
  - layer: 1
    what: "Dockerfile `|| echo` swallowed gsplat build failure"
    fix: "removed fallback; added `import gsplat` assertion"
  - layer: 2
    what: "nerfstudio base image final stage is nvidia/cuda:11.8.0-runtime (no nvcc)"
    fix: "apt install cuda-nvcc-11-8 cuda-cudart-dev-11-8; CUDA_HOME; TORCH_CUDA_ARCH_LIST"
  - layer: 2b
    what: "missing wheel/ninja -> invalid command 'bdist_wheel'"
    fix: "pip install wheel ninja"
  - layer: 3
    what: "e3nn unimportable (missing opt_einsum_fx + scipy from --no-deps); DA3 swallows the ImportError in a bare try/except, surfacing as NameError: matrix_to_angles inside rotate_sh()"
    fix: "pip install opt_einsum_fx scipy; assert `from e3nn.o3 import matrix_to_angles, wigner_D`"
  - layer: 4
    what: "CUDA OOM in the Gaussian head on t4-small (16GB) at 80 frames; flavor was chosen from model weight size rather than activation peak"
    fix: "24GB floor (a10g-small, l4x1) + da3_max_sfm_frames=40; --max-sfm-frames existed but reconstruction.py never passed it"
  - layer: 5
    what: "DA3's exporter writes the PLY then renders a preview video, dying on moviepy fps=None; entrypoint treated the whole inference call as fatal and discarded a completed 4.5M gaussian run"
    fix: "export failure is non-fatal when a PLY was written"

image: hf.co/spaces/clopeux/vw-studio-da3-gs
gsplat_version: 1.5.2
gsplat_commit: 0b4dddf04cb687367602c01196913cde6a743d70
gsplat_build_time_s: 1510
e3nn_version: 0.6.0
da3_model: depth-anything/DA3-GIANT-1.1
da3_model_size_gb: 5.42

zerogpu_available_for_jobs: false
flavor_fallback_added: true
flavor_fallback_default: [a10g-small, l4x1]
flavor_fallback_success_path_verified: true

known_issues:
  - "opacity max is +inf; sigmoid-safe for rendering but nan-poisons any mean/sum, incl. gaussian_merge quality scoring"
  - "outlier tail: 96% of gaussians within radius 1.58 but bbox +/-600, skewness 32.7 vs 1.07 trained; breaks viewer auto-framing"

quality_vs_trained:
  da3_draft:    {gaussians: 4503479, wall_time_s: 76,   cost_usd: 0.02, sharpness: soft}
  da3_standard: {gaussians: 1839412, wall_time_s: 2100, cost_usd: 0.36, sharpness: sharp}

local_windows_build: abandoned
local_windows_blocker: "MSVC 14.51 yvals_core.h STL1002 static_assert requires CUDA 13.2+; -allow-unsupported-compiler does not bypass it"

next_steps:
  - "repoint da3-draft/da3-incremental at vw-studio-da3 (drop the -gs override in presets.py)"
  - "clamp opacity before da3-incremental is trusted"
  - "percentile clip in convert_splat_outputs to fix framing + gravity estimate"
  - "da3-incremental still untested end-to-end"
```
