<!-- v1.0.0 -->
# DA3 Direct-3DGS (`--gs-only`) — Why It Never Ran, and What It Took to Fix

> **Thu, 30 Jul 2026 12:13 PM EDT** — Investigation into why the `da3-draft` and
> `da3-incremental` presets had never produced output. Companion to
> `da3-standard-benchmark-20260713.md`, which covers the path that *did* work.
> Formatted for both human and agent consumption.

---

## Summary

`da3-draft` (DA3-GIANT with `infer_gs=True`, no splatfacto training) had **never
run successfully a single time** since it was written. Not "crashed repeatedly" —
never executed. Three independent failures were stacked on top of each other,
each one hidden by the layer above it, and each only reachable after fixing the
previous one.

| Layer | Failure | Hidden by |
|---|---|---|
| 1 | gsplat CUDA extension never built | `\|\| echo "...skipped"` in the Dockerfile |
| 2 | No `nvcc` in the base image | Layer 1's silent fallback |
| 2b | No `wheel` / `ninja` → `invalid command 'bdist_wheel'` | Only surfaced once layer 2 was fixed |
| 3 | `e3nn` unimportable → `NameError` mid-inference | DA3's own bare `try/except` |

`da3-standard` and `da3-high` were never affected — they use DA3 for pose+depth
only and never touch the Gaussian head, so none of these paths execute.

**Total cost to find all of it: $0.03** (one 152-second GPU job).

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

## The diagnostic job

| Metric | Value |
|---|---|
| Job ID | `6a6b65ef23ed89c748ec7c94` |
| Flavor | `l4x1` |
| Duration | 151.7 s |
| **Actual cost** | **$0.03** |
| Outcome | ERROR — `NameError: matrix_to_angles` |
| Frames | 500 extracted → 80 subsampled for DA3 |
| Model | `depth-anything/DA3-GIANT-1.1` (5.42 GB) |

It reached DA3 inference, loaded a 5.4 GB model, and failed on the very last
step before producing gaussians. Cheapest possible way to learn all of that —
and none of it was reachable from static inspection, because the bug lives in a
dependency's exception handler.

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
| `da3-draft` end-to-end | ⏳ Never completed — blocked on GPU capacity |
| `da3-incremental` | ⏳ Untested (depends on same `--gs-only` path) |
| `da3-standard` / `da3-high` | ✅ Unaffected, still on proven `vw-studio-da3` |
| Local Windows gsplat build | ❌ Abandoned (MSVC/CUDA STL assert) |

`da3-draft` and `da3-incremental` point at `vw-studio-da3-gs` via
`sfm_image_override` so the proven `vw-studio-da3` image is untouched. **Repoint
them at `vw-studio-da3` once a green run exists** — the two Spaces then differ
only in the nvcc/e3nn additions, which are safe for the pose+depth presets too.

To resume: `.venv\Scripts\python.exe tools\retry_da3_draft_recon.py local-run-20260730-105436`
(frames already extracted; reruns the reconstruction stage only).

---

## Agent-Readable Summary

```yaml
investigation: da3-direct-3dgs-gs-only
date: 2026-07-30
status: BLOCKED_ON_GPU_CAPACITY
preset_affected: [da3-draft, da3-incremental]
preset_unaffected: [da3-standard, da3-high]
never_ran_before: true
diagnostic_cost_usd: 0.03
diagnostic_job_id: 6a6b65ef23ed89c748ec7c94
diagnostic_duration_s: 151.7

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

image: hf.co/spaces/clopeux/vw-studio-da3-gs
gsplat_version: 1.5.2
gsplat_commit: 0b4dddf04cb687367602c01196913cde6a743d70
gsplat_build_time_s: 1510
e3nn_version: 0.6.0
da3_model: depth-anything/DA3-GIANT-1.1
da3_model_size_gb: 5.42

zerogpu_available_for_jobs: false
flavor_fallback_added: true
flavor_fallback_default: [t4-small, a10g-small]
flavor_fallback_success_path_verified: false

local_windows_build: abandoned
local_windows_blocker: "MSVC 14.51 yvals_core.h STL1002 static_assert requires CUDA 13.2+; -allow-unsupported-compiler does not bypass it"

resume_command: "tools/retry_da3_draft_recon.py local-run-20260730-105436"
```
