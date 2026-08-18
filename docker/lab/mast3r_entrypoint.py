"""MASt3R-SfM stage entrypoint: frames.zip -> processed_min.zip.

Drop-in replacement for docker/worker/recon_entrypoint.py's --sfm-only path.
Uses MASt3R-SfM instead of COLMAP to estimate camera poses + a sparse point
cloud. Emits the exact same processed_min.zip layout the prod worker's
--train-only path expects:

    transforms.json    nerfstudio-format cam poses + intrinsics
    sparse_pc.ply      sparse 3D points (colored) for splatfacto init

Also archives the raw MASt3R state (poses, focals, dense pts3d) under
mast3r_debug/ for post-hoc inspection.

Why bother: COLMAP starves on low-texture / grass / sky / low-parallax
captures. MASt3R's matcher is learned and handles those cases -- swapping it
in for the SfM stage gives us dense reconstructions where the COLMAP path
would silently register 3/280 images and produce garbage.

Run inside the vw-studio-recon-lab image (Docker Space), invoked by vw_stage.py
just like the COLMAP entrypoint. VW_IN / VW_OUT set by the bootstrap.

Envelope (arguments mirror recon_entrypoint.py so the launcher stays uniform):
    --sfm-only       kept for parity; this entrypoint is always SfM-only
    --downscale N    passed through to transforms.json (does not affect SfM)
    --keep-checkpoint  kept for parity; MASt3R state is always archived
    --image-size N   MASt3R input size (default 512 -- the pretrained weights
                     expect 512; going higher rarely helps and costs a lot)
    --max-images N   cap frames processed (0 = all). Useful for smoke tests.
    --retrieval K    number of retrieval neighbors per image (default 20;
                     3000 frames * 20 = 60k pairs, ~15-30 min on l4)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

# Windows/console encoding safety even in Linux containers -- huggingface_hub
# progress bars and other libs sometimes emit non-ASCII into stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def _fail(out_dir: Path, code: str, detail: str) -> int:
    print(f"[mast3r] FAILED: {code} -- {detail}", flush=True)
    (out_dir / "error.json").write_text(
        json.dumps({"code": code, "detail": detail}, indent=2), encoding="utf-8"
    )
    return 1


def _to_numpy(obj):
    """Convert tensor / list-of-tensors to numpy arrays."""
    import numpy as np
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, (list, tuple)):
        return [_to_numpy(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj
    return np.asarray(obj)


def _write_nerfstudio_transforms(
    out_path: Path,
    image_names: list[str],
    cams2world_opencv,
    focals,
    principal_points,
    width: int,
    height: int,
) -> None:
    """Write a nerfstudio-compatible transforms.json.

    MASt3R returns cam2world in the OpenCV / COLMAP convention (X right, Y down,
    Z forward). Nerfstudio's nerfstudio-data dataparser expects OpenGL / Blender
    convention (X right, Y up, Z backward). Convert by flipping columns 1, 2.
    """
    import numpy as np

    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    frames = []
    for i, name in enumerate(image_names):
        c2w_cv = np.asarray(cams2world_opencv[i], dtype=np.float32)
        if c2w_cv.shape == (3, 4):
            c2w_cv = np.vstack([c2w_cv, np.array([[0, 0, 0, 1]], dtype=np.float32)])
        c2w_gl = c2w_cv @ cv_to_gl
        frames.append({
            "file_path": f"images/{name}",
            "transform_matrix": c2w_gl.tolist(),
        })

    # focals may be [N] or [N, 1] or [N, 2] (fx, fy). Normalize to fx=fy scalar.
    focals_arr = np.asarray(focals, dtype=np.float32).reshape(len(image_names), -1)
    fl_x = float(focals_arr[:, 0].mean())
    fl_y = float(focals_arr[:, -1].mean()) if focals_arr.shape[1] >= 2 else fl_x
    # principal points: [N, 2] (cx, cy) -- use median so a single bad image doesn't skew
    pps = np.asarray(principal_points, dtype=np.float32).reshape(len(image_names), 2)
    cx = float(np.median(pps[:, 0]))
    cy = float(np.median(pps[:, 1]))
    payload = {
        "camera_model": "OPENCV",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": int(width),
        "h": int(height),
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        # nerfstudio-data reads this top-level key when load_3D_points=True to
        # seed splatfacto's initial gaussians. Without it, splatfacto silently
        # falls back to random init and prints "no point cloud found" -- which
        # burns training steps that could have been on real geometry.
        "ply_file_path": "sparse_pc.ply",
        "frames": frames,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_sparse_pc_ply(out_path: Path, points, colors) -> None:
    """Write a colored sparse point cloud PLY for splatfacto initialization."""
    import numpy as np
    from plyfile import PlyData, PlyElement

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    n = min(len(points), len(colors))
    points, colors = points[:n], colors[:n]
    if n > 0 and colors.max() <= 1.5:
        colors = (colors * 255.0).clip(0, 255)
    colors = colors.astype(np.uint8)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    verts = np.empty(n, dtype=dtype)
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["red"], verts["green"], verts["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(out_path))


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm-only", action="store_true", help="kept for parity")
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--image-size", type=int, default=512,
                        help="MASt3R input size (512 default -- weights trained for it)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="cap frames processed (0 = all)")
    parser.add_argument("--swin-window", type=int, default=5,
                        help="Sliding-window size when scene_graph is swin. K neighbors "
                             "per image (with cyclic closure) => K*N pairs total.")
    parser.add_argument(
        "--scene-graph", default="retrieval-25-10",
        help=(
            "MASt3R scene_graph. 'retrieval-Na-k' uses the retrieval model to "
            "pick pairs across the whole sequence. Falls back to 'swin-<N>' if "
            "retrieval weights aren't installed in the image."
        ),
    )
    parser.add_argument("--niter1", type=int, default=300,
                        help="sparse_global_alignment coarse iters (3D matching loss).")
    parser.add_argument("--niter2", type=int, default=300,
                        help="sparse_global_alignment fine iters (2D reproj loss). 0 skips.")
    # --retrieval kept as accepted-and-mapped for launcher parity with older invocations.
    parser.add_argument("--retrieval", type=int, default=0,
                        help="alias for --swin-window when >0 (kept for launcher parity)")
    # Accept-and-ignore flags the launcher may pass (parity with recon_entrypoint).
    parser.add_argument("--train-args", default="[]")
    parser.add_argument("--vocab-tree-num-images", type=int, default=0)
    parser.add_argument("--sift-max-image-size", type=int, default=0)
    args = parser.parse_args()

    in_dir = Path(os.environ["VW_IN"])
    out_dir = Path(os.environ["VW_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)

    work = Path("/tmp/mast3r")
    images_dir = work / "images"
    cache_dir = work / "cache"
    for d in (images_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    frames_zip = in_dir / "frames.zip"
    if not frames_zip.exists():
        return _fail(out_dir, "missing_input", "frames.zip not found in stage inputs")

    started_all = time.monotonic()
    print(f"[mast3r] unpacking {frames_zip.name} -> {images_dir}", flush=True)
    with zipfile.ZipFile(frames_zip) as archive:
        archive.extractall(images_dir)

    image_paths = sorted(
        [p for p in images_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if args.max_images > 0 and len(image_paths) > args.max_images:
        # Uniform temporal subsample -- ffmpeg numbered frames in time order, so
        # picking every N-th preserves scene coverage over the whole capture.
        # Preferred over take-first-N (which biases to the first fraction of
        # the walk) especially when the launcher sends us an unpruned 3000-set.
        if args.max_images == 1:
            indices = [0]
        else:
            indices = [
                int(round(i * (len(image_paths) - 1) / (args.max_images - 1)))
                for i in range(args.max_images)
            ]
        original_count = len(image_paths)
        image_paths = [image_paths[i] for i in indices]
        print(
            f"[mast3r] uniformly subsampled {len(image_paths)}/{original_count} "
            "frames (including the first and last frames)", flush=True,
        )
    if not image_paths:
        return _fail(out_dir, "no_frames", "frames.zip contained no jpg/png files")
    print(f"[mast3r] {len(image_paths)} frames to process (image_size={args.image_size})", flush=True)

    timings: dict[str, float] = {}

    # ---- Load MASt3R model ----
    started = time.monotonic()
    try:
        import torch  # noqa: F401
        from mast3r.model import AsymmetricMASt3R
    except Exception as exc:  # noqa: BLE001
        return _fail(out_dir, "import_failed",
                     f"MASt3R model imports failed: {type(exc).__name__}: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return _fail(out_dir, "no_gpu",
                     "MASt3R requires a CUDA GPU. This job was scheduled on a CPU flavor.")

    weights_id = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    try:
        model = AsymmetricMASt3R.from_pretrained(weights_id).to(device).eval()
    except Exception as exc:  # noqa: BLE001
        return _fail(out_dir, "model_load_failed",
                     f"AsymmetricMASt3R.from_pretrained({weights_id!r}) raised "
                     f"{type(exc).__name__}: {exc}")
    timings["model_load_s"] = round(time.monotonic() - started, 1)
    print(f"[mast3r] model loaded in {timings['model_load_s']}s", flush=True)

    # torch 2.6 flipped torch.load's weights_only default to True. MASt3R's
    # retrieval processor's pickled argparse.Namespace fails the strict loader,
    # and upstream (mast3r/retrieval/processor.py) hasn't been patched. Only
    # affects the retrieval branch; safetensors loading is unaffected.
    _orig_torch_load = torch.load
    def _torch_load_compat(*args, **kwargs):  # noqa: E306
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_compat  # type: ignore[assignment]

    # ---- Build image pair list via retrieval ----
    # For a few-dozen frames MASt3R's swin pair scheme is fine. For thousands
    # we MUST use retrieval or the pairwise pass is quadratic and OOMs.
    started = time.monotonic()
    try:
        from mast3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
    except Exception as exc:  # noqa: BLE001
        return _fail(out_dir, "import_failed",
                     f"make_pairs/load_images import failed: {type(exc).__name__}: {exc}")

    imgs = load_images([str(p) for p in image_paths], size=args.image_size, verbose=False)
    # Scene graph selection: prefer retrieval-Na-k (proper matching pattern
    # from mast3r/demo.py) when retrieval weights + Retriever are available;
    # fall back to swin-K otherwise. The image bundles the retrieval weights
    # via docker/lab/Dockerfile's wget from Naver's CDN, so retrieval is the
    # default path in practice.
    scene_graph = args.scene_graph
    sim_matrix = None
    RETRIEVAL_PTH = Path("/root/.cache/torch/hub/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")
    CODEBOOK_PTH = Path("/root/.cache/torch/hub/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl")
    if scene_graph.startswith("retrieval") and not (RETRIEVAL_PTH.exists() and CODEBOOK_PTH.exists()):
        window = args.retrieval if args.retrieval > 0 else args.swin_window
        scene_graph = f"swin-{window}"
        print(f"[mast3r] retrieval weights or codebook not baked in, falling back to {scene_graph}", flush=True)

    if scene_graph.startswith("retrieval"):
        try:
            from mast3r.retrieval.processor import Retriever
        except Exception as exc:  # noqa: BLE001
            window = args.retrieval if args.retrieval > 0 else args.swin_window
            scene_graph = f"swin-{window}"
            print(f"[mast3r] Retriever import failed ({exc}); falling back to {scene_graph}", flush=True)
        else:
            retriever = Retriever(str(RETRIEVAL_PTH), backbone=model, device=device)
            with torch.no_grad():
                sim_matrix = retriever([str(p) for p in image_paths])
            print(f"[mast3r] sim_matrix computed: shape={tuple(sim_matrix.shape) if hasattr(sim_matrix, 'shape') else '?'}", flush=True)

    print(f"[mast3r] building pairs (scene_graph={scene_graph}, {len(image_paths)} frames)", flush=True)
    try:
        pairs = make_pairs(
            imgs,
            scene_graph=scene_graph,
            prefilter=None,
            symmetrize=True,
            sim_mat=sim_matrix,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(out_dir, "make_pairs_failed",
                     f"make_pairs({scene_graph!r}) raised {type(exc).__name__}: {exc}")
    timings["make_pairs_s"] = round(time.monotonic() - started, 1)
    print(f"[mast3r] {len(pairs)} pairs built in {timings['make_pairs_s']}s", flush=True)

    # ---- Run sparse global alignment ----
    started = time.monotonic()
    try:
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    except Exception as exc:  # noqa: BLE001
        return _fail(out_dir, "import_failed",
                     f"sparse_global_alignment import failed: {type(exc).__name__}: {exc}")
    try:
        scene = sparse_global_alignment(
            [str(p) for p in image_paths],
            pairs,
            str(cache_dir),
            model,
            device=device,
            shared_intrinsics=True,  # single-camera capture (iPhone) -- big prior
            niter1=args.niter1,
            niter2=args.niter2,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(out_dir, "sparse_ga_failed",
                     f"sparse_global_alignment raised {type(exc).__name__}: {exc}")
    timings["sparse_ga_s"] = round(time.monotonic() - started, 1)
    print(f"[mast3r] sparse_global_alignment completed in {timings['sparse_ga_s']}s", flush=True)

    # ---- Extract outputs ----
    started = time.monotonic()
    try:
        cams2world = _to_numpy(scene.get_im_poses())          # [N, 4, 4] cam2w
        focals = _to_numpy(scene.get_focals())                # [N] or [N, 1]
        principal_points = _to_numpy(scene.get_principal_points())  # [N, 2]
        pts3d_list = scene.get_sparse_pts3d()                 # list of tensors
        colors_list = scene.get_pts3d_colors()                # list of tensors
    except Exception as exc:  # noqa: BLE001
        return _fail(out_dir, "extract_failed",
                     f"scene attribute extraction raised {type(exc).__name__}: {exc}")

    # Flatten per-image pts3d + colors into a single sparse point cloud.
    import numpy as np
    all_pts = []
    all_cols = []
    for p, c in zip(pts3d_list, colors_list):
        p_np = _to_numpy(p).reshape(-1, 3)
        c_np = _to_numpy(c).reshape(-1, 3)
        n = min(len(p_np), len(c_np))
        all_pts.append(p_np[:n])
        all_cols.append(c_np[:n])
    pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), dtype=np.float32)
    cols = np.concatenate(all_cols, axis=0) if all_cols else np.zeros((0, 3), dtype=np.float32)
    # Cap the initial sparse cloud to a reasonable size for splatfacto init.
    if len(pts) > 500_000:
        idx = np.random.default_rng(seed=0).choice(len(pts), 500_000, replace=False)
        pts, cols = pts[idx], cols[idx]
    print(f"[mast3r] extracted: {len(pts)} sparse points, {len(cams2world)} poses", flush=True)

    # ---- Write processed_min.zip payload ----
    processed = work / "processed"
    (processed / "images").mkdir(parents=True, exist_ok=True)
    # Copy the frames the launcher's --train-only path expects at processed/images/.
    for src in image_paths:
        shutil.copy(src, processed / "images" / src.name)

    # Resolve output resolution: use the first image's true pixel dimensions.
    from PIL import Image as _PILImage
    with _PILImage.open(image_paths[0]) as im:
        width_px, height_px = im.size

    # MASt3R focals + principal points are in the internal image_size resolution,
    # NOT the original. Scale them to the output resolution so nerfstudio's
    # dataparser reads matching intrinsics vs the frames it actually loads.
    scale_x = width_px / args.image_size
    scale_y = height_px / args.image_size
    focals_scaled = np.asarray(focals, dtype=np.float32).reshape(len(image_paths), -1).copy()
    focals_scaled[:, 0] *= scale_x
    if focals_scaled.shape[1] >= 2:
        focals_scaled[:, 1] *= scale_y
    pps_scaled = np.asarray(principal_points, dtype=np.float32).reshape(len(image_paths), 2).copy()
    pps_scaled[:, 0] *= scale_x
    pps_scaled[:, 1] *= scale_y

    _write_nerfstudio_transforms(
        processed / "transforms.json",
        image_names=[p.name for p in image_paths],
        cams2world_opencv=cams2world,
        focals=focals_scaled,
        principal_points=pps_scaled,
        width=width_px,
        height=height_px,
    )
    _write_sparse_pc_ply(processed / "sparse_pc.ply", pts, cols)
    timings["write_outputs_s"] = round(time.monotonic() - started, 1)

    # Pack processed_min.zip. Prod worker's --train-only unpacks the same keys:
    # transforms.json, sparse_pc.ply. images/ is optional there (worker copies
    # from VW_IN/frames.zip itself), but including it here is idempotent-safe.
    started = time.monotonic()
    zip_path = out_dir / "processed_min.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(processed / "transforms.json", "transforms.json")
        archive.write(processed / "sparse_pc.ply", "sparse_pc.ply")
    timings["zip_s"] = round(time.monotonic() - started, 1)
    print(f"[mast3r] wrote {zip_path.name} ({zip_path.stat().st_size // 1_000_000} MB)", flush=True)

    # summary.json alongside processed_min.zip so the launcher can inspect it.
    total_s = round(time.monotonic() - started_all, 1)
    (out_dir / "summary.json").write_text(
        json.dumps({
            "engine": "mast3r-sfm",
            "frames": len(image_paths),
            "registered_images": len(cams2world),
            "sparse_points": int(len(pts)),
            "image_size": args.image_size,
            "retrieval_k": args.retrieval,
            "scene_graph": scene_graph,
            "timings": timings,
            "total_s": total_s,
        }, indent=2),
        encoding="utf-8",
    )

    # Free CUDA memory before returning so the container can shut down cleanly.
    del scene, model, imgs, pairs, pts3d_list, colors_list
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    print(f"[mast3r] complete in {total_s}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
