"""DA3 (Depth Anything 3) reconstruction entrypoint.

Replaces COLMAP/MASt3R for the SfM stage. Runs inside the vw-studio-worker
image with VW_IN / VW_OUT set by vw_stage.py.

Modes:
  --sfm-only       Run DA3 inference, produce processed_min.zip (transforms.json
                   + sparse_pc.ply) for splatfacto --train-only. [split Job A]
  --gs-only        Run DA3-GIANT with infer_gs=True, produce da3_gaussians.ply
                   directly. No splatfacto training. [da3-draft preset]
  --train-only     Skip DA3, consume processed_min.zip from VW_IN and run
                   splatfacto + ns-export. [split Job B, same as recon_entrypoint]
  (no flag)        Full pipeline: DA3 SfM + splatfacto training + export.

Inputs (VW_IN):
  frames.zip         Input images (jpg/png)

Outputs (VW_OUT):
  splat.ply          Full-attribute 3DGS PLY (from ns-export or DA3 direct)
  summary.json       Job metadata
  model.zip          Training checkpoint (when --keep-checkpoint)
  processed_min.zip  transforms.json + sparse_pc.ply (SfM output for split jobs)
  da3_gaussians.ply  Direct 3DGS output (when --gs-only)
  error.json         Structured failure info
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[da3-recon] {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log(f"$ {' '.join(cmd)}")
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    result = subprocess.run(cmd, check=False, **kwargs)
    if result.returncode != 0 and result.stderr:
        log(f"stderr: {result.stderr[-2000:]}")
    return result


def run_streaming(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command with live stdout/stderr streaming to log."""
    log(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout.splitlines():
        log(f"  {line}")
    if result.returncode != 0:
        log(f"  exit code: {result.returncode}")
    return result


def fail(out_dir: Path, code: str, detail: str) -> int:
    log(f"FAILED: {code} — {detail}")
    (out_dir / "error.json").write_text(
        json.dumps({"code": code, "detail": detail}, indent=2), encoding="utf-8"
    )
    return 1


def load_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)


def da3_inference(
    images: list[Path],
    model_id: str,
    infer_gs: bool,
    export_dir: Path | None = None,
    export_format: str = "mini_npz",
) -> dict:
    """Run DA3 inference and return the prediction outputs.

    Returns a dict with keys: depth, extrinsics, intrinsics, conf,
    processed_images, and optionally gaussians.
    """
    import torch
    from depth_anything_3.api import DepthAnything3

    # Reduce fragmentation on L4 (22GB) — DA3's multi-view attention is memory-hungry.
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading DA3 model: {model_id} on {device}")
    model = DepthAnything3.from_pretrained(model_id).to(device)

    image_paths = [str(p) for p in images]
    log(f"Running DA3 inference on {len(image_paths)} images")

    inference_kwargs: dict = {}
    if infer_gs:
        inference_kwargs["infer_gs"] = True
    if export_dir is not None:
        inference_kwargs["export_dir"] = str(export_dir)
        inference_kwargs["export_format"] = export_format

    prediction = model.inference(image=image_paths, **inference_kwargs)

    result = {
        "depth": prediction.depth,
        "extrinsics": prediction.extrinsics,
        "intrinsics": prediction.intrinsics,
        "conf": getattr(prediction, "conf", None),
        "processed_images": prediction.processed_images,
    }
    if infer_gs and hasattr(prediction, "aux") and "gaussians" in prediction.aux:
        result["gaussians"] = prediction.aux["gaussians"]
        log("DA3 Gaussian prediction available")
    return result


def da3_to_transforms(
    prediction: dict,
    images: list[Path],
    images_dir: Path,
    output_dir: Path,
) -> Path:
    """Convert DA3 output to nerfstudio transforms.json format.

    DA3 gives us:
      - extrinsics: (N, 3, 4) world-to-camera in OpenCV convention
      - intrinsics: (N, 3, 3) camera intrinsics

    Nerfstudio transforms.json expects:
      - per-frame transform_matrix: 4x4 camera-to-world in OpenGL/Blender convention
      - fl_x, fl_y, cx, cy, w, h per frame
      - file_path relative to data root
    """
    exts = prediction["extrinsics"]  # (N, 3, 4) w2c
    ixts = prediction["intrinsics"]  # (N, 3, 3)
    depths = prediction["depth"]     # (N, H, W)
    processed_imgs = prediction["processed_images"]  # (N, H, W, 3)

    n_frames = len(images)
    frames = []

    # DA3 resizes images internally; we need to scale intrinsics back to
    # the original image resolution so nerfstudio can load the full-res files.
    from PIL import Image as PILImage

    for i in range(n_frames):
        # Convert w2c (3x4) to c2w (4x4)
        w2c = np.eye(4)
        w2c[:3, :] = exts[i]
        c2w = np.linalg.inv(w2c)

        # Convert OpenCV (x-right, y-down, z-forward) to OpenGL (x-right, y-up, z-backward)
        # by flipping Y and Z axes of the camera coordinate system
        opengl_to_cv = np.diag([1, -1, -1, 1]).astype(np.float64)
        c2w_opengl = c2w @ opengl_to_cv

        # DA3's intrinsics are for the resized image — scale to original resolution
        da_h, da_w = depths[i].shape[:2]
        with PILImage.open(images[i]) as img:
            orig_w, orig_h = img.size
        scale_x = orig_w / da_w
        scale_y = orig_h / da_h

        ixt = ixts[i]  # (3, 3)
        fl_x = float(ixt[0, 0]) * scale_x
        fl_y = float(ixt[1, 1]) * scale_y
        cx = float(ixt[0, 2]) * scale_x
        cy = float(ixt[1, 2]) * scale_y

        h, w = orig_h, orig_w

        frame = {
            "file_path": f"images/{images[i].name}",
            "transform_matrix": c2w_opengl.tolist(),
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": int(w),
            "h": int(h),
        }
        frames.append(frame)

    # Use first frame's intrinsics as the applied scale reference
    first_ixt = ixts[0]
    camera_angle_x = float(2 * np.arctan2(first_ixt[0, 2], first_ixt[0, 0]))

    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": frames,
        "ply_file_path": "sparse_pc.ply",
    }

    transforms_path = output_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    log(f"Wrote transforms.json with {n_frames} frames to {transforms_path}")
    return transforms_path


def da3_to_sparse_pc(
    prediction: dict,
    output_dir: Path,
    max_points: int = 500_000,
) -> Path:
    """Fuse DA3 depth maps into a sparse point cloud (PLY).

    Back-projects each frame's depth map using the predicted intrinsics and
    extrinsics, then concatenates and downsamples to max_points.
    """
    depths = prediction["depth"]       # (N, H, W)
    exts = prediction["extrinsics"]    # (N, 3, 4) w2c
    ixts = prediction["intrinsics"]    # (N, 3, 3)
    confs = prediction.get("conf")     # (N, H, W) or None
    processed = prediction["processed_images"]  # (N, H, W, 3)

    all_points = []
    all_colors = []

    for i in range(len(depths)):
        depth = depths[i]  # (H, W)
        h, w = depth.shape
        ixt = ixts[i]

        # Create pixel coordinate grid
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        # Normalize to camera coordinates
        z = depth
        x = (xs - ixt[0, 2]) * z / ixt[0, 0]
        y = (ys - ixt[1, 2]) * z / ixt[1, 1]
        # Stack into (H*W, 3) camera coordinates
        pts_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

        # Filter by confidence if available
        if confs is not None:
            conf = confs[i].flatten()
            # Keep top 60% confidence points
            thresh = np.percentile(conf, 40)
            mask = conf > thresh
            pts_cam = pts_cam[mask]
            colors = processed[i].reshape(-1, 3)[mask]
        else:
            colors = processed[i].reshape(-1, 3)

        # Filter invalid depth (zero, negative, or extreme)
        valid = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[valid]
        colors = colors[valid]

        # Transform to world coordinates: inv(w2c) * [x, y, z, 1]
        w2c = np.eye(4)
        w2c[:3, :] = exts[i]
        c2w = np.linalg.inv(w2c)
        pts_homog = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (c2w @ pts_homog.T).T[:, :3]

        all_points.append(pts_world)
        all_colors.append(colors)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Downsample if needed
    if len(all_points) > max_points:
        idx = np.random.choice(len(all_points), max_points, replace=False)
        all_points = all_points[idx]
        all_colors = all_colors[idx]

    log(f"Sparse point cloud: {len(all_points)} points")

    # Write PLY
    ply_path = output_dir / "sparse_pc.ply"
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(all_points, all_colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} ")
            r = int(np.clip(col[0], 0, 1) * 255) if col[0] <= 1.0 else int(col[0])
            g = int(np.clip(col[1], 0, 1) * 255) if col[1] <= 1.0 else int(col[1])
            b = int(np.clip(col[2], 0, 1) * 255) if col[2] <= 1.0 else int(col[2])
            f.write(f"{r} {g} {b}\n")

    log(f"Wrote sparse_pc.ply to {ply_path}")
    return ply_path


def export_da3_gs_ply(prediction: dict, output_path: Path) -> Path:
    """Export DA3's direct 3DGS prediction to a standard 3DGS PLY file.

    The DA3 Gaussian output is in the prediction.aux['gaussians'] dict,
    containing positions, rotations, scales, colors, and opacities.
    This function converts them to the standard 3DGS PLY format that
    our splat_io.py can read.
    """
    gaussians = prediction.get("gaussians")
    if gaussians is None:
        raise RuntimeError("No Gaussian data in DA3 prediction (infer_gs=True required)")

    # DA3's Gaussian output format may vary; we try common keys
    positions = gaussians.get("positions") or gaussians.get("means")
    if positions is None:
        raise RuntimeError(f"DA3 gaussians dict keys: {list(gaussians.keys())}")

    # Convert to numpy if needed
    positions = np.asarray(positions)
    n_gaussians = len(positions)
    log(f"DA3 produced {n_gaussians} gaussians")

    # Try to get other attributes
    scales = gaussians.get("scales") or gaussians.get("scale")
    rotations = gaussians.get("rotations") or gaussians.get("quats")
    colors = gaussians.get("colors") or gaussians.get("sh_dc") or gaussians.get("rgb")
    opacities = gaussians.get("opacities") or gaussians.get("opacity")

    # Defaults if missing
    if scales is None:
        scales = np.ones((n_gaussians, 3)) * 0.01
    else:
        scales = np.asarray(scales)

    if rotations is None:
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n_gaussians, 1))
    else:
        rotations = np.asarray(rotations)

    if colors is None:
        colors = np.ones((n_gaussians, 3)) * 128
    else:
        colors = np.asarray(colors)

    if opacities is None:
        opacities = np.ones(n_gaussians) * 0.9
    else:
        opacities = np.asarray(opacities)

    # Write 3DGS PLY
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format binary_little_endian 1.0\n")
        f.write(f"element vertex {n_gaussians}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        for i in range(3):
            f.write(f"property float f_dc_{i}\n")
        for i in range(3):
            f.write(f"property float f_scale_{i}\n")
        for i in range(4):
            f.write(f"property float f_rot_{i}\n")
        f.write("property float opacity\n")
        f.write("end_header\n")

        # SH DC coefficient: color * 0.28209479 (SH C0)
        SH_C0 = 0.28209479177387814
        import struct

        for i in range(n_gaussians):
            pos = positions[i]
            sc = np.exp(scales[i]) if np.any(scales[i] < 0) else scales[i]  # log-space → linear
            rot = rotations[i]
            col = colors[i]
            opa = opacities[i]

            # Sigmoid for opacity if raw logits
            if opa < 0 or opa > 1:
                opa = 1.0 / (1.0 + np.exp(-opa))

            f_dc = [float(col[0] / 255.0 - 0.5) / SH_C0,
                    float(col[1] / 255.0 - 0.5) / SH_C0,
                    float(col[2] / 255.0 - 0.5) / SH_C0]

            data = struct.pack(
                "3f3f3f3f4ff",
                float(pos[0]), float(pos[1]), float(pos[2]),
                0.0, 0.0, 0.0,  # normals (unused)
                f_dc[0], f_dc[1], f_dc[2],
                float(sc[0]), float(sc[1]), float(sc[2]),
                float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]),
                float(opa),
            )
            f.write(data)

    log(f"Wrote DA3 3DGS PLY ({n_gaussians} gaussians) to {output_path}")
    return output_path


def write_depth_maps(prediction: dict, images: list[Path], output_dir: Path) -> None:
    """Write per-frame depth (and confidence) maps as .npy files.

    Named after the source frame stem, not the inference index, so a map can
    always be traced back to the frame it came from. splatfacto does not
    consume these — see make_depth_bundle for why we keep them anyway.
    """
    depths_dir = output_dir / "depths"
    depths_dir.mkdir(parents=True, exist_ok=True)
    depths = prediction["depth"]
    for i, depth in enumerate(depths):
        stem = images[i].stem if i < len(images) else f"frame_{i:05d}"
        np.save(depths_dir / f"{stem}.npy", depth)

    confs = prediction.get("conf")
    if confs is not None:
        conf_dir = output_dir / "confidence"
        conf_dir.mkdir(parents=True, exist_ok=True)
        for i, conf in enumerate(confs):
            stem = images[i].stem if i < len(images) else f"frame_{i:05d}"
            np.save(conf_dir / f"{stem}.npy", conf)
        log(f"Wrote {len(confs)} confidence maps to {conf_dir}")

    log(f"Wrote {len(depths)} depth maps to {depths_dir}")


def make_depth_bundle(processed: Path, output_path: Path) -> Path | None:
    """Bundle depths/ + confidence/ into depths.zip.

    Kept deliberately separate from processed_min.zip: that archive is the
    SfM -> training handoff and stays small (~8 MB) because both legs pay to
    transfer it. The depth and confidence fields are a genuine DA3 output that
    nothing downstream consumes yet, so they ride in their own artifact rather
    than being discarded inside the container. The runner downloads everything
    under out/, so this lands in <stage>/remote_out/depths.zip automatically.
    """
    members: list[tuple[Path, str]] = []
    for folder in ("depths", "confidence"):
        source = processed / folder
        if not source.is_dir():
            continue
        for path in sorted(source.glob("*.npy")):
            members.append((path, f"{folder}/{path.name}"))
    if not members:
        log("No depth maps to bundle")
        return None
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in members:
            archive.write(path, arcname)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log(f"Wrote {output_path} ({len(members)} maps, {size_mb:.1f} MB)")
    return output_path


def make_processed_min(processed: Path, output_path: Path) -> None:
    """Bundle transforms.json + sparse_pc.ply into processed_min.zip."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("transforms.json", "sparse_pc.ply"):
            candidate = processed / name
            if candidate.exists():
                archive.write(candidate, name)
        # No colmap_database.db for DA3 — we don't use COLMAP at all.
        # nerfstudio's dataparser only needs transforms.json. Depth maps go in
        # depths.zip instead (make_depth_bundle) so this stays a small handoff.
    log(f"Wrote {output_path}")


def filter_supported_flags(train_args: list[str]) -> list[str]:
    """Drop --flag value pairs that the installed ns-train doesn't know."""
    help_text = ""
    probe = run(["ns-train", "splatfacto", "--help"], capture_output=True, text=True)
    if probe.returncode == 0:
        help_text = probe.stdout + probe.stderr
    if not help_text:
        return train_args
    if "--vis" in train_args:
        import re

        index = train_args.index("--vis") + 1
        window = re.search(r"--vis\b(.{0,300})", help_text, re.DOTALL)
        choices = window.group(1) if window else ""
        if index < len(train_args) and train_args[index] == "none" and "none" not in choices:
            log("--vis none unsupported in this nerfstudio; using tensorboard")
            train_args[index] = "tensorboard"
    kept: list[str] = []
    index = 0
    while index < len(train_args):
        arg = train_args[index]
        if arg.startswith("--") and arg not in help_text:
            log(f"dropping unsupported flag: {arg}")
            index += 2 if index + 1 < len(train_args) and not train_args[index + 1].startswith("--") else 1
            continue
        kept.append(arg)
        index += 1
    return kept


def finish_export(
    args, out_dir: Path, processed: Path, train_out: Path, export: Path,
    timings: dict, frame_count: int, matching_used: str,
) -> int:
    """Shared post-train flow: ns-export, copy outputs, archive bundles, summary."""
    configs = sorted(train_out.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not configs:
        return fail(out_dir, "no_config", "no config.yml produced by training")
    started = time.monotonic()
    exported = run([
        "ns-export", "gaussian-splat",
        "--load-config", str(configs[0]),
        "--output-dir", str(export),
    ])
    timings["export_s"] = round(time.monotonic() - started, 1)
    if exported.returncode != 0:
        return fail(out_dir, "export_failed", f"ns-export exit {exported.returncode}")

    plys = sorted(export.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not plys:
        return fail(out_dir, "no_ply", "ns-export produced no PLY")
    shutil.copyfile(plys[0], out_dir / "splat.ply")

    if args.keep_checkpoint:
        shutil.make_archive(str(out_dir / "model"), "zip", root_dir=str(train_out))
        make_processed_min(processed, out_dir / "processed_min.zip")
        log("checkpoint archived (model.zip + processed_min.zip)")

    # Full mode ran DA3 in this same container, so the depth fields are here
    # to keep. In --train-only they were produced by Job A and already shipped.
    depth_bundle = make_depth_bundle(processed, out_dir / "depths.zip")

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "frames": frame_count,
                "matching_method": matching_used,
                "train_args": json.loads(args.train_args),
                "depth_maps": bool(depth_bundle),
                "timings": timings,
                "sfm_engine": "da3",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log("complete")
    return 0


def main() -> int:  # noqa: PLR0911, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--train-args", default="[]", help="JSON list of ns-train args")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument(
        "--da3-model", default="depth-anything/DA3-LARGE-1.1",
        help="HuggingFace model ID for DA3 (e.g. depth-anything/DA3-GIANT-1.1)",
    )
    parser.add_argument(
        "--gs-only", action="store_true",
        help="Run DA3 with infer_gs=True and output da3_gaussians.ply directly. No splatfacto.",
    )
    parser.add_argument(
        "--sfm-only", action="store_true",
        help="Run DA3 SfM only, produce processed_min.zip. No splatfacto training.",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Skip DA3. Consume processed_min.zip from VW_IN, run splatfacto + ns-export.",
    )
    parser.add_argument(
        "--refine-mode", action="store_true",
        help="Refine an existing splat: VW_IN must contain model.zip + processed_min.zip.",
    )
    parser.add_argument(
        "--merge-splat", action="store_true",
        help="Incremental merge: run DA3 direct 3DGS on new frames, then merge with base splat.ply from VW_IN.",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.05,
        help="Voxel edge length for deduplication during merge (scene units).",
    )
    parser.add_argument(
        "--dynamic-cull-radius", type=float, default=0.10,
        help="Cull base gaussians within this radius of new ones (0 = disable).",
    )
    parser.add_argument(
        "--icp-max-distance", type=float, default=0.5,
        help="ICP max correspondence distance for alignment (0 = skip alignment).",
    )
    parser.add_argument(
        "--max-sfm-frames", type=int, default=80,
        help="Max frames passed to DA3 inference (subsampled evenly). Prevents OOM on 22GB L4.",
    )
    args = parser.parse_args()

    in_dir = Path(os.environ["VW_IN"])
    out_dir = Path(os.environ["VW_OUT"])
    work = Path("/tmp/da3_recon")
    images = work / "images"
    processed = work / "processed"
    train_out = work / "train"
    export = work / "export"
    for folder in (images, processed, train_out, export):
        folder.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    frames_zip = in_dir / "frames.zip"
    if not frames_zip.exists():
        return fail(out_dir, "missing_input", "frames.zip not found in stage inputs")
    with zipfile.ZipFile(frames_zip) as archive:
        archive.extractall(images)
    image_paths = load_images(images)
    frame_count = len(image_paths)
    log(f"{frame_count} frames extracted")

    # Subsample frames for DA3 inference — the multi-view attention scales
    # quadratically with frame count and OOMs on 22GB L4 above ~100 frames.
    # All frames are still passed to splatfacto training via processed/images/.
    sfm_frames = image_paths
    if not args.train_only and frame_count > args.max_sfm_frames:
        step = frame_count / args.max_sfm_frames
        indices = [int(i * step) for i in range(args.max_sfm_frames)]
        sfm_frames = [image_paths[i] for i in indices]
        log(f"Subsampled {frame_count} → {len(sfm_frames)} frames for DA3 SfM (max-sfm-frames={args.max_sfm_frames})")

    if args.gs_only and (args.sfm_only or args.train_only):
        return fail(out_dir, "bad_args", "--gs-only is mutually exclusive with --sfm-only and --train-only")
    if args.sfm_only and args.train_only:
        return fail(out_dir, "bad_args", "--sfm-only and --train-only are mutually exclusive")
    if args.merge_splat and not args.gs_only:
        return fail(out_dir, "bad_args", "--merge-splat requires --gs-only")

    # ---- --gs-only: DA3 direct 3DGS, no splatfacto ----
    if args.gs_only:
        log(f"DA3 direct 3DGS mode (model: {args.da3_model})")
        started = time.monotonic()
        prediction = da3_inference(
            sfm_frames,
            model_id=args.da3_model,
            infer_gs=True,
            export_dir=out_dir,
            export_format="gs_ply",
        )
        timings["da3_inference_s"] = round(time.monotonic() - started, 1)

        # Try to find the gs_ply exported by DA3's export pipeline
        gs_ply_candidates = sorted(out_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
        if gs_ply_candidates:
            shutil.copyfile(gs_ply_candidates[0], out_dir / "da3_gaussians.ply")
            shutil.copyfile(gs_ply_candidates[0], out_dir / "splat.ply")
            log(f"DA3 gs_ply exported to {out_dir / 'splat.ply'}")
        else:
            # Fallback: export from prediction.aux['gaussians']
            try:
                export_da3_gs_ply(prediction, out_dir / "da3_gaussians.ply")
                shutil.copyfile(out_dir / "da3_gaussians.ply", out_dir / "splat.ply")
            except Exception as exc:
                return fail(out_dir, "gs_export_failed", str(exc))

        (out_dir / "summary.json").write_text(
            json.dumps({
                "frames": frame_count,
                "sfm_engine": "da3",
                "da3_model": args.da3_model,
                "direct_gs": True,
                "timings": timings,
            }, indent=2),
            encoding="utf-8",
        )
        # ---- --merge-splat: merge new DA3 3DGS with base splat ----
        if args.merge_splat:
            base_ply = in_dir / "splat.ply"
            if not base_ply.exists():
                return fail(out_dir, "missing_input",
                            "--merge-splat requires splat.ply (base splat) in $VW_IN")
            log(f"Merging new DA3 splat with base: {base_ply}")
            from gaussian_merge import MergeConfig, merge_splats
            from splat_io import read_gaussian_ply, write_gaussian_ply

            merge_config = MergeConfig(
                voxel_size=args.voxel_size,
                icp_max_distance=args.icp_max_distance,
                dynamic_cull_radius=args.dynamic_cull_radius,
                prefer_new=True,
            )
            merge_started = time.monotonic()
            base_splat = read_gaussian_ply(base_ply)
            new_splat = read_gaussian_ply(out_dir / "splat.ply")
            log(f"Base: {base_splat.count} gaussians | New: {new_splat.count} gaussians")
            merged = merge_splats(base_splat, new_splat, merge_config)
            timings["merge_s"] = round(time.monotonic() - merge_started, 1)
            log(f"Merged: {merged.count} gaussians (culled {base_splat.count + new_splat.count - merged.count})")
            write_gaussian_ply(merged, out_dir / "splat.ply")

            (out_dir / "summary.json").write_text(
                json.dumps({
                    "frames": frame_count,
                    "sfm_engine": "da3",
                    "da3_model": args.da3_model,
                    "direct_gs": True,
                    "incremental_merge": True,
                    "base_gaussians": base_splat.count,
                    "new_gaussians": new_splat.count,
                    "merged_gaussians": merged.count,
                    "merge_config": {
                        "voxel_size": args.voxel_size,
                        "dynamic_cull_radius": args.dynamic_cull_radius,
                        "icp_max_distance": args.icp_max_distance,
                    },
                    "timings": timings,
                }, indent=2),
                encoding="utf-8",
            )
            log("DA3 incremental merge complete")
            return 0

        log("DA3 direct 3DGS complete")
        return 0

    # ---- --train-only: skip DA3, consume processed_min.zip ----
    if args.train_only:
        processed_zip = in_dir / "processed_min.zip"
        if not processed_zip.exists():
            return fail(out_dir, "missing_input",
                        "--train-only requires processed_min.zip in $VW_IN")
        with zipfile.ZipFile(processed_zip) as archive:
            archive.extractall(processed)
        # Only move extracted frames into processed/images/ if the SfM zip
        # didn't already include them. DA3's processed_min.zip contains only
        # the subsampled SfM frames — moving all 500 would cause a mismatch
        # between transforms.json (80 frames) and images/ (500 files).
        target_images = processed / "images"
        if not target_images.exists() and images.exists():
            shutil.move(str(images), str(target_images))

        train_args = filter_supported_flags(json.loads(args.train_args))
        # Disable torch.compile/dynamo — nerfstudio's splatfacto uses @torch.compile
        # which triggers a broken inductor import chain when torch version is mismatched.
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        started = time.monotonic()
        train_cmd = [
            "ns-train", "splatfacto",
            "--data", str(processed),
            "--output-dir", str(train_out),
            *train_args,
        ]
        if args.refine_mode:
            bundle_model = in_dir / "model.zip"
            if bundle_model.exists():
                base_train = Path("/tmp/da3_recon/base_train")
                base_train.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(bundle_model) as archive:
                    archive.extractall(base_train)
                configs = sorted(base_train.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
                if configs:
                    train_cmd.extend(["--load-dir", str(configs[0].parent)])
                    log(f"Refine: loading checkpoint from {configs[0].parent}")
        result = run_streaming(train_cmd)
        timings["train_s"] = round(time.monotonic() - started, 1)
        if result.returncode != 0:
            return fail(out_dir, "train_failed", f"ns-train exit {result.returncode}")
        return finish_export(args, out_dir, processed, train_out, export, timings, frame_count, "da3")

    # ---- --sfm-only or full: run DA3 SfM first ----
    log(f"DA3 SfM mode (model: {args.da3_model})")
    started = time.monotonic()
    prediction = da3_inference(
        sfm_frames,
        model_id=args.da3_model,
        infer_gs=False,
    )
    timings["da3_inference_s"] = round(time.monotonic() - started, 1)

    # Convert DA3 output to nerfstudio format (use sfm_frames — prediction only has these)
    da3_to_transforms(prediction, sfm_frames, images, processed)
    da3_to_sparse_pc(prediction, processed)
    # Depth + confidence fields are kept as their own artifact. They are NOT
    # referenced from transforms.json — pointing splatfacto at them is what
    # broke the first run (depth path mismatch), and splatfacto ignores them
    # regardless. Retained because they're a real DA3 output we may want for
    # occupancy grids / mesh extraction later.
    write_depth_maps(prediction, sfm_frames, processed)

    # Move ALL images into processed/images/ for nerfstudio dataparser
    # (splatfacto can use more frames than DA3 processed for training supervision)
    target_images = processed / "images"
    target_images.mkdir(parents=True, exist_ok=True)
    for img in image_paths:
        shutil.copyfile(img, target_images / img.name)

    if args.sfm_only:
        make_processed_min(processed, out_dir / "processed_min.zip")
        depth_bundle = make_depth_bundle(processed, out_dir / "depths.zip")
        (out_dir / "summary.json").write_text(
            json.dumps({
                "frames": frame_count,
                "frames_sfm": len(sfm_frames),
                "sfm_engine": "da3",
                "da3_model": args.da3_model,
                "depth_maps": bool(depth_bundle),
                "timings": timings,
            }, indent=2),
            encoding="utf-8",
        )
        log("DA3 SfM-only complete")
        return 0

    # Full mode: continue to splatfacto training
    train_args = filter_supported_flags(json.loads(args.train_args))
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    started = time.monotonic()
    train_cmd = [
        "ns-train", "splatfacto",
        "--data", str(processed),
        "--output-dir", str(train_out),
        *train_args,
    ]
    result = run_streaming(train_cmd)
    timings["train_s"] = round(time.monotonic() - started, 1)
    if result.returncode != 0:
        return fail(out_dir, "train_failed", f"ns-train exit {result.returncode}")
    return finish_export(args, out_dir, processed, train_out, export, timings, frame_count, "da3")


if __name__ == "__main__":
    sys.exit(main())
