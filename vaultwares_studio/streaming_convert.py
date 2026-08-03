"""Convert DA3-Streaming output into the nerfstudio bundle splatfacto expects.

DA3-Streaming writes its own flat format:

    camera_poses.txt   one flattened 4x4 camera-to-world matrix per line
    intrinsic.txt      "fx fy cx cy" per line, one per frame
    pcd/combined_pcd.ply   merged, already-filtered point cloud

Downstream we need ``transforms.json`` + ``sparse_pc.ply``, i.e. exactly the
contract ``make_processed_min`` bundles, so Job B / splatfacto needs no changes.

Two conventions matter, and getting either wrong produces a reconstruction that
trains without complaint and looks subtly wrong:

**C2W, not W2C.** ``da3_to_transforms`` in the entrypoint handles DA3's *direct*
prediction, which is ``(N,3,4)`` world-to-camera, and inverts it. Streaming has
already inverted — ``camera_poses.txt`` is camera-to-world. Inverting again would
be silent and wrong, so this module never inverts.

**OpenCV to OpenGL.** DA3 works in OpenCV camera axes (x-right, y-down,
z-forward); nerfstudio wants OpenGL/Blender (x-right, y-up, z-backward). The fix
is a right-multiply by ``diag(1,-1,-1,1)``, which negates the Y and Z basis
columns and leaves the translation column untouched. Because position is
unaffected, comparing camera *positions* against a reference validates the
C2W/W2C choice, while comparing *orientations* validates this flip — the two
tests are independent.

**Intrinsics are for the resized frames.** Streaming is fed downscaled images
(504x280 rather than 1920x1080), so fx/fy/cx/cy come back in that resolution's
pixels while splatfacto loads the full-res originals. They have to be scaled up,
the same trick ``da3_to_transforms`` uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Right-multiplying a c2w by this converts OpenCV camera axes to OpenGL.
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def load_streaming_poses(stream_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read camera_poses.txt and intrinsic.txt.

    Returns ``(poses, intrinsics)`` with shapes ``(N,4,4)`` camera-to-world and
    ``(N,4)`` as ``fx, fy, cx, cy``.
    """
    poses_path = stream_dir / "camera_poses.txt"
    intr_path = stream_dir / "intrinsic.txt"
    for path in (poses_path, intr_path):
        if not path.exists():
            raise FileNotFoundError(f"DA3-Streaming output missing: {path}")

    poses = np.loadtxt(poses_path, dtype=np.float64)
    if poses.ndim == 1:  # a single frame collapses to 1-D
        poses = poses[None, :]
    if poses.shape[1] != 16:
        raise ValueError(f"camera_poses.txt has {poses.shape[1]} columns, expected 16")
    poses = poses.reshape(-1, 4, 4)

    intrinsics = np.loadtxt(intr_path, dtype=np.float64)
    if intrinsics.ndim == 1:
        intrinsics = intrinsics[None, :]
    if intrinsics.shape[1] != 4:
        raise ValueError(f"intrinsic.txt has {intrinsics.shape[1]} columns, expected 4")

    if len(poses) != len(intrinsics):
        raise ValueError(
            f"pose/intrinsic count mismatch: {len(poses)} vs {len(intrinsics)}"
        )
    return poses, intrinsics


def validate_poses(poses: np.ndarray, *, rtol: float = 1e-3) -> None:
    """Fail loudly on malformed pose matrices.

    Cheap, and the alternative is a training run that burns GPU time before
    anyone notices the geometry is nonsense.
    """
    if not np.isfinite(poses).all():
        raise ValueError("camera poses contain non-finite values")

    bottom = poses[:, 3, :]
    if not np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("camera poses have a malformed bottom row (expected [0,0,0,1])")

    rotations = poses[:, :3, :3]
    dets = np.linalg.det(rotations)
    if not np.allclose(dets, 1.0, rtol=rtol, atol=rtol):
        raise ValueError(
            f"camera rotations are not proper (det range {dets.min():.5f}..{dets.max():.5f}); "
            "a determinant of -1 means a reflection crept in"
        )


def streaming_to_transforms(
    stream_dir: Path,
    image_names: list[str],
    *,
    stream_size: tuple[int, int],
    original_size: tuple[int, int],
    ply_file_path: str = "sparse_pc.ply",
) -> dict:
    """Build the nerfstudio transforms.json payload.

    ``image_names`` must be in the same order streaming consumed them — it sorts
    the input directory, so sorted order is the contract.
    ``stream_size`` / ``original_size`` are ``(width, height)``.
    """
    poses, intrinsics = load_streaming_poses(stream_dir)
    validate_poses(poses)

    if len(image_names) != len(poses):
        raise ValueError(
            f"{len(image_names)} image names but {len(poses)} poses — these must "
            "correspond one-to-one and in order"
        )

    stream_w, stream_h = stream_size
    orig_w, orig_h = original_size
    scale_x = orig_w / stream_w
    scale_y = orig_h / stream_h

    frames = []
    for i, name in enumerate(image_names):
        # Already camera-to-world: do NOT invert. Only rebase the axes.
        c2w_opengl = poses[i] @ OPENCV_TO_OPENGL
        fx, fy, cx, cy = intrinsics[i]
        frames.append(
            {
                "file_path": f"images/{name}",
                "transform_matrix": c2w_opengl.tolist(),
                "fl_x": float(fx) * scale_x,
                "fl_y": float(fy) * scale_y,
                "cx": float(cx) * scale_x,
                "cy": float(cy) * scale_y,
                "w": int(orig_w),
                "h": int(orig_h),
            }
        )

    fx0, _, cx0, _ = intrinsics[0]
    camera_angle_x = float(2.0 * np.arctan2(cx0, fx0))
    return {
        "camera_angle_x": camera_angle_x,
        "frames": frames,
        "ply_file_path": ply_file_path,
    }


def write_processed_bundle(
    stream_dir: Path,
    image_names: list[str],
    output_dir: Path,
    *,
    stream_size: tuple[int, int],
    original_size: tuple[int, int],
) -> tuple[Path, Path]:
    """Write transforms.json + sparse_pc.ply into ``output_dir``.

    The point cloud is copied straight across: streaming already applies its own
    ``depth_threshold`` and confidence filtering, and its output has a max/p95
    radius ratio around 1.4 (against 416 for the direct-3DGS path), so the
    aggressive spatial filter in ``splat_filter`` is not warranted here.
    """
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    transforms = streaming_to_transforms(
        stream_dir,
        image_names,
        stream_size=stream_size,
        original_size=original_size,
    )
    transforms_path = output_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")

    source_ply = stream_dir / "pcd" / "combined_pcd.ply"
    if not source_ply.exists():
        raise FileNotFoundError(f"DA3-Streaming point cloud missing: {source_ply}")
    sparse_path = output_dir / "sparse_pc.ply"
    shutil.copyfile(source_ply, sparse_path)
    return transforms_path, sparse_path
