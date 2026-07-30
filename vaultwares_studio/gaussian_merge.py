"""Gaussian splat merging for incremental reconstruction.

Merges two 3DGS PLYs (a base splat + a new splat from fresh footage) into a
single coherent splat. Handles:

- **Alignment**: ICP on the point clouds when the two splats come from
  independent reconstructions (different coordinate frames).
- **Deduplication**: Voxel-grid deduplication — within each voxel, keep the
  Gaussian with the higher quality score (opacity × volume).
- **Dynamic object culling**: Base Gaussians that fall within a cull radius
  of new Gaussians are removed — the new capture represents current reality
  (e.g., a person walked into the scene, furniture was moved).

Used by the ``da3-incremental`` preset and the ``--merge-splat`` entrypoint mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .splat_io import GaussianSplat, read_gaussian_ply, write_gaussian_ply


@dataclass
class MergeConfig:
    """Parameters controlling the merge behavior."""

    # Voxel edge length (in scene units, typically meters) for deduplication.
    # Smaller = more aggressive dedup, larger = keeps more overlapping gaussians.
    voxel_size: float = 0.05
    # ICP max correspondence distance (scene units). Set to 0 to skip alignment.
    icp_max_distance: float = 0.5
    # Cull base gaussians within this radius of any new gaussian position.
    # Set to 0 to disable dynamic culling (pure concatenation + dedup).
    dynamic_cull_radius: float = 0.10
    # Remove merged gaussians with sigmoid opacity below this threshold.
    opacity_threshold: float = 0.005
    # When True, prefer new gaussians over base in overlapping voxels
    # (newer capture = more current reality). When False, keep higher quality.
    prefer_new: bool = True


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _quality_score(splat: GaussianSplat) -> np.ndarray:
    """Per-gaussian quality: opacity × volume (larger + more opaque = better)."""
    opacity = _sigmoid(splat.opacity)
    scales = np.exp(splat.scales)  # log-scale → linear
    volume = np.prod(scales, axis=1)
    return opacity * volume


def _apply_transform(splat: GaussianSplat, transform: np.ndarray) -> GaussianSplat:
    """Apply a 4x4 rigid transform to a GaussianSplat.

    Transforms positions and rotates the quaternion + scale axes.
    """
    positions = splat.positions.astype(np.float64)
    homog = np.hstack([positions, np.ones((len(positions), 1))])
    new_positions = (transform @ homog.T).T[:, :3].astype(np.float32)

    # Extract rotation from the 4x4 transform
    rotation = transform[:3, :3].astype(np.float32)
    # Apply rotation to quaternions: q_new = R_quat * q_old
    # For simplicity, convert rotation matrix to quaternion, then quaternion multiply
    rot_quat = _rotation_matrix_to_quaternion(rotation)
    new_rotations = np.zeros_like(splat.rotations)
    for i in range(len(splat.rotations)):
        new_rotations[i] = _quaternion_multiply(rot_quat, splat.rotations[i])

    return GaussianSplat(
        positions=new_positions,
        sh0=splat.sh0,
        opacity=splat.opacity,
        scales=splat.scales,
        rotations=new_rotations,
        sh_rest=splat.sh_rest,
    )


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Simple point-to-point ICP alignment.

    Returns a 4x4 rigid transform that maps source onto target.
    Uses nearest-neighbor correspondences with a distance threshold.
    """
    if max_distance <= 0 or len(source) < 10 or len(target) < 10:
        return np.eye(4)

    # Downsample for speed
    n_sample = min(10000, len(source), len(target))
    src = source[np.random.choice(len(source), n_sample, replace=False)]
    tgt = target[np.random.choice(len(target), n_sample, replace=False)]

    transform = np.eye(4)
    prev_error = float("inf")

    for _ in range(max_iterations):
        # Transform source
        src_h = np.hstack([src, np.ones((len(src), 1))])
        src_t = (transform @ src_h.T).T[:, :3]

        # Find nearest neighbors
        from scipy.spatial import cKDTree

        tree = cKDTree(tgt)
        distances, indices = tree.query(src_t, k=1)
        valid = distances < max_distance
        if not np.any(valid):
            break

        src_valid = src_t[valid]
        tgt_valid = tgt[indices[valid]]

        # Compute rigid transform via SVD
        src_centroid = src_valid.mean(axis=0)
        tgt_centroid = tgt_valid.mean(axis=0)
        src_centered = src_valid - src_centroid
        tgt_centered = tgt_valid - tgt_centroid

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid

        new_transform = np.eye(4)
        new_transform[:3, :3] = R
        new_transform[:3, 3] = t
        transform = new_transform @ transform

        mean_error = np.mean(distances[valid])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return transform


def voxel_dedup(
    base: GaussianSplat,
    new: GaussianSplat,
    voxel_size: float,
    prefer_new: bool = True,
) -> GaussianSplat:
    """Merge two splats with voxel-grid deduplication.

    Within each voxel, keeps only the highest-quality Gaussian.
    When prefer_new=True, new Gaussians win ties over base ones.
    """
    all_positions = np.vstack([base.positions, new.positions])
    n_base = base.count
    n_new = new.count

    # Compute voxel indices
    voxel_idx = np.floor(all_positions / voxel_size).astype(np.int64)
    # Encode 3D voxel index to 1D key
    voxel_keys = voxel_idx[:, 0] * 73856093 ^ voxel_idx[:, 1] * 19349663 ^ voxel_idx[:, 2] * 83492791

    # Quality scores
    base_quality = _quality_score(base)
    new_quality = _quality_score(new)
    all_quality = np.concatenate([base_quality, new_quality])

    # For prefer_new, add a small bonus to new gaussians
    if prefer_new:
        all_quality[n_base:] *= 1.1

    # Group by voxel key and keep the best in each
    unique_keys, inverse = np.unique(voxel_keys, return_inverse=True)
    best_in_voxel = np.full(len(unique_keys), -1, dtype=np.int64)
    for i in range(len(all_quality)):
        v = inverse[i]
        if best_in_voxel[v] == -1 or all_quality[i] > all_quality[best_in_voxel[v]]:
            best_in_voxel[v] = i

    keep_indices = best_in_voxel[best_in_voxel >= 0]
    keep_indices.sort()

    # Reconstruct the merged splat
    is_base = keep_indices < n_base
    is_new = ~is_base

    merged = GaussianSplat(
        positions=all_positions[keep_indices],
        sh0=np.vstack([base.sh0, new.sh0])[keep_indices],
        opacity=np.concatenate([base.opacity, new.opacity])[keep_indices],
        scales=np.vstack([base.scales, new.scales])[keep_indices],
        rotations=np.vstack([base.rotations, new.rotations])[keep_indices],
        sh_rest=(
            np.vstack([base.sh_rest, new.sh_rest])[keep_indices]
            if base.sh_rest is not None and new.sh_rest is not None
            else None
        ),
    )
    return merged


def cull_near_new(
    base: GaussianSplat,
    new: GaussianSplat,
    cull_radius: float,
) -> GaussianSplat:
    """Remove base Gaussians that are within cull_radius of any new Gaussian.

    This handles dynamic objects: if a person appears in the new capture,
    their Gaussians replace the stale background Gaussians that were there before.
    """
    if cull_radius <= 0 or base.count == 0 or new.count == 0:
        return base

    from scipy.spatial import cKDTree

    new_tree = cKDTree(new.positions)
    # Query which base points have a new point within cull_radius
    distances, _ = new_tree.query(base.positions, k=1)
    keep = distances > cull_radius
    culled = int(np.sum(~keep))

    return GaussianSplat(
        positions=base.positions[keep],
        sh0=base.sh0[keep],
        opacity=base.opacity[keep],
        scales=base.scales[keep],
        rotations=base.rotations[keep],
        sh_rest=base.sh_rest[keep] if base.sh_rest is not None else None,
    )


def filter_opacity(splat: GaussianSplat, threshold: float) -> GaussianSplat:
    """Remove Gaussians with sigmoid opacity below threshold."""
    if threshold <= 0:
        return splat
    opacity = _sigmoid(splat.opacity)
    keep = opacity >= threshold
    return GaussianSplat(
        positions=splat.positions[keep],
        sh0=splat.sh0[keep],
        opacity=splat.opacity[keep],
        scales=splat.scales[keep],
        rotations=splat.rotations[keep],
        sh_rest=splat.sh_rest[keep] if splat.sh_rest is not None else None,
    )


def merge_splats(
    base: GaussianSplat,
    new: GaussianSplat,
    config: MergeConfig = MergeConfig(),
) -> GaussianSplat:
    """Merge two Gaussian splats into one.

    Steps:
    1. Align new to base via ICP (if icp_max_distance > 0)
    2. Cull base Gaussians near new ones (dynamic object replacement)
    3. Voxel deduplicate overlapping Gaussians
    4. Filter low-opacity Gaussians
    """
    # Step 1: Alignment
    if config.icp_max_distance > 0:
        transform = icp_align(
            new.positions.astype(np.float64),
            base.positions.astype(np.float64),
            max_distance=config.icp_max_distance,
        )
        if not np.allclose(transform, np.eye(4)):
            new = _apply_transform(new, transform)

    # Step 2: Dynamic object culling
    base_culled = cull_near_new(base, new, config.dynamic_cull_radius)

    # Step 3: Voxel deduplication
    merged = voxel_dedup(base_culled, new, config.voxel_size, prefer_new=config.prefer_new)

    # Step 4: Opacity filter
    merged = filter_opacity(merged, config.opacity_threshold)

    return merged


def merge_ply_files(
    base_path: str,
    new_path: str,
    output_path: str,
    config: MergeConfig = MergeConfig(),
) -> dict:
    """Merge two 3DGS PLY files and write the result.

    Returns a summary dict with counts and timing info.
    """
    import time

    from pathlib import Path

    started = time.monotonic()
    base = read_gaussian_ply(Path(base_path))
    new = read_gaussian_ply(Path(new_path))
    base_count = base.count
    new_count = new.count

    merged = merge_splats(base, new, config)
    merged_count = merged.count

    write_gaussian_ply(merged, Path(output_path))
    elapsed = round(time.monotonic() - started, 2)

    return {
        "base_gaussians": base_count,
        "new_gaussians": new_count,
        "merged_gaussians": merged_count,
        "culled": base_count + new_count - merged_count,
        "elapsed_seconds": elapsed,
    }
