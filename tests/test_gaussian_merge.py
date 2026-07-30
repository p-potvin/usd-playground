"""Tests for gaussian_merge.py — incremental splat merging."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from vaultwares_studio.gaussian_merge import (
    MergeConfig,
    GaussianSplat,
    _sigmoid,
    _quality_score,
    voxel_dedup,
    cull_near_new,
    filter_opacity,
    merge_splats,
)
from vaultwares_studio.splat_io import write_gaussian_ply, read_gaussian_ply


def _make_splat(n: int, offset: float = 0.0, opacity: float = 5.0) -> GaussianSplat:
    """Create a simple test splat with n gaussians."""
    rng = np.random.default_rng(42)
    return GaussianSplat(
        positions=rng.uniform(-1, 1, (n, 3)).astype(np.float32) + offset,
        sh0=rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32),
        opacity=np.full(n, opacity, dtype=np.float32),
        scales=np.full((n, 3), -3.0, dtype=np.float32),  # log(0.05)
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        sh_rest=None,
    )


def test_sigmoid():
    assert _sigmoid(np.array([0.0])) == pytest.approx(0.5)
    assert _sigmoid(np.array([100.0])) > 0.99
    assert _sigmoid(np.array([-100.0])) < 0.01


def test_quality_score():
    splat = _make_splat(10)
    scores = _quality_score(splat)
    assert scores.shape == (10,)
    assert np.all(scores > 0)  # opacity > 0, volume > 0


def test_voxel_dedup_removes_overlaps():
    base = _make_splat(100, offset=0.0)
    new = _make_splat(100, offset=0.0)  # Same positions → heavy overlap
    merged = voxel_dedup(base, new, voxel_size=0.1, prefer_new=True)
    assert merged.count < base.count + new.count
    assert merged.count > 0


def test_voxel_dedup_no_overlap():
    base = _make_splat(50, offset=0.0)
    new = _make_splat(50, offset=100.0)  # Far away → no overlap
    merged = voxel_dedup(base, new, voxel_size=0.1)
    # Allow minor collisions within each set (RNG may place 2 points in same voxel)
    assert merged.count >= base.count + new.count - 5
    assert merged.count > base.count


def test_cull_near_new():
    base = _make_splat(100, offset=0.0)
    new = _make_splat(10, offset=0.0)  # Overlapping positions
    culled = cull_near_new(base, new, cull_radius=0.5)
    assert culled.count < base.count
    assert culled.count > 0


def test_cull_near_new_zero_radius():
    base = _make_splat(50)
    new = _make_splat(50, offset=100.0)
    culled = cull_near_new(base, new, cull_radius=0.0)
    assert culled.count == base.count  # No culling


def test_filter_opacity():
    splat = GaussianSplat(
        positions=np.zeros((5, 3), dtype=np.float32),
        sh0=np.zeros((5, 3), dtype=np.float32),
        opacity=np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32),
        scales=np.full((5, 3), -3.0, dtype=np.float32),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)).astype(np.float32),
    )
    filtered = filter_opacity(splat, threshold=0.5)
    # sigmoid(0) = 0.5, sigmoid(1) ≈ 0.73, sigmoid(10) ≈ 1.0
    assert filtered.count == 3  # opacity >= 0.5 → indices 2, 3, 4


def test_merge_splats_basic():
    base = _make_splat(50, offset=0.0)
    new = _make_splat(50, offset=10.0)  # Far apart, no overlap
    config = MergeConfig(
        voxel_size=0.1,
        icp_max_distance=0.0,  # Skip ICP
        dynamic_cull_radius=0.0,  # No culling
        opacity_threshold=0.0,
    )
    merged = merge_splats(base, new, config)
    # Allow minor within-set voxel collisions
    assert merged.count >= base.count + new.count - 5


def test_merge_splats_with_culling():
    base = _make_splat(100, offset=0.0)
    new = _make_splat(20, offset=0.0)  # Overlapping
    config = MergeConfig(
        voxel_size=0.05,
        icp_max_distance=0.0,
        dynamic_cull_radius=0.3,
        opacity_threshold=0.0,
    )
    merged = merge_splats(base, new, config)
    assert merged.count < base.count + new.count
    assert merged.count > 0


def test_merge_ply_roundtrip(tmp_path: Path):
    base = _make_splat(30, offset=0.0)
    new = _make_splat(30, offset=5.0)
    base_path = tmp_path / "base.ply"
    new_path = tmp_path / "new.ply"
    out_path = tmp_path / "merged.ply"
    write_gaussian_ply(base, base_path)
    write_gaussian_ply(new, new_path)

    from vaultwares_studio.gaussian_merge import merge_ply_files
    summary = merge_ply_files(
        str(base_path), str(new_path), str(out_path),
        config=MergeConfig(icp_max_distance=0.0, dynamic_cull_radius=0.0),
    )
    assert summary["base_gaussians"] == 30
    assert summary["new_gaussians"] == 30
    assert summary["merged_gaussians"] > 0
    assert out_path.exists()

    # Verify the merged PLY is readable
    merged = read_gaussian_ply(out_path)
    assert merged.count > 0


def test_da3_incremental_preset_exists():
    from vaultwares_studio.presets import PRESETS, SfmMethod
    assert "da3-incremental" in PRESETS
    p = PRESETS["da3-incremental"]
    assert p.sfm_method == SfmMethod.DA3
    assert p.da3_direct_gs is True
    assert p.da3_model == "depth-anything/DA3-GIANT-1.1"
