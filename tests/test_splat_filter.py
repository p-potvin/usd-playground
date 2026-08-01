"""splat_filter: clamp non-finite opacity, drop the degenerate position tail.

Both failure modes are real, observed on the first green da3-draft run
(2026-07-30): DA3-GIANT returned opacity logits containing +inf, and ~1% of
gaussians sat far enough out to stretch the bounding box 416x and flip the PCA
gravity estimate so the scene spawned upside down.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaultwares_studio.splat_filter import (
    OPACITY_LOGIT_LIMIT,
    clamp_opacity,
    sanitize_splat,
)
from vaultwares_studio.splat_io import GaussianSplat


def _splat(positions: np.ndarray, opacity: np.ndarray | None = None) -> GaussianSplat:
    n = len(positions)
    return GaussianSplat(
        positions=positions.astype(np.float32),
        sh0=np.zeros((n, 3), dtype=np.float32),
        opacity=(np.zeros(n, dtype=np.float32) if opacity is None else opacity.astype(np.float32)),
        scales=np.zeros((n, 3), dtype=np.float32),
        rotations=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        sh_rest=None,
    )


def _core_with_tail(n_core: int = 1000, n_tail: int = 10, seed: int = 0):
    """A dense core plus a handful of far-flung outliers — the DA3 shape."""
    rng = np.random.default_rng(seed)
    core = rng.normal(0.0, 1.0, size=(n_core, 3))
    tail = rng.normal(0.0, 1.0, size=(n_tail, 3)) + rng.choice([-500, 500], size=(n_tail, 3))
    return np.vstack([core, tail])


# -- opacity ------------------------------------------------------------------


def test_clamp_opacity_replaces_non_finite():
    opacity = np.array([0.0, np.inf, -np.inf, np.nan, 5.0])
    splat, touched = clamp_opacity(_splat(np.zeros((5, 3)), opacity))

    assert touched == 3
    assert np.isfinite(splat.opacity).all()
    assert splat.opacity[1] == pytest.approx(OPACITY_LOGIT_LIMIT)
    # -inf and nan both become fully transparent: an invisible gaussian is a
    # safer default than an opaque one when the model had no opinion.
    assert splat.opacity[2] == pytest.approx(-OPACITY_LOGIT_LIMIT)
    assert splat.opacity[3] == pytest.approx(-OPACITY_LOGIT_LIMIT)
    # Finite values pass through untouched.
    assert splat.opacity[0] == pytest.approx(0.0)
    assert splat.opacity[4] == pytest.approx(5.0)


def test_clamp_opacity_is_a_no_op_when_already_finite():
    opacity = np.array([-3.0, 0.0, 3.0])
    splat, touched = clamp_opacity(_splat(np.zeros((3, 3)), opacity))
    assert touched == 0
    assert splat.opacity == pytest.approx(opacity)


def test_inf_opacity_would_poison_a_mean_without_the_clamp():
    """The reason this matters: gaussian_merge scores quality with a mean."""
    opacity = np.array([0.0, np.inf, 1.0])
    assert not np.isfinite(opacity.mean())

    splat, _ = clamp_opacity(_splat(np.zeros((3, 3)), opacity))
    assert np.isfinite(splat.opacity.mean())


# -- spatial filter -----------------------------------------------------------


def test_sanitize_drops_the_outlier_tail_and_shrinks_extent():
    positions = _core_with_tail()
    splat, report = sanitize_splat(_splat(positions), keep_quantile=0.95)

    assert report.count_before == 1010
    assert report.count_after < report.count_before
    # The whole point: extent collapses by orders of magnitude.
    assert report.radius_after < report.radius_before / 50


def test_sanitize_keeps_the_bulk_of_the_core():
    positions = _core_with_tail(n_core=1000, n_tail=10)
    _, report = sanitize_splat(_splat(positions), keep_quantile=0.95)
    # A per-axis 95% envelope compounds across 3 axes, so expect ~86% kept,
    # not 95%. Guard the floor so an over-aggressive filter is caught.
    assert report.count_after > 800


def test_keep_quantile_one_clamps_opacity_but_keeps_every_gaussian():
    """Trained splatfacto output must not pay for the spatial filter."""
    positions = _core_with_tail()
    opacity = np.zeros(len(positions))
    opacity[0] = np.inf
    splat, report = sanitize_splat(_splat(positions, opacity), keep_quantile=1.0)

    assert report.count_after == report.count_before
    assert report.dropped == 0
    assert report.opacity_clamped == 1
    assert np.isfinite(splat.opacity).all()


def test_all_attribute_arrays_stay_aligned_after_filtering():
    """A mask applied to some arrays but not others silently corrupts colour."""
    positions = _core_with_tail()
    splat, report = sanitize_splat(_splat(positions), keep_quantile=0.95)

    n = report.count_after
    assert splat.positions.shape == (n, 3)
    assert splat.sh0.shape == (n, 3)
    assert splat.opacity.shape == (n,)
    assert splat.scales.shape == (n, 3)
    assert splat.rotations.shape == (n, 4)


def test_rejects_nonsense_quantiles():
    positions = _core_with_tail()
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="keep_quantile"):
            sanitize_splat(_splat(positions), keep_quantile=bad)


# -- the bug this was built for ----------------------------------------------


def test_outlier_tail_flips_the_gravity_estimate_and_filtering_fixes_it():
    """Regression guard for the upside-down spawn.

    A ground-plane-ish cloud whose true up is +Y, plus a sparse tail. PCA on the
    raw cloud picks the wrong axis; after filtering it recovers +Y.
    """
    from vaultwares_studio.gravity_align import compute_alignment

    rng = np.random.default_rng(7)
    # Wide and flat in XZ, thin in Y => smallest-variance axis is Y.
    core = np.column_stack([
        rng.normal(0, 10.0, 4000),
        rng.normal(0, 0.4, 4000),
        rng.normal(0, 10.0, 4000),
    ])
    # Sparse tail displaced hard along +Y, enough to dominate the covariance.
    tail = np.column_stack([
        rng.normal(0, 5.0, 60),
        rng.normal(900.0, 50.0, 60),
        rng.normal(0, 5.0, 60),
    ])
    positions = np.vstack([core, tail])

    raw = compute_alignment(positions)
    filtered_splat, _ = sanitize_splat(_splat(positions), keep_quantile=0.95)
    filtered = compute_alignment(np.asarray(filtered_splat.positions, dtype=np.float64))

    # The tail wrecks the raw estimate; the filtered one locks onto Y.
    assert abs(filtered.up_before[1]) > 0.99
    assert abs(raw.up_before[1]) < abs(filtered.up_before[1])


def test_alignment_tilt_is_never_nan_for_an_aligned_cloud():
    """arccos(abs(dot)) returned nan when dot rounded past 1.0."""
    from vaultwares_studio.gravity_align import compute_alignment

    rng = np.random.default_rng(3)
    positions = np.column_stack([
        rng.normal(0, 10.0, 2000),
        rng.normal(0, 0.01, 2000),  # near-perfectly flat => up lands on Y
        rng.normal(0, 10.0, 2000),
    ])
    result = compute_alignment(positions)
    assert np.isfinite(result.angle_from_y_degrees)
