"""Sanitise raw predicted splats before anything downstream consumes them.

Feed-forward models (DA3's Gaussian head, and DA3-Streaming's fused point cloud)
emit two kinds of garbage that trained splatfacto output does not:

**Non-finite opacity.** DA3-GIANT returns opacity logits containing ``+inf``.
Harmless through a sigmoid when rendering, but ``nan``-poisons any mean or sum —
including ``gaussian_merge``'s quality scoring, which is what ``da3-incremental``
ranks gaussians with.

**A long tail of degenerate positions.** Background and sky pixels have
effectively no depth signal, so they back-project to enormous distances. On the
first green ``da3-draft`` run (2026-07-30) roughly 1% of gaussians sat beyond 10x
the core radius and the furthest was 416x, stretching the bounding box from ~1.4
to ~581. That one percent broke three things at once: the viewer's auto-framing
put the camera inside the geometry, the packed ``.splat`` wasted precision range
on empty space, and — worst — PCA gravity alignment picked the wrong axis and
spawned the scene upside down.

Note on the filter statistic: standard deviation is the wrong tool here. With 1%
of the mass at 400x the core, sigma is enormous and a 2-sigma cut keeps virtually
everything. Quantiles are robust to exactly this shape, so that is what we use;
the precise cut point barely matters (95 / 98 / 99 all recover the same up-vector
to three decimals).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .splat_io import GaussianSplat

# Opacity is stored as a logit. sigmoid(+-12) is within 1e-5 of 1/0, so this
# clamp is visually lossless while keeping every downstream mean/sum finite.
OPACITY_LOGIT_LIMIT = 12.0
DEFAULT_KEEP_QUANTILE = 0.95


@dataclass(frozen=True)
class FilterReport:
    """What sanitise_splat actually changed, for logging and summary.json."""

    count_before: int
    count_after: int
    opacity_clamped: int
    radius_before: float
    radius_after: float

    @property
    def dropped(self) -> int:
        return self.count_before - self.count_after

    def as_dict(self) -> dict:
        return {
            "count_before": self.count_before,
            "count_after": self.count_after,
            "dropped": self.dropped,
            "opacity_clamped": self.opacity_clamped,
            "radius_before": round(self.radius_before, 3),
            "radius_after": round(self.radius_after, 3),
        }


def clamp_opacity(splat: GaussianSplat) -> tuple[GaussianSplat, int]:
    """Replace non-finite opacity logits with a finite equivalent.

    Returns the splat and how many values were touched. ``+inf`` becomes
    ``+OPACITY_LOGIT_LIMIT`` (still fully opaque), ``-inf`` and ``nan`` become
    ``-OPACITY_LOGIT_LIMIT`` (fully transparent) — nan is treated as "no
    opinion", and an invisible gaussian is safer than an opaque one.
    """
    opacity = np.asarray(splat.opacity, dtype=np.float32)
    bad = ~np.isfinite(opacity)
    touched = int(bad.sum())
    if touched == 0:
        return splat, 0
    cleaned = opacity.copy()
    cleaned[np.isposinf(opacity)] = OPACITY_LOGIT_LIMIT
    cleaned[np.isneginf(opacity) | np.isnan(opacity)] = -OPACITY_LOGIT_LIMIT
    cleaned = np.clip(cleaned, -OPACITY_LOGIT_LIMIT, OPACITY_LOGIT_LIMIT)
    return replace(splat, opacity=cleaned), touched


def _keep_mask(positions: np.ndarray, keep_quantile: float) -> np.ndarray:
    """Per-axis quantile envelope around the median.

    An axis-aligned box rather than a radial cut: scenes are usually much wider
    than they are tall (this backyard is ~4x), and a sphere sized to keep the
    horizontal extent would keep a column of sky with it.
    """
    tail = (1.0 - keep_quantile) / 2.0
    low = np.percentile(positions, tail * 100.0, axis=0)
    high = np.percentile(positions, (1.0 - tail) * 100.0, axis=0)
    return np.all((positions >= low) & (positions <= high), axis=1)


def sanitize_splat(
    splat: GaussianSplat,
    keep_quantile: float = DEFAULT_KEEP_QUANTILE,
) -> tuple[GaussianSplat, FilterReport]:
    """Clamp opacity and drop the degenerate position tail.

    ``keep_quantile`` is the fraction of gaussians retained per axis. Pass 1.0
    to clamp opacity only — trained splatfacto output does not need the spatial
    filter and should not pay for it.
    """
    if not 0.0 < keep_quantile <= 1.0:
        raise ValueError(f"keep_quantile must be in (0, 1], got {keep_quantile}")

    positions = np.asarray(splat.positions, dtype=np.float64)
    centre = np.median(positions, axis=0)
    radius_before = float(np.percentile(np.linalg.norm(positions - centre, axis=1), 100.0))

    splat, opacity_clamped = clamp_opacity(splat)
    count_before = splat.count

    if keep_quantile >= 1.0:
        return splat, FilterReport(
            count_before=count_before,
            count_after=count_before,
            opacity_clamped=opacity_clamped,
            radius_before=radius_before,
            radius_after=radius_before,
        )

    mask = _keep_mask(positions, keep_quantile)
    if not mask.any():
        raise ValueError("Spatial filter would drop every gaussian")

    filtered = replace(
        splat,
        positions=splat.positions[mask],
        sh0=splat.sh0[mask],
        opacity=splat.opacity[mask],
        scales=splat.scales[mask],
        rotations=splat.rotations[mask],
        sh_rest=None if splat.sh_rest is None else splat.sh_rest[mask],
    )
    kept = positions[mask]
    radius_after = float(
        np.percentile(np.linalg.norm(kept - np.median(kept, axis=0), axis=1), 100.0)
    )
    return filtered, FilterReport(
        count_before=count_before,
        count_after=filtered.count,
        opacity_clamped=opacity_clamped,
        radius_before=radius_before,
        radius_after=radius_after,
    )
