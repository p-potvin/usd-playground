"""Frame extraction stage: extract frames, keep previews, capture sampling metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import DigitalTwinStudioRunner, StageRecord


def run(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> None:
    from ..pipeline import (
        DEPENDENCY_INSTALL_HINTS,
        FRAME_EXTRACT_TARGET,
        FRAME_KEEP_TARGET,
        compute_extraction_fps,
        list_frames,
        resolve_binary,
        save_job_manifest,
    )
    from ..presets import get_preset
    from ..pipeline import StageState

    ffmpeg = resolve_binary("ffmpeg")
    if ffmpeg is None:
        stage.state = StageState.NEEDS_INSTALL.value
        stage.message = DEPENDENCY_INSTALL_HINTS["ffmpeg"]
        save_job_manifest(ctx.manifest)
        raise RuntimeError("ffmpeg is required for frame extraction.")
    ctx.frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in list_frames(ctx.frames_dir):
        stale.unlink(missing_ok=True)
    duration = None
    intake = ctx.stage_for("video_intake")
    try:
        duration = float(intake.metadata["probe"]["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        pass
    preset = get_preset(ctx.manifest.metadata.get("preset"))
    if preset.unrestricted_frames and preset.frame_cap > 0:
        extract_target = preset.frame_cap
        keep_target = preset.frame_cap
    else:
        extract_target = FRAME_EXTRACT_TARGET
        keep_target = FRAME_KEEP_TARGET
    fps = compute_extraction_fps(duration, target_frames=extract_target)
    if preset.unrestricted_frames:
        ctx.log(
            f"Sampling at {fps} fps (duration: {duration or 'unknown'}s); "
            f"lab mode — no sharpness prune, hard-cap {preset.frame_cap}"
        )
    else:
        ctx.log(
            f"Sampling at {fps} fps (duration: {duration or 'unknown'}s); "
            f"keeping the sharpest ~{keep_target} frames"
        )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        ctx.manifest.source_video,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(ctx.frames_dir / "frame_%05d.jpg"),
    ]
    ctx._run_command(cmd, "Frame extraction failed.", timeout_seconds=1800)
    extracted_count = len(list_frames(ctx.frames_dir))
    kept_count = extracted_count
    if preset.unrestricted_frames:
        if preset.frame_cap > 0 and extracted_count > preset.frame_cap:
            for stale in list_frames(ctx.frames_dir)[preset.frame_cap:]:
                stale.unlink(missing_ok=True)
            extracted_count = preset.frame_cap
            kept_count = preset.frame_cap
            ctx.log(f"Lab mode: hard-capped to {preset.frame_cap} frames (no sharpness prune).")
    elif extracted_count > FRAME_KEEP_TARGET:
        try:
            from ..frame_selection import prune_to_sharpest

            extracted_count, kept_count = prune_to_sharpest(ctx.frames_dir, FRAME_KEEP_TARGET)
            ctx.log(f"Kept {kept_count}/{extracted_count} sharpest frames (motion-blur filter).")
        except ImportError:
            ctx.log("OpenCV unavailable; skipping blur-aware frame selection.")
    base_frames_zip = ctx.manifest.metadata.get("refine_base_frames_zip")
    extras_prefix_base = "clip" if base_frames_zip else "extra"
    if base_frames_zip:
        for source in list_frames(ctx.frames_dir):
            if not source.name.startswith("clip"):
                shutil.move(str(source), str(ctx.frames_dir / f"clip0_{source.name}"))
    for index, extra_video in enumerate(ctx.manifest.metadata.get("extra_videos", []) or []):
        offset = 1 if base_frames_zip else 0
        extract_extra_video(
            ctx,
            extra_video,
            prefix=f"{extras_prefix_base}{index + offset}_",
            ffmpeg=ffmpeg,
        )
    if base_frames_zip:
        merge_refine_base_frames(ctx, Path(base_frames_zip))
    frame_paths = list_frames(ctx.frames_dir)
    if not frame_paths:
        raise RuntimeError("No frames were extracted.")
    preview_manifest = ctx.frames_dir / "frames_manifest.json"
    preview_manifest.write_text(
        json.dumps(
            {
                "frameCount": len(frame_paths),
                "sampleFrames": [str(path) for path in frame_paths[:6]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stage.metadata = {"frameCount": len(frame_paths), "fps": fps, "extracted": extracted_count}
    stage.message = f"Extracted {len(frame_paths)} frames."
    ctx._add_artifact(stage, "Frames Manifest", "json", preview_manifest, "Frame extraction summary.")
    for index, frame in enumerate(frame_paths[:3], start=1):
        ctx._add_artifact(stage, f"Frame Preview {index}", "image", frame, "Sample extracted frame.")


def extract_extra_video(
    ctx: "DigitalTwinStudioRunner", video_path: str, prefix: str, ffmpeg: str
) -> None:
    """Extract an additional clip into ctx.frames_dir, prefixed and pruned.

    Each extra clip gets its own temp dir for the ffmpeg dump and its own
    sharpness pass, so the prune budget is per-video. After pruning, files
    are moved into ctx.frames_dir as ``<prefix>frame_NNNNN.jpg``. We try
    to read duration from ffprobe; on failure (e.g. ffprobe not in PATH)
    we fall back to the same fps the primary chose.
    """
    from ..pipeline import (
        FRAME_EXTRACT_TARGET,
        FRAME_KEEP_TARGET,
        compute_extraction_fps,
        list_frames,
    )

    video = Path(video_path)
    if not video.exists():
        ctx.log(f"Extra video skipped: {video} does not exist.")
        return
    duration: float | None = None
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip()) if probe.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    fps = compute_extraction_fps(duration, target_frames=FRAME_EXTRACT_TARGET)
    with tempfile.TemporaryDirectory(prefix=f"vw-{prefix.strip('_')}-") as tmpdir:
        tmp_path = Path(tmpdir)
        ctx.log(f"Extracting extra video {video.name} at {fps} fps (duration: {duration or 'unknown'}s)")
        cmd = [
            ffmpeg, "-y", "-i", str(video),
            "-vf", f"fps={fps}", "-q:v", "2",
            str(tmp_path / "frame_%05d.jpg"),
        ]
        ctx._run_command(cmd, f"Extra-video extraction failed for {video.name}.", timeout_seconds=1800)
        raw_count = len(list_frames(tmp_path))
        kept_count = raw_count
        if raw_count > FRAME_KEEP_TARGET:
            try:
                from ..frame_selection import prune_to_sharpest

                raw_count, kept_count = prune_to_sharpest(tmp_path, FRAME_KEEP_TARGET)
            except ImportError:
                ctx.log("OpenCV unavailable; skipping blur-aware prune for extra video.")
        for source in list_frames(tmp_path):
            shutil.move(str(source), str(ctx.frames_dir / f"{prefix}{source.name}"))
        ctx.log(f"Extra video {video.name}: kept {kept_count}/{raw_count} frames (prefix={prefix}).")


def merge_refine_base_frames(ctx: "DigitalTwinStudioRunner", base_zip: Path) -> None:
    """Unpack the base job's frames into ctx.frames_dir with ORIGINAL names.

    The worker's --refine-mode looks up each image in the base
    colmap_database.db by name to know which features are already cached.
    Prefixing the base frames would make colmap see them as new images
    and re-extract features, defeating the point of refine. New-clip
    frames are prefixed (clip<N>_) on the extraction side instead.
    """
    from ..pipeline import list_frames

    if not base_zip.exists():
        ctx.log(f"Refine: base frames zip not found at {base_zip}; skipping merge.")
        return
    merged = 0
    with zipfile.ZipFile(base_zip) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            bare = Path(info.filename).name
            if not bare:
                continue
            target = ctx.frames_dir / bare
            with archive.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            merged += 1
    new_count = sum(
        1 for p in list_frames(ctx.frames_dir)
        if p.name.startswith("clip")
    )
    ctx.log(
        f"Refine: merged {merged} base frames into {ctx.frames_dir.name} "
        f"(new={new_count}, total={new_count + merged})."
    )
