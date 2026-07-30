"""Video intake stage: inspect the input video and initialize the job profile."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import DigitalTwinStudioRunner, StageRecord


def run(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> None:
    source_video = Path(ctx.manifest.source_video)
    if not source_video.exists():
        raise FileNotFoundError(f"Missing source video: {source_video}")
    metadata = {
        "sourceVideo": str(source_video),
        "fileSizeBytes": source_video.stat().st_size,
        "executionProfile": ctx.manifest.execution_profile,
        "mode": ctx.manifest.mode,
    }
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(source_video),
        ]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode == 0 and completed.stdout:
            metadata["probe"] = json.loads(completed.stdout)
    metadata_path = ctx.root / "input_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    stage.metadata = metadata
    stage.message = "Video metadata captured and job initialized."
    ctx._add_artifact(stage, "Input Metadata", "json", metadata_path, "Video intake metadata.")
