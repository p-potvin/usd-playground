"""Run the da3-stream preset (DA3-Streaming SfM over every frame).

Reuses an existing job's extracted frames rather than re-extracting, so repeat
attempts don't re-upload 251MB or re-run ffmpeg.

Usage:
    .venv\\Scripts\\python.exe tools\\queue_da3_stream.py <source-job-id> [scheduling-timeout-s]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vaultwares_studio.pipeline import (  # noqa: E402
    DigitalTwinStudioRunner,
    create_job_manifest,
    load_job_manifest,
    save_job_manifest,
)
from vaultwares_studio.runners.hf_jobs import HfJobsConfig, HfJobsStageRunner  # noqa: E402


def _job_dir(job_id: str) -> Path:
    for base in (ROOT / "data" / "jobs", Path("D:/vaultwares-studio-jobs/data/jobs")):
        if (base / job_id / "manifest.json").exists():
            return base / job_id
    raise SystemExit(f"No manifest.json found for {job_id}")


def main() -> int:
    source_id = sys.argv[1]
    scheduling_timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0

    source = load_job_manifest(_job_dir(source_id) / "manifest.json")
    source_frames = Path(source.output_dir) / "frames"
    frames = sorted(source_frames.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No frames in {source_frames}")

    config = HfJobsConfig.load()
    if not config.enabled:
        raise SystemExit("Remote compute not enabled (data/remote_compute.json).")

    manifest = create_job_manifest(Path(source.source_video))
    manifest.metadata["preset"] = "da3-stream"
    manifest.metadata["flavor_scheduling_timeout_seconds"] = scheduling_timeout

    dest_frames = Path(manifest.output_dir) / "frames"
    dest_frames.mkdir(parents=True, exist_ok=True)
    for f in frames:
        shutil.copyfile(f, dest_frames / f.name)

    # Frames are already present, so mark the upstream stages done and run only
    # reconstruction — no ffmpeg, no re-extraction.
    from vaultwares_studio.pipeline import StageState

    for stage in manifest.stages:
        if stage.key in ("video_intake", "frame_extraction"):
            stage.state = StageState.COMPLETE.value
    save_job_manifest(manifest)

    print(f"[stream] job_id={manifest.job_id} preset=da3-stream frames={len(frames)}")
    print(f"[stream] scheduling_timeout={scheduling_timeout:.0f}s per flavor")

    runner = DigitalTwinStudioRunner(
        manifest,
        lambda m: print(m, flush=True),
        strict_mode=True,
        remote_runner=HfJobsStageRunner(
            config=config,
            confirm_cost=lambda est: print(f"[stream] cost approved: {est.summary()}") or True,
        ),
    )
    runner.run_stage("reconstruction")

    recon = Path(manifest.output_dir) / "reconstruction"
    for label, path in (
        ("cloud.ply", recon / "cloud.ply"),
        ("cloud.splat", recon / "cloud.splat"),
        ("summary.json", recon / "summary.json"),
    ):
        size = f"{path.stat().st_size / 1024 / 1024:.1f} MB" if path.exists() else "MISSING"
        print(f"[stream] {label}: {size}")
    summary = recon / "summary.json"
    if summary.exists():
        print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
