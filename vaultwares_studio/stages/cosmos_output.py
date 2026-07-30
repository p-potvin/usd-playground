"""Cosmos output stage: write Cosmos annotations and produce the final walkthrough video."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import DigitalTwinStudioRunner, StageRecord


def run(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> None:
    from ..pipeline import resolve_binary, save_job_manifest
    from ..runners import CostDeniedError, StageCancelledError, StageContext, record_spend

    ctx.cosmos_dir.mkdir(parents=True, exist_ok=True)
    ctx.deliverables_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = ctx.cosmos_dir / "cosmos_annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "sourceStage": str(ctx.usd_stage_path),
                "model": "cosmos-reason2 (placeholder-safe)",
                "annotations": [
                    {"label": "environment", "path": "/World/Environment"},
                    {"label": "digital-twin", "path": "/World/DigitalTwin"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    transfer_path = ctx.cosmos_dir / "cosmos_transfer_notes.txt"
    transfer_path.write_text(
        "Placeholder-safe Cosmos Transfer notes.\n"
        f"Source stage: {ctx.usd_stage_path}\n",
        encoding="utf-8",
    )
    rendered_remotely = False
    if ctx.remote_runner is not None:
        try:
            rendered_remotely = _run_remote_render(ctx, stage)
        except StageCancelledError:
            raise
        except CostDeniedError as exc:
            ctx.log(f"{exc} Falling back to the preview slideshow.")
        except Exception as exc:  # noqa: BLE001
            if ctx.strict_mode:
                raise
            ctx.log(f"Remote walkthrough render failed, using preview slideshow: {exc}")

    if rendered_remotely:
        ctx.manifest.walkthrough_video = str(ctx.splat_walkthrough_path)
        ctx._add_artifact(
            stage, "Walkthrough Video", "video", ctx.splat_walkthrough_path,
            "Splat-rendered camera-path walkthrough.",
        )
        stage.message = "Cosmos artifacts written; splat walkthrough rendered along the camera path."
    else:
        _build_walkthrough_video(ctx)
        ctx.manifest.walkthrough_video = str(ctx.walkthrough_path)
        ctx._add_artifact(stage, "Walkthrough Video", "video", ctx.walkthrough_path, "Final MP4 walkthrough.")
        stage.message = "Cosmos artifacts written and walkthrough video rendered."
    ctx._add_artifact(stage, "Cosmos Annotation", "json", annotation_path, "Reason model output.")
    ctx._add_artifact(stage, "Cosmos Transfer Notes", "text", transfer_path, "Transfer model notes.")


def _run_remote_render(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> bool:
    """Render the authored camera path with the trained splat (HF Job).

    Needs the recon stage's checkpoint bundle in the artifact dataset
    (model.zip + processed_min.zip, produced by remote reconstructions
    from M2 onward) and the camera_path.json authored by camera staging.
    """
    from ..runners import StageContext, record_spend

    remote_out = ctx.root / "reconstruction" / "remote_out"
    bundle_ok = (remote_out / "model.zip").exists() and (remote_out / "processed_min.zip").exists()
    if not bundle_ok or not ctx.camera_render_path.exists():
        ctx.log(
            "No render bundle for this job (model.zip + processed_min.zip + camera_path.json) — "
            "re-run reconstruction to bank one. Using the preview slideshow."
        )
        return False
    runner_config = getattr(ctx.remote_runner, "config", None)
    image_name = getattr(runner_config, "worker_image", "") if runner_config else ""
    if not image_name or image_name.startswith("python:"):
        raise RuntimeError("Remote worker image not configured.")

    dataset_prefix = f"jobs/{ctx.manifest.job_id}/reconstruction/out"
    ctx_obj = StageContext(
        job_dir=ctx.root,
        job_id=ctx.manifest.job_id,
        stage_key="walkthrough_render",
        params={
            "image": image_name,
            "image_has_hub": True,
            "flavor": "l4x1",
            "est_minutes": 8,
            "timeout_seconds": 2400,
            "command": ["python", "/opt/vw/render_entrypoint.py"],
            "extra_repo_inputs": [
                f"{dataset_prefix}/model.zip",
                f"{dataset_prefix}/processed_min.zip",
            ],
        },
        inputs=[ctx.camera_render_path],
        expected_outputs=[ctx.splat_walkthrough_path],
        log=ctx.log,
        cancel=ctx.cancel_token,
    )
    result = ctx.remote_runner.run(ctx_obj)
    if result.metadata:
        record_spend(ctx.manifest, "cosmos_output", result.metadata)
    return ctx.splat_walkthrough_path.exists()


def _build_walkthrough_video(ctx: "DigitalTwinStudioRunner") -> None:
    from ..pipeline import resolve_binary

    ffmpeg = resolve_binary("ffmpeg")
    preview_paths = sorted(ctx.cameras_dir.glob("shot_*.png"))
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render the final walkthrough video.")
    if not preview_paths:
        raise RuntimeError("Camera previews must exist before rendering the walkthrough video.")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        "1",
        "-i",
        str(ctx.cameras_dir / "shot_%02d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(ctx.walkthrough_path),
    ]
    ctx._run_command(cmd, "Walkthrough render failed.", timeout_seconds=1800)
