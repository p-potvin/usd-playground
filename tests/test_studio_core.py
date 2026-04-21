from pathlib import Path

from studio_core.camera_director import build_camera_bundle
from studio_core.integration import build_vaultflows_workflow
from studio_core.pipeline import (
    DEFAULT_SOURCE_VIDEO,
    StageState,
    create_job_manifest,
    load_job_manifest,
)


def test_create_job_manifest_writes_resumable_job():
    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    manifest_path = Path(manifest.output_dir) / "manifest.json"

    assert manifest_path.exists()
    loaded = load_job_manifest(manifest_path)
    assert loaded.job_id == manifest.job_id
    assert loaded.state == StageState.QUEUED.value
    assert [stage.key for stage in loaded.stages] == [
        "video_intake",
        "frame_extraction",
        "reconstruction",
        "usd_cameras",
        "cosmos_output",
    ]


def test_camera_bundle_contains_presets_and_prompt_plan():
    bundle = build_camera_bundle("show me the desk from the doorway, then orbit left and rise")

    assert bundle["presets"]
    assert bundle["promptPlan"]
    assert any(shot["name"] == "Doorway Start" for shot in bundle["promptPlan"])
    assert any(shot["name"] == "Orbit Move" for shot in bundle["promptPlan"])
    assert any(shot["name"] == "Rise Shot" for shot in bundle["promptPlan"])
    assert len(bundle["allShots"]) >= len(bundle["presets"])


def test_vaultflows_workflow_export_shape():
    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    workflow = build_vaultflows_workflow(manifest)

    assert workflow["id"] == manifest.job_id
    assert workflow["category"] == "Digital Twin"
    assert workflow["pin"] is True
    assert workflow["favorite"] is True
    assert len(workflow["steps"]) == len(manifest.stages)
    assert workflow["steps"][0]["id"] == "video_intake"
