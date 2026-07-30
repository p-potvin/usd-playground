"""Camera staging stage: compose the USD stage and stage cameras."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import DigitalTwinStudioRunner, StageRecord


def run(ctx: "DigitalTwinStudioRunner", stage: "StageRecord") -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    from ..camera_director import build_camera_bundle
    from ..camera_paths import (
        CameraEntity,
        CameraKeyframe,
        author_usd_camera,
        build_visit_path,
        load_captured_entities,
        to_nerfstudio_camera_path,
    )
    from ..pipeline import DEFAULT_CAMERA_PROMPT, _slugify, save_job_manifest
    from ..pipeline import StageState

    already_paused = bool(stage.metadata.get("pausedForUserInput"))
    ctx.usd_dir.mkdir(parents=True, exist_ok=True)
    ctx.cameras_dir.mkdir(parents=True, exist_ok=True)
    stage_path = Usd.Stage.CreateNew(str(ctx.usd_stage_path))
    UsdGeom.SetStageUpAxis(stage_path, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage_path, "/World")
    stage_path.SetDefaultPrim(world.GetPrim())
    ground = UsdGeom.Cube.Define(stage_path, "/World/Environment/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(20.0, 0.1, 20.0))
    ground.AddTranslateOp().Set(Gf.Vec3f(0.0, -0.05, 0.0))
    light = UsdLux.DistantLight.Define(stage_path, "/World/Environment/Sun")
    light.CreateIntensityAttr(900.0)
    twin = UsdGeom.Xform.Define(stage_path, "/World/DigitalTwin")
    twin.GetPrim().CreateAttribute("sourceVideo", Sdf.ValueTypeNames.String, custom=True).Set(ctx.manifest.source_video)
    if ctx.recon_stage_path.exists():
        twin.GetPrim().GetReferences().AddReference(str(ctx.recon_stage_path))

    bundle = build_camera_bundle(str(ctx.manifest.metadata.get("cameraPrompt", DEFAULT_CAMERA_PROMPT)))
    UsdGeom.Xform.Define(stage_path, "/World/Navigation")

    center = _scene_center(ctx)
    entities: list[CameraEntity] = []
    for shot in bundle["allShots"]:
        entities.append(
            CameraEntity(
                name=shot["name"],
                source=shot["source"],
                keyframes=[
                    CameraKeyframe(t=0.0, position=list(shot["position"]), look_at=list(center))
                ],
            )
        )
    captured = load_captured_entities(Path(ctx.manifest.output_dir) / "usd" / "captured_cameras.json")
    entities.extend(captured)
    visit_path = build_visit_path(captured)
    if visit_path is not None:
        entities.append(visit_path)

    for entity in entities:
        author_usd_camera(stage_path, f"/World/Navigation/{_slugify(entity.name)}", entity)
    stage_path.GetRootLayer().Save()

    path_entity = visit_path or _default_orbit_path(ctx, center)
    ctx.camera_render_path.write_text(
        json.dumps(to_nerfstudio_camera_path(path_entity), indent=2), encoding="utf-8"
    )
    ctx.manifest.metadata["cameras"] = [entity.to_dict() for entity in entities]
    ctx.camera_plan_path.write_text(
        json.dumps({**bundle, "entities": ctx.manifest.metadata["cameras"]}, indent=2),
        encoding="utf-8",
    )
    preview_paths = _render_camera_previews(ctx, bundle["allShots"])
    stage.metadata = {
        "cameraCount": len(entities),
        "capturedCount": len(captured),
        "renderPath": path_entity.name,
    }
    stage.message = (
        f"USD stage composed with {len(entities)} cameras "
        f"({len(captured)} captured); render path: {path_entity.name}."
    )

    if ctx.manifest.mode == "guided" and len(captured) == 0 and not already_paused:
        stage.metadata["pausedForUserInput"] = True
        stage.message = (
            f"Generated {len(entities)} default cameras. Capture poses in "
            "the viewport to add more, then re-run this step. Re-run as-is "
            "to accept the defaults."
        )
        stage.state = StageState.NEEDS_USER_INPUT.value
        ctx.log(stage.message)
        return
    ctx._add_artifact(stage, "USD Stage", "usd", ctx.usd_stage_path, "Composed digital twin stage.")
    ctx._add_artifact(stage, "Camera Plan", "json", ctx.camera_plan_path, "Cameras and paths.")
    ctx._add_artifact(stage, "Render Camera Path", "json", ctx.camera_render_path, "ns-render camera path.")
    for index, preview in enumerate(preview_paths[:3], start=1):
        ctx._add_artifact(stage, f"Camera Preview {index}", "image", preview, "Generated camera preview.")


def _scene_center(ctx: "DigitalTwinStudioRunner") -> list[float]:
    """Robust centroid of the reconstruction (preview cloud percentiles)."""
    try:
        import numpy as np
        from plyfile import PlyData

        vertex = PlyData.read(str(ctx.recon_preview_ply_path))["vertex"]
        points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
        low, high = np.percentile(points, [5, 95], axis=0)
        return [float(v) for v in (low + high) / 2]
    except Exception:  # noqa: BLE001 - placeholder scenes have no preview
        return [0.0, 0.0, 0.0]


def _default_orbit_path(
    ctx: "DigitalTwinStudioRunner", center: list[float], seconds: float = 12.0
) -> "CameraEntity":
    from ..camera_paths import CameraEntity
    from ..walk_patterns import SceneBounds, bounds_from_preview_ply, orbit

    try:
        bounds = bounds_from_preview_ply(ctx.recon_preview_ply_path)
        bounds = SceneBounds(center=tuple(float(value) for value in center), radius=bounds.radius)
    except Exception:  # noqa: BLE001 - placeholder scenes have no preview cloud
        bounds = SceneBounds(center=tuple(float(value) for value in center), radius=2.0)
    return orbit(bounds, seconds=seconds, name="Scene Orbit")


def _render_camera_previews(ctx: "DigitalTwinStudioRunner", shots: list[dict]) -> list[Path]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return []
    paths: list[Path] = []
    palette = {
        "background": "#F5F5DC",
        "surface": "#FDFDFD",
        "accent": "#006994",
        "accent_alt": "#D4AF37",
        "text": "#222222",
    }
    for index, shot in enumerate(shots, start=1):
        image = Image.new("RGB", (1280, 720), palette["background"])
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((48, 48, 1232, 672), radius=28, fill=palette["surface"], outline=palette["accent"], width=4)
        draw.rectangle((88, 116, 1188, 520), fill=palette["accent"])
        draw.rectangle((124, 152, 1152, 484), fill=palette["accent_alt"])
        draw.text((88, 72), f"{index:02d}. {shot['name']}", fill=palette["text"])
        draw.text((88, 540), shot["description"], fill=palette["text"])
        draw.text((88, 600), f"Source: {shot['source']} | Position: {tuple(shot['position'])}", fill=palette["text"])
        path = ctx.cameras_dir / f"shot_{index:02d}.png"
        image.save(path)
        paths.append(path)
    return paths
