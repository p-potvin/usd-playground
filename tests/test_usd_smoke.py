from pathlib import Path

from pxr import Usd

from usd_smoke import DEFAULT_OUTPUT_PATH, build_smoke_stage


def test_smoke_stage_writes_openable_usd():
    result = build_smoke_stage()

    assert result.output_path == Path(DEFAULT_OUTPUT_PATH).resolve()
    assert result.output_path.exists()
    assert result.output_path.suffix == ".usda"

    stage = Usd.Stage.Open(str(result.output_path))
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath().pathString == "/World"

    reconstruction = stage.GetPrimAtPath("/World/DigitalTwin/Reconstruction")
    assert reconstruction.IsValid()
    assert len(reconstruction.GetAttribute("points").Get()) == result.reconstruction_points

    source_video = stage.GetPrimAtPath("/World/DigitalTwin").GetAttribute("sourceVideo").Get()
    assert source_video.endswith("test_input.mp4")

    print(f"Generated USD artifact: {result.output_path}")
