"""One-off functional probe for da3-draft (--gs-only direct 3DGS).

Runs the REAL production code path (DigitalTwinStudioRunner.run_stage) against
real footage, reusing the source video from the proven da3-standard run, on a
fresh job so it can't clobber that job's results. Only runs video_intake,
frame_extraction, and reconstruction — camera_staging/cosmos_output aren't
needed to answer "does infer_gs=True produce valid gaussians".

Usage:
    .venv\\Scripts\\python.exe tools\\probe_da3_draft.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vaultwares_studio.pipeline import create_job_manifest, DEFAULT_CAMERA_PROMPT, DigitalTwinStudioRunner  # noqa: E402
from vaultwares_studio.runners.hf_jobs import HfJobsStageRunner, HfJobsConfig  # noqa: E402

SOURCE_VIDEO = ROOT / "inputs" / "current_training" / "backyard_134s_sunny.mp4"


def main() -> int:
    if not SOURCE_VIDEO.exists():
        print(f"Source video not found: {SOURCE_VIDEO}", file=sys.stderr)
        return 1

    config = HfJobsConfig.load()
    if not config.enabled:
        print("Remote compute not enabled (data/remote_compute.json).", file=sys.stderr)
        return 1
    remote_runner = HfJobsStageRunner(config=config, confirm_cost=lambda est: print(f"[probe] cost pre-approved: {est.summary()}") or True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    manifest = create_job_manifest(SOURCE_VIDEO, DEFAULT_CAMERA_PROMPT)
    manifest.metadata["preset"] = "da3-draft"
    print(f"[probe] job_id={manifest.job_id} preset=da3-draft")

    runner = DigitalTwinStudioRunner(manifest, log, strict_mode=True, remote_runner=remote_runner)
    for stage_key in ("video_intake", "frame_extraction", "reconstruction"):
        print(f"[probe] running stage: {stage_key}")
        runner.run_stage(stage_key)

    splat_path = runner.recon_dir / "splat.ply"
    summary_path = runner.recon_dir / "summary.json"
    print(f"[probe] splat.ply exists: {splat_path.exists()}", "size:", splat_path.stat().st_size if splat_path.exists() else "n/a")
    print(f"[probe] summary.json: {summary_path.read_text(encoding='utf-8') if summary_path.exists() else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
