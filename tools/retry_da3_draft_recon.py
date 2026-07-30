"""Retry the reconstruction stage on an existing job (frames already extracted).

Usage:
    .venv\\Scripts\\python.exe tools\\retry_da3_draft_recon.py <job-id> [scheduling-timeout-seconds]

The optional second argument is how long each flavor candidate may sit in HF's
SCHEDULING state before the runner gives up and tries the next one. Defaults to
600s here rather than the runner's interactive 120s: this tool is for unattended
retries, where waiting out a busy GPU pool beats exhausting the candidate list
in four minutes. SCHEDULING is not billed, so a long wait is free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vaultwares_studio.pipeline import load_job_manifest, DigitalTwinStudioRunner  # noqa: E402
from vaultwares_studio.runners.hf_jobs import HfJobsStageRunner, HfJobsConfig  # noqa: E402


def main() -> int:
    job_id = sys.argv[1]
    manifest_candidates = [
        ROOT / "data" / "jobs" / job_id / "manifest.json",
        Path("D:/vaultwares-studio-jobs/data/jobs") / job_id / "manifest.json",
    ]
    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    if manifest_path is None:
        print(f"No manifest.json found for {job_id}", file=sys.stderr)
        return 1

    config = HfJobsConfig.load()
    if not config.enabled:
        print("Remote compute not enabled.", file=sys.stderr)
        return 1
    remote_runner = HfJobsStageRunner(config=config, confirm_cost=lambda est: print(f"[retry] cost pre-approved: {est.summary()}") or True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    manifest = load_job_manifest(manifest_path)
    scheduling_timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    manifest.metadata["flavor_scheduling_timeout_seconds"] = scheduling_timeout
    print(
        f"[retry] job_id={manifest.job_id} preset={manifest.metadata.get('preset')} "
        f"scheduling_timeout={scheduling_timeout:.0f}s per flavor"
    )

    runner = DigitalTwinStudioRunner(manifest, log, strict_mode=True, remote_runner=remote_runner)
    runner.run_stage("reconstruction")

    splat_path = runner.recon_dir / "splat.ply"
    summary_path = runner.recon_dir / "summary.json"
    print(f"[retry] splat.ply exists: {splat_path.exists()}", "size:", splat_path.stat().st_size if splat_path.exists() else "n/a")
    print(f"[retry] summary.json: {summary_path.read_text(encoding='utf-8') if summary_path.exists() else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
