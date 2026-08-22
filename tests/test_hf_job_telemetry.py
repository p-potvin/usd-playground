"""The HF Jobs runner records one model run per job.

Exercised through a stubbed _run so nothing here contacts Hugging Face or
needs a token: the point is the wrapper's exit paths, not the job itself.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from vaultwares_studio import telemetry
from vaultwares_studio.runners.base import (
    CostDeniedError,
    StageCancelledError,
    StageContext,
    StageResult,
)
from vaultwares_studio.runners.hf_jobs import HfJobsConfig, HfJobsStageRunner

pytestmark = pytest.mark.skipif(
    not telemetry.available(), reason="vaultwares-adk submodule not initialised"
)

COST_METADATA = {
    "job_id": "hfj-abc",
    "job_url": "https://hf.co/jobs/hfj-abc",
    "flavor": "l4x1",
    "est_usd": 0.30,
    "actual_usd_estimate": 0.27,
    "duration_seconds": 1215.4,
    "rate_source": "approximate public rates, recorded 2026-06",
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path):
    from vaultwares_adk.telemetry import configure

    configure(
        spool_dir=str(tmp_path / "spool"),
        api_url="http://127.0.0.1:9",
        post_timeout_s=0.15,
        enabled=True,
    )


def _ctx() -> StageContext:
    return StageContext(
        job_dir=Path(tempfile.mkdtemp()),
        job_id="studio-job-1",
        stage_key="reconstruction",
    )


def _runner(behaviour):
    runner = HfJobsStageRunner(config=HfJobsConfig())
    runner._run = behaviour  # type: ignore[method-assign]
    return runner


def _last_record():
    from vaultwares_adk.telemetry.worker import get_worker

    worker = get_worker()
    worker._drain_once(__import__("vaultwares_adk.telemetry", fromlist=["get_config"]).get_config())
    assert worker._pending, "no run was recorded"
    return worker._pending[-1]


def test_successful_job_records_cost_and_flavor():
    runner = _runner(lambda ctx: StageResult(
        status="complete", artifacts=[Path("a.usda")], metadata=COST_METADATA))

    result = runner.run(_ctx())

    assert result.status == "complete"
    record = _last_record()
    assert record.provider == "huggingface"
    assert record.runtime == "hf-job"
    assert record.cost_usd == 0.27
    assert record.extra["flavor"] == "l4x1"


def test_cost_is_flagged_inexact():
    # FLAVOR_RATES_USD_PER_HOUR is documented as approximate and the real
    # figure is the HF invoice, which is never read. A computed estimate must
    # not read as billed spend.
    runner = _runner(lambda ctx: StageResult(status="complete", metadata=COST_METADATA))
    runner.run(_ctx())

    record = _last_record()
    assert record.priced_exactly is False
    assert record.cost_state == "settled"
    assert record.is_free is False


def test_rented_gpu_is_not_marked_free():
    runner = _runner(lambda ctx: StageResult(status="complete", metadata=COST_METADATA))
    runner.run(_ctx())
    assert _last_record().is_free is False


def test_user_cancel_is_cancelled_not_error():
    # Aborting a twenty-minute reconstruction is a decision. Counting it in the
    # failure rate would make the runner look unreliable.
    def _cancel(ctx):
        raise StageCancelledError("Remote job cancelled.")

    with pytest.raises(StageCancelledError):
        _runner(_cancel).run(_ctx())

    assert _last_record().status == "cancelled"


def test_declined_cost_is_rejected_not_error():
    def _denied(ctx):
        raise CostDeniedError("user declined $0.30")

    with pytest.raises(CostDeniedError):
        _runner(_denied).run(_ctx())

    record = _last_record()
    assert record.status == "rejected"
    assert record.error_class == "CostDenied"


def test_remote_failure_is_recorded_then_reraised():
    def _boom(ctx):
        raise RuntimeError("Remote job ended in ERROR.")

    with pytest.raises(RuntimeError):
        _runner(_boom).run(_ctx())

    record = _last_record()
    assert record.status == "error"
    assert record.error_class == "RuntimeError"


def test_remote_seconds_recorded_separately_from_wall_clock():
    # The runner times the remote job; the recorder's own clock would also
    # include upload and download.
    runner = _runner(lambda ctx: StageResult(status="complete", metadata=COST_METADATA))
    runner.run(_ctx())
    assert _last_record().extra["remote_seconds"] == 1215.4


def test_missing_metadata_does_not_raise():
    runner = _runner(lambda ctx: StageResult(status="complete", metadata={}))
    assert runner.run(_ctx()).status == "complete"
