"""Model-run telemetry bridge for vaultwares-studio.

Records one run per Hugging Face Job, so rented GPU reconstruction sits in the
same series as local ASR, ComfyUI and the HF inference gateway rather than
being tracked only in the per-job manifest.

Same two rules as the other bridges: resolved lazily and cached, and never
allowed to break the caller. A reconstruction that took twenty minutes on an
L4 must not fail at the last step because a telemetry post did.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

_adk: Any = None
_state = "unloaded"  # unloaded | ready | unavailable


def _load() -> bool:
    global _adk, _state
    if _state != "unloaded":
        return _state == "ready"

    here = Path(__file__).resolve()
    candidates = [os.environ.get("VW_ADK_PATH")]
    for parent in list(here.parents)[:4]:
        candidates.append(str(parent / "vaultwares-adk"))
    for candidate in candidates:
        if candidate and Path(candidate).is_dir() and candidate not in sys.path:
            sys.path.insert(0, candidate)

    try:
        from vaultwares_adk import telemetry  # type: ignore

        _adk = telemetry
        _state = "ready"
    except Exception:
        _adk = None
        _state = "unavailable"
    return _state == "ready"


def available() -> bool:
    return _load()


def job_run(
    *,
    model: str,
    flavor: Optional[str] = None,
    project: Optional[str] = None,
    **fields: Any,
):
    """Context manager for one Hugging Face Job.

    Cost is recorded but flagged inexact. FLAVOR_RATES_USD_PER_HOUR is
    explicitly "approximate public rates" and the authoritative figure is the
    HF invoice, which we never read -- so `priced_exactly=False` keeps a
    computed estimate from being mistaken for billed spend. It is still
    `settled`, because nothing will ever come back to correct it.
    """
    if not _load():
        return _NullRun()
    try:
        return _adk.ModelRun(
            provider="huggingface",
            runtime="hf-job",
            model=model,
            task="job",
            project=project or os.environ.get("VW_PROJECT") or "vaultwares-studio",
            service="studio-reconstruction",
            priced_exactly=False,
            cost_state="settled",
            is_free=False,
            flavor=flavor,
            **fields,
        )
    except Exception:
        return _NullRun()


def flush(timeout: float = 15.0) -> None:
    if not _load():
        return
    try:
        _adk.shutdown(timeout=timeout)
    except Exception:
        pass


class _NullRun:
    record = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return self

        return _noop
