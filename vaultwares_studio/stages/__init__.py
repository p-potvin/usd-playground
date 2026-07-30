"""Stage handler modules for the digital twin pipeline.

Each module implements the body of a pipeline stage, extracted from
``DigitalTwinStudioRunner`` to keep the orchestrator thin.  Stage functions
receive the runner as ``ctx`` and a ``StageRecord``; they call back into the
runner for shared helpers (``_run_command``, ``_add_artifact``, etc.).

Modules are imported lazily by ``pipeline.py`` to avoid circular imports.
"""
