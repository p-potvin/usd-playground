"""Static checks on the worker entrypoints.

These files run only inside the job container, so nothing else in the suite
imports them and ordinary mistakes reach production unexamined. On 2026-08-03 a
second `def run_streaming` was added alongside the existing one; Python kept the
later definition, and every `--train-only` call site started raising
`TypeError: run_streaming() missing 1 required positional argument`. The job
failed 15 seconds in, after paying to schedule a GPU.

Deliberately narrow. Broader variants (every called name resolves, every CLI
flag is consumed) were tried and produced false positives on pre-existing code —
a check that cries wolf gets deleted, so this one only asserts what it can prove.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "docker" / "worker"
ENTRYPOINTS = sorted(WORKER_DIR.glob("*_entrypoint.py"))


def test_entrypoints_are_discoverable():
    assert ENTRYPOINTS, f"no *_entrypoint.py found under {WORKER_DIR}"


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: p.name)
def test_entrypoint_parses(path: Path):
    """Nothing else imports these, so a syntax error would ship."""
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: p.name)
def test_no_duplicate_top_level_definitions(path: Path):
    """A redefinition silently shadows the first, often with a new signature."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen:
                duplicates.append(
                    f"{node.name} (lines {seen[node.name]} and {node.lineno})"
                )
            seen[node.name] = node.lineno
    assert not duplicates, f"{path.name} redefines: {'; '.join(duplicates)}"
