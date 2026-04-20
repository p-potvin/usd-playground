# Testing usd-playground

This repo has two very different layers:

1. A small, practical OpenUSD smoke test you can run locally.
2. A larger research/demo pipeline that depends on Redis, ffmpeg, Nerfstudio, COLMAP, and extra agent code.

For everyday users, start with the smoke test.

## Recommended Test

Run this from the repo root.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install usd-core pytest redis
python -m pytest -s
```

### What success looks like

- `pytest` exits with code `0`
- the test summary reports all tests passed
- `data/test_outputs/smoke_scene.usda` exists
- the console prints the generated USD path

## Generated Artifact

The smoke test writes:

- `data/test_outputs/smoke_scene.usda`

You can generate the same file without pytest:

```powershell
.\.venv\Scripts\python.exe .\usd_smoke.py
```

## Why The Default Test Scope Is Limited

`pytest.ini` intentionally limits discovery to:

- `tests/`
- `vaultwares_agentciation/omx_integration/tests/`

It skips:

- `.venv/` because that contains third-party package tests
- `cosmos-reason2/` because it is an upstream submodule with additional platform-specific requirements
- `vault-themes/` because it is not part of the Python test surface for this repo

Without that scoping, a normal `pytest` run tries to execute unrelated vendored tests and fails for reasons that have nothing to do with `usd-playground`.

## About The Bigger Pipeline

These files are still present:

- `run_pipeline_demo.py`
- `worker_runner.py`
- `manager_runner.py`
- `gui_app.py`

They are better treated as advanced demos than as the default test story. They may require:

- a running Redis instance
- ffmpeg
- Nerfstudio
- COLMAP
- extra GPU-heavy dependencies

If you only want to verify that the repo can author OpenUSD output locally, the smoke test is the right tool.

## Troubleshooting

If `from pxr import Usd` fails:

- install `usd-core` into the active environment
- confirm you are using the same Python interpreter for both install and test commands

If `pytest` tries to run submodule or `.venv` tests anyway:

- make sure you are running from the repo root
- confirm `pytest.ini` is present

If the USD file is missing after a green test run:

- check whether the repo root is writable
- rerun with `python -m pytest -s tests/test_usd_smoke.py`
