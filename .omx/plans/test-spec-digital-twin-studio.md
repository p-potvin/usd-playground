# Test Spec: Digital Twin Studio

## Verification Targets

### Pipeline core

- stage definitions are deterministic
- job manifests serialize and reload correctly
- state transitions preserve valid lifecycle values
- artifact registration writes expected paths and metadata

### Camera logic

- preset generation returns expected named views
- prompt-driven camera planning returns bounded steps and preview metadata

### Walkthrough output

- final output registration includes MP4 artifact metadata
- missing preview inputs degrade gracefully

### Dependency health

- missing binaries/modules surface as `needs-install`
- available tools resolve correctly into health summaries

### Existing smoke behavior

- `usd_smoke.py` and `tests/test_usd_smoke.py` continue to pass

## Planned Commands

- `python -m pytest -q tests`
- targeted import / syntax checks on touched modules
- if the UI remains import-safe, a direct module import smoke check for `gui_app.py`
