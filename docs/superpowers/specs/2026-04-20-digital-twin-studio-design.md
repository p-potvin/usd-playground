# Digital Twin Studio Design

## Goal

Deliver a native-first Digital Twin Studio for local Windows hardware that turns a user video into a resumable digital-twin job with explicit stages, artifact outputs, a final walkthrough video, and an optional live 3D viewer.

## Product Shape

- Native desktop app first, built on the existing `PySide6` / `qfluentwidgets` shell.
- Guided-first workflow for local use.
- Future-ready pipeline contract so the same job model can move into `vaultwares-pipelines` and later become a `vault-flows` workflow.
- Local-first execution profile tuned for `RTX 3060 12GB VRAM + 32GB RAM`.

## Core UX

### Main layout

- Persistent left rail with:
  - current job summary
  - one lifecycle state card
  - ordered stage list
- Main content column with:
  - top-row state card spanning the viewer width
  - active step viewer with explanation, previews, artifacts, and controls
  - finish panel replacing the active step viewer when the job completes
- Clicking any earlier stage reopens that stage viewer after completion.

### Workflow stages

1. Video Intake
2. Frame Extraction
3. Reconstruction
4. USD + Cameras
5. Cosmos + Output

### Lifecycle states

- `queued`
- `running`
- `needs-install`
- `needs-user-input`
- `complete`
- `failed`

## Architecture

### Native Studio Shell

The desktop shell owns file picking, install health, stage navigation, preview surfaces, logs, and handoff actions.

### Pipeline Core

A reusable local execution layer owns:

- job manifests
- stage metadata and state transitions
- dependency detection
- execution adapters for each tool-driven step
- artifact registration
- camera generation
- walkthrough video generation

This layer must stay UI-agnostic.

### Future Workflow Contract

The same job/stage/artifact schema must be portable to:

- `vaultwares-pipelines` as a backend job API
- `vault-flows` as a workflow node / workflow run surface

## Stage Responsibilities

### Video Intake

- accept local video input
- inspect duration, fps, resolution
- choose a local-safe execution profile
- initialize the job manifest

### Frame Extraction

- extract frames with ffmpeg
- keep previews and manifest metadata
- allow retries with different sampling if needed

### Reconstruction

- run COLMAP / Nerfstudio / gsplat paths when available
- fall back to placeholder-safe outputs when tools are missing unless strict mode is enabled
- register dense artifacts, checkpoints, point cloud / PLY outputs, and logs

### USD + Cameras

- convert reconstruction output into USD stage artifacts
- compose lights, floor, and metadata
- provide both:
  - preset cameras
  - prompt-driven camera direction
- emit preview frames for stage review

### Cosmos + Output

- optional Cosmos Reason / Transfer execution
- final walkthrough render to MP4
- finish state with:
  - open walkthrough video
  - open live 3D viewer
  - open output folder / artifacts

## Camera Director

### Preset mode

Generate safe, explainable camera presets such as:

- entrance view
- orbit
- overhead
- hero shot

### Prompt-driven mode

Accept natural-language camera requests like:

- `show me the desk from the doorway, then orbit left and rise`

The prompt-driven path should resolve to a bounded camera plan with generated preview frames and path metadata. Runtime length is acceptable; memory spikes are not.

## Live 3D Viewer

- The app must offer a live 3D viewing path.
- For v1, this can be a native viewer launch surface rather than a fully embedded complex viewport.
- It should prefer a local-safe point cloud / stage preview path and avoid browser WebGL dependence.

## Outputs

### Required

- extracted frames
- reconstruction artifacts
- USD stage
- camera metadata / previews
- final walkthrough MP4
- job manifest

### Optional

- live 3D viewer session
- Cosmos annotation output
- Cosmos transfer output

## Local Hardware Rules

- Default to local-safe settings.
- Prefer lower-memory paths over maximum fidelity.
- Gate heavier tools behind explicit `needs-install` or opt-in controls.
- Avoid OOM conditions even if execution time increases.

## Branding and Design Direction

- Use `vault-themes/.github/STYLE.md` as the main visual direction source.
- Keep the look calm, legible, and structured rather than flashy.
- Use VaultWares token thinking:
  - no hardcoded ad-hoc colors
  - 8px spacing rhythm
  - explicit visual hierarchy
- The UI should feel product-grade, not like a debug console.

## Testing Strategy

- Add targeted tests for:
  - job manifest creation and updates
  - stage state transitions
  - dependency health reporting
  - camera plan generation
  - walkthrough output registration
- Keep the existing USD smoke test intact.
- Prefer deterministic unit coverage for the new pipeline core instead of UI-only testing.

## Risks

- Existing implementation is placeholder-heavy, so the refactor must preserve runnable demo behavior while improving structure.
- Embedded 3D viewing is likely the highest technical risk on the local machine.
- Some heavy dependencies remain optional and must surface as explicit install gaps, not silent failures.

## First Implementation Slice

1. Extract pipeline/job concepts out of `gui_app.py` into a reusable core.
2. Redesign the dashboard into the approved guided layout.
3. Introduce explicit job/stage/artifact manifests and statuses.
4. Add final walkthrough video output.
5. Add optional native live viewer launch.
6. Add tests for the new core.
