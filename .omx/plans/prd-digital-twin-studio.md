# PRD: Digital Twin Studio

## Summary

Build a native local-first Digital Twin Studio that converts a user video into a resumable digital-twin job with stage-by-stage visibility, USD creation, camera generation, optional Cosmos augmentation, final MP4 delivery, and optional live 3D viewing.

## User Stories

### US-001 Guided local run

As a user, I want to upload a video and move through a guided workflow so that I can create a digital twin locally without guessing what each tool is doing.

Acceptance criteria:

- guided step rail stays visible at all times
- one state card summarizes lifecycle state
- main viewer shows the active stage

### US-002 Resumable stages

As a user, I want explicit stage states and artifacts so that I can resume work, inspect failures, and later use the same workflow through an API.

Acceptance criteria:

- job manifest exists
- each stage has a state from the approved lifecycle set
- artifacts are registered per stage

### US-003 Camera generation

As a user, I want both presets and prompt-driven camera direction so that I can explore the space even if I do not know the right camera prompt.

Acceptance criteria:

- preset camera plans are generated
- prompt camera plans are generated
- stage viewer surfaces preview outputs

### US-004 Final delivery options

As a user, I want a final walkthrough MP4 and an optional live 3D viewer so that I can choose the safest or richest review mode for my machine.

Acceptance criteria:

- final MP4 is produced and registered
- UI offers open video and open live 3D viewer actions

### US-005 Future workflow portability

As a VaultWares developer, I want the demo shaped like a future API/workflow job so that it can move into `vaultwares-pipelines` and `vault-flows` with minimal rewrite.

Acceptance criteria:

- pipeline core is UI-agnostic
- job/stage contract is reusable
- install gaps surface as structured states rather than ad-hoc logs
