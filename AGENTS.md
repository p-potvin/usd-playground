# usd-playground Agent Rules

These rules focus on Figma-to-code work, native desktop UI consistency, and VaultWares branding for this repo.

## Project Structure

- The main native UI shell lives in `gui_app.py`.
- Reusable execution logic lives under `studio_core/`.
- Tests for repo-owned logic live under `tests/`.
- Runtime outputs belong under `data/jobs/` and must not be treated as source files.

## Framework and Styling Rules

- This repo is a Python desktop app, not a web app.
- Prefer `PySide6` and existing `qfluentwidgets` primitives before introducing new UI frameworks.
- Treat QFluentWidgets output as the structural baseline, not the final visual language.
- Use soft VaultWares styling aligned to `vault-themes/.github/STYLE.md`:
  - no ad-hoc hardcoded neon palettes
  - calm light/dark-capable surfaces
  - visible hierarchy
  - 8px-style spacing rhythm where practical

## Token and Visual Direction

- IMPORTANT: Never scatter unrelated one-off colors through the UI.
- Prefer a small, repeatable palette for cards, borders, backgrounds, accent, and text.
- Favor warm-neutral surfaces with controlled blue/gold accents over generic terminal aesthetics.
- Keep motion subtle and avoid decorative busywork.

## Desktop Component Rules

- Reuse shared pipeline data from `studio_core/` instead of embedding business logic in UI widgets.
- UI widgets should render manifest data, trigger actions, and display artifacts; they should not own pipeline rules.
- New pipeline features should land in `studio_core/` first, then be surfaced in the UI.

## Figma MCP Integration Rules

These rules define how to translate Figma designs into code for this repo.

### Required Flow

1. Run `get_design_context` for the exact node(s) first.
2. Run `get_screenshot` for visual parity reference.
3. Use the Figma output as a structure reference only; adapt it to `PySide6` / `qfluentwidgets`.
4. Map colors and spacing to the repo's VaultWares-aligned styling rather than copying raw values directly.
5. Reuse existing cards, layout sections, and runner-driven UI patterns before creating new ones.

### Implementation Rules

- Do not translate Figma output into web stacks such as React or Tailwind for this repo.
- Convert layouts into Qt widgets and Qt layouts directly.
- Keep the guided studio pattern intact:
  - persistent left step rail
  - top-row state card
  - main active-step viewer
  - finish panel swap on completion
- Preserve the job/stage/artifact contract when implementing new screens.

## Asset Handling

- If Figma MCP provides localhost image or SVG assets, use them directly when relevant.
- Store repo-owned static assets in the project root or a clear local asset folder near the consuming feature.
- Do not introduce new icon packs unless explicitly required.

## Testing Rules

- Add targeted tests for new logic in `studio_core/`.
- Prefer deterministic tests around manifests, stage state, camera planning, and artifact generation.
- Keep UI verification lightweight unless a change truly requires Qt-specific behavior testing.

<!-- VAULT-THEMES-SUBMODULE:START -->
## Vault Themes Submodule Rules

If this repository includes the `vault-themes` submodule, you MUST read the following files before making changes related to UI, branding, design systems, token usage, shared components, authentication UX, encrypted client-to-client communication, or Figma-derived implementation:

- `vault-themes/AGENTS.md`
- `vault-themes/.github/STYLE.md`
- `vault-themes/.github/INSTRUCTIONS.md`

When theme token roles, contrast helpers, or executable theme governance matter, also inspect:

- `vault-themes/theme_manager.py`

Treat these files as the shared VaultWares source of truth. Re-check them whenever the submodule changes or when a task touches cross-repo product rules.
<!-- VAULT-THEMES-SUBMODULE:END -->
