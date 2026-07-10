# Conductor Main Agent Guide

## Scope

This repository owns the replaceable Gradio client: layout, callbacks, UI
state, prompt editing, history presentation, and Plotly visualization.
Generation orchestration, provider routing, MIDI conversion, persistence, and
playback helpers belong to `conductor-core`.

## Key paths

- `src/conductor_main/app.py`: Gradio layout and callback adaptation.
- `src/conductor_main/visualization.py`: UI-specific piano-roll rendering.
- `tests/test_app.py`: callback and UI behavior.
- `tests/test_package_boundary.py`: client isolation checks.

## Working rules

- Keep callbacks thin and delegate generation to `LoopGenerationEngine`.
- Do not duplicate Core model metadata, provider routing, MIDI, or storage logic.
- Keep provider/model controls metadata-driven.
- Preserve optional audio behavior when FluidSynth or FFmpeg is unavailable.
- Importing the package must not launch Gradio.
- Do not make live provider calls during ordinary tests.
- Do not commit API keys, prompt experiments, generations, or build output.

## Validation

Install a compatible `conductor-core`, then run:

```powershell
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python -m build
```

Manually smoke-test provider/model selectors and generation when callback or
layout behavior changes. Before a commit, inspect `git status` and the intended
diff and leave unrelated or planning artifacts untouched.
