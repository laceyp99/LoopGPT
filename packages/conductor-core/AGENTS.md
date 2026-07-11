# Conductor Core Agent Guide

## Scope

This repository owns the reusable generation engine: public request/result
contracts, provider routing and adapters, model metadata, music models, MIDI
conversion, artifact storage, and optional playback helpers. Do not introduce
Gradio, Dash, or evaluation dependencies into Core.

## Key paths

- `src/conductor_core/engine.py`: end-to-end synchronous generation.
- `src/conductor_core/config.py`: public configuration and result contracts.
- `src/conductor_core/resources/model_list.json`: canonical provider metadata.
- `src/conductor_core/providers/`: provider request and parsing behavior.
- `src/conductor_core/midi.py`: loop/MIDI conversion.
- `src/conductor_core/storage.py`: filesystem artifacts and history.
- `tests/`: deterministic unit and package-boundary tests.
- `.agents/skills/add-model-support/`: model-support workflow.

## Working rules

- Inspect existing provider and test patterns before editing.
- Keep provider services lazy and import-safe; never require a live service at import time.
- Preserve the public API exported from `conductor_core.__init__` unless a deliberate breaking change is requested.
- Prefer metadata-driven model capabilities over hard-coded model exceptions.
- Do not make live provider calls or run broad evaluations unless explicitly requested.
- Treat FluidSynth and FFmpeg as optional external tools.
- Do not commit generated MIDI, audio, histories, build output, or credentials.

## Validation

```powershell
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python -m build
```

Run focused provider, engine, MIDI, or storage tests while iterating. Before a
commit, check `git status` and the intended diff. Do not commit planning
artifacts or unrelated changes.
