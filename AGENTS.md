# AGENTS.md - LoopGPT Agent Guide

This document provides the essential context an AI coding agent needs to work safely and productively in this repository.

## Start Here

- **Read**: [README.md](README.md) for product-level usage and quick setup.
- **Evaluation notes**: See [projects/conductor-eval/README.md](projects/conductor-eval/README.md). Root `evaluation/` files are compatibility wrappers only.
- **Don't run broad evaluation jobs** unless the user explicitly requests them — they consume many model calls.

## Quick Start (safe commands)

Install the packages editable and run the app locally (the root project is only compatibility wrappers and does not pull in the packages):

```bash
py -3.12 -m pip install -e "packages/conductor-core[providers,playback]"
py -3.12 -m pip install -e "apps/conductor-main"
python app.py
```

If you need to run tests while working on code changes, install the dev extras first:

```bash
py -3.12 -m pip install -e "packages/conductor-core[providers,playback,dev]"
py -3.12 -m pip install -e "projects/conductor-eval[dashboard,dev]"
py -3.12 -m pip install -e "apps/conductor-main[dev]"
py -3.12 -m pytest -q
```

Useful evaluation/debug commands:

```bash
py -3.12 -m conductor_eval.analysis
py -3.12 -m conductor_eval.analysis <path-to-run-directory>
py -3.12 -c "from conductor_eval.checks import scale_test, duration_test; from mido import MidiFile; midi = MidiFile('apps/conductor-main/generations/gen_<id>/loop.mid'); print(scale_test(midi, 'C', 'major')); print(duration_test(midi, 'quarter'))"
```

Avoid launching large automated evaluation runs or broad multi-model fan-outs without approval.

## Project Snapshot

LoopGPT generates 4-bar MIDI loops from natural-language prompts via a Gradio UI.

- **App entry**: [apps/conductor-main/src/conductor_main/app.py](apps/conductor-main/src/conductor_main/app.py) — Gradio UI and callback wiring. Root [app.py](app.py) is only a transitional launcher and no longer starts the app at import time.
- **Generation router**: [packages/conductor-core/src/conductor_core/routing.py](packages/conductor-core/src/conductor_core/routing.py) — Core routing across providers.
- **Provider modules**: [packages/conductor-core/src/conductor_core/providers/](packages/conductor-core/src/conductor_core/providers/) (`openai.py`, `anthropic.py`, `google.py`, `ollama.py`) — provider-specific prompt translation and API calls. Root `src/*_api.py` files are compatibility aliases.
- **Loop data & MIDI**: [packages/conductor-core/src/conductor_core/models.py](packages/conductor-core/src/conductor_core/models.py) (Pydantic loop models), [packages/conductor-core/src/conductor_core/midi.py](packages/conductor-core/src/conductor_core/midi.py) (loop→MIDI conversion).
- **Audio (optional)**: [packages/conductor-core/src/conductor_core/playback.py](packages/conductor-core/src/conductor_core/playback.py) — MIDI→audio rendering (requires FluidSynth/FFmpeg; the default SoundFont is packaged with Core, extra SoundFonts go in `apps/conductor-main/soundfonts/`).
- **History & UI data**: [packages/conductor-core/src/conductor_core/storage.py](packages/conductor-core/src/conductor_core/storage.py) and `apps/conductor-main/generations/` — recent outputs and metadata.
- **Visualization**: [apps/conductor-main/src/conductor_main/visualization.py](apps/conductor-main/src/conductor_main/visualization.py) owns the piano-roll rendering; shared musical constants and duration keywords live in `conductor_core.music`.
- **Core engine**: [packages/conductor-core/src/conductor_core/engine.py](packages/conductor-core/src/conductor_core/engine.py) owns provider-backed loop generation and MIDI persistence.
- **Evaluation**: [projects/conductor-eval/src/conductor_eval/evaluator.py](projects/conductor-eval/src/conductor_eval/evaluator.py), [projects/conductor-eval/src/conductor_eval/checks.py](projects/conductor-eval/src/conductor_eval/checks.py), and [projects/conductor-eval/src/conductor_eval/analysis.py](projects/conductor-eval/src/conductor_eval/analysis.py).
- **Legacy evaluation paths**: Root [evaluation/](evaluation/) modules only delegate to `conductor_eval`.
- **Tests**: Each project owns its suite (`packages/conductor-core/tests`, `apps/conductor-main/tests`, `projects/conductor-eval/tests`); root [tests/](tests/) holds compatibility regression tests. Root `pytest -q` runs all of them.

## Evaluation Conventions

- **Evaluator**: [projects/conductor-eval/src/conductor_eval/evaluator.py](projects/conductor-eval/src/conductor_eval/evaluator.py) orchestrates evaluation tasks and delegates generation to `LoopGenerationEngine.generate(...)`.
- **Tests selection**: Tests are selected by name and executed via `Evaluator.run_tests(...)`.
- **Auto-detection**: Duration expectations use `DURATION_KEYWORDS` from `conductor_core.music`.
- **Default outputs**: The `Evaluator` defaults to `projects/conductor-eval/evaluations`.
- **Generation boundary**: Eval must not route providers or convert loop objects to MIDI itself; those are Core responsibilities.

## Project Conventions

- **Imports**: Follow local ordering: third-party, stdlib, then local modules.
- **Logging**: Use Python's `logging` library; evaluation code writes logs to `<output_dir>/run.log`.
- **Errors**: Raise `ValueError` for invalid user-facing configuration; prefer graceful fallbacks for optional dependencies.
- **Provider resilience**: Provider modules (especially Ollama) must not fail at import time if services are unavailable.

## Data and Outputs

- **Model list**: [packages/conductor-core/src/conductor_core/resources/model_list.json](packages/conductor-core/src/conductor_core/resources/model_list.json) is the source of truth for provider metadata.
- **Generations**: `apps/conductor-main/generations/` contains recent UI outputs with `loop.mid`, optional `loop.mp3`, `messages.json`, and `metadata.json` (root `generations/` only holds legacy artifacts).

Evaluation runs produce a structured directory with `config.json`, `summary.json`, generated MIDI files, message logs, and per-run test results.

## Known Pitfalls (pay attention to these)

- **Ollama availability**: [packages/conductor-core/src/conductor_core/providers/ollama.py](packages/conductor-core/src/conductor_core/providers/ollama.py) should use lazy checks; don't assume a running local Ollama service.
- **Audio dependencies**: Audio rendering requires external FluidSynth/FFmpeg tools; treat it as optional. The default SoundFont ships inside `conductor_core.resources.soundfonts`.
- **Generated artifacts**: The repo contains generated MIDI files and prior runs under `generations/` and `apps/conductor-main/generations/`; do not treat them as source-of-truth.

## Common Change Paths (how to make common edits)

- **Add or change a model/provider**:
	1. Update [packages/conductor-core/src/conductor_core/resources/model_list.json](packages/conductor-core/src/conductor_core/resources/model_list.json).
	2. Implement or update the provider module under `packages/conductor-core/src/conductor_core/providers/`.
	3. Adjust [packages/conductor-core/src/conductor_core/routing.py](packages/conductor-core/src/conductor_core/routing.py) if the provider contract changed.
	4. Update UI controls in [apps/conductor-main/src/conductor_main/app.py](apps/conductor-main/src/conductor_main/app.py) only if the change affects user-facing options.

- **Change loop structure or MIDI semantics**:
	1. Update [packages/conductor-core/src/conductor_core/models.py](packages/conductor-core/src/conductor_core/models.py) (Pydantic models).
	2. Update [packages/conductor-core/src/conductor_core/midi.py](packages/conductor-core/src/conductor_core/midi.py) for conversion logic.
	3. Update any provider parsing logic that constructs `Loop` objects.
	4. Re-run evaluation tests and re-check the packaged prompt in [packages/conductor-core/src/conductor_core/resources/prompts/](packages/conductor-core/src/conductor_core/resources/prompts/).

- **Change evaluation behavior**:
	1. Update [projects/conductor-eval/src/conductor_eval/evaluator.py](projects/conductor-eval/src/conductor_eval/evaluator.py).
	2. Add or update checks in [projects/conductor-eval/src/conductor_eval/checks.py](projects/conductor-eval/src/conductor_eval/checks.py).
	3. Add focused tests under [projects/conductor-eval/tests](projects/conductor-eval/tests).
	4. Update [projects/conductor-eval/README.md](projects/conductor-eval/README.md) and ensure dashboard expectations match.

## How agents should work (brief guidelines)

- **Discover before editing**: Read [README.md](README.md) and this file before making changes.
- **Run a minimal test**: After edits, add or update focused tests in the owning project's `tests/` directory when the change affects behavior, then run the most relevant unit test or a focused `python -c` check for the touched slice.
- **Avoid heavy operations**: Ask for permission before running resource-heavy evaluation runs or multi-provider calls.
