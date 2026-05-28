# AGENTS.md - LoopGPT Agent Guide

This document provides the essential context an AI coding agent needs to work safely and productively in this repository.

## Start Here

- **Read**: [README.md](README.md) for product-level usage and quick setup.
- **Evaluation notes**: See [evaluation/README.md](evaluation/README.md) before changing anything under `evaluation/`.
- **Don't run broad evaluation jobs** unless the user explicitly requests them — they consume many model calls.

## Quick Start (safe commands)

Install dependencies and run the app locally:

```bash
pip install -e .
python app.py
```

If you need to run tests while working on code changes, install the dev extra first:

```bash
pip install -e ".[dev]"
```

Useful evaluation/debug commands:

```bash
python evaluation/analysis.py
python evaluation/analysis.py <path-to-run-directory>
python -c "from evaluation.tests import scale_test, duration_test; from mido import MidiFile; midi = MidiFile('output.mid'); print(scale_test(midi, 'C', 'major')); print(duration_test(midi, 'quarter'))"
```

Avoid launching large automated evaluation runs or broad multi-model fan-outs without approval.

## Project Snapshot

LoopGPT generates 4-bar MIDI loops from natural-language prompts via a Gradio UI.

- **App entry**: [app.py](app.py) — Gradio UI and callback wiring (starts the app at import time).
- **Generation router**: [src/runs.py](src/runs.py) — core routing and orchestration across providers.
- **Provider modules**: [src/openai_api.py](src/openai_api.py), [src/claude_api.py](src/claude_api.py), [src/gemini_api.py](src/gemini_api.py), [src/ollama_api.py](src/ollama_api.py) — provider-specific prompt translation and API calls.
- **Loop data & MIDI**: [src/objects.py](src/objects.py) (Pydantic loop models), [src/midi_processing.py](src/midi_processing.py) (loop→MIDI conversion).
- **Audio (optional)**: [src/audio.py](src/audio.py) — MIDI→audio rendering (requires FluidSynth/FFmpeg and a soundfont in [soundfonts/](soundfonts/)).
- **History & UI data**: [src/history.py](src/history.py) and `generations/` — recent outputs and metadata.
- **Utilities**: [src/utils.py](src/utils.py) — shared helpers (duration keywords, visualization, logging helpers).
- **Evaluation**: [evaluation/evaluator.py](evaluation/evaluator.py), [evaluation/tests.py](evaluation/tests.py), [evaluation/analysis.py](evaluation/analysis.py).
- **Tests**: See the test suite under [tests/](tests/).

## Evaluation Conventions

- **Evaluator**: [evaluation/evaluator.py](evaluation/evaluator.py) orchestrates generation, tests, result saving, and provider routing.
- **Tests selection**: Tests are selected by name and executed via `Evaluator.run_tests(...)`.
- **Auto-detection**: Duration expectations are auto-detected using keywords from [src/utils.py](src/utils.py).
- **Default outputs**: The `Evaluator` defaults `output_dir` to `evaluations`. Documentation examples may use `runs` — check callers.

## Project Conventions

- **Imports**: Follow local ordering: third-party, stdlib, then local modules.
- **Logging**: Use Python's `logging` library; evaluation code writes logs to `<output_dir>/run.log`.
- **Errors**: Raise `ValueError` for invalid user-facing configuration; prefer graceful fallbacks for optional dependencies.
- **Provider resilience**: Provider modules (especially Ollama) must not fail at import time if services are unavailable.

## Data and Outputs

- **Model list**: [model_list.json](model_list.json) is the source of truth for provider metadata.
- **Latest session**: [loop.json](loop.json) stores the latest loop-generation message history.
- **Generations**: `generations/` contains recent UI outputs with `loop.mid` and `metadata.json`.

Evaluation runs produce a structured directory with `config.json`, `summary.json`, generated MIDI files, message logs, and per-run test results.

## Known Pitfalls (pay attention to these)

- **Import-time side effects**: [app.py](app.py) launches the Gradio app on import — avoid importing it during non-UI tasks.
- **Ollama availability**: [src/ollama_api.py](src/ollama_api.py) should use lazy checks; don't assume a running local Ollama service.
- **Audio dependencies**: Audio rendering requires external tools and a soundfont in [soundfonts/](soundfonts/); treat it as optional.
- **Generated artifacts**: The repo contains generated MIDI files and prior runs under `generations/`; do not treat them as source-of-truth.

## Common Change Paths (how to make common edits)

- **Add or change a model/provider**:
	1. Update [model_list.json](model_list.json).
	2. Implement or update the provider module under `src/`.
	3. Adjust routing in [src/runs.py](src/runs.py) if the provider contract changed.
	4. Update UI controls in [app.py](app.py) only if the change affects user-facing options.

- **Change loop structure or MIDI semantics**:
	1. Update [src/objects.py](src/objects.py) (Pydantic models).
	2. Update [src/midi_processing.py](src/midi_processing.py) for conversion logic.
	3. Update any provider parsing logic that constructs `Loop` objects.
	4. Re-run evaluation tests and re-check prompt files in [Prompts/](Prompts/).

- **Change evaluation behavior**:
	1. Update [evaluation/evaluator.py](evaluation/evaluator.py).
	2. Add or update checks in [evaluation/tests.py](evaluation/tests.py).
	3. Update [evaluation/README.md](evaluation/README.md) and ensure [evaluation/analysis.py](evaluation/analysis.py) expectations match.

## How agents should work (brief guidelines)

- **Discover before editing**: Read [README.md](README.md) and this file before making changes.
- **Run a minimal test**: After edits, add or update focused tests under [tests/](tests/) when the change affects behavior, then run the most relevant unit test or a focused `python -c` check for the touched slice.
- **Avoid heavy operations**: Ask for permission before running resource-heavy evaluation runs or multi-provider calls.
