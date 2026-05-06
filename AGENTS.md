# AGENTS.md - LoopGPT Codebase Guide

This file gives AI coding agents the minimum project context needed to work safely in this repository.

## Start Here

- Read [README.md](README.md) for product-level usage and environment setup.
- Read [evaluation/README.md](evaluation/README.md) before changing anything under `evaluation/`.
- Do not run broad evaluations unless the user explicitly asks. They create many model calls and can be slow or expensive.

## Project Snapshot

LoopGPT generates 4-bar MIDI loops from natural-language prompts through a Gradio app in `app.py`.

Current major areas:

- `app.py`: Gradio UI and callback wiring.
- `src/runs.py`: Main generation router across providers.
- `src/openai_api.py`, `src/claude_api.py`, `src/gemini_api.py`, `src/ollama_api.py`: Provider-specific generation and prompt-translation logic.
- `src/midi_processing.py`: Loop object to MIDI conversion.
- `src/objects.py`: Pydantic models for loop structure.
- `src/utils.py`: Shared MIDI helpers, duration keyword detection, visualization helpers, and logging utilities.
- `src/audio.py`: Optional MIDI to audio rendering using FluidSynth and FFmpeg.
- `src/history.py`: Saves and loads recent generations in `generations/`.
- `evaluation/evaluator.py`: Unified evaluation runner.
- `evaluation/tests.py`: MIDI validation helpers such as scale and duration checks.
- `evaluation/analysis.py`: Dash dashboard for exploring evaluation runs.

## Safe Commands

Setup and app run:

```bash
pip install -r requirements.txt
python app.py
```

Safe evaluation-adjacent commands:

```bash
python evaluation/analysis.py
python evaluation/analysis.py <path-to-run-directory>
python -c "from evaluation.tests import scale_test, duration_test; from mido import MidiFile; midi = MidiFile('output.mid'); print(scale_test(midi, 'C', 'major')); print(duration_test(midi, 'quarter'))"
```

Avoid by default:

- Large evaluation runs through `Evaluator.evaluate(...)` unless the user explicitly asks.
- Any command that would fan out across many cloud models.

## Evaluation Conventions

The evaluation directory no longer uses separate runner scripts. The current pattern is:

- `evaluation/evaluator.py` exposes an `Evaluator` class that handles generation, test execution, result saving, and async versus sync provider routing.
- `test_reasoning=True` expands compatible models across thinking and effort variations.
- Validation tests are selected by name through the `tests` argument and executed via `Evaluator.run_tests(...)`.
- Duration expectations are auto-detected from prompt text using shared keywords from `src/utils.py`.

Important detail:

- `Evaluator` defaults `output_dir` to `evaluations`, while examples in docs may pass `output_dir="runs"`. Check the caller before assuming output paths.

## Project Conventions

### Imports

Follow the local style already used in the file you are editing. This repo often groups imports as:

1. Third-party packages
2. Standard library modules
3. Local `src` or `evaluation` imports

Preserve existing style when touching older files rather than reformatting unrelated imports.

### Logging

- Use `logging` for backend and evaluation code.
- Evaluation code configures root logging in `Evaluator._setup_logging()` and writes to `<output_dir>/run.log`.

### Data and Outputs

- `model_list.json` is the source of truth for provider model metadata.
- `loop.json` stores the latest loop-generation message history.
- `generations/` stores recent UI generations and metadata.
- Evaluation runs create structured directories with `config.json`, `summary.json`, generated MIDI, message logs, and per-run test results.

### Error Handling

- Raise `ValueError` for invalid user-facing configuration.
- Prefer graceful fallbacks for unavailable optional dependencies or services.
- Preserve the current Ollama behavior: avoid import-time hard failures when the local host is unavailable.

## Known Pitfalls

- `app.py` starts the Gradio app at import time, so imports used during startup must stay safe.
- `src/ollama_api.py` should not rely on import-time network success; prefer lazy or cached status checks.
- Audio playback is optional and depends on external tools plus a SoundFont in `soundfonts/`.
- The repo contains generated artifacts such as MIDI files and prior run outputs; do not treat them as source-of-truth code.

## Common Change Paths

Adding or changing model support:

1. Update `model_list.json`.
2. Update the relevant provider module in `src/`.
3. Adjust `src/runs.py` routing if the provider contract changed.
4. Update `app.py` controls only if the user-facing options changed.

Changing loop structure or MIDI semantics:

1. Update `src/objects.py`.
2. Update `src/midi_processing.py`.
3. Update any affected provider parsing logic.
4. Re-check prompt files in `Prompts/`.

Changing evaluation behavior:

1. Start with `evaluation/evaluator.py`.
2. Update or add checks in `evaluation/tests.py`.
3. Update [evaluation/README.md](evaluation/README.md) if behavior or output shape changes.
4. Keep dashboard assumptions in `evaluation/analysis.py` in sync with result schema changes.