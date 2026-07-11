# Conductor

Conductor is a prompt-to-MIDI project suite. The repository now contains three
independently installable Python projects instead of one application owning the
entire stack.

| Project | Purpose | Install when you need |
|---|---|---|
| [`conductor-core`](packages/conductor-core/README.md) | Provider routing, loop models, MIDI conversion, playback helpers, and artifact persistence | A reusable generation engine, script, notebook, service, or custom client |
| [`conductor-main`](apps/conductor-main/README.md) | Gradio UI, callbacks, UI state, and piano-roll visualization | The interactive desktop/web client |
| [`conductor-eval`](projects/conductor-eval/README.md) | Evaluation orchestration, MIDI checks, reports, and the optional dashboard | Model and prompt quality analysis |

The original root modules remain as compatibility wrappers during the
transition. New code should import the packages above directly. Do not install
the root project as the normal setup path; choose the smallest project that
matches your use case.

## Run the main app on Windows

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".\packages\conductor-core[providers,playback]"
.\.venv\Scripts\python.exe -m pip install -e ".\apps\conductor-main"
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\conductor-main.exe
```

The explicit `.venv` paths matter on Windows. `py -3.12 -m pip` selects the
registered global Python and can silently create a mixed environment even after
a venv has been created. See the [Conductor Main guide](apps/conductor-main/README.md)
for validation and compatibility-launcher commands.

## Use only the engine

The base Core install has no UI or dashboard dependency:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".\packages\conductor-core"
```

```python
from conductor_core import EngineConfig, GenerationRequest, LoopGenerationEngine

engine = LoopGenerationEngine(EngineConfig.from_defaults(artifact_root="output"))
result = engine.generate(
    GenerationRequest(
        key="C",
        scale="Major",
        description="syncopated neo-soul chords",
        model="gemini-3.1-flash-lite",
        temperature=0.3,
    )
)
print(result.midi_path)
```

Install a provider extra before making live calls. Core's README documents
credentials, provider extras, playback, and the complete result contract.

## Install evaluation tools

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".\packages\conductor-core[providers]"
.\.venv\Scripts\python.exe -m pip install -e ".\projects\conductor-eval[dashboard,dev]"
```

Evaluation jobs can make many paid model calls. They are never part of the
ordinary test workflow and should be started intentionally. See the
[evaluation guide](projects/conductor-eval/README.md) for guarded execution and
small-run examples.

## Development and validation

Install each editable project with its development dependencies, then run the
local suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\packages\conductor-core[providers,playback,dev]"
.\.venv\Scripts\python.exe -m pip install -e ".\projects\conductor-eval[dashboard,dev]"
.\.venv\Scripts\python.exe -m pip install -e ".\apps\conductor-main[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Focused package checks:

```powershell
.\.venv\Scripts\python.exe -m pytest .\packages\conductor-core\tests -q
.\.venv\Scripts\python.exe -m pytest .\projects\conductor-eval\tests -q
.\.venv\Scripts\python.exe -m pytest .\apps\conductor-main\tests -q
```

The local tests do not make live provider calls. FluidSynth and FFmpeg are
external optional requirements and are not exercised by the default suite.

## Repository layout

```text
apps/conductor-main/       Replaceable Gradio client
packages/conductor-core/   Reusable generation engine
projects/conductor-eval/   Evaluation package and dashboard
app.py                     Transitional client launcher
evaluation/                Transitional evaluation wrappers
src/                       Legacy compatibility modules
tests/                     Compatibility regression tests
generations/               Local generated artifacts
```

## Repository split decision

The three projects remain in this monorepo for now. Their package metadata,
imports, resources, tests, and documented install commands are independent, so
they can be extracted later without another architectural refactor. Keeping
them together for the next milestone makes cross-package changes and regression
testing simpler while the public APIs stabilize. A separate Git repository or
PyPI release is deferred until independent versioning, ownership, or release
cadence provides a concrete benefit.

## Current product constraints

- Generated loops are four bars in 4/4 at 120 BPM.
- Output quality and cost depend on the selected model and prompt.
- Cloud providers require network access and may incur charges.
- `Stop Waiting` detaches the UI but cannot cancel an in-flight provider call.
- Audio preview requires FluidSynth, FFmpeg, and an available SoundFont.
