# Conductor Core

`conductor-core` is the reusable prompt-to-MIDI engine behind the Conductor
applications. It can be embedded in a CLI, notebook, backend service, test
harness, or another UI without importing Gradio, Dash, Plotly, or the evaluation
package.

Core owns:

- provider routing for OpenAI, Anthropic, Google, and Ollama;
- validated four-bar loop models and provider response parsing;
- prompt assembly and model capability metadata;
- loop-to-MIDI and MIDI-to-loop conversion;
- generation workspaces, messages, metadata, and history persistence;
- optional SoundFont discovery and MIDI-to-audio rendering;
- structured generation results and progress events.

## Installation

From the `conductor-core` project directory on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

The base install supports deterministic music models and MIDI operations. Add
only the capabilities your consumer needs:

| Extra | Adds | Example use |
|---|---|---|
| `openai` | OpenAI SDK | An OpenAI-only service |
| `anthropic` | Anthropic SDK | A Claude-only client |
| `google` | Google Gen AI SDK | A Gemini-only notebook |
| `ollama` | Ollama SDK | Local generation |
| `providers` | All four provider SDKs | A client with model switching |
| `playback` | MIDI synthesis and MP3 helpers | Audio previews |
| `dev` | Pytest | Core development |

```powershell
# All providers
.\.venv\Scripts\python.exe -m pip install -e ".[providers]"

# One provider plus playback
.\.venv\Scripts\python.exe -m pip install -e ".[google,playback]"

# Complete local development install
.\.venv\Scripts\python.exe -m pip install -e ".[providers,playback,dev]"
```

Using the venv interpreter explicitly is intentional. `py -3.12 -m pip`
selects the registered global Python even when a virtual environment exists.

## Basic generation

```python
from conductor_core import EngineConfig, GenerationRequest, LoopGenerationEngine

engine = LoopGenerationEngine(
    EngineConfig.from_defaults(artifact_root="generations")
)
result = engine.generate(
    GenerationRequest(
        key="C",
        scale="Major",
        description="warm neo-soul electric piano chords",
        model="gemini-3.1-flash-lite",
        temperature=0.3,
    )
)

print(result.generation_id)
print(result.midi_path)
print(result.cost)
```

`generate()` is synchronous. It calls the selected provider, converts the
validated loop to MIDI, and persists the resulting artifacts before returning.

## Credentials and provider selection

Credentials can be injected by the calling application:

```python
from conductor_core import EngineConfig, ProviderCredentials

config = EngineConfig.from_defaults(
    artifact_root="my-output",
    provider_credentials=ProviderCredentials(
        openai_api_key="...",
        google_api_key="...",
        anthropic_api_key="...",
        ollama_host="http://localhost:11434",
    ),
)
```

If a credential is not injected, provider modules fall back to these environment
variables:

```ini
OPENAI_API_KEY="..."
GEMINI_API_KEY="..."
ANTHROPIC_API_KEY="..."
OLLAMA_API_HOST_ADDRESS="http://localhost:11434"
```

The `model` normally determines the provider from packaged model metadata. Set
`provider` on `GenerationRequest` when a caller needs to record or override the
provider label explicitly.

## Generation request options

```python
from conductor_core import GenerationRequest

request = GenerationRequest(
    key="F#",
    scale="minor",
    description="dark sixteenth-note synth arpeggio",
    model="gpt-5-mini",
    provider="OpenAI",
    temperature=0.2,
    use_thinking=False,
    effort="low",
    prompt_override=None,
    render_audio=False,
    soundfont_path=None,
)
```

| Field | Purpose |
|---|---|
| `key`, `scale`, `description` | Musical request added to the model prompt |
| `model` | Packaged model identifier used for routing and response handling |
| `provider` | Optional explicit provider metadata override |
| `temperature` | Sampling temperature for models that support it |
| `use_thinking` | Toggle-style reasoning control for supported models |
| `effort` | Model-specific reasoning effort such as `minimal`, `low`, or `high` |
| `prompt_override` | System prompt override for only this request |
| `render_audio` | Request an MP3 preview after MIDI generation |
| `soundfont_path` | SoundFont name or path for this request |

Model capabilities differ. Consumers can inspect
`conductor_core.music.get_model_info()` to build compatible controls instead of
assuming every model accepts temperature or the same reasoning settings.

## Prompt customization

Core ships with a default loop-generation prompt. Override it for every request
made by an engine:

```python
config = EngineConfig.from_defaults(
    prompt_override="Generate sparse four-bar loops using the required schema."
)
```

Or override it for one request:

```python
request = GenerationRequest(
    key="D",
    scale="Major",
    description="bright piano ostinato",
    model="gemini-3.1-flash-lite",
    prompt_override="Generate piano-only material using the required schema.",
)
```

The request override takes precedence over the engine override, which takes
precedence over the packaged prompt.

## Progress reporting

Use a callback to adapt synchronous Core work to logs, a progress bar, a queue,
or an asynchronous UI wrapper:

```python
def report_progress(event):
    print(f"[{event.stage}] {event.message}")

result = engine.generate(request, progress_callback=report_progress)
```

Current stages include provider generation, MIDI processing, and audio
rendering. The callback reports progress but does not cancel an in-flight
provider request.

## Audio rendering

```python
config = EngineConfig.from_defaults(
    artifact_root="generations",
    default_soundfont_path="FM-Piano1 20190916.sf2",
)
engine = LoopGenerationEngine(config)

result = engine.generate(
    GenerationRequest(
        key="A",
        scale="minor",
        description="soft felt-piano progression",
        model="gemini-3.1-flash-lite",
        render_audio=True,
    )
)

print(result.audio_path)
print(result.warnings)
```

Install the `playback` extra and provide FluidSynth and FFmpeg on the system
`PATH`. Audio failure does not discard a successful MIDI generation; Core
returns the MIDI with a warning and `audio_path=None`.

Useful lower-level playback helpers live in `conductor_core.playback`:

```python
from conductor_core.playback import (
    get_default_soundfont,
    is_playback_available,
    list_soundfonts,
    midi_to_mp3,
)

print(list_soundfonts())
print(get_default_soundfont())
print(is_playback_available())
```

## Results and persisted artifacts

`GenerationResult` contains:

| Attribute | Contents |
|---|---|
| `generation_id` | Unique filesystem generation identifier |
| `loop` | Validated provider-independent loop object |
| `midi_path` | Persisted MIDI path |
| `audio_path` | Persisted MP3 path, when rendering succeeds |
| `messages` | Provider conversation/response messages |
| `cost` | Provider-reported estimated cost, when available |
| `metadata` | Persisted generation metadata |
| `warnings` | Non-fatal issues such as skipped audio |

Each generation workspace contains `loop.mid`, `messages.json`,
`metadata.json`, and optionally `loop.mp3`. Bind history operations to a custom
root with `FilesystemArtifactStore`:

```python
from conductor_core.storage import FilesystemArtifactStore

store = FilesystemArtifactStore("service-data/generations")
engine = LoopGenerationEngine(config, store=store)

recent = store.load_history()
saved = store.get_generation(result.generation_id)
```

The store also supports deleting generations and updating saved audio metadata.
By default, history retains the newest 20 generations.

## Direct MIDI and music utilities

Consumers that already have loop data can use Core without a provider call:

```python
from mido import MidiFile
from conductor_core.midi import loop_to_midi, midi_to_loop

midi = MidiFile()
loop_to_midi(midi, loop, times_as_string=False)
midi.save("loop.mid")

restored_loop = midi_to_loop("loop.mid", times_as_string=False)
```

Additional packaged utilities and models are available from:

- `conductor_core.models` for loop, bar, note, and timing models;
- `conductor_core.music` for model metadata, prompts, scales, and durations;
- `conductor_core.routing` for lower-level provider routing;
- `conductor_core.storage` for artifact and history management;
- `conductor_core.playback` for optional audio operations.

Prefer `LoopGenerationEngine` for complete generation workflows so persistence,
cleanup, prompt handling, and provider behavior stay consistent.

## Error behavior

Provider, parsing, and MIDI conversion errors are raised to the caller. If an
error occurs after a workspace is allocated, Core removes the unfinished
workspace. Callers should catch exceptions at their application boundary and
decide how to display, retry, or log them.

## Logging

Core emits log records under the `conductor_core` logger namespace and never
configures handlers or global logging itself (a `NullHandler` is attached so
unconfigured consumers see no warnings). To surface Core logs, configure
logging in the application:

```python
import logging

logging.basicConfig(level=logging.INFO)          # everything to the console
# or route only Core records somewhere specific:
logging.getLogger("conductor_core").addHandler(my_handler)
```

## Validate Core independently

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests are deterministic and do not make live provider calls or require the
audio toolchain.
