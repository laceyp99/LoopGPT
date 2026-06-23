# LoopGPT: AI-Powered Music Generation

This project is a music generation tool that enables users to create 4 bar loops using an intuitive Gradio web interface. By leveraging AI model APIs from multiple providers, the application generates MIDI compositions based on user-defined parameters, making it a great resource for musicians, producers, and AI music enthusiasts.

## Features

- Interact with a sleek Gradio interface for an inviting user experience
- Audio playback to listen to generated MIDI inside the browser
- Switch between installed SoundFonts, refresh the SoundFont list in-app, and re-render audio without regenerating MIDI
- Visualize MIDI output using piano roll display
- Manage your last 20 generations from the history sidebar
- View/Edit the system prompt that instruct the model through the generation process
- Save full API message history to JSON files for training examples

## Quick Start
### Installation
Clone this repository:
```sh
git clone https://github.com/laceyp99/LoopGPT.git
cd LoopGPT
```
Install required dependencies:
```sh
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

### Audio Playback Setup (Optional)
To enable audio playback of generated MIDI loops, you'll need:

1. **FluidSynth** - MIDI synthesizer
   - Windows: Download from [FluidSynth releases](https://github.com/FluidSynth/fluidsynth/releases) and add to PATH
   - macOS: `brew install fluidsynth`
   - Linux: `apt install fluidsynth`

2. **FFmpeg** - Audio encoding
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `apt install ffmpeg`

3. **Bundled SoundFont** - Piano samples
   - The repo includes `soundfonts/FM-Piano1 20190916.sf2` as the default playback SoundFont
   - Source: [FreePats FM Synthesized Piano](https://freepats.zenvoid.org/ElectricPiano/synthesized-piano.html)
   - License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
   - You can still add additional `.sf2` files to the `soundfonts/` directory for alternate playback voices
   - If the app is already running, click **Refresh SoundFonts** in the UI after adding new files

*Note: Audio playback is optional. The app will still generate and download MIDI files without these dependencies.*

Set up your API keys by copying `src/.env.example` to `src/.env`, or enter them via the Gradio "API Keys" panel:
```ini
OPENAI_API_KEY="your-openai-api-key-here"
GEMINI_API_KEY="your-gemini-api-key-here"
ANTHROPIC_API_KEY="your-anthropic-api-key-here"
OLLAMA_API_HOST_ADDRESS="your-ollama-base-url-here"
```
### Run the Gradio UI
To start the web interface locally, run:
```sh
python app.py
```
Then visit [localhost](http://127.0.0.1:7860/) to access the UI.

## Testing

LoopGPT now has a local pytest suite for deterministic regression coverage. Use it for fast feedback on core music logic, MIDI conversion, history persistence, and provider routing without making live API calls.

Install the development extra if you want to run the tests:
```sh
pip install -e ".[dev]"
```

Run the full suite:
```sh
pytest
```

Run a focused file during development:
```sh
pytest tests/test_utils.py
pytest tests/test_midi_processing.py
pytest tests/test_runs.py
pytest tests/test_audio.py
pytest tests/test_app.py
```

The pytest suite is intentionally local and hermetic:

- `tests/` covers deterministic behavior in `src/`
- audio and app helper tests cover SoundFont discovery, selection, rerender decisions, and history metadata without needing live model calls
- generated test files use pytest temporary directories instead of writing into repo folders like `generations/`
- live provider calls, Gradio UI behavior, and optional audio-toolchain behavior are not part of the default pytest suite

## Usage

1. **Select your parameters**  
   - Key & scale (Major/minor)  
   - Text description (e.g. "Daniel Caesar R&B soul piano loop with sevenths and ninths interspersed throughout")  
   - Provider, Model, and model specific generation parameters (temperature, reasoning, etc.)
   - Optional: Edit the System Prompts in the Prompt Editor tab

2. **Generate & Download**  
   - Click **Generate Loop**  
   - Download the MIDI file via the "Download Generated MIDI" widget  
   - Listen to the audio preview rendered with the currently selected SoundFont (if playback is configured)
   - View the piano-roll image

3. **Work With SoundFonts**
   - Use the **SoundFont** dropdown between Playback and MIDI Visualization to choose the render voice
   - Click **Refresh SoundFonts** if you add new `.sf2` files while the app is already open
   - Click **Re-render Audio** to audition the currently loaded MIDI through a different SoundFont without regenerating the loop
   - Loaded history items reopen with their saved audio immediately; re-render only when you want a different SoundFont version

4. **Browse History**
   - Click **History** to open the sidebar
   - Select a previous generation from the dropdown
   - Click **Load** to view/play it, or **Delete** to remove it
   - History entries remember which SoundFont produced the saved audio preview

5. **Inspect JSON logs**  
   - Each saved generation includes `messages.json` with the full message exchange and reasoning

## Cost and Prompt Caching

LoopGPT estimates provider costs from the token usage fields returned by each API. Prompt-cache savings, when a provider reports them, are included in those estimates.

Normal LoopGPT generations should not be expected to benefit much from prompt caching. The reusable system prompt is relatively short, and each request includes a variable user prompt, so typical requests are below many provider cache thresholds. Avoid padding prompts solely to force cache eligibility; it adds cost and can make generations less predictable.

## Evaluation Framework

The evaluation framework lives in the `evaluation/` directory and is documented separately in [`evaluation/README.md`](evaluation/README.md).

Use pytest for fast local regression tests, and use the evaluation framework for slower prompt-to-model quality checks across providers. The two workflows are complementary rather than interchangeable.

## Project Structure

```
LoopGPT/
├── app.py                      # Gradio UI entry point
├── model_list.json             # Model pricing, rate limits, and capabilities
├── pyproject.toml              # Python project metadata 
├── UI.png                      # Screenshot of the Gradio interface
│
├── Prompts/
│   └── loop gen.txt            # System prompt for 4-bar loop generation
│
├── src/
│   ├── .env                    # API keys (OPENAI, GEMINI, ANTHROPIC, OLLAMA)
│   ├── openai_api.py           # OpenAI Responses endpoints (GPT + o-series)
│   ├── gemini_api.py           # Google Gemini endpoints
│   ├── claude_api.py           # Anthropic Claude endpoints
│   ├── ollama_api.py           # Local Ollama endpoints
│   ├── runs.py                 # API routing across supported providers
│   ├── midi_processing.py      # Loop <-> MIDI conversion via mido
│   ├── objects.py              # Pydantic models (Note, Bar, Loop, _G variants)
│   ├── utils.py                # MIDI helpers, visualization, message logging
│   ├── audio.py                # MIDI to audio conversion using FluidSynth
│   └── history.py              # Session history management (save/load/delete/update audio metadata)
│
├── tests/                      # Local pytest coverage for deterministic logic and UI helpers
│   ├── test_app.py             # App helper coverage for SoundFont refresh/rerender logic
│   ├── test_audio.py           # SoundFont discovery and audio helper coverage
│   ├── test_history.py         # History persistence and cleanup coverage
│   ├── test_midi_processing.py # MIDI conversion coverage
│   ├── test_objects.py         # Pydantic model coverage
│   ├── test_runs.py            # Provider routing coverage
│   └── test_utils.py           # Utility helper coverage
│
├── evaluation/                 # Standalone evaluation framework (see evaluation/README.md)
│   ├── evaluator.py            # Evaluator class -- orchestrates model testing
│   ├── tests.py                # MIDI validation tests (scale, duration)
│   ├── analysis.py             # Interactive Plotly Dash dashboard
│   └── README.md               # Full evaluation documentation
│
├── generations/                # Session history storage (auto-managed, up to 20)
│   └── gen_<timestamp>/
│       ├── loop.mid            # Generated MIDI file
│       ├── loop.mp3            # Rendered audio (if playback configured)
│       ├── messages.json       # Provider message history
│       └── metadata.json       # Generation parameters, artifact paths, and saved SoundFont info
│
├── runs/                       # Evaluation run outputs (created by evaluator)
│   ├── run.log                 # Evaluation log file
│   └── <timestamp_run_name>/
│       ├── config.json         # Full evaluation configuration
│       ├── summary.json        # Aggregated results and statistics
│       ├── analysis/           # Exported dashboard charts (HTML)
│       └── results/            # Per-model, per-prompt, per-key test results
│
└── soundfonts/                 # Bundled/default and user-added SoundFont (.sf2) files for playback
```

## Limitations

- The application generates MIDI files at a fixed tempo of ***120 BPM***, but can easily be changed in any DAW.
- The application currently supports only ***4-bar*** segments in ***4/4*** time signature.
- Generated music may require post-processing for professional use.
- The quality of generated music depends on the prompt and model parameters.
- Requires an active internet connection for API calls to model providers.
- High API usage may incur significant costs.
- Audio preview still depends on external FluidSynth and FFmpeg installations even though a default SoundFont is bundled with the repo.
