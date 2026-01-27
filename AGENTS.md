# AGENTS.md - LoopGPT Codebase Guide

This document provides guidance for AI coding agents working on the LoopGPT codebase.

## Project Overview

LoopGPT is a music generation tool that creates 4-bar MIDI loops using AI model APIs (OpenAI, Anthropic, Google Gemini, and Ollama). It features a Gradio web interface for user interaction.

## Build/Run Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py
```
The Gradio UI will be available at http://127.0.0.1:7860/

### Running Evaluations
Don't run any evaluations. This is for human usage and evaluation across the whole model landscape. This can take a large amount of time to run completely and will incur large API costs.

Run SOTA model evaluation (costs ~$40, takes 30+ minutes):
```bash
python evaluation/sota_eval.py
```

Run local Ollama model evaluation:
```bash
python evaluation/ollama_eval.py
```

Run prompt translation evaluation:
```bash
python evaluation/prompt_translation_eval.py
```

### Running Single Tests
The project uses direct test functions rather than a test framework. Run individual tests:
```bash
python -c "from evaluation.tests import scale_test, duration_test; from mido import MidiFile; midi = MidiFile('output.mid'); print(scale_test(midi, 'C', 'major'))"
```

Don't test via the tests.py module directly:
```bash
python evaluation/tests.py
```

## Project Structure

```
LoopGPT/
├── app.py                    # Gradio UI entry point
├── model_list.json           # Model pricing and rate limits
├── requirements.txt          # Python dependencies
├── src/
│   ├── .env                  # API keys (OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY)
│   ├── gpt_api.py           # OpenAI GPT endpoints
│   ├── o_api.py             # OpenAI reasoning (o-series) endpoints
│   ├── gemini_api.py        # Google Gemini endpoints
│   ├── claude_api.py        # Anthropic Claude endpoints
│   ├── ollama_api.py        # Local Ollama endpoints
│   ├── midi_processing.py   # Loop <-> MIDI conversion
│   ├── objects.py           # Pydantic models (Note, Bar, Loop)
│   ├── utils.py             # MIDI utilities, visualization
│   └── runs.py              # API routing logic
├── Prompts/
│   ├── loop gen.txt         # System prompt for loop generation
│   └── prompt translation.txt # System prompt for prompt enrichment
└── evaluation/
    ├── tests.py             # MIDI validation tests
    ├── sota_eval.py         # Async evaluation for cloud models
    ├── ollama_eval.py       # Serial evaluation for local models
    └── analysis.py          # Evaluation data visualization
```

## Code Style Guidelines

### Imports
Organize imports in this order:
1. Third-party libraries (openai, anthropic, google.genai, gradio, mido, etc.)
2. Standard library (os, sys, json, logging, asyncio, etc.)
3. Local modules (src.utils, src.objects, etc.)

Example:
```python
from openai import OpenAI
from dotenv import load_dotenv
import src.utils as utils
import src.objects as objects
import logging
import json
import sys
import os
```

### Logging
Use the standard logging pattern at module level:
```python
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

### Type Hints
- Use type hints in function signatures, especially for Pydantic models
- Document return types in docstrings as tuples: `Returns: tuple: (content, messages, cost)`

### Docstrings
Use Google-style docstrings with Args and Returns sections:
```python
def function_name(param1, param2):
    """Brief description of the function.

    Args:
        param1 (str): Description of param1.
        param2 (int, optional): Description of param2. Defaults to 0.

    Returns:
        tuple: (result1, result2, result3)

    Raises:
        ValueError: When invalid input is provided.
    """
```

### Naming Conventions
- **Files**: snake_case (`midi_processing.py`, `gpt_api.py`)
- **Classes**: PascalCase (`Loop`, `Note`, `TimeInformation`)
- **Functions/Variables**: snake_case (`loop_gen`, `calc_price`, `midi_loop`)
- **Constants**: UPPER_SNAKE_CASE (`OPENAI_API_KEY`)
- **Pydantic Models**: Use `_G` suffix for Gemini-compatible variants (`Loop_G`, `Note_G`)

### Pydantic Models
Define models in `src/objects.py` with Field descriptions:
```python
class Note(BaseModel):
    pitch: str = Field(..., description='Pitch of the note (e.g. "C", "D")')
    octave: int = Field(..., description='Octave of the note (e.g. 1-8)')
    velocity: int = Field(..., description='Velocity of the note (e.g. 0-127)')
    time: TimeInformation
```

### Error Handling
- Use try/except blocks for API calls and file operations
- Log errors with `logger.error()` or `logger.warning()`
- Raise `ValueError` for invalid inputs with descriptive messages
- Return graceful fallbacks in UI functions (return `None, None, error_message`)

```python
try:
    loop, messages, total_cost = runs.generate_midi(...)
except Exception as e:
    error = str(e)
    return None, None, error
```

### API Client Pattern
Each API module follows this pattern:
1. Initialize client function that loads API key from `.env`
2. `calc_price()` function for cost calculation
3. `prompt_gen()` for text generation
4. `loop_gen()` for structured MIDI generation
5. Save messages to JSON for debugging/training

```python
def initialize_openai_client():
    load_dotenv(os.path.join("src", ".env"))
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or not api_key.strip():
        logger.error("OPENAI_API_KEY is not set!")
    return OpenAI(api_key=api_key)
```

### Configuration
- API keys stored in `src/.env`
- Model metadata (pricing, rate limits, capabilities) in `model_list.json`
- System prompts in `Prompts/` directory as `.txt` files

### MIDI Processing
- Use `mido` library for MIDI operations
- Ticks per beat: 480 (default mido value)
- Sixteenth note = ticks_per_beat / 4
- 4 bars = 64 sixteenth notes

### Async Patterns
For evaluation scripts, use asyncio with semaphores for rate limiting:
```python
async def call_model_async(prompt, model_choice, temp, use_thinking, translate_prompt_choice):
    def sync_call():
        return runs.generate_midi(...)
    return await asyncio.to_thread(sync_call)
```

## Key Files to Understand

1. **src/runs.py**: Central routing logic that directs requests to appropriate API modules
2. **src/objects.py**: Pydantic models defining the Loop/Bar/Note structure
3. **src/midi_processing.py**: Converts between Loop objects and MIDI files
4. **app.py**: Gradio UI with callbacks for model selection and generation

## Common Tasks

### Adding a New Model Provider
1. Create new API module in `src/` (e.g., `new_api.py`)
2. Implement `initialize_client()`, `calc_price()`, `prompt_gen()`, `loop_gen()`
3. Add model info to `model_list.json`
4. Update routing logic in `src/runs.py`
5. Update provider dropdown in `app.py`

### Modifying Loop Structure
1. Update Pydantic models in `src/objects.py`
2. Update MIDI conversion in `src/midi_processing.py`
3. Update system prompts in `Prompts/`

### Adding New Evaluation Tests
1. Add test function to `evaluation/tests.py`
2. Call from `run_midi_tests()` function
3. Update result aggregation in evaluation scripts