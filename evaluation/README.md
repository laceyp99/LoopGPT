# LoopGPT Evaluation Framework

A unified evaluation framework for testing MIDI loop generation across multiple AI models.

## Overview

The `Evaluator` class provides a flexible, extensible pipeline for:

- Testing multiple prompts across multiple models in a single run
- Automatically appending root notes and scales to your prompts
- Auto-detecting test parameters (like duration) from prompt text
- Running async for cloud providers (OpenAI, Anthropic, Google) and sync for local models (Ollama)
- Saving structured results including MIDI files, chat history, and test results

## Quick Start

```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator()

results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C", "G", "D"],
    models="openai",
    run_name="my_first_eval"
)
```

## Installation

The evaluator uses dependencies already in `requirements.txt`. No additional installation needed.

## Usage

### Basic Evaluation

```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator()

# Single prompt, multiple roots, one provider
results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C", "G"],
    models="openai",
    run_name="quarter_note_test"
)
```

### Multiple Prompts

Test different musical patterns in a single run:

```python
results = evaluator.evaluate(
    prompts=[
        "an arpeggiator using only quarter notes",
        "an arpeggiator using only eighth notes",
        "an arpeggiator using only sixteenth notes",
    ],
    roots=["C", "D", "E", "F", "G", "A", "B"],
    models="all",
    run_name="duration_comparison"
)
```

### Model Selection

The `models` parameter accepts several formats:

| Value | Description |
|-------|-------------|
| `"all"` | All models from all providers (cloud + Ollama) |
| `"openai"` | All OpenAI models |
| `"anthropic"` | All Anthropic models |
| `"google"` | All Google Gemini models |
| `"ollama"` | All local Ollama models |
| `["gpt-4o-mini", "claude-sonnet-4-5"]` | Specific models by name |

```python
# All models from one provider
results = evaluator.evaluate(
    prompts="melody",
    roots=["C"],
    models="anthropic",
    run_name="anthropic_test"
)

# Specific models
results = evaluator.evaluate(
    prompts="melody",
    roots=["C"],
    models=["gpt-4o-mini", "claude-sonnet-4-5", "gemini-2.5-flash"],
    run_name="model_comparison"
)
```

### Testing Reasoning Variations

When `test_reasoning=True`, the evaluator tests all thinking modes and effort levels for compatible models:

```python
results = evaluator.evaluate(
    prompts="complex chord progression",
    roots=["C", "G"],
    models=["o3", "claude-sonnet-4-5"],
    run_name="reasoning_test",
    test_reasoning=True  # Tests all effort levels: minimal, low, medium, high
)
```

**What gets tested:**

| Provider | Model Type | Variations |
|----------|------------|------------|
| OpenAI | o-series (o1, o3, etc.) | 4 effort levels (minimal, low, medium, high) |
| Anthropic | Extended thinking models | thinking off + thinking on with 4 effort levels |
| Google | Thinking-capable models | thinking off + thinking on with 4 effort levels |
| Ollama | All | Single standard run |

### Testing Prompt Translation

When `test_prompt_translation=True`, each combination runs twice: with and without the prompt translation feature:

```python
results = evaluator.evaluate(
    prompts="jazzy walking bass",
    roots=["E", "A"],
    models="openai",
    run_name="translation_comparison",
    test_prompt_translation=True
)
```

### Configuring Tests

The `tests` parameter controls which validation tests run on generated MIDI:

```python
results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C"],
    models="openai",
    run_name="scale_only_test",
    tests=["scale"]  # Only run scale test, skip duration
)
```

**Available tests:**

| Test | Description | Auto-Detection |
|------|-------------|----------------|
| `scale` | Validates notes belong to the specified scale | Always uses root/scale from prompt |
| `duration` | Validates note durations match expected value | Detects from keywords: `quarter`, `eighth`, `sixteenth`, `16th`, `8th`, `half`, `whole` |

The `scale` test always runs since root and scale are always applied to prompts.

## Output Structure

Each evaluation run creates a timestamped directory:

```
evaluations/
└── 20260207_143022_my_first_eval/
    ├── config.json                    # Full evaluation configuration
    ├── summary.json                   # Aggregated results + statistics
    └── results/
        └── OpenAI/
            └── gpt-4o-mini/
                └── an_arpeggiator_using_only_quarter_notes/
                    ├── C_major/
                    │   ├── output.mid         # Generated MIDI file
                    │   ├── messages.json      # Chat history (for fine-tuning)
                    │   └── test_results.json  # Individual test results
                    └── C_minor/
                        └── ...
```

When using `test_reasoning` or `test_prompt_translation`, variation subfolders are created:

```
C_major/
├── thinking_false/
│   └── ...
├── thinking_true_effort_low/
│   └── ...
├── thinking_true_effort_high/
│   └── ...
└── ...
```

### config.json

Stores the full configuration used for the run:

```json
{
    "run_name": "my_first_eval",
    "timestamp": "20260207_143022",
    "prompts": ["an arpeggiator using only quarter notes"],
    "roots": ["C", "G"],
    "scales": ["major", "minor"],
    "models": [["OpenAI", "gpt-4o-mini"]],
    "tests": ["scale", "duration"],
    "test_reasoning": false,
    "test_prompt_translation": false,
    "temperature": 0.0
}
```

### summary.json

Aggregated statistics for the entire run:

```json
{
    "run_id": "20260207_143022_my_first_eval",
    "totals": {
        "total_generations": 48,
        "successful_generations": 45,
        "failed_generations": 3,
        "overall_pass_count": 36,
        "overall_pass_rate": 0.75,
        "total_cost": 1.25,
        "total_time": 120.5
    },
    "by_model": {
        "gpt-4o-mini": {
            "provider": "OpenAI",
            "tested": 24,
            "passed": 20,
            "pass_rate": 0.833,
            "total_cost": 0.50,
            "avg_latency": 2.1
        }
    },
    "by_root": { ... },
    "by_scale": { ... }
}
```

### test_results.json

Individual results for each generation:

```json
{
    "model": "gpt-4o-mini",
    "provider": "OpenAI",
    "prompt": "an arpeggiator using only quarter notes in C major",
    "original_prompt": "an arpeggiator using only quarter notes",
    "root": "C",
    "scale": "major",
    "config": {
        "use_thinking": false,
        "effort": null,
        "translate_prompt": false,
        "temperature": 0.0
    },
    "metrics": {
        "api_latency": 2.34,
        "cost": 0.0025
    },
    "tests": {
        "scale": {
            "ran": true,
            "params": { "root": "C", "scale": "major" },
            "total": 16,
            "correct": 16,
            "incorrect": 0,
            "pitches": { "correct": [0, 2, 4, 5, 7, 9, 11], "incorrect": [] }
        },
        "duration": {
            "ran": true,
            "params": { "duration": "quarter" },
            "detected_from_prompt": true,
            "total": 16,
            "correct": 16,
            "incorrect": 0,
            "lengths": {}
        },
        "overall_pass": true
    }
}
```

## API Reference

### Evaluator Class

```python
class Evaluator:
    def __init__(
        self,
        output_dir: str = "evaluations",  # Base directory for outputs
        temperature: float = 0.0,          # Default generation temperature
    )
```

### evaluate() Method

```python
def evaluate(
    self,
    prompts: str | list[str],              # Prompt(s) to test
    roots: list[str],                       # Root notes (e.g., ["C", "D", "F#"])
    models: str | list[str] = "all",        # Model specification
    run_name: str,                          # Required: name for this run
    tests: list[str] = ["scale", "duration"],  # Tests to run
    test_reasoning: bool = False,           # Test thinking/effort variations
    test_prompt_translation: bool = False,  # Test with/without translation
) -> dict:
```

**Returns:** Summary dictionary with aggregated statistics.

### run_tests() Method

Run tests on a MIDI file directly (useful for custom workflows):

```python
def run_tests(
    self,
    midi_data: MidiFile,    # MIDI file to test
    root: str,              # Root note
    scale: str,             # Scale (major/minor)
    prompt: str,            # Original prompt (for param detection)
    tests: list[str],       # List of test names
) -> dict:
```

## Adding Custom Tests

To add a new test:

1. Create the test function in `tests.py`:

```python
def my_custom_test(midi, param1, param2):
    """Your test logic here."""
    return {
        "total": ...,
        "correct": ...,
        "incorrect": ...,
    }
```

2. Register it in `evaluator.py`:

```python
AVAILABLE_TESTS = {
    "scale": scale_test,
    "duration": duration_test,
    "my_test": my_custom_test,  # Add here
}
```

3. Add keyword detection if needed:

```python
def _detect_test_params(self, prompt: str, test_name: str) -> dict:
    if test_name == "my_test":
        # Your detection logic
        return {"param1": value1, "param2": value2}
    ...
```

## Examples

### Full Evaluation Suite

```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator(output_dir="my_evals", temperature=0.3)

results = evaluator.evaluate(
    prompts=[
        "an arpeggiator using only quarter notes",
        "an arpeggiator using only eighth notes",
        "a melodic bass line with half notes",
    ],
    roots=["C", "D", "E", "F", "G", "A", "B"],
    models="all",
    run_name="comprehensive_eval",
    tests=["scale", "duration"],
    test_reasoning=True,
    test_prompt_translation=True,
)

print(f"Pass rate: {results['totals']['overall_pass_rate']:.1%}")
print(f"Total cost: ${results['totals']['total_cost']:.2f}")
```

### Quick Model Comparison

```python
evaluator = Evaluator()

results = evaluator.evaluate(
    prompts="simple C major arpeggio with quarter notes",
    roots=["C"],
    models=["gpt-4o-mini", "claude-sonnet-4-5", "gemini-2.5-flash"],
    run_name="quick_comparison"
)

for model, stats in results["by_model"].items():
    print(f"{model}: {stats['pass_rate']:.1%} pass, {stats['avg_latency']:.2f}s avg")
```

### Ollama-Only Evaluation

```python
evaluator = Evaluator()

results = evaluator.evaluate(
    prompts="eighth note melody",
    roots=["C", "G"],
    models="ollama",
    run_name="local_model_test"
)
```

## Error Handling

The evaluator continues on failures, logging errors and saving partial results:

- API errors are captured in `test_results.json` with an `"error"` field
- Failed generations are counted in `summary.json` under `failed_generations`
- MIDI conversion errors are logged but don't halt the evaluation

## Performance Notes

- **Cloud providers** run asynchronously with rate limiting based on RPM from `model_list.json`
- **Ollama** runs synchronously (one request at a time)
- A live progress table displays during evaluation
- Large evaluations (many models x many prompts x many roots) can take significant time and incur API costs

## File Reference

| File | Description |
|------|-------------|
| `evaluator.py` | Main `Evaluator` class |
| `tests.py` | MIDI validation test functions |
| `analysis.py` | Visualization utilities for results |
