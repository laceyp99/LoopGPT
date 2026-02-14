# LoopGPT Evaluation Framework

A unified evaluation framework for testing MIDI loop generation across multiple AI models, with an interactive Plotly Dash dashboard for analyzing results.

## Overview

The evaluation framework has two main components:

- **`Evaluator`** -- Runs structured evaluations across models, prompts, roots, and scales, saving MIDI files, chat history, and test results.
- **`analysis.py`** -- An interactive Plotly Dash dashboard that loads evaluation run data and visualizes it across 7 tabs with 19 chart types.

## Quick Start

### Run an Evaluation

```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator(output_dir="runs", temperature=0.0)

results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C", "G", "D"],
    models="openai",
    run_name="my_first_eval"
)
```

### Launch the Dashboard

```bash
# Interactive run selection
python evaluation/analysis.py

# Direct path to a run
python evaluation/analysis.py runs/20260210_224954_arpeggiator_local_pt
```

The dashboard opens at `http://127.0.0.1:8050/`.

## Installation

All dependencies are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages: `dash`, `dash-bootstrap-components`, `pandas`, `plotly`, `mido`, `rich`.

---

## Evaluator

### Basic Evaluation

```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator(output_dir="runs", temperature=0.0)

# Single prompt, multiple roots, one provider
results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C", "G"],
    models="openai",
    run_name="quarter_note_test"
)
```

The evaluator automatically appends `" in {root} {scale}"` to each prompt and runs both major and minor scales for every root.

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

### Testing Reasoning Variations

When `test_reasoning=True`, the evaluator tests all thinking modes and effort levels for compatible models:

```python
results = evaluator.evaluate(
    prompts="complex chord progression",
    roots=["C", "G"],
    models=["o3", "claude-sonnet-4-5"],
    run_name="reasoning_test",
    test_reasoning=True
)
```

| Provider | Model Type | Variations |
|----------|------------|------------|
| OpenAI | gpt 5+ and o-series | various effort levels (none, minimal, low, medium, high, xhigh) |
| Anthropic | claude 4+ | thinking off/on, only effort levels for claude-opus-4-6 |
| Google | gemini 2.5+ | thinking off/on, only effort levels for gemini 3 family |
| Ollama | All | only thinking on if the model supports it |

### Testing Prompt Translation

When `test_prompt_translation=True`, each combination runs twice -- with and without the prompt translation feature (which enriches the user prompt via an extra API call before generation):

```python
results = evaluator.evaluate(
    prompts="jazzy walking bass",
    roots=["E", "A"],
    models="ollama",
    run_name="translation_comparison",
    test_prompt_translation=True
)
```

This roughly doubles the number of generations and, for translated runs, approximately doubles latency per generation due to the extra API call.

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

| Test | Description | Auto-Detection |
|------|-------------|----------------|
| `scale` | Validates notes belong to the specified scale | Always uses root/scale from prompt |
| `duration` | Validates note durations match expected value | Detects from keywords: `quarter`, `eighth`, `sixteenth`, `16th`, `8th`, `half`, `whole` |

The `scale` test always runs since root and scale are always applied to prompts. Duration keywords are defined in `src/utils.py` as `DURATION_KEYWORDS` and shared across the codebase.

### Evaluator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | `"evaluations"` | Base directory for all run outputs |
| `temperature` | `float` | `0.0` | Temperature for generation |

#### `evaluate()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `str \| list[str]` | required | Base prompt(s) -- `" in {root} {scale}"` is appended |
| `roots` | `list[str]` | required | Root notes to test (e.g. `["C", "F#", "Eb"]`) |
| `models` | `str \| list[str]` | `"all"` | Provider name, `"all"`, or list of model names |
| `run_name` | `str` | required | Name for this run (used in output directory) |
| `tests` | `list[str]` | `["scale", "duration"]` | Which validation tests to run |
| `test_reasoning` | `bool` | `False` | Test all thinking/effort variations |
| `test_prompt_translation` | `bool` | `False` | Test with and without prompt translation |

---

## Analysis Dashboard

The dashboard is a Plotly Dash application that loads evaluation run data and provides interactive visualization across 7 tabs.

### Launching

```bash
# Interactive selection from available runs
python evaluation/analysis.py

# Direct path
python evaluation/analysis.py runs/20260210_224954_arpeggiator_local_pt

# Just the run name (looks in runs/ automatically)
python evaluation/analysis.py 20260210_224954_arpeggiator_local_pt
```

### Global Filters

A filter bar at the top of every page lets you narrow results by:

- **Models** -- Select which models to include
- **Root Notes** -- Filter by root note (e.g. C, F#, Eb)
- **Scale Type** -- Major, minor, or both
- **Variation** -- Standard, translated, reasoning effort levels

All charts update in real time when filters change.

### Dashboard Tabs

#### Tab 1: Overview
- Metric cards: total generations, pass rate, best/worst model, total cost, average latency
- Overall pass rate by model (horizontal bar chart)

#### Tab 2: Model Performance
- Per-test breakdown: scale vs duration vs overall pass rate per model
- Major vs minor pass rate comparison per model
- Model x scale heatmap
- Model x root heatmap

#### Tab 3: Root & Scale
- Pass rate by root note
- Major vs minor pass rate per root note
- Full model x root+scale heatmap

#### Tab 4: Prompt Translation
- Side-by-side pass rate comparison: standard vs translated
- Translation impact delta (positive = translation helped)
- Latency comparison: standard vs translated

#### Tab 5: Latency
- Latency distribution box plots per model (split by standard/translated when applicable)
- Latency vs pass rate scatter plot

#### Tab 6: Cost
- Total cost by model
- Cost per generation vs pass rate scatter plot

#### Tab 7: Error Patterns
- Generation failure rate (API/conversion errors) per model
- Most common incorrect pitch classes by model (as note names)
- Incorrect intervals relative to prompted root per model (e.g. m3, P5 -- helps identify systematic confusions)
- Incorrect durations by model showing actual vs requested duration

### Exporting

Click the **Export Dashboard** button to save all 19 charts as individual HTML files plus a combined `dashboard.html` to `runs/<run>/analysis/`:

```
runs/20260210_224954_arpeggiator_local_pt/
└── analysis/
    ├── dashboard.html              # Combined single-page dashboard
    ├── pass_rate_by_model.html
    ├── per_test_breakdown.html
    ├── incorrect_intervals.html
    └── ... (19 chart files total)
```

The exported HTML files are self-contained and can be shared without a running server.

### Programmatic Usage

You can also use the data loading and chart building functions directly:

```python
from evaluation.analysis import load_run, build_pass_rate_by_model

df, config, summary = load_run("runs/20260210_224954_arpeggiator_local_pt")

# df is a pandas DataFrame with one row per generation
print(df.groupby("model")["overall_pass"].mean())

# Build individual charts
fig = build_pass_rate_by_model(df)
fig.show()
```

---

## Output Structure

Each evaluation run creates a timestamped directory:

```
runs/
└── 20260210_224954_my_first_eval/
    ├── config.json                    # Full evaluation configuration
    ├── summary.json                   # Aggregated results + statistics
    ├── analysis/                      # Created by dashboard export
    │   └── dashboard.html
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
├── output.mid              # Standard variation (no subfolder)
├── messages.json
├── test_results.json
└── standard_translated/    # Translated variation
    ├── output.mid
    ├── messages.json
    └── test_results.json
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
    "by_root": { "C": { "tested": 24, "passed": 18, "pass_rate": 0.75 } },
    "by_scale": { "major": { "tested": 24, "passed": 20, "pass_rate": 0.833 } }
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

---

## Adding Custom Tests

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
    "my_test": my_custom_test,
}
```

3. Add keyword detection in `_detect_test_params()` if the test parameters should be auto-detected from prompt text.

---

## Error Handling

The evaluator continues on failures, logging errors and saving partial results:

- API errors are captured in `test_results.json` with an `"error"` field
- Failed generations are counted in `summary.json` under `failed_generations`
- MIDI conversion errors are logged but don't halt the evaluation
- All logs are written to `<output_dir>/run.log`

## Performance Notes

- **Cloud providers** run asynchronously with rate limiting based on RPM from `model_list.json`
- **Ollama** runs synchronously, sorted by model to minimize GPU memory swaps
- A live Rich progress table displays during evaluation with per-model pass rates, latency, and cost
- Large evaluations (many models x many prompts x many roots) can take significant time and incur API costs

## File Reference

| File | Description |
|------|-------------|
| `evaluator.py` | Main `Evaluator` class -- orchestrates generation, testing, and result saving |
| `tests.py` | MIDI validation test functions (`scale_test`, `duration_test`) |
| `analysis.py` | Interactive Plotly Dash dashboard (7 tabs, 19 charts, global filters, export) |
